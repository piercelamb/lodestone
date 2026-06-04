"""Single-shot fetch of an ACL Anthology paper.

Structurally a remote analog of the ``ingest --pdf`` (local-PDF) path:
download the PDF, render to markdown via ``pymupdf4llm``, persist via the
shared ``_persist_pdf_row`` helper with ``html_source=PDF_FALLBACK``.
The Anthology exposes no HTML/LaTeX fulltext, but it does serve clean
per-paper MODS XML at ``aclanthology.org/<id>.xml`` (title / authors /
abstract / year), which we use in place of the PDF-heuristic metadata
the bare local-PDF path falls back to.

Paper identity in :class:`papers.arxiv_id` is namespaced with the
``acl:`` prefix (e.g. ``acl:2021.acl-long.285``) to avoid colliding with
arxiv ids and to make the source unambiguous at a glance.
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable

import httpx

from _system.db.connection import get_conn
from _system.schemas.paper_metadata import PaperMetadata
from _system.scripts.fetch_paper import (
    IngestExtractionFailed,
    _ArxivMetadata,
    _get_existing_paper,
    _persist_pdf_row,
)
from _system.utils.acl_urls import acl_pdf_url, acl_xml_url, parse_acl_id
from _system.utils.http import (
    make_default_client as _make_default_client,
    retry_http as _retry_http,
)
from _system.utils.logging import get_logger

__all__ = ["fetch"]

_LOG = get_logger("scripts.fetch_acl")

_MODS_NS = "http://www.loc.gov/mods/v3"


@_retry_http
def _get_acl_xml(client: httpx.Client, acl_id: str) -> str:
    url = acl_xml_url(acl_id)
    resp = client.get(url)
    resp.raise_for_status()
    return resp.text


@_retry_http
def _download_acl_pdf(client: httpx.Client, acl_id: str) -> bytes:
    url = acl_pdf_url(acl_id)
    resp = client.get(url)
    resp.raise_for_status()
    return resp.content


# MARC relator codes / text terms for roles we do NOT want to treat as
# authors. ACL proceedings frontmatter MODS lists volume editors as
# <name type="personal">; without this filter they'd be persisted as
# paper authors.
_NON_AUTHOR_ROLES = frozenset({
    "editor", "edt",
    "translator", "trl",
    "illustrator", "ill",
    "compiler", "com",
})


def _text(el: ET.Element | None) -> str:
    """Concatenated text including nested children's text and tails.

    MODS elements like <title>, <abstract>, and <namePart> may carry
    inline markup (italics, math, foreign words). Element.text returns
    only the text before the first child, which silently truncates the
    rest; itertext() walks the subtree.
    """
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _is_author_role(name_el: ET.Element) -> bool:
    """True unless the <name> carries an explicit non-author role.

    ACL author entries sometimes omit <role> entirely, so absence is
    treated as author. Explicit editor/translator/etc. roles (whether
    the marc-relator code "edt" or the text "editor") are filtered out.
    """
    for role_term in name_el.iter(f"{{{_MODS_NS}}}roleTerm"):
        term = _text(role_term).lower()
        if term in _NON_AUTHOR_ROLES:
            return False
    return True


def _parse_mods(xml_text: str, acl_id: str) -> _ArxivMetadata:
    """Parse ACL Anthology MODS XML into the internal ``_ArxivMetadata`` shape.

    The Anthology wraps a single ``mods:mods`` record inside a
    ``mods:modsCollection`` envelope. ``dateIssued`` is typically just
    the 4-digit year, so we normalize to ``YYYY-01-01`` for downstream
    code that expects an ISO-8601 date.
    """
    root = ET.fromstring(xml_text)
    if root.tag == f"{{{_MODS_NS}}}modsCollection":
        mods = root.find(f"{{{_MODS_NS}}}mods")
    else:
        mods = root
    if mods is None:
        raise IngestExtractionFailed(
            f"ACL Anthology MODS for {acl_id!r}: no <mods> element found"
        )

    title_info = mods.find(f"{{{_MODS_NS}}}titleInfo")
    if title_info is not None:
        title_main = _text(title_info.find(f"{{{_MODS_NS}}}title"))
        subtitle = _text(title_info.find(f"{{{_MODS_NS}}}subTitle"))
        title = f"{title_main}: {subtitle}" if title_main and subtitle else (title_main or subtitle)
    else:
        title = ""

    authors: list[str] = []
    for name_el in mods.findall(f"{{{_MODS_NS}}}name"):
        # Treat missing type as "personal" (some MODS variants omit the
        # attribute on author entries); skip explicit non-personal types
        # like "corporate".
        ntype = name_el.get("type")
        if ntype is not None and ntype != "personal":
            continue
        if not _is_author_role(name_el):
            continue
        given_parts: list[str] = []
        family_parts: list[str] = []
        plain_parts: list[str] = []
        for part in name_el.findall(f"{{{_MODS_NS}}}namePart"):
            text = _text(part)
            if not text:
                continue
            ptype = part.get("type")
            if ptype == "given":
                given_parts.append(text)
            elif ptype == "family":
                family_parts.append(text)
            else:
                # Untyped + termsOfAddress (e.g. "Jr.", "III") — append
                # after the typed parts so generation suffixes survive.
                plain_parts.append(text)
        combined = " ".join([*given_parts, *family_parts, *plain_parts]).strip()
        if combined:
            authors.append(combined)

    abstract = _text(mods.find(f"{{{_MODS_NS}}}abstract"))

    origin = mods.find(f"{{{_MODS_NS}}}originInfo")
    year_raw = _text(origin.find(f"{{{_MODS_NS}}}dateIssued")) if origin is not None else ""
    if not year_raw[:4].isdigit():
        raise IngestExtractionFailed(
            f"ACL Anthology MODS for {acl_id!r}: <dateIssued> missing or not "
            f"4-digit year (got {year_raw!r}); papers.date would be empty and "
            "break downstream date filters"
        )
    published = f"{year_raw[:4]}-01-01"

    return _ArxivMetadata(
        title=title,
        authors=authors,
        abstract=abstract,
        published=published,
        comment=None,
        summary=abstract or None,
        pdf_url=acl_pdf_url(acl_id),
    )


def _default_acl_lookup(client: httpx.Client, acl_id: str) -> _ArxivMetadata:
    xml_text = _get_acl_xml(client, acl_id)
    return _parse_mods(xml_text, acl_id)


def fetch(
    *,
    conn: sqlite3.Connection,
    acl_id: str,
    force: bool = False,
    domain_override: str | None = None,
    client: httpx.Client | None = None,
    acl_lookup: Callable[[httpx.Client, str], _ArxivMetadata] | None = None,
) -> PaperMetadata:
    """Fetch an ACL Anthology paper and persist a PDF-fallback ``papers`` row.

    Test hook: pass ``client=`` to inject a stubbed ``httpx.Client``, and
    optionally ``acl_lookup=`` to bypass the MODS GET. Production callers
    pass nothing and the defaults kick in.
    """
    from _system.pdf.extract import extract_markdown

    owns_client = client is None
    client = client or _make_default_client()
    lookup = acl_lookup or _default_acl_lookup

    namespaced_id = f"acl:{acl_id}"
    try:
        existing_row = _get_existing_paper(conn, namespaced_id)
        if existing_row is not None and not force:
            _LOG.info(
                "paper %s already present (status=%s), skipping fetch",
                namespaced_id, existing_row.status,
            )
            return existing_row

        meta = lookup(client, acl_id)
        pdf_bytes = _download_acl_pdf(client, acl_id)
        markdown = extract_markdown(pdf_bytes)
        if markdown is None:
            raise IngestExtractionFailed(
                f"ACL Anthology PDF for {acl_id!r}: pymupdf4llm produced no "
                "usable markdown"
            )
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()

        # client=None disables repo discovery in _persist_pdf_row (matches
        # the local-PDF convention). ACL papers don't carry a
        # PaperswithCode link keyed off arxiv_id, so the PwC lookup would
        # always miss; and the abstract isn't a reliable repo source for
        # this path.
        return _persist_pdf_row(
            conn,
            arxiv_id=namespaced_id,
            meta=meta,
            markdown=markdown,
            content_hash=content_hash,
            domain_override=domain_override,
            existing_row=existing_row,
            client=None,
        )
    finally:
        if owns_client:
            client.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch an ACL Anthology paper.")
    parser.add_argument("--acl", required=True, help="ACL Anthology id or URL")
    parser.add_argument("--db", default="lodestone.db", help="path to sqlite db")
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even if present")
    parser.add_argument("--domain", default=None, help="domain override")
    args = parser.parse_args(argv)

    acl_id = parse_acl_id(args.acl)
    conn = get_conn(Path(args.db))
    try:
        pm = fetch(
            conn=conn,
            acl_id=acl_id,
            force=args.force,
            domain_override=args.domain,
        )
    finally:
        conn.close()
    print(f"{pm.arxiv_id}: {pm.status} (paper_name={pm.paper_name})")


if __name__ == "__main__":
    main()
