"""Convert a FETCHED post's raw_html to markdown via trafilatura.

Pure compute, no network. Mirrors :mod:`_system.scripts.convert_paper`.

Failure shape:

* Empty / very-short markdown (< 200 chars after extract) → ``status =
  failed_parse`` with ``needs_review = 1``. The page parsed enough at
  fetch-time to extract metadata, but the body content didn't survive
  trafilatura's article extractor — usually a JS-only Notion / Substack
  skeleton DOM.

References: outbound arxiv-id mentions are pulled from BOTH the
markdown body (post-extract) and the original HTML's ``<a href>`` URLs
(catches links whose anchor text is "this paper" but whose target is
``arxiv.org/abs/<id>``). Forward + backward citation resolution runs
through the shared
:func:`_system.utils.citation_resolution.resolve_arxiv_citations` helper.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import lxml.html
import trafilatura

from _system.db.connection import get_conn, transaction
from _system.schemas.post_metadata import PostStatus, can_run_from
from _system.utils.arxiv_urls import (
    extract_arxiv_id_from_text,
    iter_arxiv_id_matches,
)
from _system.utils.citation_resolution import resolve_arxiv_citations
from _system.utils.source_resolution import SourceKind
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.convert_post")

_MIN_USEFUL_MARKDOWN_CHARS = 200
_REFERENCE_CONTEXT_CHARS = 240


class PostNotFound(Exception):
    pass


class RawHtmlMissing(Exception):
    pass


class StageNotAllowed(Exception):
    pass


@dataclass
class _PostReference:
    cited_arxiv_id: str
    raw_text: str


class ConvertPostResult(NamedTuple):
    post_name: str
    status: str
    markdown_chars: int
    references: int
    references_resolved_forward: int


def convert(
    *,
    post_name: str,
    conn: sqlite3.Connection,
    force: bool = False,
) -> ConvertPostResult:
    """Convert a FETCHED post's raw_html to markdown; update the posts row."""
    del force  # parity with paper-side; --force must cascade to fetch

    row = conn.execute(
        """
        SELECT id, status, raw_html
          FROM posts WHERE post_name = ?
        """,
        (post_name,),
    ).fetchone()
    if row is None:
        raise PostNotFound(f"post_name={post_name!r} not found in posts table")
    post_id, status_str, raw_html = row

    try:
        current = PostStatus(status_str)
    except ValueError as exc:
        raise StageNotAllowed(
            f"post_name={post_name!r}: unrecognized status={status_str!r}"
        ) from exc
    if not can_run_from(current, PostStatus.CONVERTED):
        extra = (
            " (terminal failure — re-fetch required)"
            if current in (PostStatus.FAILED_FETCH, PostStatus.FAILED_PARSE)
            else ""
        )
        raise StageNotAllowed(
            f"post_name={post_name!r}: cannot run CONVERTED from status="
            f"{status_str!r}{extra}"
        )

    if raw_html is None:
        raise RawHtmlMissing(f"post_name={post_name!r}: raw_html is NULL")

    markdown = trafilatura.extract(
        raw_html,
        output_format="markdown",
        include_links=True,
        include_images=True,
        include_tables=True,
        with_metadata=False,
        favor_precision=True,
    ) or ""

    references = _extract_arxiv_references(raw_html, markdown)

    needs_review = False
    target_status = PostStatus.CONVERTED
    if len(markdown) < _MIN_USEFUL_MARKDOWN_CHARS:
        _LOG.warning(
            "post_name=%s: trafilatura returned %d chars (< %d) — "
            "marking FAILED_PARSE",
            post_name, len(markdown), _MIN_USEFUL_MARKDOWN_CHARS,
        )
        needs_review = True
        target_status = PostStatus.FAILED_PARSE

    with transaction(conn):
        conn.execute(
            """
            UPDATE posts
               SET markdown = ?,
                   raw_html = NULL,
                   status   = ?,
                   needs_review = CASE WHEN ? = 1 THEN 1 ELSE needs_review END
             WHERE post_name = ?
            """,
            (markdown, target_status.value, int(needs_review), post_name),
        )

        # Replace-all on rerun: the extractor is the source of truth.
        conn.execute(
            "DELETE FROM post_references WHERE post_id = ?", (post_id,)
        )
        if references:
            conn.executemany(
                """
                INSERT INTO post_references (post_id, cited_arxiv_id, raw_text)
                VALUES (?, ?, ?)
                """,
                [
                    (post_id, r.cited_arxiv_id, r.raw_text)
                    for r in references
                ],
            )

        forward_resolved, _ = resolve_arxiv_citations(
            conn,
            kind=SourceKind.POST,
            source_id=post_id,
            source_arxiv_id=None,
        )

    _LOG.info(
        "converted post_id=%s post_name=%s markdown_chars=%d references=%d "
        "forward_resolved=%d status=%s",
        post_id, post_name, len(markdown), len(references),
        forward_resolved, target_status.value,
    )
    return ConvertPostResult(
        post_name=post_name,
        status=target_status.value,
        markdown_chars=len(markdown),
        references=len(references),
        references_resolved_forward=forward_resolved,
    )


def _extract_arxiv_references(
    raw_html: str, markdown: str,
) -> list[_PostReference]:
    """Pull outbound arxiv references from markdown body + raw <a href> URLs.

    Two passes because they catch different shapes:

    1. Markdown-body scan finds explicit ``arXiv:NNNN.NNNNN`` mentions
       and inline ``arxiv.org/abs/...`` URLs.
    2. ``<a href>`` scan in the original HTML catches links whose anchor
       text is uninformative ("this paper", "previous work") but whose
       href targets an arxiv abs/pdf URL.

    Dedup by ``cited_arxiv_id``; first-seen wins. The resulting
    ``raw_text`` is a short context window from whichever surface matched
    first — useful for surfacing the citation in search.
    """
    seen: set[str] = set()
    refs: list[_PostReference] = []

    for arxiv_id, match in iter_arxiv_id_matches(markdown):
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)
        window_start = max(0, match.start() - 40)
        window_end = min(len(markdown), match.end() + 200)
        refs.append(_PostReference(
            cited_arxiv_id=arxiv_id,
            raw_text=_trim_context(markdown[window_start:window_end]),
        ))

    try:
        tree = lxml.html.fromstring(raw_html)
    except (lxml.etree.ParserError, ValueError):
        return refs
    for href in tree.xpath("//a/@href"):
        if not isinstance(href, str):
            continue
        arxiv_id = extract_arxiv_id_from_text(href)
        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            refs.append(_PostReference(
                cited_arxiv_id=arxiv_id,
                raw_text=href[:_REFERENCE_CONTEXT_CHARS],
            ))

    return refs


def _trim_context(text: str) -> str:
    """Squeeze whitespace and cap at the reference context window."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:_REFERENCE_CONTEXT_CHARS]


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a FETCHED post's raw_html to markdown."
    )
    parser.add_argument("--post", required=True, help="posts.post_name")
    parser.add_argument(
        "--force",
        action="store_true",
        help="no-op for convert; forwarded for parity with other stages",
    )
    parser.add_argument("--db", default="lodestone.db", help="sqlite db path")
    args = parser.parse_args(argv)

    conn = get_conn(args.db)
    try:
        result = convert(post_name=args.post, conn=conn, force=args.force)
    finally:
        conn.close()
    print(json.dumps(result._asdict()))


if __name__ == "__main__":
    _main()
