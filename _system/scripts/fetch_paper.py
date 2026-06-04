"""Two-phase fetch of an arxiv paper.

Phase 1: arxiv metadata + HTML discovery + PDF download (for content-hash
dedup only) + LaTeXML parse + figure downloads + code-repo discovery.
Pure in-memory; no open sqlite transaction. Phase 2: a single
``BEGIN/COMMIT`` that writes ``papers`` + ``figures``.

On both HTML hosts failing (arxiv.org/html then ar5iv.labs.arxiv.org),
persists a stub ``papers`` row with ``status=failed_html`` and returns
without raising — ``search.py --needs-review`` surfaces the failure.

All outbound HTTP carries the ``Lodestone/1.0`` User-Agent header. The
arxiv metadata API (``export.arxiv.org/api/query``) is gated by a
file-locked 3.1s throttle (``_system.utils.arxiv_throttle``) and uses a
stricter retry policy (``retry_arxiv_api``): 3 attempts, 429-only,
honors ``Retry-After``; transport errors are *not* retried because
they indicate arxiv has escalated to silent IP-throttling. All other
outbound HTTP (HTML hosts, PDF, e-print, figures) uses ``retry_http``:
6 attempts with deterministic exponential backoff (3s, 6s, 12s, 24s,
48s capped at 60s) on 5xx / 429 / transport errors. Note we hit
``export.arxiv.org`` directly rather than via the ``arxiv`` Python
library — the library's hardcoded ``arxiv.py/<v>`` UA shares a global
throttle bucket and 429s constantly.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sqlite3
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import httpx
from PIL import Image

from _system.db.cascade import delete_paper_cascade
from _system.db.connection import get_conn, transaction
from _system.html.latexml_parser import FigureDescriptor, parse as parse_latexml
from _system.latex import LATEX_SENTINEL_PREFIX
from _system.latex import assemble as latex_assemble
from _system.latex import eprint as latex_eprint
from _system.latex import figures as latex_figures
from _system.pdf import PDF_SENTINEL_PREFIX
from _system.schemas.paper_metadata import HtmlSource, PaperMetadata, PaperStatus
from _system.utils.arxiv_throttle import wait_for_arxiv_slot
from _system.utils.arxiv_urls import base_url_for_source, parse_arxiv_id
from _system.utils.http import (
    USER_AGENT,
    is_transient as _is_transient,
    make_default_client as _make_default_client,
    retry_arxiv_api as _retry_arxiv_api,
    retry_http as _retry_http,
)
from _system.utils.logging import get_logger
from _system.utils.repo_url import extract_repo_candidates, normalize_repo_url
from _system.utils.slug import existing_slugs, generate_paper_name

__all__ = [
    "fetch",
    "IngestExtractionFailed",
    "LATEX_SENTINEL_PREFIX",
    "PDF_SENTINEL_PREFIX",
]

_LOG = get_logger("scripts.fetch_paper")


class IngestExtractionFailed(RuntimeError):
    """Raised when HTML, LaTeX-source, AND PDF extraction all fail.

    Surfaces out of ``fetch()`` -> ``ingest()`` -> MCP/CLI as a clear,
    loud failure rather than the prior silent ``failed_html`` stub row.
    """

# Per-image decompression-bomb guard; well above 1920²×4 but low enough
# that a malicious PNG cannot OOM the worker. Catch `DecompressionBombError`
# per image and skip; never abort the whole paper.
Image.MAX_IMAGE_PIXELS = 256 * 1024 * 1024

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_WIDTH = 1280
_JPEG_QUALITY = 85
_PWC_RATE_SLEEP_S = 1.0

_ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"
_AR5IV_HTML_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

_PWC_PAPER_LOOKUP = "https://paperswithcode.com/api/v1/papers/?arxiv_id={arxiv_id}"
_PWC_PAPER_REPOS = "https://paperswithcode.com/api/v1/papers/{slug}/repositories/"

_VERSION_RE = re.compile(r"v\d+$")


@dataclass
class _ArxivMetadata:
    """Shape returned by the arxiv library, reshaped for internal use."""

    title: str
    authors: list[str]
    abstract: str
    published: str  # YYYY-MM-DD
    comment: str | None
    summary: str | None
    pdf_url: str


@dataclass
class _ProcessedFigure:
    """Figure ready for DB insert: bytes already downscaled/re-encoded."""

    figure_number: int
    display_number: str | None
    figure_id: str
    caption: str
    section_context: str
    image_bytes: bytes
    mime_type: str


_ARXIV_API_URL = "https://export.arxiv.org/api/query?id_list={arxiv_id}"

_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


@_retry_arxiv_api
def _arxiv_api_get(arxiv_id: str) -> str:
    """GET arxiv's Atom export with our Lodestone UA, gated by the global throttle.

    The official ``arxiv`` Python lib sends ``user-agent: arxiv.py/<v>``,
    which shares a global throttle bucket with every other user of that
    library and 429s constantly. Our project UA carries a contact email
    and lands in arxiv's normal-citizen rate class.

    Pre-call: ``wait_for_arxiv_slot()`` blocks until at least 3.1s have
    elapsed since the most recent arxiv API call from this machine
    (persisted across processes). This is the only mechanism that
    actually prevents 429s in the user-driven CLI workflow — retries
    can recover from a single slip but not a sustained pattern.

    Retries: 429 and 503 (the documented arxiv flow-control signals)
    plus one bounded transport-error retry (Fastly first-byte-timeout,
    network artifacts, macOS sleep wedges); 3 attempts total, honoring
    Retry-After on HTTPStatusError or falling back to 60s/120s waits.
    Other 4xx and 5xx (404, 502, 504, etc.) surface immediately so the
    caller can react. The caller (`_default_arxiv_lookup`) re-wraps
    post-retry exceptions into a more actionable RuntimeError for the
    MCP envelope.

    Read timeout is 15s rather than the default 30s — arxiv responds in
    well under a second when not throttling, and a long timeout just
    extends the wait when the connection has been silently dropped (one
    bounded retry above gives self-healing for transient wedges).
    """
    wait_for_arxiv_slot()
    url = _ARXIV_API_URL.format(arxiv_id=arxiv_id)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=15.0) as c:
        resp = c.get(url)
        resp.raise_for_status()
        return resp.text


def _default_arxiv_lookup(arxiv_id: str) -> _ArxivMetadata:
    """Query arxiv's Atom export for a single id. Version suffix accepted."""
    import xml.etree.ElementTree as ET

    try:
        xml_text = _arxiv_api_get(arxiv_id)
    except httpx.TransportError as exc:
        raise RuntimeError(
            f"arxiv API did not respond within retry budget for {arxiv_id!r} "
            f"(possible arxiv throttling, CDN first-byte-timeout, or local "
            f"network issue — wait a few minutes and retry). "
            f"raw: {type(exc).__name__}: {exc}"
        ) from exc
    except httpx.HTTPStatusError as exc:
        sc = exc.response.status_code
        if sc in (429, 503):
            raise RuntimeError(
                f"arxiv API rate-limited request for {arxiv_id!r} "
                f"(HTTP {sc} after retry budget exhausted — wait several "
                f"minutes and retry; arxiv asks for >=3s between requests "
                f"across all your machines)."
            ) from exc
        raise  # 4xx and other 5xx: keep original, those carry info already.
    root = ET.fromstring(xml_text)
    entry = root.find(f"{{{_ATOM_NS}}}entry")
    if entry is None:
        raise RuntimeError(f"arxiv API returned no result for {arxiv_id!r}")

    title_el = entry.find(f"{{{_ATOM_NS}}}title")
    summary_el = entry.find(f"{{{_ATOM_NS}}}summary")
    published_el = entry.find(f"{{{_ATOM_NS}}}published")
    comment_el = entry.find(f"{{{_ARXIV_NS}}}comment")
    title = (title_el.text or "").strip() if title_el is not None else ""
    summary = (summary_el.text or "").strip() if summary_el is not None else ""
    published_raw = (published_el.text or "").strip() if published_el is not None else ""
    published = published_raw[:10]  # YYYY-MM-DD prefix of ISO 8601
    comment = (comment_el.text or "").strip() if comment_el is not None else None

    authors: list[str] = []
    for author_el in entry.findall(f"{{{_ATOM_NS}}}author"):
        name_el = author_el.find(f"{{{_ATOM_NS}}}name")
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    pdf_url: str | None = None
    for link in entry.findall(f"{{{_ATOM_NS}}}link"):
        if link.get("type") == "application/pdf":
            pdf_url = link.get("href")
            break
    if not pdf_url:
        pdf_url = _ARXIV_PDF_URL.format(arxiv_id=_strip_version(arxiv_id))

    return _ArxivMetadata(
        title=title,
        authors=authors,
        abstract=summary,
        published=published,
        comment=comment,
        summary=summary or None,
        pdf_url=pdf_url,
    )


def _strip_version(arxiv_id: str) -> str:
    return _VERSION_RE.sub("", arxiv_id)


@_retry_http
def _try_html(client: httpx.Client, url: str) -> str | None:
    """GET `url`, return body if 2xx + text/html on an `/html/` path, else None.

    ar5iv 302-redirects to ``arxiv.org/abs/{id}`` when it has no rendering
    available — a 200 body shaped like the arxiv listing page. We reject
    any response whose final URL leaves the ``/html/`` path so the listing
    page never reaches the LaTeXML parser as if it were paper content.

    Raises HTTPStatusError on 5xx so tenacity can retry.
    """
    resp = client.get(url)
    if resp.status_code == 404:
        return None
    if 500 <= resp.status_code < 600:
        resp.raise_for_status()
    if resp.status_code != 200:
        _LOG.warning("html fetch %s returned %d", url, resp.status_code)
        return None
    content_type = resp.headers.get("content-type", "")
    if not content_type.lower().startswith("text/html"):
        _LOG.warning("html fetch %s returned %r, not text/html", url, content_type)
        return None
    if "/html/" not in resp.url.path:
        _LOG.info(
            "html fetch %s redirected off /html/ to %s; treating as no-rendering",
            url, resp.url,
        )
        return None
    return resp.text


def _fetch_html_body(
    client: httpx.Client, arxiv_id: str
) -> tuple[str | None, HtmlSource | None]:
    """Try arxiv.org/html first, then ar5iv. Returns (body, source) or (None, None)."""
    for source, url_tmpl in (
        (HtmlSource.ARXIV, _ARXIV_HTML_URL),
        (HtmlSource.AR5IV, _AR5IV_HTML_URL),
    ):
        url = url_tmpl.format(arxiv_id=arxiv_id)
        body = _try_html(client, url)
        if body is not None:
            return body, source
    return None, None


@_retry_http
def _download_pdf(client: httpx.Client, arxiv_id: str) -> bytes:
    url = _ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    resp = client.get(url)
    resp.raise_for_status()
    return resp.content


def _download_figure(
    client: httpx.Client, url: str
) -> tuple[bytes, str] | None:
    """Streamed GET enforcing a 20MB cap. Returns (bytes, content-type) or None
    if the server said 4xx, the content-length was too large, or body exceeded
    the cap mid-stream. Raises on 5xx after tenacity exhaustion."""

    @_retry_http
    def _do_request():
        # raise_for_status() must live inside the retried call so that
        # transient 5xx responses are surfaced to tenacity. A status check
        # after the decorator returns would only retry transport errors.
        req = client.build_request("GET", url)
        resp = client.send(req, stream=True)
        if 500 <= resp.status_code < 600:
            resp.close()
            resp.raise_for_status()
        return resp

    resp = _do_request()
    try:
        if 400 <= resp.status_code < 500:
            _LOG.warning("figure fetch %s returned %d, skipping", url, resp.status_code)
            return None
        cl_header = resp.headers.get("content-length")
        if cl_header and cl_header.isdigit() and int(cl_header) > _MAX_IMAGE_BYTES:
            _LOG.warning(
                "figure fetch %s content-length %s exceeds %d, skipping",
                url, cl_header, _MAX_IMAGE_BYTES,
            )
            return None
        content_type = resp.headers.get("content-type", "application/octet-stream")
        buf = bytearray()
        for chunk in resp.iter_bytes():
            buf.extend(chunk)
            if len(buf) > _MAX_IMAGE_BYTES:
                _LOG.warning(
                    "figure fetch %s exceeded %d bytes mid-stream, skipping",
                    url, _MAX_IMAGE_BYTES,
                )
                return None
        return bytes(buf), content_type
    finally:
        resp.close()


def _process_figure_image(
    raw_bytes: bytes, content_type_hint: str
) -> tuple[bytes, str] | None:
    """Decode, downscale to _MAX_IMAGE_WIDTH (aspect preserved), and re-encode
    as JPEG q=_JPEG_QUALITY. Sources carrying an alpha channel stay PNG —
    JPEG can't represent transparency. Returns (bytes, mime) or None on
    decompression-bomb / decode error.

    Re-encoding is unconditional: passing source PNGs through often leaves
    photographic / schematic figures at 3-5x the JPEG-equivalent byte size
    with no perceptible quality difference at our serving dimensions.
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except Image.DecompressionBombError as exc:
        _LOG.warning("figure is a decompression bomb, skipping (%s)", exc)
        return None
    except (OSError, ValueError, SyntaxError) as exc:
        # PIL raises these for truncated / unrecognized / malformed images;
        # MemoryError / AttributeError should surface so we can debug.
        _LOG.warning("figure decode failed, skipping (%s)", exc)
        return None

    if img.width > _MAX_IMAGE_WIDTH:
        # thumbnail() only downsamples; the huge height cap preserves
        # aspect ratio without ever upscaling.
        img.thumbnail((_MAX_IMAGE_WIDTH, 10_000_000), Image.LANCZOS)

    # Many arxiv figures arrive as RGBA with a fully-opaque alpha channel
    # (matplotlib's default save mode). Treat that as "no real alpha" so it
    # can re-encode to JPEG. Only keep PNG when transparency is actually
    # used somewhere in the image.
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        alpha = img.getchannel("A")
        if alpha.getextrema()[0] < 255:
            out = io.BytesIO()
            img.save(out, format="PNG", optimize=True)
            return out.getvalue(), "image/png"

    out = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(out, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
    return out.getvalue(), "image/jpeg"


def _resolve_figure_bytes(
    client: httpx.Client, desc: FigureDescriptor
) -> tuple[bytes, str] | None:
    if desc.inline_data is not None:
        return _process_figure_image(desc.inline_data, desc.inline_mime or "")
    if desc.src_url is None:
        _LOG.debug(
            "figure %d (id=%s) has no <img> source; skipping (placeholder)",
            desc.figure_number, desc.figure_id,
        )
        return None
    downloaded = _download_figure(client, desc.src_url)
    if downloaded is None:
        return None
    raw, ct = downloaded
    return _process_figure_image(raw, ct)


def _process_figures(
    client: httpx.Client, descriptors: list[FigureDescriptor]
) -> list[_ProcessedFigure]:
    results: list[_ProcessedFigure] = []
    for desc in descriptors:
        resolved = _resolve_figure_bytes(client, desc)
        if resolved is None:
            continue
        img_bytes, mime = resolved
        results.append(
            _ProcessedFigure(
                figure_number=desc.figure_number,
                display_number=desc.display_number,
                figure_id=desc.figure_id,
                caption=desc.caption,
                section_context=desc.section_context,
                image_bytes=img_bytes,
                mime_type=mime,
            )
        )
    return results


def _layer1_paperswithcode(
    client: httpx.Client, arxiv_id: str
) -> str | None:
    """PwC: /papers/?arxiv_id={id} → slug → /papers/{slug}/repositories/ →
    pick is_official else top by stars. 1 req/sec."""
    try:
        resp = client.get(_PWC_PAPER_LOOKUP.format(arxiv_id=_strip_version(arxiv_id)))
    except httpx.HTTPError as exc:
        _LOG.warning("PwC paper lookup failed: %s", exc)
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        _LOG.warning("PwC paper lookup returned non-JSON body: %s", exc)
        return None
    results = data.get("results") or []
    if not results:
        return None
    slug = results[0].get("id") or results[0].get("slug")
    if not slug:
        return None

    _sleep(_PWC_RATE_SLEEP_S)

    try:
        resp2 = client.get(_PWC_PAPER_REPOS.format(slug=slug))
    except httpx.HTTPError as exc:
        _LOG.warning("PwC repos fetch failed: %s", exc)
        return None
    if resp2.status_code != 200:
        return None
    try:
        data2 = resp2.json()
    except json.JSONDecodeError as exc:
        _LOG.warning("PwC repos fetch returned non-JSON body: %s", exc)
        return None
    repos = data2.get("results") or []
    if not repos:
        return None
    official = [r for r in repos if r.get("is_official")]
    pool = official or sorted(
        repos, key=lambda r: r.get("stars") or 0, reverse=True
    )
    for repo in pool:
        url = repo.get("url")
        if url and (norm := normalize_repo_url(url)):
            return norm
    return None


def _discover_code_repo(
    client: httpx.Client,
    arxiv_id: str,
    html_body: str,
    meta: _ArxivMetadata,
) -> str | None:
    if pwc := _layer1_paperswithcode(client, arxiv_id):
        return pwc
    if html_hits := extract_repo_candidates(html_body):
        return html_hits[0]
    haystack = " ".join(s for s in (meta.comment or "", meta.summary or "") if s)
    if meta_hits := extract_repo_candidates(haystack):
        return meta_hits[0]
    return None


def _sleep(seconds: float) -> None:
    # Wrapped so tests can monkeypatch without touching the stdlib.
    import time
    time.sleep(seconds)


def _get_existing_paper(conn: sqlite3.Connection, arxiv_id: str) -> PaperMetadata | None:
    row = conn.execute(
        """
        SELECT arxiv_id, paper_name, title, authors, date, abstract,
               pdf_url, domain, collection, status, markdown, raw_html,
               html_source, content_hash, needs_review,
               ingested_at
          FROM papers WHERE arxiv_id = ?
        """,
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None
    return PaperMetadata(
        arxiv_id=row[0],
        paper_name=row[1],
        title=row[2],
        authors=row[3],
        date=row[4],
        abstract=row[5],
        pdf_url=row[6],
        domain=row[7],
        collection=row[8],
        status=row[9],
        markdown=row[10],
        raw_html=row[11],
        html_source=row[12],
        content_hash=row[13],
        needs_review=bool(row[14]),
        ingested_at=row[15],
    )


def _log_soft_dedup(conn: sqlite3.Connection, content_hash: str, arxiv_id: str) -> None:
    if arxiv_id.startswith("pdf:"):
        # Local-PDF chapter rows of one book intentionally share
        # content_hash by design (siblings under pdf:<hash[:12]>:chNN).
        # Suppress the warning for in-book matches; only fire on
        # genuine cross-document collisions.
        book_prefix = arxiv_id.split(":ch", 1)[0]
        # hash is hex and book_prefix has no LIKE wildcards, so the
        # LIKE pattern is safe without ESCAPE.
        row = conn.execute(
            "SELECT arxiv_id FROM papers "
            " WHERE content_hash = ? "
            "   AND arxiv_id != ? "
            "   AND arxiv_id NOT LIKE ? "
            " LIMIT 1",
            (content_hash, arxiv_id, f"{book_prefix}%"),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT arxiv_id FROM papers WHERE content_hash = ? AND arxiv_id != ? LIMIT 1",
            (content_hash, arxiv_id),
        ).fetchone()
    if row is not None:
        _LOG.warning(
            "soft-dedup: content_hash %s also found under arxiv_id %s (current: %s); continuing",
            content_hash, row[0], arxiv_id,
        )


def _persist(
    conn: sqlite3.Connection,
    pm: PaperMetadata,
    figures: list[_ProcessedFigure],
) -> None:
    """Phase 2: single transaction writing papers + figures.

    Idempotency: on a forced re-fetch, any existing papers row (and every
    dependent row across ``figures``, ``term_aliases``, ``topics``,
    and the FTS ``sections`` virtual table) is deleted inside the same
    transaction before the fresh INSERT. Callers that want paper_name /
    ingested_at preservation must set those on the incoming
    PaperMetadata — ``fetch()`` does this.

    The discovered ``code_repo`` URL travels on ``pm`` for the orchestrator
    to consume after fetch returns; it is **not** persisted on the
    ``papers`` row. Repos are first-class entities and are created in a
    follow-up stage.
    """
    with transaction(conn):
        if pm.domain:
            # Classification hasn't run yet for a plain fetch, but a
            # `--domain` override (or a force-refetch where the row
            # was previously classified) can legitimately set this.
            conn.execute(
                "INSERT OR IGNORE INTO domains (name) VALUES (?)",
                (pm.domain,),
            )

        existing = conn.execute(
            "SELECT id FROM papers WHERE arxiv_id = ?", (pm.arxiv_id,)
        ).fetchone()
        if existing is not None:
            delete_paper_cascade(conn, paper_id=existing[0])

        cursor = conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, paper_name, title, authors, date, abstract,
                domain, collection, content_hash, pdf_url,
                html_source, ingested_at, status, markdown, raw_html,
                figure_count, needs_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pm.arxiv_id, pm.paper_name, pm.title, pm.authors, pm.date,
                pm.abstract, pm.domain, pm.collection,
                pm.content_hash, pm.pdf_url, pm.html_source,
                pm.ingested_at, str(pm.status), pm.markdown, pm.raw_html,
                len(figures), int(bool(pm.needs_review)),
            ),
        )
        paper_id = cursor.lastrowid

        if figures:
            conn.executemany(
                """
                INSERT INTO figures (
                    paper_id, figure_number, display_number, figure_id,
                    caption, section_context, image, mime_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (paper_id, f.figure_number, f.display_number,
                     f.figure_id, f.caption, f.section_context,
                     f.image_bytes, f.mime_type)
                    for f in figures
                ],
            )


@dataclass
class _LatexFetchResult:
    """Output of `_try_latex_fallback`: assembled LaTeX + processed figure rows."""

    assembled_tex: str
    processed_figures: list[_ProcessedFigure]


_FLAG_FALSY = frozenset({"0", "false", "no", "off", ""})


def _flag_enabled(env_var: str, explicit: bool | None) -> bool:
    """Resolve a default-on fallback toggle: kwarg > env > on."""
    if explicit is not None:
        return explicit
    return os.environ.get(env_var, "1").strip().lower() not in _FLAG_FALSY


def _try_latex_fallback(
    client: httpx.Client, arxiv_id: str
) -> _LatexFetchResult | None:
    """Download the e-print, assemble main.tex, materialize raster figures.

    Returns None on any unrecoverable step (no e-print, malformed tarball,
    no `\\documentclass`, oversize payload). The walker is NOT run here —
    convert_paper produces markdown. We only need raw figures and the
    assembled tex string; the walker re-runs at convert-time.
    """
    fetched = latex_eprint.fetch_eprint(client, arxiv_id)
    if fetched is None:
        return None
    blob, fmt = fetched
    try:
        with latex_eprint.extract_to_tempdir(blob, fmt) as tex_root:
            main = latex_assemble.find_main_tex(tex_root)
            if main is None:
                _LOG.info(
                    "e-print %s extracted but contains no \\documentclass file",
                    arxiv_id,
                )
                return None
            assembled = latex_assemble.assemble_source(main)
            descriptors = latex_figures.discover_figures(assembled, tex_root)
            processed: list[_ProcessedFigure] = []
            for desc in descriptors:
                pf = _process_latex_figure(desc)
                if pf is not None:
                    processed.append(pf)
            return _LatexFetchResult(
                assembled_tex=assembled, processed_figures=processed,
            )
    except (ValueError, OSError, tarfile.TarError) as exc:
        _LOG.warning("latex fallback for %s failed during extraction: %s",
                     arxiv_id, exc)
        return None


def _process_latex_figure(
    desc: latex_figures.LatexFigureDescriptor,
) -> _ProcessedFigure | None:
    """Read raster bytes from disk and run them through the existing
    downscale/re-encode pipeline used by the HTML path.

    Figures with `local_path is None` (TikZ-only, PDF/EPS/SVG, missing
    file) are not persisted — the walker emits a placeholder comment at
    the right ordinal so the markdown still reads correctly. The figure
    counter in markdown stays in sync because the walker counts
    `\\begin{figure}` envs, not DB rows.
    """
    raw = latex_figures.read_figure_bytes(desc)
    if raw is None:
        return None
    img_bytes, mime_hint = raw
    processed = _process_figure_image(img_bytes, mime_hint)
    if processed is None:
        return None
    out_bytes, mime = processed
    return _ProcessedFigure(
        figure_number=desc.figure_number,
        display_number=desc.display_number,
        figure_id=desc.figure_id,
        caption=desc.caption,
        section_context=desc.section_context,
        image_bytes=out_bytes,
        mime_type=mime,
    )


@dataclass
class _PdfFetchResult:
    """Output of `_try_pdf_fallback`: extracted markdown + content hash."""

    markdown: str
    content_hash: str


def _try_pdf_fallback(
    client: httpx.Client, arxiv_id: str
) -> _PdfFetchResult | None:
    """Download the PDF and run it through pymupdf4llm.

    Returns None if the PDF can't be fetched or pymupdf4llm produces no
    usable markdown. Reuses the same `_download_pdf` helper as the HTML
    path — the bytes also feed `content_hash` so paper identity stays
    hash-anchored across all fallback tiers.
    """
    from _system.pdf.extract import extract_markdown

    try:
        pdf_bytes = _download_pdf(client, arxiv_id)
    except httpx.HTTPStatusError as exc:
        _LOG.warning(
            "pdf fallback for %s: PDF download failed: %s", arxiv_id, exc,
        )
        return None

    md = extract_markdown(pdf_bytes)
    if md is None:
        _LOG.warning(
            "pdf fallback for %s: pymupdf4llm produced no usable markdown",
            arxiv_id,
        )
        return None
    return _PdfFetchResult(
        markdown=md,
        content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
    )


def fetch(
    *,
    conn: sqlite3.Connection,
    arxiv_id: str,
    force: bool = False,
    domain_override: str | None = None,
    client: httpx.Client | None = None,
    arxiv_lookup: Callable[[str], _ArxivMetadata] | None = None,
    latex_fallback: bool | None = None,
    pdf_fallback: bool | None = None,
) -> PaperMetadata:
    """Two-phase fetch of an arxiv paper. See module docstring for details.

    Keyword-only; all stage functions in the ingest pipeline share this
    contract (see section 14). Test hooks (``client``, ``arxiv_lookup``)
    let unit tests skip the network deterministically; production callers
    pass nothing and the defaults kick in.
    """
    owns_client = client is None
    client = client or _make_default_client()
    arxiv_lookup = arxiv_lookup or _default_arxiv_lookup

    try:
        # Early dedup: skip all network if this arxiv_id is already in the DB.
        # On force=True we keep the row's paper_name + ingested_at so the
        # slug stays stable across re-fetches (downstream references and
        # human-memory don't get clobbered).
        existing_row = _get_existing_paper(conn, arxiv_id)
        if existing_row is not None and not force:
            _LOG.info(
                "paper %s already present (status=%s), skipping fetch",
                arxiv_id, existing_row.status,
            )
            return existing_row

        meta = arxiv_lookup(arxiv_id)

        html_body, html_source = _fetch_html_body(client, arxiv_id)

        if html_body is None:
            if _flag_enabled("LODESTONE_LATEX_FALLBACK", latex_fallback):
                latex_result = _try_latex_fallback(client, arxiv_id)
                if latex_result is not None:
                    return _persist_latex_fallback(
                        conn,
                        arxiv_id=arxiv_id,
                        meta=meta,
                        latex_result=latex_result,
                        domain_override=domain_override,
                        existing_row=existing_row,
                        client=client,
                    )
            if _flag_enabled("LODESTONE_PDF_FALLBACK", pdf_fallback):
                pdf_result = _try_pdf_fallback(client, arxiv_id)
                if pdf_result is not None:
                    return _persist_pdf_fallback(
                        conn,
                        arxiv_id=arxiv_id,
                        meta=meta,
                        pdf_result=pdf_result,
                        domain_override=domain_override,
                        existing_row=existing_row,
                        client=client,
                    )
            raise IngestExtractionFailed(
                f"All extraction paths failed for arxiv_id={arxiv_id!r}: "
                "arxiv.org/html, ar5iv, e-print LaTeX, and PDF (pymupdf4llm) "
                "each returned no usable content."
            )

        pdf_bytes = _download_pdf(client, arxiv_id)
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()

        _log_soft_dedup(conn, content_hash, arxiv_id)

        base_url = base_url_for_source(html_source, arxiv_id)
        parsed = parse_latexml(html_body, base_url)

        processed_figures = _process_figures(client, parsed.figures)

        code_repo = _discover_code_repo(client, arxiv_id, html_body, meta)

        paper_name, ingested_at = _resolve_slug_and_timestamp(
            conn, meta, arxiv_id, existing_row,
        )

        pm = PaperMetadata(
            arxiv_id=arxiv_id,
            paper_name=paper_name,
            title=meta.title,
            authors=json.dumps(meta.authors),
            date=meta.published,
            abstract=meta.abstract,
            pdf_url=meta.pdf_url,
            domain=domain_override or (existing_row.domain if existing_row else None),
            collection=existing_row.collection if existing_row else None,
            status=PaperStatus.FETCHED,
            markdown=None,
            raw_html=html_body,
            html_source=html_source,
            content_hash=content_hash,
            code_repo=code_repo,
            needs_review=False,
            ingested_at=ingested_at,
        )
        _persist(conn, pm, processed_figures)
        return pm
    finally:
        if owns_client:
            client.close()


def _build_fallback_metadata(
    conn: sqlite3.Connection,
    *,
    client: httpx.Client | None,
    arxiv_id: str,
    meta: _ArxivMetadata,
    raw_html: str,
    html_source: HtmlSource,
    content_hash: str,
    domain_override: str | None,
    existing_row: PaperMetadata | None,
    paper_name_override: str | None = None,
) -> PaperMetadata:
    """Common prelude for both fallback persisters: soft-dedup log, code-repo
    discovery, slug resolution, then build the ``PaperMetadata`` row.

    Code-repo discovery skips the HTML body parse and goes straight to the
    arxiv comment + PwC. Local-PDF ingest passes ``client=None`` (or an
    ``arxiv_id`` starting with ``pdf:``) and the PwC roundtrip is skipped
    — there is no upstream metadata to resolve a code repo from. The
    local-PDF orchestrator pre-computes book/chapter slugs and passes them
    via ``paper_name_override`` so the chapter slug convention isn't
    clobbered by :func:`generate_paper_name`. needs_review is left False;
    classify is the sole writer of that column — it flips it True iff the
    LLM minted a brand-new domain or collection that a human should review.
    """
    _log_soft_dedup(conn, content_hash, arxiv_id)
    if client is None or arxiv_id.startswith("pdf:"):
        code_repo = None
    else:
        code_repo = _discover_code_repo(client, arxiv_id, "", meta)
    if paper_name_override is not None:
        paper_name = paper_name_override
        if existing_row is not None and existing_row.ingested_at:
            ingested_at = existing_row.ingested_at
        else:
            ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        paper_name, ingested_at = _resolve_slug_and_timestamp(
            conn, meta, arxiv_id, existing_row,
        )
    return PaperMetadata(
        arxiv_id=arxiv_id,
        paper_name=paper_name,
        title=meta.title,
        authors=json.dumps(meta.authors),
        date=meta.published,
        abstract=meta.abstract,
        pdf_url=meta.pdf_url,
        domain=domain_override or (existing_row.domain if existing_row else None),
        collection=existing_row.collection if existing_row else None,
        status=PaperStatus.FETCHED,
        markdown=None,
        raw_html=raw_html,
        html_source=html_source,
        content_hash=content_hash,
        code_repo=code_repo,
        needs_review=False,
        ingested_at=ingested_at,
    )


def _persist_latex_fallback(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str,
    meta: _ArxivMetadata,
    latex_result: _LatexFetchResult,
    domain_override: str | None,
    existing_row: PaperMetadata | None,
    client: httpx.Client,
) -> PaperMetadata:
    """Persist a paper that succeeded via the LaTeX-source fallback.

    The PDF is still downloaded for content-hash purposes (matches HTML
    path semantics — paper identity is hash-anchored, not source-anchored).
    """
    pdf_bytes = _download_pdf(client, arxiv_id)
    pm = _build_fallback_metadata(
        conn,
        client=client,
        arxiv_id=arxiv_id,
        meta=meta,
        raw_html=LATEX_SENTINEL_PREFIX + latex_result.assembled_tex,
        html_source=HtmlSource.LATEX_LOCAL,
        content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
        domain_override=domain_override,
        existing_row=existing_row,
    )
    _persist(conn, pm, latex_result.processed_figures)
    return pm


def _persist_pdf_row(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str,
    meta: _ArxivMetadata,
    markdown: str,
    content_hash: str,
    domain_override: str | None,
    existing_row: PaperMetadata | None,
    client: httpx.Client | None,
    paper_name_override: str | None = None,
) -> PaperMetadata:
    """Persist a single papers row whose body came from a PDF.

    Shared by the arxiv PDF-fallback path and the local-PDF ingest path.
    The markdown is stashed verbatim in ``raw_html`` behind
    :data:`PDF_SENTINEL_PREFIX` — convert_paper strips the sentinel and
    uses the markdown directly (no re-extraction at convert time). No
    figures (PDF path skips figure extraction by design). Pass
    ``paper_name_override`` to bypass :func:`generate_paper_name` (the
    local-PDF orchestrator computes chapter slugs up front).
    """
    pm = _build_fallback_metadata(
        conn,
        client=client,
        arxiv_id=arxiv_id,
        meta=meta,
        raw_html=PDF_SENTINEL_PREFIX + markdown,
        html_source=HtmlSource.PDF_FALLBACK,
        content_hash=content_hash,
        domain_override=domain_override,
        existing_row=existing_row,
        paper_name_override=paper_name_override,
    )
    _persist(conn, pm, figures=[])
    return pm


def _persist_pdf_fallback(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str,
    meta: _ArxivMetadata,
    pdf_result: _PdfFetchResult,
    domain_override: str | None,
    existing_row: PaperMetadata | None,
    client: httpx.Client,
) -> PaperMetadata:
    """Thin wrapper around :func:`_persist_pdf_row` for the arxiv PDF
    fallback path."""
    return _persist_pdf_row(
        conn,
        arxiv_id=arxiv_id,
        meta=meta,
        markdown=pdf_result.markdown,
        content_hash=pdf_result.content_hash,
        domain_override=domain_override,
        existing_row=existing_row,
        client=client,
    )


def _resolve_slug_and_timestamp(
    conn: sqlite3.Connection,
    meta: _ArxivMetadata,
    arxiv_id: str,
    existing_row: PaperMetadata | None,
) -> tuple[str, str]:
    """Preserve paper_name / ingested_at on a force re-fetch, else generate
    fresh values.

    Recomputing the slug on a re-fetch would collide with the row's own
    existing slug in the `existing` set (see `generate_paper_name`) and
    append a ``_NNNNN`` suffix, breaking any downstream references to the
    old name.
    """
    if existing_row is not None and existing_row.ingested_at:
        return existing_row.paper_name, existing_row.ingested_at
    paper_name = (
        existing_row.paper_name
        if existing_row is not None
        else generate_paper_name(
            meta.title, meta.published, arxiv_id, existing_slugs(conn),
        )
    )
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return paper_name, ingested_at


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch an arxiv paper.")
    parser.add_argument("--url", required=True, help="arxiv URL or id")
    parser.add_argument("--db", default="lodestone.db", help="path to sqlite db")
    parser.add_argument("--force", action="store_true", help="re-fetch even if present")
    parser.add_argument("--domain", default=None, help="domain override")
    args = parser.parse_args(argv)

    arxiv_id = parse_arxiv_id(args.url)
    conn = get_conn(Path(args.db))
    try:
        pm = fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            force=args.force,
            domain_override=args.domain,
        )
    finally:
        conn.close()
    print(f"{pm.arxiv_id}: {pm.status} (paper_name={pm.paper_name})")


if __name__ == "__main__":
    main()
