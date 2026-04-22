"""Two-phase fetch of an arxiv paper.

Phase 1: arxiv metadata + HTML discovery + PDF download + page rendering +
LaTeXML parse + figure downloads + code-repo discovery. Pure in-memory; no
open sqlite transaction. Phase 2: a single ``BEGIN/COMMIT`` that writes
``papers`` + ``figures`` + ``page_images``.

On both HTML hosts failing (arxiv.org/html then ar5iv.labs.arxiv.org),
persists a stub ``papers`` row with ``status=failed_html`` and returns
without raising — ``search.py --needs-review`` surfaces the failure.

All outbound HTTP carries the ``Lodestone/1.0`` User-Agent header. Retries
are 3-attempt exponential backoff, triggered only on 5xx / transport
errors (per project policy: never swallow unexpected exceptions).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from _system.db.connection import get_conn, transaction
from _system.html.latexml_parser import FigureDescriptor, parse as parse_latexml
from _system.schemas.paper_metadata import HtmlSource, PaperMetadata, PaperStatus
from _system.utils.arxiv_urls import base_url_for_source
from _system.utils.logging import get_logger
from _system.utils.slug import generate_paper_name

_LOG = get_logger("scripts.fetch_paper")

USER_AGENT = "Lodestone/1.0 (mailto:pierce.lamb@getwhys.io)"

# Per-image decompression-bomb guard; well above 1920²×4 but low enough
# that a malicious PNG cannot OOM the worker. Catch `DecompressionBombError`
# per image and skip; never abort the whole paper.
Image.MAX_IMAGE_PIXELS = 256 * 1024 * 1024

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_WIDTH = 1920
_PDF_RENDER_DPI = 300
_JPEG_QUALITY = 85
_PWC_RATE_SLEEP_S = 1.0

_ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"
_AR5IV_HTML_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
_ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

_PWC_PAPER_LOOKUP = "https://paperswithcode.com/api/v1/papers/?arxiv_id={arxiv_id}"
_PWC_PAPER_REPOS = "https://paperswithcode.com/api/v1/papers/{slug}/repositories/"

_REPO_HOSTS = frozenset({"github.com", "gitlab.com", "bitbucket.org"})
_REPO_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/[^\s\"'<>]+",
    re.IGNORECASE,
)
_TRAILING_PUNCT = ".,);:!?]}>"

_VERSION_RE = re.compile(r"v\d+$")
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")


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


def _make_default_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    )


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return isinstance(exc, httpx.TransportError)


_retry_http = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=0.5, max=4.0),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)


def _default_arxiv_lookup(arxiv_id: str) -> _ArxivMetadata:
    """Query the arxiv API for a single id. Version suffix is accepted."""
    import arxiv  # lazy — tests monkeypatch this entry point

    results = list(arxiv.Search(id_list=[arxiv_id]).results())
    if not results:
        raise RuntimeError(f"arxiv API returned no result for {arxiv_id!r}")
    return _result_to_metadata(results[0])


def _result_to_metadata(result) -> _ArxivMetadata:
    authors = [getattr(a, "name", None) or str(a) for a in result.authors]
    published = result.published
    if hasattr(published, "strftime"):
        published_str = published.strftime("%Y-%m-%d")
    else:
        published_str = str(published)[:10]
    pdf_url = getattr(result, "pdf_url", None)
    if not pdf_url:
        pdf_url = _ARXIV_PDF_URL.format(arxiv_id=_strip_version(str(result.entry_id).rsplit("/", 1)[-1]))
    return _ArxivMetadata(
        title=result.title.strip(),
        authors=authors,
        abstract=(getattr(result, "summary", "") or "").strip(),
        published=published_str,
        comment=getattr(result, "comment", None),
        summary=getattr(result, "summary", None),
        pdf_url=pdf_url,
    )


def _strip_version(arxiv_id: str) -> str:
    return _VERSION_RE.sub("", arxiv_id)


@_retry_http
def _try_html(client: httpx.Client, url: str) -> str | None:
    """GET `url`, return body if 2xx + text/html, else None on 404 / non-html.

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


def _render_pdf_pages(pdf_bytes: bytes) -> list[bytes]:
    """Render each PDF page to a PNG byte string at 300 DPI."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(pdf_bytes)
    scale = _PDF_RENDER_DPI / 72.0
    out: list[bytes] = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                pil_image = page.render(scale=scale).to_pil()
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                out.append(buf.getvalue())
            except Image.DecompressionBombError as exc:
                _LOG.warning("page %d rendered an oversize image, skipping (%s)", i, exc)
            finally:
                try:
                    page.close()
                except Exception:  # pragma: no cover — pdfium quirks
                    pass
    finally:
        pdf.close()
    return out


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
    """Decode, optionally downscale to 1920 width (aspect preserved), re-encode
    JPEG at quality=85 or preserve PNG. Returns (bytes, mime) or None on
    decompression-bomb / decode error."""
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

    fmt = (img.format or "").upper()
    resized = False
    if img.width > _MAX_IMAGE_WIDTH:
        # thumbnail() only downsamples; the huge height cap preserves
        # aspect ratio without ever upscaling.
        img.thumbnail((_MAX_IMAGE_WIDTH, 10_000_000), Image.LANCZOS)
        resized = True

    out = io.BytesIO()
    if fmt == "JPEG" or "jpeg" in content_type_hint.lower():
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(out, format="JPEG", quality=_JPEG_QUALITY)
        return out.getvalue(), "image/jpeg"

    if fmt == "PNG" or "png" in content_type_hint.lower():
        if resized:
            img.save(out, format="PNG")
            return out.getvalue(), "image/png"
        return raw_bytes, "image/png"

    # Fallback: preserve format via PIL save if possible, else PNG.
    save_fmt = fmt or "PNG"
    try:
        img.save(out, format=save_fmt)
        return out.getvalue(), f"image/{save_fmt.lower()}"
    except (KeyError, OSError):
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue(), "image/png"


def _resolve_figure_bytes(
    client: httpx.Client, desc: FigureDescriptor
) -> tuple[bytes, str] | None:
    if desc.inline_data is not None:
        return _process_figure_image(desc.inline_data, desc.inline_mime or "")
    if desc.src_url is None:
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


def _normalize_repo_url(raw: str) -> str | None:
    """Return a canonical ``https://host/owner/repo`` URL, or None if the
    input isn't a repo root.

    The plan requires path depth **exactly 2** (``/owner/repo``). A deeper
    path (``/owner/repo/issues/42``, ``/owner/repo/blob/main/file.md``) is
    rejected outright rather than truncated — truncation produces
    false-positive repo links for bibliography entries that happen to cite
    a specific file on github.
    """
    raw = raw.strip().rstrip(_TRAILING_PUNCT)
    try:
        parsed = urlparse(raw)
    except ValueError:
        return None
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in _REPO_HOSTS:
        return None
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) != 2:
        return None
    owner, repo = segments[0], segments[1]
    repo = repo.removesuffix(".git")
    return f"https://{host}/{owner}/{repo}"


def _extract_repo_candidates(text: str) -> list[str]:
    return [
        norm
        for raw in _REPO_URL_RE.findall(text)
        if (norm := _normalize_repo_url(raw)) is not None
    ]


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
        if url and (norm := _normalize_repo_url(url)):
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
    if html_hits := _extract_repo_candidates(html_body):
        return html_hits[0]
    haystack = " ".join(s for s in (meta.comment or "", meta.summary or "") if s)
    if meta_hits := _extract_repo_candidates(haystack):
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
               html_source, content_hash, code_repo, needs_review,
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
        code_repo=row[14],
        needs_review=bool(row[15]),
        ingested_at=row[16],
    )


def _existing_paper_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("SELECT paper_name FROM papers").fetchall()
    return {r[0] for r in rows}


def _log_soft_dedup(conn: sqlite3.Connection, content_hash: str, arxiv_id: str) -> None:
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
    page_images: list[bytes],
) -> None:
    """Phase 2: single transaction writing papers + figures + page_images.

    Idempotency: on a forced re-fetch, any existing papers row (and every
    dependent row across ``figures``, ``page_images``, ``entities``,
    ``paper_topics``, and the FTS ``abstracts`` / ``sections`` virtual
    tables) is deleted inside the same transaction before the fresh INSERT.
    Callers that want paper_name / ingested_at preservation must set those
    on the incoming PaperMetadata — ``fetch()`` does this.
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
            old_id = existing[0]
            # Clear every FK-backed child (PRAGMA foreign_keys=ON in
            # connection.py). Missing any of these would raise
            # FOREIGN KEY constraint failed on the paper DELETE.
            conn.execute("DELETE FROM figures WHERE paper_id = ?", (old_id,))
            conn.execute("DELETE FROM page_images WHERE paper_id = ?", (old_id,))
            conn.execute("DELETE FROM entities WHERE paper_id = ?", (old_id,))
            conn.execute("DELETE FROM paper_topics WHERE paper_id = ?", (old_id,))
            # FTS tables: paper_id is UNINDEXED (no real FK), but the
            # rows would otherwise linger and pollute search results.
            conn.execute("DELETE FROM abstracts WHERE paper_id = ?", (old_id,))
            conn.execute("DELETE FROM sections WHERE paper_id = ?", (old_id,))
            conn.execute("DELETE FROM papers WHERE id = ?", (old_id,))

        cursor = conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, paper_name, title, authors, date, abstract,
                domain, collection, code_repo, content_hash, pdf_url,
                html_source, ingested_at, status, markdown, raw_html,
                figure_count, needs_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pm.arxiv_id, pm.paper_name, pm.title, pm.authors, pm.date,
                pm.abstract, pm.domain, pm.collection, pm.code_repo,
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

        if page_images:
            conn.executemany(
                "INSERT INTO page_images (paper_id, page_number, image) VALUES (?, ?, ?)",
                [(paper_id, i + 1, img) for i, img in enumerate(page_images)],
            )


def fetch(
    *,
    conn: sqlite3.Connection,
    arxiv_id: str,
    force: bool = False,
    domain_override: str | None = None,
    client: httpx.Client | None = None,
    arxiv_lookup: Callable[[str], _ArxivMetadata] | None = None,
    render_pages: Callable[[bytes], list[bytes] | None] = None,
) -> PaperMetadata:
    """Two-phase fetch of an arxiv paper. See module docstring for details.

    Keyword-only; all stage functions in the ingest pipeline share this
    contract (see section 14). Test hooks (``client``, ``arxiv_lookup``,
    ``render_pages``) let unit tests skip the network and PDF rendering
    deterministically; production callers pass nothing and the defaults
    kick in.
    """
    owns_client = client is None
    client = client or _make_default_client()
    arxiv_lookup = arxiv_lookup or _default_arxiv_lookup
    render_pages = render_pages or _render_pdf_pages

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
            return _persist_failed_html(
                conn, arxiv_id, meta, domain_override, existing_row
            )

        pdf_bytes = _download_pdf(client, arxiv_id)
        content_hash = hashlib.sha256(pdf_bytes).hexdigest()

        _log_soft_dedup(conn, content_hash, arxiv_id)

        page_images = render_pages(pdf_bytes)

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
        _persist(conn, pm, processed_figures, page_images)
        return pm
    finally:
        if owns_client:
            client.close()


def _persist_failed_html(
    conn: sqlite3.Connection,
    arxiv_id: str,
    meta: _ArxivMetadata,
    domain_override: str | None,
    existing_row: PaperMetadata | None,
) -> PaperMetadata:
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
        status=PaperStatus.FAILED_HTML,
        markdown=None,
        raw_html=None,
        html_source=None,
        content_hash=None,
        code_repo=None,
        needs_review=False,
        ingested_at=ingested_at,
    )
    _persist(conn, pm, figures=[], page_images=[])
    return pm


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
            meta.title, meta.published, arxiv_id, _existing_paper_names(conn),
        )
    )
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return paper_name, ingested_at


def _parse_arxiv_id_from_url(raw: str) -> str:
    """Accept either a bare id (``2301.12345`` / ``2301.12345v2``) or a full
    arxiv URL. Version suffix is preserved — that's the identity policy."""
    m = _ARXIV_ID_RE.search(raw)
    if not m:
        raise ValueError(f"could not locate arxiv id in {raw!r}")
    return m.group(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch an arxiv paper.")
    parser.add_argument("--url", required=True, help="arxiv URL or id")
    parser.add_argument("--db", default="lodestone.db", help="path to sqlite db")
    parser.add_argument("--force", action="store_true", help="re-fetch even if present")
    parser.add_argument("--domain", default=None, help="domain override")
    args = parser.parse_args(argv)

    arxiv_id = _parse_arxiv_id_from_url(args.url)
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
