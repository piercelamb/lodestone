"""Two-phase fetch of a blog post URL.

Mirrors :mod:`_system.scripts.fetch_paper` but for arbitrary HTML pages.
Phase 1: HTTP GET, parse canonical URL, run trafilatura's metadata
extractor for title/author/site_name/description, run htmldate for the
publication date, scan the HTML body for repo URLs. Phase 2: a single
``BEGIN/COMMIT`` that writes one ``posts`` row at ``status='fetched'``.

Two terminal failure modes (vs paper's one ``FAILED_HTML``):
``FAILED_FETCH`` for network refusals / 4xx, and ``FAILED_PARSE`` for
HTML that trafilatura returned empty / unusable from. ``FAILED_PARSE``
is reached at convert-time, not here.

All outbound HTTP carries the shared ``Lodestone/1.0`` UA via
:mod:`_system.utils.http`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from urllib.parse import urljoin, urlparse

import httpx
import lxml.html
import trafilatura
from htmldate import find_date

from _system.db.cascade import delete_post_cascade
from _system.db.connection import get_conn, transaction
from _system.schemas.post_metadata import PostMetadata, PostStatus
from _system.utils.http import make_default_client, retry_http
from _system.utils.logging import get_logger
from _system.utils.repo_url import extract_repo_candidates
from _system.utils.slug import existing_slugs, generate_post_name, sanitize_domain

__all__ = ["fetch", "fetch_post", "fetch_from_file"]

_LOG = get_logger("scripts.fetch_post")


_CANONICAL_XPATH = "//link[@rel='canonical']/@href"
_OG_URL_XPATH = "//meta[@property='og:url']/@content"
_OG_SITE_XPATH = "//meta[@property='og:site_name']/@content"
_OG_DESC_XPATH = "//meta[@property='og:description']/@content"

_ABSTRACT_FALLBACK_CHARS = 400


@dataclass
class _ExtractedMetadata:
    title: str
    author: str | None
    site_name: str | None
    description: str
    date: str
    canonical_url: str


@retry_http
def _http_get(client: httpx.Client, url: str) -> httpx.Response:
    """Wrapped GET with our retry policy. 5xx / 429 / transport errors retry;
    4xx returns through to the caller for failed_fetch handling.
    """
    resp = client.get(url)
    if 500 <= resp.status_code < 600:
        resp.raise_for_status()
    return resp


def _resolve_canonical_url(html: str, post_redirect_url: str) -> str:
    """Pick the best canonical URL: <link rel=canonical> > og:url > redirect URL.

    Many sites (Jekyll/Hugo defaults included) emit root-relative hrefs
    like ``/blog/<slug>.html`` or protocol-relative ``//host/path``.
    We urljoin against the post-redirect URL so the stored canonical is
    always absolute. A canonical with no host after urljoin (degenerate
    page-served-without-protocol case) falls back to the redirect URL —
    posts.canonical_url is UNIQUE NOT NULL and a relative path could
    falsely collide across sites.
    """
    try:
        tree = lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):
        return post_redirect_url
    for xp in (_CANONICAL_XPATH, _OG_URL_XPATH):
        hits = tree.xpath(xp)
        if not hits:
            continue
        raw = (hits[0] or "").strip()
        if not raw:
            continue
        joined = urljoin(post_redirect_url, raw)
        parsed = urlparse(joined)
        if parsed.scheme and parsed.netloc:
            return joined
    return post_redirect_url


def _extract_meta_dom(html: str) -> tuple[str | None, str | None]:
    """Pull ``og:site_name`` and ``og:description`` straight from the DOM.

    Trafilatura already exposes ``sitename`` and ``description`` but the
    DOM lookup is cheap and lets us cross-check / fall back when
    trafilatura's sniffers disagree.
    """
    try:
        tree = lxml.html.fromstring(html)
    except (lxml.etree.ParserError, ValueError):
        return None, None
    site = tree.xpath(_OG_SITE_XPATH)
    desc = tree.xpath(_OG_DESC_XPATH)
    return (
        ((site[0] or "").strip() or None) if site else None,
        ((desc[0] or "").strip() or None) if desc else None,
    )


def _extract_metadata(
    html: str, fallback_url: str,
) -> _ExtractedMetadata | None:
    """Run trafilatura's bare_extraction + htmldate for the metadata block.

    Returns None when trafilatura can't recognize the input as an article;
    caller treats that as ``FAILED_FETCH`` (we got bytes but the page is
    not a readable post — e.g. a JS-only skeleton DOM).
    """
    doc = trafilatura.bare_extraction(
        html,
        url=fallback_url,
        with_metadata=True,
        favor_precision=True,
    )
    if doc is None:
        return None
    bare = doc.as_dict() if hasattr(doc, "as_dict") else doc

    title = (bare.get("title") or "").strip()
    if not title:
        # No title means we can't generate a post_name slug. Treat as
        # unparseable and let caller mark FAILED_FETCH.
        return None

    author = (bare.get("author") or "").strip() or None
    site_name = (bare.get("sitename") or "").strip() or None

    og_site, og_desc = _extract_meta_dom(html)
    if site_name is None:
        site_name = og_site

    description = (bare.get("description") or "").strip()
    if not description and og_desc:
        description = og_desc
    if not description:
        body_text = (bare.get("text") or "").strip()
        description = body_text[:_ABSTRACT_FALLBACK_CHARS]
    if not description:
        # Honest empty body — mark for review at convert time.
        description = title

    canonical_url = _resolve_canonical_url(html, fallback_url)

    date_str = find_date(html, original_date=True)
    if not date_str:
        # Last-resort: trafilatura's own date estimator.
        date_str = (bare.get("date") or "").strip()
    if not date_str:
        # Use today as a soft fallback; convert/classify can still run.
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return _ExtractedMetadata(
        title=title,
        author=author,
        site_name=site_name,
        description=description,
        date=date_str[:10],
        canonical_url=canonical_url,
    )


_POST_FULL_COLS = (
    "id, post_name, source_url, canonical_url, title, author, "
    "site_name, date, abstract, domain, collection, status, "
    "markdown, raw_html, content_hash, etag, last_modified, "
    "needs_review, ingested_at"
)


def _row_to_post(row: tuple) -> tuple[int, PostMetadata]:
    """Hydrate a posts row (cols match :data:`_POST_FULL_COLS`) into PostMetadata."""
    return int(row[0]), PostMetadata(
        post_name=row[1],
        source_url=row[2],
        canonical_url=row[3],
        title=row[4],
        author=row[5],
        site_name=row[6],
        date=row[7],
        abstract=row[8],
        domain=row[9],
        collection=row[10],
        status=row[11],
        markdown=row[12],
        raw_html=row[13],
        content_hash=row[14],
        etag=row[15],
        last_modified=row[16],
        needs_review=bool(row[17]),
        ingested_at=row[18],
    )


def _load_post_by_url(
    conn: sqlite3.Connection, *, column: str, value: str,
) -> tuple[int, PostMetadata] | None:
    """Look up by ``column`` (canonical_url or source_url) and hydrate."""
    row = conn.execute(
        f"SELECT {_POST_FULL_COLS} FROM posts WHERE {column} = ?",
        (value,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_post(row)


def _log_soft_dedup(
    conn: sqlite3.Connection, content_hash: str, canonical_url: str,
) -> None:
    row = conn.execute(
        "SELECT canonical_url FROM posts "
        " WHERE content_hash = ? AND canonical_url != ? LIMIT 1",
        (content_hash, canonical_url),
    ).fetchone()
    if row is not None:
        _LOG.warning(
            "soft-dedup: post content_hash %s also at %s (current: %s); continuing",
            content_hash, row[0], canonical_url,
        )


def _persist_post(
    conn: sqlite3.Connection,
    pm: PostMetadata,
    *,
    existing_id: int | None,
) -> int:
    """Phase 2: single transaction.

    Cascade-delete the existing post (if force-refetch) BEFORE inserting
    so post_name + ingested_at can be preserved on the row carried in
    ``pm`` without violating the post_name UNIQUE constraint.
    """
    with transaction(conn):
        if pm.domain:
            conn.execute(
                "INSERT OR IGNORE INTO domains (name) VALUES (?)",
                (pm.domain,),
            )
        if existing_id is not None:
            delete_post_cascade(conn, post_id=existing_id)

        cursor = conn.execute(
            """
            INSERT INTO posts (
                post_name, source_url, canonical_url, title, author,
                site_name, date, abstract, domain, collection,
                content_hash, etag, last_modified, raw_html, markdown,
                ingested_at, status, needs_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pm.post_name, pm.source_url, pm.canonical_url, pm.title,
                pm.author, pm.site_name, pm.date, pm.abstract,
                pm.domain, pm.collection, pm.content_hash,
                pm.etag, pm.last_modified, pm.raw_html, pm.markdown,
                pm.ingested_at, str(pm.status), int(bool(pm.needs_review)),
            ),
        )
        return int(cursor.lastrowid)


def _record_failed_fetch(
    conn: sqlite3.Connection, source_url: str,
) -> PostMetadata:
    """Stub a posts row in FAILED_FETCH so search.py --needs-review surfaces it.

    For unrecoverable network/4xx/parse cases we still want a row so the
    user can see the URL was tried. ``canonical_url`` falls back to
    ``source_url`` when no body parsed (it's UNIQUE NOT NULL on posts).
    """
    existing = conn.execute(
        "SELECT id, post_name, ingested_at FROM posts WHERE source_url = ?",
        (source_url,),
    ).fetchone()
    now = datetime.now(timezone.utc)
    if existing is None:
        existing_id: int | None = None
        post_name = generate_post_name(
            "post",
            now.strftime("%Y-%m-%d"),
            source_url,
            existing_slugs(conn),
        )
        ingested_at = now.isoformat(timespec="seconds")
    else:
        existing_id = int(existing[0])
        post_name = existing[1]
        ingested_at = existing[2] or now.isoformat(timespec="seconds")
    pm = PostMetadata(
        post_name=post_name,
        source_url=source_url,
        canonical_url=source_url,
        title="(unparseable)",
        author=None,
        site_name=None,
        date=now.strftime("%Y-%m-%d"),
        abstract="(unparseable)",
        domain=None,
        collection=None,
        status=PostStatus.FAILED_FETCH,
        markdown=None,
        raw_html=None,
        content_hash=None,
        needs_review=True,
        ingested_at=ingested_at,
    )
    _persist_post(conn, pm, existing_id=existing_id)
    return pm


def _persist_extracted_post(
    conn: sqlite3.Connection,
    *,
    meta: _ExtractedMetadata,
    html_body: str,
    source_url: str,
    force: bool,
    domain_override: str | None,
    etag: str | None,
    last_modified: str | None,
) -> PostMetadata:
    """Phase-2 tail shared by the HTTP and local-file fetch front-ends.

    Resolves the canonical-keyed dedup/force decision, computes the
    content hash, scans the body for a repo URL, picks the slug
    (preserving an existing row's ``post_name`` / ``ingested_at`` on a
    force-refetch), and writes one ``status='fetched'`` posts row.

    ``etag`` / ``last_modified`` are caller-supplied: the HTTP response
    headers for :func:`fetch`, ``None`` for :func:`fetch_from_file`
    (a saved file carries no transport metadata). The "already present
    and not force" short-circuit returns the existing row untouched.
    """
    existing = _load_post_by_url(
        conn, column="canonical_url", value=meta.canonical_url,
    )
    if existing is not None and not force:
        _, existing_pm = existing
        _LOG.info(
            "post %s already present (status=%s), skipping fetch",
            meta.canonical_url, existing_pm.status,
        )
        return existing_pm

    content_hash = hashlib.sha256(
        (meta.canonical_url + html_body).encode("utf-8")
    ).hexdigest()
    _log_soft_dedup(conn, content_hash, meta.canonical_url)

    # Repo discovery — feeds the orchestrator's standalone-repo branch.
    # The discovered URL travels on the in-memory PostMetadata; it is
    # never persisted to the posts row.
    repo_hits = extract_repo_candidates(html_body)
    code_repo = repo_hits[0] if repo_hits else None

    if existing is not None:
        existing_id = existing[0]
        post_name = existing[1].post_name
        ingested_at = existing[1].ingested_at or datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")
    else:
        existing_id = None
        post_name = generate_post_name(
            meta.title,
            meta.date,
            meta.canonical_url,
            existing_slugs(conn),
        )
        ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    pm = PostMetadata(
        post_name=post_name,
        source_url=source_url,
        canonical_url=meta.canonical_url,
        title=meta.title,
        author=meta.author,
        site_name=meta.site_name,
        date=meta.date,
        abstract=meta.description,
        domain=domain_override,
        collection=None,
        status=PostStatus.FETCHED,
        markdown=None,
        raw_html=html_body,
        content_hash=content_hash,
        etag=etag,
        last_modified=last_modified,
        code_repo=code_repo,
        needs_review=False,
        ingested_at=ingested_at,
    )
    _persist_post(conn, pm, existing_id=existing_id)
    return pm


def fetch(
    *,
    conn: sqlite3.Connection,
    url: str,
    force: bool = False,
    domain_override: str | None = None,
    client: httpx.Client | None = None,
) -> PostMetadata:
    """Two-phase fetch of a blog post URL.

    Returns the populated :class:`PostMetadata` (status=FETCHED on
    success, FAILED_FETCH on terminal network or parse-rejection). Caller
    inspects ``status`` and proceeds accordingly. All-or-nothing inside
    a single transaction at the persist boundary; mid-fetch network
    failures bubble up uncaught (resumable via re-run).
    """
    owns_client = client is None
    client = client or make_default_client()
    try:
        # Pre-check by source_url: if the user re-runs the exact same URL
        # without --force, skip the network. canonical_url-based dedup
        # still runs after fetch to catch the same post reached via a
        # different source URL (with/without trailing slash, etc.).
        if not force:
            cached = _load_post_by_url(conn, column="source_url", value=url)
            if cached is not None:
                _LOG.info(
                    "post %s already present (status=%s), skipping fetch",
                    url, cached[1].status,
                )
                return cached[1]

        try:
            resp = _http_get(client, url)
        except httpx.HTTPError as exc:
            _LOG.warning("post fetch %s failed at network layer: %s", url, exc)
            return _record_failed_fetch(conn, url)

        if 400 <= resp.status_code < 500:
            _LOG.warning("post fetch %s returned %d, marking FAILED_FETCH",
                         url, resp.status_code)
            return _record_failed_fetch(conn, url)
        html_body = resp.text
        post_redirect_url = str(resp.url)

        meta = _extract_metadata(html_body, post_redirect_url)
        if meta is None:
            return _record_failed_fetch(conn, url)

        return _persist_extracted_post(
            conn,
            meta=meta,
            html_body=html_body,
            source_url=url,
            force=force,
            domain_override=domain_override,
            etag=resp.headers.get("etag"),
            last_modified=resp.headers.get("last-modified"),
        )
    finally:
        if owns_client:
            client.close()


# Public alias matching the convention used by other stages.
fetch_post = fetch


def fetch_from_file(
    *,
    conn: sqlite3.Connection,
    html_path: Path,
    force: bool = False,
    domain_override: str | None = None,
) -> PostMetadata:
    """Ingest a locally-saved ``.html`` file through the post pipeline.

    The file front-end to :func:`fetch`: no network, no HTTP headers.
    Used for paywalled / JS-rendered pages a cookieless GET can't capture
    but a browser "Save Page As → Complete" can. Identity comes from the
    file's own ``<link rel=canonical>`` / ``og:url``; a ``file://<abspath>``
    sentinel is the fallback (canonical_url == source_url, which is unique
    and stable). Returns the populated :class:`PostMetadata` — status
    FETCHED on success, FAILED_FETCH when the file didn't parse as an
    article. Persistence reuses the exact phase-2 tail as the HTTP lane,
    so ``--force`` cascades the matching canonical in place (slug
    preserved).
    """
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found at {html_path}")
    html_body = html_path.read_text(encoding="utf-8", errors="replace")
    base_url = _resolve_canonical_url(html_body, html_path.resolve().as_uri())

    meta = _extract_metadata(html_body, base_url)
    if meta is None:
        # There is no network in the file lane — _record_failed_fetch's
        # "fetch" name is a pragmatic reuse. The file simply didn't parse
        # as a readable article (no title / not an article DOM).
        _LOG.warning(
            "post file %s did not parse as an article, marking FAILED_FETCH",
            html_path,
        )
        return _record_failed_fetch(conn, base_url)

    return _persist_extracted_post(
        conn,
        meta=meta,
        html_body=html_body,
        source_url=meta.canonical_url,
        force=force,
        domain_override=domain_override,
        etag=None,
        last_modified=None,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fetch a blog post URL.")
    parser.add_argument("--url", required=True, help="post URL")
    parser.add_argument("--db", default="lodestone.db", help="path to sqlite db")
    parser.add_argument("--force", action="store_true", help="re-fetch even if present")
    parser.add_argument("--domain", default=None, help="domain override")
    args = parser.parse_args(argv)

    if args.domain is not None:
        sanitized = sanitize_domain(args.domain)
        if not sanitized:
            parser.error(
                f"--domain={args.domain!r} sanitizes to empty string"
            )
        args.domain = sanitized

    conn = get_conn(Path(args.db))
    try:
        pm = fetch(
            conn=conn,
            url=args.url,
            force=args.force,
            domain_override=args.domain,
        )
    finally:
        conn.close()
    print(json.dumps({
        "post_name": pm.post_name,
        "status": str(pm.status),
        "canonical_url": pm.canonical_url,
        "title": pm.title,
    }))


if __name__ == "__main__":
    main()
