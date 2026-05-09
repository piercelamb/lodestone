"""Stage 1 of the standalone-repo path: turn a URL into a ``repos`` row.

Parses a GitHub/GitLab/Bitbucket URL, derives a deterministic ``repo_slug``
(``{host_short}-{owner}-{name}`` with collision suffixes), optionally hits
the GitHub API for description / default_branch / topics, and inserts a
``repos`` row with status ``RESOLVED``.

Metadata fetch is **best-effort**: missing GitHub token, rate-limit, 404
all log a warning and proceed with metadata fields left NULL. A repo
that doesn't exist on the host will fail in the next stage (clone), at
which point status becomes ``FAILED_REPO``. We do not block resolve on
remote reachability — keeping the stages independent matches the
paper-side pipeline's two-phase shape.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import httpx

from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.schemas.repo_metadata import RepoStatus
from _system.utils.logging import get_logger
from _system.utils.repo_url import RepoUrlParts, parse_repo_url, repo_slug_base

_LOG = get_logger("scripts.resolve_repo")

USER_AGENT = "Lodestone/1.0 (mailto:richard.pierce.lamb@gmail.com)"

_GITHUB_API_REPO = "https://api.github.com/repos/{owner}/{name}"
_GITHUB_TOKEN_ENV_VARS = ("GITHUB_TOKEN", "GH_TOKEN", "LODESTONE_GITHUB_TOKEN")


class ResolveRepoError(Exception):
    """Base class for resolve_repo failures."""


class InvalidRepoUrlError(ResolveRepoError):
    """URL is not a recognized owner/repo on a supported host."""


class _ResolvedMetadata(NamedTuple):
    description: str | None
    default_branch: str | None
    topics: tuple[str, ...]


class ResolveResult(NamedTuple):
    repo_id: int
    repo_slug: str
    url: str
    host: str
    owner: str
    name: str
    paper_id: int | None
    status: str
    metadata: _ResolvedMetadata


def resolve(
    *,
    conn: sqlite3.Connection,
    repo_url: str,
    paper_id: int | None = None,
    domain: str | None = None,
    collection: str | None = None,
    client: httpx.Client | None = None,
) -> ResolveResult:
    """Insert (or update) a ``repos`` row for ``repo_url`` and return its identity.

    ``paper_id`` ties the repo to a paper (paper-linked path). When
    set, ``domain`` / ``collection`` should also be supplied so the new
    repo inherits its taxonomy without re-running CLASSIFY. Standalone
    repos pass paper_id=None and let CLASSIFY fill in the taxonomy from
    the README.

    The DB only contains a single row per canonical URL — re-resolving
    an existing URL updates its mutable fields (description, default
    branch) but never moves a paper-linked repo onto a different paper.
    """
    parts = parse_repo_url(repo_url)
    if parts is None:
        raise InvalidRepoUrlError(
            f"unsupported or malformed repo URL: {repo_url!r}; "
            f"must be https://{{github.com|gitlab.com|bitbucket.org}}/owner/repo"
        )

    owns_client = client is None
    client = client or _make_default_client()

    try:
        meta = _fetch_metadata(client, parts)
    finally:
        if owns_client:
            client.close()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with transaction(conn):
        if domain:
            conn.execute(
                "INSERT OR IGNORE INTO domains (name) VALUES (?)",
                (domain,),
            )

        existing = conn.execute(
            "SELECT id, repo_slug, paper_id, status FROM repos WHERE url = ?",
            (parts.canonical_url,),
        ).fetchone()

        if existing is not None:
            repo_id, existing_slug, existing_paper_id, existing_status = existing
            # Update mutable metadata in place; preserve identity columns.
            conn.execute(
                """
                UPDATE repos
                   SET description = COALESCE(?, description),
                       default_branch = COALESCE(?, default_branch),
                       paper_id = COALESCE(paper_id, ?),
                       domain = COALESCE(domain, ?),
                       collection = COALESCE(collection, ?)
                 WHERE id = ?
                """,
                (
                    meta.description, meta.default_branch,
                    paper_id, domain, collection, repo_id,
                ),
            )
            return ResolveResult(
                repo_id=repo_id,
                repo_slug=existing_slug,
                url=parts.canonical_url,
                host=parts.host,
                owner=parts.owner,
                name=parts.name,
                paper_id=existing_paper_id or paper_id,
                status=existing_status,
                metadata=meta,
            )

        slug = _allocate_slug(conn, parts)
        cursor = conn.execute(
            """
            INSERT INTO repos (
                repo_slug, url, host, owner, name, paper_id,
                description, default_branch, ingested_at,
                domain, collection, status, has_readme, file_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """,
            (
                slug, parts.canonical_url, parts.host, parts.owner,
                parts.name, paper_id,
                meta.description, meta.default_branch, now,
                domain, collection,
                RepoStatus.RESOLVED.value,
            ),
        )
        repo_id = int(cursor.lastrowid)

    _LOG.info(
        "resolved repo %s url=%s paper_id=%s domain=%s",
        slug, parts.canonical_url, paper_id, domain,
    )
    return ResolveResult(
        repo_id=repo_id,
        repo_slug=slug,
        url=parts.canonical_url,
        host=parts.host,
        owner=parts.owner,
        name=parts.name,
        paper_id=paper_id,
        status=RepoStatus.RESOLVED.value,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Slug allocation
# ---------------------------------------------------------------------------


def _allocate_slug(conn: sqlite3.Connection, parts: RepoUrlParts) -> str:
    """Return a unique ``repo_slug`` for ``parts``.

    Starts with the canonical base form (``gh-{owner}-{name}``); if that
    is already taken by a different URL, append ``-2``, ``-3``, ... up to
    a small bound. The PK on ``repos.url`` already prevents the common
    case (re-resolving the same URL); collision-handling here is for the
    rare case of two different hosts/owner combinations slugging the same.
    """
    base = repo_slug_base(parts.host, parts.owner, parts.name)
    candidate = base
    suffix = 2
    while True:
        existing = conn.execute(
            "SELECT 1 FROM repos WHERE repo_slug = ?", (candidate,)
        ).fetchone()
        if existing is None:
            return candidate
        candidate = f"{base}-{suffix}"
        suffix += 1
        if suffix > 100:
            raise RuntimeError(
                f"could not allocate unique repo_slug for {parts.canonical_url}: "
                f"base={base!r} collided 100 times"
            )


# ---------------------------------------------------------------------------
# Metadata fetch (best-effort)
# ---------------------------------------------------------------------------


def _make_default_client() -> httpx.Client:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(headers=headers, timeout=15.0, follow_redirects=True)


def _github_token() -> str | None:
    for var in _GITHUB_TOKEN_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    return None


def _fetch_metadata(client: httpx.Client, parts: RepoUrlParts) -> _ResolvedMetadata:
    """Best-effort GitHub metadata fetch. Non-GitHub hosts return empty."""
    if parts.host != "github.com":
        return _ResolvedMetadata(description=None, default_branch=None, topics=())

    url = _GITHUB_API_REPO.format(owner=parts.owner, name=parts.name)
    try:
        resp = client.get(url)
    except httpx.HTTPError as exc:
        _LOG.warning("github metadata fetch failed for %s: %s", url, exc)
        return _ResolvedMetadata(description=None, default_branch=None, topics=())

    if resp.status_code == 404:
        _LOG.warning("github repo %s not found via API; continuing", url)
        return _ResolvedMetadata(description=None, default_branch=None, topics=())
    if resp.status_code == 401 or resp.status_code == 403:
        _LOG.warning(
            "github metadata fetch %s returned %d (auth/rate-limit); continuing",
            url, resp.status_code,
        )
        return _ResolvedMetadata(description=None, default_branch=None, topics=())
    if resp.status_code != 200:
        _LOG.warning(
            "github metadata fetch %s returned %d; continuing", url, resp.status_code,
        )
        return _ResolvedMetadata(description=None, default_branch=None, topics=())

    try:
        data: dict[str, Any] = resp.json()
    except json.JSONDecodeError as exc:
        _LOG.warning("github metadata for %s is non-JSON: %s", url, exc)
        return _ResolvedMetadata(description=None, default_branch=None, topics=())

    raw_topics = data.get("topics") or []
    topics = tuple(str(t).strip() for t in raw_topics if t)
    return _ResolvedMetadata(
        description=(data.get("description") or None),
        default_branch=(data.get("default_branch") or None),
        topics=topics,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a repo URL into a repos row (status=RESOLVED)."
    )
    parser.add_argument("--repo", required=True, help="github/gitlab/bitbucket URL")
    parser.add_argument(
        "--db",
        default=os.environ.get("LODESTONE_DB", "lodestone.db"),
        help="path to the sqlite db (default: $LODESTONE_DB or ./lodestone.db)",
    )
    args = parser.parse_args(argv)

    conn = get_conn(Path(args.db))
    try:
        init_db(conn)
        result = resolve(conn=conn, repo_url=args.repo)
    finally:
        conn.close()
    print(json.dumps({
        "repo_slug": result.repo_slug,
        "url": result.url,
        "host": result.host,
        "owner": result.owner,
        "name": result.name,
        "status": result.status,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
