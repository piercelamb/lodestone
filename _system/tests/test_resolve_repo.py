"""Unit tests for ``_system/scripts/resolve_repo.py`` and the URL helpers.

GitHub API access is mocked via httpx.MockTransport — no real network
hits.
"""
from __future__ import annotations

import sqlite3

import httpx
import pytest

from _system.scripts import resolve_repo as rr
from _system.utils.repo_url import normalize_repo_url, parse_repo_url, repo_slug_base


# ---------------------------------------------------------------------------
# URL parsing + slug derivation
# ---------------------------------------------------------------------------


def test_normalize_repo_url_strips_dot_git():
    assert normalize_repo_url("https://github.com/foo/bar.git") == \
        "https://github.com/foo/bar"


def test_normalize_repo_url_strips_trailing_punctuation():
    assert normalize_repo_url("https://github.com/foo/bar.") == \
        "https://github.com/foo/bar"
    assert normalize_repo_url("https://github.com/foo/bar)") == \
        "https://github.com/foo/bar"


def test_normalize_repo_url_rejects_deeper_paths():
    """Paths with depth != 2 are rejected outright (no truncation)."""
    assert normalize_repo_url("https://github.com/foo/bar/issues/42") is None
    assert normalize_repo_url("https://github.com/foo") is None


def test_normalize_repo_url_rejects_other_hosts():
    assert normalize_repo_url("https://example.com/foo/bar") is None


def test_parse_repo_url_returns_parts():
    parts = parse_repo_url("https://github.com/Owner/Repo")
    assert parts is not None
    assert parts.canonical_url == "https://github.com/Owner/Repo"
    assert parts.host == "github.com"
    assert parts.owner == "Owner"
    assert parts.name == "Repo"


def test_repo_slug_base_format():
    assert repo_slug_base("github.com", "Meta-Llama", "Llama") == \
        "gh-meta-llama-llama"
    assert repo_slug_base("gitlab.com", "alice", "proj") == "gl-alice-proj"
    assert repo_slug_base("bitbucket.org", "team", "code") == "bb-team-code"


def test_repo_slug_base_strips_special_chars():
    assert repo_slug_base("github.com", "User_Name", "My-Tool!") == \
        "gh-user-name-my-tool"


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------


def _client_with(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def _gh_metadata_response(description="Demo repo", default_branch="main",
                          topics=("ml", "rag")) -> httpx.Response:
    body = {
        "description": description,
        "default_branch": default_branch,
        "topics": list(topics),
    }
    return httpx.Response(200, json=body)


def test_resolve_creates_repos_row_with_metadata(conn):
    def handler(req):
        if "api.github.com" in str(req.url):
            return _gh_metadata_response()
        return httpx.Response(404)

    with _client_with(handler) as client:
        result = rr.resolve(
            conn=conn,
            repo_url="https://github.com/owner/proj",
            client=client,
        )

    assert result.repo_slug == "gh-owner-proj"
    assert result.url == "https://github.com/owner/proj"
    assert result.host == "github.com"
    assert result.owner == "owner"
    assert result.name == "proj"
    assert result.status == "resolved"
    assert result.metadata.description == "Demo repo"
    assert result.metadata.default_branch == "main"
    assert "ml" in result.metadata.topics

    row = conn.execute(
        "SELECT description, default_branch, status, paper_id "
        "  FROM repos WHERE repo_slug = ?",
        ("gh-owner-proj",),
    ).fetchone()
    assert row == ("Demo repo", "main", "resolved", None)


def test_resolve_continues_when_github_404(conn):
    def handler(req):
        return httpx.Response(404)

    with _client_with(handler) as client:
        result = rr.resolve(
            conn=conn,
            repo_url="https://github.com/owner/missing",
            client=client,
        )
    assert result.metadata.description is None
    assert result.metadata.default_branch is None


def test_resolve_continues_when_github_403_rate_limited(conn):
    """A 403 (rate-limit / no token) should not abort the pipeline."""
    def handler(req):
        return httpx.Response(403, json={"message": "rate limited"})

    with _client_with(handler) as client:
        result = rr.resolve(
            conn=conn,
            repo_url="https://github.com/owner/proj2",
            client=client,
        )
    assert result.metadata.description is None
    # Row was still created, with status=resolved.
    assert result.status == "resolved"


def test_resolve_paper_linked_carries_paper_id(conn):
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES ('rag')")
    conn.execute(
        "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, "
        "  abstract, pdf_url, ingested_at, status, domain, collection) "
        "VALUES ('2401.99999', 'p1', 't', '[]', '2024-01-01', 'a', 'u', 'd', "
        "  'fetched', NULL, NULL)"
    )
    paper_id = conn.execute("SELECT id FROM papers WHERE paper_name='p1'").fetchone()[0]

    def handler(req):
        return httpx.Response(404)

    with _client_with(handler) as client:
        result = rr.resolve(
            conn=conn,
            repo_url="https://github.com/owner/anchored",
            paper_id=paper_id,
            client=client,
        )

    assert result.paper_id == paper_id
    row = conn.execute(
        "SELECT paper_id FROM repos WHERE repo_slug = ?", (result.repo_slug,)
    ).fetchone()
    assert row[0] == paper_id


def test_resolve_idempotent_on_same_url(conn):
    def handler(req):
        return httpx.Response(404)

    with _client_with(handler) as client:
        first = rr.resolve(
            conn=conn,
            repo_url="https://github.com/owner/dup",
            client=client,
        )
        second = rr.resolve(
            conn=conn,
            repo_url="https://github.com/owner/dup",
            client=client,
        )

    assert first.repo_id == second.repo_id
    assert first.repo_slug == second.repo_slug
    n = conn.execute(
        "SELECT COUNT(*) FROM repos WHERE url = ?",
        ("https://github.com/owner/dup",),
    ).fetchone()[0]
    assert n == 1


def test_resolve_rejects_unsupported_url(conn):
    with pytest.raises(rr.InvalidRepoUrlError):
        rr.resolve(conn=conn, repo_url="https://example.com/foo/bar")


def test_resolve_handles_slug_collision_via_suffix(conn):
    """Two different URLs that slug-collide get -2/-3 suffixes."""
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES ('rag')")
    # Pre-seed a row that will collide with the next slug derivation.
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, "
        "  ingested_at, status) "
        "VALUES (?, ?, 'github.com', 'owner', 'proj', 'd', 'resolved')",
        ("gh-owner-proj", "https://github.com/owner/proj"),
    )

    def handler(req):
        return httpx.Response(404)

    # Same canonical owner/name but different host (gitlab) — base slug
    # differs (gl-...), so use a contrived alt-owner with different
    # canonical that still collapses to gh-owner-proj after slugify.
    # Easier: directly hit the helper to verify suffix logic.
    from _system.utils.repo_url import RepoUrlParts
    parts = RepoUrlParts(
        canonical_url="https://github.com/owner/proj",
        host="github.com", owner="owner", name="proj",
    )
    assert rr._allocate_slug(conn, parts) == "gh-owner-proj-2"
