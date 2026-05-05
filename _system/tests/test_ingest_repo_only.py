"""Unit tests for the standalone-repo ingest path.

Mocks the resolve / fetch / classify stages on the ingest module to
exercise the orchestrator's stage-routing without touching the network
or LLM.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from _system.schemas.repo_metadata import RepoStatus
from _system.scripts import ingest
from _system.scripts.resolve_repo import ResolveResult, _ResolvedMetadata


def _resolve_stub(*, conn, repo_url, paper_id=None, domain=None,
                  collection=None, client=None) -> ResolveResult:
    """Insert a minimal repos row and return a ResolveResult."""
    cur = conn.execute(
        """
        INSERT INTO repos (
            repo_slug, url, host, owner, name, paper_id,
            ingested_at, status
        ) VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, ?)
        """,
        (
            "gh-owner-tool", repo_url, paper_id,
            "2024-01-02T00:00:00+00:00", RepoStatus.RESOLVED.value,
        ),
    )
    return ResolveResult(
        repo_id=cur.lastrowid,
        repo_slug="gh-owner-tool",
        url=repo_url,
        host="github.com",
        owner="owner",
        name="tool",
        paper_id=paper_id,
        status=RepoStatus.RESOLVED.value,
        metadata=_ResolvedMetadata(description=None, default_branch=None, topics=()),
    )


@pytest.fixture
def patched(conn):
    """Patch resolve / fetch / classify stages on the ingest module."""
    calls: list[tuple[str, dict]] = []

    def _resolve(*, conn, repo_url, paper_id=None, domain=None,
                 collection=None, client=None):
        calls.append(("resolve", {"repo_url": repo_url}))
        return _resolve_stub(
            conn=conn, repo_url=repo_url, paper_id=paper_id,
            domain=domain, collection=collection,
        )

    def _fetch(**kwargs):
        calls.append(("fetch_repo", dict(kwargs)))
        c = kwargs["conn"]
        slug = kwargs["repo_slug"]
        c.execute(
            "UPDATE repos SET status = ?, has_readme = 1, file_count = 3 "
            " WHERE repo_slug = ?",
            (RepoStatus.REPO_FETCHED.value, slug),
        )

    def _classify(**kwargs):
        calls.append(("classify_repo", dict(kwargs)))
        c = kwargs["conn"]
        slug = kwargs["repo_slug"]
        c.execute(
            "INSERT OR IGNORE INTO domains (name) VALUES ('rag')"
        )
        c.execute(
            "INSERT OR IGNORE INTO collections (domain, name) VALUES ('rag', 'tools')"
        )
        c.execute(
            "UPDATE repos SET status = ?, domain = ?, collection = ? "
            " WHERE repo_slug = ?",
            (RepoStatus.CLASSIFIED.value, "rag", "tools", slug),
        )

    with patch.object(ingest, "resolve_repo_stage", side_effect=_resolve), \
         patch.object(ingest, "fetch_repo_stage", side_effect=_fetch), \
         patch.object(ingest, "classify_repo_stage", side_effect=_classify):
        yield calls


def test_fresh_standalone_repo_runs_all_three_stages(conn, patched):
    summary = ingest.ingest_repo_only(
        conn=conn,
        repo_url="https://github.com/owner/tool",
        force=False,
    )
    stages = [c[0] for c in patched]
    assert stages == ["resolve", "fetch_repo", "classify_repo"]
    assert summary["status"] == RepoStatus.CLASSIFIED.value
    assert summary["repo_slug"] == "gh-owner-tool"


def test_resume_from_resolved_skips_resolve(conn, patched):
    # Simulate a prior partial run.
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, "
        "  ingested_at, status) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?)",
        ("gh-owner-tool", "https://github.com/owner/tool",
         "2024-01-02T00:00:00+00:00", RepoStatus.RESOLVED.value),
    )
    ingest.ingest_repo_only(
        conn=conn,
        repo_url="https://github.com/owner/tool",
        force=False,
    )
    stages = [c[0] for c in patched]
    assert stages == ["fetch_repo", "classify_repo"]


def test_terminal_no_force_is_noop(conn, patched):
    conn.execute(
        "INSERT OR IGNORE INTO domains (name) VALUES ('rag')"
    )
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, ?, ?)",
        ("gh-owner-tool", "https://github.com/owner/tool",
         "2024-01-02T00:00:00+00:00", RepoStatus.CLASSIFIED.value,
         "rag", "tools"),
    )
    summary = ingest.ingest_repo_only(
        conn=conn,
        repo_url="https://github.com/owner/tool",
        force=False,
    )
    assert patched == []
    assert summary["status"] == RepoStatus.CLASSIFIED.value


def test_force_cascades_then_runs_full_pipeline(conn, patched):
    """``--force`` wipes the prior repos row and re-runs every stage."""
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES ('rag')")
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, 'rag', 'tools')",
        ("gh-owner-tool", "https://github.com/owner/tool",
         "2024-01-02T00:00:00+00:00", RepoStatus.CLASSIFIED.value),
    )

    ingest.ingest_repo_only(
        conn=conn,
        repo_url="https://github.com/owner/tool",
        force=True,
    )
    stages = [c[0] for c in patched]
    assert stages == ["resolve", "fetch_repo", "classify_repo"]
    n = conn.execute(
        "SELECT COUNT(*) FROM repos WHERE url = ?",
        ("https://github.com/owner/tool",),
    ).fetchone()[0]
    assert n == 1


def test_paper_linked_repo_url_rejected_via_standalone_path(conn, patched):
    """A URL that already exists as a paper-linked repo cannot be
    re-routed through the standalone path."""
    conn.execute(
        "INSERT OR IGNORE INTO domains (name) VALUES ('rag')"
    )
    conn.execute(
        "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, "
        "  abstract, pdf_url, ingested_at, status, domain, collection) "
        "VALUES ('2401.00001', 'p1', 't', '[]', '2024-01-01', 'a', 'u', 'd', "
        "  'fetched', NULL, NULL)"
    )
    paper_id = conn.execute(
        "SELECT id FROM papers WHERE paper_name='p1'"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, paper_id, "
        "  ingested_at, status) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, ?)",
        ("gh-owner-tool", "https://github.com/owner/tool", paper_id,
         "2024-01-02T00:00:00+00:00", RepoStatus.RESOLVED.value),
    )

    with pytest.raises(ValueError, match="paper-linked repo"):
        ingest.ingest_repo_only(
            conn=conn,
            repo_url="https://github.com/owner/tool",
            force=False,
        )
