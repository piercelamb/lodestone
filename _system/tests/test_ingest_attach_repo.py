"""Unit tests for the ``attach_repo_to_paper`` orchestrator.

Mocks ``resolve_repo_stage`` and ``fetch_repo_stage`` on the ingest
module so we exercise the orchestrator's branching without touching
the network or LLM. The paper-linked path deliberately skips
``classify_repo`` (taxonomy is inherited from the paper), so it is not
patched here — calling it would be a bug.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from _system.schemas.paper_metadata import PaperStatus
from _system.schemas.repo_metadata import RepoStatus
from _system.scripts import ingest
from _system.scripts.resolve_repo import ResolveResult, _ResolvedMetadata


_DEFAULT_URL = "https://github.com/owner/tool"


def _seed_paper(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str = "2401.00001",
    paper_name: str = "alice_2024_thing",
    status: PaperStatus = PaperStatus.INDEXED,
    domain: str | None = "rag",
    collection: str | None = "tools",
) -> int:
    """Insert a minimal papers row mirroring an indexed paper.

    Classified+ rows require both domain and collection per the schema
    invariant — the defaults satisfy that for the happy path. Pass
    ``domain=None``/``collection=None`` only with pre-classify statuses.
    """
    if domain is not None:
        conn.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", (domain,))
    if domain is not None and collection is not None:
        conn.execute(
            "INSERT OR IGNORE INTO collection_definitions "
            "(domain, name, description) VALUES (?, ?, NULL)",
            (domain, collection),
        )
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, ingested_at, status, domain, collection
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id, paper_name, "Title", '["A"]', "2024-01-01", "Abs",
            f"https://arxiv.org/pdf/{arxiv_id}",
            "2024-01-02T00:00:00+00:00",
            status.value, domain, collection,
        ),
    )
    return int(cur.lastrowid)


def _seed_collection_row(
    conn: sqlite3.Connection,
    *,
    target_kind: str,
    target_id: int,
    domain: str,
    collection: str,
    is_primary: bool = True,
) -> None:
    conn.execute(
        "INSERT INTO collections (target_kind, target_id, domain, collection, is_primary) "
        "VALUES (?, ?, ?, ?, ?)",
        (target_kind, target_id, domain, collection, int(is_primary)),
    )


def _resolve_stub(*, conn, repo_url, paper_id=None, domain=None,
                  collection=None, client=None) -> ResolveResult:
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
        repo_id=int(cur.lastrowid),
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
    calls: list[tuple[str, dict]] = []

    def _resolve(*, conn, repo_url, paper_id=None, domain=None,
                 collection=None, client=None):
        calls.append((
            "resolve",
            {"repo_url": repo_url, "paper_id": paper_id},
        ))
        return _resolve_stub(
            conn=conn, repo_url=repo_url, paper_id=paper_id,
            domain=domain, collection=collection,
        )

    def _fetch(**kwargs):
        calls.append(("fetch_repo", dict(kwargs)))
        c = kwargs["conn"]
        slug = kwargs["repo_slug"]
        c.execute(
            "UPDATE repos SET status = ?, has_readme = 1, file_count = 5 "
            " WHERE repo_slug = ?",
            (RepoStatus.REPO_FETCHED.value, slug),
        )

    with patch.object(ingest, "resolve_repo_stage", side_effect=_resolve), \
         patch.object(ingest, "fetch_repo_stage", side_effect=_fetch):
        yield calls


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_attach_runs_resolve_then_fetch_and_inherits_taxonomy(conn, patched):
    paper_id = _seed_paper(conn)
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )

    summary = ingest.attach_repo_to_paper(
        conn=conn,
        paper_identifier="alice_2024_thing",
        repo_url=_DEFAULT_URL,
        force=False,
    )

    stages = [c[0] for c in patched]
    # No classify_repo — paper-linked repos inherit taxonomy.
    assert stages == ["resolve", "fetch_repo"]

    # resolve_repo was called with paper_id wired through.
    resolve_kwargs = patched[0][1]
    assert resolve_kwargs["paper_id"] == paper_id

    repo_row = conn.execute(
        "SELECT paper_id, domain, collection, status, file_count "
        "  FROM repos WHERE url = ?",
        (_DEFAULT_URL,),
    ).fetchone()
    assert repo_row[0] == paper_id
    assert repo_row[1] == "rag"
    assert repo_row[2] == "tools"
    assert repo_row[3] == RepoStatus.REPO_FETCHED.value
    assert repo_row[4] == 5

    coll = conn.execute(
        "SELECT domain, collection, is_primary FROM collections "
        " WHERE target_kind = 'repo'"
    ).fetchall()
    assert coll == [("rag", "tools", 1)]

    # Summary envelope carries the new repo.
    assert summary["repo"] is not None
    assert summary["repo"]["repo_slug"] == "gh-owner-tool"
    assert summary["repo"]["status"] == RepoStatus.REPO_FETCHED.value


def test_attach_resolves_paper_by_arxiv_id(conn, patched):
    paper_id = _seed_paper(conn, arxiv_id="2401.99999", paper_name="bob_paper")
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )

    ingest.attach_repo_to_paper(
        conn=conn,
        paper_identifier="2401.99999",
        repo_url=_DEFAULT_URL,
        force=False,
    )

    repo_paper_id = conn.execute(
        "SELECT paper_id FROM repos WHERE url = ?",
        (_DEFAULT_URL,),
    ).fetchone()[0]
    assert repo_paper_id == paper_id


def test_attach_falls_back_to_arxiv_when_name_misses(conn, patched):
    """Name lookup is tried first; arxiv_id is the fallback."""
    paper_id = _seed_paper(conn, arxiv_id="2401.55555", paper_name="real_slug")
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )

    ingest.attach_repo_to_paper(
        conn=conn,
        paper_identifier="2401.55555",
        repo_url=_DEFAULT_URL,
        force=False,
    )
    assert [c[0] for c in patched] == ["resolve", "fetch_repo"]


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------


def test_attach_rejects_unknown_paper(conn, patched):
    with pytest.raises(ValueError, match="no paper found"):
        ingest.attach_repo_to_paper(
            conn=conn,
            paper_identifier="ghost_paper",
            repo_url=_DEFAULT_URL,
            force=False,
        )
    assert patched == []


def test_attach_rejects_paper_not_indexed(conn, patched):
    _seed_paper(
        conn, status=PaperStatus.FETCHED,
        domain=None, collection=None,
    )
    with pytest.raises(ValueError, match="requires status='indexed'"):
        ingest.attach_repo_to_paper(
            conn=conn,
            paper_identifier="alice_2024_thing",
            repo_url=_DEFAULT_URL,
            force=False,
        )
    assert patched == []


def test_attach_rejects_paper_with_existing_repo_no_force(conn, patched):
    paper_id = _seed_paper(conn)
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, paper_id, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, ?, 'rag', 'tools')",
        (
            "gh-owner-existing", "https://github.com/owner/existing", paper_id,
            "2024-01-02T00:00:00+00:00", RepoStatus.REPO_FETCHED.value,
        ),
    )

    with pytest.raises(ValueError, match="already has a linked repo"):
        ingest.attach_repo_to_paper(
            conn=conn,
            paper_identifier="alice_2024_thing",
            repo_url=_DEFAULT_URL,
            force=False,
        )
    assert patched == []


def test_attach_force_cascades_existing_repo_and_reattaches(conn, patched):
    paper_id = _seed_paper(conn)
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )
    cur = conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, paper_id, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, ?, 'rag', 'tools')",
        (
            "gh-owner-old", "https://github.com/owner/old", paper_id,
            "2024-01-02T00:00:00+00:00", RepoStatus.REPO_FETCHED.value,
        ),
    )
    old_repo_id = int(cur.lastrowid)
    _seed_collection_row(
        conn, target_kind="repo", target_id=old_repo_id,
        domain="rag", collection="tools",
    )

    ingest.attach_repo_to_paper(
        conn=conn,
        paper_identifier="alice_2024_thing",
        repo_url=_DEFAULT_URL,
        force=True,
    )

    # Old repo row gone; new one in its place linked to the same paper.
    remaining = conn.execute(
        "SELECT url FROM repos WHERE paper_id = ?", (paper_id,)
    ).fetchall()
    assert remaining == [(_DEFAULT_URL,)]
    assert [c[0] for c in patched] == ["resolve", "fetch_repo"]


def test_attach_rejects_url_already_linked_to_other_paper(conn, patched):
    paper_id = _seed_paper(conn)
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )
    other_paper_id = _seed_paper(
        conn, arxiv_id="2402.00002", paper_name="other_paper",
    )
    _seed_collection_row(
        conn, target_kind="paper", target_id=other_paper_id,
        domain="rag", collection="tools",
    )
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, paper_id, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, ?, 'rag', 'tools')",
        (
            "gh-owner-tool", _DEFAULT_URL, other_paper_id,
            "2024-01-02T00:00:00+00:00", RepoStatus.REPO_FETCHED.value,
        ),
    )

    with pytest.raises(ValueError, match="already linked to a different paper"):
        ingest.attach_repo_to_paper(
            conn=conn,
            paper_identifier="alice_2024_thing",
            repo_url=_DEFAULT_URL,
            force=False,
        )
    assert patched == []


def test_attach_rejects_url_existing_as_standalone(conn, patched):
    paper_id = _seed_paper(conn)
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, paper_id, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', NULL, ?, ?, 'rag', 'tools')",
        (
            "gh-owner-tool", _DEFAULT_URL,
            "2024-01-02T00:00:00+00:00", RepoStatus.CLASSIFIED.value,
        ),
    )

    with pytest.raises(ValueError, match="standalone repo"):
        ingest.attach_repo_to_paper(
            conn=conn,
            paper_identifier="alice_2024_thing",
            repo_url=_DEFAULT_URL,
            force=False,
        )
    assert patched == []


def test_attach_canonicalizes_url_before_conflict_check(conn, patched):
    """A non-canonical input form (`.git` suffix) must hit the same
    cross-paper rejection as the canonical form, because resolve_repo
    stores the canonical URL and its COALESCE-on-paper_id would
    otherwise silently corrupt the other paper's repo taxonomy.
    """
    paper_id = _seed_paper(conn)
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )
    other_paper_id = _seed_paper(
        conn, arxiv_id="2402.00002", paper_name="other_paper",
    )
    _seed_collection_row(
        conn, target_kind="paper", target_id=other_paper_id,
        domain="rag", collection="tools",
    )
    # Existing row stored at the CANONICAL url (resolve_repo strips
    # .git and trailing slashes), linked to other_paper.
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, paper_id, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, ?, 'rag', 'tools')",
        (
            "gh-owner-tool", _DEFAULT_URL, other_paper_id,
            "2024-01-02T00:00:00+00:00", RepoStatus.REPO_FETCHED.value,
        ),
    )

    with pytest.raises(ValueError, match="already linked to a different paper"):
        ingest.attach_repo_to_paper(
            conn=conn,
            paper_identifier="alice_2024_thing",
            repo_url="https://github.com/owner/tool.git",
            force=False,
        )
    assert patched == []


def test_attach_malformed_url_does_not_delete_existing_linked_repo(conn, patched):
    """`--force` must not cascade-delete the paper's existing linked
    repo when the new URL is malformed — the URL is validated before
    any destructive write.
    """
    paper_id = _seed_paper(conn)
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )
    cur = conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, paper_id, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'old', ?, ?, ?, 'rag', 'tools')",
        (
            "gh-owner-old", "https://github.com/owner/old", paper_id,
            "2024-01-02T00:00:00+00:00", RepoStatus.REPO_FETCHED.value,
        ),
    )
    old_repo_id = int(cur.lastrowid)

    with pytest.raises(ValueError, match="unsupported or malformed repo URL"):
        ingest.attach_repo_to_paper(
            conn=conn,
            paper_identifier="alice_2024_thing",
            repo_url="not-a-url",
            force=True,
        )
    assert patched == []
    # Original linked repo must still exist.
    remaining = conn.execute(
        "SELECT id FROM repos WHERE id = ?", (old_repo_id,),
    ).fetchone()
    assert remaining is not None


def test_attach_cross_paper_conflict_does_not_delete_existing_linked_repo(
    conn, patched,
):
    """`--force` must validate the new URL's availability before
    cascade-deleting the paper's existing linked repo; otherwise a
    URL that's already taken would silently destroy the prior fetch.
    """
    paper_id = _seed_paper(conn)
    _seed_collection_row(
        conn, target_kind="paper", target_id=paper_id,
        domain="rag", collection="tools",
    )
    cur = conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, paper_id, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'old', ?, ?, ?, 'rag', 'tools')",
        (
            "gh-owner-old", "https://github.com/owner/old", paper_id,
            "2024-01-02T00:00:00+00:00", RepoStatus.REPO_FETCHED.value,
        ),
    )
    old_repo_id = int(cur.lastrowid)
    other_paper_id = _seed_paper(
        conn, arxiv_id="2402.00002", paper_name="other_paper",
    )
    _seed_collection_row(
        conn, target_kind="paper", target_id=other_paper_id,
        domain="rag", collection="tools",
    )
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, paper_id, "
        "  ingested_at, status, domain, collection) "
        "VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, ?, 'rag', 'tools')",
        (
            "gh-owner-tool", _DEFAULT_URL, other_paper_id,
            "2024-01-02T00:00:00+00:00", RepoStatus.REPO_FETCHED.value,
        ),
    )

    with pytest.raises(ValueError, match="already linked to a different paper"):
        ingest.attach_repo_to_paper(
            conn=conn,
            paper_identifier="alice_2024_thing",
            repo_url=_DEFAULT_URL,
            force=True,
        )
    assert patched == []
    remaining = conn.execute(
        "SELECT id FROM repos WHERE id = ?", (old_repo_id,),
    ).fetchone()
    assert remaining is not None
