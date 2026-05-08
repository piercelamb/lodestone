"""Unit tests for ``_system/scripts/classify_repo.py``.

The LLM call is replaced with a deterministic stub via the ``call_llm``
seam; the embedder is the orthogonal-vector test helper from
test_classify_paper. Schema/topics writes go through the real resolver.
"""
from __future__ import annotations

import sqlite3

import pytest

from _system.schemas.repo_metadata import RepoStatus
from _system.scripts.classify_repo import (
    ClassifyRepoStateError,
    classify,
)


# ---------------------------------------------------------------------------
# Helpers (mirrors the patterns in test_classify_paper)
# ---------------------------------------------------------------------------


class _OrthogonalEmbedder:
    def __init__(self) -> None:
        self._index: dict[str, int] = {}

    def _vec_for(self, text: str) -> list[float]:
        idx = self._index.setdefault(text, len(self._index))
        v = [0.0] * 384
        v[idx % 384] = 1.0
        return v

    def embed(self, text: str) -> list[float]:
        return self._vec_for(text)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def _runner_from_dict(payload: dict):
    def _runner(system: str, user: str, schema: dict, response_model):
        return response_model.model_validate(payload)
    return _runner


def _payload(
    *,
    domain_index: int = 0,
    new_domain: str = "",
    new_domain_desc: str = "",
    collections: list[dict] | None = None,
    topics: list[str] | None = None,
) -> dict:
    """Build a multi-collection classification payload for repo classification."""
    if collections is None:
        collections = [
            {
                "index": -1,
                "new_name": "demo_collection",
                "new_desc": "Cluster of work demos.",
            }
        ]
    return {
        "domain_index": domain_index,
        "new_domain": new_domain,
        "new_domain_desc": new_domain_desc,
        "collections": collections,
        "topics": topics if topics is not None else ["t"],
    }


def _seed_domain(conn: sqlite3.Connection, name: str = "rag") -> None:
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", (name,))


def _seed_repo(
    conn: sqlite3.Connection,
    *,
    repo_slug: str = "gh-owner-tool",
    url: str = "https://github.com/owner/tool",
    has_readme: bool = True,
    readme_text: str = "# Demo Tool\n\nA hierarchical retrieval pipeline.\n",
    status: str = RepoStatus.REPO_FETCHED.value,
    domain: str | None = None,
    collection: str | None = None,
    paper_id: int | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO repos (
            repo_slug, url, host, owner, name, paper_id,
            ingested_at, status, has_readme, file_count,
            domain, collection
        ) VALUES (?, ?, 'github.com', 'owner', 'tool', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            repo_slug, url, paper_id,
            "2024-01-02T00:00:00+00:00", status,
            1 if has_readme else 0, 5, domain, collection,
        ),
    )
    repo_id = cur.lastrowid
    if has_readme:
        conn.execute(
            "INSERT INTO readmes_fts (repo_id, repo_slug, domain, path, content) "
            "VALUES (?, ?, ?, 'README.md', ?)",
            (repo_id, repo_slug, domain or "", readme_text),
        )
    return repo_id


@pytest.fixture
def db_with_domain(conn: sqlite3.Connection):
    _seed_domain(conn)
    return conn


# ---------------------------------------------------------------------------
# README-required gate / ORPHANED
# ---------------------------------------------------------------------------


def test_repo_without_readme_marked_orphaned(db_with_domain):
    """has_readme=0 → status=ORPHANED, no LLM call."""
    rid = _seed_repo(db_with_domain, has_readme=False)

    def _explode(*args, **kwargs):
        raise AssertionError("LLM must not be called for README-less repo")

    result = classify(
        repo_slug="gh-owner-tool",
        conn=db_with_domain,
        call_llm=_explode,
    )
    assert result.status == RepoStatus.ORPHANED.value
    assert result.domain is None
    assert result.collection is None
    assert result.collections == ()

    row = db_with_domain.execute(
        "SELECT status, domain, collection FROM repos WHERE id = ?", (rid,)
    ).fetchone()
    assert row == (RepoStatus.ORPHANED.value, None, None)


def test_repo_with_empty_readme_marked_orphaned(db_with_domain):
    rid = _seed_repo(db_with_domain, readme_text="   \n")
    result = classify(
        repo_slug="gh-owner-tool",
        conn=db_with_domain,
        call_llm=lambda *a, **k: pytest.fail("should not call LLM"),
    )
    assert result.status == RepoStatus.ORPHANED.value


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_repo_with_readme_classifies_into_existing_domain(db_with_domain):
    rid = _seed_repo(db_with_domain)
    embedder = _OrthogonalEmbedder()

    result = classify(
        repo_slug="gh-owner-tool",
        conn=db_with_domain,
        call_llm=_runner_from_dict(_payload(
            collections=[
                {"index": -1, "new_name": "demo_collection",
                 "new_desc": "Cluster of work demos."},
            ],
            topics=["hierarchical retrieval", "tree search"],
        )),
        embedder=embedder,
    )

    assert result.status == RepoStatus.CLASSIFIED.value
    assert result.domain == "rag"
    assert result.collection == "demo_collection"
    assert result.collections == ("demo_collection",)
    assert "hierarchical retrieval" in result.topics
    assert "tree search" in result.topics

    row = db_with_domain.execute(
        "SELECT status, domain, collection FROM repos WHERE id = ?", (rid,)
    ).fetchone()
    assert row == (RepoStatus.CLASSIFIED.value, "rag", "demo_collection")

    topic_rows = db_with_domain.execute(
        "SELECT topic FROM topics WHERE target_kind='repo' AND target_id = ? "
        "ORDER BY topic",
        (rid,),
    ).fetchall()
    topics = [r[0] for r in topic_rows]
    assert topics == ["hierarchical retrieval", "tree search"]

    # Polymorphic collections row landed.
    coll_rows = db_with_domain.execute(
        "SELECT domain, collection, is_primary FROM collections "
        " WHERE target_kind='repo' AND target_id = ?",
        (rid,),
    ).fetchall()
    assert coll_rows == [("rag", "demo_collection", 1)]


def test_repo_multi_collection_writes_primary_and_secondary(db_with_domain):
    """Repo classify supports 1..4 collections, primary first."""
    rid = _seed_repo(db_with_domain)
    embedder = _OrthogonalEmbedder()

    result = classify(
        repo_slug="gh-owner-tool",
        conn=db_with_domain,
        call_llm=_runner_from_dict(_payload(
            collections=[
                {"index": -1, "new_name": "primary_cluster",
                 "new_desc": "Primary."},
                {"index": -1, "new_name": "secondary_cluster",
                 "new_desc": "Secondary."},
            ],
            topics=["t"],
        )),
        embedder=embedder,
    )

    assert result.collection == "primary_cluster"
    assert result.collections == ("primary_cluster", "secondary_cluster")
    # needs_review because both are new collections.
    assert result.needs_review is True

    rows = db_with_domain.execute(
        "SELECT collection, is_primary FROM collections "
        " WHERE target_kind='repo' AND target_id = ? "
        " ORDER BY is_primary DESC, collection",
        (rid,),
    ).fetchall()
    assert rows == [
        ("primary_cluster", 1),
        ("secondary_cluster", 0),
    ]


def test_repo_canonical_collision_dedupes_keeping_primary(db_with_domain):
    """Two picks resolving to the same canonical collapse into one row,
    primary preserved."""
    db_with_domain.execute(
        """
        INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES ('rag', 'collection', '', 'shared', 'seed')
        """
    )
    rid = _seed_repo(db_with_domain)
    embedder = _OrthogonalEmbedder()

    classify(
        repo_slug="gh-owner-tool",
        conn=db_with_domain,
        call_llm=_runner_from_dict(_payload(
            collections=[
                {"index": -1, "new_name": "shared", "new_desc": "Primary."},
                {"index": -1, "new_name": "Shared",
                 "new_desc": "Casefold dupe."},
            ],
            topics=["t"],
        )),
        embedder=embedder,
    )

    rows = db_with_domain.execute(
        "SELECT collection, is_primary FROM collections "
        " WHERE target_kind='repo' AND target_id = ?",
        (rid,),
    ).fetchall()
    assert rows == [("shared", 1)]


def test_paper_linked_repo_refuses_classify_repo(db_with_domain):
    """Paper-linked repos inherit taxonomy; routing them through
    classify_repo is a programming error."""
    db_with_domain.execute(
        "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, "
        "  abstract, pdf_url, ingested_at, status, domain, collection) "
        "VALUES ('2401.00001', 'p1', 't', '[]', '2024-01-01', 'a', 'u', 'd', "
        "  'fetched', NULL, NULL)"
    )
    paper_id = db_with_domain.execute(
        "SELECT id FROM papers WHERE paper_name='p1'"
    ).fetchone()[0]
    _seed_repo(db_with_domain, paper_id=paper_id)
    with pytest.raises(ClassifyRepoStateError, match="paper-linked"):
        classify(
            repo_slug="gh-owner-tool",
            conn=db_with_domain,
            call_llm=lambda *a, **k: pytest.fail("should not call LLM"),
        )


def test_rerun_replaces_topics_and_collections(db_with_domain):
    """Re-running classify on a repo replaces its prior topic + collection rows."""
    rid = _seed_repo(db_with_domain)
    embedder = _OrthogonalEmbedder()

    classify(
        repo_slug="gh-owner-tool",
        conn=db_with_domain,
        call_llm=_runner_from_dict(_payload(
            collections=[
                {"index": -1, "new_name": "first_cluster",
                 "new_desc": "Cluster A."},
                {"index": -1, "new_name": "extra_cluster",
                 "new_desc": "Cluster A2."},
            ],
            topics=["alpha", "beta"],
        )),
        embedder=embedder,
    )

    # Reset to REPO_FETCHED so a re-run is allowed.
    db_with_domain.execute(
        "UPDATE repos SET status = ? WHERE id = ?",
        (RepoStatus.REPO_FETCHED.value, rid),
    )

    classify(
        repo_slug="gh-owner-tool",
        conn=db_with_domain,
        call_llm=_runner_from_dict(_payload(
            collections=[
                {"index": -1, "new_name": "first_cluster",
                 "new_desc": "Cluster A."},
            ],
            topics=["gamma"],
        )),
        embedder=embedder,
    )

    topics = [r[0] for r in db_with_domain.execute(
        "SELECT topic FROM topics WHERE target_kind='repo' AND target_id = ?",
        (rid,),
    ).fetchall()]
    assert topics == ["gamma"]

    # Only the primary collection row from the second run remains.
    rows = db_with_domain.execute(
        "SELECT collection FROM collections "
        " WHERE target_kind='repo' AND target_id = ?",
        (rid,),
    ).fetchall()
    assert [r[0] for r in rows] == ["first_cluster"]


def test_domain_override_skips_llm_domain_choice(db_with_domain):
    """``domain_override`` forces the domain regardless of LLM output."""
    db_with_domain.execute(
        "INSERT OR IGNORE INTO domains (name) VALUES ('biology')"
    )
    rid = _seed_repo(db_with_domain)
    embedder = _OrthogonalEmbedder()

    result = classify(
        repo_slug="gh-owner-tool",
        conn=db_with_domain,
        domain_override="biology",
        call_llm=_runner_from_dict(_payload(
            collections=[
                {"index": -1, "new_name": "evolution",
                 "new_desc": "Evolution work."},
            ],
            topics=["phylogeny"],
        )),
        embedder=embedder,
    )
    assert result.domain == "biology"
