"""Tests for the new repo-tree / read-code modes and `--scope` BM25.

The tiny corpus seeds two papers — one with a fresh repo + README, one
with no repo at all — so every soft-failure path has a target.
"""
from __future__ import annotations

import sqlite3

import pytest

from _system.scripts import search as search_mod


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_paper(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str,
    paper_name: str,
    domain: str = "rag",
    code_repo: str | None = "https://github.com/owner/repo",
    status: str = "repo_fetched",
    abstract: str = "Demo abstract.",
) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO domains (name) VALUES (?)", (domain,)
    )
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, ingested_at, status, domain, code_repo,
            code_repo_commit, code_repo_fetched_at, markdown
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id, paper_name, "T", '["A"]', "2024-01-01", abstract,
            f"https://arxiv.org/pdf/{arxiv_id}",
            "2024-01-02T00:00:00+00:00",
            status, domain, code_repo,
            "abc123" if code_repo else None,
            "2024-01-03T00:00:00+00:00" if code_repo else None,
            "# Abstract\n\n" + abstract + "\n",
        ),
    )
    paper_id = cur.lastrowid
    # Mirror the abstract into sections so BM25 has something to find.
    conn.execute(
        "INSERT INTO sections (paper_id, domain, paper_name, section_title, "
        "  section_level, body) VALUES (?, ?, ?, 'Abstract', '1', ?)",
        (paper_id, domain, paper_name, abstract),
    )
    return paper_id


def _seed_code_files(conn, paper_id: int, files: list[tuple[str, str | None, str]]) -> None:
    for path, lang, content in files:
        conn.execute(
            "INSERT INTO code_files (paper_id, path, language, size_bytes, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (paper_id, path, lang, len(content.encode("utf-8")), content),
        )


def _seed_readme(conn, *, paper_id: int, paper_name: str, domain: str,
                 path: str, content: str) -> None:
    conn.execute(
        "INSERT INTO readmes_fts (paper_id, domain, paper_name, path, content) "
        "VALUES (?, ?, ?, ?, ?)",
        (paper_id, domain, paper_name, path, content),
    )


@pytest.fixture
def seeded(conn: sqlite3.Connection) -> sqlite3.Connection:
    p1 = _seed_paper(
        conn,
        arxiv_id="2401.00001",
        paper_name="repo_paper_2026",
        abstract="A paper about sparse mixture of experts cross-attention.",
    )
    _seed_code_files(conn, p1, [
        ("README.md", "markdown", "# Repo Paper\n\nMoE training pipeline.\n"),
        ("src/model.py", "python", "def train():\n    pass\n"),
        ("src/utils.py", "python",
         "".join(f"line{n}\n" for n in range(1, 21))),
    ])
    _seed_readme(
        conn,
        paper_id=p1, paper_name="repo_paper_2026", domain="rag",
        path="README.md",
        content="# Repo Paper\n\nMoE training pipeline.\n",
    )

    _seed_paper(
        conn,
        arxiv_id="2402.00002",
        paper_name="no_repo_paper_2026",
        code_repo=None,
        status="indexed",
    )
    return conn


# ---------------------------------------------------------------------------
# --repo-tree
# ---------------------------------------------------------------------------


def test_repo_tree_returns_paths_sorted(seeded):
    payload = search_mod.mode_repo_tree(seeded, paper_name="repo_paper_2026")
    assert payload["mode"] == "repo_tree"
    assert payload["status"] == "ok"
    assert payload["code_repo"] == "https://github.com/owner/repo"
    assert payload["commit"] == "abc123"
    paths = [f["path"] for f in payload["files"]]
    assert paths == sorted(paths)
    assert "README.md" in paths
    assert payload["file_count"] == len(paths)


def test_repo_tree_no_repo_returns_soft_status(seeded):
    payload = search_mod.mode_repo_tree(seeded, paper_name="no_repo_paper_2026")
    assert payload["status"] == "no_repo"
    assert "hint" in payload


def test_repo_tree_failed_repo_returns_soft_status(conn):
    _seed_paper(
        conn, arxiv_id="2403.00003", paper_name="failed_paper_2026",
        status="failed_repo",
    )
    payload = search_mod.mode_repo_tree(conn, paper_name="failed_paper_2026")
    assert payload["status"] == "failed_repo"
    assert payload["code_repo"] == "https://github.com/owner/repo"


# ---------------------------------------------------------------------------
# --read-code
# ---------------------------------------------------------------------------


def test_read_code_returns_full_content(seeded):
    payload = search_mod.mode_read_code(
        seeded, paper_name="repo_paper_2026", path="src/model.py",
    )
    assert payload["status"] == "ok"
    assert payload["language"] == "python"
    assert "def train" in payload["content"]


def test_read_code_lines_slice(seeded):
    payload = search_mod.mode_read_code(
        seeded, paper_name="repo_paper_2026",
        path="src/utils.py", lines="3-5",
    )
    assert payload["status"] == "ok"
    assert payload["content"] == "line3\nline4\nline5\n"
    assert payload["lines"] == [3, 5]


def test_read_code_path_not_found_soft_failure(seeded):
    payload = search_mod.mode_read_code(
        seeded, paper_name="repo_paper_2026", path="nope.py",
    )
    assert payload["status"] == "file_not_found"
    assert "hint" in payload


@pytest.mark.parametrize("bad", ["", "10", "10-", "abc-3", "5-3", "0-2"])
def test_read_code_malformed_lines_soft_failure(seeded, bad):
    payload = search_mod.mode_read_code(
        seeded, paper_name="repo_paper_2026",
        path="src/utils.py", lines=bad,
    )
    assert payload["status"] == "malformed_lines", (bad, payload)
    assert "hint" in payload


# ---------------------------------------------------------------------------
# BM25 envelope: code_repo metadata
# ---------------------------------------------------------------------------


def test_bm25_envelope_carries_code_repo_metadata(seeded):
    payload = search_mod.mode_bm25(
        seeded, query="sparse mixture", filters={}, limit=5,
    )
    hits = [h for h in payload["results"] if h["paper_name"] == "repo_paper_2026"]
    assert hits, payload
    cr = hits[0]["code_repo"]
    assert cr is not None
    assert cr["url"] == "https://github.com/owner/repo"
    assert cr["status"] == "repo_fetched"
    assert cr["file_count"] == 3


def test_taxonomy_lookup_envelope_carries_code_repo_per_paper(conn):
    """Mode 2 (--collection lookup) decorates papers with code_repo."""
    p1 = _seed_paper(
        conn, arxiv_id="2401.10000", paper_name="collected_paper_2026",
    )
    conn.execute(
        "UPDATE papers SET collection = ? WHERE id = ?",
        ("hierarchical indexing", p1),
    )
    _seed_code_files(conn, p1, [("README.md", "markdown", "# x\n")])
    # Canonical + terms_fts row for the collection.
    coll = conn.execute(
        "INSERT INTO canonical_terms (domain, term_type, entity_type, "
        " canonical_name, first_seen_in) "
        "VALUES (?, 'collection', '', ?, ?)",
        ("rag", "hierarchical indexing", "collected_paper_2026"),
    ).lastrowid
    conn.execute(
        "INSERT INTO terms_fts "
        "  (term_id, domain, term_type, entity_type, canonical_name, aliases) "
        "VALUES (?, 'rag', 'collection', '', ?, '')",
        (coll, "hierarchical indexing"),
    )

    payload = search_mod.mode_taxonomy_lookup(
        conn, term="hierarchical indexing", kind="collection",
        filters={"domain": "rag"},
    )
    assert payload.get("error") is None, payload
    papers = payload.get("papers") or []
    assert any(p["paper_name"] == "collected_paper_2026" for p in papers)
    target = next(p for p in papers if p["paper_name"] == "collected_paper_2026")
    assert target["code_repo"] is not None
    assert target["code_repo"]["url"] == "https://github.com/owner/repo"


# ---------------------------------------------------------------------------
# --scope: sections / readmes / both
# ---------------------------------------------------------------------------


def test_bm25_scope_default_sections_unchanged(seeded):
    """No --scope passed → sections-only behavior, no `scope` key in payload."""
    payload = search_mod.mode_bm25(
        seeded, query="sparse mixture", filters={}, limit=5,
    )
    assert payload["mode"] == "sections"
    assert "scope" not in payload  # default path doesn't tag the scope
    assert any(h["paper_name"] == "repo_paper_2026" for h in payload["results"])


def test_bm25_scope_readmes_finds_paper_via_readme_only(seeded):
    """A query for a README-only term hits the paper through the README index."""
    payload = search_mod.mode_bm25(
        seeded, query="MoE training", filters={}, limit=5,
        scope=search_mod.Scope.READMES,
    )
    assert payload.get("scope") == "readmes"
    hits = [h for h in payload["results"] if h["paper_name"] == "repo_paper_2026"]
    assert hits, payload
    assert hits[0]["readme_hit"] is not None
    assert hits[0]["readme_hit"]["path"] == "README.md"


def test_bm25_scope_both_unions_hits_with_readme_hit_field(seeded):
    payload = search_mod.mode_bm25(
        seeded, query="MoE training", filters={}, limit=5,
        scope=search_mod.Scope.BOTH,
    )
    assert payload.get("scope") == "both"
    hits = [h for h in payload["results"] if h["paper_name"] == "repo_paper_2026"]
    assert hits, payload
    # README-only match → readme_hit populated, no section hits required.
    assert hits[0]["readme_hit"] is not None


def test_bm25_scope_readmes_empty_query_soft_failure(seeded):
    payload = search_mod.mode_bm25(
        seeded, query="---", filters={}, limit=5,
        scope=search_mod.Scope.READMES,
    )
    assert payload.get("status") == "empty_query"


def test_bm25_scope_without_query_errors():
    """`--scope readmes` without a positional QUERY is a CLI mistake."""
    parser = search_mod._build_parser()
    with pytest.raises(SystemExit):
        # _check_mode_conflicts calls parser.error(), which exits.
        ns = parser.parse_args(["--scope", "readmes"])
        search_mod._check_mode_conflicts(parser, ns)
