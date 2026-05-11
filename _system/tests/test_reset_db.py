"""Tests for `_system/scripts/reset_db.py`.

The load-bearing invariant is that ``reset()`` preserves the DB file's
inode — running ``lodestone-mcp`` processes pin the inode at startup
(``mcp_server.py:_check_db_inode_pinned``), so a wipe that unlinks the
file would silently orphan their writes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.scripts.reset_db import reset, _user_tables


_TAXONOMY_FIXTURE = {
    "domains": [
        {
            "name": "Test Domain",
            "description": "fixture domain for reset_db tests",
            "collections": [
                {"name": "Test Collection A",
                 "description": "fixture collection a"},
                {"name": "Test Collection B",
                 "description": "fixture collection b"},
            ],
        }
    ]
}


def _seed_some_data(conn) -> None:
    """Drop a paper, a post, a repo, and a couple canonical terms in.

    Keeps inserts minimal — we just need non-zero rows in each headline
    table so the post-reset assertions are meaningful.
    """
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, ingested_at, status, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("0001.0001v1", "stub_paper_2026", "Stub Paper", "A. Stub",
         "2026-05-09", "stub abstract",
         "https://arxiv.org/pdf/0001.0001v1.pdf",
         "2026-05-09T00:00:00Z", "fetched", "0" * 64),
    )
    conn.execute(
        """
        INSERT INTO posts (
            post_name, source_url, canonical_url, title, date, abstract,
            raw_html, content_hash, ingested_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("stub_post_2026", "https://example.com/p", "https://example.com/p",
         "Stub", "2026-05-09", "stub abstract",
         "<p>x</p>", "1" * 64, "2026-05-09T00:00:00Z", "fetched"),
    )
    conn.execute(
        """
        INSERT INTO repos (
            repo_slug, url, host, owner, name, status, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("gh-x-y", "https://github.com/x/y", "github", "x", "y",
         "resolved", "2026-05-09T00:00:00Z"),
    )
    conn.execute(
        """
        INSERT INTO canonical_terms (
            domain, term_type, canonical_name, first_seen_in
        ) VALUES (?, ?, ?, ?)
        """,
        ("stub-domain", "topic", "Stub Topic", "stub_paper_2026"),
    )
    conn.commit()


def test_reset_wipes_user_data_and_reseeds_taxonomy(conn):
    _seed_some_data(conn)
    assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM repos").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms"
    ).fetchone()[0] == 1

    summary = reset(conn=conn, taxonomy=_TAXONOMY_FIXTURE)

    for table in ("papers", "posts", "repos", "canonical_terms",
                  "term_aliases", "topics", "collections", "term_embeddings",
                  "terms_fts", "sections", "readmes_fts"):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert n == 0, f"{table} should be empty after reset, got {n}"

    domain_rows = conn.execute(
        "SELECT name FROM domains"
    ).fetchall()
    assert len(domain_rows) == 1
    coll_rows = conn.execute(
        "SELECT name FROM collection_definitions"
    ).fetchall()
    assert len(coll_rows) == 2

    assert summary["taxonomy_seeded"]["domains_inserted"] == 1
    assert summary["taxonomy_seeded"]["collections_inserted"] == 2
    assert summary["total_rows_wiped"] >= 4  # paper + post + repo + term


def test_reset_preserves_inode(db_path: Path):
    """The whole point of TRUNCATE-in-place: same inode before and after.

    A running MCP server pins the inode at startup; if reset_db
    unlinked + recreated the file, the server would silently write to
    an orphan inode until restart.
    """
    conn = get_conn(db_path)
    init_db(conn)
    _seed_some_data(conn)

    inode_before = db_path.stat().st_ino

    try:
        reset(conn=conn, taxonomy=_TAXONOMY_FIXTURE)
    finally:
        conn.close()

    inode_after = db_path.stat().st_ino
    assert inode_before == inode_after, (
        f"reset must preserve inode (before={inode_before}, "
        f"after={inode_after}); MCP servers pinning the file would orphan."
    )


def test_reset_handles_fresh_db(tmp_path: Path):
    """Pointed at a path that doesn't exist, reset should create the
    schema (via init_db in main()) and seed taxonomy. We exercise the
    same path as main() short of the CLI parsing.
    """
    db = tmp_path / "fresh.db"
    assert not db.exists()

    conn = get_conn(db)
    try:
        init_db(conn)
        summary = reset(conn=conn, taxonomy=_TAXONOMY_FIXTURE)
    finally:
        conn.close()

    assert db.exists()
    assert summary["taxonomy_seeded"]["domains_inserted"] == 1
    assert summary["taxonomy_seeded"]["collections_inserted"] == 2


def test_reset_idempotent(conn):
    """Running reset twice in a row must succeed and converge to the same
    end state. ``reset`` always wipes the taxonomy before re-seeding, so
    the seed counts will be identical across runs.
    """
    first = reset(conn=conn, taxonomy=_TAXONOMY_FIXTURE)
    second = reset(conn=conn, taxonomy=_TAXONOMY_FIXTURE)

    assert (first["taxonomy_seeded"]["domains_inserted"]
            == second["taxonomy_seeded"]["domains_inserted"] == 1)
    assert (first["taxonomy_seeded"]["collections_inserted"]
            == second["taxonomy_seeded"]["collections_inserted"] == 2)

    assert conn.execute(
        "SELECT COUNT(*) FROM domains"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM collection_definitions"
    ).fetchone()[0] == 2


def test_user_tables_excludes_sqlite_internals_and_shadows(conn):
    tables = _user_tables(conn)
    for name in tables:
        assert not name.startswith("sqlite_")
        # FTS5 / vec0 shadows would slip past as plain tables; sql IS NULL
        # filters them, so none of these suffixes should appear.
        for suffix in ("_data", "_idx", "_docsize", "_content", "_config",
                       "_chunks", "_rowids"):
            assert not name.endswith(suffix), (
                f"shadow table {name!r} leaked through _user_tables filter"
            )

    # Sanity: the headline tables we wipe are present.
    for expected in ("papers", "posts", "repos", "canonical_terms",
                     "term_aliases", "topics", "collections",
                     "domains", "collection_definitions",
                     "terms_fts", "term_embeddings",
                     "sections", "readmes_fts"):
        assert expected in tables, f"missing user table: {expected}"
