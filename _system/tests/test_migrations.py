"""Tests for idempotent schema bootstrap and per-table invariants."""
from __future__ import annotations

import sqlite3

import pytest
from sqlite_vec import serialize_float32

from _system.db.migrations import init_db

EXPECTED_TABLES = {
    "domains",
    "collections",
    "papers",
    "figures",
    "page_images",
    "abstracts",
    "sections",
    "terms_fts",
    "canonical_terms",
    "term_aliases",
    "term_embeddings",
    "entities",
    "paper_topics",
}

# Virtual tables (FTS5, vec0) create auxiliary shadow tables; filter by prefix.
_SHADOW_PREFIXES = ("abstracts_", "sections_", "terms_fts_", "term_embeddings_")


def _user_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table','virtual') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {
        r[0] for r in rows
        if not any(r[0].startswith(p) for p in _SHADOW_PREFIXES)
    }


# Plain (non-virtual) tables — support row-count queries directly.
_PLAIN_TABLES = {
    "domains", "collections", "papers", "figures", "page_images",
    "canonical_terms", "term_aliases", "entities", "paper_topics",
}


def _plain_row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in _PLAIN_TABLES
    }


def _seed_paper(conn: sqlite3.Connection, paper_id: int = 1) -> None:
    conn.execute("INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
                 ("ml", "Machine learning"))
    conn.execute(
        """
        INSERT INTO papers (
            id, arxiv_id, paper_name, title, authors, date, abstract, domain,
            content_hash, pdf_url, ingested_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (paper_id, f"2512.{paper_id:05d}", f"paper_{paper_id}", "t", "[]",
         "2026-04-20", "abstract text", "ml", "deadbeef", "http://x", "2026-04-20T00:00:00", "fetched"),
    )


def test_init_db_creates_all_12_tables(conn):
    assert _user_tables(conn) == EXPECTED_TABLES


def test_init_db_is_idempotent(conn):
    first_tables = _user_tables(conn)

    # Seed one row per user (non-virtual) table so we can check "zero new
    # rows" on repeat init_db calls. Only plain tables accept these inserts
    # — skip the virtual ones (FTS5 / vec0) since they have required columns.
    conn.execute("INSERT INTO domains (name) VALUES ('ml')")
    conn.execute(
        "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, abstract, "
        "domain, content_hash, pdf_url, ingested_at, status) "
        "VALUES ('x', 'p', 't', '[]', '2026-04-20', 'a', 'ml', 'h', 'u', 'd', 'fetched')"
    )
    baseline_counts = _plain_row_counts(conn)

    init_db(conn)

    assert _user_tables(conn) == first_tables == EXPECTED_TABLES
    # Zero new rows, zero errors on repeat init_db.
    assert _plain_row_counts(conn) == baseline_counts


def test_vec0_existence_guard(conn):
    """sqlite-vec vec0 does not support IF NOT EXISTS; idempotency must work anyway."""
    # The conn fixture already ran init_db once; a naive second run without
    # the sqlite_master guard would raise here.
    init_db(conn)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='term_embeddings'"
    ).fetchone()
    assert exists is not None


def test_canonical_terms_uniqueness(conn):
    conn.execute(
        "INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in) "
        "VALUES (?, ?, ?, ?, ?)",
        ("ml", "entity", "method", "Transformer", "paper_1"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in) "
            "VALUES (?, ?, ?, ?, ?)",
            ("ml", "entity", "method", "Transformer", "paper_2"),
        )


def test_entities_uniqueness(conn):
    _seed_paper(conn, paper_id=1)
    conn.execute(
        "INSERT INTO entities "
        "(paper_id, domain, paper_name, entity_name, entity_type, source_breadcrumb, description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "ml", "paper_1", "Transformer", "method", "# Intro", "a desc"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO entities "
            "(paper_id, domain, paper_name, entity_name, entity_type, source_breadcrumb, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "ml", "paper_1", "Transformer", "method", "# Intro", "other desc"),
        )
    # Different breadcrumb: must succeed.
    conn.execute(
        "INSERT INTO entities "
        "(paper_id, domain, paper_name, entity_name, entity_type, source_breadcrumb, description) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "ml", "paper_1", "Transformer", "method", "# Method", "d2"),
    )


def test_init_db_backfills_collections_from_legacy_papers(conn):
    """Papers predating the `collections` table must be registered by `init_db`.

    Simulates an old database: a paper already has a (domain, collection)
    pair but no row in `collections`. `init_db` runs its one-shot backfill
    (``INSERT OR IGNORE INTO collections SELECT DISTINCT ...``) and the
    pair should appear.
    """
    conn.execute(
        "INSERT OR IGNORE INTO domains (name) VALUES (?)", ("rag",)
    )
    conn.execute(
        "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, abstract, "
        "domain, collection, content_hash, pdf_url, ingested_at, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("2401.00001", "legacy_paper", "t", "[]", "2026-04-20", "a",
         "rag", "hierarchical indexing", "h", "http://x",
         "2026-04-20T00:00:00", "classified"),
    )
    # Drop the backfilled row the fixture already created (if any) to prove
    # init_db is what puts it back.
    conn.execute("DELETE FROM collections")

    init_db(conn)

    row = conn.execute(
        "SELECT domain, name, description FROM collections "
        " WHERE domain = ? AND name = ?",
        ("rag", "hierarchical indexing"),
    ).fetchone()
    assert row is not None
    assert row[2] is None  # legacy rows land with NULL description


def test_papers_raw_html_accepts_large_payload(conn):
    """`raw_html` must exist and accept a multi-megabyte string."""
    big = "x" * (2 * 1024 * 1024)  # 2 MB
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", ("ml",))
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract, domain,
            content_hash, pdf_url, ingested_at, status, raw_html
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("2512.99999", "big_paper", "t", "[]", "2026-04-20", "a", "ml",
         "hash", "http://x", "2026-04-20T00:00:00", "fetched", big),
    )
    row = conn.execute(
        "SELECT length(raw_html) FROM papers WHERE paper_name = 'big_paper'"
    ).fetchone()
    assert row[0] == len(big)


def test_term_embeddings_metadata_filters(conn):
    """vec0 metadata columns (term_type/entity_type/domain) must be usable in WHERE.

    Seeds two rows with disjoint metadata so the filter has something to exclude;
    a silently-ignored filter would leak the second row and fail the test.
    """
    # Two embeddings at the same point in space so both are equally-ranked by KNN.
    vec = serialize_float32([1.0] + [0.0] * 383)
    conn.execute(
        "INSERT INTO canonical_terms (id, domain, term_type, entity_type, canonical_name, first_seen_in) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (42, "ml", "entity", "method", "Transformer", "paper_1"),
    )
    conn.execute(
        "INSERT INTO canonical_terms (id, domain, term_type, entity_type, canonical_name, first_seen_in) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (43, "bio", "collection", "", "Genomics", "paper_2"),
    )
    conn.execute(
        "INSERT INTO term_embeddings (term_id, embedding, term_type, entity_type, domain) "
        "VALUES (?, ?, ?, ?, ?)",
        (42, vec, "entity", "method", "ml"),
    )
    conn.execute(
        "INSERT INTO term_embeddings (term_id, embedding, term_type, entity_type, domain) "
        "VALUES (?, ?, ?, ?, ?)",
        (43, vec, "collection", "", "bio"),
    )

    # Each filter must return ONLY the matching row.
    for col, expected_val, expected_id in [
        ("term_type", "entity", 42),
        ("entity_type", "method", 42),
        ("domain", "ml", 42),
        ("term_type", "collection", 43),
        ("domain", "bio", 43),
    ]:
        rows = conn.execute(
            f"""
            SELECT term_id FROM term_embeddings
            WHERE embedding MATCH ? AND k = 5 AND {col} = ?
            """,
            (vec, expected_val),
        ).fetchall()
        ids = {r[0] for r in rows}
        assert ids == {expected_id}, (
            f"metadata filter on {col}={expected_val!r} returned {ids}, "
            f"expected exactly {{{expected_id}}}"
        )
