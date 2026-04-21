"""Tests for idempotent schema bootstrap and per-table invariants."""
from __future__ import annotations

import sqlite3
from contextlib import suppress

import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db

EXPECTED_TABLES = {
    "domains",
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
    "domains", "papers", "figures", "page_images",
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


def test_init_db_creates_all_12_tables(db_path):
    c = get_conn(db_path)
    try:
        init_db(c)
        assert _user_tables(c) == EXPECTED_TABLES
    finally:
        c.close()


def test_init_db_is_idempotent(db_path):
    c = get_conn(db_path)
    try:
        init_db(c)
        first_tables = _user_tables(c)

        # Seed one row per user (non-virtual) table so we can check "zero new
        # rows" on repeat init_db calls. Only plain tables accept these inserts
        # — skip the virtual ones (FTS5 / vec0) since they have required columns.
        c.execute("INSERT INTO domains (name) VALUES ('ml')")
        c.execute(
            "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, abstract, "
            "domain, content_hash, pdf_url, ingested_at, status) "
            "VALUES ('x', 'p', 't', '[]', '2026-04-20', 'a', 'ml', 'h', 'u', 'd', 'fetched')"
        )
        baseline_counts = _plain_row_counts(c)

        init_db(c)
        init_db(c)

        assert _user_tables(c) == first_tables == EXPECTED_TABLES
        # Plan: "zero new rows, zero errors" on repeat init_db.
        assert _plain_row_counts(c) == baseline_counts
    finally:
        c.close()


def test_vec0_existence_guard(db_path):
    """sqlite-vec vec0 does not support IF NOT EXISTS; idempotency must work anyway."""
    c = get_conn(db_path)
    try:
        init_db(c)
        # A naive second run without the sqlite_master guard would raise here.
        init_db(c)
        # Table must exist and be usable.
        exists = c.execute(
            "SELECT 1 FROM sqlite_master WHERE name='term_embeddings'"
        ).fetchone()
        assert exists is not None
    finally:
        c.close()


def test_canonical_terms_uniqueness(db_path):
    c = get_conn(db_path)
    try:
        init_db(c)
        c.execute(
            "INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in) "
            "VALUES (?, ?, ?, ?, ?)",
            ("ml", "entity", "method", "Transformer", "paper_1"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            c.execute(
                "INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in) "
                "VALUES (?, ?, ?, ?, ?)",
                ("ml", "entity", "method", "Transformer", "paper_2"),
            )
    finally:
        c.close()


def test_entities_uniqueness(db_path):
    c = get_conn(db_path)
    try:
        init_db(c)
        _seed_paper(c, paper_id=1)
        c.execute(
            "INSERT INTO entities "
            "(paper_id, domain, paper_name, entity_name, entity_type, source_breadcrumb, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "ml", "paper_1", "Transformer", "method", "# Intro", "a desc"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            c.execute(
                "INSERT INTO entities "
                "(paper_id, domain, paper_name, entity_name, entity_type, source_breadcrumb, description) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (1, "ml", "paper_1", "Transformer", "method", "# Intro", "other desc"),
            )
        # Different breadcrumb: must succeed.
        c.execute(
            "INSERT INTO entities "
            "(paper_id, domain, paper_name, entity_name, entity_type, source_breadcrumb, description) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "ml", "paper_1", "Transformer", "method", "# Method", "d2"),
        )
    finally:
        c.close()


def test_papers_raw_html_accepts_large_payload(db_path):
    """`raw_html` must exist and accept a multi-megabyte string."""
    c = get_conn(db_path)
    try:
        init_db(c)
        big = "x" * (2 * 1024 * 1024)  # 2 MB
        c.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", ("ml",))
        c.execute(
            """
            INSERT INTO papers (
                arxiv_id, paper_name, title, authors, date, abstract, domain,
                content_hash, pdf_url, ingested_at, status, raw_html
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("2512.99999", "big_paper", "t", "[]", "2026-04-20", "a", "ml",
             "hash", "http://x", "2026-04-20T00:00:00", "fetched", big),
        )
        row = c.execute(
            "SELECT length(raw_html) FROM papers WHERE paper_name = 'big_paper'"
        ).fetchone()
        assert row[0] == len(big)
    finally:
        c.close()


def test_term_embeddings_metadata_filters(db_path):
    """vec0 metadata columns (term_type/entity_type/domain) must be usable in WHERE.

    Seeds two rows with disjoint metadata so the filter has something to exclude;
    a silently-ignored filter would leak the second row and fail the test.
    """
    c = get_conn(db_path)
    try:
        init_db(c)
        # Two embeddings at the same point in space so both are equally-ranked by KNN.
        vec = [1.0] + [0.0] * 383
        c.execute(
            "INSERT INTO canonical_terms (id, domain, term_type, entity_type, canonical_name, first_seen_in) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (42, "ml", "entity", "method", "Transformer", "paper_1"),
        )
        c.execute(
            "INSERT INTO canonical_terms (id, domain, term_type, entity_type, canonical_name, first_seen_in) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (43, "bio", "collection", "", "Genomics", "paper_2"),
        )
        c.execute(
            "INSERT INTO term_embeddings (term_id, embedding, term_type, entity_type, domain) "
            "VALUES (?, ?, ?, ?, ?)",
            (42, _pack_f32(vec), "entity", "method", "ml"),
        )
        c.execute(
            "INSERT INTO term_embeddings (term_id, embedding, term_type, entity_type, domain) "
            "VALUES (?, ?, ?, ?, ?)",
            (43, _pack_f32(vec), "collection", "", "bio"),
        )

        # Each filter must return ONLY the matching row.
        for col, expected_val, expected_id in [
            ("term_type", "entity", 42),
            ("entity_type", "method", 42),
            ("domain", "ml", 42),
            ("term_type", "collection", 43),
            ("domain", "bio", 43),
        ]:
            rows = c.execute(
                f"""
                SELECT term_id FROM term_embeddings
                WHERE embedding MATCH ? AND k = 5 AND {col} = ?
                """,
                (_pack_f32(vec), expected_val),
            ).fetchall()
            ids = {r[0] for r in rows}
            assert ids == {expected_id}, (
                f"metadata filter on {col}={expected_val!r} returned {ids}, "
                f"expected exactly {{{expected_id}}}"
            )
    finally:
        c.close()


def _pack_f32(values):
    """Serialize a list[float] as sqlite-vec expects (little-endian float32 bytes)."""
    import struct
    return struct.pack(f"{len(values)}f", *values)
