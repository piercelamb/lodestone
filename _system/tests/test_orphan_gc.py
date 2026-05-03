"""Unit tests for ``_system/db/orphan_gc.py``.

Seeds canonical_terms / paper_topics / collections / term_aliases /
term_embeddings / terms_fts directly, calls the helper, and asserts the
expected rows go (or stay).
"""
from __future__ import annotations

import sqlite3
import struct

import pytest

from _system.db.orphan_gc import gc_orphan_topic_collection_canonicals


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _embed_blob() -> bytes:
    vec = [0.0] * 384
    vec[0] = 1.0
    return struct.pack(f"{len(vec)}f", *vec)


def _seed_domain(conn: sqlite3.Connection, name: str = "rag") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO domains (name, description) VALUES (?, NULL)",
        (name,),
    )


def _seed_canonical(
    conn: sqlite3.Connection,
    *,
    domain: str,
    term_type: str,
    canonical_name: str,
    first_seen_in: str = "some_paper",
    entity_type: str = "",
    with_embedding: bool = True,
    with_fts: bool = True,
) -> int:
    conn.execute(
        """
        INSERT INTO canonical_terms
            (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES (?, ?, ?, ?, ?)
        """,
        (domain, term_type, entity_type, canonical_name, first_seen_in),
    )
    term_id = conn.execute(
        """
        SELECT id FROM canonical_terms
         WHERE domain = ? AND term_type = ? AND canonical_name = ?
        """,
        (domain, term_type, canonical_name),
    ).fetchone()[0]
    if with_fts:
        conn.execute(
            """
            INSERT INTO terms_fts
                (term_id, domain, term_type, entity_type,
                 canonical_name, aliases)
            VALUES (?, ?, ?, ?, ?, '')
            """,
            (term_id, domain, term_type, entity_type, canonical_name),
        )
    if with_embedding:
        conn.execute(
            """
            INSERT INTO term_embeddings
                (term_id, embedding, term_type, entity_type, domain)
            VALUES (?, ?, ?, ?, ?)
            """,
            (term_id, _embed_blob(), term_type, entity_type, domain),
        )
    return term_id


def _seed_paper(
    conn: sqlite3.Connection,
    *,
    paper_name: str,
    arxiv_id: str,
    domain: str,
    collection: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, ingested_at, status, domain, collection
        ) VALUES (?, ?, 'T', '[]', '2024-01-01', 'A',
                  ?, '2024-01-02T00:00:00+00:00', 'CLASSIFIED', ?, ?)
        """,
        (arxiv_id, paper_name, f"https://arxiv.org/pdf/{arxiv_id}",
         domain, collection),
    )
    return cur.lastrowid


# ===========================================================================
# Topic GC
# ===========================================================================


def test_orphan_topic_canonical_is_deleted(conn):
    _seed_domain(conn)
    term_id = _seed_canonical(
        conn, domain="rag", term_type="topic", canonical_name="orphan_topic",
    )

    counts = gc_orphan_topic_collection_canonicals(conn)

    assert counts["topics"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (term_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM terms_fts WHERE term_id = ?", (term_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM term_embeddings WHERE term_id = ?", (term_id,)
    ).fetchone()[0] == 0


def test_topic_with_paper_topics_row_survives(conn):
    _seed_domain(conn)
    term_id = _seed_canonical(
        conn, domain="rag", term_type="topic", canonical_name="bound_topic",
    )
    paper_id = _seed_paper(
        conn, paper_name="bound_paper", arxiv_id="2401.99991", domain="rag",
    )
    conn.execute(
        "INSERT INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
        (paper_id, "rag", "bound_topic"),
    )

    counts = gc_orphan_topic_collection_canonicals(conn)

    assert counts["topics"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (term_id,)
    ).fetchone()[0] == 1


# ===========================================================================
# Collection GC
# ===========================================================================


def test_orphan_collection_canonical_is_deleted(conn):
    _seed_domain(conn)
    # Seed the registry row for an orphan collection — a collection
    # canonical normally lands alongside a `collections` registry row
    # (classify_paper INSERT OR IGNOREs both). The GC should remove both.
    conn.execute(
        "INSERT INTO collections (domain, name, description) "
        "VALUES (?, ?, NULL)",
        ("rag", "orphan_coll"),
    )
    term_id = _seed_canonical(
        conn,
        domain="rag",
        term_type="collection",
        canonical_name="orphan_coll",
    )

    counts = gc_orphan_topic_collection_canonicals(conn)

    assert counts["collections"] == 1
    assert counts["collections_registry"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (term_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM collections WHERE domain = ? AND name = ?",
        ("rag", "orphan_coll"),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM terms_fts WHERE term_id = ?", (term_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM term_embeddings WHERE term_id = ?", (term_id,)
    ).fetchone()[0] == 0


def test_collection_with_papers_collection_reference_survives(conn):
    _seed_domain(conn)
    conn.execute(
        "INSERT INTO collections (domain, name, description) "
        "VALUES (?, ?, NULL)",
        ("rag", "bound_coll"),
    )
    term_id = _seed_canonical(
        conn,
        domain="rag",
        term_type="collection",
        canonical_name="bound_coll",
    )
    _seed_paper(
        conn,
        paper_name="bound_coll_paper",
        arxiv_id="2401.99992",
        domain="rag",
        collection="bound_coll",
    )

    counts = gc_orphan_topic_collection_canonicals(conn)

    assert counts["collections"] == 0
    assert counts["collections_registry"] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (term_id,)
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM collections WHERE name = ?", ("bound_coll",),
    ).fetchone()[0] == 1


# ===========================================================================
# Entity scope (out of scope — never GC'd)
# ===========================================================================


def test_entity_canonical_never_gcd_even_when_orphaned(conn):
    """Entity canonicals are out of scope under the synonym-index regime —
    tier-1 mentions leave no per-paper trace, so substantiation can't be
    proven. The GC must leave them alone even when they have no
    term_aliases rows."""
    _seed_domain(conn)
    term_id = _seed_canonical(
        conn,
        domain="rag",
        term_type="entity",
        canonical_name="LonelyMethod",
        entity_type="method",
    )

    counts = gc_orphan_topic_collection_canonicals(conn)

    assert "entities" not in counts
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (term_id,)
    ).fetchone()[0] == 1


# ===========================================================================
# term_aliases cleanup
# ===========================================================================


def test_term_aliases_rows_for_orphan_topic_are_also_deleted(conn):
    """Topics can land tier-2/3/4 alias rows when the resolver fuzzy-matches
    a synonym to an existing canonical. If the canonical orphans, those
    alias rows go too (they reference a no-longer-existent term_id)."""
    _seed_domain(conn)
    term_id = _seed_canonical(
        conn, domain="rag", term_type="topic",
        canonical_name="orphan_topic_with_alias",
    )
    conn.execute(
        """
        INSERT INTO term_aliases
            (term_id, alias, source_paper, match_tier)
        VALUES (?, ?, ?, ?)
        """,
        (term_id, "fuzzy_form", "some_paper", 2),
    )

    gc_orphan_topic_collection_canonicals(conn)

    assert conn.execute(
        "SELECT COUNT(*) FROM term_aliases WHERE term_id = ?", (term_id,)
    ).fetchone()[0] == 0


# ===========================================================================
# Counts dict
# ===========================================================================


def test_returns_counts_dict(conn):
    _seed_domain(conn)
    # one bound topic
    bound_topic_id = _seed_canonical(
        conn, domain="rag", term_type="topic", canonical_name="bound_t",
    )
    paper_id = _seed_paper(
        conn, paper_name="anchor", arxiv_id="2401.10001", domain="rag",
    )
    conn.execute(
        "INSERT INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
        (paper_id, "rag", "bound_t"),
    )
    # two orphan topics
    _seed_canonical(
        conn, domain="rag", term_type="topic", canonical_name="orphan_t1",
    )
    _seed_canonical(
        conn, domain="rag", term_type="topic", canonical_name="orphan_t2",
    )
    # one orphan collection (registry row + canonical)
    conn.execute(
        "INSERT INTO collections (domain, name, description) "
        "VALUES (?, ?, NULL)",
        ("rag", "orphan_c1"),
    )
    _seed_canonical(
        conn,
        domain="rag",
        term_type="collection",
        canonical_name="orphan_c1",
    )

    counts = gc_orphan_topic_collection_canonicals(conn)

    assert counts == {
        "topics": 2,
        "collections": 1,
        "collections_registry": 1,
    }
    # Bound topic survives.
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (bound_topic_id,)
    ).fetchone()[0] == 1
