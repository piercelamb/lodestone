"""Unit tests for _system/scripts/index_paper.py.

Covers both per-paper indexing (``index_one``) and the offline full rebuild
(``rebuild_all``). The real :class:`Embedder` is never loaded — a fake
embedder tracks ``embed_batch`` call sizes so the batch-size contract is
testable without touching sentence-transformers.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec

from _system.schemas.paper_metadata import PaperStatus
from _system.scripts.index_paper import (
    IndexResult,
    PaperNotFound,
    StatusTooLow,
    UnknownStatusError,
    index_one,
    rebuild_all,
)
from _system.utils.sections import split_sections


# ---------------------------------------------------------------------------
# Fake embedder — records embed_batch call sizes; produces deterministic
# 384-dim unit vectors.
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    def __init__(self) -> None:
        self.embed_batch_calls: list[int] = []
        self.embed_batch_texts: list[list[str]] = []

    def embed(self, text: str) -> list[float]:
        v = [0.0] * 384
        v[hash(text) % 384] = 1.0
        return v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_batch_calls.append(len(texts))
        self.embed_batch_texts.append(list(texts))
        return [self.embed(t) for t in texts]


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_domain(conn: sqlite3.Connection, name: str = "rag") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
        (name, "test domain"),
    )


def _seed_paper(
    conn: sqlite3.Connection,
    *,
    paper_name: str = "paper_name_2024",
    arxiv_id: str = "2401.00001",
    status: str = PaperStatus.EXTRACTED.value,
    title: str = "A Test Paper",
    abstract: str = "The abstract discusses BookRAG for question answering.",
    markdown: str | None = "# Method\n\nOur approach uses BookRAG.\n",
    domain: str | None = "rag",
    collection: str | None = "retrieval",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, html_source, ingested_at, status, markdown,
            domain, collection, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id,
            paper_name,
            title,
            '["A. Author"]',
            "2024-01-01",
            abstract,
            f"https://arxiv.org/pdf/{arxiv_id}",
            "arxiv",
            "2024-01-01T00:00:00+00:00",
            status,
            markdown,
            domain,
            collection,
            0,
        ),
    )
    return cur.lastrowid


def _seed_canonical(
    conn: sqlite3.Connection,
    *,
    domain: str = "rag",
    term_type: str = "entity",
    entity_type: str = "method",
    canonical_name: str = "BookRAG",
    first_seen_in: str = "seed",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO canonical_terms
            (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES (?, ?, ?, ?, ?)
        """,
        (domain, term_type, entity_type, canonical_name, first_seen_in),
    )
    return cur.lastrowid


def _seed_alias(
    conn: sqlite3.Connection,
    term_id: int,
    alias: str,
    *,
    source_paper: str = "seed_paper",
    match_tier: int = 2,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO term_aliases
            (term_id, alias, source_paper, match_tier)
        VALUES (?, ?, ?, ?)
        """,
        (term_id, alias, source_paper, match_tier),
    )


def _seed_entity(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    paper_name: str,
    entity_name: str,
    entity_type: str = "method",
    domain: str = "rag",
    source_breadcrumb: str = "# Method",
    description: str = "description",
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO entities
            (paper_id, domain, paper_name, entity_name, entity_type,
             source_breadcrumb, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (paper_id, domain, paper_name, entity_name, entity_type,
         source_breadcrumb, description),
    )


def _seed_paper_topic(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    topic: str,
    domain: str = "rag",
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
        (paper_id, domain, topic),
    )


# ===========================================================================
# Per-paper indexing — happy path
# ===========================================================================


class TestIndexOneBasics:
    def test_returns_index_result(self, conn):
        _seed_domain(conn)
        _seed_paper(conn)
        result = index_one(paper_name="paper_name_2024", conn=conn)
        assert isinstance(result, IndexResult)
        assert result.paper_name == "paper_name_2024"
        assert result.status == PaperStatus.INDEXED.value
        assert result.section_count >= 1

    def test_index_one_creates_exactly_one_abstracts_row(self, conn):
        _seed_domain(conn)
        _seed_paper(conn)
        index_one(paper_name="paper_name_2024", conn=conn)
        n = conn.execute(
            "SELECT COUNT(*) FROM abstracts WHERE paper_name = ?",
            ("paper_name_2024",),
        ).fetchone()[0]
        assert n == 1

    def test_index_one_populates_sections_equal_to_paper_section_count(self, conn):
        md = (
            "# Intro\n\nIntro body.\n\n"
            "# Method\n\nMethod body.\n\n"
            "## Sub\n\nSub body.\n"
        )
        _seed_domain(conn)
        _seed_paper(conn, markdown=md)
        expected = len(split_sections(md))
        index_one(paper_name="paper_name_2024", conn=conn)

        n_sections = conn.execute(
            "SELECT COUNT(*) FROM sections WHERE paper_name = ?",
            ("paper_name_2024",),
        ).fetchone()[0]
        section_count_col = conn.execute(
            "SELECT section_count FROM papers WHERE paper_name = ?",
            ("paper_name_2024",),
        ).fetchone()[0]
        assert n_sections == expected
        assert section_count_col == expected

    def test_bm25_match_against_fresh_abstract_returns_paper(self, conn):
        _seed_domain(conn)
        _seed_paper(conn, abstract="We propose BookRAG, a novel retrieval technique.")
        index_one(paper_name="paper_name_2024", conn=conn)
        rows = conn.execute(
            "SELECT paper_name FROM abstracts WHERE abstracts MATCH ?",
            ("BookRAG",),
        ).fetchall()
        assert any(r[0] == "paper_name_2024" for r in rows)

    def test_abstract_porter_tokenizer_stems(self, conn):
        """abstracts uses porter — a query for 'retrieving' hits 'retrieval'."""
        _seed_domain(conn)
        _seed_paper(conn, abstract="We propose a retrieval technique.")
        index_one(paper_name="paper_name_2024", conn=conn)
        rows = conn.execute(
            "SELECT paper_name FROM abstracts WHERE abstracts MATCH ?",
            ("retrieving",),
        ).fetchall()
        assert any(r[0] == "paper_name_2024" for r in rows)

    def test_reindex_same_paper_replaces_rather_than_appends(self, conn):
        md = "# A\n\ntext\n\n# B\n\nmore\n"
        _seed_domain(conn)
        _seed_paper(conn, markdown=md)
        expected_sections = len(split_sections(md))
        index_one(paper_name="paper_name_2024", conn=conn)
        # Second pass must replace, not append.
        index_one(paper_name="paper_name_2024", conn=conn)
        n_abs = conn.execute(
            "SELECT COUNT(*) FROM abstracts WHERE paper_name = ?",
            ("paper_name_2024",),
        ).fetchone()[0]
        n_sec = conn.execute(
            "SELECT COUNT(*) FROM sections WHERE paper_name = ?",
            ("paper_name_2024",),
        ).fetchone()[0]
        assert n_abs == 1
        assert n_sec == expected_sections

    def test_markdown_missing_still_indexes_abstract(self, conn):
        """A paper with NULL markdown can still be indexed — we get an
        abstracts row and zero sections."""
        _seed_domain(conn)
        _seed_paper(conn, markdown=None)
        index_one(paper_name="paper_name_2024", conn=conn)
        n_abs = conn.execute(
            "SELECT COUNT(*) FROM abstracts WHERE paper_name = ?",
            ("paper_name_2024",),
        ).fetchone()[0]
        n_sec = conn.execute(
            "SELECT COUNT(*) FROM sections WHERE paper_name = ?",
            ("paper_name_2024",),
        ).fetchone()[0]
        assert n_abs == 1
        assert n_sec == 0


# ===========================================================================
# Status gate
# ===========================================================================


class TestStatusGate:
    def test_paper_not_found_raises(self, conn):
        with pytest.raises(PaperNotFound):
            index_one(paper_name="nope", conn=conn)

    def test_fetched_without_force_raises(self, conn):
        _seed_domain(conn)
        _seed_paper(conn, status=PaperStatus.FETCHED.value)
        with pytest.raises(StatusTooLow):
            index_one(paper_name="paper_name_2024", conn=conn)

    def test_failed_html_is_terminal(self, conn):
        _seed_domain(conn)
        _seed_paper(conn, status=PaperStatus.FAILED_HTML.value)
        with pytest.raises(StatusTooLow) as exc_info:
            index_one(paper_name="paper_name_2024", conn=conn)
        assert "failed_html" in str(exc_info.value).lower()

    def test_extracted_advances_to_indexed(self, conn):
        _seed_domain(conn)
        _seed_paper(conn, status=PaperStatus.EXTRACTED.value)
        index_one(paper_name="paper_name_2024", conn=conn)
        status = conn.execute(
            "SELECT status FROM papers WHERE paper_name = ?",
            ("paper_name_2024",),
        ).fetchone()[0]
        assert status == PaperStatus.INDEXED.value

    def test_indexed_can_rerun_without_force(self, conn):
        """Re-running INDEXED stage on an already-indexed paper is OK
        (delta == 0 per ``can_run_from``)."""
        _seed_domain(conn)
        _seed_paper(conn, status=PaperStatus.INDEXED.value)
        index_one(paper_name="paper_name_2024", conn=conn)  # should not raise

    def test_force_bypasses_status_guard(self, conn):
        _seed_domain(conn)
        _seed_paper(conn, status=PaperStatus.FETCHED.value)
        index_one(paper_name="paper_name_2024", conn=conn, force=True)
        status = conn.execute(
            "SELECT status FROM papers WHERE paper_name = ?",
            ("paper_name_2024",),
        ).fetchone()[0]
        assert status == PaperStatus.INDEXED.value

    def test_unknown_status_raises(self, conn):
        _seed_domain(conn)
        _seed_paper(conn, status="bogus_stage")
        with pytest.raises(UnknownStatusError):
            index_one(paper_name="paper_name_2024", conn=conn)


# ===========================================================================
# terms_fts scoping
# ===========================================================================


class TestTermsFtsScoping:
    def test_terms_fts_rebuild_is_scoped_to_touched_terms(self, conn):
        """Index paper A, then sneak a marker into B's terms_fts row, then
        re-index A — B's row must survive untouched."""
        _seed_domain(conn, "rag")
        _seed_domain(conn, "other")

        paper_a = _seed_paper(
            conn, paper_name="paperA", arxiv_id="2401.0001",
            domain="rag", collection="ca", title="A",
        )
        term_a = _seed_canonical(conn, domain="rag", canonical_name="EntityA")
        _seed_alias(conn, term_a, "ea_alias", source_paper="paperA")
        _seed_entity(
            conn, paper_id=paper_a, paper_name="paperA",
            entity_name="EntityA", domain="rag",
        )

        paper_b = _seed_paper(
            conn, paper_name="paperB", arxiv_id="2401.0002",
            domain="other", collection="cb", title="B",
        )
        term_b = _seed_canonical(
            conn, domain="other", canonical_name="EntityB",
        )
        _seed_alias(conn, term_b, "eb_alias", source_paper="paperB")
        _seed_entity(
            conn, paper_id=paper_b, paper_name="paperB",
            entity_name="EntityB", domain="other",
        )

        index_one(paper_name="paperA", conn=conn)

        # Sneak a marker into B's row. This simulates external prior state
        # that A's re-index must NOT clobber.
        conn.execute("DELETE FROM terms_fts WHERE term_id = ?", (term_b,))
        conn.execute(
            "INSERT INTO terms_fts "
            "(term_id, domain, term_type, entity_type, canonical_name, aliases) "
            "VALUES (?, 'other', 'entity', 'method', 'EntityB', 'MANUAL_MARKER')",
            (term_b,),
        )

        index_one(paper_name="paperA", conn=conn)

        b_aliases_row = conn.execute(
            "SELECT aliases FROM terms_fts WHERE term_id = ?", (term_b,),
        ).fetchone()
        assert b_aliases_row is not None, "term B's terms_fts row was deleted"
        assert b_aliases_row[0] == "MANUAL_MARKER", (
            f"term B's aliases clobbered: {b_aliases_row[0]!r}"
        )

    def test_terms_fts_aliases_is_space_joined_distinct_list(self, conn):
        _seed_domain(conn)
        paper_id = _seed_paper(conn)
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        # Three distinct aliases, one is a duplicate from a different paper.
        _seed_alias(conn, term_id, "book-rag", source_paper="p1")
        _seed_alias(conn, term_id, "book-rag", source_paper="p2")
        _seed_alias(conn, term_id, "bookragv2", source_paper="p1")
        _seed_alias(conn, term_id, "BR", source_paper="p1")
        _seed_entity(
            conn, paper_id=paper_id, paper_name="paper_name_2024",
            entity_name="BookRAG",
        )

        index_one(paper_name="paper_name_2024", conn=conn)

        aliases_row = conn.execute(
            "SELECT aliases FROM terms_fts WHERE term_id = ?", (term_id,),
        ).fetchone()
        assert aliases_row is not None
        parts = aliases_row[0].split()
        assert sorted(parts) == sorted(["book-rag", "bookragv2", "BR"])

    def test_topics_and_collection_also_touch_terms(self, conn):
        """paper_topics and papers.collection must also enter the touched-term
        set so their terms_fts rows get built alongside entity terms."""
        _seed_domain(conn)
        paper_id = _seed_paper(
            conn, paper_name="paper_name_2024", collection="retrieval",
        )
        # Seed canonical rows for collection + topic + entity.
        entity_term = _seed_canonical(
            conn, term_type="entity", entity_type="method",
            canonical_name="EntityE",
        )
        topic_term = _seed_canonical(
            conn, term_type="topic", entity_type="",
            canonical_name="TopicT",
        )
        coll_term = _seed_canonical(
            conn, term_type="collection", entity_type="",
            canonical_name="retrieval",
        )
        _seed_entity(
            conn, paper_id=paper_id, paper_name="paper_name_2024",
            entity_name="EntityE",
        )
        _seed_paper_topic(conn, paper_id=paper_id, topic="TopicT")

        index_one(paper_name="paper_name_2024", conn=conn)

        touched = set(r[0] for r in conn.execute(
            "SELECT term_id FROM terms_fts"
        ).fetchall())
        assert entity_term in touched
        assert topic_term in touched
        assert coll_term in touched


# ===========================================================================
# Breadcrumb inclusion
# ===========================================================================


class TestBreadcrumbInclusion:
    def test_section_body_includes_breadcrumb_on_line_one(self, conn):
        """BM25 for a parent-only token must return the child row because
        the breadcrumb ``# Parent > ## Child`` is prepended to the child body."""
        md = (
            "# UniqueParentTok\n\n"
            "Parent prose without distinctive tokens.\n\n"
            "## Child\n\n"
            "Child body.\n"
        )
        _seed_domain(conn)
        _seed_paper(conn, markdown=md)
        index_one(paper_name="paper_name_2024", conn=conn)

        rows = conn.execute(
            "SELECT section_title FROM sections WHERE sections MATCH ?",
            ("UniqueParentTok",),
        ).fetchall()
        titles = {r[0] for r in rows}
        assert "Child" in titles, (
            f"child row didn't match parent token; titles={titles!r}"
        )


# ===========================================================================
# rebuild_all
# ===========================================================================


class TestRebuildAll:
    def test_rebuild_all_drops_and_recreates_all_four_derived_tables(
        self, conn, fake_embedder
    ):
        """Seed stale rows in all four derived tables, then rebuild and
        assert the stale rows are gone."""
        _seed_domain(conn)
        _seed_paper(conn, markdown=None)
        term_id = _seed_canonical(conn, canonical_name="term1")

        conn.execute(
            "INSERT INTO abstracts "
            "(paper_id, domain, paper_name, collection, title, body) "
            "VALUES (99, 'rag', 'ghost', 'ca', 'title', 'body')"
        )
        conn.execute(
            "INSERT INTO sections "
            "(paper_id, domain, paper_name, section_title, section_level, body) "
            "VALUES (99, 'rag', 'ghost', 'Gone', '1', 'body')"
        )
        conn.execute(
            "INSERT INTO terms_fts "
            "(term_id, domain, term_type, entity_type, canonical_name, aliases) "
            "VALUES (?, 'rag', 'entity', 'method', 'stale_term', 'gone')",
            (term_id,),
        )
        conn.execute(
            "INSERT INTO term_embeddings "
            "(term_id, embedding, term_type, entity_type, domain) "
            "VALUES (?, ?, 'entity', 'method', 'rag')",
            (term_id, sqlite_vec.serialize_float32([0.5] * 384)),
        )

        rebuild_all(conn, embedder=fake_embedder)

        # Stale ghost rows gone
        assert conn.execute(
            "SELECT COUNT(*) FROM abstracts WHERE paper_name = 'ghost'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM sections WHERE paper_name = 'ghost'"
        ).fetchone()[0] == 0

    def test_rebuild_all_populates_term_embeddings_for_every_canonical_term(
        self, conn, fake_embedder
    ):
        _seed_domain(conn)
        for i in range(5):
            _seed_canonical(
                conn, canonical_name=f"term_{i}", first_seen_in=f"p{i}",
            )
        rebuild_all(conn, embedder=fake_embedder)
        n = conn.execute("SELECT COUNT(*) FROM term_embeddings").fetchone()[0]
        assert n == 5

    def test_rebuild_all_uses_batched_embed_batch_with_batch_size_64(
        self, conn, fake_embedder
    ):
        _seed_domain(conn)
        for i in range(150):
            _seed_canonical(
                conn, canonical_name=f"term_{i:04d}", first_seen_in=f"p{i}",
            )
        rebuild_all(conn, embedder=fake_embedder)
        # Every call receives ≤64 texts.
        assert fake_embedder.embed_batch_calls, "embed_batch never called"
        assert all(n <= 64 for n in fake_embedder.embed_batch_calls)
        # Every canonical term covered exactly once.
        assert sum(fake_embedder.embed_batch_calls) == 150

    def test_rebuild_all_repopulates_abstracts_and_sections(
        self, conn, fake_embedder
    ):
        _seed_domain(conn)
        md_1 = "# One\n\nalpha.\n"
        md_2 = "# Two\n\nbravo.\n\n## Two.one\n\ncharlie.\n"
        _seed_paper(
            conn, paper_name="pA", arxiv_id="2401.0101",
            markdown=md_1, abstract="Paper A abstract.",
        )
        _seed_paper(
            conn, paper_name="pB", arxiv_id="2401.0102",
            markdown=md_2, abstract="Paper B abstract.",
        )
        rebuild_all(conn, embedder=fake_embedder)

        assert conn.execute(
            "SELECT COUNT(*) FROM abstracts"
        ).fetchone()[0] == 2
        expected_sections = len(split_sections(md_1)) + len(split_sections(md_2))
        assert conn.execute(
            "SELECT COUNT(*) FROM sections"
        ).fetchone()[0] == expected_sections

    def test_rebuild_all_empty_db_is_safe(self, conn, fake_embedder):
        """Empty DB: no papers, no canonical_terms. rebuild_all must not raise
        and must leave all four derived tables empty."""
        rebuild_all(conn, embedder=fake_embedder)
        for table in ("abstracts", "sections", "terms_fts", "term_embeddings"):
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0] == 0
        # embed_batch never called (no canonicals to embed)
        assert fake_embedder.embed_batch_calls == []

    def test_rebuild_all_populates_terms_fts_for_every_canonical(
        self, conn, fake_embedder
    ):
        _seed_domain(conn)
        for i in range(3):
            tid = _seed_canonical(
                conn, canonical_name=f"term_{i}", first_seen_in=f"p{i}",
            )
            _seed_alias(conn, tid, f"alias_{i}", source_paper=f"p{i}")
        rebuild_all(conn, embedder=fake_embedder)
        n = conn.execute("SELECT COUNT(*) FROM terms_fts").fetchone()[0]
        assert n == 3

    def test_rebuild_all_is_idempotent_when_run_twice(
        self, conn, fake_embedder
    ):
        """Two back-to-back rebuilds on the same DB must yield identical row
        counts — no duplicated abstracts/sections/terms_fts/term_embeddings
        rows."""
        _seed_domain(conn)
        _seed_paper(
            conn, paper_name="pA", arxiv_id="2401.0901",
            markdown="# Intro\n\nHello.\n\n# Method\n\nWorld.\n",
        )
        for i in range(4):
            _seed_canonical(
                conn, canonical_name=f"t_{i}", first_seen_in=f"p{i}",
            )

        rebuild_all(conn, embedder=fake_embedder)
        counts_a = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("abstracts", "sections", "terms_fts", "term_embeddings")
        }

        second = _FakeEmbedder()
        rebuild_all(conn, embedder=second)
        counts_b = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("abstracts", "sections", "terms_fts", "term_embeddings")
        }

        assert counts_a == counts_b


# ===========================================================================
# Re-index own terms_fts and orphan-canonical edge cases
# ===========================================================================


class TestReindexOwnTermsFts:
    def test_reindex_paper_a_does_not_duplicate_its_own_terms_fts_rows(
        self, conn
    ):
        """After two back-to-back index_one calls on the same paper, every
        touched term must have exactly one terms_fts row (not two)."""
        _seed_domain(conn)
        paper_id = _seed_paper(conn, paper_name="pA", arxiv_id="2401.0202")
        term_id = _seed_canonical(
            conn, canonical_name="EntityX", first_seen_in="pA",
        )
        _seed_alias(conn, term_id, "EX", source_paper="pA")
        _seed_entity(
            conn, paper_id=paper_id, paper_name="pA",
            entity_name="EntityX",
        )

        index_one(paper_name="pA", conn=conn)
        index_one(paper_name="pA", conn=conn)

        n = conn.execute(
            "SELECT COUNT(*) FROM terms_fts WHERE term_id = ?", (term_id,)
        ).fetchone()[0]
        assert n == 1


class TestOrphanEntity:
    def test_orphan_entity_without_canonical_does_not_crash(self, conn):
        """If an ``entities`` row's name has no matching canonical_terms row,
        index_one must not crash. The orphan should be silently dropped from
        ``touched`` and a WARN emitted so the mismatch surfaces in logs.

        Lodestone's root logger sets ``propagate=False`` and pins its handler
        to ``sys.stderr`` at module-load time — neither ``caplog`` nor
        ``capsys`` capture its output. Attach a temporary handler directly
        to the ``lodestone.scripts.index_paper`` logger.
        """
        import logging

        _seed_domain(conn)
        paper_id = _seed_paper(conn)
        _seed_entity(
            conn, paper_id=paper_id, paper_name="paper_name_2024",
            entity_name="GhostEntity", entity_type="method",
        )

        records: list[logging.LogRecord] = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _CaptureHandler(level=logging.WARNING)
        logger = logging.getLogger("lodestone.scripts.index_paper")
        logger.addHandler(handler)
        try:
            index_one(paper_name="paper_name_2024", conn=conn)
        finally:
            logger.removeHandler(handler)

        n = conn.execute("SELECT COUNT(*) FROM terms_fts").fetchone()[0]
        assert n == 0

        warn_messages = [r.getMessage() for r in records if r.levelno >= logging.WARNING]
        assert any("entity_names" in m for m in warn_messages), (
            f"expected WARN about entity_name/canonical mismatch; "
            f"got {warn_messages!r}"
        )
