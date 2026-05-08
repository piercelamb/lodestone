"""Unit tests for _system/scripts/search.py.

Covers all five search modes (BM25 / taxonomy / browse / ToC / content
extraction), the argparse routing, lazy-import discipline, and the dual
JSON / human output formatters.

The seeded DB fixture builds a small, self-consistent corpus — two papers
with matching ``sections`` / ``terms_fts`` rows, a canonical ``RAPTOR``
term with aliases + embedding, and a figure BLOB keyed on both
``figure_number`` and ``display_number``.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest
import sqlite_vec

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.scripts import search as search_mod
from _system.utils.sections import split_sections


# ---------------------------------------------------------------------------
# Small valid PNG blob — 1x1 pixel, used for every figure in the fixtures
# so we only need one BLOB constant.
# ---------------------------------------------------------------------------

_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "89000000134944415478da63f8cfc0c000c40000000300015d6b2d540000"
    "0000000049454e44ae426082"
)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


_PAPER1_MD = (
    "# Abstract\n\n"
    "We introduce BookRAG, a hierarchical indexing technique for retrieval.\n\n"
    "# Introduction\n\n"
    "Intro content mentioning RAPTOR.\n\n"
    "# Method\n\n"
    "Method content discussing the BookRAG approach.\n\n"
    "## Setup\n\n"
    "Setup inside Method.\n\n"
    "# Experiments\n\n"
    "Experiments content.\n\n"
    "## Setup\n\n"
    "Setup inside Experiments.\n"
)


def _seed_domain(conn: sqlite3.Connection, name: str = "rag") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
        (name, f"{name} domain"),
    )


def _insert_paper(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str,
    paper_name: str,
    title: str,
    abstract: str,
    markdown: str | None,
    domain: str,
    collection: str | None,
    needs_review: int,
    ingested_at: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, html_source, ingested_at, status, markdown,
            domain, collection, needs_review, section_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id,
            paper_name,
            title,
            '["A. Author"]',
            ingested_at[:10],
            abstract,
            f"https://arxiv.org/pdf/{arxiv_id}",
            "arxiv",
            ingested_at,
            "indexed",
            markdown,
            domain,
            collection,
            needs_review,
            0,
        ),
    )
    paper_id = cur.lastrowid
    # Mirror production: a classified paper carries a primary
    # `collections` row pointing at its denormalized collection.
    if collection is not None:
        conn.execute(
            "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
            "VALUES (?, ?, NULL)",
            (domain, collection),
        )
        conn.execute(
            "INSERT OR IGNORE INTO collections "
            " (target_kind, target_id, domain, collection, is_primary) "
            " VALUES ('paper', ?, ?, ?, 1)",
            (paper_id, domain, collection),
        )
    return paper_id


def _add_secondary_collection(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    domain: str,
    collection: str,
) -> None:
    """Attach a SECONDARY `collections` row (is_primary=0) — keeps
    `papers.collection` (the primary) untouched."""
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES (?, ?, NULL)",
        (domain, collection),
    )
    conn.execute(
        "INSERT INTO collections "
        " (target_kind, target_id, domain, collection, is_primary) "
        " VALUES ('paper', ?, ?, ?, 0)",
        (paper_id, domain, collection),
    )


def _insert_sections_for_md(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    domain: str,
    paper_name: str,
    markdown: str,
) -> int:
    rows = [
        (paper_id, domain, paper_name, chunk.title, str(chunk.level), chunk.body)
        for chunk in split_sections(markdown)
    ]
    conn.executemany(
        """
        INSERT INTO sections
            (paper_id, domain, paper_name, section_title, section_level, body)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.execute(
        "UPDATE papers SET section_count = ? WHERE paper_name = ?",
        (len(rows), paper_name),
    )
    return len(rows)


def _insert_canonical(
    conn: sqlite3.Connection,
    *,
    domain: str,
    term_type: str,
    entity_type: str,
    canonical_name: str,
    first_seen_in: str,
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


def _insert_alias(
    conn: sqlite3.Connection,
    term_id: int,
    alias: str,
    source_paper: str,
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


def _insert_term_embedding(
    conn: sqlite3.Connection,
    *,
    term_id: int,
    term_type: str,
    entity_type: str,
    domain: str,
    vec: list[float],
) -> None:
    conn.execute(
        """
        INSERT INTO term_embeddings
            (term_id, embedding, term_type, entity_type, domain)
        VALUES (?, ?, ?, ?, ?)
        """,
        (term_id, sqlite_vec.serialize_float32(vec), term_type, entity_type, domain),
    )


def _insert_terms_fts(
    conn: sqlite3.Connection,
    *,
    term_id: int,
    domain: str,
    term_type: str,
    entity_type: str,
    canonical_name: str,
    aliases: str,
) -> None:
    conn.execute(
        """
        INSERT INTO terms_fts
            (term_id, domain, term_type, entity_type, canonical_name, aliases)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (term_id, domain, term_type, entity_type, canonical_name, aliases),
    )


def _insert_entity(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    domain: str,
    paper_name: str,
    entity_name: str,
    entity_type: str,
    source_breadcrumb: str,
    description: str = "",
) -> None:
    """Compatibility shim: seed an entity-typed canonical (if needed)
    and ONE synonym row pointing at it. ``paper_id``,
    ``source_breadcrumb``, and ``description`` are unused — the
    appearance log was reverted to a synonym index. The seeded synonym
    is ``f"{entity_name.lower()}_alt"`` so it differs from the canonical
    and respects the synonym-index invariant; tests that need a real
    synonym for BM25 enrichment / preview have a row to find."""
    del paper_id, source_breadcrumb, description
    conn.execute(
        """
        INSERT OR IGNORE INTO canonical_terms
            (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES (?, 'entity', ?, ?, ?)
        """,
        (domain, entity_type, entity_name, paper_name),
    )
    term_id = conn.execute(
        """
        SELECT id FROM canonical_terms
         WHERE domain = ? AND term_type = 'entity' AND canonical_name = ?
        """,
        (domain, entity_name),
    ).fetchone()[0]
    conn.execute(
        """
        INSERT OR IGNORE INTO term_aliases
            (term_id, alias, source_paper, match_tier)
        VALUES (?, ?, ?, 2)
        """,
        (term_id, f"{entity_name.lower()}_alt", paper_name),
    )


def _insert_paper_topic(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    domain: str,
    topic: str,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO topics (target_kind, target_id, domain, topic) VALUES ('paper', ?, ?, ?)",
        (paper_id, domain, topic),
    )


def _insert_figure(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    figure_number: int,
    display_number: str | None,
    figure_id: str | None,
    caption: str,
    section_context: str,
    image: bytes,
    mime_type: str = "image/png",
) -> None:
    conn.execute(
        """
        INSERT INTO figures
            (paper_id, figure_number, display_number, figure_id, caption,
             section_context, image, mime_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            figure_number,
            display_number,
            figure_id,
            caption,
            section_context,
            image,
            mime_type,
        ),
    )


# ---------------------------------------------------------------------------
# Main seed fixture — stands up the tiny corpus used by almost every test.
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    _seed_domain(conn, "rag")
    _seed_domain(conn, "other")

    # Paper 1: bookrag_2024 (rag / hierarchical indexing)
    p1_id = _insert_paper(
        conn,
        arxiv_id="2401.00001",
        paper_name="bookrag_2024",
        title="BookRAG: Hierarchical Indexing for Retrieval",
        abstract=(
            "We introduce BookRAG, a hierarchical indexing approach for "
            "document retrieval."
        ),
        markdown=_PAPER1_MD,
        domain="rag",
        collection="hierarchical indexing",
        needs_review=0,
        ingested_at="2024-01-01T00:00:00+00:00",
    )
    _insert_sections_for_md(
        conn,
        paper_id=p1_id,
        domain="rag",
        paper_name="bookrag_2024",
        markdown=_PAPER1_MD,
    )

    # Paper 2: stale_2024 with needs_review=1. needs_review is set when
    # classify_paper proposes a brand-new domain name and asks a human
    # to confirm it; the paper still has both domain and collection
    # (system invariant for classified+ rows).
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES ('other', 'misc', NULL)"
    )
    p2_id = _insert_paper(
        conn,
        arxiv_id="2402.00002",
        paper_name="stale_2024",
        title="Stale Paper",
        abstract="A paper that needs review.",
        markdown=None,
        domain="other",
        collection="misc",
        needs_review=1,
        ingested_at="2024-02-01T00:00:00+00:00",
    )
    # Canonical term: RAPTOR (entity/method, rag domain)
    raptor_id = _insert_canonical(
        conn,
        domain="rag",
        term_type="entity",
        entity_type="method",
        canonical_name="RAPTOR",
        first_seen_in="bookrag_2024",
    )
    _insert_alias(conn, raptor_id, "raptor", source_paper="raptor_2024")
    _insert_alias(conn, raptor_id, "RAPTOR method", source_paper="bookrag_2024")

    raptor_vec = [0.0] * 384
    raptor_vec[0] = 1.0
    _insert_term_embedding(
        conn,
        term_id=raptor_id,
        term_type="entity",
        entity_type="method",
        domain="rag",
        vec=raptor_vec,
    )
    _insert_terms_fts(
        conn,
        term_id=raptor_id,
        domain="rag",
        term_type="entity",
        entity_type="method",
        canonical_name="RAPTOR",
        aliases="raptor RAPTOR method",
    )

    # Canonical term: BookRAG (entity/method, rag)
    bookrag_id = _insert_canonical(
        conn,
        domain="rag",
        term_type="entity",
        entity_type="method",
        canonical_name="BookRAG",
        first_seen_in="bookrag_2024",
    )
    _insert_alias(conn, bookrag_id, "book rag", source_paper="bookrag_2024")
    book_vec = [0.0] * 384
    book_vec[1] = 1.0
    _insert_term_embedding(
        conn,
        term_id=bookrag_id,
        term_type="entity",
        entity_type="method",
        domain="rag",
        vec=book_vec,
    )
    _insert_terms_fts(
        conn,
        term_id=bookrag_id,
        domain="rag",
        term_type="entity",
        entity_type="method",
        canonical_name="BookRAG",
        aliases="book rag",
    )

    # Canonical term: "entity resolution" as a topic
    topic_id = _insert_canonical(
        conn,
        domain="rag",
        term_type="topic",
        entity_type="",
        canonical_name="entity resolution",
        first_seen_in="bookrag_2024",
    )
    topic_vec = [0.0] * 384
    topic_vec[2] = 1.0
    _insert_term_embedding(
        conn,
        term_id=topic_id,
        term_type="topic",
        entity_type="",
        domain="rag",
        vec=topic_vec,
    )
    _insert_terms_fts(
        conn,
        term_id=topic_id,
        domain="rag",
        term_type="topic",
        entity_type="",
        canonical_name="entity resolution",
        aliases="",
    )

    # Canonical term: "hierarchical indexing" as a collection
    coll_id = _insert_canonical(
        conn,
        domain="rag",
        term_type="collection",
        entity_type="",
        canonical_name="hierarchical indexing",
        first_seen_in="bookrag_2024",
    )
    coll_vec = [0.0] * 384
    coll_vec[3] = 1.0
    _insert_term_embedding(
        conn,
        term_id=coll_id,
        term_type="collection",
        entity_type="",
        domain="rag",
        vec=coll_vec,
    )
    _insert_terms_fts(
        conn,
        term_id=coll_id,
        domain="rag",
        term_type="collection",
        entity_type="",
        canonical_name="hierarchical indexing",
        aliases="",
    )

    # Entities for bookrag_2024
    _insert_entity(
        conn,
        paper_id=p1_id,
        domain="rag",
        paper_name="bookrag_2024",
        entity_name="BookRAG",
        entity_type="method",
        source_breadcrumb="# Method",
        description="A novel RAG approach.",
    )
    _insert_entity(
        conn,
        paper_id=p1_id,
        domain="rag",
        paper_name="bookrag_2024",
        entity_name="RAPTOR",
        entity_type="method",
        source_breadcrumb="# Introduction",
        description="A tree-based retrieval method.",
    )
    _insert_entity(
        conn,
        paper_id=p1_id,
        domain="rag",
        paper_name="bookrag_2024",
        entity_name="MMLongBench",
        entity_type="dataset",
        source_breadcrumb="# Experiments",
        description="Benchmark dataset.",
    )

    # Paper topic
    _insert_paper_topic(conn, paper_id=p1_id, domain="rag", topic="entity resolution")

    # Figures (used by BLOB extraction tests)
    _insert_figure(
        conn,
        paper_id=p1_id,
        figure_number=3,
        display_number="Figure 3",
        figure_id="F3",
        caption="An illustration of BookRAG.",
        section_context="Method",
        image=_PNG_1x1,
    )
    _insert_figure(
        conn,
        paper_id=p1_id,
        figure_number=4,
        display_number="Figure 3a",
        figure_id="F3a",
        caption="Subfigure a.",
        section_context="Method",
        image=_PNG_1x1,
    )
    return conn


# ===========================================================================
# Lazy-import / CLI startup
# ===========================================================================


def _search_script_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "search.py"
    )


@pytest.mark.slow
def test_help_subprocess_under_300ms():
    """``search.py --help`` subprocess must finish in < 300 ms.

    This enforces the no-ML-at-module-scope rule end-to-end: loading
    sentence_transformers / torch at import time would blow past 300 ms on
    most hardware. Uses ``python -m _system.scripts.search`` so imports
    mirror the installed-script entry point.
    """
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "_system.scripts.search", "--help"],
        capture_output=True,
        timeout=5,
    )
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, result.stderr
    assert elapsed < 0.300, f"--help took {elapsed:.3f}s, expected < 0.300s"


def test_help_does_not_import_ml_libs():
    """After ``search.py --help`` in a subprocess, sys.modules must not
    include sentence_transformers / gliner2 / gliner / torch."""
    code = (
        "import sys, runpy;\n"
        "try:\n"
        "    runpy.run_module('_system.scripts.search', run_name='__main__', "
        "alter_sys=True)\n"
        "except SystemExit:\n"
        "    pass\n"
        "banned = ('sentence_transformers','gliner2','gliner','torch')\n"
        "present = sorted(m for m in sys.modules if any(m == b or m.startswith(b+'.') for b in banned))\n"
        "print(repr(present))\n"
    )
    # Simulate `--help` by passing argv:
    result = subprocess.run(
        [sys.executable, "-c", code, "--help"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    # `argparse --help` calls SystemExit(0); we catch it above. Stdout is
    # the argparse help text followed by our print; the last line is the
    # banned-module list.
    output = result.stdout.strip().splitlines()
    banned_line = output[-1]
    assert banned_line == "[]", (
        f"search.py --help imported ML libs: {banned_line!r}\n"
        f"stderr={result.stderr}"
    )


# ===========================================================================
# Mode 1 — BM25
# ===========================================================================


class TestModeBM25:
    def test_abstract_text_surfaces_via_sections(self, seeded_db):
        """The abstract is now indexed as the # Abstract chunk inside
        ``sections``, so a BM25 query over abstract text still resolves
        the paper — just under the Abstract section title."""
        r = search_mod.mode_bm25(
            seeded_db,
            query="hierarchical indexing",
            filters={},
            limit=10,
        )
        assert r["mode"] == "sections"
        assert r["query"] == "hierarchical indexing"
        assert isinstance(r["results"], list)
        hits = [h for h in r["results"] if h["paper_name"] == "bookrag_2024"]
        assert hits, r["results"]
        group = hits[0]
        # Enrichment — topics, entities preview, hit_count
        assert "hit_count" in group
        assert "entities_preview" in group
        assert "topics" in group
        # Abstract chunk was hit (the seeded markdown opens with # Abstract).
        section_titles = [s["section_title"] for s in group["sections"]]
        assert "Abstract" in section_titles, section_titles

    def test_sections_groups_hits_by_paper(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db,
            query="BookRAG",
            filters={},
            limit=10,
        )
        assert r["mode"] == "sections"
        assert isinstance(r["results"], list)
        # Sections are grouped by paper_name; the group must preserve hit_count.
        hits = [h for h in r["results"] if h.get("paper_name") == "bookrag_2024"]
        assert hits, r
        group = hits[0]
        assert "hit_count" in group
        assert group["hit_count"] >= 1
        assert "sections" in group
        assert isinstance(group["sections"], list)
        assert len(group["sections"]) == group["hit_count"]

    def test_domain_filter_narrows_results(self, seeded_db):
        # Seed a sections row in the 'other' domain whose body mentions BookRAG.
        p2_id = seeded_db.execute(
            "SELECT id FROM papers WHERE paper_name = ?", ("stale_2024",)
        ).fetchone()[0]
        seeded_db.execute(
            """
            INSERT INTO sections
                (paper_id, domain, paper_name, section_title, section_level, body)
            VALUES (?, 'other', 'stale_2024', 'Mentions', '1',
                    'BookRAG also mentioned here')
            """,
            (p2_id,),
        )
        r_all = search_mod.mode_bm25(
            seeded_db, query="BookRAG", filters={}, limit=10
        )
        r_rag = search_mod.mode_bm25(
            seeded_db,
            query="BookRAG",
            filters={"domain": "rag"},
            limit=10,
        )
        assert any(h["paper_name"] == "stale_2024" for h in r_all["results"])
        assert not any(
            h["paper_name"] == "stale_2024" for h in r_rag["results"]
        )

    def test_enrichment_includes_entities_and_figures(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db,
            query="BookRAG",
            filters={},
            limit=10,
        )
        hit = next(h for h in r["results"] if h["paper_name"] == "bookrag_2024")
        assert hit["entities_preview"], "expected entity preview"
        assert "figures" in hit
        assert isinstance(hit["figures"], dict)
        assert hit["figures"]["count"] >= 1

    def test_hyphenated_query_does_not_crash(self, seeded_db):
        """Hyphens in user queries used to be parsed as the FTS5 NOT
        operator, causing a hard crash. The sanitizer must wrap each
        token in double quotes so the hyphen becomes phrase content."""
        # Seed a section body containing 'tree-sitter' so we have a real
        # hit to confirm the phrase semantics work too.
        p1_id = seeded_db.execute(
            "SELECT id FROM papers WHERE paper_name = ?", ("bookrag_2024",)
        ).fetchone()[0]
        seeded_db.execute(
            """
            INSERT INTO sections
                (paper_id, domain, paper_name, section_title, section_level, body)
            VALUES (?, 'rag', 'bookrag_2024', 'Tools', '1',
                    'We use tree-sitter for source parsing.')
            """,
            (p1_id,),
        )
        r = search_mod.mode_bm25(
            seeded_db,
            query="tree-sitter",
            filters={},
            limit=10,
        )
        assert r["mode"] == "sections"
        # Either the seed row is found (good) or we get an empty result
        # set (also acceptable) — what we must NOT get is an exception.
        assert isinstance(r.get("results", []), list)
        # The seeded body should be matched by the phrase "tree" + "sitter".
        assert any(
            h.get("paper_name") == "bookrag_2024" for h in r["results"]
        ), r

    def test_punctuation_only_query_returns_soft_failure(self, seeded_db):
        """A query with no word characters at all (``---``) hits the
        sanitizer's empty-tokens guard — soft-fail payload, not crash."""
        r = search_mod.mode_bm25(
            seeded_db, query="---", filters={}, limit=10
        )
        assert r["mode"] == "sections"
        assert r["status"] == "empty_query"
        assert "hint" in r

    def test_special_chars_in_query_do_not_crash(self, seeded_db):
        """Slashes, parens, and colons must also reduce to phrase
        content rather than FTS5 syntax. ``method:foo`` is now parsed as
        a qualifier (unknown one → malformed_query soft-fail) but still
        returns a structured payload, never an exception."""
        for q in ["BAAI/bge-small", "O(1)", "method:foo", "(parens)"]:
            r = search_mod.mode_bm25(
                seeded_db, query=q, filters={}, limit=10
            )
            assert r["mode"] == "sections", q
            # No KeyError, no OperationalError — that's the contract.
            assert isinstance(r.get("results", []), list)


# ===========================================================================
# Mode 1 — BM25 pagination
# ===========================================================================


def _seed_paginated_sections(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    paper_name: str,
    domain: str,
    count: int,
    body_token: str,
    title_prefix: str = "Page Section",
) -> None:
    """Insert ``count`` extra sections whose body all carry ``body_token``
    so a BM25 query for ``body_token`` matches every one of them. Lets
    pagination tests build a deterministic-sized result set without
    fighting FTS5 ranking edge cases."""
    rows = [
        (
            paper_id,
            domain,
            paper_name,
            f"{title_prefix} {i:02d}",
            "1",
            f"{body_token} payload {i} {body_token}",
        )
        for i in range(count)
    ]
    conn.executemany(
        """
        INSERT INTO sections
            (paper_id, domain, paper_name, section_title, section_level, body)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


class TestModeBM25Pagination:
    """Pagination contract for ``mode_bm25``: offset slicing,
    total_hits / has_more echo, soft-fail on negative offset, and the
    skip-COUNT optimization on the first-page-undersize case."""

    def _seed_15_matches(self, seeded_db) -> None:
        p1_id = seeded_db.execute(
            "SELECT id FROM papers WHERE paper_name = ?", ("bookrag_2024",)
        ).fetchone()[0]
        _seed_paginated_sections(
            seeded_db,
            paper_id=p1_id,
            paper_name="bookrag_2024",
            domain="rag",
            count=15,
            body_token="paginatortoken",
        )

    def test_total_hits_matches_full_result_count(self, seeded_db):
        self._seed_15_matches(seeded_db)
        r = search_mod.mode_bm25(
            seeded_db, query="paginatortoken", filters={}, limit=100
        )
        assert r["total_hits"] == 15
        assert r["offset"] == 0
        assert r["limit"] == 100
        assert r["has_more"] is False
        # All 15 hits live under one paper (bookrag_2024); the result
        # rows are 15 sections, not 1 paper.
        all_rows = sum(int(g["hit_count"]) for g in r["results"])
        assert all_rows == 15

    def test_offset_slices_correctly(self, seeded_db):
        self._seed_15_matches(seeded_db)
        page1 = search_mod.mode_bm25(
            seeded_db, query="paginatortoken", filters={},
            limit=5, offset=0,
        )
        page2 = search_mod.mode_bm25(
            seeded_db, query="paginatortoken", filters={},
            limit=5, offset=5,
        )

        def _titles(payload):
            return {
                s["section_title"]
                for g in payload["results"]
                for s in g.get("sections", [])
            }

        t1, t2 = _titles(page1), _titles(page2)
        assert t1, page1
        assert t2, page2
        assert t1.isdisjoint(t2), (t1, t2)

    def test_has_more_true_when_more_pages(self, seeded_db):
        self._seed_15_matches(seeded_db)
        r = search_mod.mode_bm25(
            seeded_db, query="paginatortoken", filters={},
            limit=5, offset=0,
        )
        assert r["has_more"] is True
        assert r["total_hits"] == 15

    def test_has_more_false_at_end(self, seeded_db):
        self._seed_15_matches(seeded_db)
        r = search_mod.mode_bm25(
            seeded_db, query="paginatortoken", filters={},
            limit=5, offset=10,
        )
        assert r["has_more"] is False
        rows = sum(int(g["hit_count"]) for g in r["results"])
        assert rows == 5

    def test_offset_beyond_total_returns_empty(self, seeded_db):
        self._seed_15_matches(seeded_db)
        r = search_mod.mode_bm25(
            seeded_db, query="paginatortoken", filters={},
            limit=5, offset=100,
        )
        # Recoverable: empty page, total still reported, has_more false.
        assert r["results"] == []
        assert r["total_hits"] == 15
        assert r["has_more"] is False
        # Not a soft-fail — Claude can recover by paging back.
        assert "status" not in r

    def test_negative_offset_soft_fails(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db, query="paginatortoken", filters={},
            limit=5, offset=-1,
        )
        assert r["status"] == "invalid_pagination"
        assert ">= 0" in r["error"]
        assert r["offset"] == -1
        assert r["limit"] == 5
        assert "total_hits" not in r
        assert r["status"] in search_mod._SOFT_FAILURE_STATUSES

    def test_offset_with_filters(self, seeded_db):
        # Seed 15 hits in 'rag', then 3 hits in a new domain. Domain
        # filter must narrow BOTH the page AND the total.
        self._seed_15_matches(seeded_db)
        seeded_db.execute(
            "INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
            ("solo", "solo domain"),
        )
        seeded_db.execute(
            "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
            "VALUES ('solo', 'misc', NULL)"
        )
        solo_id = _insert_paper(
            seeded_db,
            arxiv_id="2403.00003",
            paper_name="solo_2024",
            title="Solo",
            abstract="abstract",
            markdown=None,
            domain="solo",
            collection="misc",
            needs_review=0,
            ingested_at="2024-03-01T00:00:00+00:00",
        )
        _seed_paginated_sections(
            seeded_db,
            paper_id=solo_id,
            paper_name="solo_2024",
            domain="solo",
            count=3,
            body_token="paginatortoken",
        )

        r = search_mod.mode_bm25(
            seeded_db, query="paginatortoken", filters={"domain": "rag"},
            limit=5, offset=5,
        )
        assert r["total_hits"] == 15
        names = {g["paper_name"] for g in r["results"]}
        assert "solo_2024" not in names
        assert "bookrag_2024" in names

    def test_skip_count_optimization_when_first_page_undersize(
        self, seeded_db
    ):
        """When ``offset == 0`` and the page came back smaller than
        ``limit``, COUNT(*) is skipped — total_hits is exactly
        page_size. We verify the optimization by attaching a sqlite
        trace callback and asserting no COUNT(*) statement fires."""
        seen_sql: list[str] = []
        seeded_db.set_trace_callback(seen_sql.append)
        try:
            # 'BookRAG' hits the abstract chunk only (1 row); limit=10
            # leaves plenty of headroom — skip-COUNT applies.
            r = search_mod.mode_bm25(
                seeded_db, query="BookRAG", filters={}, limit=10, offset=0,
            )
        finally:
            seeded_db.set_trace_callback(None)
        assert r["total_hits"] == sum(
            int(g["hit_count"]) for g in r["results"]
        )
        count_stmts = [
            sql for sql in seen_sql
            if "FROM sections" in sql and "COUNT(*)" in sql
        ]
        assert count_stmts == [], count_stmts


# ===========================================================================
# GitHub-code-search-style query parser
# ===========================================================================


class TestParseGithubQuery:
    """Pure parser tests — no DB, no FTS5 execution. Verifies the
    fts_expression / qualifiers / typed-error contract."""

    def test_bare_token_defang_preserves_punctuation(self):
        p = search_mod._parse_github_query("tree-sitter")
        assert p.fts_expression == '"tree-sitter"'
        assert p.qualifiers == {}

    def test_multiple_bare_tokens_are_implicit_and(self):
        p = search_mod._parse_github_query("chain of thought")
        # Each token defanged independently. We emit explicit AND between
        # adjacent operands because FTS5's implicit-AND grammar refuses
        # `(group) prefix*` and similar combinations.
        assert p.fts_expression == '"chain" AND "of" AND "thought"'

    def test_phrase_query(self):
        p = search_mod._parse_github_query('"chain of thought"')
        assert p.fts_expression == '"chain of thought"'

    def test_escaped_quote_in_phrase(self):
        p = search_mod._parse_github_query('"name = \\"x\\""')
        # FTS5 escape doubles the internal quote
        assert p.fts_expression == '"name = ""x"""'

    def test_or_operator_passthrough(self):
        p = search_mod._parse_github_query("reasoning OR planning")
        assert p.fts_expression == '"reasoning" OR "planning"'

    def test_not_operator_passthrough(self):
        p = search_mod._parse_github_query("reasoning NOT supervised")
        assert p.fts_expression == '"reasoning" NOT "supervised"'

    def test_and_operator_passthrough(self):
        p = search_mod._parse_github_query("a AND b")
        assert p.fts_expression == '"a" AND "b"'

    def test_lowercase_or_is_a_search_term(self):
        # GitHub-style: only uppercase AND/OR/NOT are operators.
        p = search_mod._parse_github_query("a or b")
        assert p.fts_expression == '"a" AND "or" AND "b"'

    def test_parens_grouping(self):
        p = search_mod._parse_github_query("(monte carlo) tree")
        assert p.fts_expression == '( "monte" AND "carlo" ) AND "tree"'

    def test_parens_with_prefix_after(self):
        # Regression: FTS5 rejects implicit AND between a paren-group and
        # a prefix marker. Explicit AND fixes it.
        p = search_mod._parse_github_query("(monte carlo) tree*")
        assert p.fts_expression == '( "monte" AND "carlo" ) AND "tree"*'

    def test_prefix_token(self):
        p = search_mod._parse_github_query("tree*")
        assert p.fts_expression == '"tree"*'

    def test_paper_qualifier(self):
        p = search_mod._parse_github_query("paper:bookrag_2024 indexing")
        assert p.qualifiers == {"paper": "bookrag_2024"}
        assert p.fts_expression == '"indexing"'

    def test_quoted_qualifier_value(self):
        p = search_mod._parse_github_query(
            'collection:"hierarchical indexing" foo'
        )
        assert p.qualifiers == {"collection": "hierarchical indexing"}
        assert p.fts_expression == '"foo"'

    def test_multiple_qualifiers(self):
        p = search_mod._parse_github_query(
            "domain:rag kind:entity reasoning"
        )
        assert p.qualifiers == {"domain": "rag", "kind": "entity"}
        assert p.fts_expression == '"reasoning"'

    def test_surface_qualifier_validates(self):
        p = search_mod._parse_github_query("surface:sections foo")
        assert p.qualifiers == {"surface": "sections"}

    def test_kind_qualifier_validates(self):
        p = search_mod._parse_github_query("kind:topic foo")
        assert p.qualifiers == {"kind": "topic"}

    # --- error paths ---------------------------------------------------

    def test_regex_rejected(self):
        with pytest.raises(search_mod.RegexNotSupportedError):
            search_mod._parse_github_query("/foo.*/")

    def test_unclosed_quote(self):
        with pytest.raises(search_mod.UnclosedQuoteError):
            search_mod._parse_github_query('"unclosed phrase')

    def test_unmatched_open_paren(self):
        with pytest.raises(search_mod.UnmatchedParenError):
            search_mod._parse_github_query("(foo")

    def test_unmatched_close_paren(self):
        with pytest.raises(search_mod.UnmatchedParenError):
            search_mod._parse_github_query("foo)")

    def test_dangling_leading_operator(self):
        with pytest.raises(search_mod.DanglingOperatorError):
            search_mod._parse_github_query("OR foo")

    def test_dangling_trailing_operator(self):
        with pytest.raises(search_mod.DanglingOperatorError):
            search_mod._parse_github_query("foo OR")

    def test_dangling_adjacent_operators(self):
        with pytest.raises(search_mod.DanglingOperatorError):
            search_mod._parse_github_query("foo OR AND bar")

    def test_unknown_qualifier(self):
        with pytest.raises(search_mod.UnknownQualifierError):
            search_mod._parse_github_query("method:foo bar")

    def test_invalid_kind_value(self):
        with pytest.raises(search_mod.InvalidQualifierValueError):
            search_mod._parse_github_query("kind:bogus foo")

    def test_invalid_surface_value(self):
        with pytest.raises(search_mod.InvalidQualifierValueError):
            search_mod._parse_github_query("surface:bogus foo")

    def test_surface_taxonomy_value_accepted(self):
        p = search_mod._parse_github_query("surface:taxonomy reasoning")
        assert p.qualifiers == {"surface": "taxonomy"}

    def test_qualifier_repeated_with_conflicting_value(self):
        with pytest.raises(search_mod.ConflictingFilterError):
            search_mod._parse_github_query("paper:a paper:b foo")

    def test_qualifier_repeated_same_value_ok(self):
        p = search_mod._parse_github_query("paper:a paper:a foo")
        assert p.qualifiers == {"paper": "a"}

    def test_empty_query(self):
        with pytest.raises(search_mod.EmptyQueryError):
            search_mod._parse_github_query("")

    def test_whitespace_only_query(self):
        with pytest.raises(search_mod.EmptyQueryError):
            search_mod._parse_github_query("   ")

    def test_qualifier_only_query_returns_empty_fts(self):
        # qualifier-only queries parse, but their fts_expression is "" — the
        # dispatch layer is what surfaces this as an empty_query soft fail
        # for surfaces that require text.
        p = search_mod._parse_github_query("paper:foo")
        assert p.fts_expression == ""
        assert p.qualifiers == {"paper": "foo"}

    def test_punctuation_only_drops_to_empty_fts(self):
        # `---` carries no word characters → defang drops it. With no
        # qualifiers either, parse succeeds but fts_expression is "".
        p = search_mod._parse_github_query("---")
        assert p.fts_expression == ""


# ===========================================================================
# Mode 2 — Taxonomy
# ===========================================================================


class TestModeTaxonomy:
    def test_finds_canonical_with_aliases_inlined(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="RAPTOR", filters={}
        )
        assert r["mode"] == "lookup"
        assert r["query"] == "RAPTOR"
        hits = r["hits"]
        assert hits, r
        h = next(h for h in hits if h["canonical_name"] == "RAPTOR")
        assert h["kind"] == "entity"
        # aliases inlined per hit (was a follow-up call before)
        assert h["aliases"], h
        assert all(set(a.keys()) == {"alias", "source_paper"} for a in h["aliases"])
        # papers_count is omitted on entity hits — the underlying papers
        # list is alias-derived and misses tier-1 canonical-surface
        # mentions; publishing a count there would mislead.
        assert "papers_count" not in h, h
        # papers carry the same code_repo enrichment as before
        for paper in h["papers"]:
            assert set(paper.keys()) == {"paper_name", "code_repo"}

    def test_finds_via_alias(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="raptor", filters={}
        )
        names = [h["canonical_name"] for h in r["hits"]]
        assert "RAPTOR" in names

    def test_reports_no_hits_for_unknown_term(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="totally_made_up_thing", filters={}
        )
        assert r["mode"] == "lookup"
        assert r["hits"] == []
        # No status — empty hits is a normal outcome, not a soft fail.
        assert r.get("status") is None

    def test_kind_qualifier_narrows(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="kind:entity RAPTOR", filters={}
        )
        assert all(h["kind"] == "entity" for h in r["hits"])
        assert any(h["canonical_name"] == "RAPTOR" for h in r["hits"])

    def test_topic_kind(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db,
            query="kind:topic \"entity resolution\"",
            filters={},
        )
        names = [h["canonical_name"] for h in r["hits"]]
        assert "entity resolution" in names
        topic_hit = next(
            h for h in r["hits"] if h["canonical_name"] == "entity resolution"
        )
        assert topic_hit["kind"] == "topic"
        assert topic_hit["papers"]
        # topics/collections still carry papers_count — the binding
        # tables (paper_topics / papers.collection) are complete.
        assert topic_hit["papers_count"] == len(topic_hit["papers"])

    def test_collection_kind(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db,
            query="kind:collection \"hierarchical indexing\"",
            filters={},
        )
        names = [h["canonical_name"] for h in r["hits"]]
        assert "hierarchical indexing" in names
        coll_hit = next(
            h
            for h in r["hits"]
            if h["canonical_name"] == "hierarchical indexing"
        )
        assert coll_hit["kind"] == "collection"
        assert any(p["paper_name"] == "bookrag_2024" for p in coll_hit["papers"])
        assert coll_hit["papers_count"] == len(coll_hit["papers"])

    def test_domain_qualifier_filters(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="domain:rag RAPTOR", filters={},
        )
        assert all(h["domain"] == "rag" for h in r["hits"])

    def test_domain_kwarg_equivalent_to_qualifier(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="RAPTOR", filters={"domain": "rag"},
        )
        assert all(h["domain"] == "rag" for h in r["hits"])

    def test_or_operator_returns_union(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db,
            query="RAPTOR OR \"entity resolution\"",
            filters={},
        )
        names = {h["canonical_name"] for h in r["hits"]}
        # FTS5 ranks union; both should appear when both exist.
        assert "RAPTOR" in names or "entity resolution" in names
        # At minimum we got at least one mixed-kind hit from a single query.
        assert len(r["hits"]) >= 1

    def test_empty_query_soft_fail(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="", filters={},
        )
        assert r["mode"] == "lookup"
        assert r["status"] == "empty_query"

    def test_qualifier_only_query_soft_fail(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="kind:entity domain:rag", filters={},
        )
        assert r["status"] == "empty_query"

    def test_unsupported_qualifier_soft_fail(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="paper:bookrag_2024 RAPTOR", filters={},
        )
        assert r["status"] == "malformed_query"
        assert "paper" in r["error"]

    def test_invalid_kind_soft_fail(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="kind:bogus RAPTOR", filters={},
        )
        assert r["status"] == "malformed_query"

    def test_limit_caps_hits(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="RAPTOR OR \"entity resolution\"",
            filters={}, limit=1,
        )
        assert len(r["hits"]) <= 1

    def test_no_embedder_loaded(self, seeded_db, monkeypatch):
        """Lookup must be FTS-only — no torch / sentence_transformers /
        sqlite_vec import. If the path ever regresses to load Embedder,
        this stub raises and the test fails loudly."""
        import _system.resolution.embeddings as emb_mod

        def _boom(*a, **kw):
            raise AssertionError(
                "Embedder must NOT be loaded by mode_taxonomy_lookup"
            )

        monkeypatch.setattr(emb_mod, "Embedder", _boom)
        # Both a hit case and a miss case — neither should construct Embedder.
        search_mod.mode_taxonomy_lookup(seeded_db, query="RAPTOR", filters={})
        search_mod.mode_taxonomy_lookup(
            seeded_db, query="zzznosuchterm", filters={}
        )


class TestModeTaxonomyPagination:
    """Pagination contract for ``mode_taxonomy_lookup``: total_hits +
    has_more, offset slicing, ``kind:`` qualifier honored across pages."""

    def _seed_n_entity_canonicals(
        self, seeded_db, count: int, prefix: str = "PagingEntity"
    ) -> None:
        """Seed ``count`` extra entity canonicals all containing
        ``"PagingEntity"`` so a single FTS MATCH hits all of them."""
        for i in range(count):
            tid = _insert_canonical(
                seeded_db,
                domain="rag",
                term_type="entity",
                entity_type="method",
                canonical_name=f"{prefix} {i:02d}",
                first_seen_in="bookrag_2024",
            )
            _insert_terms_fts(
                seeded_db,
                term_id=tid,
                domain="rag",
                term_type="entity",
                entity_type="method",
                canonical_name=f"{prefix} {i:02d}",
                aliases="",
            )

    def test_total_hits_and_paging(self, seeded_db):
        self._seed_n_entity_canonicals(seeded_db, 12)
        page1 = search_mod.mode_taxonomy_lookup(
            seeded_db, query="PagingEntity", filters={},
            limit=5, offset=0,
        )
        page2 = search_mod.mode_taxonomy_lookup(
            seeded_db, query="PagingEntity", filters={},
            limit=5, offset=5,
        )
        assert page1["total_hits"] == 12
        assert page1["has_more"] is True
        assert page2["total_hits"] == 12
        assert page2["has_more"] is True
        names1 = {h["canonical_name"] for h in page1["hits"]}
        names2 = {h["canonical_name"] for h in page2["hits"]}
        assert names1.isdisjoint(names2), (names1, names2)
        assert len(page1["hits"]) == 5
        assert len(page2["hits"]) == 5

        # Final page reaches end_of_results
        last = search_mod.mode_taxonomy_lookup(
            seeded_db, query="PagingEntity", filters={},
            limit=5, offset=10,
        )
        assert last["has_more"] is False
        assert len(last["hits"]) == 2

    def test_offset_with_kind_filter(self, seeded_db):
        # Seed 6 entity canonicals + 4 topic canonicals all sharing the
        # token "PagingMixed". With kind:entity, total_hits should be 6.
        self._seed_n_entity_canonicals(
            seeded_db, 6, prefix="PagingMixed"
        )
        for i in range(4):
            tid = _insert_canonical(
                seeded_db,
                domain="rag",
                term_type="topic",
                entity_type="",
                canonical_name=f"PagingMixed topic {i:02d}",
                first_seen_in="bookrag_2024",
            )
            _insert_terms_fts(
                seeded_db,
                term_id=tid,
                domain="rag",
                term_type="topic",
                entity_type="",
                canonical_name=f"PagingMixed topic {i:02d}",
                aliases="",
            )

        r = search_mod.mode_taxonomy_lookup(
            seeded_db,
            query="kind:entity PagingMixed",
            filters={},
            limit=3,
            offset=0,
        )
        assert r["total_hits"] == 6
        assert r["has_more"] is True
        assert all(h["kind"] == "entity" for h in r["hits"])

        r2 = search_mod.mode_taxonomy_lookup(
            seeded_db,
            query="kind:entity PagingMixed",
            filters={},
            limit=3,
            offset=3,
        )
        assert r2["total_hits"] == 6
        assert r2["has_more"] is False
        assert all(h["kind"] == "entity" for h in r2["hits"])

    def test_negative_offset_soft_fails(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="RAPTOR", filters={},
            limit=5, offset=-1,
        )
        assert r["status"] == "invalid_pagination"
        assert ">= 0" in r["error"]
        assert r["status"] in search_mod._SOFT_FAILURE_STATUSES


# ===========================================================================
# Mode 2.5 — Search (composite)
# ===========================================================================


class TestModeSearch:
    def test_taxonomy_mixes_kinds(self, seeded_db):
        """A query that resolves a topic should also surface other-kind
        canonicals when they share lexical/stemmed overlap. The seeded DB
        has 'entity resolution' (topic) and 'hierarchical indexing'
        (collection) — query "indexing" hits the collection, query
        "RAPTOR" hits the entity. Verify the kind tag is plumbed through."""
        r = search_mod.mode_search(
            seeded_db, query="indexing", filters={"domain": "rag"}, limit=5
        )
        assert r["mode"] == "search"
        kinds = {row["kind"] for row in r["taxonomy"]}
        assert "collection" in kinds, r["taxonomy"]

        r2 = search_mod.mode_search(
            seeded_db, query="RAPTOR", filters={"domain": "rag"}, limit=5
        )
        kinds2 = {row["kind"] for row in r2["taxonomy"]}
        assert "entity" in kinds2, r2["taxonomy"]

    def test_buckets_present_and_shaped(self, seeded_db):
        r = search_mod.mode_search(
            seeded_db, query="BookRAG", filters={"domain": "rag"}, limit=5
        )
        assert set(r.keys()) >= {
            "mode", "query", "domain", "taxonomy", "sections", "readmes"
        }
        assert r["query"] == "BookRAG"
        assert r["domain"] == "rag"
        for row in r["taxonomy"]:
            assert set(row.keys()) == {
                "canonical_name", "kind", "entity_type", "domain",
            }
        # Sections bucket: slim shape — paper_name + hit_count + hits[].
        if r["sections"]:
            sec = r["sections"][0]
            assert set(sec.keys()) == {"paper_name", "hit_count", "hits"}
            assert isinstance(sec["hits"], list)
            if sec["hits"]:
                assert set(sec["hits"][0].keys()) == {
                    "section_title", "breadcrumb", "snippet",
                }
        # Readmes bucket: slim shape — paper_name + hit_count + path + snippet.
        if r["readmes"]:
            rd = r["readmes"][0]
            assert set(rd.keys()) == {"paper_name", "hit_count", "path", "snippet"}

    def test_empty_query_soft_fail(self, seeded_db):
        r = search_mod.mode_search(seeded_db, query="", filters={}, limit=5)
        assert r["mode"] == "search"
        assert r["status"] == "empty_query"
        # Status is in the project soft-failure set so the MCP wrapper
        # surfaces it as isError=false (agent-recoverable).
        assert r["status"] in _system_search_soft_failures()

    def test_whitespace_query_soft_fail(self, seeded_db):
        r = search_mod.mode_search(seeded_db, query="   ", filters={}, limit=5)
        assert r["status"] == "empty_query"

    def test_domain_filter_narrows_taxonomy(self, seeded_db):
        # All seeded canonicals are in 'rag'; querying with domain='other'
        # should yield no taxonomy hits (and the KNN fallback is gated by
        # domain too, so cross-domain leakage doesn't smuggle them in).
        r = search_mod.mode_search(
            seeded_db, query="RAPTOR", filters={"domain": "other"}, limit=5
        )
        assert r["taxonomy"] == [], r["taxonomy"]

    def test_fts_only_no_knn_fallback(self, seeded_db, monkeypatch):
        """A lexically-novel query returns an empty taxonomy bucket — there
        is no KNN fallback. The Embedder must NOT be touched on this path
        (the whole point of removing Tier B was to drop the heavy ML
        import from the orientation tool)."""

        class _ExplodingEmbedder:
            def __init__(self) -> None:
                raise AssertionError(
                    "Embedder must not be constructed on the search path"
                )

        import _system.resolution.embeddings as emb_mod

        monkeypatch.setattr(emb_mod, "Embedder", _ExplodingEmbedder)

        r = search_mod.mode_search(
            seeded_db, query="zzznosuchterm", filters={"domain": "rag"}, limit=5
        )
        assert r["taxonomy"] == [], r["taxonomy"]

    def test_markdown_format_render(self, seeded_db):
        """Markdown formatter is what reaches Claude on the MCP path.
        Smoke-check top-level headers, kind-grouped taxonomy subheaders,
        and absence of the entity_type tag."""
        r = search_mod.mode_search(
            seeded_db, query="RAPTOR", filters={"domain": "rag"}, limit=5
        )
        md = search_mod.format_search_markdown(r)
        assert md.startswith("# search 'RAPTOR'")
        assert "## taxonomy" in md
        assert "## sections" in md
        assert "## readmes" in md
        # Taxonomy is grouped under per-kind subheaders.
        if any(row["kind"] == "entity" for row in r["taxonomy"]):
            assert "### entity" in md
        # entity_type tag is intentionally omitted now.
        assert "[entity:" not in md
        # No JSON braces leak into the markdown.
        assert "{" not in md and "}" not in md

    def test_markdown_format_empty_query(self):
        md = search_mod.format_search_markdown({
            "mode": "search", "status": "empty_query", "query": "",
            "hint": "search needs at least one word; pass a non-empty query.",
        })
        assert md.startswith("# search (empty query)")
        assert "non-empty query" in md

    def test_markdown_format_malformed_query(self):
        md = search_mod.format_search_markdown({
            "mode": "search", "status": "malformed_query",
            "query": '"unclosed',
            "error": "unclosed quote",
            "hint": "Supported syntax: bare words ...",
        })
        assert md.startswith("# search (malformed query)")
        assert "unclosed quote" in md


# ===========================================================================
# GitHub-style query syntax — integration through mode_bm25 / mode_search
# ===========================================================================


class TestQuerySyntaxBM25:
    """End-to-end coverage of the GitHub-style operators / qualifiers
    flowing through mode_bm25 against the real seeded FTS5 index."""

    def test_or_returns_union_of_hits(self, seeded_db):
        # Both 'BookRAG' and 'RAPTOR' appear in the seeded paper. OR should
        # find at least the union of those two single-token queries.
        r_or = search_mod.mode_bm25(
            seeded_db, query="BookRAG OR RAPTOR", filters={}, limit=20,
        )
        assert r_or["mode"] == "sections"
        assert r_or.get("status") != "malformed_query", r_or
        names = {h["paper_name"] for h in r_or["results"]}
        assert "bookrag_2024" in names

    def test_not_excludes(self, seeded_db):
        # `BookRAG NOT supervised` matches sections that contain BookRAG
        # but not the word 'supervised'. None of the seeded sections
        # contain 'supervised' so all BookRAG hits survive.
        r_full = search_mod.mode_bm25(
            seeded_db, query="BookRAG", filters={}, limit=20,
        )
        r_not = search_mod.mode_bm25(
            seeded_db, query="BookRAG NOT supervised", filters={}, limit=20,
        )
        assert r_not.get("status") != "malformed_query", r_not
        full_count = sum(h["hit_count"] for h in r_full["results"])
        not_count = sum(h["hit_count"] for h in r_not["results"])
        assert not_count == full_count

    def test_phrase_query_finds_literal_match(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db, query='"hierarchical indexing"', filters={}, limit=10,
        )
        assert r.get("status") != "malformed_query", r
        # Seeded abstract has the literal "hierarchical indexing" phrase.
        names = {h["paper_name"] for h in r["results"]}
        assert "bookrag_2024" in names

    def test_prefix_token_matches_stem(self, seeded_db):
        # Seed body uses "indexing"; prefix "index*" should hit it.
        r = search_mod.mode_bm25(
            seeded_db, query="index*", filters={}, limit=10,
        )
        assert r.get("status") != "malformed_query", r
        names = {h["paper_name"] for h in r["results"]}
        assert "bookrag_2024" in names

    def test_paper_qualifier_narrows_to_one_paper(self, seeded_db):
        # Seed an extra section in stale_2024 mentioning BookRAG, then
        # confirm `paper:bookrag_2024 BookRAG` excludes stale_2024.
        p2_id = seeded_db.execute(
            "SELECT id FROM papers WHERE paper_name = ?", ("stale_2024",)
        ).fetchone()[0]
        seeded_db.execute(
            """
            INSERT INTO sections
                (paper_id, domain, paper_name, section_title, section_level, body)
            VALUES (?, 'other', 'stale_2024', 'Mentions', '1',
                    'BookRAG also mentioned here')
            """,
            (p2_id,),
        )
        r_all = search_mod.mode_bm25(
            seeded_db, query="BookRAG", filters={}, limit=20,
        )
        r_one = search_mod.mode_bm25(
            seeded_db,
            query="paper:bookrag_2024 BookRAG",
            filters={},
            limit=20,
        )
        names_all = {h["paper_name"] for h in r_all["results"]}
        names_one = {h["paper_name"] for h in r_one["results"]}
        assert "stale_2024" in names_all
        assert names_one == {"bookrag_2024"}

    def test_domain_qualifier_in_query(self, seeded_db):
        # qualifier-form domain narrows just like the kwarg form.
        r = search_mod.mode_bm25(
            seeded_db, query="domain:rag BookRAG", filters={}, limit=20,
        )
        names = {h["paper_name"] for h in r["results"]}
        assert "bookrag_2024" in names

    def test_domain_qualifier_conflict_with_kwarg(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db,
            query="domain:rag BookRAG",
            filters={"domain": "other"},
            limit=10,
        )
        assert r["status"] == "malformed_query"

    def test_kind_qualifier_rejected_on_bm25(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db, query="kind:entity BookRAG", filters={}, limit=10,
        )
        assert r["status"] == "malformed_query"
        assert "kind" in r["error"]

    def test_unclosed_quote_returns_malformed_query(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db, query='"unclosed phrase', filters={}, limit=10,
        )
        assert r["status"] == "malformed_query"
        assert r["mode"] == "sections"

    def test_regex_returns_malformed_query(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db, query="/foo.*/", filters={}, limit=10,
        )
        assert r["status"] == "malformed_query"

    def test_qualifier_only_query_is_empty(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db, query="paper:bookrag_2024", filters={}, limit=10,
        )
        assert r["status"] == "empty_query"


class TestQuerySyntaxSearch:
    """Same coverage on mode_search — verifies surface: routing and
    kind: filtering at the taxonomy bucket."""

    def test_or_in_taxonomy_search(self, seeded_db):
        r = search_mod.mode_search(
            seeded_db, query="BookRAG OR RAPTOR",
            filters={"domain": "rag"}, limit=10,
        )
        assert r.get("status") != "malformed_query", r
        canon = {row["canonical_name"] for row in r["taxonomy"]}
        assert "BookRAG" in canon or "RAPTOR" in canon

    def test_kind_filter_narrows_taxonomy_bucket(self, seeded_db):
        # Without kind: both entity and collection rows can surface for
        # 'indexing' (entity-typed BookRAG/RAPTOR don't match, but
        # collection 'hierarchical indexing' does). With kind:collection
        # nothing else can sneak in.
        r = search_mod.mode_search(
            seeded_db, query="kind:collection indexing",
            filters={"domain": "rag"}, limit=10,
        )
        assert r.get("status") != "malformed_query", r
        kinds = {row["kind"] for row in r["taxonomy"]}
        assert kinds <= {"collection"}, r["taxonomy"]

    def test_surface_sections_skips_readmes(self, seeded_db):
        # Unqueried buckets are OMITTED from the payload (vs returning []
        # — that would suggest "searched and found nothing" which misleads
        # the agent). The markdown renderer simply drops the heading.
        r = search_mod.mode_search(
            seeded_db, query="surface:sections BookRAG",
            filters={"domain": "rag"}, limit=10,
        )
        assert r.get("status") != "malformed_query", r
        assert "readmes" not in r
        assert "sections" in r

    def test_surface_readmes_skips_sections(self, seeded_db):
        r = search_mod.mode_search(
            seeded_db, query="surface:readmes BookRAG",
            filters={"domain": "rag"}, limit=10,
        )
        assert r.get("status") != "malformed_query", r
        assert "sections" not in r
        assert "readmes" in r

    def test_surface_sections_skips_taxonomy_and_readmes(self, seeded_db):
        # Tighter contract: surface:X means ONLY bucket X populated.
        r = search_mod.mode_search(
            seeded_db, query="surface:sections BookRAG",
            filters={"domain": "rag"}, limit=10,
        )
        assert "taxonomy" not in r
        assert "readmes" not in r
        assert "sections" in r

    def test_surface_taxonomy_only_populates_taxonomy(self, seeded_db):
        # `BookRAG` is a canonical term in the taxonomy. surface:taxonomy
        # produces only the taxonomy bucket; sections + readmes keys absent.
        r = search_mod.mode_search(
            seeded_db, query="surface:taxonomy BookRAG",
            filters={"domain": "rag"}, limit=10,
        )
        assert r.get("status") != "malformed_query", r
        assert "sections" not in r
        assert "readmes" not in r
        canon = {row["canonical_name"] for row in r["taxonomy"]}
        assert "BookRAG" in canon

    def test_surface_taxonomy_with_kind_filter(self, seeded_db):
        # Combine surface:taxonomy with kind: — the taxonomy bucket
        # narrows further, sections/readmes keys absent.
        r = search_mod.mode_search(
            seeded_db, query="surface:taxonomy kind:collection indexing",
            filters={"domain": "rag"}, limit=10,
        )
        assert "sections" not in r
        assert "readmes" not in r
        kinds = {row["kind"] for row in r["taxonomy"]}
        assert kinds <= {"collection"}

    def test_markdown_omits_unqueried_buckets(self, seeded_db):
        r = search_mod.mode_search(
            seeded_db, query="surface:taxonomy reasoning",
            filters={"domain": "rag"}, limit=10,
        )
        md = search_mod.format_search_markdown(r)
        # The taxonomy heading is present; sections / readmes are NOT.
        assert "## taxonomy" in md
        assert "## sections" not in md
        assert "## readmes" not in md

    def test_surface_taxonomy_rejected_on_bm25(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db, query="surface:taxonomy BookRAG",
            filters={}, limit=10,
        )
        assert r["status"] == "malformed_query"
        assert "taxonomy" in r["error"]

    def test_paper_qualifier_in_search(self, seeded_db):
        r = search_mod.mode_search(
            seeded_db, query="paper:bookrag_2024 BookRAG",
            filters={}, limit=10,
        )
        names = {g["paper_name"] for g in r["sections"]}
        assert names == {"bookrag_2024"} or names == set()  # any others excluded

    def test_malformed_search_query_is_soft_fail(self, seeded_db):
        r = search_mod.mode_search(
            seeded_db, query='"unclosed', filters={}, limit=10,
        )
        assert r["status"] == "malformed_query"
        assert r["mode"] == "search"

    def test_search_qualifier_only_is_empty_query(self, seeded_db):
        r = search_mod.mode_search(
            seeded_db, query="paper:bookrag_2024",
            filters={}, limit=10,
        )
        assert r["status"] == "empty_query"


class TestModeSearchMulti:
    """Multi-query fan-out: each query runs through mode_search
    independently; per-query payloads concatenated into one envelope."""

    def test_envelope_shape(self, seeded_db):
        r = search_mod.mode_search_multi(
            seeded_db,
            queries=["BookRAG", "RAPTOR"],
            filters={"domain": "rag"},
            limit=5,
        )
        assert r["mode"] == "search"
        assert r["multi"] is True
        assert r["queries"] == ["BookRAG", "RAPTOR"]
        assert r["domain"] == "rag"
        assert isinstance(r["results"], list) and len(r["results"]) == 2
        # Each sub-payload is a full mode_search envelope.
        for sub in r["results"]:
            assert sub["mode"] == "search"
            assert "taxonomy" in sub
            assert "sections" in sub
            assert "readmes" in sub

    def test_each_query_runs_independently(self, seeded_db):
        # Two queries hitting different canonicals — fan-out preserves
        # per-query routing rather than OR-ing them into one big query.
        r = search_mod.mode_search_multi(
            seeded_db,
            queries=["BookRAG", "RAPTOR"],
            filters={"domain": "rag"},
            limit=5,
        )
        sub_a, sub_b = r["results"]
        assert sub_a["query"] == "BookRAG"
        assert sub_b["query"] == "RAPTOR"

    def test_per_query_qualifiers_independent(self, seeded_db):
        # Different surface qualifiers per query → each sub-payload only
        # populates its own bucket subset.
        r = search_mod.mode_search_multi(
            seeded_db,
            queries=["surface:taxonomy BookRAG", "surface:sections BookRAG"],
            filters={"domain": "rag"},
            limit=5,
        )
        a, b = r["results"]
        assert "taxonomy" in a and "sections" not in a and "readmes" not in a
        assert "sections" in b and "taxonomy" not in b and "readmes" not in b

    def test_per_query_soft_failure_preserved(self, seeded_db):
        # One good query + one malformed query: the malformed one comes
        # back as a soft-failure inside the per-query payload, the good
        # one runs normally. The envelope itself is NOT marked as failed.
        r = search_mod.mode_search_multi(
            seeded_db,
            queries=["BookRAG", '"unclosed'],
            filters={"domain": "rag"},
            limit=5,
        )
        assert r.get("status") is None  # envelope itself OK
        good, bad = r["results"]
        assert good.get("status") is None
        assert bad["status"] == "malformed_query"

    def test_empty_queries_list_soft_fails(self, seeded_db):
        r = search_mod.mode_search_multi(
            seeded_db, queries=[], filters={}, limit=5,
        )
        assert r["status"] == "empty_query"
        assert r["mode"] == "search"

    def test_too_many_queries_soft_fails(self, seeded_db):
        # Cap at _MAX_SEARCH_MULTI_QUERIES; over-cap returns malformed
        # rather than silently truncating.
        cap = search_mod._MAX_SEARCH_MULTI_QUERIES
        r = search_mod.mode_search_multi(
            seeded_db,
            queries=["foo"] * (cap + 1),
            filters={},
            limit=5,
        )
        assert r["status"] == "malformed_query"
        assert "too many" in r["error"]

    def test_filters_apply_uniformly(self, seeded_db):
        # domain kwarg applies to every fan-out query.
        r = search_mod.mode_search_multi(
            seeded_db,
            queries=["BookRAG", "RAPTOR"],
            filters={"domain": "other"},  # no seeded paper in 'other'
            limit=5,
        )
        for sub in r["results"]:
            assert sub["taxonomy"] == []

    def test_markdown_renders_each_query(self, seeded_db):
        r = search_mod.mode_search_multi(
            seeded_db,
            queries=["BookRAG", "RAPTOR"],
            filters={"domain": "rag"},
            limit=5,
        )
        md = search_mod.format_search_markdown(r)
        assert md.startswith("# search (multi: 2 queries)")
        assert "## query 1: 'BookRAG'" in md
        assert "## query 2: 'RAPTOR'" in md
        # Sub-buckets are demoted by one level so they nest cleanly.
        assert "### taxonomy" in md
        assert "### sections" in md
        # The sub-payload's own H1 ("# search 'q'") was stripped.
        assert "# search 'BookRAG'" not in md
        assert "# search 'RAPTOR'" not in md

    def test_markdown_renders_per_query_soft_failure(self, seeded_db):
        r = search_mod.mode_search_multi(
            seeded_db,
            queries=["BookRAG", '"unclosed'],
            filters={"domain": "rag"},
            limit=5,
        )
        md = search_mod.format_search_markdown(r)
        # The per-query header is present for both; the malformed status
        # is tagged onto the H2 of the failing query so the diagnostic
        # is visible in the document outline.
        assert "## query 1: 'BookRAG'" in md
        assert '## query 2 (malformed query): \'"unclosed\'' in md
        # And the inline error/hint surface in the body.
        assert "unclosed quote" in md


def _system_search_soft_failures() -> frozenset[str]:
    """Tiny helper so the empty-query test can verify the status string is
    a member of the project's soft-failure set without re-importing the
    private constant inside the test class."""
    from _system.scripts.search import _SOFT_FAILURE_STATUSES
    return _SOFT_FAILURE_STATUSES


# ===========================================================================
# Mode 3 — Browse
# ===========================================================================


class TestModeBrowse:
    def test_collections_by_domain(self, seeded_db):
        r = search_mod.mode_browse(
            seeded_db, which="collections", filters={"domain": "rag"}
        )
        assert r["mode"] == "collections"
        names = [row["collection"] for row in r["results"]]
        assert "hierarchical indexing" in names

    def test_topics_by_domain(self, seeded_db):
        r = search_mod.mode_browse(
            seeded_db, which="topics", filters={"domain": "rag"}
        )
        assert r["mode"] == "topics"
        names = [row["topic"] for row in r["results"]]
        assert "entity resolution" in names

    def test_topics_by_collection_includes_papers_and_repos(self, seeded_db):
        # A second paper in the SAME collection contributes its topic.
        p_same_id = _insert_paper(
            seeded_db,
            arxiv_id="2403.00003",
            paper_name="hier_followup_2024",
            title="Hierarchical Indexing Follow-up",
            abstract="More on hierarchical indexing.",
            markdown=None,
            domain="rag",
            collection="hierarchical indexing",
            needs_review=0,
            ingested_at="2024-03-01T00:00:00+00:00",
        )
        _insert_paper_topic(
            seeded_db,
            paper_id=p_same_id,
            domain="rag",
            topic="tree summarization",
        )
        # A paper in a DIFFERENT collection in the same domain MUST be
        # excluded — its topic should not appear in the rollup.
        p_other_id = _insert_paper(
            seeded_db,
            arxiv_id="2404.00004",
            paper_name="dense_retrieval_2024",
            title="Dense Retrieval",
            abstract="Dense retrieval approaches.",
            markdown=None,
            domain="rag",
            collection="dense retrieval",
            needs_review=0,
            ingested_at="2024-04-01T00:00:00+00:00",
        )
        _insert_paper_topic(
            seeded_db,
            paper_id=p_other_id,
            domain="rag",
            topic="dense embeddings",
        )
        # A repo in the target collection contributes via the polymorphic
        # `collections` junction.
        seeded_db.execute(
            "INSERT INTO repos (repo_slug, url, host, owner, name, "
            "  ingested_at, status, domain, collection) "
            "VALUES ('owner-hier', 'https://github.com/owner/hier', "
            "  'github.com', 'owner', 'hier', "
            "  '2024-05-01T00:00:00+00:00', 'classified', "
            "  'rag', 'hierarchical indexing')"
        )
        repo_id = seeded_db.execute(
            "SELECT id FROM repos WHERE repo_slug = 'owner-hier'"
        ).fetchone()[0]
        seeded_db.execute(
            "INSERT INTO collection_definitions (domain, name) VALUES (?, ?) "
            " ON CONFLICT DO NOTHING",
            ("rag", "hierarchical indexing"),
        )
        seeded_db.execute(
            "INSERT INTO collections "
            " (target_kind, target_id, domain, collection, is_primary) "
            " VALUES ('repo', ?, 'rag', 'hierarchical indexing', 1)",
            (repo_id,),
        )
        seeded_db.execute(
            "INSERT INTO topics (target_kind, target_id, domain, topic) "
            "VALUES ('repo', ?, 'rag', 'tree retrieval')",
            (repo_id,),
        )
        # A repo in a DIFFERENT collection MUST be excluded.
        seeded_db.execute(
            "INSERT INTO repos (repo_slug, url, host, owner, name, "
            "  ingested_at, status, domain, collection) "
            "VALUES ('owner-dense', 'https://github.com/owner/dense', "
            "  'github.com', 'owner', 'dense', "
            "  '2024-05-02T00:00:00+00:00', 'classified', "
            "  'rag', 'dense retrieval')"
        )
        other_repo_id = seeded_db.execute(
            "SELECT id FROM repos WHERE repo_slug = 'owner-dense'"
        ).fetchone()[0]
        seeded_db.execute(
            "INSERT INTO collection_definitions (domain, name) VALUES (?, ?) "
            " ON CONFLICT DO NOTHING",
            ("rag", "dense retrieval"),
        )
        seeded_db.execute(
            "INSERT INTO collections "
            " (target_kind, target_id, domain, collection, is_primary) "
            " VALUES ('repo', ?, 'rag', 'dense retrieval', 1)",
            (other_repo_id,),
        )
        seeded_db.execute(
            "INSERT INTO topics (target_kind, target_id, domain, topic) "
            "VALUES ('repo', ?, 'rag', 'dense embeddings')",
            (other_repo_id,),
        )

        r = search_mod.mode_browse(
            seeded_db,
            which="topics",
            filters={"collection": "hierarchical indexing"},
        )
        assert r["mode"] == "topics"
        rows_by_topic = {row["topic"]: row for row in r["results"]}
        # Included: paper topic from p1, paper topic from the in-collection
        # follow-up, and the repo topic.
        assert "entity resolution" in rows_by_topic
        assert "tree summarization" in rows_by_topic
        assert "tree retrieval" in rows_by_topic
        # Excluded: topics whose only sources sit in other collections.
        assert "dense embeddings" not in rows_by_topic
        # Per-kind counts split the union correctly.
        assert rows_by_topic["entity resolution"]["paper_count"] == 1
        assert rows_by_topic["entity resolution"]["repo_count"] == 0
        assert rows_by_topic["tree retrieval"]["paper_count"] == 0
        assert rows_by_topic["tree retrieval"]["repo_count"] == 1

    def test_topics_by_collection_disambiguates_with_domain(self, seeded_db):
        # Re-use the existing collection name "misc" (only registered under
        # the 'other' domain by the fixture) and also create it under 'rag'.
        seeded_db.execute(
            "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
            "VALUES ('rag', 'misc', NULL)"
        )
        rag_misc_id = _insert_paper(
            seeded_db,
            arxiv_id="2405.00005",
            paper_name="rag_misc_2024",
            title="RAG Miscellany",
            abstract="A grab bag.",
            markdown=None,
            domain="rag",
            collection="misc",
            needs_review=0,
            ingested_at="2024-05-03T00:00:00+00:00",
        )
        _insert_paper_topic(
            seeded_db,
            paper_id=rag_misc_id,
            domain="rag",
            topic="rag-misc-topic",
        )
        # The 'other' domain's misc paper (stale_2024) carries no topics
        # in the fixture, so add one so the cross-domain mix would be
        # detectable.
        other_misc_id = seeded_db.execute(
            "SELECT id FROM papers WHERE paper_name = 'stale_2024'"
        ).fetchone()[0]
        _insert_paper_topic(
            seeded_db,
            paper_id=other_misc_id,
            domain="other",
            topic="other-misc-topic",
        )

        # Without domain: both 'misc' collections collapse into the rollup.
        both = search_mod.mode_browse(
            seeded_db,
            which="topics",
            filters={"collection": "misc"},
        )
        topics_both = {row["topic"] for row in both["results"]}
        assert "rag-misc-topic" in topics_both
        assert "other-misc-topic" in topics_both

        # With domain='rag': only the rag-side misc collection contributes.
        only_rag = search_mod.mode_browse(
            seeded_db,
            which="topics",
            filters={"collection": "misc", "domain": "rag"},
        )
        topics_rag = {row["topic"] for row in only_rag["results"]}
        assert "rag-misc-topic" in topics_rag
        assert "other-misc-topic" not in topics_rag

    def test_entity_type_list(self, seeded_db):
        r = search_mod.mode_browse(
            seeded_db, which="entity_type", filters={"entity_type": "method"}
        )
        assert r["mode"] == "entity_type"
        # Each result row is just the canonical name — no `paper_count`
        # column under the synonym-index regime. Drill into a canonical
        # via `--entity NAME` for paper-by-paper detail.
        for row in r["results"]:
            assert set(row.keys()) == {"entity_name"}
        names = [row["entity_name"] for row in r["results"]]
        assert "BookRAG" in names

    def test_aliases_includes_provenance(self, seeded_db):
        r = search_mod.mode_browse(
            seeded_db, which="aliases", filters={"aliases_term": "RAPTOR"}
        )
        assert r["mode"] == "aliases"
        assert r["results"]
        for row in r["results"]:
            assert "alias" in row
            assert "source_paper" in row
            assert "match_tier" in row

    def test_needs_review_returns_flagged_papers(self, seeded_db):
        r = search_mod.mode_browse(seeded_db, which="needs_review", filters={})
        assert r["mode"] == "needs_review"
        names = [row["paper_name"] for row in r["results"]]
        assert "stale_2024" in names
        assert "bookrag_2024" not in names


# ===========================================================================
# Mode 4 — ToC
# ===========================================================================


class TestModeToc:
    def test_returns_header_hierarchy(self, seeded_db):
        r = search_mod.mode_toc(seeded_db, slug="bookrag_2024")
        assert r["mode"] == "toc"
        assert r["slug"] == "bookrag_2024"
        # Flatten to titles — the fixture has # Abstract, # Introduction,
        # # Method, ## Setup, # Experiments, ## Setup at minimum.
        levels_titles = [(e["level"], e["title"]) for e in r["toc"]]
        assert (1, "Abstract") in levels_titles
        assert (1, "Method") in levels_titles
        assert (2, "Setup") in levels_titles

    def test_ignores_headers_in_fenced_code(self, seeded_db):
        md = (
            "# Real Header\n\n"
            "Some body.\n\n"
            "```\n"
            "# Not A Header\n"
            "```\n\n"
            "# Another Real Header\n"
        )
        seeded_db.execute(
            "UPDATE papers SET markdown = ? WHERE paper_name = ?",
            (md, "bookrag_2024"),
        )
        r = search_mod.mode_toc(seeded_db, slug="bookrag_2024")
        titles = [e["title"] for e in r["toc"]]
        assert "Not A Header" not in titles
        assert "Real Header" in titles
        assert "Another Real Header" in titles

    def test_unknown_slug_raises(self, seeded_db):
        with pytest.raises(ValueError):
            search_mod.mode_toc(seeded_db, slug="nope_2099")


class TestModeTocMany:
    def test_returns_per_source_results(self, seeded_db):
        r = search_mod.mode_toc_many(
            seeded_db, slugs=["bookrag_2024", "stale_2024"]
        )
        assert r["mode"] == "toc_many"
        assert r["slugs"] == ["bookrag_2024", "stale_2024"]
        names = [sub["slug"] for sub in r["results"]]
        assert names == ["bookrag_2024", "stale_2024"]
        for sub in r["results"]:
            assert sub["mode"] == "toc"
            assert "toc" in sub
        assert r["missing"] == []

    def test_missing_slug_collected_not_raised(self, seeded_db):
        r = search_mod.mode_toc_many(
            seeded_db, slugs=["bookrag_2024", "no_such_paper"],
        )
        names = [sub["slug"] for sub in r["results"]]
        assert names == ["bookrag_2024"]
        assert r["missing"] == ["no_such_paper"]

    def test_dedupes_input_order_preserved(self, seeded_db):
        r = search_mod.mode_toc_many(
            seeded_db,
            slugs=["bookrag_2024", "bookrag_2024", "stale_2024"],
        )
        assert r["slugs"] == ["bookrag_2024", "stale_2024"]
        assert len(r["results"]) == 2

    def test_empty_list_raises(self, seeded_db):
        with pytest.raises(ValueError):
            search_mod.mode_toc_many(seeded_db, slugs=[])


# ===========================================================================
# Mode 5a — Read
# ===========================================================================


class TestModeRead:
    def test_full_markdown(self, seeded_db):
        r = search_mod.mode_read(seeded_db, slug="bookrag_2024", section=None)
        assert r["mode"] == "read"
        assert r["status"] == "ok"
        assert r["slug"] == "bookrag_2024"
        assert r["section"] is None
        assert "BookRAG" in r["text"]

    def test_section_returns_hierarchical_slice(self, seeded_db):
        r = search_mod.mode_read(
            seeded_db, slug="bookrag_2024", section="Method"
        )
        assert r["status"] == "ok"
        assert r["section"] == "Method"
        # Must include the Method header and its Setup child.
        assert "# Method" in r["text"]
        assert "## Setup" in r["text"]
        # Must stop before the Experiments sibling header.
        assert "# Experiments" not in r["text"]

    def test_section_disambiguates_via_breadcrumb(self, seeded_db):
        # There are TWO "Setup" subsections — one under Method, one under
        # Experiments. The breadcrumb query picks the right one.
        r = search_mod.mode_read(
            seeded_db,
            slug="bookrag_2024",
            section="Experiments > Setup",
        )
        assert r["status"] == "ok"
        assert "Setup inside Experiments." in r["text"]
        assert "Setup inside Method." not in r["text"]

    def test_missing_section_emits_structured_payload(self, seeded_db):
        """Well-formed --section that doesn't match: structured payload, no raise."""
        r = search_mod.mode_read(
            seeded_db, slug="bookrag_2024", section="NoSuchSection"
        )
        assert r["mode"] == "read"
        assert r["status"] == "section_not_found"
        assert r["slug"] == "bookrag_2024"
        assert r["requested_section"] == "NoSuchSection"
        assert "text" not in r  # Don't leak any partial markdown.
        # The fallback hint must reference --toc and the whole-source option
        # — that's the actionable recovery path for the agent.
        assert "--toc" in r["hint"]
        assert "bookrag_2024" in r["hint"]

    def test_missing_section_lists_top_level_titles(self, seeded_db):
        """The available-sections payload exposes levels 1-2 from the source."""
        r = search_mod.mode_read(
            seeded_db, slug="bookrag_2024", section="ZZZ Nope"
        )
        assert r["status"] == "section_not_found"
        avail = r["available_top_level_sections"]
        # Fixture markdown has Abstract / Introduction / Method / Experiments
        # plus level-2 Setup duplicated. All level <= 2 titles must surface.
        assert "Abstract" in avail
        assert "Introduction" in avail
        assert "Method" in avail
        assert "Experiments" in avail

    def test_malformed_section_query_emits_structured_payload(self, seeded_db):
        """Malformed breadcrumb (empty segment): structured payload with rule message."""
        r = search_mod.mode_read(
            seeded_db, slug="bookrag_2024", section="A >> B"
        )
        assert r["status"] == "malformed_section_query"
        assert r["slug"] == "bookrag_2024"
        assert r["requested_section"] == "A >> B"
        # The error string carries the actual rule violated, not just the input.
        assert "empty segment" in r["error"]
        # The hint tells Claude the right syntax.
        assert "Parent > Child" in r["hint"]

    @pytest.mark.parametrize(
        "bad",
        ["", "   ", ">", "A >", "> B", "A\nB", "x" * 1000],
    )
    def test_malformed_section_query_variants(self, seeded_db, bad):
        """Every malformed shape gets caught and translated to the structured payload."""
        r = search_mod.mode_read(
            seeded_db, slug="bookrag_2024", section=bad
        )
        assert r["status"] == "malformed_section_query"

    def test_unknown_slug_raises(self, seeded_db):
        """Wrong slug is a hard error (not an agent miss) — keep raising."""
        with pytest.raises(ValueError):
            search_mod.mode_read(seeded_db, slug="nope_2099", section=None)


class TestModeReadFailureRendering:
    """to_human + main() behavior on the read-mode failure payloads."""

    def test_human_section_not_found_mentions_toc(self, seeded_db):
        payload = search_mod.mode_read(
            seeded_db, slug="bookrag_2024", section="ZZZ Nope"
        )
        out = search_mod.to_human(payload)
        assert "ZZZ Nope" in out
        assert "--toc" in out

    def test_human_malformed_query_includes_rule(self, seeded_db):
        payload = search_mod.mode_read(
            seeded_db, slug="bookrag_2024", section="A >> B"
        )
        out = search_mod.to_human(payload)
        assert "malformed" in out.lower()
        assert "Parent > Child" in out

    def test_main_exit_code_2_on_section_not_found(
        self, seeded_db, db_path, capsys
    ):
        del seeded_db  # Fixture only needed for its side effect on db_path.
        rc = search_mod.main(
            ["--db", str(db_path), "--read", "bookrag_2024", "--section", "ZZZ Nope"]
        )
        assert rc == 2
        captured = capsys.readouterr()
        # JSON mode (default): payload still goes to stdout — that's the
        # agent's recovery channel.
        assert '"status": "section_not_found"' in captured.out
        assert captured.err == ""

    def test_main_exit_code_2_on_malformed_query(
        self, seeded_db, db_path, capsys
    ):
        del seeded_db
        rc = search_mod.main(
            ["--db", str(db_path), "--read", "bookrag_2024", "--section", "A >> B"]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert '"status": "malformed_section_query"' in captured.out

    def test_main_exit_code_0_on_happy_read(self, seeded_db, db_path, capsys):
        del seeded_db
        rc = search_mod.main(
            ["--db", str(db_path), "--read", "bookrag_2024", "--section", "Method"]
        )
        assert rc == 0
        captured = capsys.readouterr()
        assert '"status": "ok"' in captured.out

    def test_main_human_failure_writes_stderr_not_stdout(
        self, seeded_db, db_path, capsys
    ):
        """``--human`` keeps stdout empty on soft failure so shell pipes are clean."""
        del seeded_db
        rc = search_mod.main(
            [
                "--db", str(db_path),
                "--read", "bookrag_2024",
                "--section", "ZZZ Nope",
                "--human",
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "ZZZ Nope" in captured.err
        assert "--toc" in captured.err


# ===========================================================================
# Mode 5b — Figure / page BLOB extraction
# ===========================================================================


class TestModeFigure:
    def test_extracts_blob_to_tempfile(self, seeded_db):
        r = search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="3")
        assert r["mode"] == "figure"
        path = Path(r["path"])
        # Must live under the OS tempdir and follow the mkstemp prefix rule.
        assert path.exists()
        assert path.is_file()
        # Must NOT be the predictable naive path.
        assert str(path) != "/tmp/lodestone_bookrag_2024_fig3.png"
        assert path.name.startswith("lodestone_bookrag_2024_fig3_")
        # Content matches what we seeded.
        assert path.read_bytes() == _PNG_1x1

    def test_path_is_unique_across_calls(self, seeded_db):
        r1 = search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="3")
        r2 = search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="3")
        assert r1["path"] != r2["path"]

    def test_accepts_dom_ordinal(self, seeded_db):
        r = search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="3")
        # figure_number=3 matches the bookrag_2024 figure.
        assert Path(r["path"]).exists()

    def test_falls_back_to_display_number(self, seeded_db):
        # display_number="Figure 3a" belongs to figure_number=4 in the
        # fixture. Querying by "Figure 3a" must find it via the fallback.
        r = search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="Figure 3a")
        assert Path(r["path"]).exists()

    def test_rejects_illegal_paper_name(self, seeded_db):
        with pytest.raises(ValueError):
            search_mod.mode_figure(seeded_db, paper="../evil", n="3")
        with pytest.raises(ValueError):
            search_mod.mode_figure(seeded_db, paper="NotASlug", n="3")

    def test_missing_figure_raises(self, seeded_db):
        with pytest.raises(ValueError):
            search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="999")


# ===========================================================================
# CLI mode-conflict guard
# ===========================================================================


class TestModeConflicts:
    """The dispatcher must reject ambiguous flag combinations instead of
    silently picking one via first-match precedence."""

    def _run(self, argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "_system.scripts.search", *argv],
            capture_output=True,
            text=True,
            timeout=5,
        )

    def test_query_plus_entity_is_rejected(self):
        result = self._run(["some query", "--entity", "RAPTOR"])
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr

    def test_entity_plus_topic_is_rejected(self):
        result = self._run(["--entity", "X", "--topic", "Y"])
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr

    def test_toc_plus_read_is_rejected(self):
        result = self._run(["--toc", "p1", "--read", "p2"])
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr


# ===========================================================================
# Output formatting
# ===========================================================================


class TestOutputFormatting:
    def test_each_mode_returns_dict_with_mode_key(self, seeded_db):
        bm25 = search_mod.mode_bm25(
            seeded_db,
            query="BookRAG",
            filters={},
            limit=5,
        )
        assert bm25["mode"] == "sections"

        tax = search_mod.mode_taxonomy_lookup(
            seeded_db, query="RAPTOR", filters={}
        )
        assert tax["mode"] == "lookup"

        br = search_mod.mode_browse(
            seeded_db, which="needs_review", filters={}
        )
        assert br["mode"] == "needs_review"

        toc = search_mod.mode_toc(seeded_db, slug="bookrag_2024")
        assert toc["mode"] == "toc"

        read = search_mod.mode_read(
            seeded_db, slug="bookrag_2024", section=None
        )
        assert read["mode"] == "read"

        fig = search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="3")
        assert fig["mode"] == "figure"

    def test_human_formatter_nonempty_per_mode(self, seeded_db):
        payloads = [
            search_mod.mode_bm25(
                seeded_db, query="BookRAG", filters={}, limit=5
            ),
            search_mod.mode_taxonomy_lookup(
                seeded_db, query="RAPTOR", filters={}
            ),
            search_mod.mode_browse(seeded_db, which="needs_review", filters={}),
            search_mod.mode_browse(
                seeded_db, which="aliases", filters={"aliases_term": "RAPTOR"}
            ),
            search_mod.mode_toc(seeded_db, slug="bookrag_2024"),
            search_mod.mode_read(seeded_db, slug="bookrag_2024", section=None),
            search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="3"),
        ]
        for payload in payloads:
            out = search_mod.to_human(payload)
            assert isinstance(out, str)
            assert out.strip(), f"empty to_human for payload={payload!r}"


# ===========================================================================
# mode_overview
# ===========================================================================


def _seed_collection_row(
    conn: sqlite3.Connection,
    domain: str,
    name: str,
    description: str | None = None,
) -> None:
    # UPSERT so an explicit description here overrides any prior NULL row
    # left by _insert_paper's auto-seed of (domain, name).
    conn.execute(
        "INSERT INTO collection_definitions (domain, name, description) VALUES (?, ?, ?) "
        "ON CONFLICT(domain, name) DO UPDATE SET description = excluded.description",
        (domain, name, description),
    )


def test_mode_overview_basic(seeded_db):
    _seed_collection_row(
        seeded_db, "rag", "hierarchical indexing", "multi-level toc"
    )
    payload = search_mod.mode_overview(seeded_db, filters={})
    assert payload["mode"] == "overview"
    assert payload["domain"] is None

    domains = payload["domains"]
    names = [d["name"] for d in domains]
    # "rag" has bookrag_2024 (1 paper); "other" has stale_2024 (1 paper).
    assert "rag" in names
    assert "other" in names

    rag = next(d for d in domains if d["name"] == "rag")
    assert rag["paper_count"] == 1
    assert rag["collection_count"] == 1
    assert rag["collections"][0]["name"] == "hierarchical indexing"
    assert rag["collections"][0]["paper_count"] == 1
    # uncategorized_count was a leak that violated the system invariant
    # (every classified paper has a domain AND a collection); the field
    # is no longer surfaced.
    assert "uncategorized_count" not in rag


def test_mode_overview_skips_empty_collections(seeded_db):
    _seed_collection_row(seeded_db, "rag", "hierarchical indexing")
    _seed_collection_row(seeded_db, "rag", "ghost_collection")  # no papers

    payload = search_mod.mode_overview(seeded_db, filters={})
    rag = next(d for d in payload["domains"] if d["name"] == "rag")
    coll_names = [c["name"] for c in rag["collections"]]
    assert "hierarchical indexing" in coll_names
    assert "ghost_collection" not in coll_names


def test_mode_overview_no_uncategorized_field(seeded_db):
    """The system invariant is that every classified paper has both a
    domain and a collection. ``uncategorized_count`` was a leak from
    an earlier draft and must not surface."""
    payload = search_mod.mode_overview(seeded_db, filters={})
    for d in payload["domains"]:
        assert "uncategorized_count" not in d


def test_mode_overview_domain_filter(seeded_db):
    _seed_collection_row(seeded_db, "rag", "hierarchical indexing")
    payload = search_mod.mode_overview(seeded_db, filters={"domain": "rag"})
    assert payload["domain"] == "rag"
    names = [d["name"] for d in payload["domains"]]
    assert names == ["rag"]


# ===========================================================================
# mode_collection
# ===========================================================================


def test_mode_collection_single_string(seeded_db):
    _seed_collection_row(
        seeded_db, "rag", "hierarchical indexing", "multi-level toc"
    )
    _insert_paper_topic(
        seeded_db,
        paper_id=seeded_db.execute(
            "SELECT id FROM papers WHERE paper_name = 'bookrag_2024'"
        ).fetchone()[0],
        domain="rag",
        topic="hierarchical retrieval",
    )

    payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["hierarchical indexing"],
        filters={},
    )
    assert payload["mode"] == "collection"
    assert payload["missing"] == []
    assert len(payload["collections"]) == 1
    entry = payload["collections"][0]
    assert entry["domain"] == "rag"
    assert entry["collection"] == "hierarchical indexing"
    assert entry["description"] == "multi-level toc"
    assert entry["paper_count"] == 1
    assert entry["papers_truncated"] is False
    assert len(entry["papers"]) == 1
    paper = entry["papers"][0]
    assert paper["paper_name"] == "bookrag_2024"
    assert "abstract" in paper
    assert paper["abstract"]
    # Topic was inserted with topic="hierarchical retrieval" plus the seeded
    # "entity resolution" topic — both should surface.
    assert set(paper["topics"]) >= {"hierarchical retrieval", "entity resolution"}


def test_mode_collection_multi(seeded_db):
    _seed_collection_row(seeded_db, "rag", "hierarchical indexing")
    # Add a second collection with its own paper.
    p_id = _insert_paper(
        seeded_db,
        arxiv_id="2403.00003",
        paper_name="hybrid_2024",
        title="Hybrid Search",
        abstract="dense+sparse fusion.",
        markdown=None,
        domain="rag",
        collection="hybrid",
        needs_review=0,
        ingested_at="2024-03-01T00:00:00+00:00",
    )
    _seed_collection_row(seeded_db, "rag", "hybrid")
    del p_id

    payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["hierarchical indexing", "hybrid"],
        filters={},
    )
    coll_names = [e["collection"] for e in payload["collections"]]
    assert set(coll_names) == {"hierarchical indexing", "hybrid"}
    assert payload["missing"] == []


def test_mode_collection_missing_name(seeded_db):
    _seed_collection_row(seeded_db, "rag", "hierarchical indexing")
    payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["hierarchical indexing", "no_such_collection"],
        filters={},
    )
    coll_names = [e["collection"] for e in payload["collections"]]
    assert coll_names == ["hierarchical indexing"]
    assert payload["missing"] == ["no_such_collection"]


def test_mode_collection_cross_domain_no_filter(seeded_db):
    # Same name "shared" registered under two domains.
    _seed_collection_row(seeded_db, "rag", "shared")
    _seed_collection_row(seeded_db, "other", "shared")

    payload = search_mod.mode_collection(
        seeded_db, collection_names=["shared"], filters={}
    )
    pairs = {(e["domain"], e["collection"]) for e in payload["collections"]}
    assert pairs == {("rag", "shared"), ("other", "shared")}


def test_mode_collection_cross_domain_with_filter(seeded_db):
    _seed_collection_row(seeded_db, "rag", "shared")
    _seed_collection_row(seeded_db, "other", "shared")

    payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["shared"],
        filters={"domain": "rag"},
    )
    assert len(payload["collections"]) == 1
    assert payload["collections"][0]["domain"] == "rag"


def test_mode_collection_include_flags_off(seeded_db):
    _seed_collection_row(seeded_db, "rag", "hierarchical indexing")
    payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["hierarchical indexing"],
        filters={},
        include_abstracts=False,
        include_topics=False,
    )
    paper = payload["collections"][0]["papers"][0]
    assert "abstract" not in paper
    assert "topics" not in paper


def test_mode_collection_limit_truncates(seeded_db):
    _seed_collection_row(seeded_db, "rag", "big")
    for i in range(15):
        _insert_paper(
            seeded_db,
            arxiv_id=f"2410.{i:05d}",
            paper_name=f"big_{i:02d}",
            title=f"Paper {i}",
            abstract="abs",
            markdown=None,
            domain="rag",
            collection="big",
            needs_review=0,
            ingested_at=f"2024-10-{(i % 28) + 1:02d}T00:00:00+00:00",
        )

    payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["big"],
        filters={},
        limit=5,
    )
    entry = payload["collections"][0]
    assert entry["paper_count"] == 15
    assert len(entry["papers"]) == 5
    assert entry["papers_truncated"] is True


def test_mode_collection_empty_input_raises(seeded_db):
    with pytest.raises(ValueError):
        search_mod.mode_collection(
            seeded_db, collection_names=[], filters={}
        )


def test_mode_collection_returns_secondary_membership(seeded_db):
    """A paper whose primary is 'A' but who carries 'B' as a secondary
    in `paper_collections` must surface under both `mode_collection(["A"])`
    and `mode_collection(["B"])`."""
    p_id = _insert_paper(
        seeded_db,
        arxiv_id="2405.01010",
        paper_name="multi_2024",
        title="Bridging A and B",
        abstract="abs",
        markdown=None,
        domain="rag",
        collection="hierarchical indexing",  # primary = A
        needs_review=0,
        ingested_at="2024-05-01T00:00:00+00:00",
    )
    _add_secondary_collection(
        seeded_db, paper_id=p_id, domain="rag", collection="hybrid"
    )
    _seed_collection_row(seeded_db, "rag", "hierarchical indexing")
    _seed_collection_row(seeded_db, "rag", "hybrid")

    primary_payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["hierarchical indexing"],
        filters={"domain": "rag"},
    )
    primary_entry = primary_payload["collections"][0]
    assert "multi_2024" in {p["paper_name"] for p in primary_entry["papers"]}

    secondary_payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["hybrid"],
        filters={"domain": "rag"},
    )
    secondary_entry = secondary_payload["collections"][0]
    assert secondary_entry["paper_count"] == 1
    assert "multi_2024" in {p["paper_name"] for p in secondary_entry["papers"]}


def test_bm25_collection_filter_matches_secondary(seeded_db):
    """`bm25(query, collection='B')` must match a paper whose secondary
    membership is 'B' even when its primary `papers.collection` is 'A'."""
    p_id = _insert_paper(
        seeded_db,
        arxiv_id="2405.02020",
        paper_name="multi_bm25_2024",
        title="Hybrid Hierarchies",
        abstract="hybrid hierarchies for retrieval over books.",
        markdown=(
            "# Hybrid Hierarchies\n\n"
            "We introduce HybridHierarchies, a unique-marker phrase.\n"
        ),
        domain="rag",
        collection="hierarchical indexing",  # primary = A
        needs_review=0,
        ingested_at="2024-05-02T00:00:00+00:00",
    )
    _add_secondary_collection(
        seeded_db, paper_id=p_id, domain="rag", collection="hybrid"
    )
    _insert_sections_for_md(
        seeded_db,
        paper_id=p_id,
        domain="rag",
        paper_name="multi_bm25_2024",
        markdown=(
            "# Hybrid Hierarchies\n\n"
            "We introduce HybridHierarchies, a unique-marker phrase.\n"
        ),
    )

    payload = search_mod.mode_bm25(
        seeded_db,
        query="HybridHierarchies",
        filters={"collection": "hybrid"},
        limit=10,
    )
    paper_names = {r["paper_name"] for r in payload["results"]}
    assert "multi_bm25_2024" in paper_names


def test_browse_collections_counts_primary_and_secondary(seeded_db):
    """browse(view='collections') aggregates from `paper_collections`,
    so a paper that's a secondary in 'B' counts toward 'B' too."""
    p_id = _insert_paper(
        seeded_db,
        arxiv_id="2405.03030",
        paper_name="multi_browse_2024",
        title="t",
        abstract="a",
        markdown=None,
        domain="rag",
        collection="hierarchical indexing",
        needs_review=0,
        ingested_at="2024-05-03T00:00:00+00:00",
    )
    _add_secondary_collection(
        seeded_db, paper_id=p_id, domain="rag", collection="hybrid"
    )
    payload = search_mod.mode_browse(
        seeded_db, which="collections", filters={"domain": "rag"}
    )
    counts = {r["collection"]: r["count"] for r in payload["results"]}
    # The paper counts in both buckets thanks to the secondary row.
    assert counts.get("hybrid") == 1


# ===========================================================================
# Formatters / to_human integration for the new modes
# ===========================================================================


def test_format_overview_tree_renders_count_style(seeded_db):
    _seed_collection_row(
        seeded_db, "rag", "hierarchical indexing", "multi-level toc"
    )
    payload = search_mod.mode_overview(seeded_db, filters={})
    text = search_mod.format_overview_tree(payload)
    assert "# overview" in text
    assert "rag" in text
    # Tree connector + count annotation.
    assert "├──" in text or "└──" in text
    assert "paper" in text


def test_format_collection_text_renders_papers(seeded_db):
    _seed_collection_row(seeded_db, "rag", "hierarchical indexing")
    payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["hierarchical indexing"],
        filters={},
    )
    text = search_mod.format_collection_text(payload)
    assert "rag / hierarchical indexing" in text
    assert "bookrag_2024" in text
    # Tree connector for the single paper.
    assert "└──" in text or "├──" in text


def test_to_human_routes_overview_and_collection(seeded_db):
    _seed_collection_row(seeded_db, "rag", "hierarchical indexing")
    overview_payload = search_mod.mode_overview(seeded_db, filters={})
    assert "rag" in search_mod.to_human(overview_payload)

    coll_payload = search_mod.mode_collection(
        seeded_db,
        collection_names=["hierarchical indexing"],
        filters={},
    )
    assert "bookrag_2024" in search_mod.to_human(coll_payload)


# ===========================================================================
# Pagination footer rendering
# ===========================================================================


class TestPagingFooter:
    def test_bm25_renders_paging_footer_when_more(self, seeded_db):
        p1_id = seeded_db.execute(
            "SELECT id FROM papers WHERE paper_name = ?", ("bookrag_2024",)
        ).fetchone()[0]
        _seed_paginated_sections(
            seeded_db, paper_id=p1_id, paper_name="bookrag_2024",
            domain="rag", count=15, body_token="footertoken",
        )
        r = search_mod.mode_bm25(
            seeded_db, query="footertoken", filters={}, limit=5, offset=0,
        )
        assert r["has_more"] is True
        out = search_mod.to_human(r)
        assert "Re-call with --offset 5" in out, out
        assert "of 15" in out, out

    def test_bm25_renders_end_marker_when_no_more(self, seeded_db):
        p1_id = seeded_db.execute(
            "SELECT id FROM papers WHERE paper_name = ?", ("bookrag_2024",)
        ).fetchone()[0]
        _seed_paginated_sections(
            seeded_db, paper_id=p1_id, paper_name="bookrag_2024",
            domain="rag", count=8, body_token="endtoken",
        )
        r = search_mod.mode_bm25(
            seeded_db, query="endtoken", filters={}, limit=5, offset=5,
        )
        assert r["has_more"] is False
        out = search_mod.to_human(r)
        assert "end of results" in out, out

    def test_lookup_footer_when_paging(self, seeded_db):
        for i in range(7):
            tid = _insert_canonical(
                seeded_db,
                domain="rag",
                term_type="entity",
                entity_type="method",
                canonical_name=f"FooterEntity {i:02d}",
                first_seen_in="bookrag_2024",
            )
            _insert_terms_fts(
                seeded_db,
                term_id=tid,
                domain="rag",
                term_type="entity",
                entity_type="method",
                canonical_name=f"FooterEntity {i:02d}",
                aliases="",
            )
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, query="FooterEntity", filters={}, limit=3, offset=0,
        )
        out = search_mod.to_human(r)
        assert "Re-call with --offset 3" in out, out
        assert "of 7" in out, out

    def test_invalid_pagination_renders_error(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db, query="anything", filters={}, limit=5, offset=-1,
        )
        out = search_mod.to_human(r)
        assert "invalid pagination" in out.lower(), out
        assert ">= 0" in out, out


# ===========================================================================
# DB introspection + read-only SQL escape hatch
# ===========================================================================


class TestModeTables:
    def test_lists_user_tables_and_marks_virtual(self, conn):
        r = search_mod.mode_tables(conn)
        assert r["mode"] == "tables"
        assert r["status"] == "ok"
        assert r["include_internal"] is False
        names = {t["name"]: t["type"] for t in r["tables"]}
        # Core tables exist after init_db.
        assert names.get("papers") == "table"
        assert names.get("canonical_terms") == "table"
        # Virtual tables tagged correctly.
        assert names.get("sections") == "virtual"
        assert names.get("term_embeddings") == "virtual"
        # Internal shadow tables filtered out by default.
        for n in names:
            assert not (
                n.endswith("_data") or n.endswith("_idx") or
                n.endswith("_content") or n.endswith("_docsize") or
                n.endswith("_config")
            ), f"shadow table leaked: {n}"

    def test_include_internal_surfaces_shadow_tables(self, conn):
        r = search_mod.mode_tables(conn, include_internal=True)
        names = {t["name"] for t in r["tables"]}
        # FTS5 shadow tables for sections show up.
        assert "sections_data" in names or "sections_idx" in names, names


class TestModeSchema:
    def test_returns_ddl_columns_indexes(self, conn):
        r = search_mod.mode_schema(conn, table_names=["papers"])
        assert r["mode"] == "schema"
        assert r["status"] == "ok"
        assert r["missing"] == []
        assert len(r["tables"]) == 1
        t = r["tables"][0]
        assert t["name"] == "papers"
        assert t["type"] == "table"
        assert "CREATE TABLE" in (t["sql"] or "")
        col_names = {c["name"] for c in t["columns"]}
        assert {"id", "arxiv_id", "paper_name", "title"} <= col_names
        # PK on id.
        pk = next(c for c in t["columns"] if c["name"] == "id")
        assert pk["pk"] == 1

    def test_unknown_tables_land_in_missing(self, conn):
        r = search_mod.mode_schema(
            conn, table_names=["papers", "no_such_table"]
        )
        names = [t["name"] for t in r["tables"]]
        assert names == ["papers"]
        assert r["missing"] == ["no_such_table"]

    def test_virtual_table_tagged(self, conn):
        r = search_mod.mode_schema(conn, table_names=["sections"])
        t = r["tables"][0]
        assert t["type"] == "virtual"
        assert "CREATE VIRTUAL TABLE" in (t["sql"] or "")

    def test_empty_input_raises(self, conn):
        with pytest.raises(ValueError):
            search_mod.mode_schema(conn, table_names=[])


class TestModeQuery:
    def test_select_returns_rows(self, seeded_db, db_path):
        # Commit any seeded writes so the read-only conn opened inside
        # mode_query sees them.
        if seeded_db.in_transaction:
            seeded_db.commit()
        r = search_mod.mode_query(
            seeded_db,
            sql="SELECT paper_name, title FROM papers ORDER BY paper_name",
            db_path=db_path,
        )
        assert r["mode"] == "query"
        assert r["status"] == "ok"
        assert r["columns"] == ["paper_name", "title"]
        assert r["row_count"] >= 1
        assert r["truncated"] is False
        names = [row["paper_name"] for row in r["rows"]]
        assert "bookrag_2024" in names

    def test_write_attempt_is_read_only_violation(self, seeded_db, db_path):
        r = search_mod.mode_query(
            seeded_db,
            sql="DROP TABLE papers",
            db_path=db_path,
        )
        assert r["status"] == "read_only_violation"
        # Source table still there.
        assert seeded_db.execute(
            "SELECT count(*) FROM papers"
        ).fetchone()[0] >= 1

    def test_insert_attempt_is_read_only_violation(self, seeded_db, db_path):
        r = search_mod.mode_query(
            seeded_db,
            sql="INSERT INTO papers (arxiv_id, paper_name, title, "
                "authors, date, abstract, pdf_url, ingested_at, status) "
                "VALUES ('x','x','x','[]','2024-01-01','x','x','x','x')",
            db_path=db_path,
        )
        assert r["status"] == "read_only_violation"

    def test_multiple_statements_soft_fails(self, seeded_db, db_path):
        r = search_mod.mode_query(
            seeded_db, sql="SELECT 1; SELECT 2", db_path=db_path,
        )
        assert r["status"] == "multiple_statements"
        # Trailing terminator alone is also caught.
        r2 = search_mod.mode_query(
            seeded_db,
            sql="SELECT 1; DROP TABLE papers;",
            db_path=db_path,
        )
        assert r2["status"] == "multiple_statements"

    def test_syntax_error_is_query_failed(self, seeded_db, db_path):
        r = search_mod.mode_query(
            seeded_db, sql="SELEKT 1", db_path=db_path,
        )
        assert r["status"] == "query_failed"
        assert "SELEKT" in r["error"] or "syntax" in r["error"].lower()

    def test_unknown_table_is_query_failed(self, seeded_db, db_path):
        r = search_mod.mode_query(
            seeded_db, sql="SELECT * FROM no_such_table",
            db_path=db_path,
        )
        assert r["status"] == "query_failed"

    def test_blob_column_is_summarized(self, seeded_db, db_path):
        r = search_mod.mode_query(
            seeded_db,
            sql="SELECT figure_number, image FROM figures "
                "ORDER BY figure_number LIMIT 1",
            db_path=db_path,
        )
        assert r["status"] == "ok"
        assert r["row_count"] == 1
        row = r["rows"][0]
        blob = row["image"]
        assert isinstance(blob, dict)
        assert blob.get("_blob") is True
        assert blob.get("size_bytes", 0) > 0

    def test_row_ceiling_truncates(self, seeded_db, db_path):
        # Use a tiny ceiling and verify truncated=true.
        r = search_mod.mode_query(
            seeded_db,
            sql="SELECT name FROM sqlite_master ORDER BY name",
            db_path=db_path,
            max_rows=2,
        )
        assert r["status"] == "ok"
        assert r["truncated"] is True
        assert r["row_count"] == 2

    def test_virtual_table_query_works(self, seeded_db, db_path):
        # FTS5 queries against the sections virtual table should run on
        # the read-only conn (sqlite-vec / FTS5 are loaded via _load_vec).
        r = search_mod.mode_query(
            seeded_db,
            sql="SELECT paper_name FROM sections "
                "WHERE sections MATCH 'BookRAG' LIMIT 5",
            db_path=db_path,
        )
        assert r["status"] == "ok"
        assert r["row_count"] >= 1

    def test_query_timeout_short_budget(self, seeded_db, db_path):
        # Force the wall-clock budget into the past so the progress handler
        # interrupts on the very first check.
        r = search_mod.mode_query(
            seeded_db,
            sql=(
                "WITH RECURSIVE c(i) AS ("
                " SELECT 1 UNION ALL SELECT i+1 FROM c WHERE i < 1000000"
                ") SELECT count(*) FROM c"
            ),
            db_path=db_path,
            timeout_seconds=0.0,
        )
        assert r["status"] == "query_timeout", r


class TestIntrospectionToHuman:
    def test_tables_to_human(self, conn):
        r = search_mod.mode_tables(conn)
        out = search_mod.to_human(r)
        assert "tables" in out
        assert "papers" in out

    def test_schema_to_human(self, conn):
        r = search_mod.mode_schema(conn, table_names=["papers"])
        out = search_mod.to_human(r)
        assert "papers" in out
        assert "columns" in out

    def test_query_ok_to_human(self, seeded_db, db_path):
        r = search_mod.mode_query(
            seeded_db,
            sql="SELECT paper_name FROM papers LIMIT 1",
            db_path=db_path,
        )
        out = search_mod.to_human(r)
        assert "query" in out

    def test_query_soft_fail_to_human(self, seeded_db, db_path):
        r = search_mod.mode_query(
            seeded_db, sql="DROP TABLE papers", db_path=db_path,
        )
        out = search_mod.to_human(r)
        assert "read_only_violation" in out


class TestIntrospectionConflicts:
    def _run(self, argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "_system.scripts.search", *argv],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_tables_plus_sql_is_rejected(self):
        result = self._run(["--tables", "--sql", "SELECT 1"])
        assert result.returncode != 0
        assert "mutually exclusive" in result.stderr

    def test_include_internal_without_tables_is_rejected(self):
        result = self._run(["--include-internal", "--sql", "SELECT 1"])
        assert result.returncode != 0
        assert "include-internal" in result.stderr


# ===========================================================================
# Mode 6b — Citations
# ===========================================================================


def _insert_post_for_citations(
    conn: sqlite3.Connection,
    *,
    post_name: str,
    title: str,
    domain: str,
    collection: str,
    date: str,
    ingested_at: str,
) -> int:
    """Minimal post row sufficient to back the citations seeds.

    Posts have a NOT NULL invariant on (domain, collection) once their
    status is past `converted`, so we land them straight at `classified`
    with both fields populated.
    """
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES (?, ?, NULL)",
        (domain, collection),
    )
    cur = conn.execute(
        """
        INSERT INTO posts (
            post_name, source_url, canonical_url, title, author, site_name,
            date, abstract, domain, collection, ingested_at, status,
            section_count, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            post_name,
            f"https://example.com/{post_name}",
            f"https://example.com/{post_name}",
            title,
            "Author",
            "example.com",
            date,
            "Stub abstract.",
            domain,
            collection,
            ingested_at,
            "classified",
            0,
            0,
        ),
    )
    return cur.lastrowid


def _insert_paper_reference(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    bibitem_id: str | None,
    ref_number: int,
    raw_text: str,
    cited_arxiv_id: str | None,
    cited_paper_id: int | None,
) -> None:
    conn.execute(
        """
        INSERT INTO paper_references
            (paper_id, bibitem_id, ref_number, raw_text,
             cited_arxiv_id, cited_paper_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (paper_id, bibitem_id, ref_number, raw_text,
         cited_arxiv_id, cited_paper_id),
    )


def _insert_post_reference(
    conn: sqlite3.Connection,
    *,
    post_id: int,
    raw_text: str,
    cited_arxiv_id: str | None,
    cited_paper_id: int | None,
) -> None:
    conn.execute(
        """
        INSERT INTO post_references
            (post_id, raw_text, cited_arxiv_id, cited_paper_id)
        VALUES (?, ?, ?, ?)
        """,
        (post_id, raw_text, cited_arxiv_id, cited_paper_id),
    )


@pytest.fixture
def seeded_citations_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Tiny corpus tailored to mode_citations.

    Layout:
      - paper ``citing_2024`` (paper_id=p_citer)  — has 4 outbound refs:
        one resolved to ``cited_2023``, one resolved to ``stale_2022``
        (status=``failed_fetch`` to exercise non-`converted` cited_status),
        one missing arxiv id ``9999.99999``, one unresolvable.
      - paper ``cited_2023``  — referenced by the citer paper AND by the
        post; serves as the inbound target.
      - paper ``stale_2022`` (status=failed_fetch) — second resolved ref.
      - post ``post_2024`` — single outbound ref pointing at ``cited_2023``.
    """
    _seed_domain(conn, "rag")

    # Cited paper — the inbound-target.
    p_cited = _insert_paper(
        conn,
        arxiv_id="2301.12345",
        paper_name="cited_2023",
        title="Cited: Foundational Method",
        abstract="Foundational method.",
        markdown="# Abstract\n\nFoundation.\n",
        domain="rag",
        collection="hierarchical indexing",
        needs_review=0,
        ingested_at="2023-01-15T00:00:00+00:00",
    )

    # Second cited paper, with non-`converted` status to exercise the
    # cited_status passthrough. It still gets a domain/collection so the
    # invariant trigger is satisfied (status='failed_fetch' is exempt
    # from the invariant, but it's harmless to set them).
    p_stale = _insert_paper(
        conn,
        arxiv_id="2202.00002",
        paper_name="stale_2022",
        title="Stale: Failed Fetch",
        abstract="Stub.",
        markdown=None,
        domain="rag",
        collection="hierarchical indexing",
        needs_review=0,
        ingested_at="2022-02-01T00:00:00+00:00",
    )
    conn.execute(
        "UPDATE papers SET status = 'failed_fetch' WHERE id = ?",
        (p_stale,),
    )

    # Citer paper — the outbound subject.
    p_citer = _insert_paper(
        conn,
        arxiv_id="2401.00001",
        paper_name="citing_2024",
        title="Citing: Survey",
        abstract="A survey.",
        markdown="# Abstract\n\nSurvey.\n",
        domain="rag",
        collection="hierarchical indexing",
        needs_review=0,
        ingested_at="2024-01-01T00:00:00+00:00",
    )

    # 4 outbound paper_references on citer.
    _insert_paper_reference(
        conn,
        paper_id=p_citer,
        bibitem_id="bib.bib1",
        ref_number=1,
        raw_text="Cited et al., 2023, arXiv:2301.12345",
        cited_arxiv_id="2301.12345",
        cited_paper_id=p_cited,
    )
    _insert_paper_reference(
        conn,
        paper_id=p_citer,
        bibitem_id="bib.bib2",
        ref_number=2,
        raw_text="Stale et al., 2022, arXiv:2202.00002",
        cited_arxiv_id="2202.00002",
        cited_paper_id=p_stale,
    )
    _insert_paper_reference(
        conn,
        paper_id=p_citer,
        bibitem_id="bib.bib3",
        ref_number=3,
        raw_text="Future et al., 2025, arXiv:9999.99999",
        cited_arxiv_id="9999.99999",
        cited_paper_id=None,
    )
    _insert_paper_reference(
        conn,
        paper_id=p_citer,
        bibitem_id=None,
        ref_number=4,
        raw_text="Smith et al., NeurIPS 2023.",
        cited_arxiv_id=None,
        cited_paper_id=None,
    )

    # Post that cites cited_2023 — exercises inbound union and outbound
    # post path simultaneously.
    p_post = _insert_post_for_citations(
        conn,
        post_name="post_2024",
        title="Blog: Discussion of Cited",
        domain="rag",
        collection="hierarchical indexing",
        date="2024-03-15",
        ingested_at="2024-03-15T00:00:00+00:00",
    )
    _insert_post_reference(
        conn,
        post_id=p_post,
        raw_text="See https://arxiv.org/abs/2301.12345 for the foundation.",
        cited_arxiv_id="2301.12345",
        cited_paper_id=p_cited,
    )

    # Standalone repo so the unsupported_direction-vs-not_found split has
    # a real repo_slug to land on.
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES ('rag', 'standalone', NULL)"
    )
    conn.execute(
        """
        INSERT INTO repos (
            repo_slug, url, host, owner, name, description,
            default_branch, fetched_at, ingested_at,
            domain, collection, status, needs_review,
            file_count, has_readme
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "gh-acme-demo", "https://github.com/acme/demo", "github.com",
            "acme", "demo", "demo repo",
            "main", "2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00",
            "rag", "standalone", "ingested", 0, 0, 1,
        ),
    )

    return conn


class TestModeCitations:
    def test_outbound_paper_buckets_resolved_missing_unresolvable(
        self, seeded_citations_db
    ):
        r = search_mod.mode_citations(
            seeded_citations_db, slug="citing_2024",
        )
        assert r["mode"] == "citations"
        assert r["status"] == "ok"
        assert r["kind"] == "paper"
        assert r["direction"] == "outbound"
        assert r["resolved_count"] == 2
        assert r["missing_count"] == 1
        assert r["unresolvable_count"] == 1
        assert r["total"] == 4
        assert r["truncated"] is False

        resolved_slugs = {row["slug"] for row in r["resolved"]}
        assert resolved_slugs == {"cited_2023", "stale_2022"}
        # Resolved rows carry the citing-side numbering.
        cited_row = next(x for x in r["resolved"] if x["slug"] == "cited_2023")
        assert cited_row["arxiv_id"] == "2301.12345"
        assert cited_row["ref_number"] == 1
        assert cited_row["bibitem_id"] == "bib.bib1"
        assert cited_row["title"].startswith("Cited:")

        missing_row = r["missing"][0]
        assert missing_row["arxiv_id"] == "9999.99999"
        assert missing_row["ref_number"] == 3
        assert "ingest_paper" in missing_row["ingest_hint"]
        assert "9999.99999" in missing_row["ingest_hint"]

        unresolvable_row = r["unresolvable"][0]
        assert unresolvable_row["ref_number"] == 4
        assert "NeurIPS" in unresolvable_row["raw_text"]
        # Unresolvable rows omit arxiv_id / ingest_hint.
        assert "arxiv_id" not in unresolvable_row
        assert "ingest_hint" not in unresolvable_row

    def test_outbound_paper_resolved_row_carries_cited_status(
        self, seeded_citations_db
    ):
        r = search_mod.mode_citations(
            seeded_citations_db, slug="citing_2024",
        )
        by_slug = {row["slug"]: row for row in r["resolved"]}
        assert by_slug["cited_2023"]["cited_status"] == "indexed"
        # `failed_fetch` cited paper passes through verbatim — agent
        # uses this to decide whether the cited source is readable.
        assert by_slug["stale_2022"]["cited_status"] == "failed_fetch"

    def test_outbound_post_omits_ref_number_and_bibitem(
        self, seeded_citations_db
    ):
        r = search_mod.mode_citations(
            seeded_citations_db, slug="post_2024",
        )
        assert r["status"] == "ok"
        assert r["kind"] == "post"
        assert r["direction"] == "outbound"
        assert r["resolved_count"] == 1
        assert r["missing_count"] == 0
        assert r["unresolvable_count"] == 0
        row = r["resolved"][0]
        assert row["slug"] == "cited_2023"
        assert "ref_number" not in row
        assert "bibitem_id" not in row

    def test_outbound_repo_returns_unsupported_direction(
        self, seeded_citations_db
    ):
        r = search_mod.mode_citations(
            seeded_citations_db, slug="gh-acme-demo",
        )
        assert r["status"] == "unsupported_direction"
        assert r["kind"] == "repo"
        assert r["direction"] == "outbound"
        assert "hint" in r

    def test_outbound_truncated_at_500(self, seeded_citations_db):
        # Pile up 501 cheap unresolvable refs on cited_2023 so the cap
        # path fires without dragging in 501 papers.
        cited_id = seeded_citations_db.execute(
            "SELECT id FROM papers WHERE paper_name = 'cited_2023'"
        ).fetchone()[0]
        for n in range(1, 502):
            _insert_paper_reference(
                seeded_citations_db,
                paper_id=cited_id,
                bibitem_id=None,
                ref_number=n,
                raw_text=f"Filler ref {n}",
                cited_arxiv_id=None,
                cited_paper_id=None,
            )
        r = search_mod.mode_citations(
            seeded_citations_db, slug="cited_2023",
        )
        assert r["status"] == "ok"
        assert r["truncated"] is True
        assert r["unresolvable_count"] == 500
        assert r["total"] == 500

    def test_inbound_paper_unions_papers_and_posts_recency_ordered(
        self, seeded_citations_db
    ):
        r = search_mod.mode_citations(
            seeded_citations_db, slug="cited_2023", direction="inbound",
        )
        assert r["status"] == "ok"
        assert r["kind"] == "paper"
        assert r["direction"] == "inbound"
        # Two citers: paper 2024-01-01, post 2024-03-15. Recency-first.
        kinds_in_order = [row["kind"] for row in r["results"]]
        assert kinds_in_order == ["post", "paper"]
        assert r["results"][0]["slug"] == "post_2024"
        assert r["results"][1]["slug"] == "citing_2024"
        # Paper rows carry ref_number; post rows do not.
        assert "ref_number" not in r["results"][0]
        assert r["results"][1]["ref_number"] == 1
        assert r["total_hits"] == 2
        assert r["has_more"] is False

    def test_inbound_paginates_with_limit_offset_has_more(
        self, seeded_citations_db
    ):
        r1 = search_mod.mode_citations(
            seeded_citations_db, slug="cited_2023",
            direction="inbound", limit=1, offset=0,
        )
        assert len(r1["results"]) == 1
        assert r1["results"][0]["slug"] == "post_2024"
        assert r1["total_hits"] == 2
        assert r1["has_more"] is True

        r2 = search_mod.mode_citations(
            seeded_citations_db, slug="cited_2023",
            direction="inbound", limit=1, offset=1,
        )
        assert len(r2["results"]) == 1
        assert r2["results"][0]["slug"] == "citing_2024"
        assert r2["has_more"] is False

    def test_inbound_post_returns_unsupported_direction(
        self, seeded_citations_db
    ):
        r = search_mod.mode_citations(
            seeded_citations_db, slug="post_2024", direction="inbound",
        )
        assert r["status"] == "unsupported_direction"
        assert r["kind"] == "post"
        assert r["direction"] == "inbound"

    def test_inbound_repo_returns_unsupported_direction(
        self, seeded_citations_db
    ):
        r = search_mod.mode_citations(
            seeded_citations_db, slug="gh-acme-demo", direction="inbound",
        )
        assert r["status"] == "unsupported_direction"
        assert r["kind"] == "repo"
        assert r["direction"] == "inbound"

    def test_unknown_slug_returns_not_found(self, seeded_citations_db):
        r = search_mod.mode_citations(
            seeded_citations_db, slug="no_such_slug",
        )
        assert r["status"] == "not_found"
        assert r["slug"] == "no_such_slug"

    def test_outbound_with_zero_references_returns_empty_buckets_status_ok(
        self, seeded_citations_db
    ):
        # cited_2023 has no outbound references in the fixture.
        r = search_mod.mode_citations(
            seeded_citations_db, slug="cited_2023",
        )
        assert r["status"] == "ok"
        assert r["kind"] == "paper"
        assert r["resolved"] == []
        assert r["missing"] == []
        assert r["unresolvable"] == []
        assert r["total"] == 0
        assert r["truncated"] is False

    def test_invalid_direction_raises_value_error(self, seeded_citations_db):
        with pytest.raises(ValueError):
            search_mod.mode_citations(
                seeded_citations_db,
                slug="citing_2024",
                direction="sideways",
            )
