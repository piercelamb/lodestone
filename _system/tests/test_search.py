"""Unit tests for _system/scripts/search.py.

Covers all five search modes (BM25 / taxonomy / browse / ToC / content
extraction), the argparse routing, lazy-import discipline, and the dual
JSON / human output formatters.

The seeded DB fixture builds a small, self-consistent corpus — two papers
with matching ``abstracts`` / ``sections`` / ``terms_fts`` rows, a
canonical ``RAPTOR`` term with aliases + embedding, a figure BLOB keyed
on both ``figure_number`` and ``display_number``, and a page image.
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
# Small valid PNG blob — 1x1 pixel, used for every figure / page image in
# the fixtures so we only need one BLOB constant.
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
    return cur.lastrowid


def _insert_abstract(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    domain: str,
    paper_name: str,
    collection: str | None,
    title: str,
    abstract: str,
) -> None:
    conn.execute(
        """
        INSERT INTO abstracts
            (paper_id, domain, paper_name, collection, title, body)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (paper_id, domain, paper_name, collection, title, abstract),
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
    conn.execute(
        """
        INSERT OR IGNORE INTO entities
            (paper_id, domain, paper_name, entity_name, entity_type,
             source_breadcrumb, description)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_id,
            domain,
            paper_name,
            entity_name,
            entity_type,
            source_breadcrumb,
            description,
        ),
    )


def _insert_paper_topic(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    domain: str,
    topic: str,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
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


def _insert_page_image(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    page_number: int,
    image: bytes,
) -> None:
    conn.execute(
        "INSERT INTO page_images (paper_id, page_number, image) VALUES (?, ?, ?)",
        (paper_id, page_number, image),
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
    _insert_abstract(
        conn,
        paper_id=p1_id,
        domain="rag",
        paper_name="bookrag_2024",
        collection="hierarchical indexing",
        title="BookRAG: Hierarchical Indexing for Retrieval",
        abstract=(
            "We introduce BookRAG, a hierarchical indexing approach for "
            "document retrieval."
        ),
    )
    _insert_sections_for_md(
        conn,
        paper_id=p1_id,
        domain="rag",
        paper_name="bookrag_2024",
        markdown=_PAPER1_MD,
    )

    # Paper 2: stale_2024 with needs_review=1
    p2_id = _insert_paper(
        conn,
        arxiv_id="2402.00002",
        paper_name="stale_2024",
        title="Stale Paper",
        abstract="A paper that needs review.",
        markdown=None,
        domain="other",
        collection=None,
        needs_review=1,
        ingested_at="2024-02-01T00:00:00+00:00",
    )
    _insert_abstract(
        conn,
        paper_id=p2_id,
        domain="other",
        paper_name="stale_2024",
        collection=None,
        title="Stale Paper",
        abstract="A paper that needs review.",
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

    # Figure + page image (used by BLOB extraction tests)
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
    _insert_page_image(conn, paper_id=p1_id, page_number=7, image=_PNG_1x1)

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
    def test_abstracts_returns_paper_hits(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db,
            query="hierarchical indexing",
            scope="abstracts",
            filters={},
            limit=10,
        )
        assert r["mode"] == "abstracts"
        assert r["query"] == "hierarchical indexing"
        assert isinstance(r["results"], list)
        assert any(
            h.get("paper_name") == "bookrag_2024" for h in r["results"]
        ), r["results"]
        hit = next(h for h in r["results"] if h["paper_name"] == "bookrag_2024")
        # Enrichment — topics, entities preview, snippet
        assert "snippet" in hit
        assert "entities_preview" in hit
        assert "topics" in hit

    def test_sections_routes_to_sections_table(self, seeded_db):
        r = search_mod.mode_bm25(
            seeded_db,
            query="BookRAG",
            scope="sections",
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
        # Seed a row in 'other' domain whose abstract mentions BookRAG.
        seeded_db.execute(
            "UPDATE abstracts SET body = ? WHERE paper_name = ?",
            ("BookRAG also mentioned here", "stale_2024"),
        )
        r_all = search_mod.mode_bm25(
            seeded_db, query="BookRAG", scope="abstracts", filters={}, limit=10
        )
        r_rag = search_mod.mode_bm25(
            seeded_db,
            query="BookRAG",
            scope="abstracts",
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
            scope="abstracts",
            filters={},
            limit=10,
        )
        hit = next(h for h in r["results"] if h["paper_name"] == "bookrag_2024")
        assert hit["entities_preview"], "expected entity preview"
        assert "figures" in hit
        assert isinstance(hit["figures"], dict)
        assert hit["figures"]["count"] >= 1


# ===========================================================================
# Mode 2 — Taxonomy
# ===========================================================================


class TestModeTaxonomy:
    def test_finds_via_terms_fts_exact(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, term="RAPTOR", kind="entity", filters={}
        )
        assert r["mode"] == "entity"
        assert r["canonical"]["name"] == "RAPTOR"
        assert r["resolved_via"] in ("exact", "alias", "fts")
        assert r["aliases"], r
        assert r["papers"]
        # The RAPTOR entity in the fixture was flagged in the Introduction
        # section.
        assert any(
            "Introduction" in s
            for paper in r["papers"]
            for s in paper.get("sections", [])
        )

    def test_finds_via_alias(self, seeded_db):
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, term="raptor", kind="entity", filters={}
        )
        assert r["canonical"]["name"] == "RAPTOR"
        assert r["resolved_via"] in ("alias", "fts", "exact")

    def test_reports_not_found(self, seeded_db, monkeypatch):
        """A term that misses Tier A forces Tier B. Stub the Embedder with an
        orthogonal vector so the KNN result's cosine falls below the 0.80
        gate — exercises the "term not found" path without paying torch's
        import cost (which breaks cross-test in the full suite on Py 3.14)."""

        class _StubEmbedder:
            def __init__(self) -> None:
                pass

            def embed(self, text: str) -> list[float]:
                v = [0.0] * 384
                v[50] = 1.0  # orthogonal to every seeded canonical vector
                return v

        import _system.resolution.embeddings as emb_mod

        monkeypatch.setattr(emb_mod, "Embedder", _StubEmbedder)

        r = search_mod.mode_taxonomy_lookup(
            seeded_db, term="totally_made_up_thing", kind="entity", filters={}
        )
        assert r["mode"] == "entity"
        assert r.get("error") == "term not found"

    def test_falls_back_to_vec_knn(self, seeded_db, monkeypatch):
        """Force Tier A miss by making terms_fts return nothing; Tier B
        should hit via vec0 KNN on a stub Embedder whose vector coincides
        exactly with the stored RAPTOR embedding."""

        class _StubEmbedder:
            def __init__(self) -> None:
                pass

            def embed(self, text: str) -> list[float]:
                v = [0.0] * 384
                v[0] = 1.0  # matches RAPTOR's stored embedding exactly
                return v

        # Patch Embedder import inside mode_taxonomy_lookup to return the stub.
        import _system.resolution.embeddings as emb_mod

        monkeypatch.setattr(emb_mod, "Embedder", _StubEmbedder)

        # Pass a term that will miss terms_fts but match the stored vector.
        r = search_mod.mode_taxonomy_lookup(
            seeded_db, term="zzznosuchterm", kind="entity", filters={}
        )
        assert r["canonical"]["name"] == "RAPTOR"
        assert r["resolved_via"] == "vector"

    def test_topic_and_collection_share_fts_path(self, seeded_db):
        r_topic = search_mod.mode_taxonomy_lookup(
            seeded_db, term="entity resolution", kind="topic", filters={}
        )
        assert r_topic["mode"] == "topic"
        assert r_topic["canonical"]["name"] == "entity resolution"
        assert r_topic["papers"]

        r_coll = search_mod.mode_taxonomy_lookup(
            seeded_db,
            term="hierarchical indexing",
            kind="collection",
            filters={},
        )
        assert r_coll["mode"] == "collection"
        assert r_coll["canonical"]["name"] == "hierarchical indexing"
        assert any(p["paper_name"] == "bookrag_2024" for p in r_coll["papers"])


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

    def test_entity_type_list(self, seeded_db):
        r = search_mod.mode_browse(
            seeded_db, which="entity_type", filters={"entity_type": "method"}
        )
        assert r["mode"] == "entity_type"
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
        r = search_mod.mode_toc(seeded_db, paper_name="bookrag_2024")
        assert r["mode"] == "toc"
        assert r["paper_name"] == "bookrag_2024"
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
        r = search_mod.mode_toc(seeded_db, paper_name="bookrag_2024")
        titles = [e["title"] for e in r["toc"]]
        assert "Not A Header" not in titles
        assert "Real Header" in titles
        assert "Another Real Header" in titles

    def test_unknown_paper_raises(self, seeded_db):
        with pytest.raises(ValueError):
            search_mod.mode_toc(seeded_db, paper_name="nope_2099")


# ===========================================================================
# Mode 5a — Read
# ===========================================================================


class TestModeRead:
    def test_full_markdown(self, seeded_db):
        r = search_mod.mode_read(seeded_db, paper_name="bookrag_2024", section=None)
        assert r["mode"] == "read"
        assert r["paper_name"] == "bookrag_2024"
        assert r["section"] is None
        assert "BookRAG" in r["text"]

    def test_section_returns_hierarchical_slice(self, seeded_db):
        r = search_mod.mode_read(
            seeded_db, paper_name="bookrag_2024", section="Method"
        )
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
            paper_name="bookrag_2024",
            section="Experiments > Setup",
        )
        assert "Setup inside Experiments." in r["text"]
        assert "Setup inside Method." not in r["text"]

    def test_missing_section_raises(self, seeded_db):
        with pytest.raises(ValueError):
            search_mod.mode_read(
                seeded_db, paper_name="bookrag_2024", section="NoSuchSection"
            )

    def test_unknown_paper_raises(self, seeded_db):
        with pytest.raises(ValueError):
            search_mod.mode_read(seeded_db, paper_name="nope_2099", section=None)


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


class TestModePage:
    def test_extracts_blob_to_tempfile(self, seeded_db):
        r = search_mod.mode_page(seeded_db, paper="bookrag_2024", n=7)
        assert r["mode"] == "page"
        path = Path(r["path"])
        assert path.exists()
        assert path.name.startswith("lodestone_bookrag_2024_page7_")
        assert path.read_bytes() == _PNG_1x1

    def test_rejects_illegal_paper_name(self, seeded_db):
        with pytest.raises(ValueError):
            search_mod.mode_page(seeded_db, paper="../evil", n=7)

    def test_missing_page_raises(self, seeded_db):
        with pytest.raises(ValueError):
            search_mod.mode_page(seeded_db, paper="bookrag_2024", n=999)


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
            scope="abstracts",
            filters={},
            limit=5,
        )
        assert bm25["mode"] == "abstracts"

        tax = search_mod.mode_taxonomy_lookup(
            seeded_db, term="RAPTOR", kind="entity", filters={}
        )
        assert tax["mode"] == "entity"

        br = search_mod.mode_browse(
            seeded_db, which="needs_review", filters={}
        )
        assert br["mode"] == "needs_review"

        toc = search_mod.mode_toc(seeded_db, paper_name="bookrag_2024")
        assert toc["mode"] == "toc"

        read = search_mod.mode_read(
            seeded_db, paper_name="bookrag_2024", section=None
        )
        assert read["mode"] == "read"

        fig = search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="3")
        assert fig["mode"] == "figure"

        page = search_mod.mode_page(seeded_db, paper="bookrag_2024", n=7)
        assert page["mode"] == "page"

    def test_human_formatter_nonempty_per_mode(self, seeded_db):
        payloads = [
            search_mod.mode_bm25(
                seeded_db, query="BookRAG", scope="abstracts", filters={}, limit=5
            ),
            search_mod.mode_bm25(
                seeded_db, query="BookRAG", scope="sections", filters={}, limit=5
            ),
            search_mod.mode_taxonomy_lookup(
                seeded_db, term="RAPTOR", kind="entity", filters={}
            ),
            search_mod.mode_browse(seeded_db, which="needs_review", filters={}),
            search_mod.mode_browse(
                seeded_db, which="aliases", filters={"aliases_term": "RAPTOR"}
            ),
            search_mod.mode_toc(seeded_db, paper_name="bookrag_2024"),
            search_mod.mode_read(seeded_db, paper_name="bookrag_2024", section=None),
            search_mod.mode_figure(seeded_db, paper="bookrag_2024", n="3"),
            search_mod.mode_page(seeded_db, paper="bookrag_2024", n=7),
        ]
        for payload in payloads:
            out = search_mod.to_human(payload)
            assert isinstance(out, str)
            assert out.strip(), f"empty to_human for payload={payload!r}"
