"""Unit tests for _system/scripts/taxonomy_tree.py.

Covers the loader (DB → list[DomainNode]) and the renderer (list[DomainNode]
→ tree text) in isolation. The renderer regression for ``classify_paper``'s
prompt format is the index-style snapshot.
"""
from __future__ import annotations

import sqlite3

import pytest

from _system.scripts.taxonomy_tree import (
    CollectionNode,
    DomainNode,
    TaxonomyTreeStyle,
    load_taxonomy,
    render_taxonomy_tree,
)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _seed_domain(conn: sqlite3.Connection, name: str, description: str | None = None) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
        (name, description),
    )


def _seed_collection(
    conn: sqlite3.Connection,
    domain: str,
    name: str,
    description: str | None = None,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES (?, ?, ?)",
        (domain, name, description),
    )


def _seed_paper(
    conn: sqlite3.Connection,
    *,
    paper_name: str,
    domain: str | None,
    collection: str | None,
    arxiv_id: str | None = None,
) -> None:
    aid = arxiv_id or f"arxiv-{paper_name}"
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, html_source, ingested_at, status,
            domain, collection, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            aid, paper_name, "Title", '["A"]', "2024-01-01", "abs",
            f"https://arxiv.org/pdf/{aid}", "arxiv",
            "2024-01-01T00:00:00+00:00", "classified",
            domain, collection, 0,
        ),
    )
    if domain is not None and collection is not None:
        conn.execute(
            "INSERT OR IGNORE INTO collections "
            " (target_kind, target_id, domain, collection, is_primary) "
            " VALUES ('paper', ?, ?, ?, 1)",
            (cur.lastrowid, domain, collection),
        )


# ---------------------------------------------------------------------------
# load_taxonomy
# ---------------------------------------------------------------------------


def test_load_taxonomy_full(conn):
    _seed_domain(conn, "rag", "retrieval augmented generation")
    _seed_domain(conn, "agents", "multi-agent systems")
    _seed_collection(conn, "rag", "hybrid", "dense+sparse fusion")
    _seed_collection(conn, "rag", "hier_indexing", None)
    _seed_collection(conn, "agents", "tool_use", None)

    _seed_paper(conn, paper_name="p1", domain="rag", collection="hybrid")
    _seed_paper(conn, paper_name="p2", domain="rag", collection="hybrid")
    _seed_paper(conn, paper_name="p3", domain="rag", collection="hier_indexing")
    _seed_paper(conn, paper_name="p4", domain="agents", collection="tool_use")

    nodes = load_taxonomy(conn)
    assert isinstance(nodes, list)
    assert all(isinstance(n, DomainNode) for n in nodes)

    # Ordering: rag has 3 papers, agents has 1 → rag first.
    assert nodes[0].name == "rag"
    assert nodes[0].paper_count == 3
    assert nodes[1].name == "agents"
    assert nodes[1].paper_count == 1

    # Collection ordering inside rag: hybrid (2 papers) before hier_indexing (1).
    rag_colls = [c.name for c in nodes[0].collections]
    assert rag_colls == ["hybrid", "hier_indexing"]
    assert nodes[0].collections[0].paper_count == 2
    assert nodes[0].collections[1].paper_count == 1


def test_load_taxonomy_drop_empty_collections(conn):
    _seed_domain(conn, "rag")
    _seed_collection(conn, "rag", "used")
    _seed_collection(conn, "rag", "unused")
    _seed_paper(conn, paper_name="p1", domain="rag", collection="used")

    with_empty = load_taxonomy(conn, include_empty_collections=True)
    names = [c.name for c in with_empty[0].collections]
    assert set(names) == {"used", "unused"}

    without_empty = load_taxonomy(conn, include_empty_collections=False)
    names = [c.name for c in without_empty[0].collections]
    assert names == ["used"]


def test_load_taxonomy_drop_empty_domains(conn):
    _seed_domain(conn, "alive")
    _seed_domain(conn, "dead")
    _seed_collection(conn, "alive", "first")
    _seed_paper(conn, paper_name="p1", domain="alive", collection="first")

    with_empty = load_taxonomy(conn, include_empty_domains=True)
    assert {n.name for n in with_empty} == {"alive", "dead"}

    without_empty = load_taxonomy(conn, include_empty_domains=False)
    assert [n.name for n in without_empty] == ["alive"]


def test_load_taxonomy_truncation(conn):
    _seed_domain(conn, "big")
    for i in range(25):
        _seed_collection(conn, "big", f"c{i:02d}")
        _seed_paper(
            conn, paper_name=f"p{i}", domain="big", collection=f"c{i:02d}"
        )

    nodes = load_taxonomy(conn, collections_per_domain_limit=10)
    assert len(nodes) == 1
    assert len(nodes[0].collections) == 10
    assert nodes[0].overflow == 15


def test_load_taxonomy_domain_filter(conn):
    _seed_domain(conn, "rag")
    _seed_domain(conn, "agents")
    _seed_collection(conn, "rag", "hybrid")
    _seed_collection(conn, "agents", "tool_use")
    _seed_paper(conn, paper_name="p1", domain="rag", collection="hybrid")
    _seed_paper(conn, paper_name="p2", domain="agents", collection="tool_use")

    nodes = load_taxonomy(conn, domain="rag")
    assert len(nodes) == 1
    assert nodes[0].name == "rag"


# ---------------------------------------------------------------------------
# render_taxonomy_tree
# ---------------------------------------------------------------------------


def test_render_tree_count_style():
    nodes = [
        DomainNode(
            name="rag",
            description="retrieval augmented generation",
            paper_count=23,
            collections=(
                CollectionNode(
                    name="hier_indexing",
                    description="multi-level toc",
                    paper_count=5,
                ),
                CollectionNode(name="hybrid", description=None, paper_count=4),
            ),
        ),
        DomainNode(
            name="agents",
            description=None,
            paper_count=8,
            collections=(),
        ),
    ]
    out = render_taxonomy_tree(nodes, style=TaxonomyTreeStyle.COUNT)

    assert "rag — retrieval augmented generation  (23 papers)" in out
    assert "├── hier_indexing — multi-level toc  (5 papers)" in out
    assert "└── hybrid  (4 papers)" in out
    # Empty-collection branch
    assert "agents  (8 papers)" in out
    assert "└── (no collections yet)" in out
    # Blank line between domain blocks.
    assert "\n\nagents" in out


def test_render_tree_index_style_classify_format():
    """Snapshot regression for classify_paper's prompt rendering."""
    nodes = [
        DomainNode(
            name="rag",
            description="retrieval augmented generation",
            paper_count=0,
            collections=(
                CollectionNode(
                    name="hybrid_search",
                    description="dense+sparse retrieval fusion",
                    paper_count=0,
                ),
                CollectionNode(name="rag_systems", description=None, paper_count=0),
            ),
        ),
        DomainNode(
            name="agents",
            description="multi-agent systems",
            paper_count=0,
            collections=(),
        ),
        DomainNode(
            name="theorem_proving",
            description=None,
            paper_count=0,
            collections=(
                CollectionNode(name="saturation_methods", description=None, paper_count=0),
                CollectionNode(name="superposition", description=None, paper_count=0),
            ),
            overflow=4,
        ),
    ]
    out = render_taxonomy_tree(
        nodes,
        style=TaxonomyTreeStyle.INDEX,
        overflow_message="(+ {n} more exist; feel free to propose new)",
    )

    expected = (
        "0. rag\n"
        "   ├── 0: hybrid_search\n"
        "   └── 1: rag_systems\n"
        "1. agents   (no existing collections)\n"
        "2. theorem_proving\n"
        "   ├── 0: saturation_methods\n"
        "   ├── 1: superposition\n"
        "   └── (+ 4 more exist; feel free to propose new)"
    )
    assert out == expected


def test_render_tree_index_style_empty_returns_sentinel():
    out = render_taxonomy_tree([], style=TaxonomyTreeStyle.INDEX)
    assert "taxonomy is empty" in out
    assert "domain_index to -1" in out
    assert "collection_index to -1" in out
