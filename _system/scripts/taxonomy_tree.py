"""Shared taxonomy loader + tree renderer.

Single source of truth for two consumers:

* ``classify_paper`` — primes the classification LLM prompt with a
  numbered tree (``style=INDEX``); needs empty collections kept (so the
  LLM can pick a registered-but-unused collection) and a per-domain cap
  to bound prompt size.
* ``mode_overview`` (search.py) — Claude's top-down corpus map
  (``style=COUNT``); drops empty rows so the tree reflects actual
  content, no truncation, and surfaces paper counts plus an
  ``uncategorized`` count for papers with ``collection IS NULL``.

The two consumers run the same SQL (one LEFT JOIN per level) and the
same renderer; only the policy toggles differ.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterable


class TaxonomyTreeStyle(StrEnum):
    INDEX = "index"   # numbered prefixes for LLM index-replace selection
    COUNT = "count"   # paper-count annotations for human/agent orientation


@dataclass(frozen=True)
class CollectionNode:
    name: str
    description: str | None
    paper_count: int
    repo_count: int = 0
    post_count: int = 0


@dataclass(frozen=True)
class DomainNode:
    name: str
    description: str | None
    paper_count: int
    collections: tuple[CollectionNode, ...]
    overflow: int = 0
    repo_count: int = 0
    post_count: int = 0


def load_taxonomy(
    conn: sqlite3.Connection,
    *,
    domain: str | None = None,
    include_empty_collections: bool = True,
    include_empty_domains: bool = True,
    collections_per_domain_limit: int | None = None,
) -> list[DomainNode]:
    """Load the full taxonomy tree from the DB as ``list[DomainNode]``.

    ``include_empty_collections=False`` drops collections with zero papers
    in their domain — used by ``mode_overview`` so the tree reflects
    content. ``include_empty_domains=False`` drops domains whose
    ``paper_count`` sums to zero — same rationale.
    ``collections_per_domain_limit`` truncates each domain's collection
    list (preserving popularity order), recording the count of hidden
    rows in ``DomainNode.overflow`` — used by ``classify_paper`` to bound
    prompt size. ``None`` means no cap.

    SQL is a single LEFT JOIN per level; empty-row filtering and
    truncation are Python-side so the same source data can be sliced
    different ways by different callers.
    """
    if domain is not None:
        domain_rows = conn.execute(
            """
            SELECT d.name, d.description, COUNT(p.id) AS paper_count
              FROM domains d
              LEFT JOIN papers p ON p.domain = d.name
             WHERE d.name = ?
             GROUP BY d.name, d.description
            """,
            (domain,),
        ).fetchall()
    else:
        domain_rows = conn.execute(
            """
            SELECT d.name, d.description, COUNT(p.id) AS paper_count
              FROM domains d
              LEFT JOIN papers p ON p.domain = d.name
             GROUP BY d.name, d.description
            """,
        ).fetchall()

    # The polymorphic `collections` junction covers papers, posts, and
    # repos — one LEFT JOIN per (domain, collection) row in the catalog
    # naturally aggregates across all kinds. SUM(CASE) keeps the
    # paper-vs-repo breakdown the renderer surfaces. Posts contribute a
    # separate `post_count` aggregate the renderer doesn't display today
    # but consumers can read off `CollectionNode.post_count`.
    if domain is not None:
        coll_rows = conn.execute(
            """
            SELECT c.domain, c.name, c.description,
                   SUM(CASE WHEN pc.target_kind='paper' THEN 1 ELSE 0 END) AS paper_count,
                   SUM(CASE WHEN pc.target_kind='post'  THEN 1 ELSE 0 END) AS post_count,
                   SUM(CASE WHEN pc.target_kind='repo'  THEN 1 ELSE 0 END) AS repo_count
              FROM collection_definitions c
              LEFT JOIN collections pc
                ON pc.domain = c.domain AND pc.collection = c.name
             WHERE c.domain = ?
             GROUP BY c.domain, c.name, c.description
            """,
            (domain,),
        ).fetchall()
    else:
        coll_rows = conn.execute(
            """
            SELECT c.domain, c.name, c.description,
                   SUM(CASE WHEN pc.target_kind='paper' THEN 1 ELSE 0 END) AS paper_count,
                   SUM(CASE WHEN pc.target_kind='post'  THEN 1 ELSE 0 END) AS post_count,
                   SUM(CASE WHEN pc.target_kind='repo'  THEN 1 ELSE 0 END) AS repo_count
              FROM collection_definitions c
              LEFT JOIN collections pc
                ON pc.domain = c.domain AND pc.collection = c.name
             GROUP BY c.domain, c.name, c.description
            """,
        ).fetchall()

    by_domain: dict[str, list[CollectionNode]] = {}
    for d, name, description, p_count, post_count, r_count in coll_rows:
        p_count = int(p_count or 0)
        post_count = int(post_count or 0)
        r_count = int(r_count or 0)
        # Drop entirely empty collections (no entries of any kind).
        if not include_empty_collections and (p_count + post_count + r_count) == 0:
            continue
        by_domain.setdefault(d, []).append(
            CollectionNode(
                name=name,
                description=description,
                paper_count=p_count,
                repo_count=r_count,
                post_count=post_count,
            )
        )

    # Sort each domain's collections: most-popular first across all
    # kinds, then alpha by name.
    for d, colls in by_domain.items():
        colls.sort(
            key=lambda c: (
                -(c.paper_count + c.post_count + c.repo_count),
                c.name,
            )
        )

    # Per-domain repo count (standalone repos only — paper-linked repos
    # are already counted via their paper).
    repo_counts_by_domain: dict[str, int] = {
        d: c
        for d, c in conn.execute(
            "SELECT domain, COUNT(*) FROM repos "
            " WHERE paper_id IS NULL AND domain IS NOT NULL "
            " GROUP BY domain"
        ).fetchall()
    }

    # Per-domain post count.
    post_counts_by_domain: dict[str, int] = {
        d: c
        for d, c in conn.execute(
            "SELECT domain, COUNT(*) FROM posts "
            " WHERE domain IS NOT NULL "
            " GROUP BY domain"
        ).fetchall()
    }

    nodes: list[DomainNode] = []
    for d_name, d_desc, d_count in domain_rows:
        d_count = int(d_count or 0)
        d_repo_count = int(repo_counts_by_domain.get(d_name, 0) or 0)
        d_post_count = int(post_counts_by_domain.get(d_name, 0) or 0)
        if (
            not include_empty_domains
            and d_count == 0 and d_repo_count == 0 and d_post_count == 0
        ):
            continue
        colls = by_domain.get(d_name, [])
        overflow = 0
        if (
            collections_per_domain_limit is not None
            and len(colls) > collections_per_domain_limit
        ):
            overflow = len(colls) - collections_per_domain_limit
            colls = colls[:collections_per_domain_limit]
        nodes.append(
            DomainNode(
                name=d_name,
                description=d_desc,
                paper_count=d_count,
                collections=tuple(colls),
                overflow=overflow,
                repo_count=d_repo_count,
                post_count=d_post_count,
            )
        )

    # Sort domains: most-content first across all kinds, then alpha.
    nodes.sort(
        key=lambda n: (
            -(n.paper_count + n.post_count + n.repo_count),
            n.name,
        )
    )
    return nodes


def _format_count(n: int, *, label: str) -> str:
    return f"{n} {label}" if n == 1 else f"{n} {label}s"


def _node_count_label(
    paper_count: int, repo_count: int, post_count: int = 0
) -> str:
    """Render the ``(N papers, M posts, K repos)`` suffix.

    Drops the post / repo halves when their counts are zero so existing
    fixtures that don't exercise those kinds render unchanged.
    """
    parts: list[str] = [_format_count(paper_count, label="paper")]
    if post_count > 0:
        parts.append(_format_count(post_count, label="post"))
    if repo_count > 0:
        parts.append(_format_count(repo_count, label="repo"))
    return ", ".join(parts)


def render_taxonomy_tree(
    domains: list[DomainNode],
    *,
    style: TaxonomyTreeStyle = TaxonomyTreeStyle.COUNT,
    overflow_message: str = "(+ {n} more)",
) -> str:
    """Render a list of ``DomainNode`` as tree text.

    ``style=INDEX`` — classify_paper format (names only; descriptions
    are intentionally omitted — the classifier picks by index, and
    descriptions blow up prompt size without changing the choice)::

        0. rag
           ├── 0: hybrid_search
           └── 1: rag_systems

    ``style=COUNT`` — overview format::

        rag — desc  (23 papers, 4 uncategorized)
        ├── hybrid_search — desc  (8 papers)
        └── rag_systems  (3 papers)

    The overflow leaf renders as ``└── (+ N more...)`` in both styles
    via ``overflow_message`` (``{n}`` is replaced with the hidden count).
    """
    if not domains:
        if style is TaxonomyTreeStyle.INDEX:
            return (
                "(taxonomy is empty — propose a new domain by setting "
                "domain_index to -1, and a new collection under it by "
                "setting collection_index to -1)"
            )
        return "(no domains)"

    if style is TaxonomyTreeStyle.INDEX:
        return _render_index(domains, overflow_message=overflow_message)
    return _render_count(domains, overflow_message=overflow_message)


def _render_index(
    domains: list[DomainNode], *, overflow_message: str
) -> str:
    lines: list[str] = []
    for i, node in enumerate(domains):
        head = f"{i}. {node.name}"
        if not node.collections:
            lines.append(f"{head}   (no existing collections)")
            continue

        lines.append(head)
        has_overflow = node.overflow > 0
        n_leaves = len(node.collections) + (1 if has_overflow else 0)
        for j, coll in enumerate(node.collections):
            connector = "└──" if j == n_leaves - 1 else "├──"
            lines.append(f"   {connector} {j}: {coll.name}")
        if has_overflow:
            tail = overflow_message.format(n=node.overflow)
            lines.append(f"   └── {tail}")
    return "\n".join(lines)


def _render_count(
    domains: list[DomainNode], *, overflow_message: str
) -> str:
    blocks: list[list[str]] = []
    for node in domains:
        block: list[str] = []
        suffix = (
            f"({_node_count_label(node.paper_count, node.repo_count, node.post_count)})"
        )
        head = (
            f"{node.name} — {node.description}  {suffix}"
            if node.description
            else f"{node.name}  {suffix}"
        )
        block.append(head)

        if not node.collections:
            block.append("└── (no collections yet)")
            blocks.append(block)
            continue

        has_overflow = node.overflow > 0
        n_leaves = len(node.collections) + (1 if has_overflow else 0)
        for j, coll in enumerate(node.collections):
            connector = "└──" if j == n_leaves - 1 else "├──"
            c_count = _node_count_label(
                coll.paper_count, coll.repo_count, coll.post_count
            )
            leaf = (
                f"{coll.name} — {coll.description}  ({c_count})"
                if coll.description
                else f"{coll.name}  ({c_count})"
            )
            block.append(f"{connector} {leaf}")
        if has_overflow:
            tail = overflow_message.format(n=node.overflow)
            block.append(f"└── {tail}")
        blocks.append(block)

    # Blank line between domains.
    return "\n\n".join("\n".join(b) for b in blocks)
