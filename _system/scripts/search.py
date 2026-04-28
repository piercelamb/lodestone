"""Lodestone search CLI — the single entry point for all retrieval.

Five modes dispatched by argparse:

1. **BM25** — positional ``QUERY`` runs against the ``sections`` FTS5 table
   (paper abstracts ride along as the ``# Abstract`` chunk). Hits are
   grouped by ``paper_name`` and enriched with an entity preview, figure
   count, and topics. User input is sanitized into a quoted-token phrase
   query so punctuation (``-``, ``/``, ``:``, parens, etc.) cannot escape
   into FTS5 operator territory.
2. **Taxonomy lookup** — ``--entity`` / ``--topic`` / bare ``--collection``
   (without a positional query) resolves a term against ``terms_fts`` with
   a vec0 KNN fallback at cosine ≥ 0.80.
3. **Browse** — ``--collections`` / ``--topics`` / ``--entity-type`` /
   ``--aliases`` / ``--needs-review`` pure SQL list queries.
4. **ToC** — ``--toc PAPER`` parses level-1..3 ATX headers from the stored
   markdown (skipping fenced code blocks).
5. **Content extraction** — ``--read PAPER [--section S]`` emits markdown
   (optionally sliced by :func:`find_hierarchical_section`);
   ``--figure PAPER N`` writes the figure BLOB to a
   :func:`tempfile.mkstemp` path and returns the path.

JSON is emitted to stdout by default. ``--human`` renders a short plaintext
per mode. All logging goes to stderr via the shared :mod:`_system.utils.logging`
logger. ``--help`` must finish in under 300 ms, so no ML library
(``sentence_transformers`` / ``gliner2`` / ``torch``) may be imported at
module scope — the sole caller (``mode_taxonomy_lookup`` Tier B) lazy-imports
the :class:`Embedder`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Any

# NB: import only cheap stdlib + the cheap internal modules here. Anything
# that pulls torch / sentence_transformers / gliner must live inside the
# function that needs it.
from _system.db.connection import get_conn
from _system.utils.logging import get_logger
from _system.utils.sections import (
    SectionQueryError,
    find_hierarchical_section,
    split_sections,
)
from _system.utils.slug import _SLUG_RE

_LOG = get_logger("scripts.search")

# Cosine threshold for Tier-B vec0 KNN fallback in taxonomy lookup.
# Looser than the resolver's 0.85 because this is a discovery tool — the
# user is typing a surface form, not a curated alias.
_TAXONOMY_VEC_MIN_COSINE = 0.80

# BM25 enrichment size cap. Keeping each follow-up query small bounds the
# JSON payload size even on queries that return many hits.
_ENTITY_PREVIEW_LIMIT = 5


class TaxonomyKind(StrEnum):
    ENTITY = "entity"
    TOPIC = "topic"
    COLLECTION = "collection"


class BrowseView(StrEnum):
    COLLECTIONS = "collections"
    TOPICS = "topics"
    ENTITY_TYPE = "entity_type"
    ALIASES = "aliases"
    NEEDS_REVIEW = "needs_review"


# ---------------------------------------------------------------------------
# Mode 1 — BM25
# ---------------------------------------------------------------------------


# Matches a single Unicode word character — used to drop tokens that hold no
# searchable content (e.g. a bare ``-`` or ``/``) after the per-token quote
# wrap. Without this filter, FTS5 raises a syntax error on the empty phrase.
_FTS5_TOKEN_HAS_WORD_CHAR = re.compile(r"\w", re.UNICODE)


class FTS5QueryError(ValueError):
    """The user's BM25 query has no searchable tokens after sanitization."""


def _sanitize_fts5_match(query: str) -> str:
    """Wrap each whitespace-delimited token in double quotes for safe MATCH.

    The FTS5 query parser treats ``-``, ``+``, ``:``, ``(``, ``)``, ``*`` as
    syntactic operators, so a bare ``tree-sitter`` parses as ``tree NOT sitter``
    with ``sitter`` read as a column name (the failure mode that motivated this
    helper). Wrapping each token in ``"..."`` reduces it to a phrase query
    whose contents are tokenized but never reinterpreted as operators —
    ``"tree-sitter"`` becomes a 2-token phrase requiring ``tree`` adjacent to
    ``sitter``.

    Internal ``"`` characters are escaped by doubling. Tokens that carry no
    word characters (e.g. a stray ``--``) are dropped; if every token drops
    out, raises :class:`FTS5QueryError` so the caller can surface a clean
    error instead of a SQLite syntax exception.
    """
    parts: list[str] = []
    for tok in query.split():
        if not _FTS5_TOKEN_HAS_WORD_CHAR.search(tok):
            continue
        parts.append('"' + tok.replace('"', '""') + '"')
    if not parts:
        raise FTS5QueryError(
            f"query has no searchable tokens after sanitization: {query!r}"
        )
    return " ".join(parts)


def mode_bm25(
    conn: sqlite3.Connection,
    *,
    query: str,
    filters: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    """BM25 text search against the ``sections`` FTS5 table.

    Returns a dict with ``mode='sections'`` and a ``results`` list grouped by
    ``paper_name``. Each group preserves the underlying row count in
    ``hit_count``. The paper's abstract is indexed as the ``# Abstract``
    chunk inside ``sections``, so a query that previously hit the dropped
    ``abstracts`` virtual table still surfaces those hits here — under the
    paper's ``Abstract`` section title.

    A query whose tokens all drop out under sanitization (e.g. pure
    punctuation like ``---``) emits ``status='empty_query'`` — agent-
    recoverable, not a crash.
    """
    domain = filters.get("domain")
    collection = filters.get("collection")
    try:
        return _bm25_sections(conn, query=query, domain=domain,
                              collection=collection, limit=limit)
    except FTS5QueryError as e:
        return {
            "mode": "sections",
            "status": "empty_query",
            "query": query,
            "error": str(e),
            "hint": (
                "BM25 query had no searchable tokens after sanitization. "
                "Pass at least one word; punctuation-only queries match "
                "nothing."
            ),
        }


def _bm25_sections(
    conn: sqlite3.Connection,
    *,
    query: str,
    domain: str | None,
    collection: str | None,
    limit: int,
) -> dict[str, Any]:
    # sections columns: (paper_id, domain, paper_name, section_title, section_level, body)
    # snippet() against 'body' = column index 5.
    match_query = _sanitize_fts5_match(query)
    sql = (
        "SELECT s.paper_id, s.domain, s.paper_name, s.section_title, "
        "       s.section_level, "
        "       snippet(sections, 5, '[', ']', '…', 10) AS snip "
        "  FROM sections s"
    )
    wheres = ["sections MATCH ?"]
    params: list[Any] = [match_query]
    if domain:
        wheres.append("s.domain = ?")
        params.append(domain)
    if collection:
        # Join papers for the collection filter — sections does not carry
        # collection in FTS5 (by design; the paper owns the collection).
        sql += " JOIN papers p ON p.id = s.paper_id"
        wheres.append("p.collection = ?")
        params.append(collection)
    sql += " WHERE " + " AND ".join(wheres)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()

    # One bucket walk to build groups; enrichment then batched across all
    # distinct paper_ids in three queries regardless of result count.
    grouped: dict[str, dict[str, Any]] = {}
    pid_order: list[int] = []
    for paper_id, dom, paper_name, section_title, section_level, snip in rows:
        group = grouped.get(paper_name)
        if group is None:
            group = {
                "paper_name": paper_name,
                "domain": dom,
                "_paper_id": paper_id,
                "hit_count": 0,
                "sections": [],
            }
            grouped[paper_name] = group
            pid_order.append(paper_id)
        group["hit_count"] += 1
        group["sections"].append({
            "section_title": section_title,
            "section_level": section_level,
            "snippet": snip,
        })

    topics_by_pid = _topics_batch(conn, pid_order)
    entities_by_pid = _entities_preview_batch(conn, pid_order)
    figures_by_pid = _figures_preview_batch(conn, pid_order)
    for group in grouped.values():
        pid = group.pop("_paper_id")
        group["topics"] = topics_by_pid.get(pid, [])
        group["entities_preview"] = entities_by_pid.get(pid, [])
        group["figures"] = figures_by_pid.get(
            pid, {"count": 0, "first_caption": None}
        )

    return {
        "mode": "sections",
        "query": query,
        "results": list(grouped.values()),
    }


def _topics_batch(
    conn: sqlite3.Connection, paper_ids: list[int]
) -> dict[int, list[str]]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"SELECT paper_id, topic FROM paper_topics "
        f" WHERE paper_id IN ({placeholders}) "
        f" ORDER BY paper_id, topic",
        paper_ids,
    ).fetchall()
    result: dict[int, list[str]] = {pid: [] for pid in paper_ids}
    for pid, topic in rows:
        result[pid].append(topic)
    return result


def _entities_preview_batch(
    conn: sqlite3.Connection, paper_ids: list[int]
) -> dict[int, list[dict[str, str]]]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    # term_aliases is a synonym index keyed by paper_name; join through
    # papers to map back to paper_id, then ROW_NUMBER caps the per-paper
    # preview at _ENTITY_PREVIEW_LIMIT in a single pass. Caveat: the
    # preview shows entities whose synonyms appeared in this paper, so
    # canonicals that only ever surface as their canonical name (no
    # synonym row) are missed. The BM25 hit's snippet is the
    # authoritative signal — this preview is just a hint.
    rows = conn.execute(
        f"""
        SELECT paper_id, canonical_name, entity_type FROM (
            SELECT paper_id, canonical_name, entity_type,
                   ROW_NUMBER() OVER (
                       PARTITION BY paper_id
                       ORDER BY entity_type, canonical_name
                   ) AS rn
              FROM (
                  SELECT p.id AS paper_id,
                         ct.canonical_name,
                         ct.entity_type
                    FROM papers p
                    JOIN term_aliases ta ON ta.source_paper = p.paper_name
                    JOIN canonical_terms ct ON ct.id = ta.term_id
                   WHERE p.id IN ({placeholders})
                     AND ct.term_type = 'entity'
                   GROUP BY p.id, ct.id
              )
        )
         WHERE rn <= ?
         ORDER BY paper_id, entity_type, canonical_name
        """,
        (*paper_ids, _ENTITY_PREVIEW_LIMIT),
    ).fetchall()
    result: dict[int, list[dict[str, str]]] = {pid: [] for pid in paper_ids}
    for pid, name, etype in rows:
        result[pid].append({"name": name, "type": etype})
    return result


def _figures_preview_batch(
    conn: sqlite3.Connection, paper_ids: list[int]
) -> dict[int, dict[str, Any]]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"""
        SELECT paper_id, cnt, caption FROM (
            SELECT paper_id,
                   COUNT(*) OVER (PARTITION BY paper_id) AS cnt,
                   caption,
                   ROW_NUMBER() OVER (
                       PARTITION BY paper_id ORDER BY figure_number
                   ) AS rn
              FROM figures
             WHERE paper_id IN ({placeholders})
        )
         WHERE rn = 1
        """,
        paper_ids,
    ).fetchall()
    result: dict[int, dict[str, Any]] = {
        pid: {"count": 0, "first_caption": None} for pid in paper_ids
    }
    for pid, cnt, caption in rows:
        result[pid] = {"count": int(cnt or 0), "first_caption": caption}
    return result


# ---------------------------------------------------------------------------
# Mode 2 — Taxonomy lookup
# ---------------------------------------------------------------------------


def mode_taxonomy_lookup(
    conn: sqlite3.Connection,
    *,
    term: str,
    kind: str,
    filters: dict[str, Any],
) -> dict[str, Any]:
    """Resolve ``term`` to a canonical row via ``terms_fts`` (Tier A) with
    a ``term_embeddings`` KNN fallback (Tier B). ``kind`` ∈ :class:`TaxonomyKind`.

    On miss returns ``{"mode": kind, "term": term, "error": "term not found"}``
    — does NOT raise.
    """
    try:
        kind_enum = TaxonomyKind(kind)
    except ValueError as exc:
        raise ValueError(f"invalid taxonomy kind: {kind!r}") from exc

    domain = filters.get("domain")

    # ------- Tier A: terms_fts MATCH with inline metadata predicates ------
    hit = _taxonomy_tier_a(conn, term=term, kind=kind_enum, domain=domain)
    resolved_via: str | None = None
    if hit is not None:
        term_id, dom, term_type, entity_type, canonical_name = hit
        resolved_via = _classify_tier_a_match(
            conn, term=term, term_id=term_id, canonical_name=canonical_name
        )

    # ------- Tier B: vec0 KNN. Lazy Embedder import. ---------------------
    if hit is None:
        # Lazy import: keeps `search.py --help` under 300 ms when Tier B
        # never fires.
        from _system.resolution.embeddings import Embedder  # noqa: PLC0415

        import sqlite_vec  # noqa: PLC0415

        embedder = Embedder()
        qvec = sqlite_vec.serialize_float32(embedder.embed(term))
        knn_sql = (
            "SELECT te.term_id, te.distance, ct.domain, ct.term_type, "
            "       ct.entity_type, ct.canonical_name "
            "  FROM term_embeddings te "
            "  JOIN canonical_terms ct ON ct.id = te.term_id "
            " WHERE te.embedding MATCH ? "
            "   AND te.term_type = ? "
            "   AND te.k = 1"
        )
        knn_params: list[Any] = [qvec, kind_enum.value]
        # Topic / collection canonicals store entity_type=''; without this
        # narrowing the KNN can return an entity-typed canonical as a
        # semantic neighbor (cross-kind pollution).
        if kind_enum in (TaxonomyKind.TOPIC, TaxonomyKind.COLLECTION):
            knn_sql += " AND te.entity_type = ?"
            knn_params.append("")
        if domain:
            knn_sql += " AND te.domain = ?"
            knn_params.append(domain)
        row = conn.execute(knn_sql, knn_params).fetchone()
        if row is None:
            return {"mode": kind_enum, "term": term, "error": "term not found"}
        term_id, distance, dom, term_type, entity_type, canonical_name = row
        # sqlite-vec returns L2 on unit-norm vectors: cos = 1 - d^2/2.
        cosine = 1.0 - (distance * distance) / 2.0
        if cosine < _TAXONOMY_VEC_MIN_COSINE:
            return {"mode": kind_enum, "term": term, "error": "term not found"}
        resolved_via = "vector"

    # ------- Build the result payload ------------------------------------
    # term_aliases is a synonym index: every row carries a non-canonical
    # surface form. SELECT DISTINCT collapses the same alias across
    # multiple papers so the keyword set is the union, with the source
    # papers kept on each row for provenance.
    aliases_rows = conn.execute(
        "SELECT DISTINCT alias, source_paper FROM term_aliases "
        " WHERE term_id = ? ORDER BY alias, source_paper",
        (term_id,),
    ).fetchall()
    aliases_payload = [
        {"alias": a, "source_paper": s} for a, s in aliases_rows
    ]

    # For entities, "type" is entity_type; for topic/collection, term_type.
    type_label = entity_type if term_type == TaxonomyKind.ENTITY else term_type
    canonical = {
        "name": canonical_name,
        "type": type_label,
        "domain": dom,
    }

    result: dict[str, Any] = {
        "mode": kind_enum,
        "term": term,
        "canonical": canonical,
        "resolved_via": resolved_via,
        "aliases": aliases_payload,
    }

    # Papers per kind
    if kind_enum is TaxonomyKind.ENTITY:
        # No per-paper sections payload under the synonym-index regime —
        # claude follows up with a positional BM25 query (e.g.
        # ``"alias1 alias2"``) to locate hit positions; sections is the
        # only BM25 path. Papers list is the union of
        # papers that contributed any synonym for this term; tier-1-only
        # mentions (canonical-as-surface-form) leave no row and so no
        # paper here, which is fine — the BM25 query catches them via
        # the canonical name as one of the OR terms.
        prows = conn.execute(
            "SELECT DISTINCT source_paper FROM term_aliases "
            " WHERE term_id = ? ORDER BY source_paper",
            (term_id,),
        ).fetchall()
        result["papers"] = [{"paper_name": r[0]} for r in prows]
    elif kind_enum is TaxonomyKind.TOPIC:
        prows = conn.execute(
            "SELECT DISTINCT p.paper_name "
            "  FROM paper_topics pt "
            "  JOIN papers p ON p.id = pt.paper_id "
            " WHERE pt.topic = ? AND pt.domain = ? ORDER BY p.paper_name",
            (canonical_name, dom),
        ).fetchall()
        result["papers"] = [{"paper_name": r[0]} for r in prows]
    else:  # TaxonomyKind.COLLECTION
        prows = conn.execute(
            "SELECT paper_name FROM papers "
            " WHERE collection = ? AND domain = ? ORDER BY paper_name",
            (canonical_name, dom),
        ).fetchall()
        result["papers"] = [{"paper_name": r[0]} for r in prows]

    return result


def _taxonomy_tier_a(
    conn: sqlite3.Connection,
    *,
    term: str,
    kind: TaxonomyKind,
    domain: str | None,
) -> tuple[int, str, str, str, str] | None:
    """Tier A: porter-stemmed match against ``terms_fts`` narrowed by
    ``term_type`` / ``domain``.

    Returns ``(term_id, domain, term_type, entity_type, canonical_name)`` or
    ``None``.
    """
    # FTS5 treats the MATCH expression as a query. The user's term may
    # include phrase-breakers (spaces); quote it as a phrase to keep the
    # porter tokenizer's stemming while avoiding column-scoped / operator
    # parsing of punctuation.
    fts_query = '"' + term.replace('"', '""') + '"'

    # ``term_type`` / ``domain`` are indexed (non-UNINDEXED) columns in
    # terms_fts — FTS5 accepts them as auxiliary WHERE predicates applied
    # after the MATCH.
    sql = (
        "SELECT term_id, domain, term_type, entity_type, canonical_name "
        "  FROM terms_fts "
        " WHERE terms_fts MATCH ? "
        "   AND term_type = ? "
    )
    params: list[Any] = [fts_query, kind.value]
    # Topic / collection canonicals store entity_type=''; without this
    # narrowing, porter-stemmed matches against an entity row's aliases
    # could surface as a topic/collection hit.
    if kind in (TaxonomyKind.TOPIC, TaxonomyKind.COLLECTION):
        sql += " AND entity_type = ? "
        params.append("")
    if domain:
        sql += " AND domain = ? "
        params.append(domain)
    sql += " ORDER BY rank LIMIT 1"
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.OperationalError as exc:
        # Degenerate MATCH queries (all punctuation, empty after quoting,
        # etc.) raise "fts5: syntax error near ...". Treat as a miss so
        # Tier B can take over. Any other OperationalError (missing table,
        # disk IO, corruption) is a real failure — re-raise.
        if "fts5" not in str(exc).lower():
            raise
        _LOG.warning(
            "terms_fts MATCH syntax error for term=%r; falling through to "
            "Tier B KNN: %s",
            term, exc,
        )
        return None
    return row


def _classify_tier_a_match(
    conn: sqlite3.Connection,
    *,
    term: str,
    term_id: int,
    canonical_name: str,
) -> str:
    """``exact`` if ``term == canonical_name``, ``alias`` if the term is a
    stored alias, else ``fts`` (the porter stemmer matched via stem-fold)."""
    if term == canonical_name:
        return "exact"
    alias_row = conn.execute(
        "SELECT 1 FROM term_aliases WHERE term_id = ? AND alias = ? LIMIT 1",
        (term_id, term),
    ).fetchone()
    if alias_row is not None:
        return "alias"
    return "fts"


# ---------------------------------------------------------------------------
# Mode 3 — Browse
# ---------------------------------------------------------------------------


def mode_browse(
    conn: sqlite3.Connection,
    *,
    which: str,
    filters: dict[str, Any],
) -> dict[str, Any]:
    """Pure-SQL list queries. ``which`` selects the view (a :class:`BrowseView`)."""
    try:
        view = BrowseView(which)
    except ValueError as exc:
        raise ValueError(f"unknown browse view: {which!r}") from exc
    domain = filters.get("domain")
    if view is BrowseView.COLLECTIONS:
        sql = (
            "SELECT collection, COUNT(*) AS n FROM papers "
            " WHERE collection IS NOT NULL "
        )
        params: list[Any] = []
        if domain:
            sql += " AND domain = ? "
            params.append(domain)
        sql += " GROUP BY collection ORDER BY n DESC, collection"
        rows = conn.execute(sql, params).fetchall()
        return {
            "mode": view,
            "results": [{"collection": r[0], "count": r[1]} for r in rows],
        }

    if view is BrowseView.TOPICS:
        sql = (
            "SELECT topic, COUNT(DISTINCT paper_id) AS n FROM paper_topics "
        )
        params = []
        if domain:
            sql += " WHERE domain = ? "
            params.append(domain)
        sql += " GROUP BY topic ORDER BY n DESC, topic"
        rows = conn.execute(sql, params).fetchall()
        return {
            "mode": view,
            "results": [{"topic": r[0], "count": r[1]} for r in rows],
        }

    if view is BrowseView.ENTITY_TYPE:
        entity_type = filters.get("entity_type")
        if not entity_type:
            raise ValueError("mode_browse(which='entity_type') requires "
                             "filters['entity_type']")
        # Pure list of canonicals in the type. Per-paper drilldown
        # happens via `--entity NAME` once the user picks one.
        sql = (
            "SELECT canonical_name FROM canonical_terms "
            " WHERE term_type = 'entity' "
            "   AND entity_type = ? "
        )
        params = [entity_type]
        if domain:
            sql += " AND domain = ? "
            params.append(domain)
        sql += " ORDER BY canonical_name"
        rows = conn.execute(sql, params).fetchall()
        return {
            "mode": view,
            "entity_type": entity_type,
            "results": [{"entity_name": r[0]} for r in rows],
        }

    if view is BrowseView.ALIASES:
        term = filters.get("aliases_term")
        if not term:
            raise ValueError("mode_browse(which='aliases') requires "
                             "filters['aliases_term']")
        # The CLI exposes a single string — if ambiguous across scopes, a
        # future caller can narrow with --domain. For now we return aliases
        # across all matching canonical rows.
        rows = conn.execute(
            "SELECT ta.alias, ta.source_paper, ta.match_tier "
            "  FROM term_aliases ta "
            "  JOIN canonical_terms ct ON ct.id = ta.term_id "
            " WHERE ct.canonical_name = ? "
            " ORDER BY ta.source_paper, ta.alias",
            (term,),
        ).fetchall()
        return {
            "mode": view,
            "term": term,
            "results": [
                {
                    "alias": r[0],
                    "source_paper": r[1],
                    "match_tier": r[2],
                }
                for r in rows
            ],
        }

    # BrowseView.NEEDS_REVIEW
    rows = conn.execute(
        "SELECT paper_name, domain, ingested_at "
        "  FROM papers WHERE needs_review = 1 "
        " ORDER BY ingested_at"
    ).fetchall()
    return {
        "mode": view,
        "results": [
            {
                "paper_name": r[0],
                "domain": r[1],
                "ingested_at": r[2],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Mode 4 — ToC
# ---------------------------------------------------------------------------


def mode_toc(conn: sqlite3.Connection, *, paper_name: str) -> dict[str, Any]:
    """Flatten ``papers.markdown`` into level-1..3 headers via the shared
    :func:`split_sections` walker — keeps ToC boundaries in lockstep with
    how the indexer chunked the paper, including fenced-code suppression.
    """
    row = conn.execute(
        "SELECT markdown FROM papers WHERE paper_name = ?", (paper_name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")
    markdown = row[0] or ""

    toc = [
        {"level": chunk.level, "title": chunk.title}
        for chunk in split_sections(markdown)
    ]
    return {"mode": "toc", "paper_name": paper_name, "toc": toc}


# ---------------------------------------------------------------------------
# Mode 5a — Read
# ---------------------------------------------------------------------------


def mode_read(
    conn: sqlite3.Connection,
    *,
    paper_name: str,
    section: str | None,
) -> dict[str, Any]:
    """Return the full markdown or a hierarchical section slice.

    Failures of the section slice are NOT raised — they're emitted as
    structured payloads so the agent can recover. ``status`` is one of:

    - missing: ``"section_not_found"`` — well-formed query, no matching
      header in this paper. Payload includes the actual top-level section
      titles + a hint pointing at ``--toc`` and the whole-paper fallback.
    - malformed: ``"malformed_section_query"`` — Claude's query violates
      breadcrumb syntax. Payload echoes the bad query and the rule message.

    A non-existent paper still raises ``ValueError`` — that's a hard error
    (the caller picked the wrong arg), not a recoverable agent miss.
    """
    row = conn.execute(
        "SELECT markdown FROM papers WHERE paper_name = ?", (paper_name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")
    markdown = row[0] or ""

    if section is None:
        return {
            "mode": "read",
            "status": "ok",
            "paper_name": paper_name,
            "section": None,
            "text": markdown,
        }

    try:
        text = find_hierarchical_section(markdown, section)
    except SectionQueryError as e:
        return {
            "mode": "read",
            "status": "malformed_section_query",
            "paper_name": paper_name,
            "requested_section": section,
            "error": str(e),
            "hint": (
                "Use a single title like 'Method' or a breadcrumb like "
                "'Parent > Child' with non-empty segments separated by ' > '."
            ),
        }

    if text is None:
        available = [
            c.title for c in split_sections(markdown) if c.level <= 2
        ]
        return {
            "mode": "read",
            "status": "section_not_found",
            "paper_name": paper_name,
            "requested_section": section,
            "available_top_level_sections": available,
            "hint": (
                f"Run --toc {paper_name} for the full hierarchy, or drop "
                f"--section to read the whole paper."
            ),
        }

    return {
        "mode": "read",
        "status": "ok",
        "paper_name": paper_name,
        "section": section,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Mode 5b — Figure BLOB extraction
# ---------------------------------------------------------------------------


def _assert_safe_paper_name(paper: str) -> None:
    if not _SLUG_RE.fullmatch(paper):
        raise ValueError(
            f"paper_name {paper!r} does not match ^[a-z0-9_]+$ — refusing "
            f"to interpolate into a tempfile prefix"
        )


def _safe_n_for_filename(n: Any) -> str:
    """Sanitize the figure identifier for use in a tempfile prefix.

    Paper names are already slug-validated, but ``n`` can legitimately be a
    caption label like ``"Figure 3a"`` with spaces. Collapse anything not in
    ``[A-Za-z0-9]`` to an underscore so mkstemp's prefix argument stays well
    behaved across platforms.
    """
    s = str(n)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return cleaned or "x"


def _lookup_paper_id(conn: sqlite3.Connection, paper: str) -> int:
    _assert_safe_paper_name(paper)
    prow = conn.execute(
        "SELECT id FROM papers WHERE paper_name = ?", (paper,)
    ).fetchone()
    if prow is None:
        raise ValueError(f"paper not found: paper_name={paper!r}")
    return prow[0]


def _write_blob_tempfile(image: bytes, *, prefix: str, suffix: str = ".png") -> str:
    """Atomically spill ``image`` bytes to a new mkstemp path and return it.

    The BaseException guard covers the narrow window where ``os.fdopen``
    raises after mkstemp handed us the fd but before the ``with`` block
    would have closed it.
    """
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(image)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return path


def mode_figure(
    conn: sqlite3.Connection,
    *,
    paper: str,
    n: str,
) -> dict[str, Any]:
    """Extract a figure BLOB to a ``tempfile.mkstemp`` path.

    ``n`` is tried first as an integer against ``figures.figure_number``
    (DOM ordinal), then as a string against ``figures.display_number``
    (caption label like ``"Figure 3a"``). Raises :class:`ValueError` if
    neither lookup hits.
    """
    paper_id = _lookup_paper_id(conn, paper)

    frow = None
    try:
        fn = int(n)
    except (TypeError, ValueError):
        fn = None
    if fn is not None:
        frow = conn.execute(
            "SELECT image, mime_type FROM figures "
            " WHERE paper_id = ? AND figure_number = ?",
            (paper_id, fn),
        ).fetchone()
    if frow is None:
        frow = conn.execute(
            "SELECT image, mime_type FROM figures "
            " WHERE paper_id = ? AND display_number = ?",
            (paper_id, str(n)),
        ).fetchone()
    if frow is None:
        raise ValueError(
            f"figure not found: paper={paper!r} n={n!r}"
        )

    image, mime_type = frow
    path = _write_blob_tempfile(
        image, prefix=f"lodestone_{paper}_fig{_safe_n_for_filename(n)}_"
    )
    return {
        "mode": "figure",
        "paper_name": paper,
        "figure_number": n,
        "path": path,
        "mime_type": mime_type or "image/png",
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def to_json(payload: dict[str, Any]) -> str:
    """Serialize ``payload`` to a single JSON string with UTF-8 passthrough."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


def to_human(payload: dict[str, Any]) -> str:
    """Short plaintext rendering per mode. Designed for terminal eyeballing,
    not programmatic reuse — pipelines should consume ``to_json``.
    """
    mode = payload.get("mode", "?")
    lines: list[str] = []

    if mode == "sections":
        if payload.get("status") == "empty_query":
            lines.append(
                f"empty BM25 query: {payload.get('query')!r}"
            )
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        else:
            lines.append(f"== BM25 sections: {payload.get('query')!r} ==")
            for hit in payload.get("results", []):
                lines.append(
                    f"- {hit['paper_name']} (hits={hit['hit_count']})"
                )
                for s in hit.get("sections", []):
                    lines.append(
                        f"    §{s['section_level']} {s['section_title']}: {s.get('snippet', '')}"
                    )

    elif mode in ("entity", "topic", "collection"):
        if payload.get("error"):
            lines.append(f"== {mode}: {payload.get('term')!r} — {payload['error']} ==")
        else:
            can = payload.get("canonical", {})
            lines.append(
                f"== {mode}: {payload.get('term')!r} → "
                f"{can.get('name')} ({can.get('type')}, {can.get('domain')}) "
                f"[{payload.get('resolved_via')}] =="
            )
            if payload.get("aliases"):
                lines.append("aliases:")
                for a in payload["aliases"]:
                    lines.append(f"  - {a['alias']} ({a['source_paper']})")
            if payload.get("papers"):
                lines.append("papers:")
                for p in payload["papers"]:
                    if "sections" in p:
                        lines.append(
                            f"  - {p['paper_name']}: {', '.join(p['sections'])}"
                        )
                    else:
                        lines.append(f"  - {p['paper_name']}")

    elif mode == "collections":
        lines.append("== collections ==")
        for row in payload.get("results", []):
            lines.append(f"  {row['collection']}  ({row['count']})")

    elif mode == "topics":
        lines.append("== topics ==")
        for row in payload.get("results", []):
            lines.append(f"  {row['topic']}  ({row['count']})")

    elif mode == "entity_type":
        lines.append(f"== entity_type: {payload.get('entity_type')!r} ==")
        for row in payload.get("results", []):
            lines.append(f"  {row['entity_name']}")

    elif mode == "aliases":
        lines.append(f"== aliases of {payload.get('term')!r} ==")
        for row in payload.get("results", []):
            lines.append(
                f"  {row['alias']}  (from {row['source_paper']}, "
                f"tier={row['match_tier']})"
            )

    elif mode == "needs_review":
        lines.append("== papers flagged for review ==")
        for row in payload.get("results", []):
            lines.append(
                f"  {row['paper_name']} [{row.get('domain')}] "
                f"ingested={row.get('ingested_at')}"
            )

    elif mode == "toc":
        lines.append(f"== ToC {payload.get('paper_name')} ==")
        for entry in payload.get("toc", []):
            indent = "  " * (entry["level"] - 1)
            lines.append(f"{indent}{'#' * entry['level']} {entry['title']}")

    elif mode == "read":
        status = payload.get("status", "ok")
        paper = payload.get("paper_name")
        if status == "section_not_found":
            lines.append(
                f"section not found in {paper}: "
                f"{payload.get('requested_section')!r}"
            )
            avail = payload.get("available_top_level_sections") or []
            if avail:
                lines.append(f"available top-level sections: {', '.join(avail)}")
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif status == "malformed_section_query":
            lines.append(
                f"malformed --section query for {paper}: "
                f"{payload.get('requested_section')!r}"
            )
            err = payload.get("error")
            if err:
                lines.append(err)
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        else:
            sec = payload.get("section")
            header = f"== {paper}"
            if sec:
                header += f" § {sec}"
            header += " =="
            lines.append(header)
            lines.append(payload.get("text", ""))

    elif mode == "figure":
        lines.append(
            f"figure {payload.get('figure_number')} of "
            f"{payload.get('paper_name')} → {payload.get('path')} "
            f"({payload.get('mime_type')})"
        )

    else:
        lines.append(f"(unknown mode: {mode!r})")
        lines.append(json.dumps(payload, indent=2))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / dispatch
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="search.py",
        description=(
            "Lodestone search CLI. Five modes routed via argparse:\n"
            "  1. BM25 (positional QUERY; searches the sections FTS5 index)\n"
            "  2. Taxonomy lookup (--entity/--topic/--collection without QUERY)\n"
            "  3. Browse (--collections/--topics/--entity-type/--aliases/--needs-review)\n"
            "  4. ToC (--toc PAPER)\n"
            "  5. Read / figure (--read / --figure)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", nargs="?", default=None,
                   help="BM25 query string (mode 1)")
    p.add_argument("--domain", default=None, help="filter by papers.domain")
    p.add_argument("--collection", default=None,
                   help="collection name — filter when QUERY is set, "
                        "otherwise Mode 2 lookup term")
    p.add_argument("--limit", type=int, default=10,
                   help="max BM25 hits (default: 10)")

    p.add_argument("--entity", default=None, help="taxonomy lookup: entity")
    p.add_argument("--topic", default=None, help="taxonomy lookup: topic")

    p.add_argument("--collections", action="store_true",
                   help="browse: list distinct collections")
    p.add_argument("--topics", action="store_true",
                   help="browse: list distinct topics")
    p.add_argument("--entity-type", dest="entity_type", default=None,
                   help="browse: list entity_names of this type")
    p.add_argument("--aliases", default=None,
                   help="browse: list aliases of a canonical_name")
    p.add_argument("--needs-review", dest="needs_review",
                   action="store_true", help="browse: papers flagged for review")

    p.add_argument("--toc", default=None, help="ToC of PAPER (paper_name)")

    p.add_argument("--read", default=None,
                   help="read full markdown of PAPER (paper_name)")
    p.add_argument("--section", default=None,
                   help="when --read is set, slice to this section "
                        "(supports 'Parent > Child' breadcrumb)")

    p.add_argument("--figure", nargs=2, metavar=("PAPER", "N"), default=None,
                   help="extract figure N from PAPER to a tempfile")

    p.add_argument("--human", action="store_true",
                   help="emit plaintext instead of JSON")
    p.add_argument("--db", default="lodestone.db", help="sqlite db path")
    return p


def _check_mode_conflicts(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject mutually exclusive mode flag combinations up front.

    The plan's routing table is "first match wins," but silent first-match
    resolution hides user mistakes (``QUERY --entity FOO`` would drop the
    positional query without warning). We enforce exclusivity explicitly:
    exactly one mode must be selected. Mode-1 filters (``--domain``,
    ``--limit``, and ``--collection`` when QUERY is present) are NOT
    counted as a mode.
    """
    modes: list[str] = []
    if args.query is not None:
        modes.append("QUERY (Mode 1 BM25)")
    # Mode 2 lookups — --collection is dual-use: only counts as Mode 2 when
    # there is no positional query.
    if args.entity is not None:
        modes.append("--entity")
    if args.topic is not None:
        modes.append("--topic")
    if args.collection is not None and args.query is None:
        modes.append("--collection (lookup)")
    # Mode 3 browse flags
    if args.collections:
        modes.append("--collections")
    if args.topics:
        modes.append("--topics")
    if args.entity_type:
        modes.append("--entity-type")
    if args.aliases:
        modes.append("--aliases")
    if args.needs_review:
        modes.append("--needs-review")
    # Mode 4 / 5
    if args.toc is not None:
        modes.append("--toc")
    if args.read is not None:
        modes.append("--read")
    if args.figure is not None:
        modes.append("--figure")

    if len(modes) > 1:
        parser.error(
            f"mutually exclusive modes selected: {', '.join(modes)}. "
            f"Pick exactly one."
        )


def _dispatch(args: argparse.Namespace, conn: sqlite3.Connection) -> dict[str, Any]:
    # Mode 5b: figure
    if args.figure is not None:
        paper, n = args.figure
        return mode_figure(conn, paper=paper, n=n)

    # Mode 5a: read
    if args.read is not None:
        return mode_read(conn, paper_name=args.read, section=args.section)

    # Mode 4: toc
    if args.toc is not None:
        return mode_toc(conn, paper_name=args.toc)

    # Mode 3: browse
    domain_filter: dict[str, Any] = {"domain": args.domain}
    if args.needs_review:
        return mode_browse(conn, which=BrowseView.NEEDS_REVIEW, filters={})
    if args.collections:
        return mode_browse(conn, which=BrowseView.COLLECTIONS, filters=domain_filter)
    if args.topics:
        return mode_browse(conn, which=BrowseView.TOPICS, filters=domain_filter)
    if args.entity_type:
        return mode_browse(
            conn,
            which=BrowseView.ENTITY_TYPE,
            filters={**domain_filter, "entity_type": args.entity_type},
        )
    if args.aliases:
        return mode_browse(
            conn,
            which=BrowseView.ALIASES,
            filters={"aliases_term": args.aliases},
        )

    # Mode 2: taxonomy lookup. --collection is DUAL-USE — it becomes a lookup
    # term only when there is no positional query. Table-driven so the three
    # near-identical branches share one call site.
    for attr, kind in (
        ("entity", TaxonomyKind.ENTITY),
        ("topic", TaxonomyKind.TOPIC),
        ("collection", TaxonomyKind.COLLECTION),
    ):
        value = getattr(args, attr)
        if value is None:
            continue
        if attr == "collection" and args.query is not None:
            continue
        return mode_taxonomy_lookup(
            conn, term=value, kind=kind, filters=domain_filter,
        )

    # Mode 1: BM25 — sections is the only path now (the legacy abstracts
    # FTS5 index was dropped; the abstract rides along as the # Abstract
    # chunk inside sections).
    if args.query is not None:
        filters: dict[str, Any] = {}
        if args.domain:
            filters["domain"] = args.domain
        if args.collection:
            filters["collection"] = args.collection
        return mode_bm25(
            conn,
            query=args.query,
            filters=filters,
            limit=args.limit,
        )

    raise SystemExit(
        "no action selected — pass a positional QUERY or one of the mode "
        "flags (--entity/--topic/--collection/--collections/--topics/"
        "--entity-type/--aliases/--needs-review/--toc/--read/--figure). "
        "Run with --help for details."
    )


_SOFT_FAILURE_STATUSES = frozenset({
    "section_not_found",
    "malformed_section_query",
    "empty_query",
})


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _check_mode_conflicts(parser, args)

    # DB connect is deferred to AFTER parse_args so `--help` never touches
    # the filesystem — argparse's auto-help short-circuits before we land
    # here.
    conn = get_conn(Path(args.db))
    try:
        result = _dispatch(args, conn)
    finally:
        conn.close()

    # Soft failures: agent-recoverable misses (e.g. --section not found,
    # malformed breadcrumb). Payload still carries the diagnostic; exit
    # code 2 signals "you asked for something that didn't work" without
    # the caller having to parse the JSON to find out.
    is_soft_failure = result.get("status") in _SOFT_FAILURE_STATUSES

    if args.human:
        # Keep stdout empty on soft failure so shell pipes don't see partial
        # output. The diagnostic still goes to the user via stderr.
        if is_soft_failure:
            sys.stderr.write(to_human(result) + "\n")
        else:
            sys.stdout.write(to_human(result) + "\n")
    else:
        # JSON mode: payload always lands on stdout (the agent reads it to
        # recover), regardless of success/failure.
        sys.stdout.write(to_json(result) + "\n")

    return 2 if is_soft_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
