"""Lodestone search CLI — the single entry point for all retrieval.

Five modes dispatched by argparse:

1. **BM25** — positional ``QUERY`` runs against ``abstracts`` (default) or
   ``sections`` (``--sections``). Enriches each hit with an entity preview,
   figure count, and topics.
2. **Taxonomy lookup** — ``--entity`` / ``--topic`` / bare ``--collection``
   (without a positional query) resolves a term against ``terms_fts`` with
   a vec0 KNN fallback at cosine ≥ 0.80.
3. **Browse** — ``--collections`` / ``--topics`` / ``--entity-type`` /
   ``--aliases`` / ``--needs-review`` pure SQL list queries.
4. **ToC** — ``--toc PAPER`` parses level-1..3 ATX headers from the stored
   markdown (skipping fenced code blocks).
5. **Content extraction** — ``--read PAPER [--section S]`` emits markdown
   (optionally sliced by :func:`find_hierarchical_section`);
   ``--figure PAPER N`` / ``--page PAPER N`` write the BLOB to a
   :func:`tempfile.mkstemp` path and return the path.

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
from pathlib import Path
from typing import Any

# NB: import only cheap stdlib + the cheap internal modules here. Anything
# that pulls torch / sentence_transformers / gliner must live inside the
# function that needs it.
from _system.db.connection import get_conn
from _system.utils.logging import get_logger
from _system.utils.sections import find_hierarchical_section, split_sections

_LOG = get_logger("scripts.search")

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Cosine threshold for Tier-B vec0 KNN fallback in taxonomy lookup.
# Looser than the resolver's 0.85 because this is a discovery tool — the
# user is typing a surface form, not a curated alias.
_TAXONOMY_VEC_MIN_COSINE = 0.80

# BM25 enrichment size cap. Keeping each follow-up query small bounds the
# JSON payload size even on queries that return many hits.
_ENTITY_PREVIEW_LIMIT = 5


# ---------------------------------------------------------------------------
# Mode 1 — BM25
# ---------------------------------------------------------------------------


def mode_bm25(
    conn: sqlite3.Connection,
    *,
    query: str,
    scope: str,
    filters: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    """BM25 text search against ``abstracts`` or ``sections``.

    ``scope`` must be ``"abstracts"`` or ``"sections"``. Returns a dict with
    ``mode`` = scope and a ``results`` list. ``abstracts`` returns paper-level
    hits; ``sections`` groups hits by paper_name and preserves the underlying
    row count in ``hit_count``.
    """
    if scope not in ("abstracts", "sections"):
        raise ValueError(f"invalid BM25 scope: {scope!r}")

    domain = filters.get("domain")
    collection = filters.get("collection")

    if scope == "abstracts":
        return _bm25_abstracts(conn, query=query, domain=domain,
                               collection=collection, limit=limit)
    return _bm25_sections(conn, query=query, domain=domain,
                          collection=collection, limit=limit)


def _bm25_abstracts(
    conn: sqlite3.Connection,
    *,
    query: str,
    domain: str | None,
    collection: str | None,
    limit: int,
) -> dict[str, Any]:
    # abstracts columns: (paper_id, domain, paper_name, collection, title, body)
    # column index for 'body' (where the match usually lives) is 5.
    sql = (
        "SELECT paper_id, domain, paper_name, collection, title, "
        "       snippet(abstracts, 5, '[', ']', '…', 10) AS snip "
        "  FROM abstracts "
        " WHERE abstracts MATCH ? "
    )
    params: list[Any] = [query]
    if domain:
        sql += " AND domain = ? "
        params.append(domain)
    if collection:
        sql += " AND collection = ? "
        params.append(collection)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    results: list[dict[str, Any]] = []
    for paper_id, dom, paper_name, coll, title, snip in rows:
        results.append({
            "paper_name": paper_name,
            "domain": dom,
            "collection": coll,
            "title": title,
            "topics": _topics_for_paper(conn, paper_id),
            "entities_preview": _entities_preview(conn, paper_id),
            "figures": _figures_preview(conn, paper_id),
            "snippet": snip,
        })
    return {"mode": "abstracts", "query": query, "results": results}


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
    sql = (
        "SELECT s.paper_id, s.domain, s.paper_name, s.section_title, "
        "       s.section_level, "
        "       snippet(sections, 5, '[', ']', '…', 10) AS snip "
        "  FROM sections s"
    )
    wheres = ["sections MATCH ?"]
    params: list[Any] = [query]
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

    grouped: dict[str, dict[str, Any]] = {}
    for paper_id, dom, paper_name, section_title, section_level, snip in rows:
        group = grouped.get(paper_name)
        if group is None:
            group = {
                "paper_name": paper_name,
                "domain": dom,
                "hit_count": 0,
                "sections": [],
                "topics": _topics_for_paper(conn, paper_id),
                "entities_preview": _entities_preview(conn, paper_id),
                "figures": _figures_preview(conn, paper_id),
            }
            grouped[paper_name] = group
        group["hit_count"] += 1
        group["sections"].append({
            "section_title": section_title,
            "section_level": section_level,
            "snippet": snip,
        })

    return {
        "mode": "sections",
        "query": query,
        "results": list(grouped.values()),
    }


def _topics_for_paper(conn: sqlite3.Connection, paper_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT topic FROM paper_topics WHERE paper_id = ? ORDER BY topic",
        (paper_id,),
    ).fetchall()
    return [r[0] for r in rows]


def _entities_preview(conn: sqlite3.Connection, paper_id: int) -> list[dict[str, str]]:
    rows = conn.execute(
        "SELECT entity_name, entity_type FROM entities "
        " WHERE paper_id = ? "
        " ORDER BY entity_type, entity_name "
        " LIMIT ?",
        (paper_id, _ENTITY_PREVIEW_LIMIT),
    ).fetchall()
    return [{"name": name, "type": etype} for name, etype in rows]


def _figures_preview(conn: sqlite3.Connection, paper_id: int) -> dict[str, Any]:
    row = conn.execute(
        "SELECT COUNT(*), "
        "       (SELECT caption FROM figures "
        "         WHERE paper_id = ? "
        "         ORDER BY figure_number LIMIT 1) "
        "  FROM figures "
        " WHERE paper_id = ?",
        (paper_id, paper_id),
    ).fetchone()
    count, first_caption = row if row else (0, None)
    return {"count": int(count or 0), "first_caption": first_caption}


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
    a ``term_embeddings`` KNN fallback (Tier B). ``kind`` ∈
    {``entity``, ``topic``, ``collection``}.

    On miss returns ``{"mode": kind, "term": term, "error": "term not found"}``
    — does NOT raise.
    """
    if kind not in ("entity", "topic", "collection"):
        raise ValueError(f"invalid taxonomy kind: {kind!r}")

    domain = filters.get("domain")

    # ------- Tier A: terms_fts MATCH with inline metadata predicates ------
    hit = _taxonomy_tier_a(conn, term=term, kind=kind, domain=domain)
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
        knn_params: list[Any] = [qvec, kind]
        # Topic / collection canonicals store entity_type=''; without this
        # narrowing the KNN can return an entity-typed canonical as a
        # semantic neighbor (cross-kind pollution).
        if kind in ("topic", "collection"):
            knn_sql += " AND te.entity_type = ?"
            knn_params.append("")
        if domain:
            knn_sql += " AND te.domain = ?"
            knn_params.append(domain)
        row = conn.execute(knn_sql, knn_params).fetchone()
        if row is None:
            return {"mode": kind, "term": term, "error": "term not found"}
        term_id, distance, dom, term_type, entity_type, canonical_name = row
        # sqlite-vec returns L2 on unit-norm vectors: cos = 1 - d^2/2.
        cosine = 1.0 - (distance * distance) / 2.0
        if cosine < _TAXONOMY_VEC_MIN_COSINE:
            return {"mode": kind, "term": term, "error": "term not found"}
        resolved_via = "vector"

    # ------- Build the result payload ------------------------------------
    aliases_rows = conn.execute(
        "SELECT alias, source_paper FROM term_aliases "
        " WHERE term_id = ? ORDER BY alias",
        (term_id,),
    ).fetchall()
    aliases_payload = [
        {"alias": a, "source_paper": s} for a, s in aliases_rows
    ]

    # For entities, "type" is entity_type; for topic/collection, term_type.
    type_label = entity_type if term_type == "entity" else term_type
    canonical = {
        "name": canonical_name,
        "type": type_label,
        "domain": dom,
    }

    result: dict[str, Any] = {
        "mode": kind,
        "term": term,
        "canonical": canonical,
        "resolved_via": resolved_via,
        "aliases": aliases_payload,
    }

    # Papers per kind
    if kind == "entity":
        prows = conn.execute(
            "SELECT DISTINCT paper_name FROM entities "
            " WHERE entity_name = ? AND domain = ? ORDER BY paper_name",
            (canonical_name, dom),
        ).fetchall()
        papers: list[dict[str, Any]] = []
        for (pn,) in prows:
            srows = conn.execute(
                "SELECT DISTINCT source_breadcrumb FROM entities "
                " WHERE paper_name = ? AND entity_name = ? "
                " ORDER BY source_breadcrumb",
                (pn, canonical_name),
            ).fetchall()
            papers.append({
                "paper_name": pn,
                "sections": [s[0] for s in srows],
            })
        result["papers"] = papers
    elif kind == "topic":
        prows = conn.execute(
            "SELECT DISTINCT p.paper_name "
            "  FROM paper_topics pt "
            "  JOIN papers p ON p.id = pt.paper_id "
            " WHERE pt.topic = ? AND pt.domain = ? ORDER BY p.paper_name",
            (canonical_name, dom),
        ).fetchall()
        result["papers"] = [{"paper_name": r[0]} for r in prows]
    else:  # kind == "collection"
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
    kind: str,
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
    params: list[Any] = [fts_query, kind]
    # Topic / collection canonicals store entity_type=''; without this
    # narrowing, porter-stemmed matches against an entity row's aliases
    # could surface as a topic/collection hit.
    if kind in ("topic", "collection"):
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
    """Pure-SQL list queries. ``which`` selects the view."""
    domain = filters.get("domain")
    if which == "collections":
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
            "mode": "collections",
            "results": [{"collection": r[0], "count": r[1]} for r in rows],
        }

    if which == "topics":
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
            "mode": "topics",
            "results": [{"topic": r[0], "count": r[1]} for r in rows],
        }

    if which == "entity_type":
        entity_type = filters.get("entity_type")
        if not entity_type:
            raise ValueError("mode_browse(which='entity_type') requires "
                             "filters['entity_type']")
        sql = (
            "SELECT entity_name, COUNT(DISTINCT paper_id) AS n "
            "  FROM entities "
            " WHERE entity_type = ? "
        )
        params = [entity_type]
        if domain:
            sql += " AND domain = ? "
            params.append(domain)
        sql += " GROUP BY entity_name ORDER BY n DESC, entity_name"
        rows = conn.execute(sql, params).fetchall()
        return {
            "mode": "entity_type",
            "entity_type": entity_type,
            "results": [
                {"entity_name": r[0], "paper_count": r[1]} for r in rows
            ],
        }

    if which == "aliases":
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
            "mode": "aliases",
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

    if which == "needs_review":
        rows = conn.execute(
            "SELECT paper_name, domain, ingested_at "
            "  FROM papers WHERE needs_review = 1 "
            " ORDER BY ingested_at"
        ).fetchall()
        return {
            "mode": "needs_review",
            "results": [
                {
                    "paper_name": r[0],
                    "domain": r[1],
                    "ingested_at": r[2],
                }
                for r in rows
            ],
        }

    raise ValueError(f"unknown browse view: {which!r}")


# ---------------------------------------------------------------------------
# Mode 4 — ToC
# ---------------------------------------------------------------------------


def mode_toc(conn: sqlite3.Connection, *, paper_name: str) -> dict[str, Any]:
    """Parse level-1..3 ATX headers from ``papers.markdown`` into a nested
    ToC. Headers inside fenced code blocks (``` or ~~~) are skipped."""
    row = conn.execute(
        "SELECT markdown FROM papers WHERE paper_name = ?", (paper_name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")
    markdown = row[0] or ""

    toc: list[dict[str, Any]] = []
    in_fence = False
    for line in markdown.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADER_RE.match(line)
        if m:
            toc.append({"level": len(m.group(1)), "title": m.group(2).strip()})

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
    """Return the full markdown or a hierarchical section slice."""
    row = conn.execute(
        "SELECT markdown FROM papers WHERE paper_name = ?", (paper_name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")
    markdown = row[0] or ""

    if section is None:
        text = markdown
    else:
        text = find_hierarchical_section(markdown, section)
        if text is None:
            raise ValueError(
                f"section not found in paper {paper_name!r}: {section!r}"
            )

    return {
        "mode": "read",
        "paper_name": paper_name,
        "section": section,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Mode 5b — Figure / page BLOB extraction
# ---------------------------------------------------------------------------


def _assert_safe_paper_name(paper: str) -> None:
    if not _SLUG_RE.fullmatch(paper):
        raise ValueError(
            f"paper_name {paper!r} does not match ^[a-z0-9_]+$ — refusing "
            f"to interpolate into a tempfile prefix"
        )


def _safe_n_for_filename(n: Any) -> str:
    """Sanitize the figure / page identifier for use in a tempfile prefix.

    Paper names are already slug-validated, but ``n`` can legitimately be a
    caption label like ``"Figure 3a"`` with spaces. Collapse anything not in
    ``[A-Za-z0-9]`` to an underscore so mkstemp's prefix argument stays well
    behaved across platforms.
    """
    s = str(n)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return cleaned or "x"


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
    _assert_safe_paper_name(paper)

    prow = conn.execute(
        "SELECT id FROM papers WHERE paper_name = ?", (paper,)
    ).fetchone()
    if prow is None:
        raise ValueError(f"paper not found: paper_name={paper!r}")
    paper_id = prow[0]

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
    n_safe = _safe_n_for_filename(n)
    prefix = f"lodestone_{paper}_fig{n_safe}_"
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(image)
    except BaseException:
        # On any error while writing, at least close the fd — os.fdopen
        # would have closed it already on exit, but os.fdopen itself could
        # have raised before wrapping, leaving fd leaked.
        try:
            os.close(fd)
        except OSError:
            pass
        raise

    return {
        "mode": "figure",
        "paper_name": paper,
        "figure_number": n,
        "path": path,
        "mime_type": mime_type or "image/png",
    }


def mode_page(
    conn: sqlite3.Connection,
    *,
    paper: str,
    n: int,
) -> dict[str, Any]:
    """Extract a page image BLOB to a ``tempfile.mkstemp`` path."""
    _assert_safe_paper_name(paper)

    prow = conn.execute(
        "SELECT id FROM papers WHERE paper_name = ?", (paper,)
    ).fetchone()
    if prow is None:
        raise ValueError(f"paper not found: paper_name={paper!r}")
    paper_id = prow[0]

    row = conn.execute(
        "SELECT image FROM page_images WHERE paper_id = ? AND page_number = ?",
        (paper_id, int(n)),
    ).fetchone()
    if row is None:
        raise ValueError(f"page not found: paper={paper!r} page={n!r}")
    image = row[0]

    n_safe = _safe_n_for_filename(n)
    prefix = f"lodestone_{paper}_page{n_safe}_"
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(image)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise

    return {
        "mode": "page",
        "paper_name": paper,
        "page_number": int(n),
        "path": path,
        "mime_type": "image/png",
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

    if mode == "abstracts":
        lines.append(f"== BM25 abstracts: {payload.get('query')!r} ==")
        for hit in payload.get("results", []):
            lines.append(f"- {hit['paper_name']} — {hit.get('title', '')}")
            if hit.get("snippet"):
                lines.append(f"    {hit['snippet']}")
            if hit.get("topics"):
                lines.append(f"    topics: {', '.join(hit['topics'])}")
            if hit.get("entities_preview"):
                ents = ", ".join(
                    f"{e['name']}({e['type']})" for e in hit["entities_preview"]
                )
                lines.append(f"    entities: {ents}")

    elif mode == "sections":
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
            lines.append(f"  {row['entity_name']}  ({row['paper_count']})")

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
        paper = payload.get("paper_name")
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

    elif mode == "page":
        lines.append(
            f"page {payload.get('page_number')} of "
            f"{payload.get('paper_name')} → {payload.get('path')}"
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
            "  1. BM25 (positional QUERY; --sections to search section bodies)\n"
            "  2. Taxonomy lookup (--entity/--topic/--collection without QUERY)\n"
            "  3. Browse (--collections/--topics/--entity-type/--aliases/--needs-review)\n"
            "  4. ToC (--toc PAPER)\n"
            "  5. Read / figure / page (--read / --figure / --page)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", nargs="?", default=None,
                   help="BM25 query string (mode 1)")
    p.add_argument("--sections", action="store_true",
                   help="BM25 against sections instead of abstracts")
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
    p.add_argument("--page", nargs=2, metavar=("PAPER", "N"), default=None,
                   help="extract page image N from PAPER to a tempfile")

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
    exactly one mode must be selected. Mode-1 filters (``--sections``,
    ``--domain``, ``--limit``, and ``--collection`` when QUERY is present)
    are NOT counted as a mode.
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
    if args.page is not None:
        modes.append("--page")

    if len(modes) > 1:
        parser.error(
            f"mutually exclusive modes selected: {', '.join(modes)}. "
            f"Pick exactly one."
        )


def _dispatch(args: argparse.Namespace, conn: sqlite3.Connection) -> dict[str, Any]:
    # Mode 5b: figure / page
    if args.figure is not None:
        paper, n = args.figure
        return mode_figure(conn, paper=paper, n=n)
    if args.page is not None:
        paper, n_str = args.page
        try:
            n_int = int(n_str)
        except ValueError as exc:
            raise ValueError(
                f"--page N must be an integer, got {n_str!r}"
            ) from exc
        return mode_page(conn, paper=paper, n=n_int)

    # Mode 5a: read
    if args.read is not None:
        return mode_read(conn, paper_name=args.read, section=args.section)

    # Mode 4: toc
    if args.toc is not None:
        return mode_toc(conn, paper_name=args.toc)

    # Mode 3: browse
    if args.needs_review:
        return mode_browse(conn, which="needs_review", filters={})
    if args.collections:
        return mode_browse(conn, which="collections",
                           filters={"domain": args.domain})
    if args.topics:
        return mode_browse(conn, which="topics",
                           filters={"domain": args.domain})
    if args.entity_type:
        return mode_browse(
            conn,
            which="entity_type",
            filters={"domain": args.domain, "entity_type": args.entity_type},
        )
    if args.aliases:
        return mode_browse(
            conn,
            which="aliases",
            filters={"aliases_term": args.aliases},
        )

    # Mode 2: taxonomy lookup. --collection is DUAL-USE: it becomes a lookup
    # term only when there is no positional query.
    if args.entity is not None:
        return mode_taxonomy_lookup(
            conn, term=args.entity, kind="entity",
            filters={"domain": args.domain},
        )
    if args.topic is not None:
        return mode_taxonomy_lookup(
            conn, term=args.topic, kind="topic",
            filters={"domain": args.domain},
        )
    if args.collection is not None and args.query is None:
        return mode_taxonomy_lookup(
            conn, term=args.collection, kind="collection",
            filters={"domain": args.domain},
        )

    # Mode 1: BM25
    if args.query is not None:
        scope = "sections" if args.sections else "abstracts"
        filters: dict[str, Any] = {}
        if args.domain:
            filters["domain"] = args.domain
        if args.collection:
            filters["collection"] = args.collection
        return mode_bm25(
            conn,
            query=args.query,
            scope=scope,
            filters=filters,
            limit=args.limit,
        )

    raise SystemExit(
        "no action selected — pass a positional QUERY or one of the mode "
        "flags (--entity/--topic/--collection/--collections/--topics/"
        "--entity-type/--aliases/--needs-review/--toc/--read/--figure/--page). "
        "Run with --help for details."
    )


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

    if args.human:
        sys.stdout.write(to_human(result) + "\n")
    else:
        sys.stdout.write(to_json(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
