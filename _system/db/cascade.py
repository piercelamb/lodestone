"""Per-paper cascade-delete helper used by both ``fetch_paper`` (force-refetch
path) and ``ingest`` (``--force`` cascade).

Canonical taxonomy rows (``canonical_terms``, ``term_embeddings``,
``terms_fts``) are not touched — those are cross-paper. ``term_aliases``
rows ARE per-paper (they're the synonym index keyed by ``source_paper``)
and are deleted alongside the rest of the paper's children.
"""
from __future__ import annotations

import sqlite3


def delete_paper_cascade(conn: sqlite3.Connection, *, paper_id: int) -> None:
    """DELETE one paper and every per-paper child row.

    The caller owns the enclosing transaction. Order matters: FK-backed
    children before the papers row (PRAGMA foreign_keys=ON); FTS5 tables
    have no FK cascade, so their rows must be deleted explicitly.
    """
    # paper_references is FK'd both inward (paper_id) and outward
    # (cited_paper_id). When deleting paper P we drop P's own refs AND
    # null any other paper's ref that pointed at P, so a future re-ingest
    # of P (or a different paper with the same arxiv_id) can re-resolve
    # without an FK violation.
    conn.execute(
        "UPDATE paper_references SET cited_paper_id = NULL "
        "WHERE cited_paper_id = ?",
        (paper_id,),
    )
    conn.execute("DELETE FROM paper_references WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM figures      WHERE paper_id = ?", (paper_id,))
    # term_aliases keys by paper_name (TEXT), not paper_id, so look up
    # the name first. Wipes entity, topic, AND collection alias rows for
    # this paper — the per-paper concepts they record are about to vanish.
    conn.execute(
        """
        DELETE FROM term_aliases
         WHERE source_paper = (SELECT paper_name FROM papers WHERE id = ?)
        """,
        (paper_id,),
    )
    conn.execute("DELETE FROM paper_topics WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM abstracts    WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM sections     WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM papers       WHERE id       = ?", (paper_id,))
