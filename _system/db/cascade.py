"""Per-paper cascade-delete helper used by both ``fetch_paper`` (force-refetch
path) and ``ingest`` (``--force`` cascade).

Per-paper rows (figures, sections, paper_topics, term_aliases, ...) are
removed alongside the ``papers`` row. Canonical taxonomy rows are touched
**only via orphan-GC** at the end of the cascade: any topic canonical
with zero remaining bindings is removed alongside its satellites in
``terms_fts``, ``term_embeddings``, and ``term_aliases``. Domains and
collections are curated categories — they survive the deletion of their
last paper so future papers can fill them; only humans delete those.
Entity canonicals are never GC'd — under the synonym-index regime,
tier-1 mentions leave no per-paper trace, so substantiation can't be
proven.
"""
from __future__ import annotations

import sqlite3

from _system.db.orphan_gc import gc_orphan_topic_canonicals


def delete_paper_cascade(conn: sqlite3.Connection, *, paper_id: int) -> None:
    """DELETE one paper and every per-paper child row.

    The caller owns the enclosing transaction. Order matters: FK-backed
    children before the papers row (PRAGMA foreign_keys=ON); FTS5 tables
    have no FK cascade, so their rows must be deleted explicitly. Orphan
    topic canonicals are GC'd at the end, after the paper is gone, when
    "zero remaining bindings" is a clean truth. Collections survive —
    they're curated categories, not per-paper concepts.
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
    conn.execute("DELETE FROM sections     WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM code_files   WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM readmes_fts  WHERE paper_id = ?", (paper_id,))
    conn.execute("DELETE FROM papers       WHERE id       = ?", (paper_id,))
    gc_orphan_topic_canonicals(conn)
