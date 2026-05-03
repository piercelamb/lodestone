"""Orphan GC for topic and collection canonicals.

Topic and collection canonicals can become orphaned in two situations:

1. Re-classifying a paper. ``classify_paper`` deletes the paper's
   ``paper_topics`` rows up front and re-runs the LLM. If the second run
   emits different topic phrasings, the first run's topic canonicals are
   left in ``canonical_terms`` with zero remaining ``paper_topics``
   references.
2. Deleting a paper. The per-paper cascade deletes ``paper_topics`` and
   resets ``papers.collection`` but leaves the canonical taxonomy alone.
   A topic or collection that only appeared in the deleted paper becomes
   orphaned.

Unlike entities (which have no per-paper binding under the synonym-index
regime), topic and collection canonicals have complete bindings:
``paper_topics`` for topics and ``papers.collection`` for collections. So
orphan detection is exact and cheap.

Entities are deliberately out of scope.
"""
from __future__ import annotations

import sqlite3


def gc_orphan_topic_collection_canonicals(
    conn: sqlite3.Connection,
) -> dict[str, int]:
    """Delete orphan topic/collection canonicals plus their satellites.

    Removes ``canonical_terms`` rows of ``term_type IN ('topic',
    'collection')`` that have zero remaining bindings, plus their
    matching rows in ``terms_fts``, ``term_embeddings``,
    ``term_aliases``, and (for collections) the first-class
    ``collections`` registry.

    Caller owns the enclosing transaction.

    Returns a counts dict::

        {"topics": N, "collections": M, "collections_registry": K}
    """
    orphan_topic_ids = [
        row[0]
        for row in conn.execute(
            """
            SELECT id FROM canonical_terms ct
             WHERE term_type = 'topic'
               AND NOT EXISTS (
                   SELECT 1 FROM paper_topics pt
                    WHERE pt.topic = ct.canonical_name
                      AND pt.domain = ct.domain
               )
            """
        )
    ]

    orphan_collections = conn.execute(
        """
        SELECT id, domain, canonical_name FROM canonical_terms ct
         WHERE term_type = 'collection'
           AND NOT EXISTS (
               SELECT 1 FROM papers p
                WHERE p.collection = ct.canonical_name
                  AND p.domain = ct.domain
           )
        """
    ).fetchall()
    orphan_collection_ids = [row[0] for row in orphan_collections]
    orphan_collection_pairs = [(row[1], row[2]) for row in orphan_collections]

    all_ids = orphan_topic_ids + orphan_collection_ids
    if all_ids:
        placeholders = ",".join("?" * len(all_ids))
        # FTS5 / vec0 virtual tables have no FK cascade; satellites
        # must be deleted explicitly. Pattern mirrors index_paper.py:306.
        conn.execute(
            f"DELETE FROM term_aliases WHERE term_id IN ({placeholders})",
            all_ids,
        )
        conn.execute(
            f"DELETE FROM terms_fts WHERE term_id IN ({placeholders})",
            all_ids,
        )
        conn.execute(
            f"DELETE FROM term_embeddings WHERE term_id IN ({placeholders})",
            all_ids,
        )
        conn.execute(
            f"DELETE FROM canonical_terms WHERE id IN ({placeholders})",
            all_ids,
        )

    registry_deleted = 0
    for domain, name in orphan_collection_pairs:
        cur = conn.execute(
            "DELETE FROM collections WHERE domain = ? AND name = ?",
            (domain, name),
        )
        registry_deleted += cur.rowcount or 0

    return {
        "topics": len(orphan_topic_ids),
        "collections": len(orphan_collection_ids),
        "collections_registry": registry_deleted,
    }
