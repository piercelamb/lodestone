"""Orphan GC for topic canonicals.

Topic canonicals can become orphaned in two situations:

1. Re-classifying a paper. ``classify_paper`` deletes the paper's
   ``paper_topics`` rows up front and re-runs the LLM. If the second run
   emits different topic phrasings, the first run's topic canonicals are
   left in ``canonical_terms`` with zero remaining ``paper_topics``
   references.
2. Deleting a paper. The per-paper cascade deletes ``paper_topics``
   alongside the paper, so a topic that only appeared in the deleted
   paper becomes orphaned.

Topics are per-paper tags — once no paper references one, it carries no
meaning and should disappear. **Domains and collections are different.**
They are curated organizational categories that exist independently of
any single paper; future papers can populate them. Deleting the last
paper in a collection must not delete the collection — only humans
delete categories. So the GC here is intentionally narrow: topics only.

Entities are deliberately out of scope — under the synonym-index regime,
tier-1 mentions leave no per-paper trace, so substantiation can't be
proven.
"""
from __future__ import annotations

import sqlite3


def gc_orphan_topic_canonicals(conn: sqlite3.Connection) -> dict[str, int]:
    """Delete orphan topic canonicals plus their satellites.

    Removes ``canonical_terms`` rows of ``term_type='topic'`` that have
    zero remaining ``paper_topics`` bindings, plus their matching rows
    in ``terms_fts``, ``term_embeddings``, and ``term_aliases``.

    Caller owns the enclosing transaction.

    Returns a counts dict::

        {"topics": N}
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

    if orphan_topic_ids:
        placeholders = ",".join("?" * len(orphan_topic_ids))
        # FTS5 / vec0 virtual tables have no FK cascade; satellites
        # must be deleted explicitly. Pattern mirrors index_paper.py:306.
        conn.execute(
            f"DELETE FROM term_aliases WHERE term_id IN ({placeholders})",
            orphan_topic_ids,
        )
        conn.execute(
            f"DELETE FROM terms_fts WHERE term_id IN ({placeholders})",
            orphan_topic_ids,
        )
        conn.execute(
            f"DELETE FROM term_embeddings WHERE term_id IN ({placeholders})",
            orphan_topic_ids,
        )
        conn.execute(
            f"DELETE FROM canonical_terms WHERE id IN ({placeholders})",
            orphan_topic_ids,
        )

    return {"topics": len(orphan_topic_ids)}
