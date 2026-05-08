"""Per-paper / per-repo cascade-delete helpers.

Used by ``fetch_paper`` (force-refetch path), ``ingest`` (``--force``
cascade), and the standalone-repo path. Per-target rows (figures,
sections, topics, code_files, polymorphic ``collections`` membership
rows, ...) are removed alongside the parent; canonical taxonomy rows
are touched only via orphan-GC at the end of the cascade. Domains and
``collection_definitions`` (the catalog) are curated categories — they
survive the deletion of their last paper / repo so future targets can
fill them; only humans delete those. The polymorphic ``collections``
junction rows DO go with the parent, since they record per-target
membership rather than the curated category itself. Entity canonicals
are never GC'd — under the synonym-index regime, tier-1 mentions leave
no per-paper trace, so substantiation can't be proven.
"""
from __future__ import annotations

import sqlite3

from _system.db.orphan_gc import gc_orphan_topic_canonicals
from _system.utils.source_resolution import SourceKind


def delete_paper_cascade(conn: sqlite3.Connection, *, paper_id: int) -> None:
    """DELETE one paper and every per-paper child row.

    The caller owns the enclosing transaction. Order matters: FK-backed
    children before the papers row (PRAGMA foreign_keys=ON); FTS5 tables
    have no FK cascade, so their rows must be deleted explicitly. Orphan
    topic canonicals are GC'd at the end, after the paper is gone, when
    "zero remaining bindings" is a clean truth. The catalog
    (``collection_definitions``) survives — it's the curated set of
    categories, not per-paper bookkeeping; only the polymorphic
    ``collections`` membership rows go with the paper.

    Any ``repos`` rows linked to this paper are cascaded too — the repo
    has no independent identity once its anchoring paper is gone.
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
    conn.execute(
        "DELETE FROM collections WHERE target_kind = ? AND target_id = ?",
        (SourceKind.PAPER.value, paper_id),
    )
    conn.execute(
        "DELETE FROM topics WHERE target_kind = ? AND target_id = ?",
        (SourceKind.PAPER.value, paper_id),
    )
    conn.execute("DELETE FROM sections     WHERE paper_id = ?", (paper_id,))

    # Cascade any linked repos. Each repo cleanup also drops topics with
    # target_kind='repo' and the repo's code_files / readmes_fts rows.
    repo_ids = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM repos WHERE paper_id = ?", (paper_id,)
        ).fetchall()
    ]
    for repo_id in repo_ids:
        delete_repo_cascade(conn, repo_id=repo_id, _gc_topics=False)

    conn.execute("DELETE FROM papers       WHERE id       = ?", (paper_id,))
    gc_orphan_topic_canonicals(conn)


def delete_post_cascade(conn: sqlite3.Connection, *, post_id: int) -> None:
    """DELETE one post and every per-post child row.

    Mirrors :func:`delete_paper_cascade` for the blog-post side. Order
    matters: FK-backed children before the posts row; FTS5 ``sections``
    rows are slug-keyed (no FK) so their delete predicate goes through
    ``post_name``. Orphan topic canonicals are GC'd at the end.

    Posts have no figures, no per-post linked repos in v1, and no
    paper_references row to null out (paper-side refs target only papers).
    """
    # Outbound arxiv refs from this post.
    conn.execute("DELETE FROM post_references WHERE post_id = ?", (post_id,))
    # term_aliases keys by slug TEXT, not id, so look up the post_name.
    conn.execute(
        """
        DELETE FROM term_aliases
         WHERE source_paper = (SELECT post_name FROM posts WHERE id = ?)
        """,
        (post_id,),
    )
    conn.execute(
        "DELETE FROM collections WHERE target_kind = ? AND target_id = ?",
        (SourceKind.POST.value, post_id),
    )
    conn.execute(
        "DELETE FROM topics WHERE target_kind = ? AND target_id = ?",
        (SourceKind.POST.value, post_id),
    )
    # sections.paper_name is the shared slug column — used here by post_name.
    conn.execute(
        """
        DELETE FROM sections
         WHERE paper_name = (SELECT post_name FROM posts WHERE id = ?)
        """,
        (post_id,),
    )
    conn.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    gc_orphan_topic_canonicals(conn)


def delete_repo_cascade(
    conn: sqlite3.Connection,
    *,
    repo_id: int,
    _gc_topics: bool = True,
) -> None:
    """DELETE one repo and every per-repo child row.

    Mirrors :func:`delete_paper_cascade` for the repo-side state. Topics
    with ``target_kind='repo'`` and the repo's polymorphic
    ``collections`` membership rows are wiped; ``code_files`` and the
    matching ``readmes_fts`` row go with the repo. Orphan topic
    canonicals are GC'd at the end (caller can suppress when chained
    inside ``delete_paper_cascade``, which runs its own GC after
    everything is gone).
    """
    conn.execute(
        "DELETE FROM topics WHERE target_kind = ? AND target_id = ?",
        (SourceKind.REPO.value, repo_id),
    )
    conn.execute(
        "DELETE FROM collections WHERE target_kind = ? AND target_id = ?",
        (SourceKind.REPO.value, repo_id),
    )
    conn.execute("DELETE FROM code_files  WHERE repo_id = ?", (repo_id,))
    conn.execute("DELETE FROM readmes_fts WHERE repo_id = ?", (repo_id,))
    conn.execute("DELETE FROM repos       WHERE id       = ?", (repo_id,))
    if _gc_topics:
        gc_orphan_topic_canonicals(conn)
