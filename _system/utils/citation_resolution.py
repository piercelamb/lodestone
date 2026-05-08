"""Forward/backward arxiv-citation resolution for papers and posts.

Both papers and posts can cite arxiv ids in their references tables
(``paper_references`` / ``post_references``). The resolution job runs
in two passes:

* Forward — for the source we just converted, populate ``cited_paper_id``
  for any reference whose ``cited_arxiv_id`` matches a paper already in
  the DB.
* Backward — when a *paper* is converted, find any other reference (in
  either references table) whose ``cited_arxiv_id`` equals the just-
  converted paper's arxiv_id and link it to this paper. This catches
  the case where paper A ingested before paper B; A's references stay
  NULL until B's CONVERT runs. Only papers can be backward targets —
  posts have no arxiv_id, so they're never citable.

Caller owns the enclosing transaction.
"""
from __future__ import annotations

import sqlite3

from _system.utils.source_resolution import SourceKind


def resolve_arxiv_citations(
    conn: sqlite3.Connection,
    *,
    kind: SourceKind,
    source_id: int,
    source_arxiv_id: str | None,
) -> tuple[int, int]:
    """Run forward + backward arxiv-citation resolution.

    Returns ``(forward_resolved, backward_resolved)``. ``backward_resolved``
    is always 0 for posts since posts can't be citation targets.
    """
    if kind is SourceKind.PAPER:
        forward_resolved = _forward_resolve_paper(conn, paper_id=source_id)
    else:
        forward_resolved = _forward_resolve_post(conn, post_id=source_id)

    backward_resolved = 0
    if kind is SourceKind.PAPER and source_arxiv_id is not None:
        backward_resolved = _backward_resolve_for_paper(
            conn, paper_id=source_id, arxiv_id=source_arxiv_id,
        )

    return forward_resolved, backward_resolved


def _forward_resolve_paper(conn: sqlite3.Connection, *, paper_id: int) -> int:
    """Forward pass for ``paper_references``.

    EXISTS guard restricts the UPDATE to rows that will actually link, so
    ``rowcount`` reflects resolved references rather than inspected ones.
    Self-citation (paper -> own arxiv_id) is allowed.
    """
    cur = conn.execute(
        """
        UPDATE paper_references
           SET cited_paper_id = (
               SELECT id FROM papers
                WHERE arxiv_id = paper_references.cited_arxiv_id
           )
         WHERE paper_id = ?
           AND cited_arxiv_id IS NOT NULL
           AND cited_paper_id IS NULL
           AND EXISTS (
               SELECT 1 FROM papers
                WHERE arxiv_id = paper_references.cited_arxiv_id
           )
        """,
        (paper_id,),
    )
    return cur.rowcount


def _forward_resolve_post(conn: sqlite3.Connection, *, post_id: int) -> int:
    """Forward pass for ``post_references``."""
    cur = conn.execute(
        """
        UPDATE post_references
           SET cited_paper_id = (
               SELECT id FROM papers
                WHERE arxiv_id = post_references.cited_arxiv_id
           )
         WHERE post_id = ?
           AND cited_arxiv_id IS NOT NULL
           AND cited_paper_id IS NULL
           AND EXISTS (
               SELECT 1 FROM papers
                WHERE arxiv_id = post_references.cited_arxiv_id
           )
        """,
        (post_id,),
    )
    return cur.rowcount


def _backward_resolve_for_paper(
    conn: sqlite3.Connection, *, paper_id: int, arxiv_id: str,
) -> int:
    """Backward pass: any *other* paper or post reference whose
    ``cited_arxiv_id`` matches THIS paper's arxiv_id gets its
    ``cited_paper_id`` populated.

    Spans both ``paper_references`` and ``post_references`` because either
    can name an arxiv id that finally landed in the corpus.
    """
    paper_cur = conn.execute(
        """
        UPDATE paper_references
           SET cited_paper_id = ?
         WHERE cited_arxiv_id = ?
           AND cited_paper_id IS NULL
        """,
        (paper_id, arxiv_id),
    )
    post_cur = conn.execute(
        """
        UPDATE post_references
           SET cited_paper_id = ?
         WHERE cited_arxiv_id = ?
           AND cited_paper_id IS NULL
        """,
        (paper_id, arxiv_id),
    )
    return paper_cur.rowcount + post_cur.rowcount
