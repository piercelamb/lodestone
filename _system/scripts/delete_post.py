"""Delete one post (stub, orphan, or fully-indexed) and its per-post rows.

    uv run _system/scripts/delete_post.py --slug post_2026 [--db PATH] [--yes]
    uv run _system/scripts/delete_post.py --url https://example.com/post [--yes]

Use this to remove a post outright — e.g. a paywalled ``failed_fetch`` /
stub you won't re-ingest, or a post that landed under the wrong identity.
Until now the only lever was ``ingest_post --force``, which *re-ingests*
rather than deletes. This wipes only post-side state via
:func:`delete_post_cascade`: the ``post_references`` rows, the slug-keyed
``sections`` FTS rows, post ``topics`` + the post's polymorphic
``collections`` membership rows, ``term_aliases`` keyed by the slug, and
the ``posts`` row itself, then GCs any topic canonical whose last binding
was this post.

What it never touches: the curated catalog (``domains``,
``collection_definitions``) — those survive the deletion of their last
member so future posts/papers can fill them; only humans delete those.
Posts carry no linked paper and no figures, so (unlike ``delete_repo.py``)
there is nothing to preserve on the other side.

Only DELETEs rows — never unlinks/recreates the DB file — so a running
``mcp_server.py`` (which pins the DB inode at startup) stays valid, same
contract as ``reset_db.py`` / ``delete_repo.py``.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple

from _system.db.cascade import delete_post_cascade
from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.utils.logging import get_logger
from _system.utils.source_resolution import SourceKind

_LOG = get_logger("scripts.delete_post")

_DEFAULT_DB = Path.home() / ".lodestone" / "lodestone.db"


class _PostRow(NamedTuple):
    id: int
    post_name: str
    canonical_url: str
    status: str


def _resolve_post(
    conn: sqlite3.Connection, *, slug: str | None, url: str | None,
) -> _PostRow | None:
    """Look up a post by its unique ``post_name`` slug or by URL.

    ``--slug`` is the primary lever (faithful sibling of
    :func:`delete_repo._resolve_repo`); ``--url`` resolves against either
    ``source_url`` or ``canonical_url`` since posts are often known by URL.
    Returns ``None`` if absent.
    """
    if slug is not None:
        row = conn.execute(
            "SELECT id, post_name, canonical_url, status FROM posts "
            " WHERE post_name = ?",
            (slug,),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, post_name, canonical_url, status FROM posts "
            " WHERE source_url = ? OR canonical_url = ? LIMIT 1",
            (url, url),
        ).fetchone()
    if row is None:
        return None
    return _PostRow(id=row[0], post_name=row[1], canonical_url=row[2], status=row[3])


def _child_counts(
    conn: sqlite3.Connection, *, post_id: int, post_name: str,
) -> dict[str, int]:
    """Per-post child-row counts, captured before the cascade for the summary."""
    one = lambda sql, params: int(conn.execute(sql, params).fetchone()[0])  # noqa: E731
    return {
        "post_references": one(
            "SELECT COUNT(*) FROM post_references WHERE post_id = ?", (post_id,)
        ),
        "sections": one(
            "SELECT COUNT(*) FROM sections WHERE paper_name = ?", (post_name,)
        ),
        "post_topics": one(
            "SELECT COUNT(*) FROM topics WHERE target_kind = ? AND target_id = ?",
            (SourceKind.POST.value, post_id),
        ),
        "collection_memberships": one(
            "SELECT COUNT(*) FROM collections WHERE target_kind = ? AND target_id = ?",
            (SourceKind.POST.value, post_id),
        ),
        "term_aliases": one(
            "SELECT COUNT(*) FROM term_aliases WHERE source_paper = ?", (post_name,)
        ),
    }


def delete_post(*, conn: sqlite3.Connection, post: _PostRow) -> dict:
    """Cascade-delete ``post`` inside one transaction; return a JSON summary.

    Snapshots the per-post child counts first, then runs the canonical
    :func:`delete_post_cascade`. No "linked paper preserved" guard — posts
    have no linked paper (the cascade is paper-free).
    """
    deleted = _child_counts(conn, post_id=post.id, post_name=post.post_name)
    deleted["post_row"] = 1

    with transaction(conn):
        delete_post_cascade(conn, post_id=post.id)

    _LOG.info("deleted post %s (id=%d)", post.post_name, post.id)
    return {
        "post_name": post.post_name,
        "post_id": post.id,
        "canonical_url": post.canonical_url,
        "status": post.status,
        "deleted": deleted,
    }


def _confirm(
    conn: sqlite3.Connection, *, post: _PostRow, counts: dict[str, int],
) -> bool:
    print(
        f"about to DELETE post: {post.post_name}  "
        f"(id={post.id}, status={post.status})"
    )
    print(f"  canonical_url: {post.canonical_url}")
    print("  rows to remove:")
    for k, v in counts.items():
        print(f"    {k}: {v}")
    print("  the curated catalog (domains, collection_definitions) stays.")
    answer = input("proceed? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete one post and its per-post child rows (sections, "
            "references, topics, collections, term aliases), leaving the "
            "curated catalog intact."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--slug", default=None,
        help="post_name to delete, e.g. post_2026",
    )
    target.add_argument(
        "--url", default=None,
        help="source_url or canonical_url of the post to delete",
    )
    parser.add_argument(
        "--db", type=Path, default=_DEFAULT_DB,
        help=f"sqlite db path (default: {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="skip the confirmation prompt",
    )
    args = parser.parse_args(argv)

    if not args.db.exists():
        print(f"db not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    conn = get_conn(args.db)
    try:
        init_db(conn)
        post = _resolve_post(conn, slug=args.slug, url=args.url)
        if post is None:
            ident = f"post_name={args.slug!r}" if args.slug else f"url={args.url!r}"
            print(f"no post with {ident} in {args.db}", file=sys.stderr)
            sys.exit(1)

        counts = _child_counts(conn, post_id=post.id, post_name=post.post_name)
        if not args.yes and not _confirm(conn, post=post, counts=counts):
            print("aborted.")
            sys.exit(1)

        summary = delete_post(conn=conn, post=post)
    finally:
        conn.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
