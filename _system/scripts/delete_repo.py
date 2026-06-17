"""Delete one repo (paper-linked or standalone) without touching the paper.

    uv run _system/scripts/delete_repo.py --slug gh-owner-name [--db PATH] [--yes]

Use this when repo discovery auto-attached the *wrong* repo to a paper —
e.g. a paper cites a generic benchmark harness and that gets registered
as the paper's linked repo. This wipes only the repo-side state:
``code_files``, the ``readmes_fts`` row, repo ``topics``, the repo's
polymorphic ``collections`` membership rows, and the ``repos`` row
itself, then GCs any topic canonicals whose last binding was this repo
(via :func:`delete_repo_cascade`).

What it never touches: the ``papers`` row, its sections / figures /
entities / paper-side topics & collections, or the curated catalog
(``domains``, ``collection_definitions``). For a paper-linked repo the
paper simply reverts to having no linked repo. The deletion is verified
to leave the linked paper row intact before the summary is printed.

Only DELETEs rows — never unlinks/recreates the DB file — so a running
``mcp_server.py`` (which pins the DB inode at startup) stays valid, same
contract as ``reset_db.py``.

NOTE: re-running ``ingest.py --url <paper> --force`` will re-run repo
discovery and likely re-attach the same wrong repo. This fixes the
current state, not future re-ingests.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple

from _system.db.cascade import delete_repo_cascade
from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.utils.logging import get_logger
from _system.utils.source_resolution import SourceKind

_LOG = get_logger("scripts.delete_repo")

_DEFAULT_DB = Path.home() / ".lodestone" / "lodestone.db"


class _RepoRow(NamedTuple):
    id: int
    repo_slug: str
    url: str
    status: str
    paper_id: int | None


def _resolve_repo(conn: sqlite3.Connection, *, slug: str) -> _RepoRow | None:
    """Look up a repo by its unique ``repo_slug``. ``None`` if absent."""
    row = conn.execute(
        "SELECT id, repo_slug, url, status, paper_id FROM repos WHERE repo_slug = ?",
        (slug,),
    ).fetchone()
    if row is None:
        return None
    return _RepoRow(id=row[0], repo_slug=row[1], url=row[2], status=row[3], paper_id=row[4])


def _linked_paper_name(conn: sqlite3.Connection, *, paper_id: int) -> str | None:
    row = conn.execute(
        "SELECT paper_name FROM papers WHERE id = ?", (paper_id,)
    ).fetchone()
    return row[0] if row else None


def _child_counts(conn: sqlite3.Connection, *, repo_id: int) -> dict[str, int]:
    """Per-repo child-row counts, captured before the cascade for the summary."""
    one = lambda sql, params: int(conn.execute(sql, params).fetchone()[0])  # noqa: E731
    return {
        "code_files": one(
            "SELECT COUNT(*) FROM code_files WHERE repo_id = ?", (repo_id,)
        ),
        "readme_rows": one(
            "SELECT COUNT(*) FROM readmes_fts WHERE repo_id = ?", (repo_id,)
        ),
        "repo_topics": one(
            "SELECT COUNT(*) FROM topics WHERE target_kind = ? AND target_id = ?",
            (SourceKind.REPO.value, repo_id),
        ),
        "collection_memberships": one(
            "SELECT COUNT(*) FROM collections WHERE target_kind = ? AND target_id = ?",
            (SourceKind.REPO.value, repo_id),
        ),
    }


def delete_repo(*, conn: sqlite3.Connection, repo: _RepoRow) -> dict:
    """Cascade-delete ``repo`` inside one transaction; assert the paper survives.

    Snapshots the per-repo child counts first, runs the canonical
    :func:`delete_repo_cascade`, then — for a paper-linked repo —
    re-reads the ``papers`` row to prove the paper was untouched. Raises
    ``RuntimeError`` if the linked paper somehow vanished (it can't, given
    the cascade never targets ``papers``; the check is a cheap guardrail
    so a future cascade-logic regression fails loud instead of silently
    eating a paper).
    """
    deleted = _child_counts(conn, repo_id=repo.id)
    deleted["repo_row"] = 1

    with transaction(conn):
        delete_repo_cascade(conn, repo_id=repo.id)

    paper_preserved: bool | None = None
    if repo.paper_id is not None:
        name = _linked_paper_name(conn, paper_id=repo.paper_id)
        paper_preserved = name is not None
        if not paper_preserved:
            raise RuntimeError(
                f"linked paper id={repo.paper_id} disappeared during repo "
                f"deletion — aborting trust in the cascade. Investigate "
                f"delete_repo_cascade before re-running."
            )

    _LOG.info("deleted repo %s (id=%d)", repo.repo_slug, repo.id)
    return {
        "repo_slug": repo.repo_slug,
        "repo_id": repo.id,
        "url": repo.url,
        "linked_paper_id": repo.paper_id,
        "paper_preserved": paper_preserved,
        "deleted": deleted,
    }


def _confirm(conn: sqlite3.Connection, *, repo: _RepoRow, counts: dict[str, int]) -> bool:
    paper = (
        _linked_paper_name(conn, paper_id=repo.paper_id)
        if repo.paper_id is not None
        else None
    )
    print(f"about to DELETE repo: {repo.repo_slug}  (id={repo.id}, status={repo.status})")
    print(f"  url:          {repo.url}")
    if paper is not None:
        print(f"  linked paper: {paper} (id={repo.paper_id}) — PRESERVED, only the repo is removed")
    else:
        print("  linked paper: none (standalone repo)")
    print("  rows to remove:")
    for k, v in counts.items():
        print(f"    {k}: {v}")
    print("  the paper and all its generated data (sections/figures/entities/topics) stay.")
    answer = input("proceed? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Delete one repo (paper-linked or standalone) and its per-repo "
            "child rows, leaving the linked paper and its data intact."
        ),
    )
    parser.add_argument(
        "--slug", required=True,
        help="repo_slug to delete, e.g. gh-owner-name",
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
        repo = _resolve_repo(conn, slug=args.slug)
        if repo is None:
            print(
                f"no repo with repo_slug={args.slug!r} in {args.db}", file=sys.stderr
            )
            sys.exit(1)

        counts = _child_counts(conn, repo_id=repo.id)
        if not args.yes and not _confirm(conn, repo=repo, counts=counts):
            print("aborted.")
            sys.exit(1)

        summary = delete_repo(conn=conn, repo=repo)
    finally:
        conn.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
