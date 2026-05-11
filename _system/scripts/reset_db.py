"""Reset ~/.lodestone/lodestone.db to a clean, taxonomy-seeded state.

    uv run _system/scripts/reset_db.py [--db PATH] [--taxonomy PATH] [--yes]

Wipes every user table (papers/posts/repos plus all auto-built taxonomy
and FTS/vec virtual tables) IN PLACE so the file's inode is preserved,
then re-seeds ``domains`` + ``collection_definitions`` from taxonomy.json.

Inode preservation matters: ``mcp_server.py`` opens the DB once at
startup and pins the inode (see ``_check_db_inode_pinned``). Unlinking
and recreating the file would silently orphan any concurrent server's
writes. This script only deletes rows, never the file.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.scripts.seed_taxonomy import seed_taxonomy
from _system.scripts.sync_plugin import sync_plugin, _resolve_dest as _plugin_dest
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.reset_db")

_DEFAULT_DB = Path.home() / ".lodestone" / "lodestone.db"
# parents[2] = lodestone repo root. taxonomy.json lives at the root so
# it ships with the plugin (rsync'd by sync_plugin into the install dir).
_DEFAULT_TAXONOMY = Path(__file__).resolve().parents[2] / "taxonomy.json"

# Tables we read for the pre-wipe summary. Headline counts only — the
# wipe itself is driven by sqlite_master enumeration so it auto-adapts.
_HEADLINE_TABLES = ("papers", "posts", "repos", "canonical_terms", "topics")


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    """Return every regular + virtual user table the wipe should touch.

    Excludes ``sqlite_*`` internals and FTS5/vec0 *shadow* tables. Some
    FTS5 shadows (``*_config``, ``*_content``, ``*_data``, ``*_idx``,
    ``*_docsize``) are emitted as real ``CREATE TABLE`` rows in
    ``sqlite_master``, so a naive ``sql IS NOT NULL`` filter leaks them.
    Instead: collect every virtual-table name first, then exclude any
    table whose name begins with ``<vtab>_``. DELETE-from-vtab cascades
    through the shadows automatically; touching them directly would
    corrupt FTS/vec internal state.
    """
    rows = conn.execute(
        """
        SELECT name, sql FROM sqlite_master
        WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
        ORDER BY name
        """
    ).fetchall()

    virtual = {n for n, sql in rows if sql.startswith("CREATE VIRTUAL TABLE")}
    shadow_prefixes = tuple(f"{name}_" for name in virtual)

    out: list[str] = []
    for name, sql in rows:
        if name in virtual:
            out.append(name)
            continue
        if not sql.startswith("CREATE TABLE"):
            continue
        if name.startswith(shadow_prefixes):
            continue
        out.append(name)
    return out


def reset(
    *,
    conn: sqlite3.Connection,
    taxonomy: dict,
) -> dict:
    """Wipe every user table in place, then re-seed the taxonomy.

    The DB file's inode is preserved — only rows are deleted. FK
    enforcement is suspended for the wipe so deletion order across
    cross-referencing tables doesn't matter, then re-enabled.

    Returns a summary dict with per-table row counts wiped plus the
    taxonomy-seed counts (passed through from :func:`seed_taxonomy`).
    """
    tables = _user_tables(conn)

    # PRAGMA foreign_keys is a no-op inside a transaction. Toggle it
    # outside the BEGIN/COMMIT block.
    conn.execute("PRAGMA foreign_keys = OFF")
    rows_wiped: dict[str, int] = {}
    try:
        with transaction(conn):
            for name in tables:
                count = conn.execute(
                    f"SELECT COUNT(*) FROM {name}"
                ).fetchone()[0]
                conn.execute(f"DELETE FROM {name}")
                rows_wiped[name] = int(count)
    finally:
        conn.execute("PRAGMA foreign_keys = ON")

    seed_summary = seed_taxonomy(conn=conn, taxonomy=taxonomy)

    return {
        "tables_wiped": len(tables),
        "total_rows_wiped": sum(rows_wiped.values()),
        "rows_wiped_by_table": rows_wiped,
        "taxonomy_seeded": seed_summary,
    }


def _headline_counts(conn: sqlite3.Connection) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in _HEADLINE_TABLES:
        row = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
        out[name] = int(row[0])
    return out


def _confirm(db_path: Path, counts: dict[str, int]) -> bool:
    print(f"about to WIPE all data in: {db_path}")
    print("current row counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("the file's inode is preserved; running MCP servers stay valid.")
    answer = input("proceed? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Wipe ~/.lodestone/lodestone.db in place and re-seed the "
            "curated taxonomy. Preserves the file's inode."
        ),
    )
    parser.add_argument(
        "--db", type=Path, default=_DEFAULT_DB,
        help=f"sqlite db path (default: {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--taxonomy", type=Path, default=_DEFAULT_TAXONOMY,
        help=f"taxonomy.json (default: {_DEFAULT_TAXONOMY})",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="skip the confirmation prompt",
    )
    parser.add_argument(
        "--vacuum", action="store_true",
        help="VACUUM after wipe to reclaim disk space (blocks; off by default)",
    )
    parser.add_argument(
        "--sync-plugin", action="store_true",
        help=(
            "after the reset, rsync the dev tree onto the installed "
            "lodestone plugin so concurrent Claude sessions pick up "
            "code changes without a /plugin update"
        ),
    )
    args = parser.parse_args(argv)

    if not args.taxonomy.exists():
        raise FileNotFoundError(
            f"taxonomy file not found: {args.taxonomy}. Pass --taxonomy "
            "to point at a valid taxonomy.json."
        )
    taxonomy = json.loads(args.taxonomy.read_text())

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = get_conn(args.db)
    try:
        init_db(conn)
        counts = _headline_counts(conn)
        if not args.yes and not _confirm(args.db, counts):
            print("aborted.")
            sys.exit(1)

        summary = reset(conn=conn, taxonomy=taxonomy)

        if args.vacuum:
            _LOG.info("running VACUUM on %s", args.db)
            conn.execute("VACUUM")
    finally:
        conn.close()

    if args.sync_plugin:
        repo_root = Path(__file__).resolve().parents[2]
        summary["plugin_synced"] = sync_plugin(
            source=repo_root,
            dest=_plugin_dest(repo_root),
            dry_run=False,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
