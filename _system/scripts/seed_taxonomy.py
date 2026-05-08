"""Bulk-seed ``domains`` and ``collection_definitions`` from a consolidated taxonomy file.

    uv run _system/scripts/seed_taxonomy.py --file path/to/taxonomy.json

Run on a fresh DB (no classified papers yet) so the LLM's per-paper
classifier sees a populated ``existing_taxonomy`` from paper #1 instead of
proposing one new entry per early ingest. Mirrors ``create_domain.py``:
``INSERT OR IGNORE`` for both tables, single transaction, idempotent.
Domain names are run through ``sanitize_domain`` so the slugged form
matches what classify_paper would have written. Collection names are
stored verbatim (the classifier's resolver canonicalizes free-form names
at runtime; pre-seeded names are treated as canonical on first sight).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.utils.logging import get_logger
from _system.utils.slug import sanitize_domain

_LOG = get_logger("scripts.seed_taxonomy")


def seed_taxonomy(
    *,
    conn: sqlite3.Connection,
    taxonomy: dict,
    force: bool = False,
) -> dict:
    """Insert every (domain, collection) row from ``taxonomy`` if absent.

    Aborts with ``RuntimeError`` when ``papers`` is non-empty unless
    ``force`` is True — seeding after classification has begun would
    introduce categories that don't reflect what the LLM has already
    been writing into the live taxonomy.
    """
    paper_count = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    if paper_count > 0 and not force:
        raise RuntimeError(
            f"refusing to seed: {paper_count} paper(s) already classified; "
            "the taxonomy has been shaped by real data. Pass --force to override."
        )

    domains_inserted = 0
    domains_skipped = 0
    collections_inserted = 0
    collections_skipped = 0

    with transaction(conn):
        for d in taxonomy["domains"]:
            sanitized = sanitize_domain(d["name"])
            if not sanitized:
                raise ValueError(
                    f"domain name {d['name']!r} sanitizes to empty string"
                )
            cur = conn.execute(
                "INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
                (sanitized, d["description"]),
            )
            if cur.rowcount == 1:
                domains_inserted += 1
                _LOG.info("inserted domain %s", sanitized)
            else:
                domains_skipped += 1
                _LOG.info("domain %s already exists, skipping", sanitized)

            for c in d["collections"]:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO collection_definitions (domain, name, description)
                    VALUES (?, ?, ?)
                    """,
                    (sanitized, c["name"], c["description"]),
                )
                if cur.rowcount == 1:
                    collections_inserted += 1
                else:
                    collections_skipped += 1

    return {
        "domains_inserted": domains_inserted,
        "domains_skipped": domains_skipped,
        "collections_inserted": collections_inserted,
        "collections_skipped": collections_skipped,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Bulk-seed Lodestone domains + collection_definitions from a taxonomy JSON file.",
    )
    parser.add_argument(
        "--file", required=True, type=Path,
        help="path to taxonomy.json (shape: {domains: [{name, description, collections: [{name, description}]}]})",
    )
    parser.add_argument("--db", default="lodestone.db", help="sqlite db path")
    parser.add_argument(
        "--force", action="store_true",
        help="seed even if papers are already classified",
    )
    args = parser.parse_args(argv)

    taxonomy = json.loads(args.file.read_text())

    conn = get_conn(Path(args.db))
    try:
        init_db(conn)
        result = seed_taxonomy(conn=conn, taxonomy=taxonomy, force=args.force)
    finally:
        conn.close()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
