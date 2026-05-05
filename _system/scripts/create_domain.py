"""Standalone idempotent domain registration.

    uv run _system/scripts/create_domain.py --name rag --description "..."

The same charset as ``slug.sanitize_domain`` is enforced here so a
manually-created domain never collides with an auto-sanitized name.
Idempotent by design — the CLI prints ``{"created": false}`` when the row
already existed rather than raising.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.utils.logging import get_logger
from _system.utils.slug import DOMAIN_MAX_LEN

_LOG = get_logger("scripts.create_domain")

_NAME_RE = re.compile(rf"^[a-z0-9_-]{{1,{DOMAIN_MAX_LEN}}}$")


def create_domain(
    *,
    conn: sqlite3.Connection,
    name: str,
    description: str,
) -> dict:
    """Insert ``(name, description)`` into ``domains`` if absent.

    Returns ``{"name": name, "created": bool}`` — ``created=False`` on an
    idempotent no-op. ``needs_review`` is 0 (manual creation is explicit;
    only classify's LLM-proposed domains flip that flag on the paper row).
    Raises ``ValueError`` on a name that doesn't match
    ``[a-z0-9_-]{1,DOMAIN_MAX_LEN}``.
    """
    if not _NAME_RE.match(name):
        raise ValueError(
            f"--name {name!r} must match [a-z0-9_-]+ and be 1–{DOMAIN_MAX_LEN} chars"
        )
    cur = conn.execute(
        "INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
        (name, description),
    )
    created = cur.rowcount == 1
    if created:
        _LOG.info("created domain %s", name)
    else:
        _LOG.info("domain %s already exists, no changes made", name)
    return {"name": name, "created": created}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Manually register a Lodestone domain (idempotent).",
    )
    parser.add_argument(
        "--name", required=True,
        help=f"domain slug, [a-z0-9_-]{{1,{DOMAIN_MAX_LEN}}}",
    )
    parser.add_argument("--description", required=True, help="human-readable blurb")
    parser.add_argument("--db", default="lodestone.db", help="sqlite db path")
    args = parser.parse_args(argv)

    conn = get_conn(Path(args.db))
    try:
        init_db(conn)
        result = create_domain(
            conn=conn,
            name=args.name,
            description=args.description,
        )
    finally:
        conn.close()
    print(json.dumps(result))


if __name__ == "__main__":
    main()
