"""Idempotent SQLite schema bootstrap for Lodestone.

Runs :file:`schema.sql` against a connection returned by
:func:`_system.db.connection.get_conn`. Plain ``CREATE TABLE`` and
``CREATE INDEX`` statements use ``IF NOT EXISTS`` and are executed via
:meth:`sqlite3.Connection.executescript` (which parses SQL comments
natively). Virtual tables (FTS5 and sqlite-vec ``vec0``) are extracted
from the script and guarded by a ``sqlite_master`` existence check —
``vec0`` does not support ``IF NOT EXISTS`` and we apply the same
pattern to FTS5 for uniformity.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_SCHEMA_FILE = Path(__file__).parent / "schema.sql"

# Matches a full `CREATE VIRTUAL TABLE <name> USING <module>(...);` block.
# `[^)]*` is safe here: our virtual-table arg lists contain no nested
# parentheses (FTS5 quoted options, vec0 `float[384]`).
VIRTUAL_TABLE_BLOCK_RE = re.compile(
    r"CREATE\s+VIRTUAL\s+TABLE\s+(\w+)\s+USING\s+\w+\s*\([^)]*\)\s*;",
    re.IGNORECASE | re.DOTALL,
)


def init_db(conn: sqlite3.Connection) -> None:
    """Run :file:`schema.sql` idempotently on ``conn``. Safe to call multiple times."""
    schema_sql = _SCHEMA_FILE.read_text(encoding="utf-8")

    virtual_blocks: list[tuple[str, str]] = []

    def _collect(m: "re.Match[str]") -> str:
        virtual_blocks.append((m.group(1), m.group(0)))
        return ""

    plain_sql = VIRTUAL_TABLE_BLOCK_RE.sub(_collect, schema_sql)

    # Plain DDL is all `IF NOT EXISTS` — executescript is safe on repeat runs.
    conn.executescript(plain_sql)

    for table_name, stmt in virtual_blocks:
        if not _table_exists(conn, table_name):
            conn.execute(stmt)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None
