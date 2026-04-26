"""Idempotent SQLite schema bootstrap for Lodestone.

Runs :file:`schema.sql` against a connection returned by
:func:`_system.db.connection.get_conn`. Plain ``CREATE TABLE`` and
``CREATE INDEX`` statements use ``IF NOT EXISTS`` and are executed via
:meth:`sqlite3.Connection.executescript` (which parses SQL comments
natively). Virtual tables (FTS5 and sqlite-vec ``vec0``) are extracted
from the script and guarded by a ``sqlite_master`` existence check —
``vec0`` does not support ``IF NOT EXISTS`` and we apply the same
pattern to FTS5 for uniformity.

One pre-script migration runs first: ``_migrate_entities_to_aliases``
folds the legacy ``entities`` table into ``term_aliases`` (adding
``source_breadcrumb`` to the PK). It runs only when the old shape is
detected; fresh DBs and already-migrated DBs skip it.
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
    _migrate_entities_to_aliases(conn)

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


def _migrate_entities_to_aliases(conn: sqlite3.Connection) -> None:
    """Fold the legacy ``entities`` table into ``term_aliases``.

    Runs once on any DB that predates the merge: detects the old
    3-column ``term_aliases`` PK via ``PRAGMA table_info``, copies both
    the existing alias rows (no breadcrumb known → ``''``) and the
    ``entities`` rows (joined to ``canonical_terms`` on
    ``(domain, term_type='entity', canonical_name)``) into a v2 table,
    then atomically swaps it in and drops ``entities``.

    Skips silently when ``term_aliases`` doesn't exist (fresh DB — the
    schema script will create the new shape) or when it already has
    ``source_breadcrumb`` (already migrated).
    """
    cols = conn.execute("PRAGMA table_info(term_aliases)").fetchall()
    if not cols:
        return  # fresh DB; schema.sql will create the new shape
    if any(c[1] == "source_breadcrumb" for c in cols):
        return  # already migrated

    with conn:
        conn.execute(
            """
            CREATE TABLE term_aliases_v2 (
                term_id INTEGER NOT NULL REFERENCES canonical_terms(id),
                alias TEXT NOT NULL,
                source_paper TEXT NOT NULL,
                source_breadcrumb TEXT NOT NULL DEFAULT '',
                match_tier INTEGER,
                PRIMARY KEY(term_id, alias, source_paper, source_breadcrumb)
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO term_aliases_v2
                (term_id, alias, source_paper, source_breadcrumb, match_tier)
            SELECT term_id, alias, source_paper, '', match_tier
              FROM term_aliases
            """
        )
        if _table_exists(conn, "entities"):
            conn.execute(
                """
                INSERT OR IGNORE INTO term_aliases_v2
                    (term_id, alias, source_paper, source_breadcrumb, match_tier)
                SELECT ct.id, e.entity_name, e.paper_name, e.source_breadcrumb, NULL
                  FROM entities e
                  JOIN canonical_terms ct
                    ON ct.domain = e.domain
                   AND ct.term_type = 'entity'
                   AND ct.canonical_name = e.entity_name
                """
            )
            conn.execute("DROP TABLE entities")
        conn.execute("DROP TABLE term_aliases")
        conn.execute("ALTER TABLE term_aliases_v2 RENAME TO term_aliases")
