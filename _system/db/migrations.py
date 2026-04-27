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
folds older shapes of ``term_aliases`` into the current
``(term_id, alias, source_paper)`` synonym-index PK. It dispatches on
``PRAGMA table_info(term_aliases)``:

- no ``term_aliases`` table → fresh DB, skip.
- 3-col PK with no ``source_breadcrumb`` → already migrated, skip.
- 3-col PK with sibling ``entities`` table → pre-PR-#20 legacy: copy
  alias rows, fold ``entities`` rows joined to ``canonical_terms`` on
  ``(domain, term_type='entity', canonical_name)``, drop ``entities``.
- 4-col PK with ``source_breadcrumb`` → post-PR-#20 appearance-log
  shape: copy synonym rows, filter out canonical-as-alias rows,
  dedupe via ``INSERT OR IGNORE`` on the new 3-col PK.

Both fold branches filter ``alias != canonical_name`` so the migration
itself respects the synonym-index invariant.
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
    """Fold legacy / appearance-log shapes of ``term_aliases`` into the
    current synonym-index PK ``(term_id, alias, source_paper)``.

    Three-state dispatcher driven by ``PRAGMA table_info(term_aliases)``:

    - **Fresh DB** (no ``term_aliases``): skip; ``schema.sql`` will
      create the new shape.
    - **Already migrated** (3-col PK, no ``source_breadcrumb``): skip.
    - **Pre-PR-#20 legacy** (3-col PK with sibling ``entities`` table):
      copy alias rows and fold ``entities`` rows, filtering out
      ``entity_name == canonical_name`` so canonicals don't reappear as
      aliases under the synonym-index invariant; drop ``entities``;
      atomic swap.
    - **Post-PR-#20 appearance log** (4-col PK with
      ``source_breadcrumb``): copy ``(term_id, alias, source_paper,
      match_tier)`` filtered to rows where the alias differs from the
      canonical; ``INSERT OR IGNORE`` deduplicates per-section
      duplicates onto the new 3-col PK; atomic swap.
    """
    cols = conn.execute("PRAGMA table_info(term_aliases)").fetchall()
    if not cols:
        return  # fresh DB; schema.sql will create the new shape

    has_breadcrumb = any(c[1] == "source_breadcrumb" for c in cols)
    has_legacy_entities = _table_exists(conn, "entities")

    if not has_breadcrumb and not has_legacy_entities:
        return  # already migrated

    with conn:
        conn.execute(
            """
            CREATE TABLE term_aliases_v2 (
                term_id INTEGER NOT NULL REFERENCES canonical_terms(id),
                alias TEXT NOT NULL,
                source_paper TEXT NOT NULL,
                match_tier INTEGER,
                PRIMARY KEY(term_id, alias, source_paper)
            )
            """
        )

        if has_breadcrumb:
            # Post-PR-#20 appearance-log shape. Copy synonym rows only —
            # rows where alias matches the canonical are filtered out so
            # the synonym-index invariant holds. Multiple breadcrumb-
            # variants of the same (term_id, alias, source_paper) collapse
            # via INSERT OR IGNORE.
            conn.execute(
                """
                INSERT OR IGNORE INTO term_aliases_v2
                    (term_id, alias, source_paper, match_tier)
                SELECT ta.term_id, ta.alias, ta.source_paper, ta.match_tier
                  FROM term_aliases ta
                  JOIN canonical_terms ct ON ct.id = ta.term_id
                 WHERE ta.alias != ct.canonical_name
                """
            )
        else:
            # Pre-PR-#20 legacy shape (3-col PK without breadcrumb).
            conn.execute(
                """
                INSERT OR IGNORE INTO term_aliases_v2
                    (term_id, alias, source_paper, match_tier)
                SELECT ta.term_id, ta.alias, ta.source_paper, ta.match_tier
                  FROM term_aliases ta
                  JOIN canonical_terms ct ON ct.id = ta.term_id
                 WHERE ta.alias != ct.canonical_name
                """
            )
            if has_legacy_entities:
                # Fold legacy `entities` rows into the synonym index.
                # Filter out entries where entity_name matches the
                # canonical — canonicals are not synonyms of themselves.
                conn.execute(
                    """
                    INSERT OR IGNORE INTO term_aliases_v2
                        (term_id, alias, source_paper, match_tier)
                    SELECT ct.id, e.entity_name, e.paper_name, NULL
                      FROM entities e
                      JOIN canonical_terms ct
                        ON ct.domain = e.domain
                       AND ct.term_type = 'entity'
                       AND ct.canonical_name = e.entity_name
                     WHERE e.entity_name != ct.canonical_name
                    """
                )
                conn.execute("DROP TABLE entities")

        conn.execute("DROP TABLE term_aliases")
        conn.execute("ALTER TABLE term_aliases_v2 RENAME TO term_aliases")
