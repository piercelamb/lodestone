"""SQLite connection factory for Lodestone.

Opens a sqlite3 connection with the pinned sqlite-vec extension loaded and
the canonical set of pragmas applied (FK on, WAL, etc). Every Lodestone
script/module acquires its DB handle through :func:`get_conn`.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Union

import sqlite_vec

# sqlite-vec is pinned at 0.1.9 (pre-v1, breaking changes expected between
# minor versions). `vec_version()` returns a string like "v0.1.9"; we assert
# the prefix at connect-time so a mismatched install fails loudly.
PINNED_VEC_PREFIX = "v0.1.9"

PathLike = Union[str, Path]


class VecVersionMismatch(RuntimeError):
    """sqlite-vec runtime version does not match the pinned project version."""


def get_conn(db_path: PathLike) -> sqlite3.Connection:
    """Open a Lodestone SQLite connection with sqlite-vec loaded and pragmas set."""
    conn = sqlite3.connect(str(db_path))
    # Explicit transaction control. Python's default ``isolation_level="deferred"``
    # starts an implicit BEGIN on the first DML statement but leaves DDL
    # outside any transaction; for a migration-heavy + batch-write app that
    # split behaviour is confusing. Autocommit + the explicit
    # ``transaction()`` helper below keeps boundaries obvious.
    conn.isolation_level = None

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)

    version = conn.execute("SELECT vec_version()").fetchone()[0]
    if not version.startswith(PINNED_VEC_PREFIX):
        conn.close()
        raise VecVersionMismatch(
            f"sqlite-vec version mismatch: expected prefix {PINNED_VEC_PREFIX!r}, "
            f"got {version!r}. Lodestone pins sqlite-vec==0.1.9 (pre-v1, breaking "
            f"changes between minor versions are expected). Run `uv sync` and "
            f"verify pyproject.toml."
        )

    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """BEGIN/COMMIT helper; rolls back on exception.

    Not re-entrant — the connection runs in autocommit mode, so nesting
    raises ``sqlite3.OperationalError: cannot start a transaction within a
    transaction``. Call sites that need composition should use SAVEPOINT
    directly.
    """
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
