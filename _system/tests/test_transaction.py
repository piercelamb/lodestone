"""Tests for the :func:`_system.db.connection.transaction` helper."""
from __future__ import annotations

import sqlite3

import pytest

from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db


def _seed_domain(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("INSERT INTO domains (name) VALUES (?)", (name,))


def test_transaction_commits_on_success(db_path):
    c = get_conn(db_path)
    try:
        init_db(c)
        with transaction(c):
            _seed_domain(c, "ml")
        # Row should survive the context-manager exit.
        row = c.execute("SELECT name FROM domains WHERE name='ml'").fetchone()
        assert row == ("ml",)
    finally:
        c.close()


def test_transaction_rolls_back_on_exception(db_path):
    c = get_conn(db_path)
    try:
        init_db(c)
        with pytest.raises(RuntimeError, match="abort"):
            with transaction(c):
                _seed_domain(c, "ml")
                raise RuntimeError("abort")
        # Domain must not be present — BEGIN/ROLLBACK wound back the insert.
        row = c.execute("SELECT name FROM domains WHERE name='ml'").fetchone()
        assert row is None
    finally:
        c.close()


def test_transaction_not_reentrant(db_path):
    """Nested transaction() blocks raise — autocommit + explicit BEGIN is not composable.

    Documents the current contract: call sites requiring nesting should use
    SAVEPOINT explicitly. Breaking this test is a real API change.
    """
    c = get_conn(db_path)
    try:
        init_db(c)
        with pytest.raises(sqlite3.OperationalError, match="transaction"):
            with transaction(c):
                with transaction(c):
                    pass
    finally:
        c.close()
