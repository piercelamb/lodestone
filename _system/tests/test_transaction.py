"""Tests for the :func:`_system.db.connection.transaction` helper."""
from __future__ import annotations

import sqlite3

import pytest

from _system.db.connection import transaction


def _seed_domain(conn: sqlite3.Connection, name: str) -> None:
    conn.execute("INSERT INTO domains (name) VALUES (?)", (name,))


def test_transaction_commits_on_success(conn):
    with transaction(conn):
        _seed_domain(conn, "ml")
    row = conn.execute("SELECT name FROM domains WHERE name='ml'").fetchone()
    assert row == ("ml",)


def test_transaction_rolls_back_on_exception(conn):
    with pytest.raises(RuntimeError, match="abort"):
        with transaction(conn):
            _seed_domain(conn, "ml")
            raise RuntimeError("abort")
    row = conn.execute("SELECT name FROM domains WHERE name='ml'").fetchone()
    assert row is None


def test_transaction_not_reentrant(conn):
    """Nested transaction() blocks raise — autocommit + explicit BEGIN is not composable.

    Documents the current contract: call sites requiring nesting should use
    SAVEPOINT explicitly. Breaking this test is a real API change.
    """
    with pytest.raises(sqlite3.OperationalError, match="transaction"):
        with transaction(conn):
            with transaction(conn):
                pass
