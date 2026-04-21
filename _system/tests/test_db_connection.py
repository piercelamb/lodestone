"""Tests for the connection factory's extension loading, pragmas, and version pin."""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from _system.db import connection as conn_mod
from _system.db.connection import VecVersionMismatch, get_conn


def test_get_conn_loads_sqlite_vec(db_path):
    c = get_conn(db_path)
    try:
        version = c.execute("SELECT vec_version()").fetchone()[0]
        assert isinstance(version, str)
        assert version, "vec_version() returned empty string"
    finally:
        c.close()


def test_vec_version_matches_pinned_prefix(db_path):
    """Asserts the real sqlite-vec version matches the project's pinned prefix.

    Prevents silent drift if pyproject.toml is bumped without updating the pin
    constant in :mod:`_system.db.connection`.
    """
    c = get_conn(db_path)
    try:
        version = c.execute("SELECT vec_version()").fetchone()[0]
        assert version.startswith(conn_mod.PINNED_VEC_PREFIX), (
            f"sqlite-vec {version!r} diverged from pin {conn_mod.PINNED_VEC_PREFIX!r}"
        )
    finally:
        c.close()


@pytest.mark.parametrize(
    "pragma, expected",
    [
        ("foreign_keys", 1),
        ("journal_mode", "wal"),
        ("synchronous", 1),   # NORMAL
        ("busy_timeout", 5000),
        ("temp_store", 2),    # MEMORY
    ],
)
def test_get_conn_sets_required_pragmas(db_path, pragma, expected):
    c = get_conn(db_path)
    try:
        value = c.execute(f"PRAGMA {pragma}").fetchone()[0]
        if isinstance(value, str):
            value = value.lower()
        assert value == expected
    finally:
        c.close()


def test_get_conn_disables_load_extension_after_sqlite_vec_failure(db_path):
    """The try/finally must re-disable extension loading even when sqlite_vec.load raises.

    Python 3.14's ``sqlite3.Connection`` is an immutable heap type, so instead
    of patching its methods we intercept ``sqlite3.connect`` with a MagicMock
    and observe the enable_load_extension call pattern.
    """
    mock_conn = MagicMock()
    with (
        patch("sqlite3.connect", return_value=mock_conn),
        patch.object(conn_mod.sqlite_vec, "load", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            get_conn(db_path)

    flags = [c.args[0] for c in mock_conn.enable_load_extension.call_args_list]
    assert flags, "enable_load_extension was never called"
    assert flags[0] is True, "must enable extension loading first"
    assert flags[-1] is False, "must re-disable extension loading in finally"


def test_get_conn_real_connection_load_extension_disabled_after_failure(db_path):
    """Real-connection counterpart: after a sqlite_vec.load failure, the
    connection must refuse subsequent ``SELECT load_extension(...)`` calls.

    This observes the flag on a real sqlite3.Connection (the MagicMock test
    above only verifies the call sequence on the mock, not actual sqlite state).
    """
    captured: dict[str, sqlite3.Connection] = {}
    real_connect = sqlite3.connect

    def capturing_connect(*args, **kwargs):
        c = real_connect(*args, **kwargs)
        captured["conn"] = c
        return c

    with (
        patch("sqlite3.connect", side_effect=capturing_connect),
        patch.object(conn_mod.sqlite_vec, "load", side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            get_conn(db_path)

    conn = captured["conn"]
    try:
        # With load_extension re-disabled, this must error (SQLite emits
        # "not authorized" when extensions are off).
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("SELECT load_extension('does_not_exist')")
    finally:
        conn.close()


def test_get_conn_vec_version_mismatch_raises(db_path, monkeypatch):
    """Wrong sqlite-vec version must raise a VecVersionMismatch with upgrade hint."""
    monkeypatch.setattr(conn_mod, "PINNED_VEC_PREFIX", "v9.9.9")
    with pytest.raises(VecVersionMismatch) as excinfo:
        get_conn(db_path)
    message = str(excinfo.value)
    assert "v9.9.9" in message
    assert "uv sync" in message or "pyproject" in message
