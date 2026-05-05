"""Unit tests for _system/scripts/create_domain.py."""
from __future__ import annotations

import json
import re

import pytest

from _system.scripts import create_domain
from _system.utils.slug import DOMAIN_MAX_LEN


def test_create_domain_inserts_row(conn):
    result = create_domain.create_domain(
        conn=conn, name="rag", description="Retrieval-augmented generation"
    )
    assert result["created"] is True
    row = conn.execute(
        "SELECT name, description FROM domains WHERE name = ?", ("rag",)
    ).fetchone()
    assert row == ("rag", "Retrieval-augmented generation")


def test_create_domain_is_idempotent(conn):
    first = create_domain.create_domain(conn=conn, name="dup", description="x")
    second = create_domain.create_domain(conn=conn, name="dup", description="y")
    assert first["created"] is True
    assert second["created"] is False
    # Only one row, and description is still the original (we INSERT OR IGNORE).
    rows = conn.execute(
        "SELECT description FROM domains WHERE name = ?", ("dup",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "x"


def test_create_domain_bad_name_raises(conn):
    with pytest.raises(ValueError, match=re.escape("RAG!")):
        create_domain.create_domain(conn=conn, name="RAG!", description="x")


def test_create_domain_empty_name_raises(conn):
    with pytest.raises(ValueError):
        create_domain.create_domain(conn=conn, name="", description="x")


def test_create_domain_long_name_raises(conn):
    with pytest.raises(ValueError):
        create_domain.create_domain(
            conn=conn, name="a" * (DOMAIN_MAX_LEN + 1), description="x"
        )


def test_create_domain_accepts_hyphens_underscores_digits(conn):
    r = create_domain.create_domain(conn=conn, name="time-series_2", description="...")
    assert r["created"] is True


def test_cli_prints_json(tmp_path, capsys):
    db_path = tmp_path / "lodestone.db"
    # init_db runs lazily inside create_domain.main, so we don't pre-create.
    create_domain.main([
        "--name", "rag",
        "--description", "Retrieval-augmented generation",
        "--db", str(db_path),
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload == {"name": "rag", "created": True}


def test_cli_idempotent_returns_created_false(tmp_path, capsys):
    db_path = tmp_path / "lodestone.db"
    create_domain.main(["--name", "a", "--description", "d", "--db", str(db_path)])
    capsys.readouterr()  # drop first payload
    create_domain.main(["--name", "a", "--description", "d", "--db", str(db_path)])
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload == {"name": "a", "created": False}
