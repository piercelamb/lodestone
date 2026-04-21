"""Shared pytest fixtures for Lodestone unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Per-test sqlite file. Never shared between tests."""
    return tmp_path / "lodestone.db"


@pytest.fixture
def conn(db_path: Path):
    """Fresh, migrated Lodestone connection."""
    c = get_conn(db_path)
    init_db(c)
    try:
        yield c
    finally:
        c.close()
