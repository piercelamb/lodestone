"""Smoke tests for lodestone/AGENTS.md — the agent navigation contract.

The file is required to live at the repo root and to cover every semantic
anchor asserted below. If the section 14 spec grows, update the required
phrase list here alongside the document.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AGENTS_MD = _REPO_ROOT / "lodestone" / "AGENTS.md"


def test_agents_md_exists():
    assert AGENTS_MD.is_file(), f"AGENTS.md not found at {AGENTS_MD}"


def test_agents_md_line_count_in_range():
    lines = AGENTS_MD.read_text(encoding="utf-8").splitlines()
    assert 80 <= len(lines) <= 200, (
        f"AGENTS.md has {len(lines)} lines; expected 80–200"
    )


def _body() -> str:
    return AGENTS_MD.read_text(encoding="utf-8")


@pytest.mark.parametrize("phrase", [
    "search.py",
    "ingest.py",
    "resumable",
    "--needs-review",
    "--read",
    "--figure",
    "--human",
    "aliases",
    "provenance",
])
def test_agents_md_contains_required_phrase(phrase):
    body = _body()
    assert phrase in body, f"AGENTS.md missing required phrase: {phrase!r}"
