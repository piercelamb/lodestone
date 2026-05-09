"""Tests for ``_system.utils.arxiv_throttle``."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from _system.utils import arxiv_throttle


@pytest.fixture(autouse=True)
def _isolate_throttle(monkeypatch, tmp_path: Path):
    """Redirect the throttle's state file to a per-test tmp path.

    Without this, every test would race against the user's real
    ``~/.lodestone/arxiv_last_request.txt`` and stomp it.
    """
    monkeypatch.setenv("LODESTONE_ARXIV_THROTTLE_PATH", str(tmp_path / "arxiv_last.txt"))


def test_first_call_does_not_sleep(monkeypatch):
    """A fresh state file means no prior call → wait_for_arxiv_slot returns immediately."""
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv_throttle.time, "sleep", lambda s: sleeps.append(s))
    arxiv_throttle.wait_for_arxiv_slot()
    assert sleeps == []


def test_second_call_within_gap_sleeps(monkeypatch):
    """Two back-to-back calls — the second sleeps for ~the configured gap."""
    monkeypatch.setenv("LODESTONE_ARXIV_MIN_GAP_S", "1.0")
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv_throttle.time, "sleep", lambda s: sleeps.append(s))

    arxiv_throttle.wait_for_arxiv_slot()
    arxiv_throttle.wait_for_arxiv_slot()

    assert len(sleeps) == 1
    # Should sleep close to 1.0s (minus the few microseconds between calls).
    assert 0.5 < sleeps[0] <= 1.0


def test_call_outside_gap_does_not_sleep(monkeypatch, tmp_path: Path):
    """A previous timestamp older than the gap → no sleep."""
    monkeypatch.setenv("LODESTONE_ARXIV_MIN_GAP_S", "1.0")
    state = tmp_path / "arxiv_last.txt"
    state.write_text(f"{time.time() - 5.0:.6f}\n")
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv_throttle.time, "sleep", lambda s: sleeps.append(s))

    arxiv_throttle.wait_for_arxiv_slot()

    assert sleeps == []


def test_zero_gap_disables_throttle(monkeypatch, tmp_path: Path):
    """Setting LODESTONE_ARXIV_MIN_GAP_S=0 short-circuits even with recent state."""
    monkeypatch.setenv("LODESTONE_ARXIV_MIN_GAP_S", "0")
    state = tmp_path / "arxiv_last.txt"
    state.write_text(f"{time.time():.6f}\n")
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv_throttle.time, "sleep", lambda s: sleeps.append(s))

    arxiv_throttle.wait_for_arxiv_slot()

    assert sleeps == []


def test_corrupt_state_file_treated_as_no_prior_call(monkeypatch, tmp_path: Path):
    """A garbled state file shouldn't crash — treat as 'no prior call'."""
    monkeypatch.setenv("LODESTONE_ARXIV_MIN_GAP_S", "1.0")
    state = tmp_path / "arxiv_last.txt"
    state.write_text("not a number\n")
    sleeps: list[float] = []
    monkeypatch.setattr(arxiv_throttle.time, "sleep", lambda s: sleeps.append(s))

    arxiv_throttle.wait_for_arxiv_slot()

    assert sleeps == []


def test_state_file_is_written(monkeypatch, tmp_path: Path):
    """After a call, the state file holds a parseable recent timestamp."""
    state = tmp_path / "arxiv_last.txt"
    monkeypatch.setattr(arxiv_throttle.time, "sleep", lambda s: None)
    before = time.time()
    arxiv_throttle.wait_for_arxiv_slot()
    after = time.time()

    raw = state.read_text().strip()
    written = float(raw)
    assert before <= written <= after


def test_invalid_min_gap_env_falls_back_to_default(monkeypatch):
    """Bad LODESTONE_ARXIV_MIN_GAP_S → fall back to the 3.1s default (logged)."""
    monkeypatch.setenv("LODESTONE_ARXIV_MIN_GAP_S", "not-a-float")
    assert arxiv_throttle._min_gap_s() == pytest.approx(3.1)


def test_state_path_env_override(monkeypatch, tmp_path: Path):
    """LODESTONE_ARXIV_THROTTLE_PATH chooses the state file location."""
    custom = tmp_path / "custom" / "arxiv.txt"
    monkeypatch.setenv("LODESTONE_ARXIV_THROTTLE_PATH", str(custom))
    assert arxiv_throttle._state_path() == custom
