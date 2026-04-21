"""Tests for the shared logger factory."""
from __future__ import annotations

import logging
import sys

import pytest


def _fresh_logging(monkeypatch, level: str | None = None):
    """Return the logger module with a clean lodestone root (handlers cleared).

    `get_logger` is idempotent across a process (handler install guarded by
    ``if not root.handlers``), so tests that exercise the configure path must
    reset the root between runs.
    """
    import _system.utils.logging as mod
    root = logging.getLogger("lodestone")
    for h in list(root.handlers):
        root.removeHandler(h)
    if level is not None:
        monkeypatch.setenv("LODESTONE_LOG_LEVEL", level)
    else:
        monkeypatch.delenv("LODESTONE_LOG_LEVEL", raising=False)
    return mod


def test_get_logger_returns_namespaced_child(monkeypatch):
    mod = _fresh_logging(monkeypatch)
    logger = mod.get_logger("db")
    assert logger.name == "lodestone.db"
    # Fully-qualified names are respected, not re-prefixed.
    logger2 = mod.get_logger("lodestone.db")
    assert logger2.name == "lodestone.db"
    # And the root name itself must not be re-prefixed.
    logger3 = mod.get_logger("lodestone")
    assert logger3.name == "lodestone"


def test_get_logger_is_idempotent(monkeypatch):
    """Calling get_logger repeatedly must not stack handlers on the root logger."""
    mod = _fresh_logging(monkeypatch)
    for _ in range(5):
        mod.get_logger(f"child_{_}")
    root = logging.getLogger("lodestone")
    assert len(root.handlers) == 1, (
        f"expected exactly one handler on lodestone root, got {len(root.handlers)}"
    )


def test_get_logger_emits_to_stderr(monkeypatch):
    mod = _fresh_logging(monkeypatch)
    mod.get_logger("stream_check")
    root = logging.getLogger("lodestone")
    handler = root.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr, "Lodestone logger must write to stderr only"


def test_get_logger_respects_env_level(monkeypatch):
    mod = _fresh_logging(monkeypatch, level="DEBUG")
    mod.get_logger("debug_check")
    # Assert on the lodestone root directly: a child's getEffectiveLevel()
    # would pass even if the env var never reached _configure_root (via
    # inheritance from the python root logger's default).
    assert logging.getLogger("lodestone").level == logging.DEBUG


def test_get_logger_default_level_is_info(monkeypatch):
    mod = _fresh_logging(monkeypatch)
    mod.get_logger("default_level")
    assert logging.getLogger("lodestone").level == logging.INFO


def test_lodestone_logger_does_not_propagate(monkeypatch):
    """Otherwise the root python logger would emit duplicate records."""
    mod = _fresh_logging(monkeypatch)
    mod.get_logger("propagation_check")
    root = logging.getLogger("lodestone")
    assert root.propagate is False
