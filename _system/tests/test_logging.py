"""Tests for the shared logger factory."""
from __future__ import annotations

import importlib
import logging
import sys

import pytest


def _reimport_logging_module(monkeypatch, level: str | None = None):
    """Reset the module-level ``_configured`` flag and re-import the logger module.

    Necessary because :func:`get_logger` is idempotent across a process; tests
    that exercise the configure path need a clean slate.
    """
    import _system.utils.logging as mod
    # Reset module state so the next get_logger() re-runs _configure_root.
    monkeypatch.setattr(mod, "_configured", False)
    # Clear the lodestone root logger's handlers so we can observe re-installation.
    root = logging.getLogger("lodestone")
    for h in list(root.handlers):
        root.removeHandler(h)
    if level is not None:
        monkeypatch.setenv("LODESTONE_LOG_LEVEL", level)
    else:
        monkeypatch.delenv("LODESTONE_LOG_LEVEL", raising=False)
    return mod


def test_get_logger_returns_namespaced_child(monkeypatch):
    mod = _reimport_logging_module(monkeypatch)
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
    mod = _reimport_logging_module(monkeypatch)
    for _ in range(5):
        mod.get_logger(f"child_{_}")
    root = logging.getLogger("lodestone")
    assert len(root.handlers) == 1, (
        f"expected exactly one handler on lodestone root, got {len(root.handlers)}"
    )


def test_get_logger_emits_to_stderr(monkeypatch):
    mod = _reimport_logging_module(monkeypatch)
    mod.get_logger("stream_check")
    root = logging.getLogger("lodestone")
    handler = root.handlers[0]
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr, "Lodestone logger must write to stderr only"


def test_get_logger_respects_env_level(monkeypatch):
    mod = _reimport_logging_module(monkeypatch, level="DEBUG")
    logger = mod.get_logger("debug_check")
    assert logger.getEffectiveLevel() == logging.DEBUG


def test_get_logger_default_level_is_info(monkeypatch):
    mod = _reimport_logging_module(monkeypatch)
    logger = mod.get_logger("default_level")
    assert logger.getEffectiveLevel() == logging.INFO


def test_lodestone_logger_does_not_propagate(monkeypatch):
    """Otherwise the root python logger would emit duplicate records."""
    mod = _reimport_logging_module(monkeypatch)
    mod.get_logger("propagation_check")
    root = logging.getLogger("lodestone")
    assert root.propagate is False
