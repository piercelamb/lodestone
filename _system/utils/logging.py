"""Shared logger factory for Lodestone.

All logs go to **stderr**; stdout is reserved for the JSON emitted by
:mod:`_system.scripts.search` and friends. Level is controlled by the
``LODESTONE_LOG_LEVEL`` env var (default ``INFO``).
"""
from __future__ import annotations

import logging
import os
import sys

_ROOT_LOGGER_NAME = "lodestone"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    level_name = os.environ.get("LODESTONE_LOG_LEVEL", "INFO").upper()
    root.setLevel(level_name)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a Lodestone-namespaced logger configured once per process."""
    _configure_root()
    if name == _ROOT_LOGGER_NAME or name.startswith(f"{_ROOT_LOGGER_NAME}."):
        full = name
    else:
        full = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(full)
