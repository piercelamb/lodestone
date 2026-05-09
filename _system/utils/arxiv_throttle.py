"""Process-wide rate limiter for arxiv API calls.

arxiv's `Terms of Use <https://info.arxiv.org/help/api/tou.html>`_ asks
callers to keep ``>=3 seconds`` between requests to
``export.arxiv.org/api/query`` (and OAI-PMH), with at most one
connection in flight. Violating this triggers HTTP 429s; sustained
violations escalate to silent connection-level throttling — TCP
accepts but the server never sends response headers, so each retry
attempt eats the full read timeout and *worsens* the offense.

Lodestone is a CLI: each ``ingest`` invocation is a separate process,
so an in-memory limiter resets every run and won't prevent the user
from triggering 429s by ingesting several papers back-to-back. We
persist the last-request timestamp to a small file under
``~/.lodestone/`` and ``flock`` it for cross-process safety.

Override hooks (mainly for tests / ad-hoc tools):
- ``LODESTONE_ARXIV_THROTTLE_PATH`` — alternate state file path
- ``LODESTONE_ARXIV_MIN_GAP_S`` — alternate min gap (set to ``0`` to disable)
"""
from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

from _system.utils.logging import get_logger

_LOG = get_logger("utils.arxiv_throttle")

# 3.0s is arxiv's documented minimum; 100ms safety margin covers clock
# skew and the time it takes the request to actually hit their server.
_DEFAULT_MIN_GAP_S = 3.1
_DEFAULT_PATH = Path.home() / ".lodestone" / "arxiv_last_request.txt"


def _state_path() -> Path:
    override = os.environ.get("LODESTONE_ARXIV_THROTTLE_PATH")
    if override:
        return Path(override)
    return _DEFAULT_PATH


def _min_gap_s() -> float:
    override = os.environ.get("LODESTONE_ARXIV_MIN_GAP_S")
    if override is not None:
        try:
            return float(override)
        except ValueError:
            _LOG.warning(
                "invalid LODESTONE_ARXIV_MIN_GAP_S=%r; using default %.1fs",
                override, _DEFAULT_MIN_GAP_S,
            )
    return _DEFAULT_MIN_GAP_S


def wait_for_arxiv_slot() -> None:
    """Block until the configured min-gap has elapsed since the last arxiv API call.

    The state file holds a single ASCII float — the unix-epoch timestamp
    of the most recent successful slot acquisition. Concurrent processes
    serialize on an exclusive ``flock`` so two ``ingest`` calls launched
    at the same instant don't both think they have a free slot.
    """
    gap = _min_gap_s()
    if gap <= 0:
        return

    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            raw = os.read(fd, 64).decode("utf-8", errors="ignore").strip()
            try:
                last = float(raw) if raw else 0.0
            except ValueError:
                last = 0.0
            now = time.time()
            wait = (last + gap) - now
            if wait > 0:
                _LOG.info(
                    "arxiv throttle: sleeping %.2fs to honor %.1fs min gap",
                    wait, gap,
                )
                time.sleep(wait)
            os.lseek(fd, 0, 0)
            os.ftruncate(fd, 0)
            os.write(fd, f"{time.time():.6f}\n".encode("ascii"))
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
