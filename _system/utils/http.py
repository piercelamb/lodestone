"""Shared httpx client + retry helpers.

Factored out of :mod:`_system.scripts.fetch_paper` so other fetch paths
(``fetch_post``, future ``fetch_repo`` HTTP calls) reuse the same UA,
timeout, and tenacity retry policy. Keeping one shape avoids the arxiv
rate-limit bucket trap (the ``arxiv`` Python lib's hardcoded
``arxiv.py/<v>`` UA shares a global throttle bucket and 429s constantly;
our project UA carries a contact email and lands in the normal-citizen
class).
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Callable, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from _system.utils.logging import get_logger

_LOG = get_logger("utils.http")

USER_AGENT = "Lodestone/1.0 (mailto:richard.pierce.lamb@gmail.com)"

_DEFAULT_TIMEOUT_S = 30.0

# Optional out-of-band progress channel. The MCP server installs an
# ``(message, done, total)`` callback here for the duration of an
# ingest tool call so HTTP retries (429 backoff, transport blips on
# content fetches) surface in the client UI, not just stderr. CLI
# callers leave this unset and retries only emit WARNING log lines.
ProgressFn = Callable[[str, int, int], None]
_progress_hook: ContextVar[Optional[ProgressFn]] = ContextVar(
    "lodestone_http_progress_hook", default=None,
)


def set_progress_hook(callback: Optional[ProgressFn]):
    """Install a progress callback for the current context.

    Returns a token for ``reset_progress_hook``. ContextVars scope this
    to the current task / thread so concurrent MCP requests don't
    cross-pollinate.
    """
    return _progress_hook.set(callback)


def reset_progress_hook(token) -> None:
    _progress_hook.reset(token)


def _summarize_exc(exc: BaseException) -> str:
    """One-line summary of a retry-triggering exception for log lines."""
    if isinstance(exc, httpx.HTTPStatusError):
        url = str(exc.request.url) if exc.request is not None else "<unknown>"
        return f"HTTP {exc.response.status_code} {url}"
    return f"{type(exc).__name__}: {exc}"


def _log_retry(retry_state) -> None:
    """tenacity ``before_sleep`` hook: log retry + emit MCP progress if hook set."""
    outcome = retry_state.outcome
    if outcome is None or not outcome.failed:
        return
    exc = outcome.exception()
    fn = retry_state.fn.__name__ if retry_state.fn is not None else "<callable>"
    wait = retry_state.next_action.sleep if retry_state.next_action else 0.0
    summary = _summarize_exc(exc)
    _LOG.warning(
        "retry %s attempt=%d wait=%.1fs reason=%s",
        fn, retry_state.attempt_number, wait, summary,
    )
    cb = _progress_hook.get()
    if cb is not None:
        # progress=0/total=0 keeps the surrounding stage's progress bar
        # untouched — the message field carries the diagnostic.
        try:
            cb(
                f"retry {fn} (attempt {retry_state.attempt_number}, "
                f"wait {wait:.0f}s) — {summary}",
                0, 0,
            )
        except Exception as cb_exc:  # noqa: BLE001
            _LOG.warning("progress hook raised, ignoring: %r", cb_exc)


def make_default_client() -> httpx.Client:
    """Construct an httpx client with the project UA, 30s timeout, follow_redirects."""
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=_DEFAULT_TIMEOUT_S,
        follow_redirects=True,
    )


def is_transient(exc: BaseException) -> bool:
    """Tenacity predicate: retry on 5xx, 429, or transport errors.

    429 is included specifically because arxiv's Atom export throttles
    aggressively. arxiv asks callers to keep a >=3s gap between API
    hits, and a sustained 429 window can last tens of seconds.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        sc = exc.response.status_code
        return sc == 429 or 500 <= sc < 600
    return isinstance(exc, httpx.TransportError)


# 6 attempts with deterministic exponential backoff: 3s, 6s, 12s, 24s,
# 48s (capped at 60). Total wait budget ~93s before re-raising — sized
# to outlast a typical arxiv 429 window. No jitter: lodestone is a
# single-user tool, no thundering-herd risk.
retry_http = retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    retry=retry_if_exception(is_transient),
    before_sleep=_log_retry,
    reraise=True,
)


def is_429(exc: BaseException) -> bool:
    """Predicate matching only HTTP 429 — used by ``retry_arxiv_api``.

    Deliberately *excludes* ``httpx.TransportError`` (read timeout,
    connection reset). Per arxiv community evidence, those are arxiv's
    escalation mode for repeat offenders — the server accepts the TCP
    connection then never responds. Retrying just holds the connection
    open and deepens the throttle. Surface it instead so the caller
    can back off cleanly.
    """
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == 429
    )


class _wait_arxiv_429(wait_base):
    """Wait honoring the ``Retry-After`` header on 429, else 60s/120s.

    arxiv doesn't currently send ``Retry-After`` on its 429s (community
    has confirmed this in mailing-list threads), so the fallback path
    is what actually fires today. The header support is forward-compat
    for if/when they add one — there's no reason not to honor it.
    """

    _MAX_WAIT_S = 600.0  # arxiv 429 windows can run a minute or two; 10min is the ceiling
    _BASE_WAIT_S = 60.0

    def __call__(self, retry_state) -> float:
        outcome = retry_state.outcome
        exc = outcome.exception() if (outcome is not None and outcome.failed) else None
        if isinstance(exc, httpx.HTTPStatusError):
            ra = exc.response.headers.get("Retry-After")
            if ra:
                try:
                    return min(float(ra), self._MAX_WAIT_S)
                except ValueError:
                    pass
        # 60s, 120s.
        attempt = max(retry_state.attempt_number, 1)
        return min(self._BASE_WAIT_S * (2 ** (attempt - 1)), self._MAX_WAIT_S)


# arxiv's metadata API needs its own retry policy, distinct from the
# generic ``retry_http``: only retry 429 (honoring Retry-After), never
# retry transport errors (those signal IP-level escalation — see the
# is_429 docstring), and cap at 3 attempts so a sustained throttle
# event surfaces in seconds, not minutes.
retry_arxiv_api = retry(
    stop=stop_after_attempt(3),
    wait=_wait_arxiv_429(),
    retry=retry_if_exception(is_429),
    before_sleep=_log_retry,
    reraise=True,
)
