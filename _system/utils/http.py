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

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

USER_AGENT = "Lodestone/1.0 (mailto:pierce.lamb@getwhys.io)"

_DEFAULT_TIMEOUT_S = 30.0


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
    aggressively; a short backoff usually clears it.
    """
    if isinstance(exc, httpx.HTTPStatusError):
        sc = exc.response.status_code
        return sc == 429 or 500 <= sc < 600
    return isinstance(exc, httpx.TransportError)


retry_http = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=0.5, max=4.0),
    retry=retry_if_exception(is_transient),
    reraise=True,
)
