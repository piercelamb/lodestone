"""Tests for ``_system.utils.http`` retry policies."""
from __future__ import annotations

import httpx
import pytest

from _system.utils import http as http_utils
from _system.utils.http import is_429, is_transient, retry_arxiv_api


def _http_status_error(status_code: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://example.test/")
    resp = httpx.Response(status_code, headers=headers or {}, request=req)
    return httpx.HTTPStatusError("err", request=req, response=resp)


# ---- is_429 / is_transient predicates ------------------------------------


def test_is_429_true_for_429() -> None:
    assert is_429(_http_status_error(429))


@pytest.mark.parametrize("sc", [500, 502, 503, 504])
def test_is_429_false_for_5xx(sc: int) -> None:
    """is_429 must NOT match 5xx — those go through retry_http, not retry_arxiv_api."""
    assert not is_429(_http_status_error(sc))


def test_is_429_false_for_transport_error() -> None:
    """is_429 must NOT match transport errors — those signal arxiv IP-throttling
    escalation; retrying makes it worse."""
    assert not is_429(httpx.ReadTimeout("read timeout"))
    assert not is_429(httpx.ConnectError("connect failed"))


def test_is_transient_still_matches_5xx_429_and_transport() -> None:
    """retry_http's predicate must keep its broader matching for non-arxiv hosts."""
    assert is_transient(_http_status_error(429))
    assert is_transient(_http_status_error(503))
    assert is_transient(httpx.ReadTimeout("read timeout"))
    assert is_transient(httpx.ConnectError("connect failed"))


# ---- _wait_arxiv_429 wait calculation -----------------------------------


def _wait(state) -> float:
    return http_utils._wait_arxiv_429()(state)


def _retry_state(attempt: int, exc: BaseException | None) -> object:
    """Minimal stand-in for tenacity.RetryCallState with the fields we read."""

    class _Outcome:
        def __init__(self, e: BaseException | None) -> None:
            self._exc = e
            self.failed = e is not None

        def exception(self) -> BaseException | None:
            return self._exc

    class _State:
        def __init__(self, n: int, e: BaseException | None) -> None:
            self.attempt_number = n
            self.outcome = _Outcome(e)

    return _State(attempt, exc)


def test_wait_honors_retry_after_seconds() -> None:
    """A numeric Retry-After header value is used verbatim."""
    err = _http_status_error(429, headers={"Retry-After": "45"})
    assert _wait(_retry_state(1, err)) == pytest.approx(45.0)


def test_wait_caps_retry_after_at_10_minutes() -> None:
    """An absurdly long Retry-After is clamped — we don't want a runaway wait."""
    err = _http_status_error(429, headers={"Retry-After": "99999"})
    assert _wait(_retry_state(1, err)) == pytest.approx(600.0)


def test_wait_falls_back_to_60s_then_120s_without_retry_after() -> None:
    """When arxiv doesn't send Retry-After (the common case), use exponential 60→120."""
    err = _http_status_error(429)
    assert _wait(_retry_state(1, err)) == pytest.approx(60.0)
    assert _wait(_retry_state(2, err)) == pytest.approx(120.0)


def test_wait_handles_garbage_retry_after() -> None:
    """A non-numeric Retry-After (e.g. HTTP-date variant) falls back to exponential."""
    err = _http_status_error(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    assert _wait(_retry_state(1, err)) == pytest.approx(60.0)


# ---- retry_arxiv_api integration ----------------------------------------


def test_retry_arxiv_api_does_not_retry_transport_errors() -> None:
    """ReadTimeout MUST NOT be retried — it indicates IP-level throttling."""
    calls: list[int] = []

    @retry_arxiv_api
    def f() -> str:
        calls.append(1)
        raise httpx.ReadTimeout("stalled")

    with pytest.raises(httpx.ReadTimeout):
        f()
    assert len(calls) == 1


def test_retry_arxiv_api_retries_429_then_succeeds(monkeypatch) -> None:
    """A 429 followed by 200 → second attempt succeeds; sleep is honored."""
    sleeps: list[float] = []
    # tenacity uses time.sleep internally — patch it so the test runs fast.
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: sleeps.append(s))

    state = {"calls": 0}

    @retry_arxiv_api
    def f() -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            raise _http_status_error(429, headers={"Retry-After": "1"})
        return "ok"

    assert f() == "ok"
    assert state["calls"] == 2
    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(1.0)


def test_retry_arxiv_api_caps_at_three_attempts(monkeypatch) -> None:
    """Persistent 429 → 3 attempts then re-raise."""
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)

    state = {"calls": 0}

    @retry_arxiv_api
    def f() -> str:
        state["calls"] += 1
        raise _http_status_error(429)

    with pytest.raises(httpx.HTTPStatusError):
        f()
    assert state["calls"] == 3


def test_retry_arxiv_api_does_not_retry_5xx(monkeypatch) -> None:
    """Only 429 retries on the arxiv API path; 503 surfaces immediately."""
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)

    state = {"calls": 0}

    @retry_arxiv_api
    def f() -> str:
        state["calls"] += 1
        raise _http_status_error(503)

    with pytest.raises(httpx.HTTPStatusError):
        f()
    assert state["calls"] == 1


# ---- progress hook bridging ----------------------------------------------


def test_progress_hook_fires_on_retry(monkeypatch) -> None:
    """When a progress hook is installed, retries emit (message, 0, 0) ticks."""
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)

    ticks: list[tuple[str, int, int]] = []
    token = http_utils.set_progress_hook(lambda m, p, t: ticks.append((m, p, t)))
    try:
        state = {"calls": 0}

        @retry_arxiv_api
        def f() -> str:
            state["calls"] += 1
            if state["calls"] < 3:
                raise _http_status_error(429)
            return "ok"

        assert f() == "ok"
    finally:
        http_utils.reset_progress_hook(token)

    # Two retries (before attempts 2 and 3); each emits one tick.
    assert len(ticks) == 2
    msg, p, total = ticks[0]
    assert "retry" in msg.lower() and "429" in msg
    assert (p, total) == (0, 0)


def test_progress_hook_unset_means_no_tick(monkeypatch) -> None:
    """With no hook installed, retries log but don't try to dispatch a tick."""
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)

    # Sanity-check no leftover hook from another test in this process.
    assert http_utils._progress_hook.get() is None

    @retry_arxiv_api
    def f() -> str:
        raise _http_status_error(429)

    with pytest.raises(httpx.HTTPStatusError):
        f()


def test_progress_hook_exception_does_not_break_retry(monkeypatch) -> None:
    """A misbehaving hook must not derail the retry loop."""
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda s: None)

    def bad_hook(*_args, **_kwargs):
        raise RuntimeError("hook is broken")

    token = http_utils.set_progress_hook(bad_hook)
    try:
        state = {"calls": 0}

        @retry_arxiv_api
        def f() -> str:
            state["calls"] += 1
            if state["calls"] < 2:
                raise _http_status_error(429)
            return "ok"

        assert f() == "ok"
    finally:
        http_utils.reset_progress_hook(token)
