"""Tests for ``_default_arxiv_lookup``'s post-retry error wrapper.

The wrapper lives at ``_system.scripts.fetch_paper._default_arxiv_lookup``
and converts ``httpx`` exceptions that escape ``_arxiv_api_get`` (after
tenacity has exhausted its budget) into a more actionable
``RuntimeError``. The wrapped message lands verbatim in the MCP error
envelope via ``_tool_error(msg_id, repr(exc))``.
"""
from __future__ import annotations

import httpx
import pytest

from _system.scripts import fetch_paper as fp


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("GET", "https://export.arxiv.org/api/query")
    resp = httpx.Response(status_code, request=req)
    return httpx.HTTPStatusError("err", request=req, response=resp)


def test_default_arxiv_lookup_wraps_transport_error_with_friendly_message(monkeypatch) -> None:
    """ReadTimeout escaping the retry budget → RuntimeError mentioning arxiv + recovery hint."""
    def boom(_arxiv_id: str) -> str:
        raise httpx.ReadTimeout("The read operation timed out")

    monkeypatch.setattr(fp, "_arxiv_api_get", boom)

    with pytest.raises(RuntimeError) as exc_info:
        fp._default_arxiv_lookup("2605.12213")

    msg = str(exc_info.value)
    assert "arxiv" in msg.lower()
    assert "2605.12213" in msg
    # Recovery hint references at least one of throttling / CDN / network.
    assert any(token in msg.lower() for token in ("throttl", "cdn", "network"))
    # Cause chain preserved for stderr / `_log("error", ...)`.
    assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)


def test_default_arxiv_lookup_wraps_429_after_retry_exhaustion(monkeypatch) -> None:
    """HTTP 429 escaping the retry budget → friendly RuntimeError with rate-limit framing."""
    def boom(_arxiv_id: str) -> str:
        raise _http_status_error(429)

    monkeypatch.setattr(fp, "_arxiv_api_get", boom)

    with pytest.raises(RuntimeError) as exc_info:
        fp._default_arxiv_lookup("2605.12213")

    msg = str(exc_info.value)
    assert "429" in msg
    assert "2605.12213" in msg
    assert "rate" in msg.lower() or "throttl" in msg.lower() or "limit" in msg.lower()
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


def test_default_arxiv_lookup_wraps_503_after_retry_exhaustion(monkeypatch) -> None:
    """HTTP 503 escaping the retry budget → friendly RuntimeError with rate-limit framing."""
    def boom(_arxiv_id: str) -> str:
        raise _http_status_error(503)

    monkeypatch.setattr(fp, "_arxiv_api_get", boom)

    with pytest.raises(RuntimeError) as exc_info:
        fp._default_arxiv_lookup("2605.12213")

    msg = str(exc_info.value)
    assert "503" in msg
    assert "2605.12213" in msg
    assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


def test_default_arxiv_lookup_does_not_wrap_404(monkeypatch) -> None:
    """404 must propagate as raw HTTPStatusError; the caller (`fetch()`) handles it."""
    def boom(_arxiv_id: str) -> str:
        raise _http_status_error(404)

    monkeypatch.setattr(fp, "_arxiv_api_get", boom)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        fp._default_arxiv_lookup("9999.99999")
    assert exc_info.value.response.status_code == 404
