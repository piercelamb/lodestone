"""Tests for :func:`_system.llm.client.call_structured`."""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict
from tenacity import wait_random_exponential

from _system.llm import client as llm_client
from _system.llm.config import Provider
from _system.llm.errors import (
    LLMPermanentError,
    LLMTransientError,
    ProviderUnconfigured,
)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int


@pytest.fixture
def patch_resolve(monkeypatch):
    monkeypatch.setattr(
        llm_client, "resolve_provider",
        lambda: llm_client.__dict__["ResolvedProvider_fake_type"]  # replaced below
        if False else _fake_resolved(),
    )


def _fake_resolved(
    provider: Provider = Provider.ANTHROPIC,
    model: str = "m",
    temperature: float = 1.0,
):
    # ``call_structured`` unpacks via ``provider, model, temperature =
    # resolve_provider()`` — a plain tuple suffices.
    return (provider, model, temperature)


def _neuter_sleep(monkeypatch):
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda _s: None)


def test_success_returns_validated_model(monkeypatch):
    monkeypatch.setattr(llm_client, "resolve_provider", _fake_resolved)

    calls = {"n": 0}

    def _fake_anthropic_call(**_kwargs):
        calls["n"] += 1
        return {"x": 7}

    monkeypatch.setattr(
        llm_client,
        "_dispatch",
        lambda provider: _fake_anthropic_call,
    )

    result = llm_client.call_structured(
        system="s", user="u", schema={"name": "n", "description": "d", "schema": {}},
        response_model=_Payload,
    )
    assert isinstance(result, _Payload)
    assert result.x == 7
    assert calls["n"] == 1


def test_transient_error_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_client, "resolve_provider", _fake_resolved)
    _neuter_sleep(monkeypatch)

    calls = {"n": 0}

    def _flaky(**_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMTransientError(f"boom {calls['n']}")
        return {"x": 1}

    monkeypatch.setattr(llm_client, "_dispatch", lambda provider: _flaky)

    result = llm_client.call_structured(
        system="s", user="u", schema={"name": "n", "description": "d", "schema": {}},
        response_model=_Payload,
    )
    assert result.x == 1
    assert calls["n"] == 3


def test_permanent_error_surfaces_immediately(monkeypatch):
    monkeypatch.setattr(llm_client, "resolve_provider", _fake_resolved)
    _neuter_sleep(monkeypatch)

    calls = {"n": 0}

    def _perm(**_kwargs):
        calls["n"] += 1
        raise LLMPermanentError("no retry")

    monkeypatch.setattr(llm_client, "_dispatch", lambda provider: _perm)

    with pytest.raises(LLMPermanentError):
        llm_client.call_structured(
            system="s", user="u", schema={"name": "n", "description": "d", "schema": {}},
            response_model=_Payload,
        )
    assert calls["n"] == 1


def test_four_straight_transient_errors_surface_last(monkeypatch):
    monkeypatch.setattr(llm_client, "resolve_provider", _fake_resolved)
    _neuter_sleep(monkeypatch)

    calls = {"n": 0}

    def _bust(**_kwargs):
        calls["n"] += 1
        raise LLMTransientError(f"attempt {calls['n']}")

    monkeypatch.setattr(llm_client, "_dispatch", lambda provider: _bust)

    with pytest.raises(LLMTransientError) as exc_info:
        llm_client.call_structured(
            system="s", user="u", schema={"name": "n", "description": "d", "schema": {}},
            response_model=_Payload,
        )
    assert calls["n"] == 4
    assert "attempt 4" in str(exc_info.value)


def test_validation_error_wrapped_as_transient_and_retried(monkeypatch):
    monkeypatch.setattr(llm_client, "resolve_provider", _fake_resolved)
    _neuter_sleep(monkeypatch)

    calls = {"n": 0}

    def _drifter(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"not_a_valid_field": "oops"}
        return {"x": 1}

    monkeypatch.setattr(llm_client, "_dispatch", lambda provider: _drifter)

    result = llm_client.call_structured(
        system="s", user="u", schema={"name": "n", "description": "d", "schema": {}},
        response_model=_Payload,
    )
    assert result.x == 1
    assert calls["n"] == 2


def test_provider_unconfigured_not_retried(monkeypatch):
    _neuter_sleep(monkeypatch)

    def _raise(*_a, **_k):
        raise ProviderUnconfigured("no provider")

    monkeypatch.setattr(llm_client, "resolve_provider", _raise)
    monkeypatch.setattr(llm_client, "_dispatch", lambda provider: None)

    with pytest.raises(ProviderUnconfigured):
        llm_client.call_structured(
            system="s", user="u", schema={"name": "n", "description": "d", "schema": {}},
            response_model=_Payload,
        )


def test_retry_wait_strategy_is_random_exponential():
    # The tenacity.retry decorator stores its wait strategy on the wrapped
    # function. We assert the *type* here; timing is not exercised.
    retry_obj = llm_client.call_structured.retry
    assert isinstance(retry_obj.wait, wait_random_exponential)
    assert retry_obj.stop.max_attempt_number == 4
