"""Adapter-level tests for the three LLM providers.

Every SDK is lazy-imported inside the adapter's ``call`` function, so we
patch ``sys.modules`` before the call resolves it. No real network
traffic, no real SDK instantiation. Exception classes are defined once at
module scope so the exception instance's class identity matches the
class the adapter imports — otherwise ``except ClassA`` silently misses
instances of a sibling ``ClassA`` from a second module build.
"""
from __future__ import annotations

import sys
import types

import pytest

from _system.llm import anthropic_adapter, gemini_adapter, openai_adapter
from _system.llm.errors import LLMPermanentError, LLMTransientError


_SCHEMA = {
    "name": "demo",
    "description": "demo schema",
    "schema": {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": False,
    },
}


# ===========================================================================
# Anthropic
# ===========================================================================


class _AnthroAPIConnectionError(Exception):
    pass


class _AnthroAPITimeoutError(Exception):
    pass


class _AnthroAPIStatusError(Exception):
    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.status_code = status_code


def _install_anthropic(monkeypatch, *, create_impl):
    """Install a fake ``anthropic`` module with the class identities above."""

    class _Messages:
        def create(self, **kwargs):
            return create_impl(**kwargs)

    class _Client:
        def __init__(self, **_kwargs):
            self.messages = _Messages()

    module = types.SimpleNamespace(
        Anthropic=_Client,
        APIConnectionError=_AnthroAPIConnectionError,
        APITimeoutError=_AnthroAPITimeoutError,
        APIStatusError=_AnthroAPIStatusError,
    )
    monkeypatch.setitem(sys.modules, "anthropic", module)


def test_anthropic_success_extracts_tool_use_input(monkeypatch):
    def create(**_kwargs):
        return types.SimpleNamespace(
            content=[
                types.SimpleNamespace(type="text", text="thinking..."),
                types.SimpleNamespace(type="tool_use", name="demo", input={"x": 42}),
            ]
        )
    _install_anthropic(monkeypatch, create_impl=create)
    result = anthropic_adapter.call(
        system="s", user="u", schema=_SCHEMA, model="claude-x", temperature=1.0,
    )
    assert result == {"x": 42}


def test_anthropic_passes_temperature_to_sdk(monkeypatch):
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(
            content=[
                types.SimpleNamespace(type="tool_use", name="demo", input={"x": 1})
            ]
        )
    _install_anthropic(monkeypatch, create_impl=create)
    anthropic_adapter.call(
        system="s", user="u", schema=_SCHEMA, model="claude-x", temperature=0.3
    )
    assert captured["temperature"] == 0.3


def test_anthropic_missing_tool_use_raises_transient(monkeypatch):
    def create(**_kwargs):
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="no tool")]
        )
    _install_anthropic(monkeypatch, create_impl=create)
    with pytest.raises(LLMTransientError):
        anthropic_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="claude-x", temperature=1.0
        )


def test_anthropic_transport_error_raises_transient(monkeypatch):
    def create(**_kwargs):
        raise _AnthroAPIConnectionError("boom")
    _install_anthropic(monkeypatch, create_impl=create)
    with pytest.raises(LLMTransientError):
        anthropic_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="claude-x", temperature=1.0
        )


def test_anthropic_500_is_transient(monkeypatch):
    def create(**_kwargs):
        raise _AnthroAPIStatusError("boom", status_code=500)
    _install_anthropic(monkeypatch, create_impl=create)
    with pytest.raises(LLMTransientError):
        anthropic_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="claude-x", temperature=1.0
        )


def test_anthropic_429_is_transient(monkeypatch):
    def create(**_kwargs):
        raise _AnthroAPIStatusError("rate limit", status_code=429)
    _install_anthropic(monkeypatch, create_impl=create)
    with pytest.raises(LLMTransientError):
        anthropic_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="claude-x", temperature=1.0
        )


def test_anthropic_401_is_permanent(monkeypatch):
    def create(**_kwargs):
        raise _AnthroAPIStatusError("unauthorized", status_code=401)
    _install_anthropic(monkeypatch, create_impl=create)
    with pytest.raises(LLMPermanentError):
        anthropic_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="claude-x", temperature=1.0
        )


# ===========================================================================
# OpenAI
# ===========================================================================


class _OAIAPIConnectionError(Exception):
    pass


class _OAIAPITimeoutError(Exception):
    pass


class _OAIRateLimitError(Exception):
    pass


class _OAIAPIStatusError(Exception):
    def __init__(self, msg: str, status_code: int):
        super().__init__(msg)
        self.status_code = status_code


def _install_openai(monkeypatch, *, create_impl):
    class _Completions:
        def create(self, **kwargs):
            return create_impl(**kwargs)

    class _Chat:
        def __init__(self):
            self.completions = _Completions()

    class _Client:
        def __init__(self, **_kwargs):
            self.chat = _Chat()

    module = types.SimpleNamespace(
        OpenAI=_Client,
        APIConnectionError=_OAIAPIConnectionError,
        APITimeoutError=_OAIAPITimeoutError,
        RateLimitError=_OAIRateLimitError,
        APIStatusError=_OAIAPIStatusError,
    )
    monkeypatch.setitem(sys.modules, "openai", module)


def _openai_response(content: str | None):
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def test_openai_success_parses_json(monkeypatch):
    _install_openai(
        monkeypatch,
        create_impl=lambda **_: _openai_response('{"x": 42}'),
    )
    result = openai_adapter.call(
        system="s", user="u", schema=_SCHEMA, model="gpt-x", temperature=1.0
    )
    assert result == {"x": 42}


def test_openai_passes_temperature_to_sdk(monkeypatch):
    captured: dict[str, object] = {}

    def create(**kwargs):
        captured.update(kwargs)
        return _openai_response('{"x": 1}')
    _install_openai(monkeypatch, create_impl=create)
    openai_adapter.call(
        system="s", user="u", schema=_SCHEMA, model="gpt-x", temperature=0.4
    )
    assert captured["temperature"] == 0.4


def test_openai_empty_content_raises_transient(monkeypatch):
    _install_openai(
        monkeypatch,
        create_impl=lambda **_: _openai_response(""),
    )
    with pytest.raises(LLMTransientError):
        openai_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gpt-x", temperature=1.0
        )


def test_openai_invalid_json_raises_transient(monkeypatch):
    _install_openai(
        monkeypatch,
        create_impl=lambda **_: _openai_response("not json"),
    )
    with pytest.raises(LLMTransientError):
        openai_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gpt-x", temperature=1.0
        )


def test_openai_rate_limit_is_transient(monkeypatch):
    def create(**_):
        raise _OAIRateLimitError("slow down")
    _install_openai(monkeypatch, create_impl=create)
    with pytest.raises(LLMTransientError):
        openai_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gpt-x", temperature=1.0
        )


def test_openai_500_is_transient(monkeypatch):
    def create(**_):
        raise _OAIAPIStatusError("server", status_code=500)
    _install_openai(monkeypatch, create_impl=create)
    with pytest.raises(LLMTransientError):
        openai_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gpt-x", temperature=1.0
        )


def test_openai_401_is_permanent(monkeypatch):
    def create(**_):
        raise _OAIAPIStatusError("bad key", status_code=401)
    _install_openai(monkeypatch, create_impl=create)
    with pytest.raises(LLMPermanentError):
        openai_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gpt-x", temperature=1.0
        )


# ===========================================================================
# Gemini (google-genai)
# ===========================================================================


class _GenaiAPIError(Exception):
    def __init__(self, msg: str, *, code: int | None = None):
        super().__init__(msg)
        self.code = code


class _GenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _HttpOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_genai(monkeypatch, *, generate_impl):
    class _Models:
        def generate_content(self, *, model, contents, config):
            return generate_impl(model=model, contents=contents, config=config)

    class _Client:
        def __init__(self, **_kwargs):
            self.models = _Models()

    errors_module = types.SimpleNamespace(APIError=_GenaiAPIError)
    types_module = types.SimpleNamespace(
        GenerateContentConfig=_GenerateContentConfig,
        HttpOptions=_HttpOptions,
    )
    genai_module = types.SimpleNamespace(
        Client=_Client,
        errors=errors_module,
        types=types_module,
    )
    google_module = types.SimpleNamespace(genai=genai_module)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.errors", errors_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)


def test_gemini_success_parses_json(monkeypatch):
    _install_genai(
        monkeypatch,
        generate_impl=lambda **_: types.SimpleNamespace(text='{"x": 42}'),
    )
    result = gemini_adapter.call(
        system="s", user="u", schema=_SCHEMA, model="gemini-x", temperature=1.0
    )
    assert result == {"x": 42}


def test_gemini_passes_temperature_to_sdk(monkeypatch):
    captured: dict[str, object] = {}

    def generate(*, model, contents, config):
        captured["config"] = config
        return types.SimpleNamespace(text='{"x": 1}')
    _install_genai(monkeypatch, generate_impl=generate)
    gemini_adapter.call(
        system="s", user="u", schema=_SCHEMA, model="gemini-x", temperature=0.5
    )
    assert captured["config"].kwargs["temperature"] == 0.5


def test_gemini_empty_text_raises_transient(monkeypatch):
    _install_genai(
        monkeypatch,
        generate_impl=lambda **_: types.SimpleNamespace(text=""),
    )
    with pytest.raises(LLMTransientError):
        gemini_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gemini-x", temperature=1.0
        )


def test_gemini_invalid_json_raises_transient(monkeypatch):
    _install_genai(
        monkeypatch,
        generate_impl=lambda **_: types.SimpleNamespace(text="not json"),
    )
    with pytest.raises(LLMTransientError):
        gemini_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gemini-x", temperature=1.0
        )


def test_gemini_5xx_api_error_is_transient(monkeypatch):
    def generate(**_):
        raise _GenaiAPIError("server", code=500)
    _install_genai(monkeypatch, generate_impl=generate)
    with pytest.raises(LLMTransientError):
        gemini_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gemini-x", temperature=1.0
        )


def test_gemini_429_api_error_is_transient(monkeypatch):
    def generate(**_):
        raise _GenaiAPIError("rate limit", code=429)
    _install_genai(monkeypatch, generate_impl=generate)
    with pytest.raises(LLMTransientError):
        gemini_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gemini-x", temperature=1.0
        )


def test_gemini_401_api_error_is_permanent(monkeypatch):
    def generate(**_):
        raise _GenaiAPIError("unauthorized", code=401)
    _install_genai(monkeypatch, generate_impl=generate)
    with pytest.raises(LLMPermanentError):
        gemini_adapter.call(
            system="s", user="u", schema=_SCHEMA, model="gemini-x", temperature=1.0
        )
