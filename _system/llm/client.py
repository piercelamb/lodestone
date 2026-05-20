"""One-shot dispatch: provider-agnostic structured LLM call.

``call_structured`` is the only public entry point the pipeline uses.
Steps per attempt:

1. ``resolve_provider()`` (config + env) returns
   ``(Provider, model, temperature)``.
2. Dispatch to the per-provider adapter's ``call(...)`` which returns
   an already-parsed JSON dict (not text).
3. ``response_model.model_validate(raw)`` turns the dict into a
   Pydantic model; validation errors are re-raised as
   :class:`LLMTransientError` so retries burn cheap, not expensive.

Retry policy (per project LLM conventions):

- 4 attempts with ``wait_random_exponential(multiplier=1, min=1, max=30)``
  (exponential backoff + jitter).
- Only :class:`LLMTransientError` is retried. 4xx bad-request, auth
  failures, and provider-config errors surface immediately.
"""
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from _system.llm.config import Provider
from _system.llm.errors import LLMTransientError
from _system.llm.selection import resolve_provider
from _system.utils.http import heartbeat_sleep


T = TypeVar("T", bound=BaseModel)


def _dispatch(provider: Provider):
    """Lazy-load the adapter module for ``provider``.

    Each adapter's SDK import sits inside its ``call`` function, so this
    module-level import chain never triggers a heavy SDK load; only the
    adapter's thin Python wrapper imports here.
    """
    if provider is Provider.ANTHROPIC:
        from _system.llm import anthropic_adapter

        return anthropic_adapter.call
    if provider is Provider.OPENAI:
        from _system.llm import openai_adapter

        return openai_adapter.call
    if provider is Provider.GEMINI:
        from _system.llm import gemini_adapter

        return gemini_adapter.call
    raise RuntimeError(f"unknown provider {provider!r}")  # pragma: no cover


@retry(
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type(LLMTransientError),
    sleep=heartbeat_sleep,
    reraise=True,
)
def call_structured(
    *,
    system: str,
    user: str,
    schema: dict,
    response_model: type[T],
) -> T:
    """Run one LLM call and return a validated Pydantic model instance."""
    provider, model, temperature = resolve_provider()
    adapter_call = _dispatch(provider)
    raw = adapter_call(
        system=system,
        user=user,
        schema=schema,
        model=model,
        temperature=temperature,
    )
    try:
        return response_model.model_validate(raw)
    except ValidationError as exc:
        raise LLMTransientError(
            f"response failed schema validation for {response_model.__name__}: {exc}"
        ) from exc
