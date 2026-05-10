"""Anthropic Messages API adapter — structured output via tool_use.

Provider-side structured output: we declare our schema as the *only* tool
the model can call, force ``tool_choice`` to it, and pull the parsed
``input`` dict directly off the response. No JSON re-parse, no text
envelope to strip. Schema drift surfaces as an Anthropic 400 before any
billable tokens land.

Lazy import: ``anthropic`` is only imported when :func:`call` runs. Users
on another provider never pay the import cost.
"""
from __future__ import annotations

from typing import Any

from _system.llm.errors import LLMPermanentError, LLMTransientError


# Known-good models for this provider, in recommended order. The first
# entry is the default when the user doesn't pin one in their config.
# Update when newer models ship; selection.py surfaces this catalog to
# users in the interactive model-picker.
MODEL_CATALOG: list[tuple[str, str]] = [
    ("claude-opus-4-7", "most capable, recommended"),
    ("claude-sonnet-4-6", "balanced speed + intelligence"),
    ("claude-haiku-4-5", "fastest, lowest cost"),
]
DEFAULT_MODEL = MODEL_CATALOG[0][0]
_MAX_TOKENS = 4096

# See openai_adapter._TIMEOUT_S for rationale — same wedge mechanic.
_TIMEOUT_S = 180.0


def call(
    *,
    system: str,
    user: str,
    schema: dict,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    import anthropic
    from anthropic import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
    )

    client = anthropic.Anthropic(timeout=_TIMEOUT_S)
    tool = {
        "name": schema["name"],
        "description": schema["description"],
        "input_schema": schema["schema"],
    }
    try:
        response = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise LLMTransientError(f"anthropic transport error: {exc}") from exc
    except APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        if status is not None and (status == 429 or 500 <= status < 600):
            raise LLMTransientError(
                f"anthropic {status}: {exc}"
            ) from exc
        raise LLMPermanentError(
            f"anthropic {status}: {exc}"
        ) from exc

    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
            payload = block.input
            if not isinstance(payload, dict):
                raise LLMTransientError(
                    f"anthropic tool_use returned non-dict input: {type(payload).__name__}"
                )
            return payload
    raise LLMTransientError(
        "anthropic response missing expected tool_use block"
    )
