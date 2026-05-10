"""OpenAI chat.completions adapter — structured output via json_schema.

The ``response_format={"type": "json_schema", "json_schema": {...,
"strict": true}}`` mode returns a single string message whose content is
guaranteed-parseable JSON matching the declared schema. One
``json.loads`` and we're done.

Lazy import: ``openai`` is only imported when :func:`call` runs.
"""
from __future__ import annotations

import json
from typing import Any

from _system.llm.errors import LLMPermanentError, LLMTransientError


MODEL_CATALOG: list[tuple[str, str]] = [
    ("gpt-5.4", "most capable, recommended"),
    ("gpt-5.4-mini", "faster, lower cost"),
    ("gpt-5.4-pro", "deepest reasoning"),
    ("gpt-5.4-nano", "cheapest, fastest"),
]
DEFAULT_MODEL = MODEL_CATALOG[0][0]

# Cap the per-call wall-clock at 180s. The OpenAI SDK's default is 600s,
# which on the MCP server (single-threaded STDIO) means a single request
# stranded by macOS sleep blocks every subsequent tool call for 10
# minutes while the read on a dead socket waits for FIN. 180s is enough
# for slow generations and tight enough to unwedge quickly post-sleep.
_TIMEOUT_S = 180.0


def call(
    *,
    system: str,
    user: str,
    schema: dict,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    import openai
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
    )

    client = openai.OpenAI(timeout=_TIMEOUT_S)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": True,
                },
            },
        )
    except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
        raise LLMTransientError(f"openai transport error: {exc}") from exc
    except APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        if status is not None and (status == 429 or 500 <= status < 600):
            raise LLMTransientError(f"openai {status}: {exc}") from exc
        raise LLMPermanentError(f"openai {status}: {exc}") from exc

    if not response.choices:
        raise LLMTransientError("openai response had no choices")
    content = response.choices[0].message.content
    if content is None or content == "":
        raise LLMTransientError("openai response content empty")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMTransientError(f"openai response not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMTransientError(
            f"openai response JSON was not an object: {type(payload).__name__}"
        )
    return payload
