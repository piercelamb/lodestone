"""Gemini (google-genai) adapter — structured output via responseSchema.

Uses the new ``google-genai`` SDK (*not* the legacy
``google-generativeai``). ``response_mime_type="application/json"`` +
``response_schema=<dict>`` makes Gemini return a JSON string guaranteed
to match the schema. Parse once, return the dict.

Lazy import: ``google.genai`` is only imported when :func:`call` runs.
"""
from __future__ import annotations

import json
from typing import Any

from _system.llm.errors import LLMPermanentError, LLMTransientError


MODEL_CATALOG: list[tuple[str, str]] = [
    ("gemini-2.5-pro", "most capable GA, recommended"),
    ("gemini-2.5-flash", "fast, balanced"),
    ("gemini-2.5-flash-lite", "fastest, lowest cost"),
    ("gemini-3.1-pro-preview", "preview — experimental frontier"),
]
DEFAULT_MODEL = MODEL_CATALOG[0][0]

# See openai_adapter._TIMEOUT_S for rationale. google-genai takes the
# value in milliseconds via HttpOptions; everyone else takes seconds.
_TIMEOUT_MS = 180_000


def call(
    *,
    system: str,
    user: str,
    schema: dict,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai.types import GenerateContentConfig, HttpOptions

    client = genai.Client(http_options=HttpOptions(timeout=_TIMEOUT_MS))
    config = GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=schema["schema"],
        temperature=temperature,
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=config,
        )
    except genai_errors.APIError as exc:
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status is not None and (status == 429 or 500 <= status < 600):
            raise LLMTransientError(f"gemini {status}: {exc}") from exc
        raise LLMPermanentError(f"gemini {status}: {exc}") from exc
    except (ConnectionError, TimeoutError) as exc:
        raise LLMTransientError(f"gemini transport error: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise LLMTransientError("gemini response text empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMTransientError(f"gemini response not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMTransientError(
            f"gemini response JSON was not an object: {type(payload).__name__}"
        )
    return payload
