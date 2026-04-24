"""Error hierarchy for the LLM package.

Two-root tree so the dispatch layer can distinguish retriable from
terminal failures without a catch-all ``Exception`` clause:

- :class:`LLMTransientError` — retried by tenacity. Transport hiccups,
  rate limits, 5xx, empty body, malformed structured output, pydantic
  :class:`pydantic.ValidationError` (raised once by the dispatch, wrapped
  here so the retry surface stays homogeneous).
- :class:`LLMPermanentError` — surfaced immediately. Auth failures,
  4xx bad-request, anything the same prompt will keep producing.
- :class:`ProviderConfigError` — pre-flight failures before any call is
  attempted (missing env var, ambiguous selection, unparsable config).
  Never retried; these are user-fix errors.

Keep this tree shallow: adapters map SDK exceptions to exactly one of
the two runtime errors, and callers catch the base type.
"""
from __future__ import annotations


class LLMError(Exception):
    """Base class for everything the LLM package raises."""


class LLMTransientError(LLMError):
    """Temporary failure — retried by ``call_structured``."""


class LLMPermanentError(LLMError):
    """Deterministic failure — not retried."""


class ProviderConfigError(LLMError):
    """Base class for provider-selection / config failures."""


class ProviderUnconfigured(ProviderConfigError):
    """No provider config file and no matching env var set."""


class ProviderKeyMissing(ProviderConfigError):
    """Config selected a provider whose env var is not set."""


class ProviderAmbiguous(ProviderConfigError):
    """Multiple env vars set and non-TTY — cannot prompt."""


class PromptPlaceholderError(LLMError):
    """A ``{PLACEHOLDER}`` or schema string literal was not substituted."""
