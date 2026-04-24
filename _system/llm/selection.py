"""Resolve which provider + model to use for this process.

Resolution order::

    1. ~/.config/lodestone/config.toml exists with [llm].provider set
         → matching *_API_KEY present  → use it
         → matching *_API_KEY absent   → raise ProviderKeyMissing
    2. No config; inspect env vars for all three providers
         → exactly one set    → use it (silent in non-TTY)
         → multiple set, TTY  → numbered prompt
         → multiple set, non-TTY → raise ProviderAmbiguous
         → none set            → raise ProviderUnconfigured
       Then, if TTY, prompt for a model from the provider's catalog
       (enter for default, number to pick, or type a custom model ID).
       Non-TTY paths use the adapter's default model silently.
       Both provider and model are persisted to the config file.

The model string is ``config.llm.model`` if set, otherwise the
per-provider default constant on the adapter module.
"""
from __future__ import annotations

import os
import sys
from typing import NamedTuple

from _system.llm.config import (
    DEFAULT_TEMPERATURE,
    LlmConfig,
    Provider,
    _LlmSection,
    all_env_vars,
    env_var_for,
    load_config,
    save_config,
)
from _system.llm.errors import (
    ProviderAmbiguous,
    ProviderKeyMissing,
    ProviderUnconfigured,
)
from _system.utils.logging import get_logger

_LOG = get_logger("llm.selection")


class ResolvedProvider(NamedTuple):
    provider: Provider
    model: str
    temperature: float


def _adapter_module(provider: Provider):
    """Lazy-import and return the adapter module for ``provider``."""
    if provider is Provider.ANTHROPIC:
        from _system.llm import anthropic_adapter as mod
    elif provider is Provider.OPENAI:
        from _system.llm import openai_adapter as mod
    elif provider is Provider.GEMINI:
        from _system.llm import gemini_adapter as mod
    else:  # pragma: no cover — enum exhaustive
        raise RuntimeError(f"unknown provider {provider!r}")
    return mod


def _default_model_for(provider: Provider) -> str:
    return _adapter_module(provider).DEFAULT_MODEL


def _model_catalog_for(provider: Provider) -> list[tuple[str, str]]:
    return _adapter_module(provider).MODEL_CATALOG


def _providers_with_env_keys() -> list[Provider]:
    return [p for p, var in all_env_vars().items() if os.environ.get(var)]


def _print_llm_intro() -> None:
    """One-time first-run explanation before any provider/model prompt.

    Fires when ``~/.config/lodestone/config.toml`` does not yet exist and
    stdin is a TTY. Gives the user the context they need to evaluate the
    upcoming picker — what the LLM is for, how often it's called, and
    where the selection is persisted.
    """
    print(
        "\n"
        "Lodestone uses an LLM to classify each ingested paper: given the\n"
        "paper and your current research taxonomy, it assigns a domain and\n"
        "collection and tags salient topics. That's the only place an LLM\n"
        "is called — fetch, convert, extract, and index run without it.\n"
        "\n"
        "Scope of LLM usage:\n"
        "  - one structured call per paper (classify stage only)\n"
        "  - prompt is your current taxonomy (domains + collections) plus\n"
        "    the first 8000 characters of the paper body\n"
        "  - response is constrained to a fixed JSON schema (index picks\n"
        "    into the taxonomy + short topic strings)\n"
        "  - your pick is saved to ~/.config/lodestone/config.toml and\n"
        "    reused silently on future runs; edit that file to change it\n",
        file=sys.stderr,
    )


def _prompt_for_provider(candidates: list[Provider]) -> Provider:
    """Interactive numbered menu; caller has already verified TTY."""
    print(
        "Multiple provider API keys are set — pick one to use for classify:",
        file=sys.stderr,
    )
    for i, p in enumerate(candidates, start=1):
        print(f"  {i}. {p.value} ({env_var_for(p)})", file=sys.stderr)
    while True:
        raw = input(f"Pick [1-{len(candidates)}]: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1]
        print("invalid selection", file=sys.stderr)


def _prompt_for_model(provider: Provider) -> str:
    """Interactive model picker; caller has already verified TTY.

    Shows the provider's known-good catalog. Accepts an index, an empty
    line (default), or a free-form model ID for anything not listed.
    """
    catalog = _model_catalog_for(provider)
    default = catalog[0][0]
    print(f"Model for {provider.value} [default: {default}]:", file=sys.stderr)
    for i, (mid, desc) in enumerate(catalog, start=1):
        print(f"  {i}. {mid} — {desc}", file=sys.stderr)
    while True:
        raw = input(
            f"Pick [1-{len(catalog)}], enter for default, "
            f"or type a custom model ID: "
        ).strip()
        if not raw:
            return default
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(catalog):
                return catalog[idx - 1][0]
            print(
                f"invalid selection — pick 1-{len(catalog)} or type a model ID",
                file=sys.stderr,
            )
            continue
        return raw


def resolve_provider() -> ResolvedProvider:
    """Return ``(provider, model)`` per the documented resolution order."""
    config = load_config()
    if config is not None:
        provider = config.llm.provider
        env_name = env_var_for(provider)
        if not os.environ.get(env_name):
            raise ProviderKeyMissing(
                f"lodestone config selects provider={provider.value!r} "
                f"but ${env_name} is not set. "
                f"Set ${env_name} or edit the config to a provider with "
                f"a key present."
            )
        temperature = (
            config.llm.temperature if config.llm.temperature is not None
            else DEFAULT_TEMPERATURE
        )
        if config.llm.model:
            return ResolvedProvider(
                provider=provider, model=config.llm.model, temperature=temperature
            )
        # Provider is pinned but model isn't — prompt if we can, else default.
        if sys.stdin.isatty():
            model = _prompt_for_model(provider)
            upgraded = LlmConfig(
                llm=_LlmSection(
                    provider=provider, model=model, temperature=temperature
                )
            )
            written_to = save_config(upgraded)
            _LOG.info(
                "persisted model selection provider=%s model=%s to %s",
                provider.value, model, written_to,
            )
        else:
            model = _default_model_for(provider)
        return ResolvedProvider(
            provider=provider, model=model, temperature=temperature
        )

    present = _providers_with_env_keys()
    if not present:
        names = ", ".join(
            f"${var}" for var in all_env_vars().values()
        )
        raise ProviderUnconfigured(
            f"no lodestone config and no provider API key in the "
            f"environment. Set one of: {names}."
        )

    is_tty = sys.stdin.isatty()
    if is_tty:
        _print_llm_intro()
    if len(present) == 1:
        picked = present[0]
    elif is_tty:
        picked = _prompt_for_provider(present)
    else:
        names = ", ".join(env_var_for(p) for p in present)
        raise ProviderAmbiguous(
            f"multiple LLM provider keys set ({names}) and stdin is not "
            f"a TTY — cannot prompt. Write ~/.config/lodestone/config.toml "
            f"to pin one."
        )

    model = _prompt_for_model(picked) if is_tty else _default_model_for(picked)

    # Temperature is written silently on first-run persist — users who want
    # to tune it edit the config directly (no interactive prompt by design).
    new_config = LlmConfig(
        llm=_LlmSection(
            provider=picked, model=model, temperature=DEFAULT_TEMPERATURE
        )
    )
    written_to = save_config(new_config)
    _LOG.info(
        "persisted provider selection provider=%s model=%s to %s",
        picked.value, model, written_to,
    )
    return ResolvedProvider(
        provider=picked, model=model, temperature=DEFAULT_TEMPERATURE
    )
