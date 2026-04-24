"""Provider enum + on-disk TOML config for LLM selection.

Config lives at the XDG Base Directory on all POSIX platforms
(``$XDG_CONFIG_HOME/lodestone/config.toml``, defaulting to
``~/.config/lodestone/config.toml`` when unset) — matches CLI-tool
convention (gh, git, etc.) rather than Apple's GUI-app convention.
Windows fallback uses ``%APPDATA%\\lodestone\\config.toml``. Shape::

    [llm]
    provider = "anthropic"      # one of: anthropic | openai | gemini
    model = "claude-opus-4-7"   # optional; per-provider default applies
    temperature = 1.0           # optional; defaults to _DEFAULT_TEMPERATURE

Missing file is fine — callers get ``None`` and fall through to env
detection. Malformed content raises :class:`ProviderConfigError` naming
the path; extra keys raise via pydantic ``extra="forbid"``. Raise-don't-
swallow: config drift is exactly the class of bug silent fallback hides.

Temperature is written silently on first-run config-persist; it is not
part of any interactive prompt. Users who want a different value edit
the TOML directly. Rationale: classify_paper's sole LLM call benefits
from provider-side stochasticity (the model is sometimes asked to
*propose* new domains/collections/topics), so ``1.0`` is a reasonable
default — but the knob is always available.
"""
from __future__ import annotations

import os
import sys
import tomllib
from enum import StrEnum
from pathlib import Path

import tomli_w
from pydantic import BaseModel, ConfigDict

from _system.llm.errors import ProviderConfigError


_APP_NAME = "lodestone"

# Written to new config files and used when the field is absent on read.
# Tuned for classify_paper (the pipeline's only LLM call) where the
# model is routinely asked to propose new domain/collection/topic names.
DEFAULT_TEMPERATURE = 1.0


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GEMINI = "gemini"


# Env var names keyed by provider. Selection / pre-flight both read from
# this single source.
_ENV_KEYS: dict[Provider, str] = {
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.GEMINI: "GEMINI_API_KEY",
}


def env_var_for(provider: Provider) -> str:
    return _ENV_KEYS[provider]


def all_env_vars() -> dict[Provider, str]:
    return dict(_ENV_KEYS)


class _LlmSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str | None = None
    temperature: float | None = None


class LlmConfig(BaseModel):
    """Root config schema. Extra keys anywhere in the tree raise."""

    model_config = ConfigDict(extra="forbid")

    llm: _LlmSection


def config_path() -> Path:
    """Resolve the config file path. Does not touch the filesystem.

    Resolution order:

    * ``$XDG_CONFIG_HOME/<app>/config.toml`` when the env var is set
    * ``~/.config/<app>/config.toml`` on POSIX (Linux/macOS)
    * ``%APPDATA%/<app>/config.toml`` on Windows (``APPDATA`` env var)
    """
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        base = Path(xdg)
    elif sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / ".config"
    else:
        base = Path.home() / ".config"
    return base / _APP_NAME / "config.toml"


def load_config(path: Path | None = None) -> LlmConfig | None:
    """Load the TOML config. Return ``None`` if the file does not exist.

    Missing file is a normal state (first run, or user relying on env
    vars). Every other failure — unparsable TOML, missing ``[llm]``,
    extra keys, bad provider value — raises :class:`ProviderConfigError`
    naming the path.
    """
    p = path or config_path()
    if not p.exists():
        return None
    try:
        raw_bytes = p.read_bytes()
    except OSError as exc:
        raise ProviderConfigError(
            f"cannot read lodestone config at {p}: {exc}"
        ) from exc

    try:
        data = tomllib.loads(raw_bytes.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ProviderConfigError(
            f"lodestone config at {p} is not valid TOML: {exc}"
        ) from exc

    try:
        return LlmConfig.model_validate(data)
    except Exception as exc:
        raise ProviderConfigError(
            f"lodestone config at {p} failed schema validation: {exc}"
        ) from exc


def save_config(config: LlmConfig, path: Path | None = None) -> Path:
    """Write the config atomically; create the parent dir if missing."""
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"llm": config.llm.model_dump(exclude_none=True)}
    p.write_bytes(tomli_w.dumps(payload).encode("utf-8"))
    return p
