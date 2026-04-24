"""Tests for :func:`_system.llm.selection.resolve_provider`."""
from __future__ import annotations

import io
import sys

import pytest

from _system.llm import config as llm_config
from _system.llm.config import Provider
from _system.llm.errors import (
    ProviderAmbiguous,
    ProviderKeyMissing,
    ProviderUnconfigured,
)
from _system.llm.selection import resolve_provider


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """Make every test start from a clean provider env."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    """Yield a path the selection code treats as the on-disk config."""
    path = tmp_path / "config.toml"
    monkeypatch.setattr(llm_config, "config_path", lambda: path)
    return path


def test_no_env_no_config_raises_unconfigured(config_file):
    assert not config_file.exists()
    with pytest.raises(ProviderUnconfigured) as exc_info:
        resolve_provider()
    msg = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert "OPENAI_API_KEY" in msg
    assert "GEMINI_API_KEY" in msg


def test_single_env_var_non_tty_picks_default_and_persists(
    config_file, monkeypatch
):
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    resolved = resolve_provider()
    assert resolved.provider is Provider.GEMINI
    from _system.llm.gemini_adapter import DEFAULT_MODEL
    from _system.llm.config import DEFAULT_TEMPERATURE
    assert resolved.model == DEFAULT_MODEL
    assert resolved.temperature == DEFAULT_TEMPERATURE
    assert config_file.exists()
    loaded = llm_config.load_config(config_file)
    assert loaded is not None
    assert loaded.llm.provider is Provider.GEMINI
    assert loaded.llm.model == DEFAULT_MODEL
    # Temperature persists silently with the default value.
    assert loaded.llm.temperature == DEFAULT_TEMPERATURE


def test_single_env_var_tty_prompts_for_model(config_file, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    # Model-prompt: pick catalog entry 2 (gemini-2.5-flash).
    replies = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    resolved = resolve_provider()
    assert resolved.provider is Provider.GEMINI
    assert resolved.model == "gemini-2.5-flash"
    loaded = llm_config.load_config(config_file)
    assert loaded is not None
    assert loaded.llm.model == "gemini-2.5-flash"


def test_multiple_env_vars_tty_prompts_and_persists(
    config_file, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    # Force TTY + stub input(): first reply picks provider, second picks model.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    fake_input = iter(["2", ""])  # "" = default model for OpenAI
    monkeypatch.setattr("builtins.input", lambda prompt="": next(fake_input))

    resolved = resolve_provider()
    # Candidates are iterated in enum order → [ANTHROPIC, OPENAI] → pick 2 = OPENAI.
    assert resolved.provider is Provider.OPENAI
    from _system.llm.openai_adapter import DEFAULT_MODEL
    assert resolved.model == DEFAULT_MODEL
    loaded = llm_config.load_config(config_file)
    assert loaded is not None
    assert loaded.llm.provider is Provider.OPENAI
    assert loaded.llm.model == DEFAULT_MODEL


def test_multiple_env_vars_non_tty_raises_ambiguous(
    config_file, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ProviderAmbiguous) as exc_info:
        resolve_provider()
    msg = str(exc_info.value)
    assert "ANTHROPIC_API_KEY" in msg
    assert "OPENAI_API_KEY" in msg


def test_config_matching_env_uses_config(config_file, monkeypatch):
    config_file.write_text(
        '[llm]\nprovider = "anthropic"\nmodel = "claude-custom"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    resolved = resolve_provider()
    assert resolved.provider is Provider.ANTHROPIC
    assert resolved.model == "claude-custom"


def test_config_without_matching_env_raises_key_missing(
    config_file, monkeypatch
):
    config_file.write_text(
        '[llm]\nprovider = "openai"\n', encoding="utf-8"
    )
    # ANTHROPIC is set but OpenAI is what the config asks for.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    with pytest.raises(ProviderKeyMissing) as exc_info:
        resolve_provider()
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_config_provider_with_no_model_non_tty_falls_back_to_default(
    config_file, monkeypatch
):
    config_file.write_text(
        '[llm]\nprovider = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    from _system.llm.anthropic_adapter import DEFAULT_MODEL
    resolved = resolve_provider()
    assert resolved.model == DEFAULT_MODEL


def test_config_provider_with_no_model_tty_prompts_and_persists(
    config_file, monkeypatch
):
    config_file.write_text(
        '[llm]\nprovider = "anthropic"\n', encoding="utf-8"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    replies = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    resolved = resolve_provider()
    from _system.llm.anthropic_adapter import MODEL_CATALOG
    assert resolved.model == MODEL_CATALOG[1][0]
    # Upgrade was persisted so subsequent runs don't re-prompt.
    loaded = llm_config.load_config(config_file)
    assert loaded is not None
    assert loaded.llm.provider is Provider.ANTHROPIC
    assert loaded.llm.model == MODEL_CATALOG[1][0]


def test_tty_prompt_reprompts_on_invalid_input(
    config_file, monkeypatch, capsys
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    # Provider prompt: "banana", "42" rejected, "1" picks ANTHROPIC.
    # Then model prompt: "" = default.
    replies = iter(["banana", "42", "1", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    resolved = resolve_provider()
    assert resolved.provider is Provider.ANTHROPIC


# ---------------------------------------------------------------------------
# Model prompt
# ---------------------------------------------------------------------------


def test_model_prompt_empty_returns_default(config_file, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    resolved = resolve_provider()
    from _system.llm.openai_adapter import DEFAULT_MODEL
    assert resolved.model == DEFAULT_MODEL


def test_model_prompt_accepts_custom_id(config_file, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    replies = iter(["claude-experimental-0"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    resolved = resolve_provider()
    assert resolved.model == "claude-experimental-0"
    loaded = llm_config.load_config(config_file)
    assert loaded is not None
    assert loaded.llm.model == "claude-experimental-0"


def test_model_prompt_out_of_range_number_reprompts(config_file, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    # "99" is out-of-range → reprompt; "1" picks first catalog entry.
    replies = iter(["99", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    resolved = resolve_provider()
    from _system.llm.openai_adapter import MODEL_CATALOG
    assert resolved.model == MODEL_CATALOG[0][0]


def test_model_prompt_numeric_index_picks_catalog_entry(config_file, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    replies = iter(["3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))

    resolved = resolve_provider()
    from _system.llm.anthropic_adapter import MODEL_CATALOG
    assert resolved.model == MODEL_CATALOG[2][0]


# ---------------------------------------------------------------------------
# Temperature persistence
# ---------------------------------------------------------------------------


def test_config_temperature_override_is_read(config_file, monkeypatch):
    config_file.write_text(
        '[llm]\nprovider = "anthropic"\nmodel = "claude-x"\ntemperature = 0.25\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    resolved = resolve_provider()
    assert resolved.temperature == 0.25


def test_config_without_temperature_falls_back_to_default(config_file, monkeypatch):
    config_file.write_text(
        '[llm]\nprovider = "anthropic"\nmodel = "claude-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    from _system.llm.config import DEFAULT_TEMPERATURE
    resolved = resolve_provider()
    assert resolved.temperature == DEFAULT_TEMPERATURE
