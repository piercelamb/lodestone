"""Tests for the TOML-backed LLM config."""
from __future__ import annotations

from pathlib import Path

import pytest

from _system.llm import config as llm_config
from _system.llm.config import (
    LlmConfig,
    Provider,
    _LlmSection,
    config_path,
    load_config,
    save_config,
)
from _system.llm.errors import ProviderConfigError


def test_config_path_honors_xdg_config_home(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/fake/xdg")
    assert str(config_path()) == "/fake/xdg/lodestone/config.toml"


def test_config_path_falls_back_to_dot_config(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # On macOS/Linux (and any non-Windows platform) the fallback is ~/.config/
    # regardless of Apple's GUI-app convention — matches the CLI-tool norm.
    monkeypatch.setattr("_system.llm.config.sys.platform", "darwin")
    assert config_path() == tmp_path / ".config" / "lodestone" / "config.toml"


def test_config_path_windows_uses_appdata(monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("APPDATA", "/fake/appdata")
    monkeypatch.setattr("_system.llm.config.sys.platform", "win32")
    assert config_path() == Path("/fake/appdata") / "lodestone" / "config.toml"


def test_load_missing_file_returns_none(tmp_path):
    missing = tmp_path / "no-config.toml"
    assert load_config(missing) is None


def test_roundtrip_provider_only(tmp_path):
    cfg = LlmConfig(llm=_LlmSection(provider=Provider.ANTHROPIC))
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    reloaded = load_config(path)
    assert reloaded == cfg


def test_roundtrip_provider_and_model(tmp_path):
    cfg = LlmConfig(
        llm=_LlmSection(provider=Provider.OPENAI, model="gpt-5.1")
    )
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    reloaded = load_config(path)
    assert reloaded is not None
    assert reloaded.llm.provider is Provider.OPENAI
    assert reloaded.llm.model == "gpt-5.1"


def test_roundtrip_provider_model_and_temperature(tmp_path):
    cfg = LlmConfig(
        llm=_LlmSection(
            provider=Provider.ANTHROPIC, model="claude-x", temperature=0.4
        )
    )
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    reloaded = load_config(path)
    assert reloaded is not None
    assert reloaded.llm.temperature == 0.4


def test_malformed_toml_raises_with_path(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("this is :: not :: toml ::::\n", encoding="utf-8")
    with pytest.raises(ProviderConfigError) as exc_info:
        load_config(path)
    assert str(path) in str(exc_info.value)


def test_extra_key_in_llm_raises(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[llm]\nprovider = "anthropic"\nstowaway = "nope"\n',
        encoding="utf-8",
    )
    with pytest.raises(ProviderConfigError) as exc_info:
        load_config(path)
    assert str(path) in str(exc_info.value)


def test_extra_top_level_key_raises(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[llm]\nprovider = "anthropic"\n\n[extra]\nfoo = 1\n',
        encoding="utf-8",
    )
    with pytest.raises(ProviderConfigError):
        load_config(path)


def test_missing_llm_section_raises(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[other]\nfoo = 1\n", encoding="utf-8")
    with pytest.raises(ProviderConfigError):
        load_config(path)


def test_invalid_provider_value_raises(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[llm]\nprovider = "llama"\n', encoding="utf-8")
    with pytest.raises(ProviderConfigError):
        load_config(path)


def test_first_save_creates_parent_dir(tmp_path):
    nested = tmp_path / "deep" / "lodestone" / "config.toml"
    assert not nested.parent.exists()
    save_config(LlmConfig(llm=_LlmSection(provider=Provider.GEMINI)), nested)
    assert nested.exists()
    assert nested.parent.is_dir()


def test_env_vars_maps_all_three_providers():
    mapping = llm_config.all_env_vars()
    assert mapping[Provider.ANTHROPIC] == "ANTHROPIC_API_KEY"
    assert mapping[Provider.OPENAI] == "OPENAI_API_KEY"
    assert mapping[Provider.GEMINI] == "GEMINI_API_KEY"
