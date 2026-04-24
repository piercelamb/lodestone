"""Tests for _system.scripts.validate_models."""
from __future__ import annotations

import pytest

from _system.llm import config as llm_config
from _system.llm.errors import (
    ProviderAmbiguous,
    ProviderKeyMissing,
    ProviderUnconfigured,
)
from _system.scripts import validate_models as vm_mod


@pytest.fixture
def patch_snapshot_download(monkeypatch):
    """Replace ``huggingface_hub.snapshot_download`` with a call recorder."""
    import huggingface_hub

    calls: list[str] = []

    def fake(repo_id: str, *args, **kwargs):
        calls.append(repo_id)
        return f"/fake/hf/cache/{repo_id}"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return calls


@pytest.fixture
def provider_anthropic(tmp_path, monkeypatch):
    """Pin provider selection to Anthropic via env var + empty config dir."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Point config loader at a writable tmp path with no pre-existing file.
    # First resolve_provider() call will persist the selection silently.
    cfg = tmp_path / "config.toml"
    monkeypatch.setattr(llm_config, "config_path", lambda: cfg)


class TestProviderUnconfigured:
    def test_no_env_no_config_raises_unconfigured(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = tmp_path / "absent.toml"
        monkeypatch.setattr(llm_config, "config_path", lambda: cfg)

        with pytest.raises(ProviderUnconfigured) as exc_info:
            vm_mod.check_models()
        msg = str(exc_info.value)
        assert "ANTHROPIC_API_KEY" in msg
        assert "OPENAI_API_KEY" in msg
        assert "GEMINI_API_KEY" in msg

    def test_config_with_missing_env_key_raises_key_missing(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        cfg = tmp_path / "config.toml"
        cfg.write_text('[llm]\nprovider = "openai"\n', encoding="utf-8")
        monkeypatch.setattr(llm_config, "config_path", lambda: cfg)

        with pytest.raises(ProviderKeyMissing) as exc_info:
            vm_mod.check_models()
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_multiple_env_vars_non_tty_raises_ambiguous(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
        monkeypatch.setenv("OPENAI_API_KEY", "b")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        missing = tmp_path / "absent.toml"
        monkeypatch.setattr(llm_config, "config_path", lambda: missing)
        # Force non-TTY.
        import sys
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

        with pytest.raises(ProviderAmbiguous):
            vm_mod.check_models()


class TestModelLoadersInvoked:
    def test_pinned_model_ids_are_passed_to_snapshot_download(
        self, provider_anthropic, patch_snapshot_download
    ):
        returned = vm_mod.check_models()
        assert returned == "anthropic"
        assert patch_snapshot_download == [
            str(vm_mod.ModelId.BGE),
            str(vm_mod.ModelId.GLINER2),
        ]


class TestModelLoadErrorContext:
    def _patch_snapshot_raises(self, monkeypatch, exc: Exception):
        import huggingface_hub

        def boom(_repo_id, *args, **kwargs):
            raise exc

        monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)

    def test_oserror_surfaces_cache_and_hint_for_bge(
        self, provider_anthropic, monkeypatch
    ):
        self._patch_snapshot_raises(monkeypatch, OSError("cache miss"))
        with pytest.raises(vm_mod.ModelLoadError) as excinfo:
            vm_mod.check_models()
        msg = str(excinfo.value)
        assert str(vm_mod.ModelId.BGE) in msg
        assert "hf hub download" in msg
        assert "huggingface" in msg.lower()
        assert isinstance(excinfo.value.__cause__, OSError)

    def test_connectionerror_surfaces_cache_and_hint_for_bge(
        self, provider_anthropic, monkeypatch
    ):
        self._patch_snapshot_raises(monkeypatch, ConnectionError("no net"))
        with pytest.raises(vm_mod.ModelLoadError) as excinfo:
            vm_mod.check_models()
        msg = str(excinfo.value)
        assert str(vm_mod.ModelId.BGE) in msg
        assert "hf hub download" in msg
        assert "huggingface" in msg.lower()
        assert isinstance(excinfo.value.__cause__, ConnectionError)


class TestMainCLI:
    def test_main_prints_provider_model_and_hf_lines(
        self, provider_anthropic, patch_snapshot_download, capsys
    ):
        vm_mod.main()
        out = capsys.readouterr().out
        non_empty_lines = [l for l in out.splitlines() if l.strip()]
        assert len(non_empty_lines) == 4
        joined = "\n".join(non_empty_lines)
        assert "provider: anthropic" in joined
        assert "model:" in joined
        assert str(vm_mod.ModelId.BGE) in joined
        assert str(vm_mod.ModelId.GLINER2) in joined


class TestImportIsolation:
    def test_import_does_not_pull_db_or_heavy_deps(self):
        import importlib
        import sys

        shed = [
            m
            for m in sys.modules
            if m.startswith(
                (
                    "_system.scripts.validate_models",
                    "_system.db",
                    "sentence_transformers",
                    "gliner2",
                    "torch",
                )
            )
        ]
        for m in shed:
            del sys.modules[m]

        importlib.import_module("_system.scripts.validate_models")

        forbidden = [
            m
            for m in sys.modules
            if m.startswith(("_system.db", "sentence_transformers", "gliner2", "torch"))
        ]
        assert forbidden == [], (
            f"validate_models import must not pull heavy deps or DB; got {forbidden}"
        )
