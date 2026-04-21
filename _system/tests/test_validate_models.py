"""Tests for _system.scripts.validate_models."""
from __future__ import annotations

import pytest

from _system.scripts import validate_models as vm_mod


@pytest.fixture
def patch_snapshot_download(monkeypatch):
    """Replace ``huggingface_hub.snapshot_download`` with a call recorder.

    The validator lazy-imports inside ``_check_model``, so we patch the
    public symbol on ``huggingface_hub`` and the deferred import picks
    up the fake.
    """
    import huggingface_hub

    calls: list[str] = []

    def fake(repo_id: str, *args, **kwargs):
        calls.append(repo_id)
        return f"/fake/hf/cache/{repo_id}"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    return calls


@pytest.fixture
def patch_claude_present(monkeypatch):
    monkeypatch.setattr(
        "shutil.which",
        lambda name: "/fake/bin/claude" if name == "claude" else None,
    )


class TestClaudeCLIMissing:
    def test_raises_when_shutil_which_returns_none(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _: None)
        with pytest.raises(vm_mod.ClaudeCLIMissing) as excinfo:
            vm_mod.check_models()
        msg = str(excinfo.value)
        assert "claude" in msg.lower()
        assert (
            "brew" in msg.lower()
            or "install" in msg.lower()
            or "docs.claude.com" in msg.lower()
        )


class TestModelLoadersInvoked:
    def test_pinned_model_ids_are_passed_to_snapshot_download(
        self, patch_claude_present, patch_snapshot_download
    ):
        returned = vm_mod.check_models()
        assert returned == "/fake/bin/claude"
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
        self, patch_claude_present, monkeypatch
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
        self, patch_claude_present, monkeypatch
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
    def test_main_prints_three_status_lines(
        self, patch_claude_present, patch_snapshot_download, capsys
    ):
        vm_mod.main()
        out = capsys.readouterr().out
        non_empty_lines = [l for l in out.splitlines() if l.strip()]
        assert len(non_empty_lines) == 3
        joined = "\n".join(non_empty_lines)
        assert "claude" in joined.lower()
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
