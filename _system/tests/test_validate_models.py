"""Tests for _system.scripts.validate_models.

These tests must never actually load the heavy ML models. The module under
test performs lazy imports inside ``check_models``, so we patch the public
``SentenceTransformer`` / ``GLiNER2`` symbols on their parent modules; the
delayed imports then pick up the fakes.
"""
from __future__ import annotations

import pytest

from _system.scripts import validate_models as vm_mod


class _FakeSentenceTransformer:
    """Records every constructor call made through ``sentence_transformers``."""

    calls: list[str] = []

    def __init__(self, name: str) -> None:
        self.name = name
        _FakeSentenceTransformer.calls.append(name)


class _FakeGLiNER2:
    """Stand-in for ``gliner2.GLiNER2`` with a ``from_pretrained`` loader."""

    calls: list[str] = []

    @classmethod
    def from_pretrained(cls, name: str):
        cls.calls.append(name)
        return cls()


@pytest.fixture(autouse=True)
def _reset_fakes():
    _FakeSentenceTransformer.calls = []
    _FakeGLiNER2.calls = []
    yield


@pytest.fixture
def patch_st(monkeypatch):
    import sentence_transformers

    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer
    )
    return _FakeSentenceTransformer


@pytest.fixture
def patch_gliner2(monkeypatch):
    import gliner2

    monkeypatch.setattr(gliner2, "GLiNER2", _FakeGLiNER2)
    return _FakeGLiNER2


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
    def test_pinned_model_ids_are_passed_to_loaders(
        self, patch_claude_present, patch_st, patch_gliner2
    ):
        vm_mod.check_models()
        assert patch_st.calls == [str(vm_mod.ModelId.BGE)]
        assert patch_gliner2.calls == [str(vm_mod.ModelId.GLINER2)]


class TestModelLoadErrorContext:
    def test_oserror_from_sentence_transformer_surfaces_cache_and_hint(
        self, patch_claude_present, monkeypatch, patch_gliner2
    ):
        import sentence_transformers

        def boom(_name):
            raise OSError("cache miss")

        monkeypatch.setattr(sentence_transformers, "SentenceTransformer", boom)
        with pytest.raises(vm_mod.ModelLoadError) as excinfo:
            vm_mod.check_models()
        msg = str(excinfo.value)
        assert str(vm_mod.ModelId.BGE) in msg
        assert "hf hub download" in msg
        # Cache path: huggingface_hub's canonical default contains "huggingface".
        assert "huggingface" in msg.lower()
        assert isinstance(excinfo.value.__cause__, OSError)

    def test_connectionerror_from_gliner2_surfaces_cache_and_hint(
        self, patch_claude_present, patch_st, monkeypatch
    ):
        import gliner2

        class _BoomGLiNER2:
            @classmethod
            def from_pretrained(cls, _name):
                raise ConnectionError("no net")

        monkeypatch.setattr(gliner2, "GLiNER2", _BoomGLiNER2)
        with pytest.raises(vm_mod.ModelLoadError) as excinfo:
            vm_mod.check_models()
        msg = str(excinfo.value)
        assert str(vm_mod.ModelId.GLINER2) in msg
        assert "hf hub download" in msg
        assert "huggingface" in msg.lower()
        assert isinstance(excinfo.value.__cause__, ConnectionError)


class TestMainCLI:
    def test_main_prints_three_status_lines(
        self, patch_claude_present, patch_st, patch_gliner2, capsys
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
    """Confirm `ingest.py` can safely import this module with no side effects.

    Guards the Acceptance Criterion: ``ingest.py`` can
    ``from _system.scripts.validate_models import check_models`` without any
    DB or heavy-model dependency.
    """

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
