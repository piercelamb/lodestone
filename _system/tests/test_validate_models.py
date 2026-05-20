"""Tests for _system.scripts.validate_models."""
from __future__ import annotations

import time

import pytest

from _system.llm import config as llm_config
from _system.llm.errors import (
    ProviderAmbiguous,
    ProviderKeyMissing,
    ProviderUnconfigured,
)
from _system.scripts import validate_models as vm_mod
from _system.utils import http as http_mod


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


class TestEnsureModelCachedNoHook:
    """When no progress hook is set, ensure_model_cached must call
    snapshot_download WITHOUT tqdm_class (preserves the default stderr
    bar) — verifies the warm-cache fast path doesn't accidentally
    install an opaque tqdm shim that would swallow HF's own logging.
    """

    def test_no_hook_falls_through_without_tqdm_class(self, monkeypatch):
        import huggingface_hub

        captured: dict = {}

        def fake(repo_id, *args, **kwargs):
            captured["repo_id"] = repo_id
            captured["kwargs"] = kwargs
            return f"/fake/{repo_id}"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
        # Force no hook.
        assert http_mod._progress_hook.get() is None
        vm_mod.ensure_model_cached(vm_mod.ModelId.BGE)
        assert captured["repo_id"] == str(vm_mod.ModelId.BGE)
        assert "tqdm_class" not in captured["kwargs"]


class TestEnsureModelCachedProgress:
    """When a progress hook is set, ensure_model_cached should pass a
    tqdm_class whose ``update(n)`` increments emit byte-level frames
    through the hook (monotone + rate-limited).
    """

    def _patch_snapshot_with_chunks(self, monkeypatch, chunks: list[int]):
        """Drive a fake snapshot_download that walks ``chunks`` through the
        passed tqdm_class context manager. Returns the calls list to be
        populated by the active hook."""
        import huggingface_hub

        def fake(repo_id, *args, **kwargs):
            tqdm_class = kwargs.get("tqdm_class")
            assert tqdm_class is not None, "expected tqdm_class to be passed"
            # Simulate per-file tqdm context like HF's downloader does.
            bar = tqdm_class(total=sum(chunks), unit="B", unit_scale=True)
            with bar as b:
                for n in chunks:
                    b.update(n)
            return f"/fake/{repo_id}"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)

    def test_hook_receives_monotone_progress(self, monkeypatch):
        self._patch_snapshot_with_chunks(monkeypatch, [10, 20, 30, 40])
        calls: list[tuple[str, int, int]] = []

        def hook(msg: str, done: int, total: int):
            calls.append((msg, done, total))

        # Disable rate-limit so every update emits.
        monkeypatch.setattr(vm_mod, "_PROGRESS_RATE_LIMIT_S", 0.0)
        token = http_mod.set_progress_hook(hook)
        try:
            vm_mod.ensure_model_cached(vm_mod.ModelId.BGE)
        finally:
            http_mod.reset_progress_hook(token)

        assert calls, "expected at least one progress frame"
        # Bytes monotone non-decreasing.
        bytes_seq = [c[1] for c in calls]
        assert bytes_seq == sorted(bytes_seq)
        # Label includes the model id.
        assert all(str(vm_mod.ModelId.BGE) in c[0] for c in calls)
        # Final byte count == sum of chunks.
        assert calls[-1][1] == 10 + 20 + 30 + 40

    def test_hook_calls_rate_limited(self, monkeypatch):
        # 100 small chunks in a tight loop — with the 0.75s default
        # rate limit and a frozen clock, we should see at most 1 frame.
        self._patch_snapshot_with_chunks(monkeypatch, [1] * 100)
        calls: list = []

        # Freeze time.monotonic so the rate limiter believes no time
        # passed between updates. (The first update emits because
        # last_emit starts at 0.0 and now - 0.0 >= 0.75 only if now
        # is large — we deliberately make it small.)
        monkeypatch.setattr(time, "monotonic", lambda: 0.1)

        def hook(msg, done, total):
            calls.append((msg, done, total))

        token = http_mod.set_progress_hook(hook)
        try:
            vm_mod.ensure_model_cached(vm_mod.ModelId.BGE)
        finally:
            http_mod.reset_progress_hook(token)

        # Zero frames is the expected outcome (frozen clock + non-zero
        # initial last_emit gap of 0.1 - 0.0 = 0.1 < 0.75 rate limit).
        assert len(calls) <= 2, (
            f"rate limit failed — got {len(calls)} frames for 100 chunks"
        )


class TestCumulativeProgress:
    """_CumulativeProgress must keep upstream progress monotone across
    multiple ensure_model_cached stages and add stage offsets to
    per-stage byte counters."""

    def test_two_stages_yield_monotone_cumulative_stream(self, monkeypatch):
        import huggingface_hub

        # Two fake models — BGE downloads 50 bytes, GLINER downloads 100.
        chunks_by_repo = {
            str(vm_mod.ModelId.BGE):     [25, 25],
            str(vm_mod.ModelId.GLINER2): [50, 50],
        }

        def fake(repo_id, *args, **kwargs):
            tqdm_class = kwargs["tqdm_class"]
            bar = tqdm_class(total=sum(chunks_by_repo[repo_id]))
            with bar as b:
                for n in chunks_by_repo[repo_id]:
                    b.update(n)
            return f"/fake/{repo_id}"

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
        monkeypatch.setattr(vm_mod, "_PROGRESS_RATE_LIMIT_S", 0.0)

        upstream_calls: list[tuple[str, int, int]] = []

        def upstream(msg, done, total):
            upstream_calls.append((msg, done, total))

        cp = vm_mod._CumulativeProgress([
            (vm_mod.ModelId.BGE,     "bge"),
            (vm_mod.ModelId.GLINER2, "gliner"),
        ])
        token = http_mod.set_progress_hook(upstream)
        try:
            for model_id, label in cp.stages_with_labels:
                with cp.stage(model_id, label):
                    vm_mod.ensure_model_cached(model_id)
        finally:
            http_mod.reset_progress_hook(token)

        bytes_seq = [c[1] for c in upstream_calls]
        assert bytes_seq, "expected upstream frames"
        # Monotone non-decreasing across both stages.
        assert bytes_seq == sorted(bytes_seq), (
            f"non-monotone cumulative bytes: {bytes_seq}"
        )
        # Final value reflects both stages' planned totals.
        assert upstream_calls[-1][1] >= (
            vm_mod._MODEL_BYTE_ESTIMATE[vm_mod.ModelId.BGE]
        )
        # Total is sum of estimates.
        assert all(
            c[2] == cp.total for c in upstream_calls
        )


class TestProgressTqdmHFContract:
    """Regression: the tqdm-shaped class we hand HF must satisfy the
    full contract huggingface_hub + tqdm.contrib.concurrent rely on,
    not just ``update(n)``.

    HF's snapshot_download:
      - reads/writes ``bytes_progress.total`` (``+= total``).
      - calls ``set_description`` on completion.
    tqdm.contrib.concurrent.thread_map (which HF uses for parallel
    per-file downloads):
      - calls ``tqdm_class.get_lock()`` and ``tqdm_class.set_lock(lock)``
        as **classmethods** on the class object, before any instance
        exists.
    Missing any of these crashes with ``AttributeError`` before bytes
    are fetched.
    """

    def test_classmethods_get_lock_and_set_lock(self):
        cls = vm_mod._build_progress_tqdm_class(
            hook=lambda *a: None, label="x", total_hint=0,
        )
        lock = cls.get_lock()
        assert lock is not None
        # set_lock must round-trip — thread_map relies on this to
        # restore the original lock after parallel work.
        cls.set_lock(lock)
        assert cls.get_lock() is lock

    def test_instance_has_total_attribute_for_inplace_addition(self):
        cls = vm_mod._build_progress_tqdm_class(
            hook=lambda *a: None, label="x", total_hint=0,
        )
        bar = cls(total=100, unit="B")
        # HF does: bytes_progress.total += <new>.
        bar.total += 50
        assert bar.total == 150

    def test_instance_accepts_unknown_kwargs(self):
        cls = vm_mod._build_progress_tqdm_class(
            hook=lambda *a: None, label="x", total_hint=0,
        )
        # HF passes name=, disable=, desc=, initial=, unit=, unit_scale=
        # through _create_progress_bar's fallback branch.
        cls(name="hf.snapshot", disable=False, desc="d",
            initial=0, unit="B", unit_scale=True, total=1)

    def test_positional_iterable_passes_through(self):
        # tqdm.contrib.concurrent.thread_map does
        # ``list(tqdm_class(ex.map(fn, items), ...))`` — if our class
        # doesn't pass through the wrapped iterable, HF's parallel
        # downloader sees zero files.
        cls = vm_mod._build_progress_tqdm_class(
            hook=lambda *a: None, label="x", total_hint=0,
        )
        assert list(cls(iter([1, 2, 3]))) == [1, 2, 3]

    def test_thread_map_round_trips_results(self):
        """End-to-end: drive the actual tqdm.contrib.concurrent.thread_map
        with our class, the same call site HF uses.
        """
        from tqdm.contrib.concurrent import thread_map

        cls = vm_mod._build_progress_tqdm_class(
            hook=lambda *a: None, label="x", total_hint=0,
        )
        result = thread_map(lambda x: x * 2, [1, 2, 3],
                            tqdm_class=cls, max_workers=2)
        assert sorted(result) == [2, 4, 6]


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
