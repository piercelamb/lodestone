"""Pre-flight check: LLM provider + HF-cached ML models.

Invoked by ``ingest.py`` before any pipeline work. Confirms that an LLM
provider (Anthropic / OpenAI / Gemini) is configured via
``~/.config/lodestone/config.toml`` and matching env var, and that the
two ML models used downstream (``BAAI/bge-small-en-v1.5`` for the
resolver, ``fastino/gliner2-large-v1`` for entity extraction) are present
in the HuggingFace cache. First runs transparently populate the cache;
download failures surface with enough context (cache path + ``hf hub
download`` hint) that the user can recover manually.

Uses ``huggingface_hub.snapshot_download`` rather than instantiating the
models themselves: confirming cache presence does not require importing
torch or materializing weights, so the pre-flight stays sub-second on
warm runs and does not spike RSS for a value that is immediately
discarded.

``ensure_model_cached`` extends the same primitive with optional
byte-level progress streaming through the ``_progress_hook`` ContextVar
in :mod:`_system.utils.http`. When set, downloads emit rate-limited
``(message, bytes_so_far, total_bytes)`` callbacks suitable for routing
to MCP ``notifications/progress`` frames. ``_CumulativeProgress``
aggregates per-model byte counts so a multi-model prefetch surfaces as
one monotone progress stream (MCP spec requirement).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from enum import StrEnum
from typing import Iterator

from _system.llm import resolve_provider
from _system.utils.logging import get_logger

# http (httpx/tenacity) is imported lazily inside the helpers below so
# ``import validate_models`` stays sub-second and the existing
# import-isolation contract (no heavy deps pulled by a pre-flight check)
# is preserved.

_LOG = get_logger("scripts.validate_models")


class ModelId(StrEnum):
    """HuggingFace repo IDs for every ML model this project depends on."""

    BGE = "BAAI/bge-small-en-v1.5"
    GLINER2 = "fastino/gliner2-large-v1"


class ModelLoadError(RuntimeError):
    """Raised when a HuggingFace model fails to load from cache."""


# Best-effort byte estimates used only for the human-readable
# `total` in MCP progress frames. Actual download may exceed these
# (HF revisions add safetensors/onnx variants over time). The MCP
# spec allows progress > total when the estimate was off — we
# deliberately do NOT clamp.
_MODEL_BYTE_ESTIMATE: dict[ModelId, int] = {
    ModelId.BGE:     133_000_000,
    ModelId.GLINER2: 285_000_000,
}

# Rate-limit progress frames so a thrashing 8 KB chunk loop doesn't
# generate hundreds of MCP frames per second. One frame every ~750 ms
# matches typical UI refresh expectations and keeps the SSE/stdout
# channel calm.
_PROGRESS_RATE_LIMIT_S = 0.75


def _build_progress_tqdm_class(hook, label: str, total_hint: int):
    """Return a tqdm-shaped class that diverts ``update(n)`` calls to ``hook``.

    ``huggingface_hub._create_progress_bar`` instantiates whatever class
    is passed via ``tqdm_class`` and uses it as a context manager with
    ``update(n)`` calls per downloaded chunk. We accept any kwargs HF
    chooses to pass (``desc``, ``total``, ``initial``, ``unit``, etc.)
    and ignore them — the byte counter is what we care about.

    A closure-captured counter is shared across all instances built from
    this class, so multi-file snapshots (one tqdm per file) feed one
    cumulative stream.
    """
    state = {"bytes": 0, "last_emit": 0.0}

    class _ProgressTqdm:
        # huggingface_hub's _snapshot_download mutates `bytes_progress.total`
        # in-place (`bytes_progress.total += total`), so the attribute must
        # exist on every instance. tqdm.contrib.concurrent.thread_map also
        # treats the class itself as a lock-holder, calling
        # `tqdm_class.get_lock()` / `tqdm_class.set_lock(lock)` as
        # classmethods — without these, the parallel download path crashes
        # with `AttributeError: type object '_ProgressTqdm' has no
        # attribute 'get_lock'` before any bytes are fetched.
        _lock = None

        @classmethod
        def get_lock(cls):
            if cls._lock is None:
                import threading
                cls._lock = threading.RLock()
            return cls._lock

        @classmethod
        def set_lock(cls, lock):
            cls._lock = lock

        def __init__(self, *args, **kwargs):
            # tqdm.contrib.concurrent.thread_map calls
            # ``tqdm_class(ex.map(fn, *iterables), ...)`` and then iterates
            # — so when an iterable is passed positionally we must let it
            # pass through, otherwise HF's parallel file download loop
            # sees zero files.
            self._iterable = args[0] if args else None
            self.total = kwargs.get("total", 0) or 0
            self.n = kwargs.get("initial", 0) or 0
            self.disable = kwargs.get("disable", False)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def update(self, n=1):
            try:
                n_int = int(n) if n is not None else 0
            except (TypeError, ValueError):
                n_int = 0
            state["bytes"] += n_int
            now = time.monotonic()
            if now - state["last_emit"] < _PROGRESS_RATE_LIMIT_S:
                return
            state["last_emit"] = now
            try:
                hook(label, state["bytes"], total_hint)
            except Exception as cb_exc:  # noqa: BLE001
                _LOG.warning("progress hook raised, ignoring: %r", cb_exc)

        # Tqdm API surface that HF or its tqdm subclass may call. We
        # implement them as no-ops so attribute-access errors don't
        # bubble out of the download path.
        def close(self):
            return

        def set_description(self, *a, **kw):
            return

        def set_postfix(self, *a, **kw):
            return

        def set_postfix_str(self, *a, **kw):
            return

        def refresh(self, *a, **kw):
            return

        def reset(self, *a, **kw):
            return

        def display(self, *a, **kw):
            return

        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

    return _ProgressTqdm


def ensure_model_cached(model_id: ModelId) -> None:
    """Idempotent: download a HF model to the local cache.

    If a ``_progress_hook`` is set on the current context (see
    :mod:`_system.utils.http`), emits byte-level progress through it —
    suitable for routing to MCP ``notifications/progress``. Otherwise
    falls back to ``snapshot_download``'s default stderr tqdm bar.

    Monotone-progress contract is the caller's responsibility: when
    downloading multiple models in one request, wrap the hook with
    :class:`_CumulativeProgress` so per-model byte counters get
    translated into a single cumulative stream.
    """
    _LOG.info("ensuring model cached: %s", model_id)
    from huggingface_hub import constants, snapshot_download

    from _system.utils.http import _progress_hook

    hook = _progress_hook.get()
    total_hint = _MODEL_BYTE_ESTIMATE.get(model_id, 0)
    try:
        if hook is None:
            snapshot_download(str(model_id))
        else:
            tqdm_class = _build_progress_tqdm_class(
                hook, label=str(model_id), total_hint=total_hint,
            )
            snapshot_download(str(model_id), tqdm_class=tqdm_class)
    except (OSError, ConnectionError) as e:
        raise ModelLoadError(
            f"failed to load {model_id} from HuggingFace cache "
            f"({constants.HF_HUB_CACHE}). "
            f"To download manually, run: hf hub download {model_id}"
        ) from e
    _LOG.info("model present: %s", model_id)


class _CumulativeProgress:
    """Aggregate per-model byte counters into one monotone progress stream.

    MCP ``notifications/progress`` requires monotonically-increasing
    ``progress`` over a request's lifetime. When we prefetch bge then
    gliner inside one tool call, the naive approach (each model emits
    0→N independently) violates that. ``_CumulativeProgress`` swaps the
    active ``_progress_hook`` for the duration of each stage with a
    wrapped hook that adds completed-stage bytes to the in-flight
    counter before re-emitting upstream.

    Usage::

        cp = _CumulativeProgress([
            (ModelId.BGE,     "bge-small-en-v1.5"),
            (ModelId.GLINER2, "gliner2-large-v1"),
        ])
        for model_id, label in cp.stages_with_labels:
            with cp.stage(model_id, label):
                ensure_model_cached(model_id)
    """

    def __init__(self, stages: list[tuple[ModelId, str]]) -> None:
        self.stages_with_labels = list(stages)
        self.total = sum(
            _MODEL_BYTE_ESTIMATE.get(m, 0) for m, _ in self.stages_with_labels
        ) or 1
        self._done_bytes = 0
        self._last_progress = 0  # enforce monotonicity even if estimates lie

    @contextmanager
    def stage(self, model_id: ModelId, label: str) -> Iterator[None]:
        """Wrap the active progress hook for the duration of one model download.

        The wrapped hook adds ``_done_bytes`` (cumulative bytes from
        prior stages) to whatever the per-model byte counter reports,
        then forwards to the outer hook with a "Downloading lodestone
        models — <label>" message.
        """
        from _system.utils.http import (
            _progress_hook,
            reset_progress_hook,
            set_progress_hook,
        )

        outer = _progress_hook.get()
        stage_est = _MODEL_BYTE_ESTIMATE.get(model_id, 0)
        cum = self

        def wrapped(msg: str, bytes_in_stage: int, _total_unused: int) -> None:
            if outer is None:
                return
            agg = cum._done_bytes + max(bytes_in_stage, 0)
            # Enforce monotone progress upstream even if a retry/resume
            # makes the per-stage counter regress.
            if agg < cum._last_progress:
                agg = cum._last_progress
            cum._last_progress = agg
            try:
                outer(
                    f"Downloading lodestone models — {label}",
                    agg, cum.total,
                )
            except Exception as cb_exc:  # noqa: BLE001
                _LOG.warning("progress hook raised, ignoring: %r", cb_exc)

        token = set_progress_hook(wrapped)
        try:
            yield
        finally:
            reset_progress_hook(token)
            # Advance the cumulative baseline by the planned stage size
            # (not the observed bytes — observed counter may have been
            # short-circuited by warm-cache shortcut, in which case the
            # next stage should still start from the planned offset).
            self._done_bytes += stage_est
            if self._done_bytes > self._last_progress:
                self._last_progress = self._done_bytes
            if outer is not None:
                try:
                    outer(
                        f"Downloaded {label}", self._done_bytes, self.total,
                    )
                except Exception as cb_exc:  # noqa: BLE001
                    _LOG.warning("progress hook raised, ignoring: %r", cb_exc)


def _check_model(model_id: ModelId) -> None:
    _LOG.info("ensuring model cached: %s", model_id)
    from huggingface_hub import constants, snapshot_download

    try:
        snapshot_download(str(model_id))
    except (OSError, ConnectionError) as e:
        raise ModelLoadError(
            f"failed to load {model_id} from HuggingFace cache "
            f"({constants.HF_HUB_CACHE}). "
            f"To download manually, run: hf hub download {model_id}"
        ) from e
    _LOG.info("model present: %s", model_id)


def check_models() -> str:
    """Assert provider configured and both ML models cached.

    Returns the resolved provider name (string). Raises
    :class:`_system.llm.ProviderConfigError` subclasses if no provider can
    be selected, or :class:`ModelLoadError` if either ML model is
    uncached and cannot be downloaded.
    """
    resolved = resolve_provider()
    _LOG.info(
        "provider configured: %s (model=%s)",
        resolved.provider.value, resolved.model,
    )
    _check_model(ModelId.BGE)
    _check_model(ModelId.GLINER2)
    return resolved.provider.value


def main() -> None:
    provider_name = check_models()
    # Reload to surface the resolved model alongside the provider name.
    resolved = resolve_provider()
    print(f"provider: {provider_name}")
    print(f"model: {resolved.model}")
    print(f"{ModelId.BGE}: present")
    print(f"{ModelId.GLINER2}: present")


if __name__ == "__main__":
    main()
