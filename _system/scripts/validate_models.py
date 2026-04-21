"""Pre-flight check: ``claude`` CLI + HF-cached ML models.

Invoked by ``ingest.py`` before any pipeline work. Confirms that the
``claude`` CLI is on PATH and that the two ML models used downstream
(``BAAI/bge-small-en-v1.5`` for the resolver, ``fastino/gliner2-base-v1``
for entity extraction) are loadable from the HuggingFace cache. First
runs transparently populate the cache; download failures surface with
enough context (cache path + ``hf hub download`` hint) that the user
can recover manually.

Heavy imports (``sentence_transformers``, ``gliner2``) are deferred to
the check helpers so that ``import validate_models`` stays cheap — both
``ingest.py`` and the standalone CLI pay no import-time cost if a caller
only needs the symbols (e.g. ``ClaudeCLIMissing``).
"""
from __future__ import annotations

import shutil
from enum import StrEnum

from _system.utils.logging import get_logger

_LOG = get_logger("scripts.validate_models")


class ModelId(StrEnum):
    """HuggingFace repo IDs for every ML model this project depends on."""

    BGE = "BAAI/bge-small-en-v1.5"
    GLINER2 = "fastino/gliner2-base-v1"


_CLAUDE_INSTALL_HINT = (
    "claude CLI not found on PATH — install via `brew install claude-code` "
    "or see https://docs.claude.com/en/docs/claude-code/setup"
)


class ClaudeCLIMissing(RuntimeError):
    """Raised when the ``claude`` CLI is not on PATH.

    The message must include an install hint (``brew install claude-code``
    / docs.claude.com) so the user can recover without re-reading source.
    """


class ModelLoadError(RuntimeError):
    """Raised when a HuggingFace model fails to load from cache."""


def _hf_cache_path() -> str:
    """Return the HuggingFace hub cache directory.

    ``huggingface_hub`` is a transitive dependency of both
    ``sentence-transformers`` and ``gliner2`` (pinned in ``pyproject.toml``),
    so we rely on its canonical resolution of ``HF_HOME`` /
    ``HUGGINGFACE_HUB_CACHE`` / the ``~/.cache/huggingface/hub`` default
    rather than re-implementing it.
    """
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE


def _model_load_error(model_id: str, original: Exception) -> ModelLoadError:
    cache = _hf_cache_path()
    return ModelLoadError(
        f"failed to load {model_id} from HuggingFace cache ({cache}). "
        f"To download manually, run: hf hub download {model_id}"
    )


def _check_bge() -> None:
    """Load ``BAAI/bge-small-en-v1.5``; wrap load failures with cache context."""
    _LOG.info("loading: %s", ModelId.BGE)
    from sentence_transformers import SentenceTransformer

    try:
        SentenceTransformer(str(ModelId.BGE))
    except (OSError, ConnectionError) as e:
        raise _model_load_error(str(ModelId.BGE), e) from e
    _LOG.info("model present: %s", ModelId.BGE)


def _check_gliner2() -> None:
    """Load ``fastino/gliner2-base-v1``; wrap load failures with cache context."""
    _LOG.info("loading: %s", ModelId.GLINER2)
    from gliner2 import GLiNER2

    try:
        GLiNER2.from_pretrained(str(ModelId.GLINER2))
    except (OSError, ConnectionError) as e:
        raise _model_load_error(str(ModelId.GLINER2), e) from e
    _LOG.info("model present: %s", ModelId.GLINER2)


def check_models() -> None:
    """Assert ``claude`` CLI present and both ML models loadable from cache.

    Raises :class:`ClaudeCLIMissing` if the CLI is not on PATH, or
    :class:`ModelLoadError` if either ML model fails to load. All other
    exceptions propagate unchanged so they are visible during debugging.
    """
    claude_path = shutil.which("claude")
    if claude_path is None:
        raise ClaudeCLIMissing(_CLAUDE_INSTALL_HINT)
    _LOG.info("claude CLI present: %s", claude_path)
    _check_bge()
    _check_gliner2()


def main() -> None:
    """Standalone CLI entry point: run checks and print a status report."""
    check_models()
    print(f"claude CLI: {shutil.which('claude')}")
    print(f"{ModelId.BGE}: present")
    print(f"{ModelId.GLINER2}: present")


if __name__ == "__main__":
    main()
