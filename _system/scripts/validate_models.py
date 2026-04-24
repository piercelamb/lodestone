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
"""
from __future__ import annotations

from enum import StrEnum

from _system.llm import resolve_provider
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.validate_models")


class ModelId(StrEnum):
    """HuggingFace repo IDs for every ML model this project depends on."""

    BGE = "BAAI/bge-small-en-v1.5"
    GLINER2 = "fastino/gliner2-large-v1"


class ModelLoadError(RuntimeError):
    """Raised when a HuggingFace model fails to load from cache."""


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
