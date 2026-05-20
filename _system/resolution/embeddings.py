"""Thin wrapper around sentence-transformers BAAI/bge-small-en-v1.5.

Used by the term resolver (tier 4) and by ``index_paper`` for taxonomy
embedding rebuilds. The underlying model is loaded lazily on first use
so that ``import embeddings`` stays cheap for callers that may never
touch the model (e.g. ``search.py`` when tier 4 is not triggered).
"""
from __future__ import annotations


class Embedder:
    """Wraps ``sentence-transformers`` BAAI/bge-small-en-v1.5 (384-dim, normalized).

    The model is loaded on the first call to :meth:`embed` or
    :meth:`embed_batch`. Callers that want a process-wide singleton
    should construct ``Embedder()`` once and reuse it.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Ensure the HF cache is populated first. On a cold cache this
        # emits byte-level progress via the active _progress_hook (set
        # by the MCP server for the duration of an ingest tool call);
        # SentenceTransformer then finds the snapshot present and its
        # constructor stays silent.
        from _system.scripts.validate_models import ModelId, ensure_model_cached

        ensure_model_cached(ModelId.BGE)
        # Import inside the method so ``import embeddings`` does not pull in
        # torch / sentence_transformers until someone actually embeds text.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> list[float]:
        """Return a 384-dim embedding for ``text`` as a list of floats."""
        self._ensure_loaded()
        # ``normalize_embeddings=True`` makes ``1 - d^2/2`` a usable cosine
        # similarity on the sqlite-vec L2 distance returned by term_embeddings.
        vec = self._model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return [float(x) for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one 384-dim embedding per input text, in input order."""
        self._ensure_loaded()
        if not texts:
            return []
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(x) for x in row] for row in vecs]
