"""Tests for _system.resolution.embeddings.Embedder.

The real model (BAAI/bge-small-en-v1.5) is several hundred MB; tests
that load it are marked ``@pytest.mark.slow`` so CI can skip them.
"""
from __future__ import annotations

import pytest

from _system.resolution import embeddings as emb_mod


class _FakeModel:
    """Stand-in for ``sentence_transformers.SentenceTransformer``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0

    def encode(self, texts, normalize_embeddings: bool = False, convert_to_numpy: bool = False):
        # Deterministic: each text produces a 384-vector whose first element is
        # ``len(text) / 100`` and the rest are zeros. Good enough for identity
        # and batch-matching tests.
        self.call_count += 1
        out = []
        for t in texts:
            v = [0.0] * 384
            v[0] = len(t) / 100.0
            out.append(v)
        # encode() normally returns a numpy.ndarray when convert_to_numpy=True;
        # a plain list-of-lists is fine for the code under test since we only
        # iterate and index into it.
        return out


@pytest.fixture
def fake_st(monkeypatch):
    """Patch sentence_transformers.SentenceTransformer with a cheap fake."""
    instances: list[_FakeModel] = []

    def factory(name):
        m = _FakeModel(name)
        instances.append(m)
        return m

    # _ensure_loaded imports sentence_transformers lazily, so we patch the
    # real module so the delayed import picks up the fake.
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", factory)
    return instances


class TestLazyLoading:
    def test_model_not_loaded_until_embed(self, fake_st):
        e = emb_mod.Embedder()
        assert e._model is None
        assert fake_st == []  # constructor not called

        e.embed("hello")
        assert e._model is not None
        assert len(fake_st) == 1

    def test_model_loaded_only_once(self, fake_st):
        e = emb_mod.Embedder()
        e.embed("one")
        e.embed("two")
        e.embed_batch(["three", "four"])
        assert len(fake_st) == 1, "SentenceTransformer was constructed more than once"


class TestEmbedShape:
    def test_embed_returns_384_floats(self, fake_st):
        e = emb_mod.Embedder()
        out = e.embed("BookRAG")
        assert isinstance(out, list)
        assert len(out) == 384
        assert all(isinstance(x, float) for x in out)

    def test_embed_batch_empty(self, fake_st):
        e = emb_mod.Embedder()
        assert e.embed_batch([]) == []


class TestEmbedBatchMatchesPerText:
    def test_batch_matches_sequential(self, fake_st):
        e = emb_mod.Embedder()
        texts = ["foo", "bar", "baz"]
        per_text = [e.embed(t) for t in texts]
        batch = e.embed_batch(texts)
        assert len(batch) == len(texts)
        for a, b in zip(per_text, batch):
            assert len(a) == len(b) == 384
            for x, y in zip(a, b):
                assert abs(x - y) < 1e-5


@pytest.mark.slow
class TestRealModel:
    """Smoke test that loads the real sentence-transformers model.

    Runs only when slow tests are explicitly enabled. Kept minimal: one
    load, one embed, one shape/range assertion.
    """

    def test_real_embed_produces_unit_vector(self):
        import math

        e = emb_mod.Embedder()
        v = e.embed("attention is all you need")
        assert len(v) == 384
        norm = math.sqrt(sum(x * x for x in v))
        # bge-small returns L2-normalized embeddings; allow a small tolerance.
        assert abs(norm - 1.0) < 1e-3
