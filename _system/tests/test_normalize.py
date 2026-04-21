"""Tests for _system.resolution.normalize.normalize_term."""
from __future__ import annotations

import pytest

from _system.resolution.normalize import TRAILING_SUFFIXES, normalize_term


class TestNormalizeTermBasics:
    def test_lowercases(self):
        assert normalize_term("BookRAG") == "bookrag"

    def test_strips_trailing_model_suffix(self):
        # Per the hyphen policy: strip ALL punctuation including inner hyphens.
        # "Book-RAG Model" -> "book rag model" -> strip " model" -> "book rag"
        assert normalize_term("Book-RAG Model") == "book rag"

    def test_strips_trailing_dataset_suffix(self):
        assert normalize_term("MMLongBench Dataset") == "mmlongbench"

    def test_collapses_whitespace_and_strips_punctuation(self):
        # "  BookRAG!!   (v2) " -> lower, punct->space, collapse -> "bookrag v2"
        assert normalize_term("  BookRAG!!   (v2) ") == "bookrag v2"


class TestNormalizeTermIdempotent:
    @pytest.mark.parametrize(
        "raw",
        [
            "BookRAG",
            "Book-RAG Model",
            "  BookRAG!!   (v2) ",
            "MMLongBench Dataset",
            "Attention Is All You Need",
            "X Approach",
            "alignment",
            "feed_forward",
        ],
    )
    def test_idempotent(self, raw: str):
        once = normalize_term(raw)
        twice = normalize_term(once)
        assert once == twice


class TestNormalizeTermSuffixStripping:
    def test_strips_only_once(self):
        # "Model Model" -> "model model" -> strip " model" once -> "model"
        assert normalize_term("Model Model") == "model"

    def test_strips_approach(self):
        assert normalize_term("X Approach") == "x"

    def test_does_not_strip_approaches(self):
        # "Approaches" is not in TRAILING_SUFFIXES; must remain.
        assert normalize_term("X Approaches") == "x approaches"

    def test_whole_word_only_suffix_match(self):
        # "alignment" ends with "ment" but must NOT be stripped — whole-word only.
        assert normalize_term("alignment") == "alignment"

    def test_whole_word_only_method(self):
        # "benchmark" ends with neither suffix as a separate token.
        assert normalize_term("overmethod") == "overmethod"

    def test_single_word_suffix_not_stripped(self):
        # No leading space means no whole-word suffix match.
        assert normalize_term("model") == "model"

    def test_strips_method(self):
        assert normalize_term("Some Method") == "some"

    def test_strips_framework(self):
        assert normalize_term("Foo Framework") == "foo"

    def test_strips_benchmark(self):
        assert normalize_term("Bar Benchmark") == "bar"

    def test_strips_system(self):
        assert normalize_term("Baz System") == "baz"


class TestNormalizeTermEdgeCases:
    def test_empty_string(self):
        assert normalize_term("") == ""

    def test_whitespace_only(self):
        assert normalize_term("   ") == ""

    def test_punctuation_only(self):
        assert normalize_term("!@#$%") == ""

    def test_preserves_underscores(self):
        # Underscores are \w, so they survive punctuation stripping.
        assert normalize_term("feed_forward") == "feed_forward"

    def test_unicode_preserved(self):
        # normalize_term does NOT ASCII-fold; only slug.py does that.
        # But diacritics should be preserved as-is (lowercased).
        result = normalize_term("Étude")
        assert result == "étude"

    def test_pure_no_side_effects(self):
        s = "Book-RAG Model"
        before = s
        normalize_term(s)
        assert s == before


class TestNormalizeTermConstants:
    def test_trailing_suffixes_exact(self):
        expected = (
            " model",
            " method",
            " framework",
            " dataset",
            " benchmark",
            " system",
            " approach",
        )
        assert TRAILING_SUFFIXES == expected

    def test_all_suffixes_begin_with_space(self):
        for suf in TRAILING_SUFFIXES:
            assert suf.startswith(" "), f"suffix {suf!r} must start with a space"
