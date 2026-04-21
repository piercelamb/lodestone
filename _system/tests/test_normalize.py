"""Tests for _system.resolution.normalize.normalize_term."""
from __future__ import annotations

import pytest

from _system.resolution.normalize import TRAILING_SUFFIXES, normalize_term


class TestNormalizeTermBasics:
    def test_lowercases(self):
        assert normalize_term("BookRAG") == "bookrag"

    def test_strips_trailing_model_suffix(self):
        assert normalize_term("Book-RAG Model") == "book rag"

    def test_collapses_whitespace_and_strips_punctuation(self):
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
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Foo Model", "foo"),
            ("Some Method", "some"),
            ("Foo Framework", "foo"),
            ("MMLongBench Dataset", "mmlongbench"),
            ("Bar Benchmark", "bar"),
            ("Baz System", "baz"),
            ("X Approach", "x"),
        ],
    )
    def test_strips_each_suffix(self, raw: str, expected: str):
        assert normalize_term(raw) == expected

    def test_strips_only_once(self):
        # "Model Model" -> strip trailing " model" once -> "model", not "".
        assert normalize_term("Model Model") == "model"

    def test_does_not_strip_approaches(self):
        # "approaches" is not in TRAILING_SUFFIXES; whole-word match only.
        assert normalize_term("X Approaches") == "x approaches"

    def test_whole_word_only_suffix_match(self):
        # Requires a preceding space; "alignment" must not be stripped.
        assert normalize_term("alignment") == "alignment"

    def test_whole_word_only_method(self):
        assert normalize_term("overmethod") == "overmethod"

    def test_single_word_suffix_not_stripped(self):
        assert normalize_term("model") == "model"


class TestNormalizeTermEdgeCases:
    def test_empty_string(self):
        assert normalize_term("") == ""

    def test_whitespace_only(self):
        assert normalize_term("   ") == ""

    def test_punctuation_only(self):
        assert normalize_term("!@#$%") == ""

    def test_preserves_underscores(self):
        assert normalize_term("feed_forward") == "feed_forward"

    def test_unicode_preserved(self):
        # normalize_term lowercases but does NOT ASCII-fold (slug.py does).
        assert normalize_term("Étude") == "étude"

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
