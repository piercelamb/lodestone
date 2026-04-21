"""Tests for _system.utils.slug.generate_paper_name."""
from __future__ import annotations

import re

import pytest

from _system.utils.slug import STOP_WORDS, generate_paper_name

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def _assert_valid_slug(slug: str) -> None:
    """Every slug from generate_paper_name must match ^[a-z0-9_]+$."""
    assert _SLUG_RE.fullmatch(slug), f"slug {slug!r} violates ^[a-z0-9_]+$"


class TestGeneratePaperNameColonBranch:
    def test_colon_prefix_bookrag(self):
        slug = generate_paper_name(
            "BookRAG: Adaptive Long-Document RAG",
            "2024-01-01",
            "2512.03413",
            set(),
        )
        assert slug == "bookrag_2024"
        _assert_valid_slug(slug)

    def test_colon_prefix_strips_hyphen(self):
        # Hyphen inside colon-prefix must be stripped to uphold ^[a-z0-9_]+$.
        slug = generate_paper_name(
            "Book-RAG: Adaptive RAG",
            "2024-06-15",
            "2406.01234",
            set(),
        )
        assert slug == "bookrag_2024"
        _assert_valid_slug(slug)

    def test_colon_prefix_strips_all_punctuation(self):
        slug = generate_paper_name(
            "Foo.Bar_Baz: a subtitle",
            "2023-03-03",
            "2303.00001",
            set(),
        )
        # Only [a-z0-9] survive from the colon-prefix.
        assert slug == "foobarbaz_2023"
        _assert_valid_slug(slug)


class TestGeneratePaperNameStopWordBranch:
    def test_stop_words_skipped(self):
        slug = generate_paper_name(
            "Attention Is All You Need",
            "2017-06-12",
            "1706.03762",
            set(),
        )
        # "is" is in STOP_WORDS; first 3 survivors are attention/all/you.
        assert slug == "attention_all_you_2017"
        _assert_valid_slug(slug)

    def test_takes_first_three_surviving_tokens(self):
        slug = generate_paper_name(
            "Alpha Beta Gamma Delta Epsilon",
            "2021-05-05",
            "2105.00002",
            set(),
        )
        assert slug == "alpha_beta_gamma_2021"
        _assert_valid_slug(slug)

    def test_fewer_than_three_tokens(self):
        slug = generate_paper_name(
            "Quick Brown",
            "2020-02-02",
            "2002.00003",
            set(),
        )
        assert slug == "quick_brown_2020"
        _assert_valid_slug(slug)

    def test_unicode_title_is_ascii_folded(self):
        slug = generate_paper_name(
            "Étude sur Modèles",
            "2019-09-09",
            "1909.00004",
            set(),
        )
        # Diacritics stripped, not raw bytes.
        assert slug == "etude_sur_modeles_2019"
        _assert_valid_slug(slug)
        assert slug.isascii()


class TestGeneratePaperNameFallback:
    def test_only_stop_words_falls_back_to_arxiv_id(self):
        slug = generate_paper_name(
            "A The Of",
            "2024-07-07",
            "2407.00005",
            set(),
        )
        # Fallback = stripped arxiv_id + year.
        assert slug == "240700005_2024"
        _assert_valid_slug(slug)

    def test_only_punctuation_falls_back(self):
        slug = generate_paper_name(
            "!!! ??? ...",
            "2022-12-12",
            "2212.00006",
            set(),
        )
        assert slug == "221200006_2022"
        _assert_valid_slug(slug)


class TestGeneratePaperNameCollision:
    def test_collision_appends_last_5_arxiv_digits(self):
        existing = {"bookrag_2024"}
        slug = generate_paper_name(
            "BookRAG: Adaptive Long-Document RAG",
            "2024-01-01",
            "2512.03413",
            existing,
        )
        # Stripped arxiv_id = "251203413"; last 5 = "03413".
        assert slug == "bookrag_2024_03413"
        _assert_valid_slug(slug)

    def test_collision_strips_version_suffix_before_suffix(self):
        existing = {"bookrag_2024"}
        slug = generate_paper_name(
            "BookRAG: Adaptive Long-Document RAG",
            "2024-01-01",
            "2301.12345v2",
            existing,
        )
        # Stripped arxiv_id = "230112345"; last 5 = "12345".
        assert slug == "bookrag_2024_12345"
        _assert_valid_slug(slug)

    def test_no_collision_returns_base(self):
        existing = {"something_else_2024"}
        slug = generate_paper_name(
            "BookRAG: Adaptive Long-Document RAG",
            "2024-01-01",
            "2512.03413",
            existing,
        )
        assert slug == "bookrag_2024"
        _assert_valid_slug(slug)

    def test_both_forms_in_existing_raises(self):
        # Distinct arxiv IDs that happen to share both the base slug AND the
        # same last-5 collision suffix is a hard-stop. We simulate it by
        # pre-populating `existing` with both forms for the same arxiv_id.
        existing = {"bookrag_2024", "bookrag_2024_03413"}
        with pytest.raises(ValueError):
            generate_paper_name(
                "BookRAG: Adaptive Long-Document RAG",
                "2024-01-01",
                "2512.03413",
                existing,
            )


class TestGeneratePaperNameInvariants:
    @pytest.mark.parametrize(
        "title,date,arxiv,existing",
        [
            ("BookRAG: Adaptive", "2024-01-01", "2512.03413", set()),
            ("Attention Is All You Need", "2017-06-12", "1706.03762", set()),
            ("Étude sur Modèles", "2019-09-09", "1909.00004", set()),
            ("A The Of", "2024-07-07", "2407.00005", set()),
            ("Book-RAG: foo", "2024-06-15", "2406.01234", {"bookrag_2024"}),
            ("BookRAG: x", "2024-01-01", "2301.12345v2", {"bookrag_2024"}),
            ("!!! ??? ...", "2022-12-12", "2212.00006", set()),
        ],
    )
    def test_slug_always_matches_regex(self, title, date, arxiv, existing):
        slug = generate_paper_name(title, date, arxiv, existing)
        _assert_valid_slug(slug)
        assert slug not in existing

    def test_stop_words_constant_exported(self):
        assert "a" in STOP_WORDS
        assert "is" in STOP_WORDS
        assert "with" in STOP_WORDS
        # Exact set per spec.
        expected = frozenset({
            "a", "the", "on", "of", "for", "and", "in", "to",
            "with", "is", "are", "be",
        })
        assert STOP_WORDS == expected
