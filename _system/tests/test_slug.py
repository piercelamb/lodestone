"""Tests for _system.utils.slug.generate_paper_name."""
from __future__ import annotations

import pytest

from _system.utils.slug import (
    _SLUG_RE,
    STOP_WORDS,
    generate_book_slug,
    generate_chapter_slug,
    generate_paper_name,
)


def _assert_valid_slug(slug: str) -> None:
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
        # Hard-stop: both base and collision-suffix forms already taken.
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
        expected = frozenset({
            "a", "the", "on", "of", "for", "and", "in", "to",
            "with", "is", "are", "be",
        })
        assert STOP_WORDS == expected


# A 64-char hex string used as a stand-in for a real sha256 content_hash.
_BOOK_HASH = "deadbeefcafebabe" * 4


class TestGenerateBookSlug:
    def test_base_form_matches_paper_slug_shape(self):
        slug = generate_book_slug(
            "Tractatus Logico Philosophicus",
            "1922-01-01",
            _BOOK_HASH,
            set(),
        )
        assert slug == "tractatus_logico_philosophicus_1922"
        _assert_valid_slug(slug)

    def test_collision_uses_content_hash(self):
        existing = {"tractatus_logico_philosophicus_1922"}
        slug = generate_book_slug(
            "Tractatus Logico Philosophicus",
            "1922-01-01",
            _BOOK_HASH,
            existing,
        )
        assert slug.endswith(f"_{_BOOK_HASH[-5:]}")
        _assert_valid_slug(slug)

    def test_only_stop_words_falls_back_to_book(self):
        slug = generate_book_slug(
            "A The Of",
            "2020-01-01",
            _BOOK_HASH,
            set(),
        )
        assert slug == "book_2020"
        _assert_valid_slug(slug)

    def test_both_forms_taken_raises(self):
        suffix = _BOOK_HASH[-5:]
        existing = {
            "tractatus_logico_philosophicus_1922",
            f"tractatus_logico_philosophicus_1922_{suffix}",
        }
        with pytest.raises(ValueError):
            generate_book_slug(
                "Tractatus Logico Philosophicus",
                "1922-01-01",
                _BOOK_HASH,
                existing,
            )


class TestGenerateChapterSlug:
    def test_zero_pads_chapter_number(self):
        slug = generate_chapter_slug(
            "tractatus_logico_philosophicus_1922",
            3,
            "Objects and States of Affairs",
            set(),
        )
        assert slug == "tractatus_logico_philosophicus_1922__ch03_objects_states_affairs"
        _assert_valid_slug(slug)

    def test_uses_double_underscore_separator(self):
        slug = generate_chapter_slug(
            "book_2020", 1, "Intro Talk", set(),
        )
        assert "__ch01_" in slug

    def test_validates_against_slug_re(self):
        slug = generate_chapter_slug(
            "book_2020", 12, "Hash & Bang!", set(),
        )
        _assert_valid_slug(slug)

    def test_falls_back_to_chN_when_title_empty(self):
        slug = generate_chapter_slug(
            "book_2020", 7, "", set(),
        )
        assert slug == "book_2020__ch07_ch7"
        _assert_valid_slug(slug)

    def test_falls_back_to_chN_when_title_all_stopwords(self):
        slug = generate_chapter_slug(
            "book_2020", 2, "A The Of", set(),
        )
        assert slug == "book_2020__ch02_ch2"
        _assert_valid_slug(slug)

    def test_collision_raises(self):
        existing = {"book_2020__ch01_intro"}
        with pytest.raises(ValueError):
            generate_chapter_slug("book_2020", 1, "Intro", existing)

    def test_rejects_chapter_index_above_99(self):
        # Two-digit zero-pad caps the slug shape; beyond 99 lex sort
        # breaks ('ch100' < 'ch11'). The function rejects the input
        # rather than silently emitting an out-of-order slug.
        with pytest.raises(ValueError, match=r"1\.\.99"):
            generate_chapter_slug("book_2020", 100, "Intro", set())

    def test_rejects_chapter_index_zero_or_negative(self):
        with pytest.raises(ValueError, match=r"1\.\.99"):
            generate_chapter_slug("book_2020", 0, "Intro", set())
        with pytest.raises(ValueError, match=r"1\.\.99"):
            generate_chapter_slug("book_2020", -1, "Intro", set())

    def test_sorts_in_chapter_order(self):
        # Verifies the zero-padded `ch<NN>` is the load-bearing property
        # behind `ORDER BY paper_name` returning chapters in TOC order.
        names = [
            generate_chapter_slug("b_2020", i, f"chapter {i}", set())
            for i in (1, 2, 10, 11)
        ]
        assert sorted(names) == names
