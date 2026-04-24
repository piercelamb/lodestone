"""Tests for _system/resolution/acronyms.py (Schwartz-Hearst)."""
from __future__ import annotations

import pytest

from _system.resolution.acronyms import extract_acronym_pairs


# ---------------------------------------------------------------------------
# Happy path: standard Long Form (SHORT) patterns
# ---------------------------------------------------------------------------


class TestCanonicalPatterns:
    def test_simple_initialism(self):
        text = "We use Reciprocal Rank Fusion (RRF) to combine retrievers."
        assert ("RRF", "Reciprocal Rank Fusion") in extract_acronym_pairs(text)

    def test_multiple_pairs_in_one_text(self):
        text = (
            "Our work combines Reciprocal Rank Fusion (RRF) with "
            "Maximum Marginal Relevance (MMR) for re-ranking."
        )
        pairs = dict(extract_acronym_pairs(text))
        assert pairs == {
            "RRF": "Reciprocal Rank Fusion",
            "MMR": "Maximum Marginal Relevance",
        }

    def test_mixed_case_short_form_rejected(self):
        """CamelCase tokens aren't acronyms — they're proper names of
        datasets / systems (``SciFact``, ``BookRAG``, ``ArguAna``). The
        all-uppercase short-form filter rejects them so we don't
        spuriously "expand" them into whatever nearby text happens to
        contain the letters.
        """
        text = "reciprocal rank fusion (rrf)"
        assert extract_acronym_pairs(text) == []
        text = "We introduce Book Rapid Access Generation (BookRAG) for QA."
        assert extract_acronym_pairs(text) == []
        text = "on biomedical text (SciFact) but underperforms"
        assert extract_acronym_pairs(text) == []

    def test_mixed_case_preserved_in_output_long_form(self):
        text = "Approximate Nearest Neighbor (ANN) search is used."
        pairs = extract_acronym_pairs(text)
        assert ("ANN", "Approximate Nearest Neighbor") in pairs


# ---------------------------------------------------------------------------
# Rejection: things that look like acronym defs but aren't
# ---------------------------------------------------------------------------


class TestRejection:
    def test_non_matching_initials_rejected(self):
        """The paren content is in length range but its letters don't match
        any word-initial progression of the preceding text.
        """
        text = "The quick brown fox jumps (XYZ)."
        assert extract_acronym_pairs(text) == []

    def test_dataset_name_reference_rejected(self):
        """Real-world false-positive: ``(SciDocs)`` used as a dataset
        reference must not spuriously "expand" to nearby text.
        """
        text = "underperforms on CS papers (SciDocs) and financial queries"
        assert extract_acronym_pairs(text) == []

    def test_self_definition_rejected(self):
        """Real-world false-positive: ``Name (Name)`` where the paren
        repeats a dataset name already in text is not an acronym def.
        """
        text = (
            "5 BEIR benchmarks (Table 2). The distance signal was validated "
            "on these 5 BEIR benchmarks using 50,425 queries (ArguAna)."
        )
        pairs = extract_acronym_pairs(text)
        # Neither Table 2 (space-containing) nor ArguAna (mixed-case)
        # passes the short-form shape filter.
        shorts = [p[0] for p in pairs]
        assert "Table 2" not in shorts
        assert "ArguAna" not in shorts

    def test_whitespace_containing_short_form_rejected(self):
        """``(Table 2)`` isn't an acronym — short forms must be single tokens."""
        text = "See details in the table (Table 2) below."
        assert extract_acronym_pairs(text) == []

    def test_too_short_rejected(self):
        text = "We use Data (D) extensively."  # single char
        assert extract_acronym_pairs(text) == []

    def test_too_long_rejected(self):
        """Acronyms over 10 chars are filtered out — probably not real."""
        text = "Very Long Acronym Definition (VLADEFINITION12345) here."
        assert extract_acronym_pairs(text) == []

    def test_all_digits_rejected(self):
        text = "We observed 42 failures (42) in total."
        assert extract_acronym_pairs(text) == []

    def test_parenthetical_without_definition_nearby_rejected(self):
        """Paren at start of text has no preceding long-form to match."""
        text = "(RRF) was proposed."
        assert extract_acronym_pairs(text) == []

    def test_too_many_words_in_paren_rejected(self):
        """Acronyms are 1-2 words; longer parentheticals aren't acronyms."""
        text = "Some method X Y Z (a b c d) here."
        assert extract_acronym_pairs(text) == []


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_first_definition_wins_on_duplicate_short(self):
        text = (
            "Reciprocal Rank Fusion (RRF) is common. "
            "Later, Random Result Filter (RRF) was proposed."
        )
        pairs = extract_acronym_pairs(text)
        # Only one entry for RRF, and it takes the first definition.
        rrf_pairs = [p for p in pairs if p[0] == "RRF"]
        assert len(rrf_pairs) == 1
        assert rrf_pairs[0][1] == "Reciprocal Rank Fusion"

    def test_repeated_same_definition_emits_one_pair(self):
        text = (
            "Reciprocal Rank Fusion (RRF) is nice. "
            "We again use Reciprocal Rank Fusion (RRF) later."
        )
        pairs = extract_acronym_pairs(text)
        assert pairs.count(("RRF", "Reciprocal Rank Fusion")) == 1


# ---------------------------------------------------------------------------
# Real-world markdown shapes
# ---------------------------------------------------------------------------


class TestRealWorld:
    def test_works_inside_markdown_prose(self):
        markdown = (
            "## Introduction\n\n"
            "We propose a new retrieval method using Reciprocal Rank "
            "Fusion (RRF). RRF combines results from multiple ranked lists.\n\n"
            "## Method\n\nOur approach uses Dense Passage Retrieval (DPR) embeddings.\n"
        )
        pairs = dict(extract_acronym_pairs(markdown))
        assert pairs["RRF"] == "Reciprocal Rank Fusion"
        assert pairs["DPR"] == "Dense Passage Retrieval"

    def test_long_form_across_markdown_linebreak(self):
        markdown = (
            "We use Reciprocal Rank\nFusion (RRF) here."
        )
        pairs = dict(extract_acronym_pairs(markdown))
        # Whitespace (including newlines) is collapsed for matching;
        # the long-form string itself may contain the original whitespace.
        assert "RRF" in pairs

    def test_empty_input(self):
        assert extract_acronym_pairs("") == []

    def test_no_parentheses(self):
        assert extract_acronym_pairs("Plain prose with no defs.") == []


# ---------------------------------------------------------------------------
# Regression: Schwartz-Hearst specifics
# ---------------------------------------------------------------------------


class TestSchwartzHearstSpecifics:
    def test_long_form_must_be_word_initial_for_first_char(self):
        """The first character of the short form must align with a
        word-initial char of the long form. Otherwise random substrings
        could spuriously match.
        """
        # "osing" does not start a word; LOM shouldn't match "just lOsing sOme Money".
        text = "just losing some money (LSM) today"
        pairs = extract_acronym_pairs(text)
        # 'L' = initial of "losing", 'S' = initial of "some", 'M' = initial of "money" → valid.
        assert ("LSM", "losing some money") in pairs

    def test_skips_when_long_form_cannot_cover_all_short_letters(self):
        """If the preceding text has fewer matchable chars than the short
        form requires, the pair is skipped.
        """
        text = "A (ABCDE) something"
        assert extract_acronym_pairs(text) == []

    @pytest.mark.parametrize(
        "short,text_before,expected_long",
        [
            ("NDCG", "normalized discounted cumulative gain", "normalized discounted cumulative gain"),
            ("BM25", "Best Match 25", "Best Match 25"),
            ("FTS5", "Full Text Search 5", "Full Text Search 5"),
        ],
    )
    def test_numeric_suffixes_pass(self, short, text_before, expected_long):
        text = f"We use {text_before} ({short}) in retrieval."
        pairs = extract_acronym_pairs(text)
        assert (short, expected_long) in pairs
