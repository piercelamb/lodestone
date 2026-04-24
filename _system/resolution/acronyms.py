"""Schwartz-Hearst parenthetical acronym detection.

Scans text for ``Long Form (SHORT)`` patterns and returns
``(short, long)`` pairs. Pure / offline — no DB, no network, stdlib only.

Based on Schwartz & Hearst 2003, "A simple algorithm for identifying
abbreviation definitions in biomedical text." The matching is the
classic right-to-left walk: every character of the short form must
match a character of the long form in order, the first short-form
character must be word-initial in the long form, and intra-word
multi-letter matches (e.g. ``BookRAG`` ↔ ``Book Rapid Access Generation``)
are allowed.

Integration: :func:`_system.scripts.extract_entities.extract` calls
:func:`extract_acronym_pairs` once on the paper markdown and uses the
result to rewrite acronym spans to their long forms before the label
vote, collapsing ``RRF``/``Reciprocal Rank Fusion`` mentions into one
canonical. The short form is persisted as a ``term_aliases`` row with
``match_tier = 0`` (paper-native, pre-resolver).
"""
from __future__ import annotations

import re

# Parenthetical with 2-40 chars of non-paren content. Upper bound keeps us
# from matching long explanatory parentheticals that aren't acronym defs.
_PAREN_RE = re.compile(r"\(([^()]{2,40})\)")

_WORD_RE = re.compile(r"\w+")

_MIN_SHORT_LEN = 2
_MAX_SHORT_LEN = 10
_MAX_SHORT_WORDS = 2
# How far back from the opening paren to look for the long form.
_LONG_FORM_WINDOW_CHARS = 200
# Classical Schwartz-Hearst bound on long-form length relative to short form.
_LONG_FORM_EXTRA_WORDS = 5


def _is_short_form_candidate(text: str) -> bool:
    """Shape filter for acronym short forms.

    Requires: length 2-10, no internal whitespace, at least one alpha char,
    and ALL alpha chars uppercase. The all-upper rule is what actually
    discriminates acronyms (``RRF``, ``MMR``, ``BM25``, ``NDCG``) from
    CamelCase dataset / model names (``SciFact``, ``ArguAna``, ``BGE-small``),
    which are proper names — not abbreviations needing expansion — and which
    Schwartz-Hearst's character-assembly matching otherwise spuriously
    "expands" into whatever long text happens to contain the letters. Without
    this filter, parentheticals like ``(SciDocs)`` match ``"on CS papers"``
    in the preceding sentence.
    """
    s = text.strip()
    if not (_MIN_SHORT_LEN <= len(s) <= _MAX_SHORT_LEN):
        return False
    if any(c.isspace() for c in s):
        return False
    alphas = [c for c in s if c.isalpha()]
    if not alphas:
        return False
    if not all(c.isupper() for c in alphas):
        return False
    return True


def _best_long_form(short: str, long: str) -> str | None:
    """Schwartz-Hearst right-to-left matching.

    Returns the substring of ``long`` that defines ``short``, or None if
    no valid match exists. ``short`` need not be uppercase — matching is
    case-insensitive. The first character of ``short`` must align with a
    word-initial character of the returned substring.
    """
    short_lower = short.lower()
    long_lower = long.lower()
    s_idx = len(short_lower) - 1
    l_idx = len(long_lower) - 1

    while s_idx >= 0:
        curr = short_lower[s_idx]
        if not curr.isalnum():
            s_idx -= 1
            continue
        # Walk l_idx left until we find a match for curr. For the first
        # char of the short form (s_idx == 0), require the match to sit
        # at a word boundary in the long form — i.e. the preceding long-
        # form char is non-alnum (or we're at position 0).
        while l_idx >= 0:
            if long_lower[l_idx] == curr:
                if s_idx == 0 and l_idx > 0 and long_lower[l_idx - 1].isalnum():
                    l_idx -= 1
                    continue
                break
            l_idx -= 1
        if l_idx < 0:
            return None
        l_idx -= 1
        s_idx -= 1

    # l_idx now sits one-before the first matched char; advance to it.
    return long[l_idx + 1 :]


def extract_acronym_pairs(text: str) -> list[tuple[str, str]]:
    """Return de-duplicated ``(short, long)`` pairs from the Schwartz-Hearst
    ``Long Form (SHORT)`` pattern in ``text``.

    First definition wins on duplicate short forms. Short form is the
    verbatim parenthetical content (whitespace-trimmed); long form is the
    exact source substring preceding the paren that Schwartz-Hearst matched
    (case preserved from the original text).
    """
    pairs: list[tuple[str, str]] = []
    seen_short: set[str] = set()

    for match in _PAREN_RE.finditer(text):
        inner = match.group(1).strip()
        if not _is_short_form_candidate(inner):
            continue
        short_key = inner.lower()
        if short_key in seen_short:
            continue

        start = max(0, match.start() - _LONG_FORM_WINDOW_CHARS)
        before = text[start : match.start()].rstrip()
        if not before:
            continue

        n_short_letters = sum(1 for c in inner if c.isalnum())
        max_long_words = n_short_letters * 2 + _LONG_FORM_EXTRA_WORDS

        word_matches = list(_WORD_RE.finditer(before))
        if not word_matches:
            continue

        # Try long-form windows of increasing word count, shortest first.
        # Minimum floor is 1, not n_short_letters — a single long word can
        # cover multiple short-form chars via intra-word matching (e.g.
        # ``BookRAG`` ↔ "Book Rapid Access Generation" uses four letters of
        # "Book"). Upper bound follows Schwartz-Hearst's heuristic.
        upper = min(max_long_words, len(word_matches))
        for n_words in range(1, upper + 1):
            window_start = word_matches[-n_words].start()
            candidate = before[window_start:]
            matched = _best_long_form(inner, candidate)
            if matched is None:
                continue
            matched_stripped = matched.strip()
            # Defense-in-depth: reject ``X (X)`` and related shapes where
            # the short form appears verbatim inside the matched long form.
            # Schwartz-Hearst's character-scatter match would otherwise
            # accept ``ArguAna 11,405). It achieves F1 = 0.996 ...`` as a
            # "long form" for ``ArguAna`` because the letters A-r-g-u-a-n-a
            # can be assembled from positions scattered through it.
            if inner.lower() in matched_stripped.lower():
                continue
            pairs.append((inner, matched_stripped))
            seen_short.add(short_key)
            break

    return pairs
