"""Term normalization used by the resolver's Tier 2 lookup and by the
alias-acceptance filter.

The function is intentionally conservative: it does NOT stem, lemmatize, or
apply a synonym dictionary. Those behaviours live in Tiers 3 (rapidfuzz)
and 4 (embeddings).
"""

from __future__ import annotations

import re

TRAILING_SUFFIXES: tuple[str, ...] = (
    " model",
    " method",
    " framework",
    " dataset",
    " benchmark",
    " system",
    " approach",
)

_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_term(s: str) -> str:
    """Normalize a term for Tier 2 matching and alias deduplication.

    Steps (in order):
      1. Lowercase.
      2. Strip all punctuation (including inner hyphens) — replace with ' '.
      3. Collapse runs of whitespace into a single space; trim.
      4. If the result ends with any string in TRAILING_SUFFIXES (matched
         as a whole trailing token, i.e., preceded by a space), strip it.
         Only ONE suffix is stripped (no iterative stripping).

    Properties:
      - Idempotent: normalize_term(normalize_term(x)) == normalize_term(x).
      - Pure: no side effects, no IO.
      - Whole-word suffix match: "alignment" is untouched even though it
        ends with "ment".
    """
    lowered = s.lower()
    depuncted = _PUNCT_RE.sub(" ", lowered)
    collapsed = _WS_RE.sub(" ", depuncted).strip()

    for suffix in TRAILING_SUFFIXES:
        if collapsed.endswith(suffix):
            return collapsed[: -len(suffix)]

    return collapsed
