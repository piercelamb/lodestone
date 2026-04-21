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

_CLEAN_RE = re.compile(r"[^\w]+")
_SUFFIX_RE = re.compile(
    r"\s(?:" + "|".join(s.lstrip() for s in TRAILING_SUFFIXES) + r")$"
)


def normalize_term(s: str) -> str:
    """Normalize a term for Tier 2 matching and alias deduplication.

    Lowercases, replaces punctuation (including inner hyphens) with spaces,
    collapses whitespace, and strips at most one trailing type-suffix from
    TRAILING_SUFFIXES. Idempotent and pure.
    """
    if not s:
        return ""
    collapsed = _CLEAN_RE.sub(" ", s.lower()).strip()
    return _SUFFIX_RE.sub("", collapsed, count=1)
