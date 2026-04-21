"""Paper name slug generation.

The slug is a readable, filesystem-safe identifier stamped onto every paper
at fetch time. It is not a canonical identity — `papers.arxiv_id` is the
identity column. The slug only needs to be unique within the `papers` table
and match the regex ^[a-z0-9_]+$.
"""

from __future__ import annotations

import re
import unicodedata

STOP_WORDS: frozenset[str] = frozenset({
    "a", "the", "on", "of", "for", "and", "in", "to", "with", "is", "are", "be",
})

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_NON_ALNUM_OR_SPACE_RE = re.compile(r"[^a-z0-9\s]+")


def _ascii_fold(s: str) -> str:
    """NFKD decompose and drop combining marks, yielding ASCII-clean text."""
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _strip_arxiv_id(arxiv_id: str) -> str:
    """Remove trailing vN version and the dot separator. Returns digits only."""
    without_version = _VERSION_SUFFIX_RE.sub("", arxiv_id)
    return without_version.replace(".", "")


def _colon_branch(title: str) -> str:
    """Take text before the first colon; ASCII-fold, lowercase, keep [a-z0-9]."""
    prefix = title.split(":", 1)[0]
    folded = _ascii_fold(prefix).lower()
    return _NON_ALNUM_RE.sub("", folded)


def _stop_word_branch(title: str) -> str:
    """Tokenize, drop stop words, take first 3 surviving tokens, join with '_'."""
    folded = _ascii_fold(title).lower()
    spaced = _NON_ALNUM_OR_SPACE_RE.sub(" ", folded)
    tokens = [t for t in spaced.split() if t and t not in STOP_WORDS]
    return "_".join(tokens[:3])


def generate_paper_name(
    title: str,
    date_yyyy_mm_dd: str,
    arxiv_id: str,
    existing: set[str],
) -> str:
    """Generate a unique, readable paper_name slug.

    Algorithm:
      1. If `title` has a colon -> take text before the first colon,
         lowercase, ASCII-fold (NFKD + drop combining marks), strip every
         char not in [a-z0-9].
      2. Else -> tokenize on whitespace, lowercase, drop STOP_WORDS, take the
         first 3 surviving tokens, ASCII-fold each, strip non-[a-z0-9],
         underscore-join.
      3. If the branch-1 or branch-2 result is empty (e.g., title is only
         stop words or only punctuation), fall back to the stripped arxiv_id
         itself.
      4. Append '_YYYY' taken from the first 4 chars of `date_yyyy_mm_dd`.
      5. If the result is in `existing`, append '_' + last 5 digits of the
         stripped arxiv_id (strip the dot and any 'vN' version suffix first).
         Assert the final slug matches ^[a-z0-9_]+$; raise ValueError if not.

    Returns:
        The slug. Guaranteed to match ^[a-z0-9_]+$ and not be in `existing`
        (or, if both base and collision forms are in `existing`, raises).
    """
    stripped_arxiv = _strip_arxiv_id(arxiv_id)

    if ":" in title:
        base = _colon_branch(title)
    else:
        base = _stop_word_branch(title)

    if not base:
        base = stripped_arxiv

    year = date_yyyy_mm_dd[:4]
    slug = f"{base}_{year}"

    if slug in existing:
        slug = f"{slug}_{stripped_arxiv[-5:]}"
        if slug in existing:
            raise ValueError(
                f"paper_name collision unresolved: {slug!r} already in existing"
            )

    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(f"generated slug violates ^[a-z0-9_]+$: {slug!r}")

    return slug
