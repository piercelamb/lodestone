"""Paper-name slug generation and identifier sanitization.

Papers and posts share the slug namespace: a slug is unique across the
union of ``papers.paper_name`` and ``posts.post_name``. Downstream
slug-keyed tables (``sections``, ``term_aliases``, ``topics``) use a bare
TEXT column and treat slugs as opaque ids, so the sharing is invisible at
the storage layer but observable at the lookup layer (``resolve_slug``
forks on which table the slug lives in).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import unicodedata

STOP_WORDS: frozenset[str] = frozenset({
    "a", "the", "on", "of", "for", "and", "in", "to", "with", "is", "are", "be",
})

_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
_VERSION_SUFFIX_RE = re.compile(r"v\d+$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_NON_ALNUM_OR_SPACE_RE = re.compile(r"[^a-z0-9\s]+")

DOMAIN_MAX_LEN = 64
_WS_OR_SLASH_RE = re.compile(r"[\s/]+")
_DOMAIN_ALLOWED_RE = re.compile(r"[^a-z0-9_-]")


def sanitize_domain(proposed: str) -> str:
    """Lowercase, collapse whitespace/slashes to ``_``, drop other non-[a-z0-9_-],
    truncate to :data:`DOMAIN_MAX_LEN`, and strip leading/trailing ``_-``."""
    lowered = proposed.lower()
    collapsed = _WS_OR_SLASH_RE.sub("_", lowered)
    stripped = _DOMAIN_ALLOWED_RE.sub("", collapsed)
    trimmed = stripped[:DOMAIN_MAX_LEN]
    return trimmed.strip("_-")


def _fold_lower(s: str) -> str:
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def _strip_arxiv_id(arxiv_id: str) -> str:
    return _VERSION_SUFFIX_RE.sub("", arxiv_id).replace(".", "")


def _colon_branch(title: str) -> str:
    prefix = title.split(":", 1)[0]
    return _NON_ALNUM_RE.sub("", _fold_lower(prefix))


def _stop_word_branch(title: str) -> str:
    spaced = _NON_ALNUM_OR_SPACE_RE.sub(" ", _fold_lower(title))
    tokens = [t for t in spaced.split() if t not in STOP_WORDS]
    return "_".join(tokens[:3])


def generate_paper_name(
    title: str,
    date_yyyy_mm_dd: str,
    arxiv_id: str,
    existing: set[str],
) -> str:
    """Generate a readable paper_name slug guaranteed to match ^[a-z0-9_]+$.

    If `title` contains ':', uses the pre-colon prefix; otherwise uses the
    first three non-stopword tokens. Appends the YYYY year. On collision with
    any name in `existing`, appends the last 5 digits of the arxiv_id (with
    any 'vN' version suffix and the dot stripped first). Raises ValueError if
    the collision form is also in `existing` or the result violates the regex.
    """
    stripped_arxiv = _strip_arxiv_id(arxiv_id)
    base = _colon_branch(title) if ":" in title else _stop_word_branch(title)
    if not base:
        base = stripped_arxiv

    slug = f"{base}_{date_yyyy_mm_dd[:4]}"

    if slug in existing:
        slug = f"{slug}_{stripped_arxiv[-5:]}"
        if slug in existing:
            raise ValueError(
                f"paper_name collision unresolved: {slug!r} already in existing"
            )

    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(f"generated slug violates ^[a-z0-9_]+$: {slug!r}")

    return slug


def generate_post_name(
    title: str,
    date_yyyy_mm_dd: str,
    canonical_url: str,
    existing: set[str],
) -> str:
    """Generate a readable post_name slug guaranteed to match ^[a-z0-9_]+$.

    Same shape as :func:`generate_paper_name` but with a URL-derived
    collision tiebreaker (last 5 hex chars of ``sha1(canonical_url)``)
    since posts have no arxiv_id. ``existing`` should union papers +
    posts (see :func:`existing_slugs`) so the namespace stays globally
    unique.
    """
    base = _colon_branch(title) if ":" in title else _stop_word_branch(title)
    if not base:
        base = "post"

    slug = f"{base}_{date_yyyy_mm_dd[:4]}"

    if slug in existing:
        url_hash = hashlib.sha1(canonical_url.encode("utf-8")).hexdigest()
        slug = f"{slug}_{url_hash[-5:]}"
        if slug in existing:
            raise ValueError(
                f"post_name collision unresolved: {slug!r} already in existing"
            )

    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(f"generated slug violates ^[a-z0-9_]+$: {slug!r}")

    return slug


def generate_book_slug(
    title: str,
    date_yyyy_mm_dd: str,
    content_hash: str,
    existing: set[str],
) -> str:
    """Generate a book-level slug analogous to :func:`generate_paper_name`.

    Local PDFs have no arxiv_id, so the collision tiebreaker is the last
    5 hex chars of the PDF's sha256 ``content_hash``. The slug shape is
    ``<title-tokens>_<YYYY>`` (or ``_<YYYY>_<hash5>`` on collision). The
    result is the prefix used by :func:`generate_chapter_slug`.
    """
    base = _colon_branch(title) if ":" in title else _stop_word_branch(title)
    if not base:
        base = "book"

    slug = f"{base}_{date_yyyy_mm_dd[:4]}"

    if slug in existing:
        slug = f"{slug}_{content_hash[-5:]}"
        if slug in existing:
            raise ValueError(
                f"book_slug collision unresolved: {slug!r} already in existing"
            )

    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(f"generated slug violates ^[a-z0-9_]+$: {slug!r}")

    return slug


CHAPTER_INDEX_MAX = 99


def generate_chapter_slug(
    book_slug: str,
    chapter_index: int,
    chapter_title: str,
    existing: set[str],
) -> str:
    """Compose ``<book_slug>__ch<NN>_<chapter-tokens>``.

    ``__`` is the book/chapter separator — :func:`generate_paper_name`
    joins tokens with single ``_`` and never emits ``__`` naturally, so
    the boundary is unambiguous. ``ch<NN>`` is zero-padded so
    ``ORDER BY paper_name`` gives chapter order for free. The
    two-digit padding caps each book at 99 chapters — above that the
    natural width of ``f"{N:02d}"`` grows (``ch100``) and lexicographic
    sort breaks (``ch100`` < ``ch11``), so we reject the input up-front
    rather than silently producing rows that ORDER BY paper_name
    misorders.

    Chapter title is tokenized the same way as :func:`_stop_word_branch`
    (lowercase, NFKD-fold, drop stopwords, take first 3 tokens). If the
    title produces no usable tokens, falls back to ``ch<chapter_index>``
    (which keeps the slug regex-valid).
    """
    if not (1 <= chapter_index <= CHAPTER_INDEX_MAX):
        raise ValueError(
            f"chapter_index must be in 1..{CHAPTER_INDEX_MAX} "
            f"(two-digit zero-pad keeps ORDER BY paper_name in TOC order); "
            f"got {chapter_index}"
        )

    chapter_tokens = _stop_word_branch(chapter_title)
    if not chapter_tokens:
        chapter_tokens = f"ch{chapter_index}"

    slug = f"{book_slug}__ch{chapter_index:02d}_{chapter_tokens}"

    if slug in existing:
        raise ValueError(
            f"chapter slug collision unresolved: {slug!r} already in existing"
        )

    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(f"generated slug violates ^[a-z0-9_]+$: {slug!r}")

    return slug


def existing_slugs(conn: sqlite3.Connection) -> set[str]:
    """Union of every paper_name and post_name in the DB.

    The slug namespace is shared: ``generate_paper_name`` /
    ``generate_post_name`` must collision-check across both tables so a
    paper can't be assigned a slug that already names a post (or vice
    versa). Two SELECTs unioned because ``papers`` and ``posts`` are
    disjoint tables; ``UNION ALL`` is fine — duplicates within either
    column are forbidden by the UNIQUE constraint.
    """
    rows = conn.execute(
        "SELECT paper_name FROM papers "
        "UNION ALL "
        "SELECT post_name FROM posts"
    ).fetchall()
    return {r[0] for r in rows}
