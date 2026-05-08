"""Slug → (kind, id) resolver for the shared papers/posts namespace.

Pipeline stages that historically keyed only on ``papers.paper_name``
(``classify``, ``extract``, ``index``, plus several MCP read-tool slug
lookups) now route through :func:`resolve_slug`. Papers are checked
first; misses fall through to posts. A slug found in neither raises
:class:`SlugNotFound` so callers don't have to implement their own
fallback paths.

The shared namespace is enforced at write time by
:func:`_system.utils.slug.existing_slugs`, which unions both tables so a
new paper can't be assigned a slug that already names a post (or vice
versa).
"""
from __future__ import annotations

import sqlite3
from enum import StrEnum
from typing import NamedTuple


class SourceKind(StrEnum):
    """Discriminator for any corpus item kind.

    Used in two distinct roles:

    - Slug-namespace dispatch in classify/extract/index/citation_resolution.
      :func:`resolve_slug` only resolves PAPER and POST — repos live in a
      separate ``repo_slug`` namespace and are looked up directly by
      ``repos.repo_slug``.
    - Discriminator column for polymorphic tables (``topics``,
      ``collections`` junction). All three values appear here.
    """

    PAPER = "paper"
    POST = "post"
    REPO = "repo"


class SlugNotFound(Exception):
    """Raised when a slug doesn't match any papers.paper_name or posts.post_name."""


class ResolvedSource(NamedTuple):
    kind: SourceKind
    id: int
    domain: str | None
    status: str | None


def resolve_slug(conn: sqlite3.Connection, slug: str) -> ResolvedSource:
    """Look up ``slug`` in papers, then posts.

    Returns a (kind, id, domain, status) tuple. Domain may be NULL when
    the row is pre-classify; status carries the table's raw status string.
    Raises :class:`SlugNotFound` when neither table holds the slug.
    """
    row = conn.execute(
        "SELECT id, domain, status FROM papers WHERE paper_name = ?",
        (slug,),
    ).fetchone()
    if row is not None:
        return ResolvedSource(SourceKind.PAPER, int(row[0]), row[1], row[2])
    row = conn.execute(
        "SELECT id, domain, status FROM posts WHERE post_name = ?",
        (slug,),
    ).fetchone()
    if row is not None:
        return ResolvedSource(SourceKind.POST, int(row[0]), row[1], row[2])
    raise SlugNotFound(
        f"slug={slug!r} not found in papers.paper_name or posts.post_name"
    )


def lookup_slug(
    conn: sqlite3.Connection, slug: str
) -> ResolvedSource | None:
    """Non-raising form of :func:`resolve_slug`. Returns None on miss."""
    try:
        return resolve_slug(conn, slug)
    except SlugNotFound:
        return None
