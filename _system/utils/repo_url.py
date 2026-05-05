"""Shared helpers for parsing repo hosting URLs (github / gitlab / bitbucket).

Used by both ``fetch_paper`` (discovers code repo URLs in a paper's body)
and ``resolve_repo`` (entry point for standalone repo ingest).
"""
from __future__ import annotations

import re
from typing import NamedTuple
from urllib.parse import urlparse

REPO_HOSTS: frozenset[str] = frozenset(
    {"github.com", "gitlab.com", "bitbucket.org"}
)

REPO_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/[^\s\"'<>]+",
    re.IGNORECASE,
)

_TRAILING_PUNCT = ".,);:!?]}>"

# Short host token used as the leading segment of a repo_slug. The slug
# format is ``{host_short}-{owner}-{name}`` so a github.com repo named
# ``owner/repo`` becomes ``gh-owner-repo``.
_HOST_SHORT: dict[str, str] = {
    "github.com": "gh",
    "gitlab.com": "gl",
    "bitbucket.org": "bb",
}

_SLUG_SAFE_RE = re.compile(r"[^a-z0-9]+")


class RepoUrlParts(NamedTuple):
    canonical_url: str
    host: str
    owner: str
    name: str


def normalize_repo_url(raw: str) -> str | None:
    """Return a canonical ``https://host/owner/repo`` URL, or None if the
    input isn't a repo root.

    Path depth must be **exactly 2** (``/owner/repo``). A deeper path
    (``/owner/repo/issues/42``, ``/owner/repo/blob/main/file.md``) is
    rejected outright rather than truncated — truncation produces
    false-positive repo links for bibliography entries that happen to
    cite a specific file on github.
    """
    raw = raw.strip().rstrip(_TRAILING_PUNCT)
    try:
        parsed = urlparse(raw)
    except ValueError:
        return None
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host not in REPO_HOSTS:
        return None
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) != 2:
        return None
    owner, repo = segments[0], segments[1]
    repo = repo.removesuffix(".git")
    return f"https://{host}/{owner}/{repo}"


def parse_repo_url(raw: str) -> RepoUrlParts | None:
    """Normalize and split into ``(canonical_url, host, owner, name)``.

    Returns None for any URL that isn't a recognized repo root.
    """
    canonical = normalize_repo_url(raw)
    if canonical is None:
        return None
    parsed = urlparse(canonical)
    host = parsed.netloc.lower()
    segments = [s for s in parsed.path.split("/") if s]
    owner, name = segments[0], segments[1]
    return RepoUrlParts(canonical_url=canonical, host=host, owner=owner, name=name)


def extract_repo_candidates(text: str) -> list[str]:
    """Pull canonical repo URLs out of an arbitrary text body."""
    return [
        norm
        for raw in REPO_URL_RE.findall(text)
        if (norm := normalize_repo_url(raw)) is not None
    ]


def repo_slug_base(host: str, owner: str, name: str) -> str:
    """Derive the ``repo_slug`` base (no collision suffix) for a repo.

    Format: ``{host_short}-{owner}-{name}``. Owner and name are lowered
    and any non-alphanumeric characters are folded to ``-``. Adjacent
    separators collapse and leading/trailing separators are stripped so
    the result matches ``[a-z0-9-]+``. Collision-handling (the ``-2``,
    ``-3`` suffixes) lives in :mod:`_system.scripts.resolve_repo` since
    it requires a DB lookup.
    """
    short = _HOST_SHORT.get(host.lower())
    if short is None:
        # Unknown host slipped through normalize_repo_url's gate — caller
        # bug. Falls back to a 2-letter host token from the TLD.
        short = host.split(".")[0][:2].lower() or "xx"
    parts = [short, _slugify(owner), _slugify(name)]
    return "-".join(p for p in parts if p)


def _slugify(s: str) -> str:
    out = _SLUG_SAFE_RE.sub("-", s.lower()).strip("-")
    return out
