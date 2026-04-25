"""Canonical URL helpers for arxiv-hosted paper sources."""
from __future__ import annotations

import re
from typing import Optional

from _system.schemas.paper_metadata import HtmlSource

# Trailing slash is load-bearing: urljoin drops the last path segment of a
# base URL that lacks a trailing separator, breaking relative figure src
# resolution on ar5iv.
_ARXIV_HTML_BASE = "https://arxiv.org/html/{arxiv_id}/"
_AR5IV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}/"

# Strict arxiv_id validators (anchored match, not search). Old-form
# `cat/NNNNNNN[vN]` is still valid on arxiv for pre-2007 papers — accept
# both. Version suffix is preserved verbatim: `2301.12345v1` and
# `2301.12345v2` are different papers by Lodestone's identity policy.
_ARXIV_NEW_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_OLD_RE = re.compile(r"^[a-z\-]+/\d{7}(v\d+)?$")

# Free-text scanner for arxiv-id mentions inside a bibliography entry.
# Three accepted forms:
#   `arXiv:2310.08560` / `arXiv: 2310.08560v3` / `ArXiv:2310.08560`
#   `arXiv:cs/0701006` (legacy archive prefix)
#   `https://arxiv.org/abs/2310.08560` / bare `arxiv.org/abs/2310.08560v2`
# The version suffix is captured but discarded by `extract_arxiv_id_from_text`.
# Defensive: a typo like `arXiv: 2310.085` (4 digits after the dot) won't
# match the modern shape — that's fine, it stays NULL rather than mis-link.
_ARXIV_ID_FREE_RE = re.compile(
    r"""
    (?:
        ar[xX]iv \s* : \s*
        (?:
            (?P<legacy>[a-z\-]+ / \d{7})
          | (?P<modern>\d{4}\.\d{4,5})
        )
        (?:v\d+)?
      |
        (?:https?://)? (?:www\.)? arxiv\.org / abs /
        (?P<urlid>\d{4}\.\d{4,5})
        (?:v\d+)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_URL_PREFIXES: tuple[str, ...] = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "https://arxiv.org/pdf/",
    "http://arxiv.org/pdf/",
)


def base_url_for_source(source: HtmlSource, arxiv_id: str) -> str:
    if source is HtmlSource.ARXIV:
        return _ARXIV_HTML_BASE.format(arxiv_id=arxiv_id)
    return _AR5IV_HTML_BASE.format(arxiv_id=arxiv_id)


def parse_arxiv_id(raw: str) -> str:
    """Extract the canonical arxiv id from a URL or bare id.

    Preserves the version suffix verbatim. Does **not** normalize.
    Raises ``ValueError`` on an input that doesn't look like either the
    new-form (`YYMM.NNNNN[vN]`) or old-form (`cat/NNNNNNN[vN]`) id.
    """
    if not raw or not raw.strip():
        raise ValueError("arxiv id / URL is empty")
    value = raw.strip()
    for prefix in _URL_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.removesuffix(".pdf")
    if _ARXIV_NEW_RE.match(value) or _ARXIV_OLD_RE.match(value):
        return value
    raise ValueError(
        f"unrecognized arxiv id / URL: {raw!r} "
        "(expected e.g. 2301.12345, 2301.12345v2, or hep-th/9901001)"
    )


def extract_arxiv_id_from_text(text: str) -> Optional[str]:
    """Find the first arxiv-id mention inside free text.

    Used by reference extraction to pull a citable arxiv-id out of a
    bibliography entry's prose. Returns the bare canonical form (no
    version suffix), matching the format `papers.arxiv_id` is normally
    stored in. Returns ``None`` when no recognizable mention exists —
    NeurIPS-only references and bibitems with the eprint field omitted
    are the common no-match cases.

    Multiple mentions within one entry (e.g. a bibitem citing both a
    journal version and a preprint) are not unusual; this returns the
    first match by document order. Trailing punctuation and parentheses
    on the surrounding text don't matter — the regex matches the id
    proper, so `(arXiv:2310.08560).` resolves cleanly.
    """
    if not text:
        return None
    m = _ARXIV_ID_FREE_RE.search(text)
    if m is None:
        return None
    legacy = m.group("legacy")
    if legacy:
        return legacy.lower()
    return m.group("modern") or m.group("urlid")
