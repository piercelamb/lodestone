"""Canonical URL helpers for arxiv-hosted paper sources."""
from __future__ import annotations

import re

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
