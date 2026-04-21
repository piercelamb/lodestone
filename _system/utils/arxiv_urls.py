"""Canonical URL helpers for arxiv-hosted paper sources."""
from __future__ import annotations

from _system.schemas.paper_metadata import HtmlSource

# Trailing slash is load-bearing: urljoin drops the last path segment of a
# base URL that lacks a trailing separator, breaking relative figure src
# resolution on ar5iv.
_ARXIV_HTML_BASE = "https://arxiv.org/html/{arxiv_id}/"
_AR5IV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}/"


def base_url_for_source(source: HtmlSource, arxiv_id: str) -> str:
    if source is HtmlSource.ARXIV:
        return _ARXIV_HTML_BASE.format(arxiv_id=arxiv_id)
    return _AR5IV_HTML_BASE.format(arxiv_id=arxiv_id)
