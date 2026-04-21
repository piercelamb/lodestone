"""Canonical URL helpers for arxiv-hosted paper sources.

Centralizes the ``html_source → base_url`` mapping so ``fetch_paper`` and
``convert_paper`` cannot drift. The trailing slash is load-bearing —
``lxml.html.fromstring`` + ``urljoin`` drops the last path segment of a
base URL that lacks a trailing separator, which silently breaks relative
figure ``src`` resolution on ar5iv (verified in section-07 tests).
"""
from __future__ import annotations

from _system.schemas.paper_metadata import HtmlSource

_ARXIV_HTML_BASE = "https://arxiv.org/html/{arxiv_id}/"
_AR5IV_HTML_BASE = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}/"


def base_url_for_source(source: HtmlSource, arxiv_id: str) -> str:
    """Return the ``<base href>`` URL for an LaTeXML document."""
    if source is HtmlSource.ARXIV:
        return _ARXIV_HTML_BASE.format(arxiv_id=arxiv_id)
    return _AR5IV_HTML_BASE.format(arxiv_id=arxiv_id)
