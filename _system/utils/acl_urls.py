"""Canonical URL helpers for ACL Anthology-hosted paper sources.

Mirrors :mod:`_system.utils.arxiv_urls` for the ``--acl`` ingest entry.

The Anthology does not publish an HTML/LaTeX fulltext equivalent to
arxiv's ``arxiv.org/html`` or e-print tarball — only the PDF. We rely
on per-paper MODS XML metadata at ``aclanthology.org/<id>.xml`` for
title/authors/abstract/year, and fall back to PDF rendering via
``pymupdf4llm`` for the body. Paper identity carries the ``acl:`` prefix
to avoid colliding with arxiv ids in the ``papers.arxiv_id`` column.
"""
from __future__ import annotations

import re

# Modern Anthology ids: ``YYYY.<venue>-<track>.N``. Venue and track are
# short kebab-case tokens (letters and digits, optional hyphens — e.g.
# ``acl-long``, ``acl-industry``, ``emnlp-main``, ``findings-acl``).
_ACL_MODERN_RE = re.compile(r"^\d{4}\.[a-z][a-z0-9]*(?:-[a-z0-9]+)*\.\d+$")

# Legacy pre-2020 ids: a single uppercase letter + 2-digit year + 4-digit
# sequence. Examples: ``P19-1001``, ``D18-1234``, ``W17-0101``.
_ACL_LEGACY_RE = re.compile(r"^[A-Z]\d{2}-\d{4}$")

_ACL_PDF_URL = "https://aclanthology.org/{acl_id}.pdf"
_ACL_XML_URL = "https://aclanthology.org/{acl_id}.xml"


def _looks_like_acl_id(value: str) -> bool:
    return bool(_ACL_MODERN_RE.match(value) or _ACL_LEGACY_RE.match(value))


def parse_acl_id(raw: str) -> str:
    """Extract the canonical ACL Anthology id from a URL or bare id.

    Accepts:
      - bare id: ``2021.acl-long.285``, ``P19-1001``
      - landing page: ``https://aclanthology.org/2021.acl-long.285`` (with or
        without trailing slash)
      - asset URLs: ``.pdf``, ``.xml``, ``.bib`` variants of the above

    Returns the canonical bare id. Raises :class:`ValueError` on input
    that doesn't match either id shape after URL stripping.
    """
    if not raw or not raw.strip():
        raise ValueError("acl id / URL is empty")
    value = raw.strip()

    # Strip the host prefix (with or without scheme + www).
    for scheme in ("https://", "http://", ""):
        for host in ("www.aclanthology.org/", "aclanthology.org/"):
            prefix = scheme + host
            if prefix and value.startswith(prefix):
                value = value[len(prefix):]
                break
        else:
            continue
        break

    value = value.rstrip("/")
    for suffix in (".pdf", ".xml", ".bib"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break

    if _looks_like_acl_id(value):
        return value
    raise ValueError(
        f"unrecognized ACL Anthology id / URL: {raw!r} "
        "(expected e.g. 2021.acl-long.285, 2025.acl-industry.35, or P19-1001)"
    )


def acl_pdf_url(acl_id: str) -> str:
    """Build the canonical ``aclanthology.org/<id>.pdf`` URL."""
    return _ACL_PDF_URL.format(acl_id=acl_id)


def acl_xml_url(acl_id: str) -> str:
    """Build the canonical ``aclanthology.org/<id>.xml`` (MODS) URL."""
    return _ACL_XML_URL.format(acl_id=acl_id)
