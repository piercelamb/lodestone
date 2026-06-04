"""Unit tests for _system/utils/acl_urls.py."""
from __future__ import annotations

import pytest

from _system.utils.acl_urls import acl_pdf_url, acl_xml_url, parse_acl_id


# ---------------------------------------------------------------------------
# Bare-id parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "2021.acl-long.285",
        "2025.acl-long.1191",
        "2025.acl-industry.35",
        "2024.emnlp-main.1",
        "2024.findings-acl.42",
        "2023.findings-emnlp.999",
    ],
)
def test_parse_modern_id_roundtrips(raw):
    assert parse_acl_id(raw) == raw


@pytest.mark.parametrize(
    "raw",
    ["P19-1001", "D18-1234", "W17-0101", "N18-2002"],
)
def test_parse_legacy_id_roundtrips(raw):
    assert parse_acl_id(raw) == raw


# ---------------------------------------------------------------------------
# URL shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://aclanthology.org/2021.acl-long.285", "2021.acl-long.285"),
        ("https://aclanthology.org/2021.acl-long.285/", "2021.acl-long.285"),
        ("http://aclanthology.org/2021.acl-long.285/", "2021.acl-long.285"),
        ("https://www.aclanthology.org/2021.acl-long.285", "2021.acl-long.285"),
        ("https://aclanthology.org/2021.acl-long.285.pdf", "2021.acl-long.285"),
        ("https://aclanthology.org/2021.acl-long.285.xml", "2021.acl-long.285"),
        ("https://aclanthology.org/2021.acl-long.285.bib", "2021.acl-long.285"),
        ("https://aclanthology.org/P19-1001/", "P19-1001"),
        ("https://aclanthology.org/P19-1001.pdf", "P19-1001"),
    ],
)
def test_parse_url_extracts_id(raw, expected):
    assert parse_acl_id(raw) == expected


# ---------------------------------------------------------------------------
# Whitespace and surface tolerance
# ---------------------------------------------------------------------------


def test_parse_strips_surrounding_whitespace():
    assert parse_acl_id("  2021.acl-long.285\n") == "2021.acl-long.285"


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "not-an-id",
        "2021.acl-long",       # missing the sequence number
        "2021.acl.long.285",   # dots in venue (must be hyphenated)
        "p19-1001",            # lowercase legacy prefix (legacy is uppercase only)
        "P190-1001",           # legacy year must be 2 digits
        "P19-100",             # legacy sequence must be 4 digits
        "arxiv.org/abs/2301.12345",   # wrong domain
        "https://arxiv.org/abs/2301.12345",
        "ftp://aclanthology.org/2021.acl-long.285",
    ],
)
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_acl_id(bad)


# ---------------------------------------------------------------------------
# URL builders
# ---------------------------------------------------------------------------


def test_acl_pdf_url():
    assert (
        acl_pdf_url("2021.acl-long.285")
        == "https://aclanthology.org/2021.acl-long.285.pdf"
    )


def test_acl_xml_url():
    assert (
        acl_xml_url("2021.acl-long.285")
        == "https://aclanthology.org/2021.acl-long.285.xml"
    )
