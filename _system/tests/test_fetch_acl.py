"""Integration tests for _system/scripts/fetch_acl.py."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from _system.pdf import PDF_SENTINEL_PREFIX
from _system.scripts.fetch_acl import _parse_mods, fetch
from _system.scripts.fetch_paper import IngestExtractionFailed
from _system.utils.http import USER_AGENT


FIXTURE_PDF = Path(__file__).parent / "fixtures" / "pdf" / "sample.pdf"


SAMPLE_MODS = """<?xml version="1.0" encoding="UTF-8"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods ID="2021.acl-long.285">
    <titleInfo>
      <title>Toy Title For Testing</title>
    </titleInfo>
    <name type="personal">
      <namePart type="given">Ada</namePart>
      <namePart type="family">Lovelace</namePart>
      <role>
        <roleTerm authority="marcrelator" type="text">author</roleTerm>
      </role>
    </name>
    <name type="personal">
      <namePart type="given">Grace</namePart>
      <namePart type="family">Hopper</namePart>
      <role>
        <roleTerm authority="marcrelator" type="text">author</roleTerm>
      </role>
    </name>
    <abstract>A tiny abstract for unit tests.</abstract>
    <originInfo>
      <publisher>Association for Computational Linguistics</publisher>
      <dateIssued>2021</dateIssued>
    </originInfo>
    <identifier type="doi">10.18653/v1/2021.acl-long.285</identifier>
    <location>
      <url>https://aclanthology.org/2021.acl-long.285</url>
    </location>
  </mods>
</modsCollection>
"""


def _client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": USER_AGENT},
        timeout=5.0,
        follow_redirects=True,
    )


# ---------------------------------------------------------------------------
# MODS parsing
# ---------------------------------------------------------------------------


def test_parse_mods_extracts_title_authors_abstract_year():
    meta = _parse_mods(SAMPLE_MODS, "2021.acl-long.285")
    assert meta.title == "Toy Title For Testing"
    assert meta.authors == ["Ada Lovelace", "Grace Hopper"]
    assert meta.abstract == "A tiny abstract for unit tests."
    assert meta.published == "2021-01-01"
    assert meta.pdf_url == "https://aclanthology.org/2021.acl-long.285.pdf"


def test_parse_mods_handles_missing_abstract():
    xml = """<?xml version="1.0"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods>
    <titleInfo><title>No Abstract Here</title></titleInfo>
    <name type="personal">
      <namePart type="given">X</namePart>
      <namePart type="family">Y</namePart>
    </name>
    <originInfo><dateIssued>2024</dateIssued></originInfo>
  </mods>
</modsCollection>
"""
    meta = _parse_mods(xml, "2024.acl-long.1")
    assert meta.title == "No Abstract Here"
    assert meta.authors == ["X Y"]
    assert meta.abstract == ""
    assert meta.published == "2024-01-01"


def test_parse_mods_skips_corporate_names():
    """Only ``name type='personal'`` should land in authors."""
    xml = """<?xml version="1.0"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods>
    <titleInfo><title>Mixed Names</title></titleInfo>
    <name type="personal">
      <namePart type="given">Alice</namePart>
      <namePart type="family">Author</namePart>
    </name>
    <name type="corporate">
      <namePart>Association for Computational Linguistics</namePart>
    </name>
    <originInfo><dateIssued>2024</dateIssued></originInfo>
  </mods>
</modsCollection>
"""
    meta = _parse_mods(xml, "2024.acl-long.1")
    assert meta.authors == ["Alice Author"]


def test_parse_mods_preserves_mixed_content_abstract():
    """Inline markup in an abstract must not truncate the surrounding text."""
    xml = """<?xml version="1.0"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods>
    <titleInfo><title>Mixed Content</title></titleInfo>
    <name type="personal">
      <namePart type="given">X</namePart>
      <namePart type="family">Y</namePart>
    </name>
    <abstract>Lead text <i>italic phrase</i> tail content.</abstract>
    <originInfo><dateIssued>2024</dateIssued></originInfo>
  </mods>
</modsCollection>
"""
    meta = _parse_mods(xml, "2024.acl-long.1")
    assert "Lead text" in meta.abstract
    assert "italic phrase" in meta.abstract
    assert "tail content" in meta.abstract


def test_parse_mods_skips_editor_role():
    """A <name type='personal'> with roleTerm 'editor' must not become an author."""
    xml = """<?xml version="1.0"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods>
    <titleInfo><title>Proceedings</title></titleInfo>
    <name type="personal">
      <namePart type="given">Alice</namePart>
      <namePart type="family">Author</namePart>
      <role><roleTerm authority="marcrelator" type="text">author</roleTerm></role>
    </name>
    <name type="personal">
      <namePart type="given">Eve</namePart>
      <namePart type="family">Editor</namePart>
      <role><roleTerm authority="marcrelator" type="text">editor</roleTerm></role>
    </name>
    <name type="personal">
      <namePart type="given">Tina</namePart>
      <namePart type="family">Translator</namePart>
      <role><roleTerm authority="marcrelator" type="code">trl</roleTerm></role>
    </name>
    <originInfo><dateIssued>2024</dateIssued></originInfo>
  </mods>
</modsCollection>
"""
    meta = _parse_mods(xml, "2024.acl-long.0")
    assert meta.authors == ["Alice Author"]


def test_parse_mods_concatenates_subtitle():
    xml = """<?xml version="1.0"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods>
    <titleInfo>
      <title>Lodestone</title>
      <subTitle>An ARA Compiler</subTitle>
    </titleInfo>
    <name type="personal">
      <namePart type="given">X</namePart>
      <namePart type="family">Y</namePart>
    </name>
    <originInfo><dateIssued>2024</dateIssued></originInfo>
  </mods>
</modsCollection>
"""
    meta = _parse_mods(xml, "2024.acl-long.1")
    assert meta.title == "Lodestone: An ARA Compiler"


def test_parse_mods_raises_on_missing_date():
    xml = """<?xml version="1.0"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods>
    <titleInfo><title>No Date Here</title></titleInfo>
    <name type="personal">
      <namePart type="given">X</namePart>
      <namePart type="family">Y</namePart>
    </name>
  </mods>
</modsCollection>
"""
    with pytest.raises(IngestExtractionFailed, match="dateIssued"):
        _parse_mods(xml, "2024.acl-long.1")


def test_parse_mods_raises_on_non_numeric_date():
    xml = """<?xml version="1.0"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods>
    <titleInfo><title>Forthcoming</title></titleInfo>
    <name type="personal">
      <namePart type="given">X</namePart>
      <namePart type="family">Y</namePart>
    </name>
    <originInfo><dateIssued>forthcoming</dateIssued></originInfo>
  </mods>
</modsCollection>
"""
    with pytest.raises(IngestExtractionFailed, match="dateIssued"):
        _parse_mods(xml, "2024.acl-long.1")


def test_parse_mods_accepts_name_without_type_attribute():
    """A <name> without a 'type' attr should default to personal/author."""
    xml = """<?xml version="1.0"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods>
    <titleInfo><title>Untyped Name</title></titleInfo>
    <name>
      <namePart type="given">Alice</namePart>
      <namePart type="family">Author</namePart>
    </name>
    <originInfo><dateIssued>2024</dateIssued></originInfo>
  </mods>
</modsCollection>
"""
    meta = _parse_mods(xml, "2024.acl-long.1")
    assert meta.authors == ["Alice Author"]


def test_parse_mods_preserves_generation_suffix():
    """Untyped <namePart> like 'Jr.' must survive when typed parts are also present."""
    xml = """<?xml version="1.0"?>
<modsCollection xmlns="http://www.loc.gov/mods/v3">
  <mods>
    <titleInfo><title>With Suffix</title></titleInfo>
    <name type="personal">
      <namePart type="given">Martin Luther</namePart>
      <namePart type="family">King</namePart>
      <namePart>Jr.</namePart>
    </name>
    <originInfo><dateIssued>2024</dateIssued></originInfo>
  </mods>
</modsCollection>
"""
    meta = _parse_mods(xml, "2024.acl-long.1")
    assert meta.authors == ["Martin Luther King Jr."]


# ---------------------------------------------------------------------------
# Full fetch() integration
# ---------------------------------------------------------------------------


def test_fetch_persists_pdf_fallback_row(conn):
    acl_id = "2021.acl-long.285"
    pdf_blob = FIXTURE_PDF.read_bytes()

    seen_urls: list[str] = []

    def handler(req):
        url = str(req.url)
        seen_urls.append(url)
        if url.endswith("/2021.acl-long.285.xml"):
            return httpx.Response(
                200, content=SAMPLE_MODS.encode("utf-8"),
                headers={"content-type": "application/xml"},
            )
        if url.endswith("/2021.acl-long.285.pdf"):
            return httpx.Response(
                200, content=pdf_blob,
                headers={"content-type": "application/pdf"},
            )
        return httpx.Response(404)

    with _client(handler) as c:
        pm = fetch(conn=conn, acl_id=acl_id, client=c)

    assert pm.arxiv_id == "acl:2021.acl-long.285"
    assert pm.title == "Toy Title For Testing"
    assert pm.html_source == "pdf_fallback"
    assert pm.raw_html.startswith(PDF_SENTINEL_PREFIX)
    assert pm.pdf_url == "https://aclanthology.org/2021.acl-long.285.pdf"
    assert pm.date == "2021-01-01"
    assert pm.content_hash is not None

    # Persisted row reflects the same.
    row = conn.execute(
        "SELECT arxiv_id, title, html_source, pdf_url, date "
        "  FROM papers WHERE arxiv_id = ?",
        ("acl:2021.acl-long.285",),
    ).fetchone()
    assert row is not None
    assert row[0] == "acl:2021.acl-long.285"
    assert row[1] == "Toy Title For Testing"
    assert row[2] == "pdf_fallback"
    assert row[3] == "https://aclanthology.org/2021.acl-long.285.pdf"
    assert row[4] == "2021-01-01"

    # Both the XML and the PDF endpoints were hit; no PwC roundtrip.
    assert any(u.endswith(".xml") for u in seen_urls)
    assert any(u.endswith(".pdf") for u in seen_urls)
    assert not any("paperswithcode.com" in u for u in seen_urls)


def test_fetch_is_idempotent_when_present(conn):
    """A second fetch with force=False should short-circuit on existing row."""
    acl_id = "2021.acl-long.285"
    pdf_blob = FIXTURE_PDF.read_bytes()

    calls = {"count": 0}

    def handler(req):
        calls["count"] += 1
        url = str(req.url)
        if url.endswith(".xml"):
            return httpx.Response(
                200, content=SAMPLE_MODS.encode("utf-8"),
                headers={"content-type": "application/xml"},
            )
        if url.endswith(".pdf"):
            return httpx.Response(
                200, content=pdf_blob,
                headers={"content-type": "application/pdf"},
            )
        return httpx.Response(404)

    with _client(handler) as c:
        fetch(conn=conn, acl_id=acl_id, client=c)
        first_count = calls["count"]
        fetch(conn=conn, acl_id=acl_id, client=c)
        second_count = calls["count"]

    # Second call should hit no endpoints — fetch returns the existing row.
    assert second_count == first_count
