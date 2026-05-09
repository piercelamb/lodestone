"""Tests covering convert_paper's PDF-fallback branch."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.scripts import fetch_paper as fp
from _system.scripts.convert_paper import convert
from _system.scripts.fetch_paper import (
    PDF_SENTINEL_PREFIX,
    USER_AGENT,
    _ArxivMetadata,
    fetch,
)


FIXTURE_PDF = Path(__file__).parent / "fixtures" / "pdf" / "sample.pdf"


def _meta() -> _ArxivMetadata:
    return _ArxivMetadata(
        title="PDF Paper",
        authors=["Author"],
        abstract="Abstract.",
        published="2026-04-09",
        comment=None,
        summary=None,
        pdf_url="https://arxiv.org/pdf/2604.23644",
    )


def _client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": USER_AGENT},
        timeout=5.0,
        follow_redirects=True,
    )


@pytest.fixture
def conn(tmp_path: Path):
    p = tmp_path / "lodestone.db"
    c = get_conn(p)
    init_db(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def silence_pwc(monkeypatch):
    monkeypatch.setattr(fp, "_sleep", lambda s: None)


def _ingest_pdf(conn, arxiv_id: str = "2604.23644") -> str:
    pdf_blob = FIXTURE_PDF.read_bytes()

    def handler(req):
        url = str(req.url)
        if "/html/" in url:
            return httpx.Response(404)
        if "/e-print/" in url:
            return httpx.Response(404)
        if "/pdf/" in url:
            return httpx.Response(
                200, content=pdf_blob,
                headers={"content-type": "application/pdf"},
            )
        return httpx.Response(404)

    with _client(handler) as c:
        pm = fetch(conn=conn, arxiv_id=arxiv_id, client=c, arxiv_lookup=lambda _: _meta())
    return pm.paper_name


def test_convert_pdf_fallback_produces_markdown_and_flags_review(conn):
    name = _ingest_pdf(conn)
    result = convert(paper_name=name, conn=conn)

    assert result.markdown_chars > 0
    assert result.figures == 0
    assert result.references == 0

    row = conn.execute(
        """
        SELECT markdown, raw_html, status, needs_review
          FROM papers WHERE paper_name = ?
        """,
        (name,),
    ).fetchone()
    markdown, raw_html, status, needs_review = row
    assert "xyzzy-marker" in markdown
    assert not markdown.startswith(PDF_SENTINEL_PREFIX)
    assert raw_html is None
    assert status == "converted"
    assert needs_review == 1

    refs = conn.execute(
        "SELECT COUNT(*) FROM paper_references WHERE paper_id = "
        "(SELECT id FROM papers WHERE paper_name = ?)",
        (name,),
    ).fetchone()[0]
    assert refs == 0


def test_convert_pdf_fallback_normalizes_heading_hierarchy(conn, monkeypatch):
    """Heading levels from font-size clustering get re-derived from title text.

    pymupdf4llm collapses parent and child sections to the same level on
    PDFs where they share a font size. ``convert_paper`` runs the
    PDF-fallback markdown through ``normalize_pdf_headings``, so a child
    like ``6.1 Layout Detection`` lands at L3 under its ``6. Results``
    L2 parent instead of as a sibling.
    """
    from _system.pdf import extract as pdf_extract

    synthetic = (
        "## 1. Introduction\n"
        "intro body with xyzzy-marker.\n"
        "\n"
        "## 1.1 Contributions\n"
        "contribs body.\n"
        "\n"
        "## 6. Results\n"
        "results body.\n"
        "\n"
        "## 6.1 Layout Detection\n"
        "layout body.\n"
        "\n"
        "### Conclusion\n"
        "conclusion body.\n"
    )
    monkeypatch.setattr(pdf_extract, "extract_markdown", lambda _b: synthetic)

    name = _ingest_pdf(conn)
    convert(paper_name=name, conn=conn)

    markdown = conn.execute(
        "SELECT markdown FROM papers WHERE paper_name = ?", (name,),
    ).fetchone()[0]

    lines = markdown.splitlines()
    # Numeric authoritative: parents at H2, children at H3.
    assert "## 1. Introduction" in lines
    assert "### 1.1 Contributions" in lines
    assert "## 6. Results" in lines
    assert "### 6.1 Layout Detection" in lines
    # Canonical promotion: H3 `Conclusion` → H2.
    assert "## Conclusion" in lines
    # Confirm the broken sibling form is gone.
    assert "## 1.1 Contributions" not in lines
    assert "## 6.1 Layout Detection" not in lines
