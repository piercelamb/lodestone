"""Integration tests for the PDF (pymupdf4llm) fallback path in fetch_paper."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.scripts import fetch_paper as fp
from _system.scripts.fetch_paper import (
    PDF_SENTINEL_PREFIX,
    USER_AGENT,
    IngestExtractionFailed,
    _ArxivMetadata,
    fetch,
)


FIXTURE_PDF = Path(__file__).parent / "fixtures" / "pdf" / "sample.pdf"


def _meta() -> _ArxivMetadata:
    return _ArxivMetadata(
        title="PDF Paper",
        authors=["PD Author"],
        abstract="An abstract.",
        published="2026-04-09",
        comment=None,
        summary=None,
        pdf_url="https://arxiv.org/pdf/2604.23644",
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "lodestone.db"
    conn = get_conn(p)
    init_db(conn)
    conn.close()
    return p


@pytest.fixture
def conn(db_path: Path):
    c = get_conn(db_path)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture(autouse=True)
def silence_pwc_sleep(monkeypatch):
    monkeypatch.setattr(fp, "_sleep", lambda s: None)


def _client(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": USER_AGENT},
        timeout=5.0,
        follow_redirects=True,
    )


def test_html_fail_then_pdf_succeeds(conn):
    arxiv_id = "2604.23644"
    pdf_blob = FIXTURE_PDF.read_bytes()

    def handler(req):
        url = str(req.url)
        if "/html/" in url:
            return httpx.Response(404)
        if "arxiv.org/e-print/" in url:
            return httpx.Response(404)
        if "/pdf/" in url:
            return httpx.Response(
                200, content=pdf_blob,
                headers={"content-type": "application/pdf"},
            )
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    with _client(handler) as c:
        pm = fetch(conn=conn, arxiv_id=arxiv_id, client=c, arxiv_lookup=lambda _: _meta())

    assert pm.status == "fetched"
    assert pm.html_source == "pdf_fallback"
    assert pm.raw_html.startswith(PDF_SENTINEL_PREFIX)
    assert "xyzzy-marker" in pm.raw_html
    assert pm.content_hash is not None

    fig_count, html_source = conn.execute(
        "SELECT figure_count, html_source FROM papers WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    assert fig_count == 0
    assert html_source == "pdf_fallback"


def test_html_fail_latex_fail_pdf_fail_raises(conn):
    arxiv_id = "2604.23000"

    def handler(req):
        return httpx.Response(404)

    with _client(handler) as c:
        with pytest.raises(IngestExtractionFailed) as exc_info:
            fetch(conn=conn, arxiv_id=arxiv_id, client=c, arxiv_lookup=lambda _: _meta())

    assert "All extraction paths failed" in str(exc_info.value)
    # No row should be persisted on full failure.
    n = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()[0]
    assert n == 0


def test_latex_disabled_pdf_succeeds(conn):
    arxiv_id = "2604.23001"
    pdf_blob = FIXTURE_PDF.read_bytes()

    captured: list[str] = []

    def handler(req):
        url = str(req.url)
        captured.append(url)
        if "/html/" in url:
            return httpx.Response(404)
        if "/pdf/" in url:
            return httpx.Response(
                200, content=pdf_blob,
                headers={"content-type": "application/pdf"},
            )
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    with _client(handler) as c:
        pm = fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=c,
            arxiv_lookup=lambda _: _meta(),
            latex_fallback=False,
        )

    assert pm.html_source == "pdf_fallback"
    # When LaTeX fallback is disabled, no e-print URL should be hit.
    assert not any("/e-print/" in u for u in captured)


def test_pdf_disabled_full_failure(conn):
    arxiv_id = "2604.23002"
    pdf_blob = FIXTURE_PDF.read_bytes()

    captured: list[str] = []

    def handler(req):
        url = str(req.url)
        captured.append(url)
        if "/html/" in url:
            return httpx.Response(404)
        if "/pdf/" in url:
            return httpx.Response(
                200, content=pdf_blob,
                headers={"content-type": "application/pdf"},
            )
        return httpx.Response(404)

    with _client(handler) as c:
        with pytest.raises(IngestExtractionFailed):
            fetch(
                conn=conn,
                arxiv_id=arxiv_id,
                client=c,
                arxiv_lookup=lambda _: _meta(),
                latex_fallback=False,
                pdf_fallback=False,
            )
    # Even though the PDF endpoint would have returned 200, the disabled
    # flag should have prevented the request entirely.
    assert not any("/pdf/" in u for u in captured)


def test_pdf_disabled_via_env(monkeypatch, conn):
    arxiv_id = "2604.23003"
    monkeypatch.setenv("LODESTONE_PDF_FALLBACK", "0")
    monkeypatch.setenv("LODESTONE_LATEX_FALLBACK", "0")

    captured: list[str] = []

    def handler(req):
        captured.append(str(req.url))
        return httpx.Response(404)

    with _client(handler) as c:
        with pytest.raises(IngestExtractionFailed):
            fetch(conn=conn, arxiv_id=arxiv_id, client=c, arxiv_lookup=lambda _: _meta())
    assert not any("/pdf/" in u for u in captured)
    assert not any("/e-print/" in u for u in captured)
