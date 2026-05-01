"""Integration tests for the LaTeX-source fallback path in fetch_paper."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.scripts import fetch_paper as fp
from _system.scripts.fetch_paper import (
    LATEX_SENTINEL_PREFIX,
    USER_AGENT,
    _ArxivMetadata,
    fetch,
)


FIXTURE_SIMPLE = Path(__file__).parent / "fixtures" / "latex" / "simple.tar.gz"
FIXTURE_TIKZ = Path(__file__).parent / "fixtures" / "latex" / "tikz_only.tar.gz"


def _meta() -> _ArxivMetadata:
    return _ArxivMetadata(
        title="LaTeX Paper",
        authors=["LX Author"],
        abstract="Things.",
        published="2025-10-09",
        comment=None,
        summary=None,
        pdf_url="https://arxiv.org/pdf/2510.07233",
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


def test_html_fail_then_eprint_succeeds(conn):
    arxiv_id = "2510.07233"
    blob = FIXTURE_SIMPLE.read_bytes()

    def handler(req):
        url = str(req.url)
        if "/html/" in url:
            return httpx.Response(404)
        if "arxiv.org/e-print/" in url:
            return httpx.Response(
                200, content=blob,
                headers={"content-type": "application/x-eprint-tar"},
            )
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.7\n%%EOF\n")
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    with _client(handler) as c:
        pm = fetch(conn=conn, arxiv_id=arxiv_id, client=c, arxiv_lookup=lambda _: _meta())

    assert pm.status == "fetched"
    assert pm.html_source == "latex_local"
    assert pm.raw_html.startswith(LATEX_SENTINEL_PREFIX)
    assert "\\section{Introduction}" in pm.raw_html

    rows = conn.execute(
        "SELECT figure_number, caption, mime_type FROM figures "
        "WHERE paper_id = (SELECT id FROM papers WHERE arxiv_id = ?)",
        (arxiv_id,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "Caption A."
    assert rows[0][2] in {"image/jpeg", "image/png"}


def test_eprint_pdf_only_falls_through_to_failed_html(conn):
    arxiv_id = "2510.07999"

    def handler(req):
        url = str(req.url)
        if "/html/" in url:
            return httpx.Response(404)
        if "arxiv.org/e-print/" in url:
            return httpx.Response(
                200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"}
            )
        return httpx.Response(404)

    with _client(handler) as c:
        pm = fetch(conn=conn, arxiv_id=arxiv_id, client=c, arxiv_lookup=lambda _: _meta())

    assert pm.status == "failed_html"
    assert pm.raw_html is None


def test_latex_fallback_disabled_via_kwarg(conn):
    arxiv_id = "2510.07000"

    captured_urls: list[str] = []

    def handler(req):
        url = str(req.url)
        captured_urls.append(url)
        if "/html/" in url:
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

    assert pm.status == "failed_html"
    # When the fallback is disabled, no e-print URL should be hit.
    assert not any("/e-print/" in u for u in captured_urls)


def test_latex_fallback_disabled_via_env(monkeypatch, conn):
    arxiv_id = "2510.07001"
    monkeypatch.setenv("LODESTONE_LATEX_FALLBACK", "0")

    captured: list[str] = []

    def handler(req):
        captured.append(str(req.url))
        if "/html/" in str(req.url):
            return httpx.Response(404)
        return httpx.Response(404)

    with _client(handler) as c:
        pm = fetch(conn=conn, arxiv_id=arxiv_id, client=c, arxiv_lookup=lambda _: _meta())

    assert pm.status == "failed_html"
    assert not any("/e-print/" in u for u in captured)


def test_tikz_only_paper_persists_with_zero_figure_rows(conn):
    arxiv_id = "2510.07050"
    blob = FIXTURE_TIKZ.read_bytes()

    def handler(req):
        url = str(req.url)
        if "/html/" in url:
            return httpx.Response(404)
        if "arxiv.org/e-print/" in url:
            return httpx.Response(
                200, content=blob,
                headers={"content-type": "application/x-eprint-tar"},
            )
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.7")
        return httpx.Response(404)

    with _client(handler) as c:
        pm = fetch(conn=conn, arxiv_id=arxiv_id, client=c, arxiv_lookup=lambda _: _meta())

    assert pm.html_source == "latex_local"
    n = conn.execute(
        "SELECT figure_count FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()[0]
    assert n == 0  # tikz figures don't materialize as DB rows
