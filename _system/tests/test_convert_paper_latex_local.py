"""Tests covering convert_paper's LaTeX-source branch."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.scripts import convert_paper as cp
from _system.scripts import fetch_paper as fp
from _system.scripts.convert_paper import convert
from _system.scripts.fetch_paper import LATEX_SENTINEL_PREFIX, USER_AGENT, fetch, _ArxivMetadata


FIXTURE_SIMPLE = Path(__file__).parent / "fixtures" / "latex" / "simple.tar.gz"


def _meta() -> _ArxivMetadata:
    return _ArxivMetadata(
        title="LaTeX Paper",
        authors=["Author"],
        abstract="Abstract.",
        published="2025-10-09",
        comment=None,
        summary=None,
        pdf_url="https://arxiv.org/pdf/2510.07233",
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


def _ingest_latex(conn, arxiv_id: str = "2510.07233") -> str:
    """Run fetch with HTML 404 + e-print 200 so we end up in LaTeX path."""
    blob = FIXTURE_SIMPLE.read_bytes()

    def handler(req):
        url = str(req.url)
        if "/html/" in url:
            return httpx.Response(404)
        if "/e-print/" in url:
            return httpx.Response(
                200, content=blob,
                headers={"content-type": "application/x-eprint-tar"},
            )
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.7")
        return httpx.Response(404)

    with _client(handler) as c:
        pm = fetch(conn=conn, arxiv_id=arxiv_id, client=c, arxiv_lookup=lambda _: _meta())
    return pm.paper_name


def test_convert_latex_local_produces_markdown(conn):
    name = _ingest_latex(conn)
    result = convert(paper_name=name, conn=conn)
    assert result.markdown_chars > 0

    md = conn.execute(
        "SELECT markdown FROM papers WHERE paper_name = ?", (name,)
    ).fetchone()[0]
    assert "# Introduction" in md
    assert "Hello world" in md
    assert "![Figure 1: Caption A.](figure:1)" in md


def test_convert_latex_local_clears_raw_html(conn):
    name = _ingest_latex(conn)
    convert(paper_name=name, conn=conn)
    raw = conn.execute(
        "SELECT raw_html FROM papers WHERE paper_name = ?", (name,)
    ).fetchone()[0]
    assert raw is None


def test_convert_latex_local_extracts_references(conn):
    name = _ingest_latex(conn)
    convert(paper_name=name, conn=conn)
    rows = conn.execute(
        """
        SELECT r.bibitem_id, r.cited_arxiv_id, r.raw_text
          FROM paper_references r
          JOIN papers p ON p.id = r.paper_id
         WHERE p.paper_name = ?
        """,
        (name,),
    ).fetchall()
    assert len(rows) == 1
    bid, cited, raw_text = rows[0]
    assert bid == "x"
    assert cited == "2310.08560"
    assert "A. Author" in raw_text


def test_convert_latex_local_logs_unknown_macros_without_flagging_review(conn, monkeypatch, caplog):
    """A paper using \\fancyhighlight logs a walker warning but does NOT flag
    needs_review — that column is reserved for new-taxonomy review only.
    """
    import logging
    arxiv_id = "2510.07555"

    # Build a custom tarball with an unknown macro.
    import gzip
    import io
    import tarfile
    src = rb"""\documentclass{article}
\begin{document}
\section{S}
\fancyhighlight{important point}.
\end{document}
"""
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w") as tf:
        info = tarfile.TarInfo(name="main.tex")
        info.size = len(src)
        tf.addfile(info, io.BytesIO(src))
    blob = gzip.compress(inner.getvalue())

    def handler(req):
        url = str(req.url)
        if "/html/" in url:
            return httpx.Response(404)
        if "/e-print/" in url:
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

    # The lodestone root logger has propagate=False, so attach caplog's
    # handler directly (same pattern as test_ingest.py).
    logger = logging.getLogger("lodestone.scripts.convert_paper")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="lodestone.scripts.convert_paper"):
            convert(paper_name=pm.paper_name, conn=conn)
    finally:
        logger.removeHandler(caplog.handler)
    needs_review = conn.execute(
        "SELECT needs_review FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()[0]
    assert needs_review == 0
    assert any(
        "latex walker partial conversion" in r.message for r in caplog.records
    ), "walker partial-conversion warning should still fire for observability"


def test_convert_latex_local_raises_on_missing_sentinel(conn):
    """Defensive: raw_html without the sentinel must trip the guard."""
    name = _ingest_latex(conn)
    # Corrupt raw_html to drop the sentinel.
    conn.execute(
        "UPDATE papers SET raw_html = ?, status = ?  WHERE paper_name = ?",
        ("not-a-sentinel\n\\section{X}", "fetched", name),
    )
    conn.commit()
    with pytest.raises(cp.RawHtmlMissing):
        convert(paper_name=name, conn=conn)
