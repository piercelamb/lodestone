"""Unit tests for _system/scripts/convert_paper.py.

The convert stage is pure compute: it re-parses ``papers.raw_html`` with the
LaTeXML parser, writes ``papers.markdown``, and NULLs out ``raw_html``.
These tests seed an in-memory-ish sqlite DB with a single papers row plus
matching figures, then exercise the convert entry point.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.scripts import convert_paper as cp
from _system.scripts.convert_paper import (
    ConvertResult,
    FigureCountMismatch,
    PaperNotFound,
    RawHtmlMissing,
    StageNotAllowed,
    convert,
)
from _system.schemas.paper_metadata import HtmlSource, PaperStatus


# ---------------------------------------------------------------------------
# Fixture HTML / DB seeding helpers
# ---------------------------------------------------------------------------


_TWO_FIGURE_HTML = """<!doctype html>
<html><body>
<section class="ltx_section" id="S1">
  <h2 class="ltx_title ltx_title_section"><span class="ltx_tag">1.</span> Method</h2>
  <p>Method prose.</p>
  <figure class="ltx_figure" id="S1.F1">
    <img src="fig1.png"/>
    <figcaption class="ltx_caption"><span class="ltx_tag">Figure 1.</span> Overview</figcaption>
  </figure>
  <p>Between figures.</p>
  <figure class="ltx_figure" id="S1.F2">
    <img src="fig2.png"/>
    <figcaption class="ltx_caption"><span class="ltx_tag">Figure 2.</span> Detail</figcaption>
  </figure>
</section>
</body></html>
"""


def _seed(
    conn: sqlite3.Connection,
    *,
    status: str = PaperStatus.FETCHED.value,
    raw_html: str | None = _TWO_FIGURE_HTML,
    html_source: str = HtmlSource.ARXIV.value,
    figure_count: int = 2,
    arxiv_id: str = "2301.00001",
    paper_name: str = "paper_name_2023",
) -> int:
    """Insert a papers row + `figure_count` figures rows; return paper_id."""
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, html_source, ingested_at, status, raw_html,
            figure_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id,
            paper_name,
            "Title",
            '["A. Author"]',
            "2023-01-01",
            "Abstract text.",
            f"https://arxiv.org/pdf/{arxiv_id}",
            html_source,
            "2024-01-01T00:00:00+00:00",
            status,
            raw_html,
            figure_count,
        ),
    )
    paper_id = cur.lastrowid
    for n in range(1, figure_count + 1):
        conn.execute(
            """
            INSERT INTO figures (
                paper_id, figure_number, display_number, figure_id,
                caption, section_context, image, mime_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (paper_id, n, str(n), f"S1.F{n}", f"caption {n}", "Method", b"\x00", "image/png"),
        )
    return paper_id


@pytest.fixture
def seeded_db(conn: sqlite3.Connection):
    """Fresh migrated conn with a single FETCHED papers row + 2 figures."""
    _seed(conn)
    return conn


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_successful_convert_sets_markdown_and_nulls_raw_html(seeded_db):
    convert(paper_name="paper_name_2023", conn=seeded_db)
    row = seeded_db.execute(
        "SELECT markdown, raw_html FROM papers WHERE paper_name = ?",
        ("paper_name_2023",),
    ).fetchone()
    assert row[0] is not None
    assert row[0].lstrip().startswith("#"), f"markdown did not start with header: {row[0]!r}"
    assert "figure:1" in row[0]
    assert "figure:2" in row[0]
    assert row[1] is None


def test_status_after_success_is_converted(seeded_db):
    convert(paper_name="paper_name_2023", conn=seeded_db)
    row = seeded_db.execute(
        "SELECT status FROM papers WHERE paper_name = ?",
        ("paper_name_2023",),
    ).fetchone()
    assert row[0] == PaperStatus.CONVERTED.value


def test_converted_status_allows_rerun_replace_markdown(conn):
    """A paper re-entering convert with raw_html restored (e.g. after a fetch
    force-refetch) must overwrite a stale markdown and re-null raw_html."""
    _seed(conn, status=PaperStatus.CONVERTED.value)
    conn.execute(
        "UPDATE papers SET markdown = ? WHERE paper_name = ?",
        ("# stale", "paper_name_2023"),
    )
    convert(paper_name="paper_name_2023", conn=conn)
    row = conn.execute(
        "SELECT markdown, raw_html, status FROM papers WHERE paper_name = ?",
        ("paper_name_2023",),
    ).fetchone()
    assert row[0] != "# stale"
    assert "figure:1" in row[0]
    assert row[1] is None
    assert row[2] == PaperStatus.CONVERTED.value


# ---------------------------------------------------------------------------
# RawHtmlMissing
# ---------------------------------------------------------------------------


def test_raw_html_null_raises_raw_html_missing(conn):
    _seed(conn, raw_html=None)
    with pytest.raises(RawHtmlMissing) as exc_info:
        convert(paper_name="paper_name_2023", conn=conn)
    assert "paper_name_2023" in str(exc_info.value)


def test_raw_html_null_with_force_still_raises(conn):
    """--force in convert alone is a no-op; caller must cascade to fetch."""
    _seed(conn, raw_html=None)
    with pytest.raises(RawHtmlMissing):
        convert(paper_name="paper_name_2023", conn=conn, force=True)


# ---------------------------------------------------------------------------
# FigureCountMismatch
# ---------------------------------------------------------------------------


def test_figure_count_mismatch_on_dangling_reference_raises(conn):
    """markdown references figure:3 but only 2 figures rows exist."""
    paper_id = _seed(conn)
    # Insert a row where the parser will emit `figure:1..2` but we create only
    # 1 figure row -> mismatch on count.
    conn.execute("DELETE FROM figures WHERE paper_id = ? AND figure_number = 2", (paper_id,))
    conn.execute("UPDATE papers SET figure_count = 1 WHERE id = ?", (paper_id,))
    with pytest.raises(FigureCountMismatch) as exc_info:
        convert(paper_name="paper_name_2023", conn=conn)
    assert "paper_name_2023" in str(exc_info.value)


def test_figure_count_mismatch_on_db_inconsistency_raises(conn):
    """papers.figure_count disagrees with COUNT(*) from figures — raise."""
    paper_id = _seed(conn)
    conn.execute("UPDATE papers SET figure_count = 99 WHERE id = ?", (paper_id,))
    with pytest.raises(FigureCountMismatch):
        convert(paper_name="paper_name_2023", conn=conn)


# ---------------------------------------------------------------------------
# Status gates
# ---------------------------------------------------------------------------


def test_status_failed_html_blocks_convert(conn):
    _seed(conn, status=PaperStatus.FAILED_HTML.value, raw_html=None)
    with pytest.raises(StageNotAllowed) as exc_info:
        convert(paper_name="paper_name_2023", conn=conn)
    assert "failed_html" in str(exc_info.value).lower()


def test_status_empty_blocks_convert(conn):
    """An empty-string status (invalid state) is treated as not-allowed."""
    _seed(conn, status="")
    with pytest.raises(StageNotAllowed):
        convert(paper_name="paper_name_2023", conn=conn)


def test_status_indexed_blocks_convert_as_moving_backwards(conn):
    """can_run_from(INDEXED, CONVERTED) is False — we can't go backwards."""
    _seed(conn, status=PaperStatus.INDEXED.value)
    with pytest.raises(StageNotAllowed):
        convert(paper_name="paper_name_2023", conn=conn)


# ---------------------------------------------------------------------------
# Paper not found
# ---------------------------------------------------------------------------


def test_paper_not_found_raises(conn):
    with pytest.raises(PaperNotFound) as exc_info:
        convert(paper_name="no_such_paper", conn=conn)
    assert "no_such_paper" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Base URL wiring
# ---------------------------------------------------------------------------


def test_html_source_arxiv_uses_arxiv_base_url(seeded_db, monkeypatch):
    captured: dict[str, str] = {}
    real_parse = cp.latexml_parser.parse

    def spy_parse(html: str, base_url: str):
        captured["base_url"] = base_url
        return real_parse(html, base_url)

    monkeypatch.setattr(cp.latexml_parser, "parse", spy_parse)
    convert(paper_name="paper_name_2023", conn=seeded_db)
    assert captured["base_url"].startswith("https://arxiv.org/html/2301.00001")


def test_html_source_ar5iv_uses_ar5iv_base_url(conn, monkeypatch):
    _seed(conn, html_source=HtmlSource.AR5IV.value)
    captured: dict[str, str] = {}
    real_parse = cp.latexml_parser.parse

    def spy_parse(html: str, base_url: str):
        captured["base_url"] = base_url
        return real_parse(html, base_url)

    monkeypatch.setattr(cp.latexml_parser, "parse", spy_parse)
    convert(paper_name="paper_name_2023", conn=conn)
    assert captured["base_url"].startswith("https://ar5iv.labs.arxiv.org/html/2301.00001")


def test_unknown_html_source_raises(conn):
    _seed(conn, html_source="bogus")
    with pytest.raises(ValueError):
        convert(paper_name="paper_name_2023", conn=conn)


# ---------------------------------------------------------------------------
# No network
# ---------------------------------------------------------------------------


def test_convert_paper_source_has_no_httpx_import():
    """Static check: the convert module must not import httpx at all.

    A runtime monkeypatch approach is unreliable because httpx is usually
    already loaded by sibling modules (fetch_paper) in the test session.
    Grepping the module source is the honest check.
    """
    source_path = Path(cp.__file__)
    source = source_path.read_text()
    assert "import httpx" not in source, (
        f"{source_path} must not import httpx (convert is zero-network)"
    )
    assert "from httpx" not in source


def test_convert_does_not_construct_httpx_client(seeded_db, monkeypatch):
    """Runtime sentinel: force-load httpx and assert convert doesn't call it."""
    import httpx

    class _Boom:
        def __init__(self, *a, **kw):
            raise AssertionError("convert() must not construct httpx.Client")

    monkeypatch.setattr(httpx, "Client", _Boom)
    convert(paper_name="paper_name_2023", conn=seeded_db)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_prints_json_summary_on_success(tmp_path: Path, capsys, monkeypatch):
    db_path = tmp_path / "lodestone.db"
    conn_ = get_conn(db_path)
    init_db(conn_)
    _seed(conn_)
    conn_.close()

    cp._main(["--paper", "paper_name_2023", "--db", str(db_path)])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["paper_name"] == "paper_name_2023"
    assert payload["status"] == PaperStatus.CONVERTED.value
    assert payload["figures"] == 2
    assert payload["markdown_chars"] > 0


def test_convert_returns_convert_result(seeded_db):
    result = convert(paper_name="paper_name_2023", conn=seeded_db)
    assert isinstance(result, ConvertResult)
    assert result.paper_name == "paper_name_2023"
    assert result.status == PaperStatus.CONVERTED.value
    assert result.figures == 2
    assert result.markdown_chars > 0
