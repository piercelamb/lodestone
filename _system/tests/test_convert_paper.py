"""Unit tests for _system/scripts/convert_paper.py."""
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


_PRE_CLASSIFY_STATUSES = {
    PaperStatus.FETCHED.value,
    PaperStatus.CONVERTED.value,
    PaperStatus.FAILED_HTML.value,
    PaperStatus.FAILED_REPO.value,
}


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
    # Schema invariant: classified+ rows must carry both domain and
    # collection. Auto-supply them for any status outside the exempt
    # set (also covers test sentinels like ""/"not_a_real_status" that
    # exercise unknown-status paths — those fall under the trigger).
    if status in _PRE_CLASSIFY_STATUSES:
        domain = None
        collection = None
    else:
        domain = "rag"
        collection = "demo_collection"
        conn.execute(
            "INSERT OR IGNORE INTO domains (name) VALUES ('rag')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO collections (domain, name, description) "
            "VALUES ('rag', 'demo_collection', NULL)"
        )
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, html_source, ingested_at, status, raw_html,
            figure_count, domain, collection
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            domain,
            collection,
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
    _seed(conn)
    return conn


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


def test_raw_html_null_raises_raw_html_missing(conn):
    _seed(conn, raw_html=None)
    with pytest.raises(RawHtmlMissing) as exc_info:
        convert(paper_name="paper_name_2023", conn=conn)
    assert "paper_name_2023" in str(exc_info.value)


def test_raw_html_null_with_force_still_raises(conn):
    # --force is a no-op in convert; caller must cascade to fetch.
    _seed(conn, raw_html=None)
    with pytest.raises(RawHtmlMissing):
        convert(paper_name="paper_name_2023", conn=conn, force=True)


def test_figure_count_mismatch_parser_vs_db_raises(conn):
    paper_id = _seed(conn)
    conn.execute("DELETE FROM figures WHERE paper_id = ? AND figure_number = 2", (paper_id,))
    conn.execute("UPDATE papers SET figure_count = 1 WHERE id = ?", (paper_id,))
    with pytest.raises(FigureCountMismatch) as exc_info:
        convert(paper_name="paper_name_2023", conn=conn)
    assert "paper_name_2023" in str(exc_info.value)


def test_figure_count_mismatch_on_db_inconsistency_raises(conn):
    paper_id = _seed(conn)
    conn.execute("UPDATE papers SET figure_count = 99 WHERE id = ?", (paper_id,))
    with pytest.raises(FigureCountMismatch):
        convert(paper_name="paper_name_2023", conn=conn)


_FIGURE_WITH_PLACEHOLDER_HTML = """<!doctype html>
<html><body>
<section class="ltx_section" id="S1">
  <h2 class="ltx_title ltx_title_section"><span class="ltx_tag">1.</span> Method</h2>
  <p>Method prose.</p>
  <figure class="ltx_figure" id="S1.F1">
    <img src="fig1.png"/>
    <figcaption class="ltx_caption"><span class="ltx_tag">Figure 1.</span> Overview</figcaption>
  </figure>
  <figure class="ltx_figure" id="S1.F2">
    <figcaption class="ltx_caption"><span class="ltx_tag">Figure 2.</span> Sub-panel wrapper, no img</figcaption>
  </figure>
</section>
</body></html>
"""


def test_placeholder_figure_does_not_count_against_db(conn):
    """A <figure> with no <img> child is dropped at fetch by design; convert
    must not count it as a fetch-time anomaly."""
    _seed(conn, raw_html=_FIGURE_WITH_PLACEHOLDER_HTML, figure_count=1)
    paper_id = conn.execute(
        "SELECT id FROM papers WHERE paper_name = ?", ("paper_name_2023",)
    ).fetchone()[0]
    conn.execute(
        "DELETE FROM figures WHERE paper_id = ? AND figure_number = 2", (paper_id,)
    )
    # Should NOT raise — parser produces 2 figures total but only 1 is
    # fetch-eligible (Figure 2 is a placeholder), matching DB count of 1.
    result = convert(paper_name="paper_name_2023", conn=conn)
    assert result.figures == 2  # parsed.figures still counts placeholders for markdown


def test_status_failed_html_blocks_convert(conn):
    _seed(conn, status=PaperStatus.FAILED_HTML.value, raw_html=None)
    with pytest.raises(StageNotAllowed) as exc_info:
        convert(paper_name="paper_name_2023", conn=conn)
    assert "failed_html" in str(exc_info.value).lower()


def test_status_empty_blocks_convert(conn):
    _seed(conn, status="")
    with pytest.raises(StageNotAllowed):
        convert(paper_name="paper_name_2023", conn=conn)


def test_status_indexed_blocks_convert_as_moving_backwards(conn):
    _seed(conn, status=PaperStatus.INDEXED.value)
    with pytest.raises(StageNotAllowed):
        convert(paper_name="paper_name_2023", conn=conn)


def test_paper_not_found_raises(conn):
    with pytest.raises(PaperNotFound) as exc_info:
        convert(paper_name="no_such_paper", conn=conn)
    assert "no_such_paper" in str(exc_info.value)


def _spy_parse_base_url(monkeypatch) -> dict[str, str]:
    captured: dict[str, str] = {}
    real_parse = cp.latexml_parser.parse

    def spy(html: str, base_url: str):
        captured["base_url"] = base_url
        return real_parse(html, base_url)

    monkeypatch.setattr(cp.latexml_parser, "parse", spy)
    return captured


def test_html_source_arxiv_uses_arxiv_base_url(seeded_db, monkeypatch):
    captured = _spy_parse_base_url(monkeypatch)
    convert(paper_name="paper_name_2023", conn=seeded_db)
    assert captured["base_url"].startswith("https://arxiv.org/html/2301.00001")


def test_html_source_ar5iv_uses_ar5iv_base_url(conn, monkeypatch):
    _seed(conn, html_source=HtmlSource.AR5IV.value)
    captured = _spy_parse_base_url(monkeypatch)
    convert(paper_name="paper_name_2023", conn=conn)
    assert captured["base_url"].startswith("https://ar5iv.labs.arxiv.org/html/2301.00001")


def test_unknown_html_source_raises(conn):
    _seed(conn, html_source="bogus")
    with pytest.raises(ValueError):
        convert(paper_name="paper_name_2023", conn=conn)


def test_convert_paper_source_has_no_httpx_import():
    source = Path(cp.__file__).read_text()
    assert "import httpx" not in source
    assert "from httpx" not in source


def test_cli_prints_json_summary_on_success(tmp_path: Path, capsys):
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


# --- bibliographic reference persistence ---


_TWO_FIGURE_HTML_WITH_BIBLIST = """<!doctype html>
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
<section class="ltx_bibliography">
  <h2 class="ltx_title">References</h2>
  <ul class="ltx_biblist">
    <li id="bib.bib1" class="ltx_bibitem">
      <span class="ltx_bibblock">Beltagy et al. arXiv:2004.05150.</span>
    </li>
    <li id="bib.bib2" class="ltx_bibitem">
      <span class="ltx_bibblock">Brown et al. NeurIPS, 2020.</span>
    </li>
    <li id="bib.bib3" class="ltx_bibitem">
      <span class="ltx_bibblock">Packer et al. MemGPT. arXiv:2310.08560.</span>
    </li>
  </ul>
</section>
</body></html>
"""


def test_references_inserted_on_convert(conn):
    """Standard ltx_bibitem references end up in paper_references."""
    paper_id = _seed(conn, raw_html=_TWO_FIGURE_HTML_WITH_BIBLIST)
    convert(paper_name="paper_name_2023", conn=conn)
    rows = conn.execute(
        """
        SELECT bibitem_id, ref_number, cited_arxiv_id, cited_paper_id
          FROM paper_references
         WHERE paper_id = ?
         ORDER BY ref_number
        """,
        (paper_id,),
    ).fetchall()
    assert len(rows) == 3
    assert [r[0] for r in rows] == ["bib.bib1", "bib.bib2", "bib.bib3"]
    assert [r[1] for r in rows] == [1, 2, 3]
    assert [r[2] for r in rows] == ["2004.05150", None, "2310.08560"]
    # No other paper exists, so cited_paper_id stays NULL even when an
    # arxiv-id was extracted.
    assert all(r[3] is None for r in rows)


def test_convert_result_includes_reference_counts(conn):
    _seed(conn, raw_html=_TWO_FIGURE_HTML_WITH_BIBLIST)
    result = convert(paper_name="paper_name_2023", conn=conn)
    assert result.references == 3
    assert result.references_resolved_forward == 0
    assert result.references_resolved_backward == 0


def test_forward_resolution_links_already_ingested_paper(conn):
    """Paper B already exists with arxiv_id 2310.08560; converting paper A
    that cites it should set cited_paper_id = B.id during the same txn."""
    paper_b_id = _seed(
        conn,
        arxiv_id="2310.08560",
        paper_name="memgpt_paper_2023",
        status=PaperStatus.CONVERTED.value,
        raw_html=None,
        figure_count=0,
    )
    paper_a_id = _seed(
        conn,
        arxiv_id="2401.00001",
        paper_name="paper_a_2024",
        raw_html=_TWO_FIGURE_HTML_WITH_BIBLIST,
    )
    result = convert(paper_name="paper_a_2024", conn=conn)
    assert result.references_resolved_forward == 1
    assert result.references_resolved_backward == 0
    rows = conn.execute(
        "SELECT cited_arxiv_id, cited_paper_id FROM paper_references "
        "WHERE paper_id = ? ORDER BY ref_number",
        (paper_a_id,),
    ).fetchall()
    assert (rows[2][0], rows[2][1]) == ("2310.08560", paper_b_id)
    # The other extracted arxiv-id (2004.05150) doesn't match any paper.
    assert rows[0] == ("2004.05150", None)


def test_backward_resolution_links_dangling_references(conn):
    """Paper A was converted earlier with a reference to arxiv 2310.08560
    that didn't resolve (cited_paper_id NULL because B wasn't in the DB).
    When B subsequently lands and runs CONVERT, A's row must update."""
    paper_a_id = _seed(
        conn,
        arxiv_id="2401.00001",
        paper_name="paper_a_2024",
        raw_html=_TWO_FIGURE_HTML_WITH_BIBLIST,
    )
    convert(paper_name="paper_a_2024", conn=conn)
    # Sanity: nothing resolved yet.
    pre = conn.execute(
        "SELECT cited_paper_id FROM paper_references "
        "WHERE paper_id = ? AND cited_arxiv_id = ?",
        (paper_a_id, "2310.08560"),
    ).fetchone()
    assert pre[0] is None

    # Now ingest paper B (the cited paper) and convert it.
    paper_b_id = _seed(
        conn,
        arxiv_id="2310.08560",
        paper_name="memgpt_paper_2023",
        raw_html=_TWO_FIGURE_HTML,  # B has no bibliography of its own
    )
    result = convert(paper_name="memgpt_paper_2023", conn=conn)
    assert result.references_resolved_backward == 1
    post = conn.execute(
        "SELECT cited_paper_id FROM paper_references "
        "WHERE paper_id = ? AND cited_arxiv_id = ?",
        (paper_a_id, "2310.08560"),
    ).fetchone()
    assert post[0] == paper_b_id


def test_re_convert_replaces_old_references(conn):
    """Convert is replace-all: an earlier set of stale references must not
    survive into a re-convert run."""
    paper_id = _seed(conn, raw_html=_TWO_FIGURE_HTML_WITH_BIBLIST)
    convert(paper_name="paper_name_2023", conn=conn)
    # Mutate one of the rows to detect replacement.
    conn.execute(
        "UPDATE paper_references SET raw_text = 'STALE' "
        "WHERE paper_id = ? AND ref_number = 1",
        (paper_id,),
    )
    # Restore raw_html (convert nulled it) and re-run with the same html.
    conn.execute(
        "UPDATE papers SET raw_html = ?, status = ? WHERE id = ?",
        (_TWO_FIGURE_HTML_WITH_BIBLIST, PaperStatus.CONVERTED.value, paper_id),
    )
    convert(paper_name="paper_name_2023", conn=conn)
    rows = conn.execute(
        "SELECT raw_text FROM paper_references "
        "WHERE paper_id = ? ORDER BY ref_number",
        (paper_id,),
    ).fetchall()
    assert all("STALE" not in r[0] for r in rows)


def test_cascade_deletes_paper_references(conn):
    """delete_paper_cascade removes paper_references along with the paper."""
    from _system.db.cascade import delete_paper_cascade

    paper_id = _seed(conn, raw_html=_TWO_FIGURE_HTML_WITH_BIBLIST)
    convert(paper_name="paper_name_2023", conn=conn)
    assert conn.execute(
        "SELECT COUNT(*) FROM paper_references WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()[0] == 3
    conn.execute("BEGIN")
    delete_paper_cascade(conn, paper_id=paper_id)
    conn.commit()
    assert conn.execute(
        "SELECT COUNT(*) FROM paper_references WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()[0] == 0


def test_cascade_nulls_outbound_cited_paper_id(conn):
    """When deleting paper B, any other paper's reference whose
    cited_paper_id pointed at B must be NULLed (not deleted) so the FK
    stays valid and the dangling reference can be re-resolved later."""
    from _system.db.cascade import delete_paper_cascade

    paper_b_id = _seed(
        conn,
        arxiv_id="2310.08560",
        paper_name="memgpt_paper_2023",
        status=PaperStatus.CONVERTED.value,
        raw_html=None,
        figure_count=0,
    )
    paper_a_id = _seed(
        conn,
        arxiv_id="2401.00001",
        paper_name="paper_a_2024",
        raw_html=_TWO_FIGURE_HTML_WITH_BIBLIST,
    )
    convert(paper_name="paper_a_2024", conn=conn)
    # Sanity: forward resolve linked paper A's ref to B.
    pre = conn.execute(
        "SELECT cited_paper_id FROM paper_references "
        "WHERE paper_id = ? AND cited_arxiv_id = ?",
        (paper_a_id, "2310.08560"),
    ).fetchone()
    assert pre[0] == paper_b_id
    conn.execute("BEGIN")
    delete_paper_cascade(conn, paper_id=paper_b_id)
    conn.commit()
    post = conn.execute(
        "SELECT cited_arxiv_id, cited_paper_id FROM paper_references "
        "WHERE paper_id = ? AND cited_arxiv_id = ?",
        (paper_a_id, "2310.08560"),
    ).fetchone()
    # cited_arxiv_id preserved so a re-ingest of B can re-resolve.
    assert post == ("2310.08560", None)


def test_convert_failure_rolls_back_references(conn, monkeypatch):
    """If anything inside the convert transaction raises after references
    are inserted, the transaction must roll back — paper_references rows
    must not survive."""
    paper_id = _seed(conn, raw_html=_TWO_FIGURE_HTML_WITH_BIBLIST)

    # Real parse runs; raise inside the txn afterwards by sabotaging the
    # backward-resolve UPDATE through a sqlite trigger.
    conn.execute(
        """
        CREATE TRIGGER abort_on_paper_refs
        BEFORE INSERT ON paper_references
        BEGIN
          SELECT RAISE(ABORT, 'forced abort for test');
        END
        """
    )
    with pytest.raises(sqlite3.IntegrityError):
        convert(paper_name="paper_name_2023", conn=conn)
    # Trigger ran, txn rolled back. No references should have been written.
    assert conn.execute(
        "SELECT COUNT(*) FROM paper_references WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()[0] == 0
    # And the markdown UPDATE that ran before the INSERT must also have
    # rolled back: status remains FETCHED, raw_html still set.
    row = conn.execute(
        "SELECT status, raw_html, markdown FROM papers WHERE id = ?",
        (paper_id,),
    ).fetchone()
    assert row[0] == PaperStatus.FETCHED.value
    assert row[1] is not None
    assert row[2] is None


def test_self_citation_resolves_to_same_paper(conn):
    """A v2 update referencing v1 (same arxiv_id) is harmless: forward
    resolve will set cited_paper_id = paper's own id."""
    self_cite_html = """<!doctype html>
<html><body>
<section class="ltx_bibliography">
  <h2 class="ltx_title">References</h2>
  <ul class="ltx_biblist">
    <li id="bib.bib1" class="ltx_bibitem">
      <span class="ltx_bibblock">Self. arXiv:2301.00001.</span>
    </li>
  </ul>
</section>
</body></html>
"""
    paper_id = _seed(
        conn,
        arxiv_id="2301.00001",
        raw_html=self_cite_html,
        figure_count=0,
    )
    result = convert(paper_name="paper_name_2023", conn=conn)
    assert result.references_resolved_forward == 1
    row = conn.execute(
        "SELECT cited_paper_id FROM paper_references WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    assert row[0] == paper_id


def test_paper_with_no_bibliography_inserts_zero_references(
    conn,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No bibliography section + no ltx_bibitem = empty reference list, no error."""
    import logging

    paper_id = _seed(conn)  # _TWO_FIGURE_HTML has no bibliography
    logger = logging.getLogger("lodestone.html.latexml_parser")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="lodestone.html.latexml_parser"):
            convert(paper_name="paper_name_2023", conn=conn)
    finally:
        logger.removeHandler(caplog.handler)
    assert conn.execute(
        "SELECT COUNT(*) FROM paper_references WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()[0] == 0
    assert any("no bibliography found" in r.getMessage() for r in caplog.records)
