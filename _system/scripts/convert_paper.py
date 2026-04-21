"""Convert a FETCHED paper's raw_html to markdown (pure compute, no network).

This stage re-reads ``papers.raw_html`` (persisted by ``fetch_paper.py``),
re-runs the LaTeXML parser, and atomically writes ``papers.markdown`` while
nulling ``raw_html`` to reclaim DB space. The convert is idempotent only in
the sense that it may be re-run as long as ``raw_html`` is present — the
successful path clears ``raw_html``, so a second run raises
``RawHtmlMissing`` (the caller is expected to cascade ``--force`` back to
the fetch stage).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import NamedTuple

from _system.db.connection import get_conn, transaction
from _system.html import latexml_parser
from _system.schemas.paper_metadata import HtmlSource, PaperStatus, can_run_from
from _system.utils.arxiv_urls import base_url_for_source
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.convert_paper")

_FIGURE_REF_RE = re.compile(r"figure:(\d+)")


class PaperNotFound(Exception):
    """No papers row matches the requested paper_name."""


class RawHtmlMissing(Exception):
    """papers.raw_html has already been cleared; re-fetch to re-convert."""


class FigureCountMismatch(Exception):
    """Parsed markdown's figure references don't match the figures table."""


class StageNotAllowed(Exception):
    """Current paper status does not permit running the convert stage."""


class ConvertResult(NamedTuple):
    paper_name: str
    status: str
    markdown_chars: int
    figures: int


def convert(
    *,
    paper_name: str,
    conn: sqlite3.Connection,
    force: bool = False,
) -> ConvertResult:
    """Convert a FETCHED paper's raw_html to markdown; update the papers row.

    Raises
    ------
    PaperNotFound: if no papers row matches ``paper_name``.
    RawHtmlMissing: if raw_html is NULL (already converted or never fetched).
    FigureCountMismatch: if markdown references figure:N that has no figures
        row, or the parser's figure count disagrees with papers.figure_count.
        papers.figure_count reflects the count of *successfully stored*
        figures during fetch; any fetch-time figure download drop will make
        this check fail and require a ``--force`` re-fetch (per plan).
    StageNotAllowed: if can_run_from(current_status, CONVERTED) is False, or
        if the row's status is empty / invalid.
    """
    if force:
        _LOG.info(
            "convert --force is a no-op; cascade to fetch (ingest.py --force) "
            "to re-download raw_html"
        )

    row = conn.execute(
        """
        SELECT id, arxiv_id, status, html_source, raw_html, figure_count
          FROM papers WHERE paper_name = ?
        """,
        (paper_name,),
    ).fetchone()
    if row is None:
        raise PaperNotFound(f"paper_name={paper_name!r} not found in papers table")
    paper_id, arxiv_id, status_str, html_source_str, raw_html, figure_count = row

    # Coerce status → PaperStatus. Both an empty status and an unrecognized
    # enum value are runtime-invalid states we treat as StageNotAllowed so
    # callers get a single exception type to catch with paper_name context.
    if not status_str:
        current: PaperStatus | None = None
    else:
        try:
            current = PaperStatus(status_str)
        except ValueError as exc:
            raise StageNotAllowed(
                f"paper_name={paper_name!r}: unrecognized status={status_str!r}"
            ) from exc
    # NOTE: can_run_from(None, CONVERTED) returns True (it treats None as
    # "no prior stage, runnable"), but the pipeline should never see a
    # status=NULL/empty row on a convert — the row was written by fetch and
    # carries at minimum status=FETCHED. Explicit None-check to reject that
    # invalid state loudly.
    if current is None or not can_run_from(current, PaperStatus.CONVERTED):
        extra = (
            " (FAILED_HTML is terminal — re-fetch required)"
            if current is PaperStatus.FAILED_HTML
            else ""
        )
        raise StageNotAllowed(
            f"paper_name={paper_name!r}: cannot run CONVERTED from status="
            f"{status_str!r}{extra}"
        )

    if raw_html is None:
        hint = (
            " --force is a no-op in convert; cascade to fetch via "
            "`ingest.py --arxiv-id ... --force`"
        )
        raise RawHtmlMissing(f"paper_name={paper_name!r}: raw_html is NULL.{hint}")

    try:
        source = HtmlSource(html_source_str)
    except ValueError as exc:
        raise ValueError(
            f"paper_name={paper_name!r}: unknown html_source={html_source_str!r}"
        ) from exc
    base_url = base_url_for_source(source, arxiv_id)

    parsed = latexml_parser.parse(raw_html, base_url)

    # Single query: fetch all figure numbers for this paper; derive the DB
    # count from the result length. Sanity-check vs papers.figure_count.
    db_numbers = {
        r[0]
        for r in conn.execute(
            "SELECT figure_number FROM figures WHERE paper_id = ?", (paper_id,)
        )
    }
    if len(db_numbers) != figure_count:
        raise FigureCountMismatch(
            f"paper_name={paper_name!r}: papers.figure_count={figure_count} "
            f"disagrees with COUNT(figures)={len(db_numbers)}"
        )
    # Strict equality: papers.figure_count is the successfully-stored count
    # from fetch; if fetch dropped a figure (image 404 etc.), this will fail
    # and the caller must cascade --force to the fetch stage.
    if len(parsed.figures) != len(db_numbers):
        raise FigureCountMismatch(
            f"paper_name={paper_name!r}: parser produced {len(parsed.figures)} "
            f"figures but DB has {len(db_numbers)} (figure was likely dropped "
            f"during fetch; re-fetch with --force)"
        )

    referenced = {int(m) for m in _FIGURE_REF_RE.findall(parsed.markdown)}
    dangling = sorted(referenced - db_numbers)
    if dangling:
        raise FigureCountMismatch(
            f"paper_name={paper_name!r}: markdown references figure numbers "
            f"{dangling} with no matching figures row"
        )

    with transaction(conn):
        conn.execute(
            """
            UPDATE papers
               SET markdown = ?,
                   raw_html = NULL,
                   status   = ?
             WHERE paper_name = ?
            """,
            (parsed.markdown, PaperStatus.CONVERTED.value, paper_name),
        )

    _LOG.info(
        "converted paper_id=%s paper_name=%s html_source=%s markdown_chars=%d figures=%d",
        paper_id,
        paper_name,
        source.value,
        len(parsed.markdown),
        len(parsed.figures),
    )
    return ConvertResult(
        paper_name=paper_name,
        status=PaperStatus.CONVERTED.value,
        markdown_chars=len(parsed.markdown),
        figures=len(parsed.figures),
    )


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a FETCHED paper's raw_html to markdown."
    )
    parser.add_argument("--paper", required=True, help="papers.paper_name")
    parser.add_argument(
        "--force",
        action="store_true",
        help="no-op for convert; forwarded for parity with other stages",
    )
    parser.add_argument("--db", default="lodestone.db", help="sqlite db path")
    args = parser.parse_args(argv)

    conn = get_conn(Path(args.db))
    try:
        result = convert(paper_name=args.paper, conn=conn, force=args.force)
    finally:
        conn.close()
    print(json.dumps(result._asdict()))


if __name__ == "__main__":
    _main()
