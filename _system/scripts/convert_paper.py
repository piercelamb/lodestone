"""Convert a FETCHED paper's raw_html to markdown (pure compute, no network)."""
from __future__ import annotations

import argparse
import json
import sqlite3
from typing import NamedTuple

from _system.db.connection import get_conn, transaction
from _system.html import latexml_parser
from _system.schemas.paper_metadata import HtmlSource, PaperStatus, can_run_from
from _system.utils.arxiv_urls import base_url_for_source
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.convert_paper")


class PaperNotFound(Exception):
    pass


class RawHtmlMissing(Exception):
    pass


class FigureCountMismatch(Exception):
    pass


class StageNotAllowed(Exception):
    pass


class ConvertResult(NamedTuple):
    paper_name: str
    status: str
    markdown_chars: int
    figures: int
    references: int
    references_resolved_forward: int
    references_resolved_backward: int


def convert(
    *,
    paper_name: str,
    conn: sqlite3.Connection,
    force: bool = False,
) -> ConvertResult:
    """Convert a FETCHED paper's raw_html to markdown; update the papers row."""
    del force  # no-op in convert; --force must cascade to fetch to restore raw_html

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

    try:
        current = PaperStatus(status_str)
    except ValueError as exc:
        raise StageNotAllowed(
            f"paper_name={paper_name!r}: unrecognized status={status_str!r}"
        ) from exc
    if not can_run_from(current, PaperStatus.CONVERTED):
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
        raise RawHtmlMissing(f"paper_name={paper_name!r}: raw_html is NULL")

    try:
        source = HtmlSource(html_source_str)
    except ValueError as exc:
        raise ValueError(
            f"paper_name={paper_name!r}: unknown html_source={html_source_str!r}"
        ) from exc
    base_url = base_url_for_source(source, arxiv_id)

    parsed = latexml_parser.parse(raw_html, base_url)

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
    # papers.figure_count is fetch's successfully-stored count; any drop
    # (image 404, placeholder) makes this disagree with the re-parse and
    # requires a --force re-fetch.
    if len(parsed.figures) != len(db_numbers):
        raise FigureCountMismatch(
            f"paper_name={paper_name!r}: parser produced {len(parsed.figures)} "
            f"figures but DB has {len(db_numbers)} (figure was likely dropped "
            f"during fetch; re-fetch with --force)"
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
        # Replace-all semantics on re-convert: the parser is the source of
        # truth for this paper's references. Drop everything we had stored
        # under paper_id and re-insert from the fresh parse. cited_paper_id
        # is left NULL on insert; both resolve passes below populate it.
        conn.execute(
            "DELETE FROM paper_references WHERE paper_id = ?", (paper_id,)
        )
        if parsed.references:
            conn.executemany(
                """
                INSERT INTO paper_references (
                    paper_id, bibitem_id, ref_number, raw_text, cited_arxiv_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (paper_id, r.bibitem_id, r.ref_number, r.raw_text, r.cited_arxiv_id)
                    for r in parsed.references
                ],
            )
        # Forward resolve: references this paper just inserted whose
        # cited_arxiv_id matches a paper already in the DB. The EXISTS
        # guard restricts the UPDATE to rows that will actually link, so
        # ``rowcount`` reflects resolved references rather than inspected
        # ones (a row whose cited_arxiv_id has no matching paper would
        # otherwise be touched as NULL→NULL and count toward rowcount).
        # Self-citation (paper -> own arxiv_id) is allowed.
        forward_cur = conn.execute(
            """
            UPDATE paper_references
               SET cited_paper_id = (
                   SELECT id FROM papers
                    WHERE arxiv_id = paper_references.cited_arxiv_id
               )
             WHERE paper_id = ?
               AND cited_arxiv_id IS NOT NULL
               AND cited_paper_id IS NULL
               AND EXISTS (
                   SELECT 1 FROM papers
                    WHERE arxiv_id = paper_references.cited_arxiv_id
               )
            """,
            (paper_id,),
        )
        forward_resolved = forward_cur.rowcount
        # Backward resolve: any *other* paper's previously-dangling reference
        # whose cited_arxiv_id matches THIS paper's arxiv_id. Needed because
        # paper A may have ingested before paper B; A's row stays NULL until
        # B's CONVERT runs.
        backward_cur = conn.execute(
            """
            UPDATE paper_references
               SET cited_paper_id = ?
             WHERE cited_arxiv_id = ?
               AND cited_paper_id IS NULL
            """,
            (paper_id, arxiv_id),
        )
        backward_resolved = backward_cur.rowcount

    _LOG.info(
        "converted paper_id=%s paper_name=%s html_source=%s markdown_chars=%d "
        "figures=%d references=%d forward_resolved=%d backward_resolved=%d",
        paper_id,
        paper_name,
        source.value,
        len(parsed.markdown),
        len(parsed.figures),
        len(parsed.references),
        forward_resolved,
        backward_resolved,
    )
    return ConvertResult(
        paper_name=paper_name,
        status=PaperStatus.CONVERTED.value,
        markdown_chars=len(parsed.markdown),
        figures=len(parsed.figures),
        references=len(parsed.references),
        references_resolved_forward=forward_resolved,
        references_resolved_backward=backward_resolved,
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

    conn = get_conn(args.db)
    try:
        result = convert(paper_name=args.paper, conn=conn, force=args.force)
    finally:
        conn.close()
    print(json.dumps(result._asdict()))


if __name__ == "__main__":
    _main()
