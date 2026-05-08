"""Convert a FETCHED paper's raw_html to markdown (pure compute, no network)."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import NamedTuple

from _system.db.connection import get_conn, transaction
from _system.html import latexml_parser
from _system.latex import LATEX_SENTINEL_PREFIX
from _system.latex import figures as latex_figures
from _system.latex import walker as latex_walker
from _system.schemas.paper_metadata import HtmlSource, PaperStatus, can_run_from
from _system.utils.arxiv_urls import base_url_for_source
from _system.utils.citation_resolution import resolve_arxiv_citations
from _system.utils.source_resolution import SourceKind
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

    needs_review = False
    if source is HtmlSource.LATEX_LOCAL:
        markdown, references, figures_count, needs_review = _convert_latex(
            raw_html, db_numbers, paper_name,
        )
    else:
        base_url = base_url_for_source(source, arxiv_id)
        parsed = latexml_parser.parse(raw_html, base_url)
        markdown = parsed.markdown
        references = parsed.references
        figures_count = len(parsed.figures)

        # Only figures the parser flagged as fetch-eligible (had a usable src_url
        # or inline data) should ever reach the DB. <figure> blocks with no <img>
        # child surface as descriptors with both fields None and are dropped at
        # fetch by design — excluding them here keeps the genuine "image 404 /
        # decode failure during fetch" anomaly detectable without flagging
        # structural placeholders that no re-fetch could resurrect.
        fetchable = [
            f for f in parsed.figures
            if f.src_url is not None or f.inline_data is not None
        ]
        if len(fetchable) != len(db_numbers):
            raise FigureCountMismatch(
                f"paper_name={paper_name!r}: parser produced {len(fetchable)} "
                f"fetchable figures but DB has {len(db_numbers)} (image likely "
                f"404'd or failed to decode during fetch; re-fetch with --force)"
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
            (markdown, PaperStatus.CONVERTED.value, paper_name),
        )
        if needs_review:
            conn.execute(
                "UPDATE papers SET needs_review = 1 WHERE id = ?",
                (paper_id,),
            )
        # Replace-all semantics on re-convert: the parser is the source of
        # truth for this paper's references. Drop everything we had stored
        # under paper_id and re-insert from the fresh parse. cited_paper_id
        # is left NULL on insert; both resolve passes below populate it.
        conn.execute(
            "DELETE FROM paper_references WHERE paper_id = ?", (paper_id,)
        )
        if references:
            conn.executemany(
                """
                INSERT INTO paper_references (
                    paper_id, bibitem_id, ref_number, raw_text, cited_arxiv_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (paper_id, r.bibitem_id, r.ref_number, r.raw_text, r.cited_arxiv_id)
                    for r in references
                ],
            )
        # Forward + backward arxiv-id resolution. Backward pass also
        # touches `post_references` so a post that cited this paper before
        # the paper landed gets its cited_paper_id linked.
        forward_resolved, backward_resolved = resolve_arxiv_citations(
            conn,
            kind=SourceKind.PAPER,
            source_id=paper_id,
            source_arxiv_id=arxiv_id,
        )

    _LOG.info(
        "converted paper_id=%s paper_name=%s html_source=%s markdown_chars=%d "
        "figures=%d references=%d forward_resolved=%d backward_resolved=%d",
        paper_id,
        paper_name,
        source.value,
        len(markdown),
        figures_count,
        len(references),
        forward_resolved,
        backward_resolved,
    )
    return ConvertResult(
        paper_name=paper_name,
        status=PaperStatus.CONVERTED.value,
        markdown_chars=len(markdown),
        figures=figures_count,
        references=len(references),
        references_resolved_forward=forward_resolved,
        references_resolved_backward=backward_resolved,
    )


_FIGURE_REF_RE = re.compile(r"\(figure:(\d+)\)")
_LATEX_DISCOVERY_TEX_ROOT = Path("/nonexistent-convert-time")


def _convert_latex(
    raw_html: str,
    db_numbers: set[int],
    paper_name: str,
) -> tuple[str, list, int, bool]:
    """Run the LaTeX walker over the assembled .tex stored in raw_html.

    Returns ``(markdown, references, figures_count, needs_review)``.
    ``needs_review`` is True iff the walker reported any unknown macros,
    unknown envs, or per-section failures — convert sets
    ``papers.needs_review = 1`` so ``search.py --needs-review`` surfaces
    partial conversions for human inspection.
    """
    if not raw_html.startswith(LATEX_SENTINEL_PREFIX):
        raise RawHtmlMissing(
            f"paper_name={paper_name!r}: html_source=latex_local but raw_html "
            "does not start with the LaTeX sentinel; re-fetch with --force"
        )
    assembled_tex = raw_html[len(LATEX_SENTINEL_PREFIX):]

    # Re-run discovery so the walker's `\begin{figure}` ordinal alignment
    # matches fetch-time — captions for envs without DB rows (TikZ, PDF
    # figures) come from this pass; raster bytes live in the DB. The
    # bogus tex_root is fine: paths won't resolve, but we mark
    # has_image from db_numbers below.
    discovered = latex_figures.discover_figures(
        assembled_tex, _LATEX_DISCOVERY_TEX_ROOT,
    )
    walker_figs = [
        d._replace(has_image=d.figure_number in db_numbers)
        for d in discovered
    ]

    result = latex_walker.tex_to_markdown(assembled_tex, walker_figs)

    md_referenced = {int(m) for m in _FIGURE_REF_RE.findall(result.markdown)}
    missing = db_numbers - md_referenced
    if missing:
        raise FigureCountMismatch(
            f"paper_name={paper_name!r}: latex walker emitted markdown for "
            f"{len(md_referenced)} figures but DB has rows for "
            f"{sorted(db_numbers)}; missing refs for {sorted(missing)} "
            "(re-fetch with --force)"
        )

    needs_review = bool(
        result.skipped_macros or result.skipped_envs
        or result.failed_sections or result.parse_errors
    )
    if needs_review:
        _LOG.warning(
            "paper_name=%s: latex walker partial conversion — "
            "skipped_macros=%s skipped_envs=%s failed_sections=%s parse_errors=%d",
            paper_name, result.skipped_macros, result.skipped_envs,
            result.failed_sections, result.parse_errors,
        )

    return result.markdown, list(result.references), len(walker_figs), needs_review


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
