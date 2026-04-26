"""Top-level Lodestone ingest CLI.

Runs the five-stage pipeline (fetch → convert → classify → extract → index)
for a single arxiv paper against a shared sqlite connection. Resumes from
the last completed stage (tracked via ``papers.status``) or cascades a
``--force`` delete and restarts.

The final JSON summary is printed on **stdout**; all progress / debug goes
to the shared logger (stderr), so downstream tooling can pipe the summary.

Stage function contract (formalized here): every stage accepts its shared
``sqlite3.Connection`` as a keyword-only argument named ``conn`` and
validates its own ``can_run_from`` preconditions. No subprocesses — this
module imports and calls directly so sqlite-vec extension load and model
caches (BGE singleton, GLiNER2 singleton) survive across stages within one
run.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from _system.db.cascade import delete_paper_cascade
from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.schemas.paper_metadata import PaperStatus, can_run_from
from _system.scripts.fetch_paper import fetch as fetch_stage
from _system.scripts.convert_paper import convert as convert_stage
from _system.scripts.classify_paper import classify as classify_stage
from _system.scripts.extract_entities import extract as extract_stage
from _system.scripts.index_paper import index_one as index_stage
from _system.scripts.validate_models import check_models
from _system.utils.arxiv_urls import parse_arxiv_id
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.ingest")


class Stage(StrEnum):
    FETCH = "fetch"
    CONVERT = "convert"
    CLASSIFY = "classify"
    EXTRACT = "extract"
    INDEX = "index"


class _PaperRow(NamedTuple):
    id: int
    name: str
    status: str
    needs_review: bool


def _get_paper_row(conn: sqlite3.Connection, arxiv_id: str) -> _PaperRow | None:
    row = conn.execute(
        "SELECT id, paper_name, status, needs_review FROM papers WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None
    return _PaperRow(id=row[0], name=row[1], status=row[2], needs_review=bool(row[3]))


def _force_delete_paper(conn: sqlite3.Connection, *, paper_id: int) -> None:
    """Cascade-delete one paper inside its own transaction.

    Global taxonomy (``canonical_terms``, ``term_aliases``,
    ``term_embeddings``, ``terms_fts``) is preserved — those rows are
    cross-paper. A force-delete can orphan a canonical term whose only
    producer was this paper; cleanup is deferred to a future "gardening"
    pass (phase 2).

    TODO(phase-2 gardening): reconcile orphaned rows across the four
    global-taxonomy tables once no paper references a given term_id /
    alias — until then orphans accumulate silently.
    """
    with transaction(conn):
        delete_paper_cascade(conn, paper_id=paper_id)


def _summary(conn: sqlite3.Connection, arxiv_id: str) -> dict:
    row = conn.execute(
        "SELECT id, paper_name, status, needs_review FROM papers WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return {
            "paper_name": None,
            "arxiv_id": arxiv_id,
            "status": None,
            "needs_review": False,
            "section_count": 0,
            "entity_count": 0,
            "figure_count": 0,
        }
    paper_id, paper_name, status, needs_review = row
    section_count = conn.execute(
        "SELECT COUNT(*) FROM sections WHERE paper_name = ?", (paper_name,)
    ).fetchone()[0]
    entity_count = conn.execute(
        """
        SELECT COUNT(DISTINCT ta.term_id)
          FROM term_aliases ta
          JOIN canonical_terms ct ON ct.id = ta.term_id
         WHERE ta.source_paper = ?
           AND ct.term_type = 'entity'
        """,
        (paper_name,),
    ).fetchone()[0]
    figure_count = conn.execute(
        "SELECT COUNT(*) FROM figures WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0]
    return {
        "paper_name": paper_name,
        "arxiv_id": arxiv_id,
        "status": status,
        "needs_review": bool(needs_review),
        "section_count": section_count,
        "entity_count": entity_count,
        "figure_count": figure_count,
    }


_PIPELINE: tuple[tuple[Stage, PaperStatus], ...] = (
    (Stage.FETCH, PaperStatus.FETCHED),
    (Stage.CONVERT, PaperStatus.CONVERTED),
    (Stage.CLASSIFY, PaperStatus.CLASSIFIED),
    (Stage.EXTRACT, PaperStatus.EXTRACTED),
    (Stage.INDEX, PaperStatus.INDEXED),
)


def _remaining_stages(current: PaperStatus | None) -> list[Stage]:
    """Simulate the pipeline one stage at a time through ``can_run_from``.

    Open-coding ``STATUS_ORDER[x] > STATUS_ORDER[y]`` is explicitly
    forbidden — see the section 14 Key Design Notes. ``can_run_from``'s
    terminal-state check makes FAILED_HTML short-circuit to an empty
    list, but callers handle FAILED_HTML before reaching here anyway.
    """
    remaining: list[Stage] = []
    simulated = current
    for stage, completed_status in _PIPELINE:
        if simulated is completed_status:
            continue
        if can_run_from(simulated, completed_status):
            remaining.append(stage)
            simulated = completed_status
    return remaining


def ingest(
    *,
    conn: sqlite3.Connection,
    arxiv_id: str,
    force: bool = False,
    domain: str | None = None,
) -> dict:
    """Run the ingest pipeline for one paper. Returns the JSON summary dict.

    The caller owns ``conn``'s lifecycle. Stage functions share this exact
    connection — shared transactions, shared sqlite-vec extension, and
    shared in-process model caches. NOTE: every stage manages its own
    ``BEGIN``/``COMMIT``; do NOT wrap the loop below in
    ``transaction(conn)`` — the inner ``BEGIN`` would raise "cannot start
    a transaction within a transaction".
    """
    row = _get_paper_row(conn, arxiv_id)

    if row is not None and force:
        _LOG.info(
            "force cascade: wiping paper id=%s arxiv_id=%s (taxonomy preserved)",
            row.id, arxiv_id,
        )
        _force_delete_paper(conn, paper_id=row.id)
        _LOG.warning(
            "cascade committed for arxiv_id=%s; beginning fresh ingest — "
            "if fetch fails the paper is gone until a successful rerun",
            arxiv_id,
        )
        row = None

    current: PaperStatus | None = None
    paper_name: str | None = None

    if row is not None:
        try:
            current = PaperStatus(row.status)
        except ValueError as exc:
            raise ValueError(
                f"papers.status={row.status!r} for arxiv_id={arxiv_id!r} "
                "is not a recognized PaperStatus"
            ) from exc
        paper_name = row.name

        if current is PaperStatus.FAILED_HTML:
            _LOG.info(
                "paper %s is FAILED_HTML; use --force to retry fetch", arxiv_id
            )
            return _summary(conn, arxiv_id)

        if current is PaperStatus.INDEXED:
            _LOG.info("paper %s already indexed, skipping", arxiv_id)
            return _summary(conn, arxiv_id)

    stages_to_run = _remaining_stages(current)
    _LOG.info(
        "ingest arxiv_id=%s current_status=%s stages_to_run=%s",
        arxiv_id, current, stages_to_run,
    )

    if Stage.FETCH in stages_to_run:
        fetch_stage(
            conn=conn,
            arxiv_id=arxiv_id,
            force=force,
            domain_override=domain,
        )
        # paper_name is assigned by fetch's slug generator; re-query.
        post_fetch = _get_paper_row(conn, arxiv_id)
        if post_fetch is None:
            raise RuntimeError(
                f"fetch() returned without persisting a papers row for {arxiv_id!r}"
            )
        paper_name = post_fetch.name
        if PaperStatus(post_fetch.status) is PaperStatus.FAILED_HTML:
            _LOG.warning("fetch produced FAILED_HTML for %s; halting pipeline", arxiv_id)
            return _summary(conn, arxiv_id)

    if paper_name is None:
        raise RuntimeError(
            f"internal invariant: no paper_name resolved for {arxiv_id!r} "
            "after resume lookup and without scheduling fetch"
        )

    if Stage.CONVERT in stages_to_run:
        convert_stage(conn=conn, paper_name=paper_name, force=force)
    if Stage.CLASSIFY in stages_to_run:
        classify_stage(
            conn=conn,
            paper_name=paper_name,
            force=force,
            domain_override=domain,
        )
    if Stage.EXTRACT in stages_to_run:
        extract_stage(conn=conn, paper_name=paper_name, force=force)
    if Stage.INDEX in stages_to_run:
        index_stage(conn=conn, paper_name=paper_name, force=force)

    return _summary(conn, arxiv_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Lodestone ingest orchestrator: fetch → convert → classify → extract → index.",
    )
    parser.add_argument("--url", required=True, help="arxiv URL or bare id (version preserved)")
    parser.add_argument("--force", action="store_true",
                        help="cascade-delete the paper (preserving global taxonomy) and re-ingest")
    parser.add_argument("--domain", default=None,
                        help="override the classifier's domain choice")
    parser.add_argument(
        "--db",
        default=os.environ.get("LODESTONE_DB", "lodestone.db"),
        help="path to the sqlite db (default: $LODESTONE_DB or ./lodestone.db)",
    )
    args = parser.parse_args(argv)

    # Parse the URL first — a malformed --url should fail before the
    # (cheap but non-trivial) HF cache sniff inside check_models().
    arxiv_id = parse_arxiv_id(args.url)

    # Pre-flight BEFORE opening the DB so a model-less machine never creates
    # an empty lodestone.db on first run.
    check_models()

    conn = get_conn(Path(args.db))
    try:
        init_db(conn)
        summary = ingest(
            conn=conn,
            arxiv_id=arxiv_id,
            force=args.force,
            domain=args.domain,
        )
    finally:
        conn.close()

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
