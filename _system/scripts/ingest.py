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
import re
import sqlite3
from pathlib import Path

from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.schemas.paper_metadata import PaperStatus, can_run_from
from _system.scripts.fetch_paper import fetch as fetch_stage
from _system.scripts.convert_paper import convert as convert_stage
from _system.scripts.classify_paper import classify as classify_stage
from _system.scripts.extract_entities import extract as extract_stage
from _system.scripts.index_paper import index_one as index_stage
from _system.scripts.validate_models import check_models
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.ingest")


# Matches `2301.12345` / `2301.12345v2` (new-form) and `hep-th/9901001v3`
# (old-form). The version suffix is captured as-is; arxiv_id identity is
# version-sensitive, so `v1` and `v2` are two different rows.
_ARXIV_NEW_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_ARXIV_OLD_RE = re.compile(r"^[a-z\-]+/\d{7}(v\d+)?$")

_URL_PREFIXES: tuple[str, ...] = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "https://arxiv.org/pdf/",
    "http://arxiv.org/pdf/",
)


def parse_arxiv_id(raw: str) -> str:
    """Extract the canonical arxiv id from a URL or bare id.

    Preserves the version suffix verbatim — `2301.12345v2` and
    `2301.12345v3` are different papers. Does **not** normalize.
    Raises ``ValueError`` on an input that doesn't look like either the
    new-form (`YYMM.NNNNN[vN]`) or old-form (`cat/NNNNNNN[vN]`) id.
    """
    if not raw or not raw.strip():
        raise ValueError("arxiv id / URL is empty")
    value = raw.strip()
    for prefix in _URL_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    value = value.removesuffix(".pdf")
    if _ARXIV_NEW_RE.match(value) or _ARXIV_OLD_RE.match(value):
        return value
    raise ValueError(
        f"unrecognized arxiv id / URL: {raw!r} "
        "(expected e.g. 2301.12345, 2301.12345v2, or hep-th/9901001)"
    )


def _get_paper_row(
    conn: sqlite3.Connection, arxiv_id: str
) -> tuple[int, str, str, bool] | None:
    """Return (paper_id, paper_name, status, needs_review) or None."""
    row = conn.execute(
        "SELECT id, paper_name, status, needs_review FROM papers WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None
    return (row[0], row[1], row[2], bool(row[3]))


def _force_delete_paper(
    conn: sqlite3.Connection, *, paper_id: int, paper_name: str
) -> None:
    """Cascade-delete one paper and every per-paper child row.

    Runs inside a single transaction. Order matters: FTS5 virtual tables
    ignore FK CASCADE, and the FK-backed children must be cleared before
    the papers row to satisfy PRAGMA foreign_keys=ON.

    Global taxonomy (``canonical_terms``, ``term_aliases``,
    ``term_embeddings``, ``terms_fts``) is **not** touched — those rows
    are cross-paper. A known limitation: a force-delete can orphan a
    canonical term whose only producer was this paper. Cleanup is
    deferred to a future "gardening" pass (phase 2).

    TODO(phase-2 gardening): reconcile orphaned rows across the four
    global-taxonomy tables listed above once no paper references a
    given term_id / alias — until then orphans accumulate silently.
    """
    with transaction(conn):
        conn.execute("DELETE FROM abstracts    WHERE paper_name = ?", (paper_name,))
        conn.execute("DELETE FROM sections     WHERE paper_name = ?", (paper_name,))
        conn.execute("DELETE FROM entities     WHERE paper_id   = ?", (paper_id,))
        conn.execute("DELETE FROM paper_topics WHERE paper_id   = ?", (paper_id,))
        conn.execute("DELETE FROM figures      WHERE paper_id   = ?", (paper_id,))
        conn.execute("DELETE FROM page_images  WHERE paper_id   = ?", (paper_id,))
        conn.execute("DELETE FROM papers       WHERE id         = ?", (paper_id,))


def _summary(conn: sqlite3.Connection, arxiv_id: str) -> dict:
    """Build the final JSON summary from current DB state."""
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
        "SELECT COUNT(*) FROM entities WHERE paper_id = ?", (paper_id,)
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


_PIPELINE: tuple[tuple[str, PaperStatus], ...] = (
    ("fetch", PaperStatus.FETCHED),
    ("convert", PaperStatus.CONVERTED),
    ("classify", PaperStatus.CLASSIFIED),
    ("extract", PaperStatus.EXTRACTED),
    ("index_one", PaperStatus.INDEXED),
)


def _remaining_stages(current: PaperStatus | None) -> list[str]:
    """Return the stages that still need to run given ``current`` status.

    Simulates the pipeline, advancing a local cursor one stage at a time
    and consulting :func:`can_run_from` at each step — this is the
    canonical way to chain stage guards (open-coding
    ``STATUS_ORDER[x] > STATUS_ORDER[y]`` is explicitly forbidden; see
    the section 14 Key Design Notes). FAILED_HTML short-circuits to an
    empty list via ``can_run_from``'s own terminal-state check, but
    callers handle FAILED_HTML before reaching here anyway.
    """
    remaining: list[str] = []
    simulated = current
    for name, stage in _PIPELINE:
        if simulated is stage:
            # Already at that status; don't rerun it.
            continue
        if can_run_from(simulated, stage):
            remaining.append(name)
            simulated = stage
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
        paper_id, paper_name, _status, _needs_review = row
        _LOG.info(
            "force cascade: wiping paper id=%s arxiv_id=%s (taxonomy preserved)",
            paper_id, arxiv_id,
        )
        _force_delete_paper(conn, paper_id=paper_id, paper_name=paper_name)
        _LOG.warning(
            "cascade committed for arxiv_id=%s; beginning fresh ingest — "
            "if fetch fails the paper is gone until a successful rerun",
            arxiv_id,
        )
        row = None  # fall through to full pipeline

    current: PaperStatus | None
    paper_name: str | None

    if row is not None:
        paper_id, paper_name, status_str, _needs_review = row
        try:
            current = PaperStatus(status_str)
        except ValueError as exc:
            raise ValueError(
                f"papers.status={status_str!r} for arxiv_id={arxiv_id!r} "
                "is not a recognized PaperStatus"
            ) from exc

        if current is PaperStatus.FAILED_HTML:
            _LOG.info(
                "paper %s is FAILED_HTML; use --force to retry fetch", arxiv_id
            )
            return _summary(conn, arxiv_id)

        if current is PaperStatus.INDEXED:
            _LOG.info("paper %s already indexed, skipping", arxiv_id)
            return _summary(conn, arxiv_id)
    else:
        current = None
        paper_name = None

    stages_to_run = _remaining_stages(current)
    _LOG.info(
        "ingest arxiv_id=%s current_status=%s stages_to_run=%s",
        arxiv_id, current.value if current else None, stages_to_run,
    )

    if "fetch" in stages_to_run:
        fetch_stage(
            conn=conn,
            arxiv_id=arxiv_id,
            force=force,
            domain_override=domain,
        )
        # paper_name is assigned by fetch (slug generator). Re-query.
        post_fetch = _get_paper_row(conn, arxiv_id)
        if post_fetch is None:
            raise RuntimeError(
                f"fetch() returned without persisting a papers row for {arxiv_id!r}"
            )
        _, paper_name, status_str, _ = post_fetch
        if PaperStatus(status_str) is PaperStatus.FAILED_HTML:
            _LOG.warning("fetch produced FAILED_HTML for %s; halting pipeline", arxiv_id)
            return _summary(conn, arxiv_id)
    else:
        # paper_name was populated from the resume lookup above.
        assert paper_name is not None  # invariant: non-fetch resume saw a row

    if "convert" in stages_to_run:
        convert_stage(conn=conn, paper_name=paper_name, force=force)
    if "classify" in stages_to_run:
        classify_stage(
            conn=conn,
            paper_name=paper_name,
            force=force,
            domain_override=domain,
        )
    if "extract" in stages_to_run:
        extract_stage(conn=conn, paper_name=paper_name, force=force)
    if "index_one" in stages_to_run:
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
