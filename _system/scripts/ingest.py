"""Top-level Lodestone ingest CLI.

Two entry shapes:

- ``--url <arxiv>`` — paper-first ingest. Runs fetch → convert →
  classify → extract → index. If the paper turns out to ship a code
  repo (URL discovered during fetch), the repo is registered as a
  first-class entity tied to the paper, and a follow-up FETCH_REPO
  stage clones it.
- ``--repo <github_url>`` — standalone-repo ingest. Runs resolve_repo
  → fetch_repo → classify_repo. Repos with no usable README terminate
  at ORPHANED (still searchable by name/path/file content).

Stage function contract: every stage accepts its shared
``sqlite3.Connection`` as a keyword-only argument named ``conn`` and
validates its own ``can_run_from`` preconditions. No subprocesses — this
module imports and calls directly so sqlite-vec extension load and
model caches survive across stages within one run.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

from _system.db.cascade import delete_paper_cascade, delete_repo_cascade
from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.schemas.paper_metadata import PaperStatus, can_run_from as paper_can_run_from
from _system.schemas.repo_metadata import RepoStatus, can_run_from as repo_can_run_from
from _system.scripts.classify_paper import classify as classify_paper_stage
from _system.scripts.classify_repo import classify as classify_repo_stage
from _system.scripts.convert_paper import convert as convert_stage
from _system.scripts.extract_entities import extract as extract_stage
from _system.scripts.fetch_paper import fetch as fetch_stage
from _system.scripts.fetch_repo import fetch_repo as fetch_repo_stage
from _system.scripts.index_paper import index_one as index_stage
from _system.scripts.resolve_repo import resolve as resolve_repo_stage
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
    RESOLVE_REPO = "resolve_repo"
    FETCH_REPO = "fetch_repo"
    CLASSIFY_REPO = "classify_repo"


class _PaperRow(NamedTuple):
    id: int
    name: str
    status: str
    needs_review: bool


class _RepoRow(NamedTuple):
    id: int
    repo_slug: str
    url: str
    status: str
    paper_id: int | None


def _get_paper_row(conn: sqlite3.Connection, arxiv_id: str) -> _PaperRow | None:
    row = conn.execute(
        "SELECT id, paper_name, status, needs_review FROM papers WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return None
    return _PaperRow(id=row[0], name=row[1], status=row[2], needs_review=bool(row[3]))


def _get_repo_row(conn: sqlite3.Connection, *, url: str) -> _RepoRow | None:
    row = conn.execute(
        "SELECT id, repo_slug, url, status, paper_id FROM repos WHERE url = ?",
        (url,),
    ).fetchone()
    if row is None:
        return None
    return _RepoRow(id=row[0], repo_slug=row[1], url=row[2], status=row[3], paper_id=row[4])


def _get_repo_for_paper(conn: sqlite3.Connection, paper_id: int) -> _RepoRow | None:
    row = conn.execute(
        "SELECT id, repo_slug, url, status, paper_id FROM repos WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    if row is None:
        return None
    return _RepoRow(id=row[0], repo_slug=row[1], url=row[2], status=row[3], paper_id=row[4])


def _force_delete_paper(conn: sqlite3.Connection, *, paper_id: int) -> None:
    """Cascade-delete one paper inside its own transaction.

    Per-paper children are wiped (including any linked repos and their
    code_files / readmes_fts rows); entity canonicals stay, topic +
    collection canonicals get GC'd if their last binding was this paper.
    """
    with transaction(conn):
        delete_paper_cascade(conn, paper_id=paper_id)


def _force_delete_repo(conn: sqlite3.Connection, *, repo_id: int) -> None:
    with transaction(conn):
        delete_repo_cascade(conn, repo_id=repo_id)


def _summary_paper(conn: sqlite3.Connection, arxiv_id: str) -> dict:
    row = conn.execute(
        "SELECT id, paper_name, status, needs_review, entity_count "
        "  FROM papers WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    if row is None:
        return {
            "kind": "paper",
            "paper_name": None,
            "arxiv_id": arxiv_id,
            "status": None,
            "needs_review": False,
            "section_count": 0,
            "entity_count": 0,
            "figure_count": 0,
            "repo": None,
        }
    paper_id, paper_name, status, needs_review, entity_count = row
    section_count = conn.execute(
        "SELECT COUNT(*) FROM sections WHERE paper_name = ?", (paper_name,)
    ).fetchone()[0]
    figure_count = conn.execute(
        "SELECT COUNT(*) FROM figures WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0]
    repo_row = conn.execute(
        "SELECT repo_slug, url, status FROM repos WHERE paper_id = ?",
        (paper_id,),
    ).fetchone()
    repo_envelope: dict | None = None
    if repo_row is not None:
        repo_envelope = {
            "repo_slug": repo_row[0],
            "url": repo_row[1],
            "status": repo_row[2],
        }
    return {
        "kind": "paper",
        "paper_name": paper_name,
        "arxiv_id": arxiv_id,
        "status": status,
        "needs_review": bool(needs_review),
        "section_count": section_count,
        "entity_count": entity_count or 0,
        "figure_count": figure_count,
        "repo": repo_envelope,
    }


def _summary_repo(conn: sqlite3.Connection, repo_url: str) -> dict:
    row = conn.execute(
        "SELECT repo_slug, url, status, domain, collection, "
        "       file_count, has_readme, needs_review "
        "  FROM repos WHERE url = ?",
        (repo_url,),
    ).fetchone()
    if row is None:
        return {
            "kind": "repo",
            "repo_slug": None,
            "url": repo_url,
            "status": None,
            "domain": None,
            "collection": None,
            "file_count": 0,
            "has_readme": False,
            "needs_review": False,
        }
    return {
        "kind": "repo",
        "repo_slug": row[0],
        "url": row[1],
        "status": row[2],
        "domain": row[3],
        "collection": row[4],
        "file_count": int(row[5] or 0),
        "has_readme": bool(row[6]),
        "needs_review": bool(row[7]),
    }


_PAPER_PIPELINE: tuple[tuple[Stage, PaperStatus], ...] = (
    (Stage.FETCH, PaperStatus.FETCHED),
    (Stage.CONVERT, PaperStatus.CONVERTED),
    (Stage.CLASSIFY, PaperStatus.CLASSIFIED),
    (Stage.EXTRACT, PaperStatus.EXTRACTED),
    (Stage.INDEX, PaperStatus.INDEXED),
)


_PAPER_TERMINAL: frozenset[PaperStatus] = frozenset({
    PaperStatus.INDEXED,
    PaperStatus.FAILED_HTML,
})


def _remaining_paper_stages(current: PaperStatus | None) -> list[Stage]:
    remaining: list[Stage] = []
    simulated = current
    for stage, completed_status in _PAPER_PIPELINE:
        if simulated is completed_status:
            continue
        if paper_can_run_from(simulated, completed_status):
            remaining.append(stage)
            simulated = completed_status
    return remaining


_REPO_PIPELINE: tuple[tuple[Stage, RepoStatus], ...] = (
    (Stage.RESOLVE_REPO, RepoStatus.RESOLVED),
    (Stage.FETCH_REPO, RepoStatus.REPO_FETCHED),
    (Stage.CLASSIFY_REPO, RepoStatus.CLASSIFIED),
)


_REPO_TERMINAL: frozenset[RepoStatus] = frozenset({
    RepoStatus.CLASSIFIED,
    RepoStatus.ORPHANED,
    RepoStatus.FAILED_RESOLVE,
    RepoStatus.FAILED_REPO,
})


def _remaining_repo_stages(current: RepoStatus | None) -> list[Stage]:
    remaining: list[Stage] = []
    simulated = current
    for stage, completed_status in _REPO_PIPELINE:
        if simulated is completed_status:
            continue
        if repo_can_run_from(simulated, completed_status):
            remaining.append(stage)
            simulated = completed_status
    return remaining


# ---------------------------------------------------------------------------
# Paper-first orchestrator
# ---------------------------------------------------------------------------


def ingest(
    *,
    conn: sqlite3.Connection,
    arxiv_id: str,
    force: bool = False,
    domain: str | None = None,
) -> dict:
    """Run the paper-first ingest pipeline. Returns the summary dict."""
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
            return _summary_paper(conn, arxiv_id)

    stages_to_run = _remaining_paper_stages(current)
    _LOG.info(
        "ingest arxiv_id=%s current_status=%s stages_to_run=%s",
        arxiv_id, current, stages_to_run,
    )

    discovered_repo_url: str | None = None

    if Stage.FETCH in stages_to_run:
        pm = fetch_stage(
            conn=conn,
            arxiv_id=arxiv_id,
            force=force,
            domain_override=domain,
        )
        post_fetch = _get_paper_row(conn, arxiv_id)
        if post_fetch is None:
            raise RuntimeError(
                f"fetch() returned without persisting a papers row for {arxiv_id!r}"
            )
        paper_name = post_fetch.name
        if PaperStatus(post_fetch.status) is PaperStatus.FAILED_HTML:
            _LOG.warning("fetch produced FAILED_HTML for %s; halting pipeline", arxiv_id)
            return _summary_paper(conn, arxiv_id)
        # Discovered repo URL travels on the in-memory PaperMetadata —
        # it is intentionally not persisted on the papers row anymore.
        discovered_repo_url = pm.code_repo

        # Register the repo as a first-class entity right at FETCH so the
        # URL is durable across resumes. domain/collection are filled in
        # after CLASSIFY (they're allowed to be NULL while status=RESOLVED).
        if discovered_repo_url:
            paper_id = post_fetch.id
            try:
                resolve_repo_stage(
                    conn=conn,
                    repo_url=discovered_repo_url,
                    paper_id=paper_id,
                )
            except Exception as exc:
                # Repo discovery failure must not abort paper ingest.
                # Log loud and proceed without a linked repo row.
                _LOG.warning(
                    "resolve_repo failed for %s url=%s: %s; paper ingest continues",
                    arxiv_id, discovered_repo_url, exc,
                )

    if paper_name is None:
        raise RuntimeError(
            f"internal invariant: no paper_name resolved for {arxiv_id!r} "
            "after resume lookup and without scheduling fetch"
        )

    if Stage.CONVERT in stages_to_run:
        convert_stage(conn=conn, paper_name=paper_name, force=force)
    if Stage.CLASSIFY in stages_to_run:
        classify_paper_stage(
            conn=conn,
            paper_name=paper_name,
            force=force,
            domain_override=domain,
        )
    if Stage.EXTRACT in stages_to_run:
        extract_stage(conn=conn, paper_name=paper_name, force=force)
    if Stage.INDEX in stages_to_run:
        index_stage(conn=conn, paper_name=paper_name, force=force)

    # Paper-linked repo follow-up (only if a repos row exists for this paper).
    paper_id_row = conn.execute(
        "SELECT id, domain, collection FROM papers WHERE arxiv_id = ?", (arxiv_id,),
    ).fetchone()
    if paper_id_row is not None:
        paper_id, paper_domain, paper_collection = paper_id_row
        repo_row = _get_repo_for_paper(conn, paper_id)
        if repo_row is not None:
            _propagate_taxonomy_to_repo(
                conn,
                repo_id=repo_row.id,
                domain=paper_domain,
                collection=paper_collection,
            )
            _run_paper_linked_repo_fetch(conn, repo_slug=repo_row.repo_slug)

    return _summary_paper(conn, arxiv_id)


def _propagate_taxonomy_to_repo(
    conn: sqlite3.Connection,
    *,
    repo_id: int,
    domain: str | None,
    collection: str | None,
) -> None:
    """Inherit ``papers.{domain,collection}`` onto the linked repo.

    Idempotent — re-running on a repo that already has the same values
    is a no-op COALESCE update. The schema-level trigger only enforces
    domain+collection on CLASSIFIED rows, so paper-linked repos that
    sit at REPO_FETCHED with the inherited values written are fine.
    """
    if domain is None or collection is None:
        return
    with transaction(conn):
        conn.execute(
            "UPDATE repos SET domain = ?, collection = ? WHERE id = ?",
            (domain, collection, repo_id),
        )


def _run_paper_linked_repo_fetch(conn: sqlite3.Connection, *, repo_slug: str) -> None:
    """Run FETCH_REPO on a paper-linked repo if it hasn't run yet."""
    row = conn.execute(
        "SELECT status FROM repos WHERE repo_slug = ?", (repo_slug,)
    ).fetchone()
    if row is None:
        return
    try:
        current = RepoStatus(row[0])
    except ValueError:
        _LOG.warning("repo %s has unknown status=%r; skipping fetch_repo", repo_slug, row[0])
        return
    if current in _REPO_TERMINAL:
        return
    if not repo_can_run_from(current, RepoStatus.REPO_FETCHED):
        return
    fetch_repo_stage(conn=conn, repo_slug=repo_slug)


# ---------------------------------------------------------------------------
# Standalone-repo orchestrator
# ---------------------------------------------------------------------------


def ingest_repo_only(
    *,
    conn: sqlite3.Connection,
    repo_url: str,
    force: bool = False,
    domain: str | None = None,
) -> dict:
    """Run the standalone-repo ingest pipeline. Returns the summary dict."""
    existing = _get_repo_row(conn, url=repo_url)

    if existing is not None and force:
        _LOG.info(
            "force cascade: wiping repo id=%s url=%s (taxonomy preserved)",
            existing.id, repo_url,
        )
        _force_delete_repo(conn, repo_id=existing.id)
        existing = None

    current: RepoStatus | None = None
    repo_slug: str | None = None

    if existing is not None:
        try:
            current = RepoStatus(existing.status)
        except ValueError as exc:
            raise ValueError(
                f"repos.status={existing.status!r} for url={repo_url!r} "
                "is not a recognized RepoStatus"
            ) from exc
        repo_slug = existing.repo_slug
        if existing.paper_id is not None:
            raise ValueError(
                f"url={repo_url!r} already exists as a paper-linked repo "
                f"(paper_id={existing.paper_id}); refuse to re-route through "
                "standalone path"
            )
        if current in _REPO_TERMINAL:
            _LOG.info(
                "repo %s already terminal (status=%s); use --force to re-ingest",
                repo_url, current.value,
            )
            return _summary_repo(conn, repo_url)

    stages_to_run = _remaining_repo_stages(current)
    _LOG.info(
        "ingest_repo_only url=%s current_status=%s stages_to_run=%s",
        repo_url, current, stages_to_run,
    )

    if Stage.RESOLVE_REPO in stages_to_run:
        result = resolve_repo_stage(conn=conn, repo_url=repo_url)
        repo_slug = result.repo_slug

    if repo_slug is None:
        raise RuntimeError(
            f"internal invariant: no repo_slug resolved for {repo_url!r}"
        )

    if Stage.FETCH_REPO in stages_to_run:
        fetch_repo_stage(conn=conn, repo_slug=repo_slug)

    # CLASSIFY_REPO runs only after a successful clone. If the prior
    # stage marked FAILED_REPO the next can_run_from check rejects.
    if Stage.CLASSIFY_REPO in stages_to_run:
        post_fetch = conn.execute(
            "SELECT status FROM repos WHERE repo_slug = ?", (repo_slug,)
        ).fetchone()
        if post_fetch is not None:
            try:
                fetched_status = RepoStatus(post_fetch[0])
            except ValueError:
                fetched_status = None
            if fetched_status is RepoStatus.REPO_FETCHED:
                classify_repo_stage(
                    conn=conn,
                    repo_slug=repo_slug,
                    force=force,
                    domain_override=domain,
                )

    return _summary_repo(conn, repo_url)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Lodestone ingest orchestrator: paper or standalone repo.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--url", default=None, help="arxiv URL or bare id (version preserved)")
    target.add_argument("--repo", default=None, help="github/gitlab/bitbucket URL for standalone repo ingest")
    parser.add_argument("--force", action="store_true",
                        help="cascade-delete the target (preserving global taxonomy) and re-ingest")
    parser.add_argument("--domain", default=None,
                        help="override the classifier's domain choice")
    parser.add_argument(
        "--db",
        default=os.environ.get("LODESTONE_DB", "lodestone.db"),
        help="path to the sqlite db (default: $LODESTONE_DB or ./lodestone.db)",
    )
    args = parser.parse_args(argv)

    if args.url:
        arxiv_id = parse_arxiv_id(args.url)
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
    else:
        check_models()
        conn = get_conn(Path(args.db))
        try:
            init_db(conn)
            summary = ingest_repo_only(
                conn=conn,
                repo_url=args.repo,
                force=args.force,
                domain=args.domain,
            )
        finally:
            conn.close()

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
