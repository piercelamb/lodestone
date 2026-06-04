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
- ``--acl <id-or-url>`` — ACL Anthology PDF ingest. MODS metadata +
  PDF rendering (no HTML/LaTeX fulltext on Anthology). No repo
  discovery — ACL papers don't carry an arxiv-id keyed PwC lookup.

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
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import pymupdf

# Optional callback fired between stages for progress reporting.
# Args: (message, done, total).
ProgressFn = Callable[[str, int, int], None]

from _system.db.cascade import (
    delete_paper_cascade,
    delete_post_cascade,
    delete_repo_cascade,
)
from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.schemas.paper_metadata import PaperStatus, can_run_from as paper_can_run_from
from _system.schemas.post_metadata import PostStatus, can_run_from as post_can_run_from
from _system.schemas.repo_metadata import RepoStatus, can_run_from as repo_can_run_from
from _system.scripts.classify_paper import classify as classify_paper_stage
from _system.scripts.classify_repo import classify as classify_repo_stage
from _system.scripts.convert_paper import convert as convert_stage
from _system.scripts.convert_post import convert as convert_post_stage
from _system.scripts.extract_entities import extract as extract_stage
from _system.scripts.fetch_acl import fetch as fetch_acl_stage
from _system.scripts.fetch_paper import fetch as fetch_stage
from _system.scripts.fetch_post import fetch as fetch_post_stage
from _system.scripts.fetch_repo import fetch_repo as fetch_repo_stage
from _system.scripts.index_paper import index_one as index_stage
from _system.scripts.load_pdf import (
    ChapterSpec,
    LocalPdfNoUsableOutline,
    discover_chapters,
    load_pdf_chapter,
    read_pdf_metadata,
    render_chapter_markdown,
    render_whole_book_markdown,
    synthetic_arxiv_id,
)
from _system.scripts.resolve_repo import resolve as resolve_repo_stage
from _system.scripts.validate_models import check_models
from _system.utils.acl_urls import parse_acl_id
from _system.utils.arxiv_urls import parse_arxiv_id
from _system.utils.logging import get_logger
from _system.utils.slug import (
    _SLUG_RE,
    existing_slugs,
    generate_book_slug,
    generate_chapter_slug,
)

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
    FETCH_POST = "fetch_post"
    CONVERT_POST = "convert_post"
    CLASSIFY_POST = "classify_post"
    EXTRACT_POST = "extract_post"
    INDEX_POST = "index_post"


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


def _collection_rows(
    conn: sqlite3.Connection, *, target_kind: str, target_id: int
) -> list[dict]:
    """List polymorphic collection rows for a target, primary first."""
    rows = conn.execute(
        "SELECT collection, is_primary FROM collections "
        " WHERE target_kind = ? AND target_id = ? "
        " ORDER BY is_primary DESC, collection",
        (target_kind, target_id),
    ).fetchall()
    return [
        {"collection": r[0], "is_primary": bool(r[1])} for r in rows
    ]


def _topic_count(
    conn: sqlite3.Connection, *, target_kind: str, target_id: int
) -> int:
    return int(conn.execute(
        "SELECT COUNT(*) FROM topics "
        " WHERE target_kind = ? AND target_id = ?",
        (target_kind, target_id),
    ).fetchone()[0])


def _summary_paper(conn: sqlite3.Connection, arxiv_id: str) -> dict:
    row = conn.execute(
        "SELECT id, paper_name, status, needs_review, entity_count, "
        "       domain, collection "
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
            "domain": None,
            "collection": None,
            "collections": [],
            "section_count": 0,
            "entity_count": 0,
            "figure_count": 0,
            "topic_count": 0,
            "repo": None,
        }
    (
        paper_id,
        paper_name,
        status,
        needs_review,
        entity_count,
        domain,
        collection,
    ) = row
    section_count = conn.execute(
        "SELECT COUNT(*) FROM sections WHERE paper_name = ?", (paper_name,)
    ).fetchone()[0]
    figure_count = conn.execute(
        "SELECT COUNT(*) FROM figures WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0]
    collections = _collection_rows(conn, target_kind="paper", target_id=paper_id)
    topic_count = _topic_count(conn, target_kind="paper", target_id=paper_id)
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
        "domain": domain,
        "collection": collection,
        "collections": collections,
        "section_count": section_count,
        "entity_count": entity_count or 0,
        "figure_count": figure_count,
        "topic_count": topic_count,
        "repo": repo_envelope,
    }


def _summary_repo(conn: sqlite3.Connection, repo_url: str) -> dict:
    row = conn.execute(
        "SELECT id, repo_slug, url, status, domain, collection, "
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
            "collections": [],
            "file_count": 0,
            "has_readme": False,
            "needs_review": False,
            "topic_count": 0,
        }
    repo_id = int(row[0])
    collections = _collection_rows(conn, target_kind="repo", target_id=repo_id)
    topic_count = _topic_count(conn, target_kind="repo", target_id=repo_id)
    return {
        "kind": "repo",
        "repo_slug": row[1],
        "url": row[2],
        "status": row[3],
        "domain": row[4],
        "collection": row[5],
        "collections": collections,
        "file_count": int(row[6] or 0),
        "has_readme": bool(row[7]),
        "needs_review": bool(row[8]),
        "topic_count": topic_count,
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
    progress: ProgressFn | None = None,
) -> dict:
    """Run the paper-first ingest pipeline. Returns the summary dict.

    ``progress`` fires ``(message, done, total)`` between stages so callers
    (e.g. the MCP server) can render staged progress. CLI callers leave it
    None and stages run silent.
    """
    def _tick(msg: str, done: int, total: int) -> None:
        if progress is not None:
            progress(msg, done, total)

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
            _tick("already complete", 0, 0)
            return _summary_paper(conn, arxiv_id)

    stages_to_run = _remaining_paper_stages(current)
    _LOG.info(
        "ingest arxiv_id=%s current_status=%s stages_to_run=%s",
        arxiv_id, current, stages_to_run,
    )

    total = len(stages_to_run)
    done = 0

    if total == 0:
        _tick("already complete", 0, 0)
        return _summary_paper(conn, arxiv_id)

    discovered_repo_url: str | None = None

    if Stage.FETCH in stages_to_run:
        _tick(f"starting {Stage.FETCH.value}", done, total)
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
            done += 1
            _tick("complete", done, total)
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
        done += 1

    if paper_name is None:
        raise RuntimeError(
            f"internal invariant: no paper_name resolved for {arxiv_id!r} "
            "after resume lookup and without scheduling fetch"
        )

    if Stage.CONVERT in stages_to_run:
        _tick(f"starting {Stage.CONVERT.value}", done, total)
        convert_stage(conn=conn, paper_name=paper_name, force=force)
        done += 1
    if Stage.CLASSIFY in stages_to_run:
        _tick(f"starting {Stage.CLASSIFY.value}", done, total)
        classify_paper_stage(
            conn=conn,
            paper_name=paper_name,
            force=force,
            domain_override=domain,
        )
        done += 1
    if Stage.EXTRACT in stages_to_run:
        _tick(f"starting {Stage.EXTRACT.value}", done, total)
        extract_stage(conn=conn, paper_name=paper_name, force=force)
        done += 1
    if Stage.INDEX in stages_to_run:
        _tick(f"starting {Stage.INDEX.value}", done, total)
        index_stage(conn=conn, paper_name=paper_name, force=force)
        done += 1

    _tick("complete", done, total)

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
                paper_id=paper_id,
                repo_id=repo_row.id,
                domain=paper_domain,
                collection=paper_collection,
            )
            _run_paper_linked_repo_fetch(conn, repo_slug=repo_row.repo_slug)

    return _summary_paper(conn, arxiv_id)


def ingest_acl(
    *,
    conn: sqlite3.Connection,
    acl_id: str,
    force: bool = False,
    domain: str | None = None,
    progress: ProgressFn | None = None,
) -> dict:
    """Run the ACL Anthology ingest pipeline. Returns the summary dict.

    Near-clone of :func:`ingest` minus the repo-discovery block: ACL
    papers don't carry a PaperswithCode arxiv-id keyed lookup, and the
    MODS metadata exposes no arxiv-comment analog. The paper itself
    rides on a synthetic ``arxiv_id = "acl:<id>"`` so downstream stages
    (convert / classify / extract / index) treat it identically to an
    arxiv paper that hit the PDF fallback path.
    """
    def _tick(msg: str, done: int, total: int) -> None:
        if progress is not None:
            progress(msg, done, total)

    arxiv_id = f"acl:{acl_id}"

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
            _tick("already complete", 0, 0)
            return _summary_paper(conn, arxiv_id)

    stages_to_run = _remaining_paper_stages(current)
    _LOG.info(
        "ingest_acl arxiv_id=%s current_status=%s stages_to_run=%s",
        arxiv_id, current, stages_to_run,
    )

    total = len(stages_to_run)
    done = 0

    if total == 0:
        _tick("already complete", 0, 0)
        return _summary_paper(conn, arxiv_id)

    if Stage.FETCH in stages_to_run:
        _tick(f"starting {Stage.FETCH.value}", done, total)
        fetch_acl_stage(
            conn=conn,
            acl_id=acl_id,
            force=force,
            domain_override=domain,
        )
        post_fetch = _get_paper_row(conn, arxiv_id)
        if post_fetch is None:
            raise RuntimeError(
                f"fetch_acl() returned without persisting a papers row for {arxiv_id!r}"
            )
        paper_name = post_fetch.name
        if PaperStatus(post_fetch.status) is PaperStatus.FAILED_HTML:
            _LOG.warning("fetch_acl produced FAILED_HTML for %s; halting pipeline", arxiv_id)
            done += 1
            _tick("complete", done, total)
            return _summary_paper(conn, arxiv_id)
        done += 1

    if paper_name is None:
        raise RuntimeError(
            f"internal invariant: no paper_name resolved for {arxiv_id!r} "
            "after resume lookup and without scheduling fetch"
        )

    if Stage.CONVERT in stages_to_run:
        _tick(f"starting {Stage.CONVERT.value}", done, total)
        convert_stage(conn=conn, paper_name=paper_name, force=force)
        done += 1
    if Stage.CLASSIFY in stages_to_run:
        _tick(f"starting {Stage.CLASSIFY.value}", done, total)
        classify_paper_stage(
            conn=conn,
            paper_name=paper_name,
            force=force,
            domain_override=domain,
        )
        done += 1
    if Stage.EXTRACT in stages_to_run:
        _tick(f"starting {Stage.EXTRACT.value}", done, total)
        extract_stage(conn=conn, paper_name=paper_name, force=force)
        done += 1
    if Stage.INDEX in stages_to_run:
        _tick(f"starting {Stage.INDEX.value}", done, total)
        index_stage(conn=conn, paper_name=paper_name, force=force)
        done += 1

    _tick("complete", done, total)

    return _summary_paper(conn, arxiv_id)


def _propagate_taxonomy_to_repo(
    conn: sqlite3.Connection,
    *,
    paper_id: int,
    repo_id: int,
    domain: str | None,
    collection: str | None,
) -> None:
    """Inherit the parent paper's taxonomy onto the linked repo.

    Writes the denormalized scalar ``repos.{domain, collection}`` (mirrors
    the primary collection) AND mirrors every polymorphic ``collections``
    membership the paper carries — so a paper-linked repo lives in the
    same set of collections as the paper, including secondaries.
    Idempotent: the polymorphic write is DELETE-then-INSERT.

    The schema-level trigger only enforces domain+collection on
    CLASSIFIED rows, so paper-linked repos that sit at REPO_FETCHED with
    the inherited values written are fine.
    """
    if domain is None or collection is None:
        return
    with transaction(conn):
        # Scalar primary pointer must be written first so the polymorphic
        # collections trigger reads the new repos.domain when the
        # mirrored rows land.
        conn.execute(
            "UPDATE repos SET domain = ?, collection = ? WHERE id = ?",
            (domain, collection, repo_id),
        )
        conn.execute(
            "DELETE FROM collections WHERE target_kind = 'repo' AND target_id = ?",
            (repo_id,),
        )
        conn.execute(
            """
            INSERT INTO collections (target_kind, target_id, domain, collection, is_primary)
            SELECT 'repo', ?, domain, collection, is_primary
              FROM collections
             WHERE target_kind = 'paper' AND target_id = ?
            """,
            (repo_id, paper_id),
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
    progress: ProgressFn | None = None,
) -> dict:
    """Run the standalone-repo ingest pipeline. Returns the summary dict.

    ``progress`` fires ``(message, done, total)`` between stages — see
    :func:`ingest` for the contract.
    """
    def _tick(msg: str, done: int, total: int) -> None:
        if progress is not None:
            progress(msg, done, total)

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
            _tick("already complete", 0, 0)
            return _summary_repo(conn, repo_url)

    stages_to_run = _remaining_repo_stages(current)
    _LOG.info(
        "ingest_repo_only url=%s current_status=%s stages_to_run=%s",
        repo_url, current, stages_to_run,
    )

    total = len(stages_to_run)
    done = 0

    if total == 0:
        _tick("already complete", 0, 0)
        return _summary_repo(conn, repo_url)

    if Stage.RESOLVE_REPO in stages_to_run:
        _tick(f"starting {Stage.RESOLVE_REPO.value}", done, total)
        result = resolve_repo_stage(conn=conn, repo_url=repo_url)
        repo_slug = result.repo_slug
        done += 1

    if repo_slug is None:
        raise RuntimeError(
            f"internal invariant: no repo_slug resolved for {repo_url!r}"
        )

    if Stage.FETCH_REPO in stages_to_run:
        _tick(f"starting {Stage.FETCH_REPO.value}", done, total)
        fetch_repo_stage(conn=conn, repo_slug=repo_slug)
        done += 1

    # CLASSIFY_REPO runs only after a successful clone. If the prior
    # stage marked FAILED_REPO the next can_run_from check rejects.
    if Stage.CLASSIFY_REPO in stages_to_run:
        _tick(f"starting {Stage.CLASSIFY_REPO.value}", done, total)
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
        done += 1

    _tick("complete", done, total)
    return _summary_repo(conn, repo_url)


# ---------------------------------------------------------------------------
# Post-first orchestrator
# ---------------------------------------------------------------------------


_POST_PIPELINE: tuple[tuple[Stage, PostStatus], ...] = (
    (Stage.FETCH_POST, PostStatus.FETCHED),
    (Stage.CONVERT_POST, PostStatus.CONVERTED),
    (Stage.CLASSIFY_POST, PostStatus.CLASSIFIED),
    (Stage.EXTRACT_POST, PostStatus.EXTRACTED),
    (Stage.INDEX_POST, PostStatus.INDEXED),
)


_POST_TERMINAL: frozenset[PostStatus] = frozenset({
    PostStatus.INDEXED,
    PostStatus.FAILED_FETCH,
    PostStatus.FAILED_PARSE,
})


def _remaining_post_stages(current: PostStatus | None) -> list[Stage]:
    remaining: list[Stage] = []
    simulated = current
    for stage, completed_status in _POST_PIPELINE:
        if simulated is completed_status:
            continue
        if post_can_run_from(simulated, completed_status):
            remaining.append(stage)
            simulated = completed_status
    return remaining


class _PostRow(NamedTuple):
    id: int
    name: str
    status: str


def _get_post_row(conn: sqlite3.Connection, *, url: str) -> _PostRow | None:
    """Locate an existing posts row by either source_url or canonical_url.

    On force-cascade we want to find the row even if the user passed the
    pre-canonical source URL (e.g. a syndicated mirror) — the dedup at
    fetch time is canonical-keyed, so we need the same lookup shape here.
    """
    row = conn.execute(
        """
        SELECT id, post_name, status FROM posts
         WHERE source_url = ? OR canonical_url = ?
         LIMIT 1
        """,
        (url, url),
    ).fetchone()
    if row is None:
        return None
    return _PostRow(id=int(row[0]), name=row[1], status=row[2])


def _force_delete_post(conn: sqlite3.Connection, *, post_id: int) -> None:
    with transaction(conn):
        delete_post_cascade(conn, post_id=post_id)


def _summary_post(conn: sqlite3.Connection, url: str) -> dict:
    row = conn.execute(
        """
        SELECT id, post_name, canonical_url, status, needs_review,
               domain, collection, section_count, entity_count, title
          FROM posts
         WHERE source_url = ? OR canonical_url = ?
         LIMIT 1
        """,
        (url, url),
    ).fetchone()
    if row is None:
        return {
            "kind": "post",
            "post_name": None,
            "url": url,
            "canonical_url": None,
            "status": None,
            "needs_review": False,
            "domain": None,
            "collection": None,
            "collections": [],
            "section_count": 0,
            "entity_count": 0,
            "topic_count": 0,
            "title": None,
        }
    post_id = int(row[0])
    collections = _collection_rows(conn, target_kind="post", target_id=post_id)
    topic_count = _topic_count(conn, target_kind="post", target_id=post_id)
    return {
        "kind": "post",
        "post_name": row[1],
        "url": url,
        "canonical_url": row[2],
        "status": row[3],
        "needs_review": bool(row[4]),
        "domain": row[5],
        "collection": row[6],
        "collections": collections,
        "section_count": int(row[7] or 0),
        "entity_count": int(row[8] or 0),
        "topic_count": topic_count,
        "title": row[9],
    }


def ingest_post(
    *,
    conn: sqlite3.Connection,
    url: str,
    force: bool = False,
    domain: str | None = None,
    progress: ProgressFn | None = None,
) -> dict:
    """Run the blog-post ingest pipeline. Returns the summary dict.

    Mirrors :func:`ingest`, but does NOT auto-ingest any github links
    discovered in the post body. Posts frequently mention unrelated repos
    in passing, so we don't treat a stray github URL as a "linked repo"
    here — that signal is only reliable for arxiv papers. If the user
    wants a repo ingested, they can invoke ``ingest_repo`` directly.
    """
    def _tick(msg: str, done: int, total: int) -> None:
        if progress is not None:
            progress(msg, done, total)

    row = _get_post_row(conn, url=url)
    if row is not None and force:
        _LOG.info(
            "force cascade: wiping post id=%s url=%s (taxonomy preserved)",
            row.id, url,
        )
        _force_delete_post(conn, post_id=row.id)
        _LOG.warning(
            "cascade committed for url=%s; beginning fresh ingest", url,
        )
        row = None

    current: PostStatus | None = None
    post_name: str | None = None

    if row is not None:
        try:
            current = PostStatus(row.status)
        except ValueError as exc:
            raise ValueError(
                f"posts.status={row.status!r} for url={url!r} "
                "is not a recognized PostStatus"
            ) from exc
        post_name = row.name

        if current in (PostStatus.FAILED_FETCH, PostStatus.FAILED_PARSE):
            _LOG.info(
                "post %s is %s; use --force to retry", url, current.value,
            )
            _tick("already complete", 0, 0)
            return _summary_post(conn, url)

    stages_to_run = _remaining_post_stages(current)
    _LOG.info(
        "ingest_post url=%s current_status=%s stages_to_run=%s",
        url, current, stages_to_run,
    )

    total = len(stages_to_run)
    done = 0

    if total == 0:
        _tick("already complete", 0, 0)
        return _summary_post(conn, url)

    if Stage.FETCH_POST in stages_to_run:
        _tick(f"starting {Stage.FETCH_POST.value}", done, total)
        pm = fetch_post_stage(
            conn=conn,
            url=url,
            force=force,
            domain_override=domain,
        )
        post_name = pm.post_name
        post_status = PostStatus(pm.status) if pm.status else None
        if post_status in (PostStatus.FAILED_FETCH, PostStatus.FAILED_PARSE):
            _LOG.warning("fetch_post produced %s for %s; halting pipeline",
                         post_status.value, url)
            done += 1
            _tick("complete", done, total)
            return _summary_post(conn, url)
        done += 1

    if post_name is None:
        raise RuntimeError(
            f"internal invariant: no post_name resolved for {url!r} "
            "after resume lookup and without scheduling fetch"
        )

    if Stage.CONVERT_POST in stages_to_run:
        _tick(f"starting {Stage.CONVERT_POST.value}", done, total)
        result = convert_post_stage(post_name=post_name, conn=conn, force=force)
        # convert_post may downgrade to FAILED_PARSE when trafilatura
        # returned an empty body. Halt the pipeline before classify.
        if result.status == PostStatus.FAILED_PARSE.value:
            _LOG.warning("convert_post produced FAILED_PARSE for %s; halting", url)
            done += 1
            _tick("complete", done, total)
            return _summary_post(conn, url)
        done += 1
    if Stage.CLASSIFY_POST in stages_to_run:
        _tick(f"starting {Stage.CLASSIFY_POST.value}", done, total)
        classify_paper_stage(
            conn=conn,
            paper_name=post_name,
            force=force,
            domain_override=domain,
        )
        done += 1
    if Stage.EXTRACT_POST in stages_to_run:
        _tick(f"starting {Stage.EXTRACT_POST.value}", done, total)
        extract_stage(conn=conn, paper_name=post_name, force=force)
        done += 1
    if Stage.INDEX_POST in stages_to_run:
        _tick(f"starting {Stage.INDEX_POST.value}", done, total)
        index_stage(conn=conn, paper_name=post_name, force=force)
        done += 1

    _tick("complete", done, total)

    return _summary_post(conn, url)


# ---------------------------------------------------------------------------
# Local-PDF orchestrator
# ---------------------------------------------------------------------------


class _ChapterRecord(NamedTuple):
    """One row to push through the per-paper pipeline. Filled in two
    passes: first the (arxiv_id, title, page range) from the outline,
    then ``paper_name`` from either an existing row's slug (resume) or
    a freshly generated chapter slug (new ingest)."""

    arxiv_id: str
    chapter_index: int | None  # None → whole-book (--no-split)
    title: str
    page_start: int | None
    page_end: int | None
    paper_name: str


def _force_delete_book_rows(conn: sqlite3.Connection, *, content_hash: str) -> int:
    """Cascade-delete every papers row matching ``pdf:<hash[:12]>`` or
    ``pdf:<hash[:12]>:%`` in a single transaction. Returns the count."""
    prefix = f"pdf:{content_hash[:12]}"
    rows = conn.execute(
        "SELECT id FROM papers WHERE arxiv_id = ? OR arxiv_id LIKE ?",
        (prefix, f"{prefix}:%"),
    ).fetchall()
    with transaction(conn):
        for row in rows:
            delete_paper_cascade(conn, paper_id=row[0])
    return len(rows)


def _resolve_book_slug_from_existing(
    conn: sqlite3.Connection, content_hash: str,
) -> str | None:
    """Recover the existing book_slug for partial resumes.

    Chapter rows from the same book share ``arxiv_id`` prefix
    ``pdf:<hash[:12]>`` and ``paper_name`` prefix ``<book_slug>__``.
    Prefer chapter rows over the whole-book row when both exist: the
    chapter ``paper_name`` carries the ``__`` separator so the split
    is unambiguous, and a stale whole-book row from an old ``--no-split``
    ingest shouldn't shadow the current chapter namespace. If only the
    whole-book row exists, returns it as-is (its paper_name *is* the
    book slug).
    """
    prefix = f"pdf:{content_hash[:12]}"
    # hash is hex; prefix has no LIKE wildcards. No ESCAPE needed.
    chapter_row = conn.execute(
        "SELECT paper_name FROM papers "
        " WHERE arxiv_id LIKE ? "
        " ORDER BY arxiv_id LIMIT 1",
        (f"{prefix}:%",),
    ).fetchone()
    bare_row = conn.execute(
        "SELECT paper_name FROM papers WHERE arxiv_id = ?",
        (prefix,),
    ).fetchone()
    if chapter_row is not None and bare_row is not None:
        _LOG.warning(
            "pdf:%s has BOTH a whole-book row (%s) and chapter rows; "
            "recovering book_slug from the chapter row — the whole-book "
            "row is stale and should be cleaned up via --force",
            content_hash[:12], bare_row[0],
        )
    row = chapter_row if chapter_row is not None else bare_row
    if row is None:
        return None
    name = row[0]
    if "__" in name:
        return name.split("__", 1)[0]
    return name


def _run_chapter_pipeline(
    *,
    conn: sqlite3.Connection,
    pdf_bytes: bytes,
    book_meta,
    rec: _ChapterRecord,
    force: bool,
    domain: str | None,
) -> dict:
    """Run the per-paper pipeline for a single chapter (or the whole-book
    row when ``rec.chapter_index is None``). Mirrors the resume logic in
    :func:`ingest` but for synthetic ``pdf:`` arxiv_ids — the per-row
    state machine and stage functions are reused as-is."""
    existing_row = _get_paper_row(conn, rec.arxiv_id)

    current: PaperStatus | None = None
    paper_name = rec.paper_name

    if existing_row is not None:
        try:
            current = PaperStatus(existing_row.status)
        except ValueError as exc:
            raise ValueError(
                f"papers.status={existing_row.status!r} for "
                f"arxiv_id={rec.arxiv_id!r} is not a recognized PaperStatus"
            ) from exc
        paper_name = existing_row.name
        if current is PaperStatus.FAILED_HTML:
            _LOG.info(
                "chapter %s is FAILED_HTML; use --force to retry", rec.arxiv_id,
            )
            return _summary_paper(conn, rec.arxiv_id)

    stages_to_run = _remaining_paper_stages(current)
    if not stages_to_run:
        return _summary_paper(conn, rec.arxiv_id)

    if Stage.FETCH in stages_to_run:
        if rec.page_start is None:
            # No page range → the PDF bytes ARE the chapter (manual ingest)
            # or the whole-book ingest (`--no-split`). Both paths render
            # the entire document; the difference is whether
            # ``rec.title`` carries a chapter title (manual) or the book
            # title (whole-book).
            markdown = render_whole_book_markdown(pdf_bytes)
        else:
            spec = ChapterSpec(
                index=rec.chapter_index or 0,
                title=rec.title,
                page_start=rec.page_start,
                page_end=rec.page_end,
            )
            markdown = render_chapter_markdown(pdf_bytes, spec)
        load_pdf_chapter(
            conn=conn,
            book_meta=book_meta,
            chapter_paper_name=paper_name,
            chapter_arxiv_id=rec.arxiv_id,
            chapter_title=rec.title,
            markdown=markdown,
            domain_override=domain,
        )
    if Stage.CONVERT in stages_to_run:
        convert_stage(conn=conn, paper_name=paper_name, force=force)
    if Stage.CLASSIFY in stages_to_run:
        classify_paper_stage(
            conn=conn, paper_name=paper_name, force=force, domain_override=domain,
        )
    if Stage.EXTRACT in stages_to_run:
        extract_stage(conn=conn, paper_name=paper_name, force=force)
    if Stage.INDEX in stages_to_run:
        index_stage(conn=conn, paper_name=paper_name, force=force)

    return _summary_paper(conn, rec.arxiv_id)


def ingest_pdf(
    *,
    conn: sqlite3.Connection,
    pdf_path: Path,
    force: bool = False,
    no_split: bool = False,
    domain: str | None = None,
    progress: ProgressFn | None = None,
) -> dict:
    """Ingest a local PDF, splitting by its embedded outline by default.

    Each chapter becomes one ``papers`` row with synthetic ``arxiv_id``
    ``pdf:<hash[:12]>:ch<NN>`` and ``paper_name``
    ``<book_slug>__ch<NN>_<chapter_slug>``. ``no_split=True`` skips the
    outline (and outline validation) and ingests the whole PDF as a
    single row with bare ``<book_slug>`` / ``pdf:<hash[:12]>``.

    Raises :class:`LocalPdfNoUsableOutline` when the embedded outline
    can't be split — the CLI catches this and surfaces the
    ``--no-split`` / pre-slice guidance. Per-chapter pipeline failures
    after the outline check are caught and logged so one bad chapter
    doesn't abort the rest of the book.
    """
    def _tick(msg: str, done: int, total: int) -> None:
        if progress is not None:
            progress(msg, done, total)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    book_meta = read_pdf_metadata(pdf_bytes, pdf_path)
    content_hash = book_meta.source_content_hash

    # Outline pass FIRST — must validate before any destructive --force
    # cascade so an unparseable outline (which raises LocalPdfNoUsableOutline)
    # doesn't take prior chapters down with it.
    if no_split:
        partial_records: list[tuple[str, int | None, str, int | None, int | None]] = [
            (synthetic_arxiv_id(content_hash, None), None, book_meta.title, None, None),
        ]
    else:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            specs = discover_chapters(doc)
        finally:
            doc.close()
        partial_records = [
            (
                synthetic_arxiv_id(content_hash, spec.index),
                spec.index,
                spec.title,
                spec.page_start,
                spec.page_end,
            )
            for spec in specs
        ]

    # Outline validated — safe to cascade-delete prior rows now.
    if force:
        deleted = _force_delete_book_rows(conn, content_hash=content_hash)
        if deleted:
            _LOG.info(
                "force cascade: removed %d rows for pdf:%s",
                deleted, content_hash[:12],
            )

    # Slug + pipeline pass: recover existing book_slug for partial
    # resumes, else generate fresh. Per-chapter paper_name comes from
    # the existing row (if any) or a fresh chapter slug. A slug-
    # generation failure for one chapter is captured as a failed
    # envelope and the rest of the book continues (docstring contract:
    # one bad chapter doesn't abort the rest).
    existing = existing_slugs(conn)
    recovered_book_slug = _resolve_book_slug_from_existing(conn, content_hash)
    if recovered_book_slug is not None:
        book_slug = recovered_book_slug
    else:
        book_slug = generate_book_slug(
            book_meta.title, book_meta.published, content_hash, existing,
        )

    existing_with_book = set(existing)
    if recovered_book_slug is None:
        existing_with_book.add(book_slug)

    total = len(partial_records)
    chapter_summaries: list[dict] = []

    for i, (arxiv_id, idx, title, page_start, page_end) in enumerate(partial_records):
        existing_row = _get_paper_row(conn, arxiv_id)
        if existing_row is not None:
            paper_name = existing_row.name
        elif idx is None:
            paper_name = book_slug
        else:
            try:
                paper_name = generate_chapter_slug(
                    book_slug, idx, title, existing_with_book,
                )
            except ValueError as exc:
                _LOG.exception(
                    "chapter %d (%r) slug generation failed: %s",
                    idx, title, exc,
                )
                chapter_summaries.append({
                    "kind": "paper",
                    "paper_name": None,
                    "arxiv_id": arxiv_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue
            existing_with_book.add(paper_name)

        rec = _ChapterRecord(
            arxiv_id=arxiv_id,
            chapter_index=idx,
            title=title,
            page_start=page_start,
            page_end=page_end,
            paper_name=paper_name,
        )
        _tick(f"chapter {i + 1}/{total}: {rec.paper_name}", i, total)
        try:
            summary = _run_chapter_pipeline(
                conn=conn,
                pdf_bytes=pdf_bytes,
                book_meta=book_meta,
                rec=rec,
                force=force,
                domain=domain,
            )
            chapter_summaries.append(summary)
        except Exception as exc:
            _LOG.exception(
                "chapter %s (arxiv_id=%s) failed mid-pipeline: %s",
                rec.paper_name, rec.arxiv_id, exc,
            )
            chapter_summaries.append({
                "kind": "paper",
                "paper_name": rec.paper_name,
                "arxiv_id": rec.arxiv_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })

    _tick("complete", total, total)

    return {
        "kind": "pdf",
        "pdf_path": str(pdf_path),
        "book_slug": book_slug,
        "content_hash": content_hash,
        "chapter_count": total,
        "chapters": chapter_summaries,
    }


def _validate_book_slug(slug: str) -> None:
    """Reject anything that breaks the ``<book_slug>__ch<NN>_*`` convention.

    The slug must match ``^[a-z0-9_]+$`` (same as auto-generated book
    slugs) AND must not contain ``__`` — the double underscore is
    reserved as the book/chapter separator, so a book_slug that
    embedded one would be indistinguishable from an existing chapter
    boundary on string ops.
    """
    if not _SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"--book-slug must match ^[a-z0-9_]+$; got {slug!r}"
        )
    if "__" in slug:
        raise ValueError(
            f"--book-slug must not contain '__' (reserved chapter "
            f"separator); got {slug!r}"
        )


def _escape_like(s: str, escape_char: str = "\\") -> str:
    """Escape SQL LIKE wildcards (``_`` and ``%``) in ``s``.

    ``book_slug`` is allowed to contain ``_`` (it matches ``^[a-z0-9_]+$``)
    so any LIKE pattern built from it must escape underscores or it
    matches any single char — silently expanding the deletion scope to
    rows of unrelated books whose paper_name happens to line up.
    """
    return (
        s.replace(escape_char, escape_char + escape_char)
         .replace("_", escape_char + "_")
         .replace("%", escape_char + "%")
    )


def _force_delete_chapter_slot(
    conn: sqlite3.Connection, *, book_slug: str, chapter_index: int,
    arxiv_id: str | None = None,
) -> int:
    """Cascade-delete any row in the ``<book_slug>__ch<NN>_*`` slot.

    Targets paper_name (the user-facing slug), not just arxiv_id — that
    way a force re-ingest with a *different* chapter file (different
    content_hash) still clears the old row sitting in the same slot.
    When ``arxiv_id`` is provided the row matching that synthetic id is
    ALSO deleted, so re-running with a different ``--book-slug``
    relocates cleanly (otherwise the prior row under the OLD book_slug
    would survive the slot cleanup).
    """
    literal_prefix = f"{book_slug}__ch{chapter_index:02d}_"
    escaped_prefix = _escape_like(literal_prefix)
    pattern = escaped_prefix + "%"
    if arxiv_id is None:
        rows = conn.execute(
            "SELECT id FROM papers WHERE paper_name LIKE ? ESCAPE '\\'",
            (pattern,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM papers "
            " WHERE paper_name LIKE ? ESCAPE '\\' OR arxiv_id = ?",
            (pattern, arxiv_id),
        ).fetchall()
    with transaction(conn):
        for row in rows:
            delete_paper_cascade(conn, paper_id=row[0])
    return len(rows)


def ingest_pdf_chapter(
    *,
    conn: sqlite3.Connection,
    pdf_path: Path,
    book_slug: str,
    chapter_index: int,
    chapter_title: str | None = None,
    force: bool = False,
    domain: str | None = None,
    progress: ProgressFn | None = None,
) -> dict:
    """Ingest a single hand-sliced chapter PDF under an explicit book namespace.

    Lets the user assemble a "book" from arbitrary chapter PDFs by
    declaring the shared ``book_slug`` and per-call ``chapter_index``.
    The resulting ``paper_name`` is
    ``<book_slug>__ch<NN>_<chapter_slug>`` — exact same shape as
    auto-split chapters, so ``WHERE paper_name LIKE 'book_slug__%'
    ORDER BY paper_name`` returns them in TOC order.

    ``arxiv_id`` is synthesized from the chapter PDF's own sha256
    (``pdf:<sha256[:12]>:ch<NN>``); siblings under the same book_slug
    do NOT share an arxiv_id prefix (each chapter file has its own
    hash). ``--force`` cascades by ``paper_name`` pattern instead, so
    swapping in a different file for the same slot works.

    Re-running with the same ``pdf_path`` + ``book_slug`` +
    ``chapter_index`` is idempotent — picks up at the last completed
    pipeline stage just like the arxiv path.
    """
    def _tick(msg: str, done: int, total: int) -> None:
        if progress is not None:
            progress(msg, done, total)

    _validate_book_slug(book_slug)
    if chapter_index < 1:
        raise ValueError(f"chapter_index must be >= 1; got {chapter_index}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    pdf_bytes = pdf_path.read_bytes()
    book_meta = read_pdf_metadata(pdf_bytes, pdf_path)
    content_hash = book_meta.source_content_hash
    arxiv_id = synthetic_arxiv_id(content_hash, chapter_index)

    if force:
        # Delete by EITHER the new paper_name slot OR the arxiv_id —
        # the arxiv_id branch covers the relocation case where the
        # same chapter file was previously ingested under a different
        # --book-slug (paper_name LIKE won't match the old prefix).
        deleted = _force_delete_chapter_slot(
            conn, book_slug=book_slug, chapter_index=chapter_index,
            arxiv_id=arxiv_id,
        )
        if deleted:
            _LOG.info(
                "force cascade: removed %d rows in slot %s__ch%02d_* or arxiv_id=%s",
                deleted, book_slug, chapter_index, arxiv_id,
            )

    effective_title = (chapter_title or book_meta.title).strip() or book_meta.title

    existing_row = _get_paper_row(conn, arxiv_id)
    if existing_row is not None:
        expected_prefix = f"{book_slug}__"
        if not existing_row.name.startswith(expected_prefix):
            raise ValueError(
                f"chapter PDF already ingested as paper_name="
                f"{existing_row.name!r} under a different book_slug; "
                f"pass --force to relocate to book_slug={book_slug!r}"
            )
        paper_name = existing_row.name
    else:
        paper_name = generate_chapter_slug(
            book_slug, chapter_index, effective_title, existing_slugs(conn),
        )

    rec = _ChapterRecord(
        arxiv_id=arxiv_id,
        chapter_index=chapter_index,
        title=effective_title,
        page_start=None,  # signals "render whole PDF" — the file IS the chapter
        page_end=None,
        paper_name=paper_name,
    )

    _tick(f"chapter {chapter_index}: {paper_name}", 0, 1)
    try:
        summary = _run_chapter_pipeline(
            conn=conn,
            pdf_bytes=pdf_bytes,
            book_meta=book_meta,
            rec=rec,
            force=force,
            domain=domain,
        )
    except Exception as exc:
        _LOG.exception(
            "manual chapter ingest failed: paper_name=%s arxiv_id=%s: %s",
            paper_name, arxiv_id, exc,
        )
        summary = {
            "kind": "paper",
            "paper_name": paper_name,
            "arxiv_id": arxiv_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    _tick("complete", 1, 1)

    return {
        "kind": "pdf_chapter",
        "pdf_path": str(pdf_path),
        "book_slug": book_slug,
        "chapter_index": chapter_index,
        "content_hash": content_hash,
        "chapter": summary,
    }


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
    target.add_argument("--post", default=None, help="blog post URL")
    target.add_argument("--pdf", default=None,
                        help="local PDF path; outline-split into per-chapter rows by default")
    target.add_argument("--acl", default=None,
                        help="ACL Anthology paper id or URL "
                             "(e.g. 2021.acl-long.285, P19-1001, "
                             "https://aclanthology.org/2021.acl-long.285/, "
                             "or its .pdf/.xml/.bib asset URL)")
    parser.add_argument("--no-split", action="store_true",
                        help="ingest the whole PDF as a single row (skips outline validation); "
                             "only meaningful with --pdf")
    parser.add_argument("--book-slug", default=None,
                        help="manual chapter ingest: explicit book_slug to share across "
                             "hand-sliced chapter PDFs (paper_name becomes "
                             "<book_slug>__ch<NN>_<chapter_slug>); requires --pdf and "
                             "--chapter-index")
    parser.add_argument("--chapter-index", type=int, default=None,
                        help="manual chapter ingest: 1-based chapter index (zero-padded "
                             "in the slug); requires --book-slug")
    parser.add_argument("--chapter-title", default=None,
                        help="manual chapter ingest: title for this chapter row "
                             "(defaults to the chapter PDF's title metadata)")
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

    if args.no_split and not args.pdf:
        parser.error("--no-split is only meaningful with --pdf")
    manual_chapter = (
        args.book_slug is not None
        or args.chapter_index is not None
        or args.chapter_title is not None
    )
    if manual_chapter:
        if not args.pdf:
            parser.error(
                "--book-slug / --chapter-index / --chapter-title require --pdf"
            )
        if args.book_slug is None or args.chapter_index is None:
            parser.error(
                "manual chapter ingest requires both --book-slug and --chapter-index"
            )
        if args.no_split:
            parser.error("--no-split cannot be combined with --book-slug")

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
    elif args.repo:
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
    elif args.pdf:
        check_models()
        conn = get_conn(Path(args.db))
        try:
            init_db(conn)
            if manual_chapter:
                summary = ingest_pdf_chapter(
                    conn=conn,
                    pdf_path=Path(args.pdf),
                    book_slug=args.book_slug,
                    chapter_index=args.chapter_index,
                    chapter_title=args.chapter_title,
                    force=args.force,
                    domain=args.domain,
                )
            else:
                try:
                    summary = ingest_pdf(
                        conn=conn,
                        pdf_path=Path(args.pdf),
                        force=args.force,
                        no_split=args.no_split,
                        domain=args.domain,
                    )
                except LocalPdfNoUsableOutline as exc:
                    msg = (
                        f"{exc}\n"
                        "PDF has no usable embedded outline. To ingest this book, "
                        "either: (a) re-run with --no-split to treat the whole PDF "
                        "as one paper, (b) pre-slice the PDF into per-chapter files "
                        "and ingest each with --book-slug / --chapter-index, or (c) "
                        "ingest the whole file as a single row with --no-split."
                    )
                    raise SystemExit(msg)
        finally:
            conn.close()
    elif args.acl:
        acl_id = parse_acl_id(args.acl)
        check_models()
        conn = get_conn(Path(args.db))
        try:
            init_db(conn)
            summary = ingest_acl(
                conn=conn,
                acl_id=acl_id,
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
            summary = ingest_post(
                conn=conn,
                url=args.post,
                force=args.force,
                domain=args.domain,
            )
        finally:
            conn.close()

    print(json.dumps(summary))


if __name__ == "__main__":
    main()
