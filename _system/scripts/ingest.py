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
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

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
from _system.scripts.fetch_paper import fetch as fetch_stage
from _system.scripts.fetch_post import fetch as fetch_post_stage
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
            "section_count": 0,
            "entity_count": 0,
            "title": None,
        }
    return {
        "kind": "post",
        "post_name": row[1],
        "url": url,
        "canonical_url": row[2],
        "status": row[3],
        "needs_review": bool(row[4]),
        "domain": row[5],
        "collection": row[6],
        "section_count": int(row[7] or 0),
        "entity_count": int(row[8] or 0),
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

    Mirrors :func:`ingest`. If the post links a github repo, the discovered
    URL is forwarded to the standalone-repo path
    (:func:`ingest_repo_only`) AFTER the post pipeline finishes. In v1 we
    don't link the repo back to the post (no ``repos.post_id``); the repo
    stands on its own.
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

    discovered_repo_url: str | None = None

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
        discovered_repo_url = pm.code_repo
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

    # v1: discovered repos go through the standalone path. Failures must
    # not abort the post ingest — log loud and move on.
    if discovered_repo_url:
        try:
            ingest_repo_only(
                conn=conn,
                repo_url=discovered_repo_url,
                force=False,
                domain=domain,
            )
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "post-linked repo ingest failed for %s url=%s: %s; "
                "post ingest is otherwise complete",
                url, discovered_repo_url, exc,
            )

    return _summary_post(conn, url)


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
