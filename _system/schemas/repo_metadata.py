"""Repo status enum, status ordering, and the `repos` row model.

Mirrors the shape of :mod:`_system.schemas.paper_metadata`. A repo is a
first-class entity addressable by ``repo_slug``; it may be paper-linked
(``paper_id`` set) or standalone (``paper_id`` NULL). Standalone repos
flow through their own state machine, classify themselves from a README,
and otherwise reuse the same fetch / persist machinery as paper-linked
repos.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class RepoStatus(StrEnum):
    RESOLVED = "resolved"            # repos row created with metadata
    REPO_FETCHED = "repo_fetched"    # clone walked, code_files + readmes_fts persisted
    CLASSIFIED = "classified"        # domain/collection/topics assigned (terminal happy path)
    ORPHANED = "orphaned"            # fetched but no usable README; classification skipped
    FAILED_RESOLVE = "failed_resolve"
    FAILED_REPO = "failed_repo"


class TopicTarget(StrEnum):
    """Discriminator for the unified ``topics`` table."""

    PAPER = "paper"
    REPO = "repo"
    POST = "post"


# CLASSIFIED is the standalone happy-path terminus for repos with a usable
# README; ORPHANED is the terminus for repos without one. Paper-linked repos
# inherit domain/collection from the paper and skip the CLASSIFY stage —
# they reach REPO_FETCHED and stay there.
STATUS_ORDER: dict[RepoStatus, int] = {
    RepoStatus.RESOLVED: 0,
    RepoStatus.REPO_FETCHED: 1,
    RepoStatus.CLASSIFIED: 2,
    RepoStatus.ORPHANED: -1,
    RepoStatus.FAILED_RESOLVE: -1,
    RepoStatus.FAILED_REPO: -1,
}


def can_run_from(
    current: Optional[RepoStatus], target_stage: RepoStatus
) -> bool:
    """True iff running ``target_stage`` is meaningful given ``current``.

    Mirrors :func:`_system.schemas.paper_metadata.can_run_from`. Negative
    STATUS_ORDER values are terminal (ORPHANED, FAILED_*) and short-circuit.
    """
    if current is None:
        return True
    if STATUS_ORDER[current] < 0:
        return False
    delta = STATUS_ORDER[target_stage] - STATUS_ORDER[current]
    return 0 <= delta <= 1


class RepoMetadata(BaseModel):
    """Pydantic mirror of a ``repos`` table row."""

    model_config = ConfigDict(use_enum_values=True)

    repo_slug: str
    url: str
    host: str
    owner: str
    name: str
    paper_id: Optional[int] = None
    description: Optional[str] = None
    default_branch: Optional[str] = None
    commit_sha: Optional[str] = None
    fetched_at: Optional[str] = None
    ingested_at: Optional[str] = None
    domain: Optional[str] = None
    collection: Optional[str] = None
    status: RepoStatus
    needs_review: bool = False
    file_count: int = 0
    has_readme: bool = False
