"""Post status enum, status ordering, and the `posts` row model.

Mirrors :mod:`_system.schemas.paper_metadata`. Two terminal failure
states (vs paper's one ``FAILED_HTML``) because blogs fail in two
qualitatively different ways: network refused (``FAILED_FETCH``) vs
returned HTML that trafilatura couldn't extract from (``FAILED_PARSE``).
"""
from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PostStatus(StrEnum):
    FETCHED = "fetched"
    CONVERTED = "converted"
    CLASSIFIED = "classified"
    EXTRACTED = "extracted"
    INDEXED = "indexed"
    FAILED_FETCH = "failed_fetch"
    FAILED_PARSE = "failed_parse"


STATUS_ORDER: dict[PostStatus, int] = {
    PostStatus.FETCHED: 0,
    PostStatus.CONVERTED: 1,
    PostStatus.CLASSIFIED: 2,
    PostStatus.EXTRACTED: 3,
    PostStatus.INDEXED: 4,
    PostStatus.FAILED_FETCH: -1,
    PostStatus.FAILED_PARSE: -1,
}


def can_run_from(
    current: Optional[PostStatus], target_stage: PostStatus
) -> bool:
    """True iff running ``target_stage`` is meaningful given ``current``.

    Mirrors :func:`_system.schemas.paper_metadata.can_run_from`. Negative
    STATUS_ORDER values are terminal and short-circuit.
    """
    if current is None:
        return True
    if STATUS_ORDER[current] < 0:
        return False
    delta = STATUS_ORDER[target_stage] - STATUS_ORDER[current]
    return 0 <= delta <= 1


class PostMetadata(BaseModel):
    """Pydantic mirror of a ``posts`` table row."""

    model_config = ConfigDict(use_enum_values=True)

    post_name: str
    source_url: str
    canonical_url: str
    title: str
    author: Optional[str] = None
    site_name: Optional[str] = None
    date: str
    abstract: str
    domain: Optional[str] = None
    collection: Optional[str] = None
    status: PostStatus
    markdown: Optional[str] = None
    raw_html: Optional[str] = None
    content_hash: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    needs_review: bool = False
    ingested_at: Optional[str] = None
    # Discovered repo URL — transient. Surfaced by ``fetch_post`` so the
    # ingest orchestrator can register a standalone ``repos`` row. Never
    # persisted to the ``posts`` table.
    code_repo: Optional[str] = None
