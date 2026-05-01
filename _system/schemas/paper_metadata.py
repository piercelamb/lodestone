"""Paper status enum, status ordering, and the `papers` row model."""
from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PaperStatus(StrEnum):
    FETCHED = "fetched"
    CONVERTED = "converted"
    CLASSIFIED = "classified"
    EXTRACTED = "extracted"
    INDEXED = "indexed"
    REPO_FETCHED = "repo_fetched"
    FAILED_HTML = "failed_html"
    FAILED_REPO = "failed_repo"


class HtmlSource(StrEnum):
    ARXIV = "arxiv"
    AR5IV = "ar5iv"
    LATEX_LOCAL = "latex_local"


STATUS_ORDER: dict[PaperStatus, int] = {
    PaperStatus.FETCHED: 0,
    PaperStatus.CONVERTED: 1,
    PaperStatus.CLASSIFIED: 2,
    PaperStatus.EXTRACTED: 3,
    PaperStatus.INDEXED: 4,
    PaperStatus.REPO_FETCHED: 5,
    PaperStatus.FAILED_HTML: -1,
    PaperStatus.FAILED_REPO: -1,
}


def can_run_from(
    current: Optional[PaperStatus], target_stage: PaperStatus
) -> bool:
    """True iff running `target_stage` is meaningful given `current`.

    You may rerun the current stage or advance one step past it. You may
    not skip ahead or go backwards. Terminal sentinels (FAILED_HTML,
    FAILED_REPO) short-circuit. Callers use --force to bypass this check.
    """
    if current is None:
        return True
    if STATUS_ORDER[current] < 0:
        return False
    delta = STATUS_ORDER[target_stage] - STATUS_ORDER[current]
    return 0 <= delta <= 1


SECTION_CHUNK_LEVELS: tuple[int, int, int] = (1, 2, 3)


class PaperMetadata(BaseModel):
    """Pydantic mirror of a `papers` table row."""

    model_config = ConfigDict(use_enum_values=True)

    arxiv_id: str
    paper_name: str
    title: str
    authors: str
    date: str
    abstract: str
    pdf_url: str
    domain: Optional[str] = None
    collection: Optional[str] = None
    status: PaperStatus
    markdown: Optional[str] = None
    raw_html: Optional[str] = None
    html_source: Optional[str] = None
    content_hash: Optional[str] = None
    code_repo: Optional[str] = None
    needs_review: bool = False
    ingested_at: Optional[str] = None
