"""Local-PDF loader: read metadata + outline, split into chapters, persist
each chapter as a ``papers`` row.

Counterpart to :mod:`_system.scripts.fetch_paper` for the
``ingest --pdf <path>`` entry shape. Books are split by their embedded
PDF outline (``pymupdf.Document.get_toc``); chapter rows share the
source PDF's ``content_hash`` and ride on a synthetic
``arxiv_id = "pdf:<hash[:12]>:ch<NN>"`` (``"pdf:<hash[:12]>"`` for the
whole-book ``--no-split`` shape). The PDF-fallback persistence machinery
in :mod:`_system.scripts.fetch_paper` is reused via the shared
``_persist_pdf_row`` helper — once a chapter row lands, the rest of the
pipeline (convert → classify → extract → index) treats it identically
to an arxiv paper that hit the PDF fallback path.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from _system.pdf.extract import extract_markdown
from _system.schemas.paper_metadata import PaperMetadata
from _system.scripts.fetch_paper import (
    _ArxivMetadata,
    _get_existing_paper,
    _persist_pdf_row,
)
from _system.utils.logging import get_logger

__all__ = [
    "ChapterSpec",
    "LocalPdfMetadata",
    "LocalPdfNoUsableOutline",
    "discover_chapters",
    "load_pdf_chapter",
    "read_pdf_metadata",
    "render_chapter_markdown",
]

_LOG = get_logger("scripts.load_pdf")

_AUTHOR_SPLIT_RE = re.compile(r"[;,]")
# PDF metadata creationDate format: "D:YYYYMMDDHHmmSS+HH'mm'" (PDF Date format
# per ISO 32000-1 §7.9.4). We're permissive — the leading "D:" prefix is
# optional in some producers, and we only need the YYYYMMDD prefix.
_PDF_DATE_RE = re.compile(r"D?:?(\d{4})(\d{2})?(\d{2})?")


@dataclass
class LocalPdfMetadata:
    """Synthesized book-level metadata for a local PDF."""

    title: str
    authors: list[str]
    published: str  # YYYY-MM-DD
    source_pdf_path: Path
    source_content_hash: str  # sha256 of the source PDF bytes


@dataclass
class ChapterSpec:
    """One chapter slice. ``page_start``/``page_end`` are 0-based; ``end``
    is exclusive (matches Python slice and pymupdf4llm ``pages=`` semantics)."""

    index: int  # 1-based for slug + display
    title: str
    page_start: int  # 0-based, inclusive
    page_end: int  # 0-based, exclusive


class LocalPdfNoUsableOutline(Exception):
    """Raised when a local PDF's embedded outline cannot be split into chapters.

    The caller (CLI) catches this and emits the user-facing message
    pointing at ``--no-split`` or manual pre-slicing.
    """


def _safe_filename_stem(pdf_path: Path) -> str:
    """Filename stem with extension stripped; ``"untitled"`` for empty."""
    return pdf_path.stem.strip() or "untitled"


def _parse_pdf_date(raw: str) -> str | None:
    """Parse a PDF metadata ``creationDate`` string to ``YYYY-MM-DD``.

    Returns ``None`` if the string is empty or doesn't carry a year.
    """
    if not raw:
        return None
    m = _PDF_DATE_RE.match(raw.strip())
    if not m:
        return None
    year = m.group(1)
    month = m.group(2) or "01"
    day = m.group(3) or "01"
    return f"{year}-{month}-{day}"


def read_pdf_metadata(pdf_bytes: bytes, pdf_path: Path) -> LocalPdfMetadata:
    """Synthesize a :class:`LocalPdfMetadata` from a PDF's embedded metadata.

    Empty fields are filled from heuristics:
    - title  → filename stem → ``"untitled"``
    - authors → split on ``,`` and ``;``; empty list if no usable text
    - published → ``creationDate`` parsed to ``YYYY-MM-DD``; today's date
      on miss (downstream slug builder uses ``date[:4]`` so a year is
      required even when missing in the source).
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        raw_meta = doc.metadata or {}
    finally:
        doc.close()

    raw_title = (raw_meta.get("title") or "").strip()
    title = raw_title or _safe_filename_stem(pdf_path)

    raw_authors = (raw_meta.get("author") or "").strip()
    if raw_authors:
        authors = [a.strip() for a in _AUTHOR_SPLIT_RE.split(raw_authors) if a.strip()]
    else:
        authors = []

    published = _parse_pdf_date(raw_meta.get("creationDate") or "")
    if not published:
        published = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    return LocalPdfMetadata(
        title=title,
        authors=authors,
        published=published,
        source_pdf_path=pdf_path,
        source_content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
    )


def _dedupe_consecutive_same_page(
    entries: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    """Drop entries whose page equals the previous kept entry's page.

    Defensive cleanup for TOCs that list two entries at the same start
    page (e.g. "Bibliography" and "Subject Index" both registered at the
    same page — they can't both be valid chapter starts).
    """
    out: list[tuple[str, int]] = []
    for title, page in entries:
        if out and out[-1][1] == page:
            continue
        out.append((title, page))
    return out


def discover_chapters(doc: pymupdf.Document) -> list[ChapterSpec]:
    """Read ``doc.get_toc(simple=True)`` and project TOC entries to chapters.

    Splits on level-1 entries by default. When level-1 yields fewer than 3
    entries (e.g. books split into two "volumes" or "parts" with the real
    chapters living at level-2), auto-falls back to level-2 if it has ≥3
    entries; chapter boundaries then come from the union of level-1 and
    level-2 starts so chapters stop cleanly at part headers rather than
    bleeding through them.

    Outline is usable iff after dedupe (entries on the same page are
    collapsed to one): (a) ≥3 entries at L1, or ≥3 at L2 as a fallback,
    or ≥2 at L1 (legacy short-book path); (b) page numbers strictly
    increasing across consecutive entries at the chosen level; (c) each
    chapter spans ≥1 page.

    Page numbers from pymupdf's TOC are **1-based**; the returned
    :class:`ChapterSpec` page ranges are converted to **0-based,
    half-open** for direct use with ``pymupdf4llm.to_markdown(pages=…)``.

    Raises :class:`LocalPdfNoUsableOutline` on missing/garbage outline.
    """
    raw_toc = doc.get_toc(simple=True) or []
    # pymupdf returns page=-1 for outline entries whose destination can't
    # be resolved to a page; drop those before validation so we don't
    # subtract 1 and hand pymupdf4llm a negative page index.
    level1 = _dedupe_consecutive_same_page(
        [(title, page) for level, title, page in raw_toc if level == 1 and page >= 1]
    )
    level2 = _dedupe_consecutive_same_page(
        [(title, page) for level, title, page in raw_toc if level == 2 and page >= 1]
    )

    if len(level1) >= 3:
        entries = level1
        boundary_pages = sorted({p for _, p in level1})
        chosen_level = 1
    elif len(level2) >= 3:
        entries = level2
        boundary_pages = sorted({p for _, p in level1} | {p for _, p in level2})
        chosen_level = 2
        _LOG.info(
            "PDF outline has %d level-1 entries; using %d level-2 entries as chapters",
            len(level1), len(level2),
        )
    elif len(level1) >= 2:
        entries = level1
        boundary_pages = sorted({p for _, p in level1})
        chosen_level = 1
    else:
        raise LocalPdfNoUsableOutline(
            f"PDF outline has too few entries to split: "
            f"{len(level1)} at level-1, {len(level2)} at level-2"
        )

    pages = [p for _, p in entries]
    if any(b <= a for a, b in zip(pages, pages[1:])):
        raise LocalPdfNoUsableOutline(
            f"PDF outline level-{chosen_level} pages are not strictly increasing: {pages}"
        )

    total_pages = doc.page_count

    specs: list[ChapterSpec] = []
    for i, (title, start_1based) in enumerate(entries):
        # Next boundary strictly greater than this entry's start. For the
        # L2 fallback, boundary_pages also includes L1 starts so a chapter
        # at the tail of part 1 stops cleanly when part 2 begins.
        next_start_1based = next(
            (b for b in boundary_pages if b > start_1based),
            total_pages + 1,
        )

        page_start = start_1based - 1  # 0-based, inclusive
        page_end = next_start_1based - 1  # 0-based, exclusive

        if page_end <= page_start:
            raise LocalPdfNoUsableOutline(
                f"PDF outline chapter {i + 1} ({title!r}) spans <1 page"
            )

        specs.append(ChapterSpec(
            index=i + 1,
            title=(title or "").strip(),
            page_start=page_start,
            page_end=page_end,
        ))

    return specs


def render_chapter_markdown(pdf_bytes: bytes, spec: ChapterSpec) -> str:
    """Render the chapter page range to markdown via pymupdf4llm.

    Raises :class:`RuntimeError` if pymupdf4llm produces no usable
    output for the requested page range — chapter-level extraction
    failure is louder than the arxiv whole-PDF fallback (which just
    falls through to the next tier) because the orchestrator can still
    skip this chapter without aborting the book.
    """
    pages = list(range(spec.page_start, spec.page_end))
    md = extract_markdown(pdf_bytes, pages=pages)
    if md is None:
        raise RuntimeError(
            f"pymupdf4llm produced no usable markdown for chapter "
            f"{spec.index} ({spec.title!r}, pages {spec.page_start}-{spec.page_end})"
        )
    return md


def render_whole_book_markdown(pdf_bytes: bytes) -> str:
    """Render the whole PDF to markdown for the ``--no-split`` path."""
    md = extract_markdown(pdf_bytes)
    if md is None:
        raise RuntimeError(
            "pymupdf4llm produced no usable markdown for the whole PDF"
        )
    return md


def synthetic_arxiv_id(content_hash: str, chapter_index: int | None) -> str:
    """``"pdf:<hash[:12]>"`` for whole-book; ``"pdf:<hash[:12]>:ch<NN>"`` for chapters."""
    base = f"pdf:{content_hash[:12]}"
    if chapter_index is None:
        return base
    return f"{base}:ch{chapter_index:02d}"


def synthetic_pdf_url(pdf_path: Path) -> str:
    """``file://`` URL satisfying the ``papers.pdf_url`` NOT NULL constraint.

    Uses :meth:`Path.as_uri` so spaces and other reserved chars in the
    path are percent-encoded per RFC 8089 — a raw f-string would emit
    literal spaces that downstream urlparse consumers can't handle.
    """
    return pdf_path.resolve().as_uri()


def load_pdf_chapter(
    *,
    conn: sqlite3.Connection,
    book_meta: LocalPdfMetadata,
    chapter_paper_name: str,
    chapter_arxiv_id: str,
    chapter_title: str | None,
    markdown: str,
    domain_override: str | None,
) -> PaperMetadata:
    """Persist a single book-chapter (or whole-book) row.

    The caller (the ``ingest_pdf`` orchestrator) pre-computes
    ``chapter_paper_name`` and ``chapter_arxiv_id`` so the chapter slug
    convention (``<book_slug>__ch<NN>_<title-tokens>``) is enforced
    centrally. ``existing_row`` is looked up here for ingested_at
    preservation on re-ingest.
    """
    title = (chapter_title or "").strip() or book_meta.title

    arxiv_meta = _ArxivMetadata(
        title=title,
        authors=list(book_meta.authors),
        abstract="",
        published=book_meta.published,
        comment=None,
        summary=None,
        pdf_url=synthetic_pdf_url(book_meta.source_pdf_path),
    )

    existing_row = _get_existing_paper(conn, chapter_arxiv_id)

    return _persist_pdf_row(
        conn,
        arxiv_id=chapter_arxiv_id,
        meta=arxiv_meta,
        markdown=markdown,
        content_hash=book_meta.source_content_hash,
        domain_override=domain_override,
        existing_row=existing_row,
        client=None,
        paper_name_override=chapter_paper_name,
    )
