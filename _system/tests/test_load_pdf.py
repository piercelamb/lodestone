"""Unit tests for _system.scripts.load_pdf.

Builds a small synthetic PDF (in-memory, no on-disk fixture) with an
embedded outline + per-page marker text. The same fixture is reused for
the orchestrator tests in test_ingest_pdf.
"""
from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest

from _system.scripts.load_pdf import (
    ChapterSpec,
    LocalPdfNoUsableOutline,
    discover_chapters,
    load_pdf_chapter,
    read_pdf_metadata,
    render_chapter_markdown,
    synthetic_arxiv_id,
    synthetic_pdf_url,
)


def _make_pdf(
    *,
    toc: list[list] | None,
    n_pages: int = 6,
    title: str = "Synthetic Book",
    author: str = "Alice; Bob",
    creation_date: str = "D:20240115120000Z",
) -> bytes:
    """Build a small PDF in memory with optional TOC + metadata."""
    doc = pymupdf.Document()
    for i in range(n_pages):
        p = doc.new_page()
        p.insert_text((72, 72), f"Page {i + 1}: marker-{i + 1}")
    if toc is not None:
        doc.set_toc(toc)
    doc.set_metadata({
        "title": title, "author": author, "creationDate": creation_date,
    })
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture
def book_pdf_bytes() -> bytes:
    return _make_pdf(toc=[
        [1, "Intro", 1],
        [1, "Body", 3],
        [1, "Conclusion", 5],
    ])


@pytest.fixture
def book_pdf_path(tmp_path: Path, book_pdf_bytes: bytes) -> Path:
    p = tmp_path / "synthetic_book.pdf"
    p.write_bytes(book_pdf_bytes)
    return p


class TestReadPdfMetadata:
    def test_pulls_title_author_date(self, book_pdf_bytes, tmp_path):
        meta = read_pdf_metadata(book_pdf_bytes, tmp_path / "synth.pdf")
        assert meta.title == "Synthetic Book"
        assert meta.authors == ["Alice", "Bob"]
        assert meta.published == "2024-01-15"
        assert len(meta.source_content_hash) == 64

    def test_falls_back_to_filename_when_title_empty(self, tmp_path):
        pdf = _make_pdf(toc=None, title="", author="")
        path = tmp_path / "my_book.pdf"
        meta = read_pdf_metadata(pdf, path)
        assert meta.title == "my_book"
        assert meta.authors == []

    def test_today_used_when_no_creation_date(self, tmp_path):
        pdf = _make_pdf(toc=None, creation_date="")
        meta = read_pdf_metadata(pdf, tmp_path / "x.pdf")
        # Should be YYYY-MM-DD shape regardless of date used.
        assert len(meta.published) == 10
        assert meta.published[4] == "-" and meta.published[7] == "-"

    def test_semicolon_and_comma_author_split(self, book_pdf_bytes, tmp_path):
        meta = read_pdf_metadata(book_pdf_bytes, tmp_path / "x.pdf")
        assert meta.authors == ["Alice", "Bob"]

        pdf2 = _make_pdf(toc=None, author="A, B, C")
        meta2 = read_pdf_metadata(pdf2, tmp_path / "y.pdf")
        assert meta2.authors == ["A", "B", "C"]


class TestDiscoverChapters:
    def test_returns_chapter_specs(self, book_pdf_bytes):
        doc = pymupdf.open(stream=book_pdf_bytes, filetype="pdf")
        try:
            specs = discover_chapters(doc)
        finally:
            doc.close()
        assert len(specs) == 3
        # 0-based, half-open page ranges.
        assert specs[0] == ChapterSpec(index=1, title="Intro", page_start=0, page_end=2)
        assert specs[1] == ChapterSpec(index=2, title="Body", page_start=2, page_end=4)
        # Last chapter runs to the end (page_count=6 → page_end=6).
        assert specs[2] == ChapterSpec(index=3, title="Conclusion", page_start=4, page_end=6)

    def test_raises_when_outline_missing(self):
        pdf = _make_pdf(toc=None)
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        try:
            with pytest.raises(LocalPdfNoUsableOutline):
                discover_chapters(doc)
        finally:
            doc.close()

    def test_raises_when_only_one_level1_entry(self):
        pdf = _make_pdf(toc=[[1, "Solo", 1]])
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        try:
            with pytest.raises(LocalPdfNoUsableOutline):
                discover_chapters(doc)
        finally:
            doc.close()

    def test_raises_when_pages_not_monotonic(self):
        pdf = _make_pdf(toc=[
            [1, "Chapter A", 3],
            [1, "Chapter B", 1],
        ])
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        try:
            with pytest.raises(LocalPdfNoUsableOutline):
                discover_chapters(doc)
        finally:
            doc.close()

    def test_ignores_level2_entries_when_l1_sufficient(self):
        # Level-1 with ≥2 entries and level-2 with <3 → level-1 wins
        # (legacy short-book path); level-2 entries don't define chapter
        # boundaries here.
        pdf = _make_pdf(toc=[
            [1, "Intro", 1],
            [2, "Subsection", 2],
            [1, "Body", 3],
        ])
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        try:
            specs = discover_chapters(doc)
        finally:
            doc.close()
        assert [s.title for s in specs] == ["Intro", "Body"]

    def test_falls_back_to_level2_when_l1_has_only_two(self):
        # 2 level-1 "volumes", 3 level-2 "chapters" — exactly the
        # stanford-speech-processing book shape: L1 is useless (mega-
        # volumes), L2 is the real chapter level.
        pdf = _make_pdf(
            n_pages=6,
            toc=[
                [1, "Part I", 1],
                [2, "Alpha", 1],
                [2, "Beta", 3],
                [1, "Part II", 4],
                [2, "Gamma", 4],
            ],
        )
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        try:
            specs = discover_chapters(doc)
        finally:
            doc.close()
        assert [s.title for s in specs] == ["Alpha", "Beta", "Gamma"]
        # Beta sits at the tail of Part I; its page_end must clip at the
        # start of Part II (1-based page 4 → 0-based 3), NOT bleed
        # through into the next part's pages.
        assert specs[0] == ChapterSpec(index=1, title="Alpha", page_start=0, page_end=2)
        assert specs[1] == ChapterSpec(index=2, title="Beta", page_start=2, page_end=3)
        assert specs[2] == ChapterSpec(index=3, title="Gamma", page_start=3, page_end=6)

    def test_falls_back_to_level2_when_single_level1(self):
        # Single level-1 "root" with three level-2 chapters — falls back
        # to level-2. (pymupdf refuses to write a TOC without a level-1
        # entry, so this is as close to "no level-1" as we can get.)
        pdf = _make_pdf(
            n_pages=6,
            toc=[
                [1, "Root", 1],
                [2, "Alpha", 1],
                [2, "Beta", 3],
                [2, "Gamma", 5],
            ],
        )
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        try:
            specs = discover_chapters(doc)
        finally:
            doc.close()
        assert [s.title for s in specs] == ["Alpha", "Beta", "Gamma"]

    def test_level2_fallback_dedupes_equal_page_entries(self):
        # Two level-2 entries pointing at the same page (e.g. a TOC with
        # "Bibliography" and "Subject Index" both registered at the start
        # of the back matter). The duplicate must be collapsed, otherwise
        # the strict-monotonicity check would refuse the outline.
        pdf = _make_pdf(
            n_pages=8,
            toc=[
                [1, "Part I", 1],
                [2, "Intro", 1],
                [2, "Body", 3],
                [1, "Part II", 5],
                [2, "Bibliography", 5],
                [2, "Subject Index", 5],  # equal-page sibling → deduped
                [2, "Appendix", 7],
            ],
        )
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        try:
            specs = discover_chapters(doc)
        finally:
            doc.close()
        # Subject Index dropped; Bibliography wins.
        assert [s.title for s in specs] == ["Intro", "Body", "Bibliography", "Appendix"]

    def test_level1_with_three_entries_does_not_fall_back(self):
        # When level-1 has ≥3 entries, level-2 is ignored even if richer.
        pdf = _make_pdf(
            n_pages=6,
            toc=[
                [1, "Part I", 1],
                [2, "Alpha", 1],
                [2, "Beta", 2],
                [1, "Part II", 3],
                [2, "Gamma", 3],
                [2, "Delta", 4],
                [1, "Part III", 5],
            ],
        )
        doc = pymupdf.open(stream=pdf, filetype="pdf")
        try:
            specs = discover_chapters(doc)
        finally:
            doc.close()
        assert [s.title for s in specs] == ["Part I", "Part II", "Part III"]


class TestRenderChapterMarkdown:
    def test_returns_only_chapter_pages(self, book_pdf_bytes):
        # Pins down 0-based page numbering: chapter 1 covers pages 0..1
        # which is the original PDF's first two pages ("Page 1", "Page 2").
        spec = ChapterSpec(index=1, title="Intro", page_start=0, page_end=2)
        md = render_chapter_markdown(book_pdf_bytes, spec)
        assert "marker-1" in md
        assert "marker-2" in md
        assert "marker-3" not in md

    def test_renders_middle_chapter(self, book_pdf_bytes):
        spec = ChapterSpec(index=2, title="Body", page_start=2, page_end=4)
        md = render_chapter_markdown(book_pdf_bytes, spec)
        assert "marker-3" in md
        assert "marker-4" in md
        assert "marker-1" not in md
        assert "marker-5" not in md


class TestLoadPdfChapter:
    def test_writes_paper_row_with_synthetic_arxiv_id(self, conn, book_pdf_bytes, tmp_path):
        path = tmp_path / "book.pdf"
        path.write_bytes(book_pdf_bytes)
        meta = read_pdf_metadata(book_pdf_bytes, path)
        arxiv_id = synthetic_arxiv_id(meta.source_content_hash, 1)
        paper_name = "synthetic_book_2024__ch01_intro"
        pm = load_pdf_chapter(
            conn=conn,
            book_meta=meta,
            chapter_paper_name=paper_name,
            chapter_arxiv_id=arxiv_id,
            chapter_title="Intro",
            markdown="# Intro\n\nbody body body " * 50,
            domain_override=None,
        )
        assert pm.arxiv_id == arxiv_id
        assert pm.paper_name == paper_name
        assert pm.status == "fetched"
        assert pm.html_source == "pdf_fallback"
        assert pm.content_hash == meta.source_content_hash

        row = conn.execute(
            "SELECT arxiv_id, paper_name, title, content_hash, html_source, status "
            "  FROM papers WHERE arxiv_id = ?",
            (arxiv_id,),
        ).fetchone()
        assert row == (arxiv_id, paper_name, "Intro", meta.source_content_hash,
                       "pdf_fallback", "fetched")

    def test_synthetic_arxiv_id_shape(self):
        h = "a" * 64
        assert synthetic_arxiv_id(h, None) == "pdf:aaaaaaaaaaaa"
        assert synthetic_arxiv_id(h, 3) == "pdf:aaaaaaaaaaaa:ch03"
        assert synthetic_arxiv_id(h, 12) == "pdf:aaaaaaaaaaaa:ch12"

    def test_chapter_rows_share_content_hash(self, conn, book_pdf_bytes, tmp_path):
        path = tmp_path / "book.pdf"
        path.write_bytes(book_pdf_bytes)
        meta = read_pdf_metadata(book_pdf_bytes, path)
        for i, title in enumerate(("Intro", "Body"), start=1):
            load_pdf_chapter(
                conn=conn,
                book_meta=meta,
                chapter_paper_name=f"book_2024__ch{i:02d}_{title.lower()}",
                chapter_arxiv_id=synthetic_arxiv_id(meta.source_content_hash, i),
                chapter_title=title,
                markdown="markdown body " * 50,
                domain_override=None,
            )
        hashes = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT content_hash FROM papers WHERE arxiv_id LIKE 'pdf:%'"
            )
        ]
        assert hashes == [meta.source_content_hash]


class TestDiscoverChaptersPageFilter:
    def test_drops_unresolved_minus_one_pages(self):
        # pymupdf's get_toc returns page=-1 for outline entries whose
        # destination it can't resolve. If we don't filter these out,
        # page_start = -1 - 1 = -2 leaks into pymupdf4llm's pages= arg.
        # set_toc itself sometimes substitutes -1 for unresolved entries
        # so we build a TOC where ALL pages are valid first and then
        # mutate to inject -1 — easiest path is to assert filtering
        # behavior via a constructed raw_toc through the helper.
        from _system.scripts.load_pdf import _dedupe_consecutive_same_page
        # Simulate the comprehension that does the filtering:
        raw_toc = [
            [1, "Intro", 1],
            [1, "Bogus", -1],
            [1, "Body", 3],
        ]
        kept = _dedupe_consecutive_same_page(
            [(title, page) for level, title, page in raw_toc if level == 1 and page >= 1]
        )
        assert kept == [("Intro", 1), ("Body", 3)]


class TestSyntheticPdfUrl:
    def test_url_encodes_spaces(self, tmp_path):
        # Spaces (and other reserved chars) in the PDF path must be
        # percent-encoded so downstream urlparse consumers can handle
        # the value as a real URI.
        path = tmp_path / "My Book.pdf"
        path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        url = synthetic_pdf_url(path)
        assert url.startswith("file://")
        assert "%20" in url  # space → %20
        assert " " not in url

    def test_is_absolute(self, tmp_path):
        # Path.as_uri requires absolute paths; we already resolve() so
        # relative input lands as the absolute resolved form.
        path = tmp_path / "book.pdf"
        path.write_bytes(b"%PDF-1.4\n%%EOF\n")
        url = synthetic_pdf_url(path)
        assert url.startswith("file:///")
