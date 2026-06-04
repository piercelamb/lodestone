"""End-to-end tests for the ingest_pdf orchestrator.

Mirrors test_ingest_post.py: downstream stages are stubbed at module
boundaries so the orchestrator wiring is exercised without touching the
LLM, GLiNER, or any other heavy dependency. Heavy-stage behavior is
tested in their respective unit modules.
"""
from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest

from _system.scripts import ingest as ingest_mod
from _system.scripts.ingest import ingest_pdf, ingest_pdf_chapter
from _system.scripts.load_pdf import LocalPdfNoUsableOutline


# ---------------------------------------------------------------------------
# Synthetic PDF fixture
# ---------------------------------------------------------------------------


def _make_pdf(
    *,
    toc: list[list] | None,
    n_pages: int = 6,
    title: str = "Pdf Book",
    author: str = "Alice; Bob",
    creation_date: str = "D:20240115120000Z",
) -> bytes:
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
def split_pdf(tmp_path: Path) -> Path:
    """PDF with a 3-chapter outline."""
    data = _make_pdf(toc=[
        [1, "Intro", 1],
        [1, "Body", 3],
        [1, "Conclusion", 5],
    ])
    path = tmp_path / "book.pdf"
    path.write_bytes(data)
    return path


@pytest.fixture
def outline_less_pdf(tmp_path: Path) -> Path:
    """PDF with no embedded TOC."""
    data = _make_pdf(toc=None)
    path = tmp_path / "no_outline.pdf"
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Stage stubs (no LLMs / GLiNER / network)
# ---------------------------------------------------------------------------


def _stub_convert(*, paper_name, conn, force=False):
    conn.execute(
        "UPDATE papers SET markdown = ?, raw_html = NULL, status = 'converted' "
        "WHERE paper_name = ?",
        (f"# {paper_name}\n\nbody " * 30, paper_name),
    )
    conn.commit()
    from _system.scripts.convert_paper import ConvertResult
    return ConvertResult(
        paper_name=paper_name, status="converted", markdown_chars=300,
        figures=0, references=0,
        references_resolved_forward=0, references_resolved_backward=0,
    )


def _stub_classify(*, paper_name, conn, force=False, domain_override=None, **_):
    domain = domain_override or "philosophy"
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", (domain,))
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name) VALUES (?, ?)",
        (domain, "logic"),
    )
    conn.execute(
        "UPDATE papers SET domain = ?, collection = ?, status = 'classified' "
        "WHERE paper_name = ?",
        (domain, "logic", paper_name),
    )
    conn.execute(
        """
        DELETE FROM collections WHERE target_kind = 'paper' AND target_id = (
            SELECT id FROM papers WHERE paper_name = ?
        )
        """,
        (paper_name,),
    )
    conn.execute(
        """
        INSERT INTO collections (target_kind, target_id, domain, collection, is_primary)
        SELECT 'paper', id, ?, ?, 1 FROM papers WHERE paper_name = ?
        """,
        (domain, "logic", paper_name),
    )
    conn.commit()
    from _system.scripts.classify_paper import ClassifyResult
    return ClassifyResult(
        paper_name=paper_name, domain=domain, collections=("logic",),
        topics=(), needs_review=False, status="classified",
    )


def _stub_extract(*, paper_name, conn, force=False, **_):
    conn.execute(
        "UPDATE papers SET entity_count = 2, status = 'extracted' "
        "WHERE paper_name = ?",
        (paper_name,),
    )
    conn.commit()
    from _system.scripts.extract_entities import ExtractResult
    return ExtractResult(paper_name=paper_name, entity_count=2, status="extracted")


def _stub_index(*, paper_name, conn, force=False, **_):
    conn.execute(
        "UPDATE papers SET status = 'indexed' WHERE paper_name = ?",
        (paper_name,),
    )
    conn.commit()
    from _system.scripts.index_paper import IndexResult
    return IndexResult(paper_name=paper_name, section_count=1, status="indexed")


@pytest.fixture
def patched_pipeline(monkeypatch):
    monkeypatch.setattr(ingest_mod, "convert_stage", _stub_convert)
    monkeypatch.setattr(ingest_mod, "classify_paper_stage", _stub_classify)
    monkeypatch.setattr(ingest_mod, "extract_stage", _stub_extract)
    monkeypatch.setattr(ingest_mod, "index_stage", _stub_index)


# ---------------------------------------------------------------------------
# Orchestrator behavior
# ---------------------------------------------------------------------------


class TestIngestPdf:
    def test_creates_one_row_per_chapter(self, conn, split_pdf, patched_pipeline):
        summary = ingest_pdf(conn=conn, pdf_path=split_pdf)
        assert summary["kind"] == "pdf"
        assert summary["chapter_count"] == 3
        assert len(summary["chapters"]) == 3
        assert all(c["status"] == "indexed" for c in summary["chapters"])

        # Three papers rows, all with the chapter prefix and shared content_hash.
        rows = conn.execute(
            "SELECT paper_name, arxiv_id, content_hash FROM papers "
            "ORDER BY paper_name"
        ).fetchall()
        assert len(rows) == 3
        book_slug = summary["book_slug"]
        assert all(r[0].startswith(f"{book_slug}__ch") for r in rows)
        assert all(r[1].startswith(f"pdf:{summary['content_hash'][:12]}:ch") for r in rows)
        assert len({r[2] for r in rows}) == 1  # one content_hash

    def test_chapters_sort_in_toc_order(self, conn, split_pdf, patched_pipeline):
        ingest_pdf(conn=conn, pdf_path=split_pdf)
        rows = conn.execute(
            "SELECT paper_name FROM papers ORDER BY paper_name"
        ).fetchall()
        # zero-padded ch01/ch02/ch03 → lexicographic sort matches TOC order.
        assert "__ch01_" in rows[0][0]
        assert "__ch02_" in rows[1][0]
        assert "__ch03_" in rows[2][0]

    def test_no_split_creates_one_whole_book_row(self, conn, split_pdf, patched_pipeline):
        summary = ingest_pdf(conn=conn, pdf_path=split_pdf, no_split=True)
        assert summary["chapter_count"] == 1
        rows = conn.execute(
            "SELECT paper_name, arxiv_id FROM papers"
        ).fetchall()
        assert len(rows) == 1
        # Bare book_slug (no __chNN_ suffix), bare pdf:<hash> arxiv_id.
        assert "__ch" not in rows[0][0]
        assert ":" not in rows[0][1].removeprefix("pdf:")

    def test_raises_on_unusable_outline(self, conn, outline_less_pdf, patched_pipeline):
        with pytest.raises(LocalPdfNoUsableOutline):
            ingest_pdf(conn=conn, pdf_path=outline_less_pdf)
        # No rows should have been written.
        assert conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0] == 0

    def test_no_split_works_on_outline_less_pdf(self, conn, outline_less_pdf, patched_pipeline):
        # --no-split must NOT validate the outline.
        summary = ingest_pdf(conn=conn, pdf_path=outline_less_pdf, no_split=True)
        assert summary["chapter_count"] == 1

    def test_force_cascades_all_chapters(self, conn, split_pdf, patched_pipeline):
        ingest_pdf(conn=conn, pdf_path=split_pdf)
        before = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        assert before == 3

        ingest_pdf(conn=conn, pdf_path=split_pdf, force=True)
        after = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        # Force re-runs the whole loop; three rows remain (deleted + reinserted).
        assert after == 3

    def test_resume_skips_already_indexed_chapters(self, conn, split_pdf, patched_pipeline, monkeypatch):
        ingest_pdf(conn=conn, pdf_path=split_pdf)
        # Re-run without force — none of the stage stubs should fire.
        call_log: list[str] = []

        def assert_unused(*a, **k):
            call_log.append("called")
            raise AssertionError("downstream stage should not run on resume")

        monkeypatch.setattr(ingest_mod, "convert_stage", assert_unused)
        monkeypatch.setattr(ingest_mod, "classify_paper_stage", assert_unused)
        monkeypatch.setattr(ingest_mod, "extract_stage", assert_unused)
        monkeypatch.setattr(ingest_mod, "index_stage", assert_unused)

        summary = ingest_pdf(conn=conn, pdf_path=split_pdf)
        assert summary["chapter_count"] == 3
        assert call_log == []
        # All chapter envelopes still report indexed (from prior run).
        for ch in summary["chapters"]:
            assert ch["status"] == "indexed"

    def test_partial_resume_completes_missing_chapters(self, conn, split_pdf, patched_pipeline):
        # Run once: all chapters indexed.
        ingest_pdf(conn=conn, pdf_path=split_pdf)
        # Manually rewind the last chapter to FETCHED so resume picks it up.
        conn.execute(
            "UPDATE papers SET status = 'fetched' WHERE arxiv_id LIKE 'pdf:%:ch03'"
        )
        conn.commit()
        summary = ingest_pdf(conn=conn, pdf_path=split_pdf)
        # All three end at indexed; the partial-resume chapter advanced through
        # convert→classify→extract→index.
        statuses = [c["status"] for c in summary["chapters"]]
        assert statuses == ["indexed", "indexed", "indexed"]

    def test_progress_ticks_emitted(self, conn, split_pdf, patched_pipeline):
        events: list[tuple] = []
        ingest_pdf(
            conn=conn, pdf_path=split_pdf,
            progress=lambda *a: events.append(a),
        )
        # One "chapter i/N" tick per chapter + one "complete".
        chapter_ticks = [e for e in events if e[0].startswith("chapter ")]
        assert len(chapter_ticks) == 3
        assert events[-1][0] == "complete"

    def test_domain_override_propagates(self, conn, split_pdf, patched_pipeline):
        summary = ingest_pdf(
            conn=conn, pdf_path=split_pdf, domain="custom_domain",
        )
        # The classify stub honors domain_override.
        for ch in summary["chapters"]:
            assert ch["domain"] == "custom_domain"

    def test_chapter_failure_does_not_abort_book(self, conn, split_pdf, patched_pipeline, monkeypatch):
        # Make convert blow up only on the middle chapter.
        original = ingest_mod.convert_stage

        def flaky_convert(*, paper_name, conn, force=False):
            if "__ch02_" in paper_name:
                raise RuntimeError("boom on chapter 2")
            return original(paper_name=paper_name, conn=conn, force=force)

        monkeypatch.setattr(ingest_mod, "convert_stage", flaky_convert)
        summary = ingest_pdf(conn=conn, pdf_path=split_pdf)

        statuses = {ch.get("paper_name", ch.get("arxiv_id")): ch["status"]
                    for ch in summary["chapters"]}
        # One failed envelope, two indexed.
        assert sum(1 for s in statuses.values() if s == "indexed") == 2
        assert sum(1 for s in statuses.values() if s == "failed") == 1

    def test_missing_pdf_raises(self, conn, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_pdf(conn=conn, pdf_path=tmp_path / "nope.pdf")


# ---------------------------------------------------------------------------
# Manual chapter mode (--book-slug / --chapter-index)
# ---------------------------------------------------------------------------


def _make_chapter_pdf(tmp_path: Path, name: str, marker: str) -> Path:
    """One-page PDF used as a stand-in for a hand-sliced book chapter."""
    data = _make_pdf(
        toc=None, n_pages=2, title=f"Chapter for {marker}",
        author="Book Author", creation_date="D:20200101000000Z",
    )
    # Re-render with custom marker so the chapter PDFs have distinct content
    # (and therefore distinct sha256s).
    doc = pymupdf.open(stream=data, filetype="pdf")
    page = doc.new_page()  # add an extra page with the marker
    page.insert_text((72, 72), f"marker: {marker}")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    path = tmp_path / name
    path.write_bytes(buf.getvalue())
    return path


class TestIngestPdfChapter:
    def test_single_chapter_lands_in_slot(self, conn, tmp_path, patched_pipeline):
        ch = _make_chapter_pdf(tmp_path, "ch3.pdf", "wittgenstein-3")
        summary = ingest_pdf_chapter(
            conn=conn, pdf_path=ch,
            book_slug="tractatus_1922",
            chapter_index=3,
            chapter_title="Objects and States of Affairs",
        )
        assert summary["kind"] == "pdf_chapter"
        assert summary["book_slug"] == "tractatus_1922"
        assert summary["chapter_index"] == 3
        row = conn.execute(
            "SELECT paper_name, arxiv_id, status FROM papers"
        ).fetchone()
        assert row[0] == "tractatus_1922__ch03_objects_states_affairs"
        assert row[1].startswith("pdf:") and row[1].endswith(":ch03")
        assert row[2] == "indexed"

    def test_multiple_chapters_share_book_slug(self, conn, tmp_path, patched_pipeline):
        ch1 = _make_chapter_pdf(tmp_path, "a.pdf", "alpha")
        ch5 = _make_chapter_pdf(tmp_path, "b.pdf", "bravo")
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch1,
            book_slug="my_book", chapter_index=1, chapter_title="Intro",
        )
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch5,
            book_slug="my_book", chapter_index=5, chapter_title="Body",
        )
        rows = conn.execute(
            "SELECT paper_name FROM papers WHERE paper_name LIKE 'my_book__%' "
            "ORDER BY paper_name"
        ).fetchall()
        assert [r[0] for r in rows] == [
            "my_book__ch01_intro",
            "my_book__ch05_body",
        ]
        # Each chapter PDF has its own content_hash → distinct arxiv_id prefix.
        hashes = conn.execute(
            "SELECT DISTINCT content_hash FROM papers WHERE paper_name LIKE 'my_book__%'"
        ).fetchall()
        assert len(hashes) == 2

    def test_idempotent_resume(self, conn, tmp_path, patched_pipeline, monkeypatch):
        ch = _make_chapter_pdf(tmp_path, "ch.pdf", "x")
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch,
            book_slug="b_2020", chapter_index=2, chapter_title="T",
        )

        def boom(*a, **k):
            raise AssertionError("downstream stage should not run on resume")

        monkeypatch.setattr(ingest_mod, "convert_stage", boom)
        monkeypatch.setattr(ingest_mod, "classify_paper_stage", boom)
        monkeypatch.setattr(ingest_mod, "extract_stage", boom)
        monkeypatch.setattr(ingest_mod, "index_stage", boom)

        summary = ingest_pdf_chapter(
            conn=conn, pdf_path=ch,
            book_slug="b_2020", chapter_index=2, chapter_title="T",
        )
        assert summary["chapter"]["status"] == "indexed"

    def test_force_replaces_chapter_slot(self, conn, tmp_path, patched_pipeline):
        # Run once with file A under slot ch07.
        ch_a = _make_chapter_pdf(tmp_path, "a.pdf", "alpha")
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch_a,
            book_slug="b", chapter_index=7, chapter_title="OldTitle",
        )
        before_hash = conn.execute(
            "SELECT content_hash FROM papers WHERE paper_name = 'b__ch07_oldtitle'"
        ).fetchone()[0]

        # Re-run with a DIFFERENT file (different sha256) at the same slot,
        # under --force. The slot should be wiped + replaced.
        ch_b = _make_chapter_pdf(tmp_path, "b.pdf", "bravo")
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch_b,
            book_slug="b", chapter_index=7, chapter_title="NewTitle",
            force=True,
        )
        rows = conn.execute(
            "SELECT paper_name, content_hash FROM papers WHERE paper_name LIKE 'b__ch07%'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "b__ch07_newtitle"
        assert rows[0][1] != before_hash

    def test_chapter_title_falls_back_to_pdf_metadata(self, conn, tmp_path, patched_pipeline):
        ch = _make_chapter_pdf(tmp_path, "ch.pdf", "x")
        # No chapter_title → falls back to PDF title metadata (set in
        # _make_chapter_pdf to "Chapter for x" → tokens "chapter_x" after
        # stopword strip).
        summary = ingest_pdf_chapter(
            conn=conn, pdf_path=ch,
            book_slug="b", chapter_index=1, chapter_title=None,
        )
        assert summary["chapter"]["paper_name"].startswith("b__ch01_")

    def test_invalid_book_slug_rejected(self, conn, tmp_path):
        ch = _make_chapter_pdf(tmp_path, "ch.pdf", "x")
        with pytest.raises(ValueError, match="must match"):
            ingest_pdf_chapter(
                conn=conn, pdf_path=ch,
                book_slug="Has-Hyphens", chapter_index=1,
            )
        with pytest.raises(ValueError, match="must not contain '__'"):
            ingest_pdf_chapter(
                conn=conn, pdf_path=ch,
                book_slug="has__double", chapter_index=1,
            )

    def test_zero_chapter_index_rejected(self, conn, tmp_path):
        ch = _make_chapter_pdf(tmp_path, "ch.pdf", "x")
        with pytest.raises(ValueError, match=">= 1"):
            ingest_pdf_chapter(
                conn=conn, pdf_path=ch,
                book_slug="b", chapter_index=0,
            )

    def test_order_by_paper_name_returns_chapters_in_order(
        self, conn, tmp_path, patched_pipeline,
    ):
        # Ingest out of order; the zero-padded ch<NN> means lexicographic
        # sort still gives TOC order.
        for idx, marker in ((11, "k"), (2, "b"), (1, "a"), (10, "j")):
            ch = _make_chapter_pdf(tmp_path, f"ch{idx}.pdf", marker)
            ingest_pdf_chapter(
                conn=conn, pdf_path=ch,
                book_slug="big", chapter_index=idx,
                chapter_title=f"Chapter {marker}",
            )
        rows = conn.execute(
            "SELECT paper_name FROM papers WHERE paper_name LIKE 'big__%' "
            "ORDER BY paper_name"
        ).fetchall()
        indices = [r[0].split("__ch")[1][:2] for r in rows]
        assert indices == ["01", "02", "10", "11"]

    def test_rejects_chapter_index_above_99(self, conn, tmp_path):
        ch = _make_chapter_pdf(tmp_path, "ch.pdf", "x")
        with pytest.raises(ValueError, match=r"1\.\.99"):
            ingest_pdf_chapter(
                conn=conn, pdf_path=ch,
                book_slug="b", chapter_index=100,
            )

    def test_book_slug_mismatch_raises_without_force(
        self, conn, tmp_path, patched_pipeline,
    ):
        # Ingest a chapter file under 'old_book', then attempt to
        # re-ingest the SAME file under 'new_book' without --force.
        # arxiv_id is keyed off file hash so the existing row is found
        # but its paper_name lives in the old_book namespace — the
        # orchestrator must surface the mismatch instead of silently
        # resuming under old_book.
        ch = _make_chapter_pdf(tmp_path, "ch.pdf", "x")
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch,
            book_slug="old_book", chapter_index=1, chapter_title="Intro",
        )
        with pytest.raises(ValueError, match="different book_slug"):
            ingest_pdf_chapter(
                conn=conn, pdf_path=ch,
                book_slug="new_book", chapter_index=1, chapter_title="Intro",
            )
        # Old namespace untouched.
        rows = conn.execute(
            "SELECT paper_name FROM papers ORDER BY paper_name"
        ).fetchall()
        assert [r[0] for r in rows] == ["old_book__ch01_intro"]

    def test_book_slug_mismatch_relocates_with_force(
        self, conn, tmp_path, patched_pipeline,
    ):
        # Same starting state as above, but --force should clean up the
        # arxiv_id-matched row from the old namespace and land the new
        # row under new_book.
        ch = _make_chapter_pdf(tmp_path, "ch.pdf", "x")
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch,
            book_slug="old_book", chapter_index=1, chapter_title="Intro",
        )
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch,
            book_slug="new_book", chapter_index=1, chapter_title="Intro",
            force=True,
        )
        rows = conn.execute(
            "SELECT paper_name FROM papers ORDER BY paper_name"
        ).fetchall()
        # Old row gone, new row in the new namespace.
        assert [r[0] for r in rows] == ["new_book__ch01_intro"]

    def test_force_does_not_cross_book_delete_via_underscore_wildcard(
        self, conn, tmp_path, patched_pipeline,
    ):
        # book_slug='a_b' uses SQL LIKE wildcards if the literal '_' in
        # the slug is not escaped — pattern 'a_b__ch01_%' would otherwise
        # match 'axb__ch01_intro' (different book). Ingest both shapes
        # and confirm --force on one does NOT touch the other.
        ch_ab = _make_chapter_pdf(tmp_path, "ab.pdf", "ab")
        ch_axb = _make_chapter_pdf(tmp_path, "axb.pdf", "axb")
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch_ab,
            book_slug="a_b", chapter_index=1, chapter_title="Intro",
        )
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch_axb,
            book_slug="axb", chapter_index=1, chapter_title="Intro",
        )
        before = sorted(
            r[0] for r in conn.execute("SELECT paper_name FROM papers").fetchall()
        )
        assert before == ["a_b__ch01_intro", "axb__ch01_intro"]

        # --force on a_b/ch01 with a DIFFERENT file content must only
        # touch the a_b slot, leaving axb__ch01_intro intact.
        ch_ab2 = _make_chapter_pdf(tmp_path, "ab2.pdf", "ab-take-two")
        ingest_pdf_chapter(
            conn=conn, pdf_path=ch_ab2,
            book_slug="a_b", chapter_index=1, chapter_title="Redo",
            force=True,
        )
        after = sorted(
            r[0] for r in conn.execute("SELECT paper_name FROM papers").fetchall()
        )
        # a_b slot replaced (intro → redo), axb untouched.
        assert "axb__ch01_intro" in after
        assert "a_b__ch01_intro" not in after
        assert "a_b__ch01_redo" in after


class TestIngestPdfForceAndOutline:
    def test_force_does_not_delete_prior_rows_on_outline_failure(
        self, conn, tmp_path, patched_pipeline,
    ):
        # User runs `--pdf X` successfully (3 chapters land), then
        # later swaps X for a version with no usable outline and
        # re-runs with `--force`. The outline check must run BEFORE
        # the cascade so prior rows are preserved when discovery fails.
        good_pdf = tmp_path / "good.pdf"
        good_pdf.write_bytes(_make_pdf(toc=[
            [1, "Intro", 1],
            [1, "Body", 3],
            [1, "Conclusion", 5],
        ]))
        ingest_pdf(conn=conn, pdf_path=good_pdf)
        before = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        assert before == 3
        before_hash = conn.execute(
            "SELECT DISTINCT content_hash FROM papers"
        ).fetchone()[0]

        # Swap in a same-hash-prefix outline-less PDF by overwriting
        # with content that shares content_hash[:12] is hard to engineer;
        # instead, point at a totally unrelated outline-less PDF and
        # check the PRIOR good_pdf rows survive. (The fix's invariant is
        # 'discover-before-delete', not 'cross-PDF safety'.)
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(_make_pdf(toc=None))
        with pytest.raises(LocalPdfNoUsableOutline):
            ingest_pdf(conn=conn, pdf_path=bad_pdf, force=True)

        after = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        after_hash = conn.execute(
            "SELECT DISTINCT content_hash FROM papers"
        ).fetchone()[0]
        # The good rows must still be there with the same hash — the
        # outline-less ingest's --force did NOT touch unrelated content.
        assert after == 3
        assert after_hash == before_hash


class TestIngestPdfMixedModeRecovery:
    def test_recovery_prefers_chapter_row_over_whole_book(
        self, conn, tmp_path, patched_pipeline,
    ):
        # Synthesize a corrupted-state DB: a whole-book row at
        # pdf:<hash> and chapter rows at pdf:<hash>:chNN under a
        # DIFFERENT book_slug. The recovery query sorts arxiv_id ASC
        # so the bare 'pdf:<hash>' lands first — but the function
        # should prefer the chapter row's book_slug, not the stale
        # whole-book one.
        pdf = tmp_path / "book.pdf"
        pdf.write_bytes(_make_pdf(toc=[
            [1, "Intro", 1],
            [1, "Body", 3],
            [1, "Conclusion", 5],
        ]))
        # First, ingest with --no-split (one whole-book row).
        ingest_pdf(conn=conn, pdf_path=pdf, no_split=True)
        whole_book_name = conn.execute(
            "SELECT paper_name FROM papers"
        ).fetchone()[0]

        # Manually plant a chapter row under a DIFFERENT book_slug
        # (simulates a corrupted state from interleaved runs).
        existing_hash = conn.execute(
            "SELECT content_hash FROM papers"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, "
            "  abstract, pdf_url, status, content_hash, html_source, raw_html, "
            "  needs_review, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"pdf:{existing_hash[:12]}:ch01",
                "other_book__ch01_intro",
                "Intro", "[]", "2024-01-15", "", "file:///tmp/x.pdf",
                "fetched", existing_hash, "pdf_fallback", "stub", 0,
                "2024-01-15T00:00:00",
            ),
        )
        conn.commit()

        # Re-run ingest without --force; recovery should pick the chapter
        # row's book_slug 'other_book', NOT the whole-book row's slug.
        # (A warning is logged about the mixed-mode state — caplog can't
        # capture it because the Lodestone logger has propagate=False;
        # the behavioral assertion below is the load-bearing check.)
        summary = ingest_pdf(conn=conn, pdf_path=pdf)
        assert summary["book_slug"] == "other_book"


class TestIngestPdfSlugCollision:
    def test_one_slug_collision_does_not_abort_book(
        self, conn, split_pdf, patched_pipeline, monkeypatch,
    ):
        # Monkeypatch generate_chapter_slug to raise on chapter 2; the
        # rest of the book should still ingest, and chapter 2 should
        # appear as a {status: 'failed'} envelope (NOT propagate).
        from _system.scripts import ingest as ingest_mod
        original = ingest_mod.generate_chapter_slug

        def flaky(book_slug, idx, title, existing):
            if idx == 2:
                raise ValueError(
                    f"chapter slug collision unresolved: {book_slug}__ch02_x already in existing"
                )
            return original(book_slug, idx, title, existing)

        monkeypatch.setattr(ingest_mod, "generate_chapter_slug", flaky)
        summary = ingest_pdf(conn=conn, pdf_path=split_pdf)
        # 3 chapter envelopes overall: 1 + 2 (failed) + 3.
        statuses = [c["status"] for c in summary["chapters"]]
        assert statuses.count("indexed") == 2
        assert statuses.count("failed") == 1
        # Two rows landed; the failed chapter has no paper row.
        n_rows = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        assert n_rows == 2
