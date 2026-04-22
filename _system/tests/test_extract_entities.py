"""Unit tests for _system/scripts/extract_entities.py.

GLiNER2 inference is fully mocked via the ``run_inference`` test seam — no
torch / HF cache load during the default ``uv run pytest`` run. The resolver
tier 5 path is exercised with a fake ``Embedder`` so sentence-transformers is
also never loaded.

The real-model smoke test carries ``@pytest.mark.slow`` and is excluded by
default.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.schemas.paper_metadata import PaperStatus
from _system.scripts import extract_entities as ee
from _system.scripts.extract_entities import (
    _ACRONYM_ALLOWLIST,
    MarkdownMissing,
    PaperNotFound,
    StatusTooLow,
    UnknownStatusError,
    _description_for,
    _flatten_gliner_output,
    _is_garbage,
    extract,
)


# ---------------------------------------------------------------------------
# Fake embedder — tier 5 inserts need an Embedder; we never exercise semantic
# similarity here so a deterministic 384-dim vector is enough.
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    def __init__(self) -> None:
        self.embed_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        v = [0.0] * 384
        v[hash(text) % 384] = 1.0
        return v

    def embed_batch(self, texts):  # pragma: no cover
        return [self.embed(t) for t in texts]


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


_DEFAULT_MARKDOWN = (
    "# Introduction\n"
    "\n"
    "We introduce BookRAG for long-context question answering. "
    "BookRAG uses a tree structure. Evaluation uses the MMLU benchmark.\n"
    "\n"
    "# Method\n"
    "\n"
    "Our approach leverages RAPTOR and GraphRAG as baselines. "
    "We compare against Llama2 on the SQuAD dataset.\n"
)


def _seed_domain(conn: sqlite3.Connection, name: str = "rag") -> None:
    conn.execute(
        "INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
        (name, "retrieval-augmented generation"),
    )


def _seed_paper(
    conn: sqlite3.Connection,
    *,
    paper_name: str = "paper_name_2024",
    arxiv_id: str = "2401.00001",
    status: str = PaperStatus.CLASSIFIED.value,
    abstract: str = "We propose BookRAG.",
    markdown: str | None = _DEFAULT_MARKDOWN,
    domain: str | None = "rag",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, html_source, ingested_at, status, markdown,
            domain, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id,
            paper_name,
            "A Title",
            '["A. Author"]',
            "2024-01-01",
            abstract,
            f"https://arxiv.org/pdf/{arxiv_id}",
            "arxiv",
            "2024-01-01T00:00:00+00:00",
            status,
            markdown,
            domain,
            0,
        ),
    )
    return cur.lastrowid


@pytest.fixture
def tmp_db_with_paper(conn: sqlite3.Connection) -> sqlite3.Connection:
    _seed_domain(conn)
    _seed_paper(conn)
    return conn


# ---------------------------------------------------------------------------
# Inference stub helpers
# ---------------------------------------------------------------------------


def _span(text: str, label: str, *, score: float = 0.9, start: int = 0, end: int | None = None) -> dict:
    return {
        "text": text,
        "label": label,
        "score": score,
        "start": start,
        "end": end if end is not None else start + len(text),
    }


def _static_inference(spans_per_call: list[list[dict]]):
    """Return an inference fn that emits ``spans_per_call[call_index]`` in order."""
    calls = {"n": 0, "seen_texts": [], "seen_labels": [], "seen_thresholds": []}

    def _fn(text: str, label_descriptions: dict, threshold: float):
        idx = calls["n"]
        calls["n"] += 1
        calls["seen_texts"].append(text)
        calls["seen_labels"].append(dict(label_descriptions))
        calls["seen_thresholds"].append(threshold)
        if idx < len(spans_per_call):
            return list(spans_per_call[idx])
        return []

    _fn.calls = calls  # type: ignore[attr-defined]
    return _fn


def _always_empty_inference():
    return _static_inference([])


# ===========================================================================
# Pipeline / status
# ===========================================================================


class TestPipelineStatus:
    def test_classified_advances_to_extracted(self, tmp_db_with_paper, fake_embedder):
        inf = _static_inference(
            [
                [_span("BookRAG", "Method")],
                [],
            ]
        )
        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=inf,
            embedder=fake_embedder,
        )
        status = tmp_db_with_paper.execute(
            "SELECT status FROM papers WHERE paper_name = ?", ("paper_name_2024",)
        ).fetchone()[0]
        assert status == PaperStatus.EXTRACTED.value

    def test_indexed_without_force_raises(self, conn, fake_embedder):
        _seed_domain(conn)
        _seed_paper(conn, status=PaperStatus.INDEXED.value)
        with pytest.raises(StatusTooLow):
            extract(
                paper_name="paper_name_2024",
                conn=conn,
                run_inference=_always_empty_inference(),
                embedder=fake_embedder,
            )

    def test_failed_html_without_force_raises_with_terminal_hint(self, conn, fake_embedder):
        _seed_domain(conn)
        _seed_paper(conn, status=PaperStatus.FAILED_HTML.value)
        with pytest.raises(StatusTooLow) as exc_info:
            extract(
                paper_name="paper_name_2024",
                conn=conn,
                run_inference=_always_empty_inference(),
                embedder=fake_embedder,
            )
        assert "failed_html" in str(exc_info.value).lower()

    def test_force_bypasses_status_guard(self, conn, fake_embedder):
        _seed_domain(conn)
        _seed_paper(conn, status=PaperStatus.INDEXED.value)
        extract(
            paper_name="paper_name_2024",
            conn=conn,
            force=True,
            run_inference=_static_inference([[_span("BookRAG", "Method")]]),
            embedder=fake_embedder,
        )
        status = conn.execute(
            "SELECT status FROM papers WHERE paper_name = ?", ("paper_name_2024",)
        ).fetchone()[0]
        assert status == PaperStatus.EXTRACTED.value

    def test_paper_not_found_raises(self, tmp_db_with_paper, fake_embedder):
        with pytest.raises(PaperNotFound):
            extract(
                paper_name="no_such_paper",
                conn=tmp_db_with_paper,
                run_inference=_always_empty_inference(),
                embedder=fake_embedder,
            )

    def test_markdown_missing_raises(self, conn, fake_embedder):
        _seed_domain(conn)
        _seed_paper(conn, markdown=None)
        with pytest.raises(MarkdownMissing):
            extract(
                paper_name="paper_name_2024",
                conn=conn,
                run_inference=_always_empty_inference(),
                embedder=fake_embedder,
            )

    def test_rerun_clears_existing_entities(self, tmp_db_with_paper, fake_embedder):
        paper_id = tmp_db_with_paper.execute(
            "SELECT id FROM papers WHERE paper_name = ?", ("paper_name_2024",)
        ).fetchone()[0]
        # Seed 3 stale entities for this paper
        for i in range(3):
            tmp_db_with_paper.execute(
                """
                INSERT INTO entities (paper_id, domain, paper_name, entity_name,
                                      entity_type, source_breadcrumb, description)
                VALUES (?, 'rag', 'paper_name_2024', ?, 'method', '# Stale', 'old')
                """,
                (paper_id, f"stale_entity_{i}"),
            )

        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=_static_inference([[_span("BookRAG", "Method")]]),
            embedder=fake_embedder,
        )
        rows = tmp_db_with_paper.execute(
            "SELECT entity_name FROM entities WHERE paper_id = ?", (paper_id,)
        ).fetchall()
        names = {r[0] for r in rows}
        assert not any(n.startswith("stale_entity_") for n in names)
        assert len(names) == 1


# ===========================================================================
# Section splitter integration
# ===========================================================================


class TestSectionSplitterIntegration:
    def test_gliner_input_has_no_breadcrumb(self, tmp_db_with_paper, fake_embedder):
        inf = _static_inference([[_span("BookRAG", "Method")] for _ in range(10)])
        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=inf,
            embedder=fake_embedder,
        )
        # At least one inference call happened; every text passed must NOT
        # start with a breadcrumb line ("# ..." or "# A > ## B").
        assert inf.calls["n"] > 0
        for text in inf.calls["seen_texts"]:
            first_line = text.splitlines()[0] if text else ""
            assert not first_line.startswith("#"), (
                f"inference received breadcrumb-prefixed text: {first_line!r}"
            )

    def test_sub_chunking_triggers_for_long_section(self, conn, fake_embedder):
        # Build a single section with > 350 space-separated tokens so the
        # default word-count tokenizer triggers sub-chunking.
        body_words = " ".join(f"word{i}" for i in range(500))
        long_markdown = f"# Method\n\n{body_words}\n"
        _seed_domain(conn)
        _seed_paper(conn, markdown=long_markdown)

        inf = _static_inference([[] for _ in range(20)])
        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=inf,
            embedder=fake_embedder,
        )
        # One section = at least two inference calls due to sub-chunking.
        assert inf.calls["n"] >= 2
        # Upper bound catches runaway step-size bugs in sub_chunk: 500 tokens
        # at 350-max with 20-overlap (step=330) needs ceil((500-20)/330)+1 = 3
        # chunks, definitely fewer than 10.
        assert inf.calls["n"] < 10

    def test_duplicate_spans_across_overlapping_subchunks_deduped(self, conn, fake_embedder):
        body_words = " ".join(f"word{i}" for i in range(500))
        long_markdown = f"# Method\n\n{body_words}\n"
        _seed_domain(conn)
        _seed_paper(conn, markdown=long_markdown)

        # Every sub-chunk returns the same span; dedup by (entity_type, normalize_term)
        # should collapse them into one INSERT.
        def _dup_inf(text, labels, threshold):
            return [_span("BookRAG", "Method")]

        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=_dup_inf,
            embedder=fake_embedder,
        )
        rows = conn.execute(
            "SELECT entity_name FROM entities WHERE paper_name = 'paper_name_2024'"
        ).fetchall()
        assert len(rows) == 1


# ===========================================================================
# Garbage gate
# ===========================================================================


class TestGarbageGate:
    @pytest.mark.parametrize(
        "name", ["We", "we", "Table", "TABLE", "Figure", "Using", "this", "these", "that", "it", "However"]
    )
    def test_stoplist_words_rejected(self, name):
        assert _is_garbage(name)

    def test_pure_numeric_rejected(self):
        assert _is_garbage("1")
        assert _is_garbage("2024")

    @pytest.mark.parametrize("acronym", sorted(_ACRONYM_ALLOWLIST))
    def test_acronym_allowlist_kept(self, acronym):
        assert not _is_garbage(acronym)
        assert not _is_garbage(acronym.lower())

    def test_short_non_acronym_rejected(self):
        assert _is_garbage("We")  # 2 chars, not in allowlist
        assert _is_garbage("xy")
        assert _is_garbage("a")

    @pytest.mark.parametrize("label_word", ["Method", "Dataset", "Metric", "Model", "Technique", "Benchmark"])
    def test_label_word_rejected_after_normalize(self, label_word):
        assert _is_garbage(label_word)
        assert _is_garbage(label_word.lower())

    def test_empty_or_whitespace_rejected(self):
        assert _is_garbage("")
        assert _is_garbage("   ")

    def test_normal_entity_not_rejected(self):
        assert not _is_garbage("BookRAG")
        assert not _is_garbage("GraphRAG")
        assert not _is_garbage("MMLU")


class TestGarbageGateEndToEnd:
    def test_paper_with_only_garbage_extracts_zero(self, tmp_db_with_paper, fake_embedder):
        # Every span is garbage → garbage gate rejects them all.
        inf = _static_inference(
            [
                [
                    _span("We", "Method"),
                    _span("Table", "Method"),
                    _span("1", "Metric"),
                    _span("Method", "Method"),
                ],
                [],
            ]
        )
        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=inf,
            embedder=fake_embedder,
        )
        count = tmp_db_with_paper.execute(
            "SELECT entity_count FROM papers WHERE paper_name = 'paper_name_2024'"
        ).fetchone()[0]
        assert count == 0


# ===========================================================================
# Zero-entity safety
# ===========================================================================


class TestZeroEntitySafety:
    def test_zero_entities_advances_status_and_logs(self, tmp_db_with_paper, fake_embedder, caplog):
        logger = logging.getLogger("lodestone.scripts.extract_entities")
        logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger="lodestone.scripts.extract_entities"):
                extract(
                    paper_name="paper_name_2024",
                    conn=tmp_db_with_paper,
                    run_inference=_always_empty_inference(),
                    embedder=fake_embedder,
                )
        finally:
            logger.removeHandler(caplog.handler)

        row = tmp_db_with_paper.execute(
            "SELECT status, entity_count FROM papers WHERE paper_name = 'paper_name_2024'"
        ).fetchone()
        assert row[0] == PaperStatus.EXTRACTED.value
        assert row[1] == 0
        assert any(
            "0 entities extracted" in r.getMessage() for r in caplog.records
        ), [r.getMessage() for r in caplog.records]


# ===========================================================================
# source_breadcrumb
# ===========================================================================


class TestSourceBreadcrumb:
    def test_same_entity_different_breadcrumbs_yield_distinct_rows(self, conn, fake_embedder):
        # Duplicate "Setup" subsection under two different parents.
        md = (
            "# Method\n\n"
            "## Setup\n\n"
            "The Setup uses BookRAG tree.\n\n"
            "# Experiments\n\n"
            "## Setup\n\n"
            "Experiment Setup also uses BookRAG.\n"
        )
        _seed_domain(conn)
        _seed_paper(conn, markdown=md)

        def _inf(text, labels, threshold):
            if "BookRAG" in text:
                return [_span("BookRAG", "Method", start=text.index("BookRAG"))]
            return []

        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=_inf,
            embedder=fake_embedder,
        )
        rows = conn.execute(
            "SELECT entity_name, source_breadcrumb FROM entities "
            "WHERE paper_name = 'paper_name_2024' ORDER BY source_breadcrumb"
        ).fetchall()
        # Parent chunks + child chunks both contain BookRAG, so dedup per
        # section gives one row each with distinct source_breadcrumb values.
        breadcrumbs = {r[1] for r in rows}
        assert len(breadcrumbs) >= 2
        # Every row holds the same canonical entity_name
        assert {r[0] for r in rows} == {"BookRAG"}

    def test_source_breadcrumb_populated_with_full_chain(self, conn, fake_embedder):
        md = (
            "# Method\n\n"
            "## Architecture\n\n"
            "We use BookRAG in this subsection.\n"
        )
        _seed_domain(conn)
        _seed_paper(conn, markdown=md)

        def _inf(text, labels, threshold):
            if "BookRAG" in text:
                return [_span("BookRAG", "Method", start=text.index("BookRAG"))]
            return []

        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=_inf,
            embedder=fake_embedder,
        )
        breadcrumbs = [
            r[0]
            for r in conn.execute(
                "SELECT source_breadcrumb FROM entities WHERE paper_name = 'paper_name_2024'"
            )
        ]
        # Child section's breadcrumb is "# Method > ## Architecture".
        assert any(" > " in bc for bc in breadcrumbs)


# ===========================================================================
# Description extraction
# ===========================================================================


class TestDescriptionExtraction:
    def test_uses_nearest_sentence_boundary(self):
        text = (
            "Sentence one is here. "
            "BookRAG is the method we propose. "
            "Sentence three wraps up."
        )
        offset = text.index("BookRAG")
        desc = _description_for(text, offset)
        # Description MUST start at the sentence boundary, not mid-sentence.
        assert desc.startswith("BookRAG"), (
            f"description should start at sentence boundary; got {desc!r}"
        )
        # Description ends at the sentence terminator (no leak into next sentence).
        assert "propose" in desc
        assert "Sentence one" not in desc
        assert "Sentence three" not in desc

    def test_capped_at_240_chars(self):
        # A single 500-char sentence with no internal boundary.
        text = "X" * 500
        desc = _description_for(text, 250)
        assert len(desc) <= 240

    def test_falls_back_to_raw_window_when_no_boundary(self):
        # Continuous text with no sentence boundary → falls back to raw window.
        text = "alpha beta gamma delta epsilon " * 20
        offset = len(text) // 2
        desc = _description_for(text, offset)
        # Description is non-empty and within cap.
        assert desc
        assert len(desc) <= 240

    def test_empty_text_returns_empty(self):
        assert _description_for("", 0) == ""

    def test_description_written_to_db_matches(self, tmp_db_with_paper, fake_embedder):
        # Sanity check that _description_for's output reaches the DB description column.
        inf = _static_inference(
            [
                [_span("BookRAG", "Method", start=15)],
                [],
            ]
        )
        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=inf,
            embedder=fake_embedder,
        )
        descs = [
            r[0]
            for r in tmp_db_with_paper.execute(
                "SELECT description FROM entities WHERE paper_name = 'paper_name_2024'"
            )
        ]
        assert descs
        assert all(d is not None and 0 < len(d) <= 240 for d in descs)


# ===========================================================================
# Resolver integration
# ===========================================================================


class TestResolverIntegration:
    def test_entity_name_is_canonical_not_raw_span(self, tmp_db_with_paper, fake_embedder):
        # Seed canonical "BookRAG" so the raw "book rag" span resolves to it.
        tmp_db_with_paper.execute(
            """
            INSERT INTO canonical_terms (domain, term_type, entity_type,
                                         canonical_name, first_seen_in)
            VALUES ('rag', 'entity', 'method', 'BookRAG', 'seed')
            """
        )
        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=_static_inference([[_span("book rag", "Method")]]),
            embedder=fake_embedder,
        )
        row = tmp_db_with_paper.execute(
            "SELECT entity_name FROM entities WHERE paper_name = 'paper_name_2024'"
        ).fetchone()
        assert row is not None
        assert row[0] == "BookRAG"

    def test_resolver_invoked_with_entity_term_type_and_correct_entity_type(
        self, tmp_db_with_paper, fake_embedder, monkeypatch
    ):
        captured: list[dict] = []
        real_resolve = ee.resolve

        def _spy_resolve(conn, raw, **kwargs):
            captured.append({"raw": raw, **kwargs})
            return real_resolve(conn, raw, **kwargs)

        monkeypatch.setattr(ee, "resolve", _spy_resolve)

        inf = _static_inference(
            [
                [
                    _span("BookRAG", "Method"),
                    _span("MMLU", "Benchmark"),
                ],
                [],
            ]
        )
        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=inf,
            embedder=fake_embedder,
        )
        assert captured, "resolve() was not invoked"
        for call in captured:
            assert call["term_type"] == "entity"
            assert call["source_paper"] == "paper_name_2024"
            assert call["domain"] == "rag"
        # At least one call for each EntityType we fed in
        entity_types = {c["entity_type"] for c in captured}
        assert "method" in entity_types
        assert "benchmark" in entity_types


# ===========================================================================
# Unknown label handling
# ===========================================================================


class TestUnknownLabel:
    def test_unknown_label_logs_warning_and_skips(self, tmp_db_with_paper, fake_embedder, caplog):
        logger = logging.getLogger("lodestone.scripts.extract_entities")
        logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING, logger="lodestone.scripts.extract_entities"):
                inf = _static_inference(
                    [[_span("BookRAG", "Method"), _span("MysteryThing", "NotALabel")]]
                )
                extract(
                    paper_name="paper_name_2024",
                    conn=tmp_db_with_paper,
                    run_inference=inf,
                    embedder=fake_embedder,
                )
        finally:
            logger.removeHandler(caplog.handler)

        rows = tmp_db_with_paper.execute(
            "SELECT entity_name FROM entities WHERE paper_name = 'paper_name_2024'"
        ).fetchall()
        names = {r[0] for r in rows}
        # MysteryThing must not have been written.
        assert "MysteryThing" not in names
        # Something was logged at WARNING level about the unknown label.
        warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("NotALabel" in m for m in warning_msgs), warning_msgs


# ===========================================================================
# Per-label threshold
# ===========================================================================


class TestPerLabelThreshold:
    def test_span_below_per_label_threshold_is_dropped(self, tmp_db_with_paper, fake_embedder):
        # per_label["Method"] = 0.55; a Method span at 0.50 must be filtered.
        inf = _static_inference(
            [[_span("BookRAG", "Method", score=0.50)]]
        )
        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=inf,
            embedder=fake_embedder,
        )
        count = tmp_db_with_paper.execute(
            "SELECT COUNT(*) FROM entities WHERE paper_name = 'paper_name_2024'"
        ).fetchone()[0]
        assert count == 0

    def test_span_above_per_label_threshold_is_kept(self, tmp_db_with_paper, fake_embedder):
        inf = _static_inference(
            [[_span("BookRAG", "Method", score=0.60)]]
        )
        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=inf,
            embedder=fake_embedder,
        )
        count = tmp_db_with_paper.execute(
            "SELECT COUNT(*) FROM entities WHERE paper_name = 'paper_name_2024'"
        ).fetchone()[0]
        assert count == 1


# ===========================================================================
# _flatten_gliner_output
# ===========================================================================


class TestFlattenGlinerOutput:
    def test_flatten_dict_shape(self):
        raw = {
            "entities": {
                "Method": [
                    {"text": "BookRAG", "confidence": 0.92, "start": 10, "end": 17},
                    {"text": "GraphRAG", "confidence": 0.85, "start": 30, "end": 38},
                ],
                "Dataset": [],
            }
        }
        spans = _flatten_gliner_output(raw)
        assert len(spans) == 2
        method_spans = [s for s in spans if s["label"] == "Method"]
        assert len(method_spans) == 2
        assert method_spans[0]["text"] == "BookRAG"
        assert method_spans[0]["score"] == pytest.approx(0.92)
        assert method_spans[0]["start"] == 10

    def test_flatten_missing_entities_key_returns_empty(self):
        # GLiNER2 returns {} or {"entities": {}} for a no-entity text; both map
        # to [] without raising.
        assert _flatten_gliner_output({}) == []
        assert _flatten_gliner_output({"entities": {}}) == []
        # Missing-entities-key case (other unrelated keys present) still returns [].
        assert _flatten_gliner_output({"other": "stuff"}) == []

    def test_flatten_raises_on_non_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _flatten_gliner_output(None)
        with pytest.raises(ValueError, match="must be a dict"):
            _flatten_gliner_output([])

    def test_flatten_raises_on_malformed_entities_shape(self):
        with pytest.raises(ValueError, match="entities"):
            _flatten_gliner_output({"entities": "not a dict"})
        with pytest.raises(ValueError, match="must be a list"):
            _flatten_gliner_output({"entities": {"Method": "not a list"}})
        with pytest.raises(ValueError, match="item must be a dict"):
            _flatten_gliner_output({"entities": {"Method": ["not a dict"]}})


# ===========================================================================
# entity_count update
# ===========================================================================


class TestEntityCountUpdate:
    def test_papers_entity_count_matches_row_count(self, tmp_db_with_paper, fake_embedder):
        inf = _static_inference(
            [
                [_span("BookRAG", "Method"), _span("GraphRAG", "Method")],
                [_span("MMLU", "Benchmark")],
            ]
        )
        extract(
            paper_name="paper_name_2024",
            conn=tmp_db_with_paper,
            run_inference=inf,
            embedder=fake_embedder,
        )
        count_from_row = tmp_db_with_paper.execute(
            "SELECT entity_count FROM papers WHERE paper_name = 'paper_name_2024'"
        ).fetchone()[0]
        count_from_table = tmp_db_with_paper.execute(
            "SELECT COUNT(*) FROM entities WHERE paper_name = 'paper_name_2024'"
        ).fetchone()[0]
        assert count_from_row == count_from_table
        assert count_from_row == 3


# ===========================================================================
# Reject-rate WARNING and unknown status
# ===========================================================================


class TestRejectRateWarning:
    def test_warns_when_majority_rejected_and_sample_sufficient(
        self, tmp_db_with_paper, fake_embedder, caplog
    ):
        # 6 garbage + 5 good = 11 total, ~55% rejected → WARNING fires.
        spans = [
            _span("We", "Method"),
            _span("Table", "Method"),
            _span("Figure", "Method"),
            _span("Using", "Method"),
            _span("this", "Method"),
            _span("that", "Method"),
            _span("BookRAG", "Method"),
            _span("GraphRAG", "Method"),
            _span("RAPTOR", "Method"),
            _span("LlamaRAG", "Method"),
            _span("TreeRAG", "Method"),
        ]
        inf = _static_inference([spans, []])

        logger = logging.getLogger("lodestone.scripts.extract_entities")
        logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING, logger="lodestone.scripts.extract_entities"):
                extract(
                    paper_name="paper_name_2024",
                    conn=tmp_db_with_paper,
                    run_inference=inf,
                    embedder=fake_embedder,
                )
        finally:
            logger.removeHandler(caplog.handler)

        warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("exceeds 50%" in m for m in warns), warns

    def test_no_warn_below_min_samples(self, tmp_db_with_paper, fake_embedder, caplog):
        # 1 garbage span = 100% reject, but n_total < 10 → no WARNING.
        inf = _static_inference([[_span("We", "Method")]])

        logger = logging.getLogger("lodestone.scripts.extract_entities")
        logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.WARNING, logger="lodestone.scripts.extract_entities"):
                extract(
                    paper_name="paper_name_2024",
                    conn=tmp_db_with_paper,
                    run_inference=inf,
                    embedder=fake_embedder,
                )
        finally:
            logger.removeHandler(caplog.handler)

        warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("exceeds" in m and "garbage" in m for m in warns), warns


class TestUnknownStatus:
    def test_unknown_status_raises_dedicated_error(self, conn, fake_embedder):
        _seed_domain(conn)
        _seed_paper(conn, status="not_a_real_status")
        with pytest.raises(UnknownStatusError):
            extract(
                paper_name="paper_name_2024",
                conn=conn,
                run_inference=_always_empty_inference(),
                embedder=fake_embedder,
            )


# ===========================================================================
# CLI
# ===========================================================================


class TestCLI:
    def test_cli_prints_json_summary(self, tmp_path: Path, monkeypatch, capsys):
        db_path = tmp_path / "lodestone.db"
        c = get_conn(db_path)
        init_db(c)
        _seed_domain(c)
        _seed_paper(c)
        c.close()

        # Stub inference so no model loads
        monkeypatch.setattr(
            ee,
            "_default_inference",
            lambda text, labels, threshold: [_span("BookRAG", "Method")],
        )
        # Stub the default tokenizer too so sub_chunk doesn't trigger model load.
        monkeypatch.setattr(ee, "_default_tokenize", str.split)
        # Also stub the real embedder so we don't load sentence-transformers.
        monkeypatch.setattr(ee, "Embedder", _FakeEmbedder)

        ee._main(["--paper", "paper_name_2024", "--db", str(db_path)])
        out = capsys.readouterr().out.strip()
        # GLiNER2 may print a banner to stdout if the model gets loaded
        # inadvertently; the JSON payload is the last line. Parse that.
        payload = json.loads(out.splitlines()[-1])
        assert payload["paper_name"] == "paper_name_2024"
        assert payload["status"] == PaperStatus.EXTRACTED.value
        assert payload["entity_count"] >= 1


# ===========================================================================
# Real-model smoke test
# ===========================================================================


@pytest.mark.slow
def test_real_gliner_extracts_method_entity(tmp_db_with_paper, fake_embedder):
    """Smoke test: real GLiNER2 weights extract at least one Method entity.

    Skipped unless ``-m slow`` is passed. Requires the HF cache to contain
    ``fastino/gliner2-base-v1`` (``validate_models.py`` primes it).
    """
    extract(
        paper_name="paper_name_2024",
        conn=tmp_db_with_paper,
        embedder=fake_embedder,
    )
    count = tmp_db_with_paper.execute(
        "SELECT COUNT(*) FROM entities WHERE paper_name = 'paper_name_2024' "
        "AND entity_type = 'method'"
    ).fetchone()[0]
    assert count >= 1
