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
                [_span("BookRAG", "method")],
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
            run_inference=_static_inference([[_span("BookRAG", "method")]]),
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
            run_inference=_static_inference([[_span("BookRAG", "method")]]),
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
        inf = _static_inference([[_span("BookRAG", "method")] for _ in range(10)])
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

        # Every sub-chunk returns the same span; dedup by normalized name
        # should collapse them into one INSERT.
        def _dup_inf(text, labels, threshold):
            return [_span("BookRAG", "method")]

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
# Label voting: same name across labels resolves to one canonical
# ===========================================================================


class TestLabelVoting:
    """Regression: GLiNER2 labels the same entity inconsistently across
    mentions (``RRF`` as method/benchmark/dataset/...). The paper-wide
    label vote must consolidate these into a single canonical with the
    majority label, not one canonical per label.
    """

    def test_same_name_multiple_labels_one_canonical(
        self, conn, fake_embedder
    ):
        # 5 mentions of "RRF" labelled across 4 types: Method wins 2-1-1-1.
        # Each inference call returns one span so we can control counts.
        inf = _static_inference(
            [
                [_span("RRF", "method")],
                [_span("RRF", "software")],
                [_span("RRF", "method")],
                [_span("RRF", "benchmark")],
                [_span("RRF", "dataset")],
                [],
            ]
        )
        markdown = (
            "# S1\n\nbody one\n"
            "# S2\n\nbody two\n"
            "# S3\n\nbody three\n"
            "# S4\n\nbody four\n"
            "# S5\n\nbody five\n"
        )
        _seed_domain(conn)
        _seed_paper(conn, markdown=markdown)

        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=inf,
            embedder=fake_embedder,
        )

        # Exactly one canonical row for RRF (regardless of label diversity).
        canonicals = conn.execute(
            "SELECT canonical_name, entity_type FROM canonical_terms "
            "WHERE canonical_name = 'RRF'"
        ).fetchall()
        assert len(canonicals) == 1, (
            f"expected 1 RRF canonical, got {len(canonicals)}: {canonicals}"
        )
        # Majority vote winner is Method.
        assert canonicals[0][1] == "method"

        # Every entities row for RRF carries the voted type.
        entity_types = {
            row[0] for row in conn.execute(
                "SELECT DISTINCT entity_type FROM entities "
                "WHERE entity_name = 'RRF'"
            )
        }
        assert entity_types == {"method"}

    def test_schwartz_hearst_acronym_collapses_with_expansion(
        self, conn, fake_embedder
    ):
        """Acronym defined in paper text collapses with its expansion.

        If the markdown contains ``Reciprocal Rank Fusion (RRF)`` and
        later spans for both ``RRF`` and ``Reciprocal Rank Fusion``,
        the extract stage must produce ONE canonical (the long form)
        with ``RRF`` persisted as a ``term_aliases`` row.
        """
        markdown = (
            "# Background\n\n"
            "We use Reciprocal Rank Fusion (RRF) to combine rankings.\n"
            "# Method\n\nThe RRF formula is well known.\n"
            "# Results\n\nReciprocal Rank Fusion performs well.\n"
        )
        _seed_domain(conn)
        _seed_paper(conn, markdown=markdown)

        # One span per section: RRF / RRF / Reciprocal Rank Fusion.
        inf = _static_inference(
            [
                [_span("RRF", "method")],
                [_span("RRF", "method")],
                [_span("Reciprocal Rank Fusion", "method")],
                [],
            ]
        )
        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=inf,
            embedder=fake_embedder,
        )

        # ONE canonical — the long form.
        canonicals = conn.execute(
            "SELECT canonical_name FROM canonical_terms WHERE term_type='entity'"
        ).fetchall()
        names = [r[0] for r in canonicals]
        assert names == ["Reciprocal Rank Fusion"], (
            f"expected single long-form canonical; got {names}"
        )

        # RRF persisted as an alias with match_tier=0 (Schwartz-Hearst).
        aliases = conn.execute(
            "SELECT alias, match_tier FROM term_aliases ORDER BY alias"
        ).fetchall()
        assert ("RRF", 0) in aliases

        # Every entities row references the long form.
        ent_names = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT entity_name FROM entities "
                "WHERE paper_name = 'paper_name_2024'"
            )
        }
        assert ent_names == {"Reciprocal Rank Fusion"}

    def test_argmax_per_span_collapses_multi_label_rows(
        self, conn, fake_embedder
    ):
        """GLiNER2 emits one row per label that clears the threshold for
        a single span; the OLD count-based vote treated each row as its
        own +=1 vote, so one physical mention could cast three equal
        votes. Argmax-per-span collapses multi-label rows for the same
        ``(start, end)`` back to one vote — only the top-scoring label
        per span gets to vote.

        Two sub_chunks, each with the same ``(start=0, end=3)`` span
        returned under three labels. Without argmax-per-span, each
        sub_chunk casts 3 votes (software, method, dataset); method
        and dataset would accumulate 2 votes each across the two
        sub_chunks — tying software and letting first-seen pick the
        winner. With argmax-per-span, only software votes in each
        sub_chunk; software wins outright 2-0-0.
        """
        inf = _static_inference(
            [
                [
                    _span("DPR", "software", score=0.85, start=0, end=3),
                    _span("DPR", "method",   score=0.61, start=0, end=3),
                    _span("DPR", "dataset",  score=0.55, start=0, end=3),
                ],
                [
                    _span("DPR", "software", score=0.82, start=0, end=3),
                    _span("DPR", "method",   score=0.60, start=0, end=3),
                    _span("DPR", "dataset",  score=0.50, start=0, end=3),
                ],
                [],
            ]
        )
        markdown = "# S1\n\nalpha\n# S2\n\nbeta\n"
        _seed_domain(conn)
        _seed_paper(conn, markdown=markdown)

        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=inf,
            embedder=fake_embedder,
        )

        row = conn.execute(
            "SELECT entity_type, entity_type_score FROM canonical_terms "
            "WHERE canonical_name = 'DPR'"
        ).fetchone()
        assert row is not None
        assert row[0] == "software"
        # Score is the max observed score *of the winning label*.
        assert row[1] == pytest.approx(0.85)

    def test_majority_beats_single_high_score_outlier(
        self, conn, fake_embedder
    ):
        """Count-majority is robust to single high-confidence outliers:
        three method@0.65 mentions beat one software@0.95 mention, even
        though software scored higher. The stored score is the max of
        the *winning* label's mentions (0.65), NOT the paper-wide peak
        (0.95 for the losing software mention).
        """
        inf = _static_inference(
            [
                [_span("Foo", "method", score=0.65, start=0)],
                [_span("Foo", "software", score=0.95, start=0)],
                [_span("Foo", "method", score=0.64, start=0)],
                [_span("Foo", "method", score=0.63, start=0)],
                [],
            ]
        )
        markdown = (
            "# S1\n\na\n# S2\n\nb\n# S3\n\nc\n# S4\n\nd\n"
        )
        _seed_domain(conn)
        _seed_paper(conn, markdown=markdown)
        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=inf,
            embedder=fake_embedder,
        )
        row = conn.execute(
            "SELECT entity_type, entity_type_score FROM canonical_terms "
            "WHERE canonical_name = 'Foo'"
        ).fetchone()
        assert row == ("method", pytest.approx(0.65))

    def test_winning_score_persists_to_canonical(self, conn, fake_embedder):
        """Tier 5 writes the max-score-per-winning-label onto the new
        canonical row's entity_type_score.
        """
        inf = _static_inference(
            [
                [_span("NovelTerm", "method", score=0.72)],
                [],
            ]
        )
        _seed_domain(conn)
        _seed_paper(conn, markdown="# S\n\nbody\n")
        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=inf,
            embedder=fake_embedder,
        )
        row = conn.execute(
            "SELECT entity_type, entity_type_score FROM canonical_terms "
            "WHERE canonical_name = 'NovelTerm'"
        ).fetchone()
        assert row is not None
        assert row[0] == "method"
        assert row[1] == pytest.approx(0.72)

    def test_winning_score_is_max_across_mentions(self, conn, fake_embedder):
        """Same entity mentioned twice with different scores: the stored
        entity_type_score is the MAX of the two (not first or last)."""
        inf = _static_inference(
            [
                [_span("BookRAG", "method", score=0.65)],
                [_span("BookRAG", "method", score=0.88)],
                [_span("BookRAG", "method", score=0.70)],
                [],
            ]
        )
        markdown = "# S1\n\nalpha\n# S2\n\nbeta\n# S3\n\ngamma\n"
        _seed_domain(conn)
        _seed_paper(conn, markdown=markdown)
        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=inf,
            embedder=fake_embedder,
        )
        row = conn.execute(
            "SELECT entity_type, entity_type_score FROM canonical_terms "
            "WHERE canonical_name = 'BookRAG'"
        ).fetchone()
        assert row == ("method", pytest.approx(0.88))

    def test_second_paper_with_higher_score_flips_canonical(
        self, conn, fake_embedder
    ):
        """Two-paper simulation: paper 1 establishes ``DPR`` as method at 0.4;
        paper 2 extracts ``DPR`` as software at 0.85 — canonical flips,
        paper 1's entities row follows. Confirms end-to-end wiring from the
        vote machine through resolver._maybe_flip_entity_type.
        """
        _seed_domain(conn)
        _seed_paper(
            conn,
            paper_name="paper_1",
            arxiv_id="2401.11111",
            markdown="# S\n\nalpha\n",
        )
        _seed_paper(
            conn,
            paper_name="paper_2",
            arxiv_id="2401.22222",
            markdown="# S\n\nbeta\n",
        )

        extract(
            paper_name="paper_1",
            conn=conn,
            run_inference=_static_inference(
                [[_span("DPR", "method", score=0.62)], []]
            ),
            embedder=fake_embedder,
        )
        # Canonical is now method@0.62; paper_1's entities row is method.
        row = conn.execute(
            "SELECT entity_type, entity_type_score FROM canonical_terms "
            "WHERE canonical_name = 'DPR'"
        ).fetchone()
        assert row == ("method", pytest.approx(0.62))

        extract(
            paper_name="paper_2",
            conn=conn,
            run_inference=_static_inference(
                [[_span("DPR", "software", score=0.85)], []]
            ),
            embedder=fake_embedder,
        )

        row = conn.execute(
            "SELECT entity_type, entity_type_score FROM canonical_terms "
            "WHERE canonical_name = 'DPR'"
        ).fetchone()
        assert row == ("software", pytest.approx(0.85))

        # Paper 1's historical entities row was migrated.
        paper_1_type = conn.execute(
            "SELECT entity_type FROM entities WHERE paper_name = 'paper_1' "
            "AND entity_name = 'DPR'"
        ).fetchone()[0]
        assert paper_1_type == "software"

    def test_bm25_flip_across_papers_with_multi_mention_majority(
        self, conn, fake_embedder
    ):
        """Multi-mention majority vote in both papers, with paper 2's
        majority-winning label at a higher peak than paper 1's. This is
        the scenario from the design discussion:

        * Paper 1: BM25 mentioned 3x as method, 1x as software. Majority
          wins method; stored score = max(method_scores) = 0.70.
        * Paper 2: BM25 mentioned 3x as software, 1x as method. Majority
          wins software; winner score = max(software_scores) = 0.85.
        * 0.85 > 0.70 AND software != method → flip.
        """
        _seed_domain(conn)
        _seed_paper(
            conn,
            paper_name="paper_1",
            arxiv_id="2401.11111",
            markdown="# S1\n\na\n# S2\n\nb\n# S3\n\nc\n# S4\n\nd\n",
        )
        _seed_paper(
            conn,
            paper_name="paper_2",
            arxiv_id="2401.22222",
            markdown="# S1\n\na\n# S2\n\nb\n# S3\n\nc\n# S4\n\nd\n",
        )

        # Paper 1: method wins 3-1; peak method score = 0.70.
        extract(
            paper_name="paper_1",
            conn=conn,
            run_inference=_static_inference(
                [
                    [_span("BM25", "method", score=0.70, start=0)],
                    [_span("BM25", "method", score=0.65, start=0)],
                    [_span("BM25", "method", score=0.62, start=0)],
                    [_span("BM25", "software", score=0.95, start=0)],
                    [],
                ]
            ),
            embedder=fake_embedder,
        )
        row = conn.execute(
            "SELECT entity_type, entity_type_score FROM canonical_terms "
            "WHERE canonical_name = 'BM25'"
        ).fetchone()
        # Paper 1 establishes method@0.70 (NOT software@0.95 — the
        # software mention loses the majority vote and its score is
        # discarded along with it).
        assert row == ("method", pytest.approx(0.70))

        # Paper 2: software wins 3-1; peak software score = 0.85.
        extract(
            paper_name="paper_2",
            conn=conn,
            run_inference=_static_inference(
                [
                    [_span("BM25", "software", score=0.85, start=0)],
                    [_span("BM25", "software", score=0.80, start=0)],
                    [_span("BM25", "software", score=0.76, start=0)],
                    [_span("BM25", "method", score=0.68, start=0)],
                    [],
                ]
            ),
            embedder=fake_embedder,
        )

        # Cross-paper flip: paper 2's software@0.85 overturns paper 1's
        # method@0.70.
        row = conn.execute(
            "SELECT entity_type, entity_type_score FROM canonical_terms "
            "WHERE canonical_name = 'BM25'"
        ).fetchone()
        assert row == ("software", pytest.approx(0.85))

        # Paper 1's historical entities row was migrated to software.
        paper_1_type = conn.execute(
            "SELECT entity_type FROM entities WHERE paper_name = 'paper_1' "
            "AND entity_name = 'BM25'"
        ).fetchone()[0]
        assert paper_1_type == "software"

    def test_second_paper_with_lower_score_does_not_flip(
        self, conn, fake_embedder
    ):
        _seed_domain(conn)
        _seed_paper(
            conn,
            paper_name="paper_1",
            arxiv_id="2401.11111",
            markdown="# S\n\nalpha\n",
        )
        _seed_paper(
            conn,
            paper_name="paper_2",
            arxiv_id="2401.22222",
            markdown="# S\n\nbeta\n",
        )
        extract(
            paper_name="paper_1",
            conn=conn,
            run_inference=_static_inference(
                [[_span("DPR", "software", score=0.85)], []]
            ),
            embedder=fake_embedder,
        )
        extract(
            paper_name="paper_2",
            conn=conn,
            run_inference=_static_inference(
                [[_span("DPR", "method", score=0.70)], []]
            ),
            embedder=fake_embedder,
        )
        row = conn.execute(
            "SELECT entity_type, entity_type_score FROM canonical_terms "
            "WHERE canonical_name = 'DPR'"
        ).fetchone()
        assert row == ("software", pytest.approx(0.85))

    def test_vote_tiebreak_is_first_seen(self, conn, fake_embedder):
        """On vote ties, the label seen first in the paper wins
        (Counter.most_common preserves insertion order on ties).
        """
        inf = _static_inference(
            [
                [_span("Foo", "software")],  # first
                [_span("Foo", "method")],
                [],
            ]
        )
        markdown = "# S1\n\nbody one\n# S2\n\nbody two\n"
        _seed_domain(conn)
        _seed_paper(conn, markdown=markdown)

        extract(
            paper_name="paper_name_2024",
            conn=conn,
            run_inference=inf,
            embedder=fake_embedder,
        )

        canonicals = conn.execute(
            "SELECT canonical_name, entity_type FROM canonical_terms "
            "WHERE canonical_name = 'Foo'"
        ).fetchall()
        assert len(canonicals) == 1
        assert canonicals[0][1] == "software"


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

    @pytest.mark.parametrize(
        "label_word",
        [
            "method", "dataset", "metric", "model", "benchmark",
            "software", "system", "organization", "venue",
        ],
    )
    def test_label_word_rejected_after_normalize(self, label_word):
        """Label words are rejected regardless of case — ``normalize_term``
        lowercases before the ``_LABEL_WORDS`` check, so ``'Method'``,
        ``'METHOD'``, and ``'method'`` all match.
        """
        assert _is_garbage(label_word)
        assert _is_garbage(label_word.title())
        assert _is_garbage(label_word.upper())

    def test_empty_or_whitespace_rejected(self):
        assert _is_garbage("")
        assert _is_garbage("   ")

    @pytest.mark.parametrize(
        "name",
        [
            "Pipeline Latency Breakdown\n\nTable E3",
            "Foo\nBar",
            "Foo\r\nBar",
            "Foo\tBar",
        ],
    )
    def test_control_whitespace_rejected(self, name):
        """Spans that cross paragraph / row / item boundaries are dropped.

        A newline, carriage return, or tab inside an entity name means
        GLiNER2 concatenated across a structural boundary — the 'entity'
        is two unrelated things glued together, not a single concept.
        """
        assert _is_garbage(name)

    def test_normal_entity_not_rejected(self):
        assert not _is_garbage("BookRAG")
        assert not _is_garbage("GraphRAG")
        assert not _is_garbage("MMLU")
        # Multi-word entity names with regular single spaces are legitimate.
        assert not _is_garbage("Claude Opus 4")
        assert not _is_garbage("BEIR benchmarks")
        # Common entity-name shapes with internal punctuation / digits.
        assert not _is_garbage("BGE-small")
        assert not _is_garbage("ColBERTv2")
        assert not _is_garbage("NDCG@10")
        assert not _is_garbage("sqlite-vec")

    @pytest.mark.parametrize(
        "ref",
        [
            "Table 2", "Table 5a", "Table C1", "Table E4",
            "Figure 3", "Fig. 4", "Fig 4a",
            "Appendix C", "Appendix D",
            "Section 3.1", "Chapter 2",
            "Eq. 7", "Equation 12",
        ],
    )
    def test_structural_reference_rejected(self, ref):
        """Document-structure pointers (Table N, Figure N, Appendix X) are
        not entities — they're cross-references into the paper body.
        """
        assert _is_garbage(ref)

    @pytest.mark.parametrize(
        "q",
        [
            "+5.6%", "+8.3%", "-10%", "-2.5",
            "10K", "50K", "10.9 ms", "209 docs",
            "700 chunks per second", "30 papers",
            "1.5ms", "95%",
        ],
    )
    def test_quantity_value_rejected(self, q):
        """Raw measurements and value deltas are not entities."""
        assert _is_garbage(q)

    @pytest.mark.parametrize(
        "frag",
        ["NV-", "foo/", "bar:", "baz,", "qux@", "cat#"],
    )
    def test_dangling_punctuation_rejected(self, frag):
        """Spans ending in `-/:,@#` are GLiNER2 boundary truncations."""
        assert _is_garbage(frag)

    @pytest.mark.parametrize(
        "frag",
        ["+ Dedup", "- retry", "+foo", "-bar"],
    )
    def test_leading_sign_rejected(self, frag):
        """Entity names don't start with `+` or `-`; those are markdown-list
        markers or value-delta prefixes the quantity regex can't always match
        (e.g. ``+ Dedup`` — sign-space-word).
        """
        assert _is_garbage(frag)

    def test_over_length_rejected(self):
        """Spans longer than 60 chars are phrase captures, not entities."""
        long_name = "a" * 61
        assert _is_garbage(long_name)
        # 60 is the boundary — still accepted if otherwise clean.
        assert not _is_garbage("a" * 60)

    @pytest.mark.parametrize(
        "word",
        [
            "vector", "hybrid", "scoring", "scale", "mean", "median", "distance",
            "json", "pdf", "html", "docx", "xml", "yaml", "csv",
            # Case-insensitive — matches regardless of capitalisation.
            "Vector", "JSON", "HTML",
        ],
    )
    def test_generic_noun_and_file_format_rejected(self, word):
        """Concept nouns and file formats are not entities when standalone."""
        assert _is_garbage(word)


class TestGarbageGateEndToEnd:
    def test_paper_with_only_garbage_extracts_zero(self, tmp_db_with_paper, fake_embedder):
        # Every span is garbage → garbage gate rejects them all.
        inf = _static_inference(
            [
                [
                    _span("We", "method"),
                    _span("Table", "method"),
                    _span("1", "metric"),
                    _span("Method", "method"),
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
                return [_span("BookRAG", "method", start=text.index("BookRAG"))]
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
                return [_span("BookRAG", "method", start=text.index("BookRAG"))]
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
                [_span("BookRAG", "method", start=15)],
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
            run_inference=_static_inference([[_span("book rag", "method")]]),
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
                    _span("BookRAG", "method"),
                    _span("MMLU", "benchmark"),
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
                    [[_span("BookRAG", "method"), _span("MysteryThing", "NotALabel")]]
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
        # per_label["method"] = 0.55; a Method span at 0.50 must be filtered.
        inf = _static_inference(
            [[_span("BookRAG", "method", score=0.50)]]
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
            [[_span("BookRAG", "method", score=0.60)]]
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
                "method": [
                    {"text": "BookRAG", "confidence": 0.92, "start": 10, "end": 17},
                    {"text": "GraphRAG", "confidence": 0.85, "start": 30, "end": 38},
                ],
                "dataset": [],
            }
        }
        spans = _flatten_gliner_output(raw)
        assert len(spans) == 2
        method_spans = [s for s in spans if s["label"] == "method"]
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
            _flatten_gliner_output({"entities": {"method": "not a list"}})
        with pytest.raises(ValueError, match="item must be a dict"):
            _flatten_gliner_output({"entities": {"method": ["not a dict"]}})


# ===========================================================================
# entity_count update
# ===========================================================================


class TestEntityCountUpdate:
    def test_papers_entity_count_matches_row_count(self, tmp_db_with_paper, fake_embedder):
        inf = _static_inference(
            [
                [_span("BookRAG", "method"), _span("GraphRAG", "method")],
                [_span("MMLU", "benchmark")],
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
        # Give each span a unique start offset — production GLiNER2 spans
        # sit at distinct positions in the source text, and the vote
        # machine's argmax-per-span dedup keys on (start, end) so any
        # collision in the fixture would silently collapse spans.
        texts_garbage = ["We", "Table", "Figure", "Using", "this", "that"]
        texts_good = ["BookRAG", "GraphRAG", "RAPTOR", "LlamaRAG", "TreeRAG"]
        spans = []
        pos = 0
        for text in texts_garbage + texts_good:
            spans.append(_span(text, "method", start=pos))
            pos += len(text) + 1
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
        inf = _static_inference([[_span("We", "method")]])

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
            lambda text, labels, threshold: [_span("BookRAG", "method")],
        )
        # Stub the default tokenizer so sub_chunk doesn't trigger a model load.
        # Offsets-based contract: return one (start, end) pair per whitespace-
        # delimited token in the input.
        import re as _re
        monkeypatch.setattr(
            ee,
            "_default_tokenize",
            lambda t: [(m.start(), m.end()) for m in _re.finditer(r"\S+", t)],
        )
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
    ``fastino/gliner2-large-v1`` (``validate_models.py`` primes it).
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
