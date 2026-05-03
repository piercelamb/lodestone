"""Unit tests for _system/scripts/ingest.py.

Every stage function is mocked via ``unittest.mock.patch`` so section 14 tests
never hit the network, LLM, or ML models. The ``conn`` fixture is reused from
``conftest.py``; the ``--force`` cascade tests seed a real DB directly.
"""
from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.schemas.paper_metadata import PaperMetadata, PaperStatus
from _system.scripts import ingest
from _system.scripts.classify_paper import classify as _real_classify
from _system.scripts.convert_paper import convert as _real_convert
from _system.scripts.extract_entities import extract as _real_extract
from _system.scripts.fetch_paper import fetch as _real_fetch
from _system.scripts.index_paper import index_one as _real_index_one


# ---------------------------------------------------------------------------
# arxiv_id parsing
# ---------------------------------------------------------------------------


def test_parse_arxiv_id_abs_url_preserves_version():
    assert ingest.parse_arxiv_id("https://arxiv.org/abs/2301.12345v2") == "2301.12345v2"


def test_parse_arxiv_id_pdf_url_preserves_version():
    assert ingest.parse_arxiv_id("https://arxiv.org/pdf/2301.12345v2") == "2301.12345v2"


def test_parse_arxiv_id_bare_no_version():
    assert ingest.parse_arxiv_id("2301.12345") == "2301.12345"


def test_parse_arxiv_id_bare_with_version():
    assert ingest.parse_arxiv_id("2301.12345v3") == "2301.12345v3"


def test_parse_arxiv_id_pdf_suffix_stripped():
    assert ingest.parse_arxiv_id("https://arxiv.org/pdf/2301.12345.pdf") == "2301.12345"


def test_parse_arxiv_id_old_form_preserved():
    assert ingest.parse_arxiv_id("hep-th/9901001") == "hep-th/9901001"
    assert ingest.parse_arxiv_id("hep-th/9901001v2") == "hep-th/9901001v2"


def test_parse_arxiv_id_malformed_raises():
    with pytest.raises(ValueError, match="ftp://bogus"):
        ingest.parse_arxiv_id("ftp://bogus/whatever")


def test_parse_arxiv_id_empty_raises():
    with pytest.raises(ValueError):
        ingest.parse_arxiv_id("")


# ---------------------------------------------------------------------------
# Helpers for mocked-stage tests
# ---------------------------------------------------------------------------


def _seed_paper(
    conn: sqlite3.Connection,
    arxiv_id: str = "2301.12345",
    paper_name: str = "alice_2024_thing",
    status: PaperStatus = PaperStatus.FETCHED,
    needs_review: bool = False,
) -> int:
    """Insert a minimal papers row and return its id."""
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, ingested_at, status, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id, paper_name, "Title", "[\"A\"]", "2024-01-01", "Abs",
            f"https://arxiv.org/pdf/{arxiv_id}", "2024-01-02T00:00:00+00:00",
            status.value, int(needs_review),
        ),
    )
    return cur.lastrowid


class _StageRecorder:
    """Collects (stage_name, kwargs) for every mocked stage call and stubs
    their side effects on the shared conn."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def _record(self, stage: str, **kwargs):
        self.calls.append((stage, dict(kwargs)))

    def fetch(self, **kwargs):
        self._record("fetch", **kwargs)
        conn = kwargs["conn"]
        arxiv_id = kwargs["arxiv_id"]
        # Upsert a minimal FETCHED row so the orchestrator can resolve paper_name.
        existing = conn.execute(
            "SELECT id FROM papers WHERE arxiv_id = ?", (arxiv_id,)
        ).fetchone()
        if existing is None:
            _seed_paper(conn, arxiv_id=arxiv_id, paper_name=f"slug_{arxiv_id}")
        else:
            conn.execute(
                "UPDATE papers SET status = ? WHERE id = ?",
                (PaperStatus.FETCHED.value, existing[0]),
            )
        return PaperMetadata(
            arxiv_id=arxiv_id,
            paper_name=f"slug_{arxiv_id}",
            title="Title",
            authors='["A"]',
            date="2024-01-01",
            abstract="Abs",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            status=PaperStatus.FETCHED,
            ingested_at="2024-01-02T00:00:00+00:00",
        )

    def convert(self, **kwargs):
        self._record("convert", **kwargs)
        conn = kwargs["conn"]
        conn.execute(
            "UPDATE papers SET status = ? WHERE paper_name = ?",
            (PaperStatus.CONVERTED.value, kwargs["paper_name"]),
        )
        return {"status": "converted"}

    def classify(self, **kwargs):
        self._record("classify", **kwargs)
        conn = kwargs["conn"]
        conn.execute(
            "UPDATE papers SET status = ? WHERE paper_name = ?",
            (PaperStatus.CLASSIFIED.value, kwargs["paper_name"]),
        )
        return {"status": "classified"}

    def extract(self, **kwargs):
        self._record("extract", **kwargs)
        conn = kwargs["conn"]
        conn.execute(
            "UPDATE papers SET status = ? WHERE paper_name = ?",
            (PaperStatus.EXTRACTED.value, kwargs["paper_name"]),
        )
        return {"status": "extracted"}

    def index(self, **kwargs):
        self._record("index", **kwargs)
        conn = kwargs["conn"]
        conn.execute(
            "UPDATE papers SET status = ? WHERE paper_name = ?",
            (PaperStatus.INDEXED.value, kwargs["paper_name"]),
        )
        return {"status": "indexed"}


@pytest.fixture
def rec():
    return _StageRecorder()


@pytest.fixture
def patched_stages(rec):
    """Patch every stage function on the ingest module."""

    def _fetch_repo(**kwargs):
        rec._record("fetch_repo", **kwargs)
        conn = kwargs["conn"]
        conn.execute(
            "UPDATE papers SET status = ? WHERE paper_name = ?",
            (PaperStatus.REPO_FETCHED.value, kwargs["paper_name"]),
        )
        return None

    with patch.object(ingest, "fetch_stage", side_effect=rec.fetch), \
         patch.object(ingest, "convert_stage", side_effect=rec.convert), \
         patch.object(ingest, "classify_stage", side_effect=rec.classify), \
         patch.object(ingest, "extract_stage", side_effect=rec.extract), \
         patch.object(ingest, "index_stage", side_effect=rec.index), \
         patch.object(ingest, "fetch_repo_stage", side_effect=_fetch_repo):
        yield rec


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------


def test_fresh_db_runs_all_six_stages(conn, patched_stages):
    ingest.ingest(conn=conn, arxiv_id="2301.00001", force=False, domain=None)
    stages = [c[0] for c in patched_stages.calls]
    assert stages == [
        "fetch", "convert", "classify", "extract", "index", "fetch_repo",
    ]


def test_resume_from_fetched_skips_fetch(conn, patched_stages):
    _seed_paper(conn, arxiv_id="2301.00002", paper_name="slug_2301.00002",
                status=PaperStatus.FETCHED)
    ingest.ingest(conn=conn, arxiv_id="2301.00002", force=False, domain=None)
    stages = [c[0] for c in patched_stages.calls]
    assert stages == ["convert", "classify", "extract", "index", "fetch_repo"]


def test_resume_from_converted_skips_fetch_convert(conn, patched_stages):
    _seed_paper(conn, arxiv_id="2301.00003", paper_name="slug_2301.00003",
                status=PaperStatus.CONVERTED)
    ingest.ingest(conn=conn, arxiv_id="2301.00003", force=False, domain=None)
    stages = [c[0] for c in patched_stages.calls]
    assert stages == ["classify", "extract", "index", "fetch_repo"]


def test_resume_from_classified_runs_extract_index(conn, patched_stages):
    _seed_paper(conn, arxiv_id="2301.00004", paper_name="slug_2301.00004",
                status=PaperStatus.CLASSIFIED)
    ingest.ingest(conn=conn, arxiv_id="2301.00004", force=False, domain=None)
    stages = [c[0] for c in patched_stages.calls]
    assert stages == ["extract", "index", "fetch_repo"]


def test_resume_from_extracted_runs_index_only(conn, patched_stages):
    _seed_paper(conn, arxiv_id="2301.00005", paper_name="slug_2301.00005",
                status=PaperStatus.EXTRACTED)
    ingest.ingest(conn=conn, arxiv_id="2301.00005", force=False, domain=None)
    stages = [c[0] for c in patched_stages.calls]
    assert stages == ["index", "fetch_repo"]


def test_resume_from_indexed_runs_fetch_repo_only(conn, patched_stages):
    """Previously-INDEXED papers advance to fetch_repo on next ingest run —
    natural backfill, intended side effect of adding the new stage."""
    _seed_paper(conn, arxiv_id="2301.00006", paper_name="slug_2301.00006",
                status=PaperStatus.INDEXED)
    ingest.ingest(conn=conn, arxiv_id="2301.00006", force=False, domain=None)
    stages = [c[0] for c in patched_stages.calls]
    assert stages == ["fetch_repo"]


def test_repo_fetched_no_force_is_noop(conn, patched_stages):
    _seed_paper(conn, arxiv_id="2301.00006b", paper_name="slug_2301.00006b",
                status=PaperStatus.REPO_FETCHED)
    summary = ingest.ingest(
        conn=conn, arxiv_id="2301.00006b", force=False, domain=None
    )
    assert patched_stages.calls == []
    assert summary["status"] == "repo_fetched"


def test_failed_repo_no_force_is_noop(conn, patched_stages):
    _seed_paper(conn, arxiv_id="2301.00006c", paper_name="slug_2301.00006c",
                status=PaperStatus.FAILED_REPO)
    summary = ingest.ingest(
        conn=conn, arxiv_id="2301.00006c", force=False, domain=None
    )
    assert patched_stages.calls == []
    assert summary["status"] == "failed_repo"


def test_failed_html_no_force_is_noop_with_hint(conn, patched_stages, caplog):
    import logging
    _seed_paper(conn, arxiv_id="2301.00007", paper_name="slug_2301.00007",
                status=PaperStatus.FAILED_HTML)
    logger = logging.getLogger("lodestone.scripts.ingest")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="lodestone.scripts.ingest"):
            summary = ingest.ingest(
                conn=conn, arxiv_id="2301.00007", force=False, domain=None
            )
    finally:
        logger.removeHandler(caplog.handler)
    assert patched_stages.calls == []
    assert summary["status"] == "failed_html"
    assert any("--force" in r.message for r in caplog.records), (
        f"expected --force hint in logs; got {[r.message for r in caplog.records]}"
    )


def test_failed_html_with_force_cascades_then_fetches(conn, patched_stages):
    _seed_paper(conn, arxiv_id="2301.00008", paper_name="slug_2301.00008",
                status=PaperStatus.FAILED_HTML)
    ingest.ingest(conn=conn, arxiv_id="2301.00008", force=True, domain=None)
    # Cascade wiped the row; fetch saw no existing arxiv_id and ran the pipeline.
    stages = [c[0] for c in patched_stages.calls]
    assert stages == [
        "fetch", "convert", "classify", "extract", "index", "fetch_repo",
    ]


def test_force_on_indexed_cascades_then_runs_all(conn, patched_stages):
    _seed_paper(conn, arxiv_id="2301.00009", paper_name="slug_2301.00009",
                status=PaperStatus.INDEXED)
    ingest.ingest(conn=conn, arxiv_id="2301.00009", force=True, domain=None)
    stages = [c[0] for c in patched_stages.calls]
    assert stages == [
        "fetch", "convert", "classify", "extract", "index", "fetch_repo",
    ]


# ---------------------------------------------------------------------------
# --force cascade: real DB, no mocks for the delete itself
# ---------------------------------------------------------------------------


def _seed_full_paper(conn: sqlite3.Connection, arxiv_id: str, paper_name: str,
                     domain: str = "rag") -> int:
    """Seed one paper with rows in all cascade-affected tables + FTS entries.

    Returns the new paper_id.
    """
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", (domain,))
    paper_id = _seed_paper(conn, arxiv_id=arxiv_id, paper_name=paper_name,
                           status=PaperStatus.INDEXED)
    conn.execute(
        "UPDATE papers SET domain = ?, collection = ? WHERE id = ?",
        (domain, "tree-search", paper_id),
    )
    # sections (FTS5)
    conn.execute(
        "INSERT INTO sections (paper_id, domain, paper_name, section_title, section_level, body) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (paper_id, domain, paper_name, "Method", 1,
         "distinctive_section_marker_abc"),
    )
    # paper_topics
    conn.execute(
        "INSERT INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
        (paper_id, domain, "tree retrieval"),
    )
    # figures
    conn.execute(
        "INSERT INTO figures (paper_id, figure_number, caption, image, mime_type) "
        "VALUES (?, ?, ?, ?, ?)",
        (paper_id, 1, "Fig 1", b"\x89PNG\r\n\x1a\n", "image/png"),
    )
    # Canonical taxonomy + a synonym row in term_aliases. The Widget
    # canonical is shared across all callers — INSERT OR IGNORE so
    # multiple papers can seed the same canonical without UNIQUE
    # violations. Under the synonym-index regime the alias must differ
    # from the canonical, so we seed ``Widget_alt``.
    conn.execute(
        "INSERT OR IGNORE INTO canonical_terms (domain, term_type, entity_type, "
        " canonical_name, first_seen_in) VALUES (?, ?, ?, ?, ?)",
        (domain, "entity", "method", "Widget", paper_name),
    )
    term_id = conn.execute(
        "SELECT id FROM canonical_terms WHERE canonical_name = ? AND domain = ?",
        ("Widget", domain),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO term_aliases "
        " (term_id, alias, source_paper, match_tier) "
        "VALUES (?, ?, ?, ?)",
        (term_id, "Widget_alt", paper_name, 2),
    )
    # ``ingest._summary`` reads ``papers.entity_count`` directly under
    # the synonym-index regime (no JOIN against term_aliases). Every
    # seeded paper carries one canonical, so set the column to 1.
    conn.execute(
        "UPDATE papers SET entity_count = 1 WHERE id = ?",
        (paper_id,),
    )
    # term_embeddings PK is term_id; only the first seed inserts.
    existing_emb = conn.execute(
        "SELECT 1 FROM term_embeddings WHERE term_id = ?", (term_id,)
    ).fetchone()
    if existing_emb is None:
        vec = [0.0] * 384
        vec[0] = 1.0
        import struct
        blob = struct.pack(f"{len(vec)}f", *vec)
        conn.execute(
            "INSERT INTO term_embeddings (term_id, embedding, term_type, entity_type, domain) "
            "VALUES (?, ?, ?, ?, ?)",
            (term_id, blob, "entity", "method", domain),
        )
    return paper_id


def test_force_cascade_deletes_paper_and_children(conn):
    paper_id = _seed_full_paper(conn, "2301.11111", "paper_to_wipe")
    # Seed code_files and readmes_fts rows so the cascade has something
    # to clean up.
    conn.execute(
        "INSERT INTO code_files (paper_id, path, language, size_bytes, content) "
        "VALUES (?, 'README.md', 'markdown', 5, 'hi\n')",
        (paper_id,),
    )
    conn.execute(
        "INSERT INTO readmes_fts (paper_id, domain, paper_name, path, content) "
        "VALUES (?, 'rag', 'paper_to_wipe', 'README.md', 'hi')",
        (paper_id,),
    )
    ingest._force_delete_paper(conn, paper_id=paper_id)
    assert conn.execute(
        "SELECT COUNT(*) FROM papers WHERE id = ?", (paper_id,)
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM sections WHERE paper_name = ?", ("paper_to_wipe",)
    ).fetchone()[0] == 0
    for tbl in ("paper_topics", "figures", "code_files"):
        assert conn.execute(
            f"SELECT COUNT(*) FROM {tbl} WHERE paper_id = ?", (paper_id,)
        ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM readmes_fts WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0] == 0
    # term_aliases keys by paper_name (TEXT) — the cascade wipes per-paper
    # appearance rows alongside the paper.
    assert conn.execute(
        "SELECT COUNT(*) FROM term_aliases WHERE source_paper = ?",
        ("paper_to_wipe",),
    ).fetchone()[0] == 0


def test_force_cascade_clears_sections_fts_for_paper(conn):
    paper_id = _seed_full_paper(conn, "2301.22222", "paper_xyz")
    ingest._force_delete_paper(conn, paper_id=paper_id)
    rows = conn.execute(
        "SELECT COUNT(*) FROM sections WHERE sections MATCH ?",
        ("distinctive_section_marker_abc",),
    ).fetchone()[0]
    assert rows == 0


def test_force_cascade_preserves_entity_canonicals(conn):
    """Cascade preserves entity canonicals (``canonical_terms`` rows of
    term_type='entity') and their ``term_embeddings``. Entities are out
    of scope for orphan-GC under the synonym-index regime — tier-1
    mentions leave no per-paper trace, so substantiation can't be proven.
    ``term_aliases`` is per-paper (keyed by source_paper), so this paper's
    alias rows ARE wiped; a second paper's aliases referencing the same
    canonical must survive."""
    paper_id = _seed_full_paper(conn, "2301.44444", "paper_def")
    other_paper_id = _seed_full_paper(conn, "2301.44445", "paper_other")
    terms_before = conn.execute("SELECT COUNT(*) FROM canonical_terms").fetchone()[0]
    other_aliases_before = conn.execute(
        "SELECT COUNT(*) FROM term_aliases WHERE source_paper = ?",
        ("paper_other",),
    ).fetchone()[0]
    emb_before = conn.execute("SELECT COUNT(*) FROM term_embeddings").fetchone()[0]

    ingest._force_delete_paper(conn, paper_id=paper_id)

    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms"
    ).fetchone()[0] == terms_before
    # The deleted paper's alias rows are gone; the other paper's are intact.
    assert conn.execute(
        "SELECT COUNT(*) FROM term_aliases WHERE source_paper = ?",
        ("paper_def",),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM term_aliases WHERE source_paper = ?",
        ("paper_other",),
    ).fetchone()[0] == other_aliases_before
    assert conn.execute(
        "SELECT COUNT(*) FROM term_embeddings"
    ).fetchone()[0] == emb_before


def test_force_cascade_gcs_orphan_topic_and_collection_canonicals(conn):
    """Cascade GCs topic and collection canonicals whose only binding was
    the deleted paper, alongside their satellites in ``terms_fts``,
    ``term_embeddings``, ``term_aliases``, and the first-class
    ``collections`` registry. A topic canonical also bound by another
    paper survives; same for a collection canonical."""
    paper_id = _seed_full_paper(conn, "2301.55555", "paper_solo", domain="rag")
    keep_id = _seed_full_paper(conn, "2301.55556", "paper_keep", domain="rag")
    # _seed_full_paper sets papers.collection='tree-search' but doesn't
    # touch the first-class registry; mirror what classify_paper would
    # have done so we can assert a referenced registry row survives.
    conn.execute(
        "INSERT OR IGNORE INTO collections (domain, name, description) "
        "VALUES (?, ?, NULL)",
        ("rag", "tree-search"),
    )

    # Seed a topic canonical bound only to paper_solo (orphans on delete).
    conn.execute(
        "INSERT INTO canonical_terms (domain, term_type, entity_type, "
        " canonical_name, first_seen_in) VALUES (?, ?, '', ?, ?)",
        ("rag", "topic", "solo_topic_xyz", "paper_solo"),
    )
    conn.execute(
        "INSERT INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
        (paper_id, "rag", "solo_topic_xyz"),
    )
    solo_topic_id = conn.execute(
        "SELECT id FROM canonical_terms WHERE canonical_name = ? AND term_type = 'topic'",
        ("solo_topic_xyz",),
    ).fetchone()[0]

    # Seed a topic canonical bound to BOTH papers (must survive).
    conn.execute(
        "INSERT INTO canonical_terms (domain, term_type, entity_type, "
        " canonical_name, first_seen_in) VALUES (?, ?, '', ?, ?)",
        ("rag", "topic", "shared_topic_abc", "paper_solo"),
    )
    conn.execute(
        "INSERT INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
        (paper_id, "rag", "shared_topic_abc"),
    )
    conn.execute(
        "INSERT INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
        (keep_id, "rag", "shared_topic_abc"),
    )

    # Seed a collection canonical + registry row. _seed_full_paper sets
    # papers.collection='tree-search' for both papers; only this paper's
    # collection is unique. Switch paper_solo to a unique collection.
    conn.execute(
        "UPDATE papers SET collection = ? WHERE id = ?",
        ("solo-cluster", paper_id),
    )
    conn.execute(
        "INSERT INTO collections (domain, name, description) VALUES (?, ?, NULL)",
        ("rag", "solo-cluster"),
    )
    conn.execute(
        "INSERT INTO canonical_terms (domain, term_type, entity_type, "
        " canonical_name, first_seen_in) VALUES (?, ?, '', ?, ?)",
        ("rag", "collection", "solo-cluster", "paper_solo"),
    )
    solo_coll_id = conn.execute(
        "SELECT id FROM canonical_terms WHERE canonical_name = ? "
        "AND term_type = 'collection'",
        ("solo-cluster",),
    ).fetchone()[0]

    ingest._force_delete_paper(conn, paper_id=paper_id)

    # Orphan topic canonical gone.
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (solo_topic_id,),
    ).fetchone()[0] == 0
    # Shared topic canonical survives.
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms WHERE canonical_name = ?",
        ("shared_topic_abc",),
    ).fetchone()[0] == 1
    # Orphan collection canonical + registry row gone.
    assert conn.execute(
        "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (solo_coll_id,),
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM collections WHERE name = ?", ("solo-cluster",),
    ).fetchone()[0] == 0
    # The shared 'tree-search' registry row from _seed_full_paper survives
    # because paper_keep still references it.
    assert conn.execute(
        "SELECT COUNT(*) FROM collections WHERE name = ?", ("tree-search",),
    ).fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Stage dispatch contract
# ---------------------------------------------------------------------------


def test_stage_functions_share_same_conn(conn, patched_stages):
    ingest.ingest(conn=conn, arxiv_id="2301.55551", force=False, domain=None)
    conn_ids = {id(c[1]["conn"]) for c in patched_stages.calls}
    assert len(conn_ids) == 1
    assert conn_ids == {id(conn)}


@pytest.mark.parametrize("stage_fn", [
    _real_fetch, _real_convert, _real_classify, _real_extract, _real_index_one,
])
def test_stage_function_signatures_are_keyword_only(stage_fn):
    """Every stage function exposes only ``KEYWORD_ONLY`` parameters.

    Guards against a silent regression where someone re-introduces a
    positional arg and the orchestrator still works at call-time.
    """
    sig = inspect.signature(stage_fn)
    non_keyword = [
        p.name for p in sig.parameters.values()
        if p.kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    assert not non_keyword, (
        f"{stage_fn.__qualname__} has non-keyword-only params: {non_keyword}"
    )


def test_stage_functions_get_keyword_args_only_at_runtime(conn, patched_stages):
    """Sanity check the orchestrator never passes positional args either."""
    ingest.ingest(conn=conn, arxiv_id="2301.55552", force=False, domain=None)
    for stage_name, kwargs in patched_stages.calls:
        assert "conn" in kwargs, f"{stage_name} missing conn kwarg"


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------


def test_summary_contains_required_keys(conn, patched_stages):
    summary = ingest.ingest(conn=conn, arxiv_id="2301.66666", force=False, domain=None)
    required = {"paper_name", "arxiv_id", "status", "needs_review",
                "section_count", "entity_count", "figure_count"}
    assert required.issubset(summary.keys())


def test_summary_needs_review_reflects_flag(conn, patched_stages):
    # Fresh pipeline: mocks don't flip needs_review, so it should be False.
    summary = ingest.ingest(conn=conn, arxiv_id="2301.67001", force=False, domain=None)
    assert summary["needs_review"] is False

    # Pre-flag the paper with needs_review=1; ingest's no-op INDEXED path
    # should report it.
    _seed_paper(conn, arxiv_id="2301.67002", paper_name="slug_2301.67002",
                status=PaperStatus.INDEXED, needs_review=True)
    summary2 = ingest.ingest(conn=conn, arxiv_id="2301.67002", force=False, domain=None)
    assert summary2["needs_review"] is True


def test_summary_counts_match_db(conn, patched_stages):
    # Seed a paper that's already INDEXED so the no-op path takes counts.
    paper_id = _seed_full_paper(conn, "2301.67777", "counts_paper")
    # ``papers.entity_count`` is the authoritative count under the
    # synonym-index regime; bump the seed's count to 2 to model a
    # second distinct canonical for this paper.
    conn.execute(
        "UPDATE papers SET entity_count = 2 WHERE id = ?",
        (paper_id,),
    )
    conn.execute(
        "INSERT INTO figures (paper_id, figure_number, caption, image, mime_type) "
        "VALUES (?, ?, ?, ?, ?)",
        (paper_id, 2, "Fig 2", b"\x89PNG", "image/png"),
    )
    conn.execute(
        "INSERT INTO sections (paper_id, domain, paper_name, section_title, section_level, body) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (paper_id, "rag", "counts_paper", "Results", 1, "body"),
    )

    summary = ingest.ingest(conn=conn, arxiv_id="2301.67777", force=False, domain=None)

    # section_count: seed added 1; extra 1 = 2
    assert summary["section_count"] == 2
    # entity_count comes from papers.entity_count directly.
    assert summary["entity_count"] == 2
    # figure_count: seed added 1; extra 1 = 2
    assert summary["figure_count"] == 2


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def test_check_models_runs_before_init_db(tmp_path, monkeypatch):
    calls: list[str] = []

    def fake_check_models():
        calls.append("check_models")
        raise RuntimeError("models missing")

    def fake_init_db(c):
        calls.append("init_db")

    monkeypatch.setattr(ingest, "check_models", fake_check_models)
    monkeypatch.setattr(ingest, "init_db", fake_init_db)

    db_path = tmp_path / "lodestone.db"

    with pytest.raises(RuntimeError, match="models missing"):
        ingest.main([
            "--url", "https://arxiv.org/abs/2301.00001",
            "--db", str(db_path),
        ])

    assert calls == ["check_models"]
    assert not db_path.exists(), "no DB file should be created when check_models fails"


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_requires_url():
    with pytest.raises(SystemExit):
        ingest.main(["--db", "x.db"])


def test_cli_db_override_is_used(tmp_path, monkeypatch, patched_stages):
    monkeypatch.setattr(ingest, "check_models", lambda: "anthropic")
    db_path = tmp_path / "custom.db"

    # Capture stdout to avoid polluting test output.
    ingest.main([
        "--url", "https://arxiv.org/abs/2301.88888",
        "--db", str(db_path),
    ])
    assert db_path.exists()
    # The arxiv_id the stage recorder saw must have been parsed from --url.
    assert any(
        kw["arxiv_id"] == "2301.88888" for _, kw in patched_stages.calls
    )


def test_cli_domain_override_threaded_to_fetch_and_classify(
    tmp_path, monkeypatch, patched_stages
):
    monkeypatch.setattr(ingest, "check_models", lambda: "anthropic")
    db_path = tmp_path / "domain.db"
    ingest.main([
        "--url", "2301.99998",
        "--db", str(db_path),
        "--domain", "rag",
    ])

    fetch_kwargs = next(kw for name, kw in patched_stages.calls if name == "fetch")
    classify_kwargs = next(kw for name, kw in patched_stages.calls if name == "classify")
    assert fetch_kwargs.get("domain_override") == "rag"
    assert classify_kwargs.get("domain_override") == "rag"
