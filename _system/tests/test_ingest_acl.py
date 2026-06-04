"""End-to-end smoke test for the ingest_acl orchestrator.

Mirrors test_ingest_post.py: downstream stages are stubbed at module
boundaries so the orchestrator wiring is exercised without touching
network/LLMs/GLiNER. Heavy-stage behavior lives in its own modules.
"""
from __future__ import annotations

import pytest

from _system.schemas.paper_metadata import HtmlSource, PaperMetadata, PaperStatus
from _system.scripts import ingest as ingest_mod
from _system.scripts.ingest import ingest_acl


# ---------------------------------------------------------------------------
# Stage stubs
# ---------------------------------------------------------------------------


def _stub_fetch_acl(*, conn, acl_id, force=False, domain_override=None, **_):
    arxiv_id = f"acl:{acl_id}"
    paper_name = "smith_2021_toy"
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, raw_html, html_source, content_hash,
            ingested_at, status, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id, paper_name, "Toy", "[\"Smith\"]", "2021-01-01",
            "abs", f"https://aclanthology.org/{acl_id}.pdf",
            "__PDF_MARKDOWN__\n# Toy\n\nbody", "pdf_fallback",
            "0" * 64, "2026-01-01T00:00:00",
            PaperStatus.FETCHED.value, 0,
        ),
    )
    conn.commit()
    return PaperMetadata(
        arxiv_id=arxiv_id,
        paper_name=paper_name,
        title="Toy",
        authors='["Smith"]',
        date="2021-01-01",
        abstract="abs",
        pdf_url=f"https://aclanthology.org/{acl_id}.pdf",
        html_source=HtmlSource.PDF_FALLBACK,
        status=PaperStatus.FETCHED,
        ingested_at="2026-01-01T00:00:00",
    )


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
    domain = domain_override or "nlp"
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", (domain,))
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name) VALUES (?, ?)",
        (domain, "dialogue"),
    )
    conn.execute(
        "UPDATE papers SET domain = ?, collection = ?, status = 'classified' "
        "WHERE paper_name = ?",
        (domain, "dialogue", paper_name),
    )
    conn.execute(
        """
        INSERT INTO collections (target_kind, target_id, domain, collection, is_primary)
        SELECT 'paper', id, ?, ?, 1 FROM papers WHERE paper_name = ?
        """,
        (domain, "dialogue", paper_name),
    )
    conn.commit()
    from _system.scripts.classify_paper import ClassifyResult
    return ClassifyResult(
        paper_name=paper_name, domain=domain, collections=("dialogue",),
        topics=(), needs_review=False, status="classified",
    )


def _stub_extract(*, paper_name, conn, force=False, **_):
    conn.execute(
        "UPDATE papers SET entity_count = 4, status = 'extracted' "
        "WHERE paper_name = ?",
        (paper_name,),
    )
    conn.commit()
    from _system.scripts.extract_entities import ExtractResult
    return ExtractResult(paper_name=paper_name, entity_count=4, status="extracted")


def _stub_index(*, paper_name, conn, force=False, **_):
    conn.execute(
        "UPDATE papers SET status = 'indexed' WHERE paper_name = ?",
        (paper_name,),
    )
    conn.commit()
    from _system.scripts.index_paper import IndexResult
    return IndexResult(paper_name=paper_name, section_count=7, status="indexed")


@pytest.fixture
def patched_pipeline(monkeypatch):
    monkeypatch.setattr(ingest_mod, "fetch_acl_stage", _stub_fetch_acl)
    monkeypatch.setattr(ingest_mod, "convert_stage", _stub_convert)
    monkeypatch.setattr(ingest_mod, "classify_paper_stage", _stub_classify)
    monkeypatch.setattr(ingest_mod, "extract_stage", _stub_extract)
    monkeypatch.setattr(ingest_mod, "index_stage", _stub_index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_pipeline_advances_to_indexed(conn, patched_pipeline):
    summary = ingest_acl(conn=conn, acl_id="2021.acl-long.285")
    assert summary["kind"] == "paper"
    assert summary["arxiv_id"] == "acl:2021.acl-long.285"
    assert summary["status"] == "indexed"
    # section_count is derived from the sections table; the stub doesn't
    # populate sections, so we only assert entity/domain/collection.
    assert summary["entity_count"] == 4
    assert summary["domain"] == "nlp"
    assert summary["collection"] == "dialogue"


def test_progress_ticks_emitted(conn, patched_pipeline):
    events: list[tuple[str, int, int]] = []
    ingest_acl(
        conn=conn,
        acl_id="2021.acl-long.285",
        progress=lambda m, d, t: events.append((m, d, t)),
    )
    # 5 stages → 5 "starting" ticks + 1 "complete" tick.
    assert sum(1 for e in events if e[0].startswith("starting")) == 5
    assert any(e[0] == "complete" for e in events)


def test_terminal_resume_skips_pipeline(conn, patched_pipeline):
    """If a row is already at INDEXED, ingest_acl is a no-op."""
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES ('nlp')")
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name) "
        "VALUES ('nlp', 'dialogue')",
    )
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, domain, collection, ingested_at, status, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "acl:2021.acl-long.285", "smith_2021_toy", "Toy",
            "[\"Smith\"]", "2021-01-01", "abs",
            "https://aclanthology.org/2021.acl-long.285.pdf",
            "nlp", "dialogue", "2026-01-01T00:00:00", "indexed", 0,
        ),
    )
    conn.commit()

    events: list[tuple] = []
    summary = ingest_acl(
        conn=conn,
        acl_id="2021.acl-long.285",
        progress=lambda *a: events.append(a),
    )
    assert events[0] == ("already complete", 0, 0)
    assert summary["status"] == "indexed"


def test_force_cascades_existing_row(conn, patched_pipeline):
    ingest_acl(conn=conn, acl_id="2021.acl-long.285")
    assert conn.execute(
        "SELECT COUNT(*) FROM papers WHERE arxiv_id = ?",
        ("acl:2021.acl-long.285",),
    ).fetchone()[0] == 1

    ingest_acl(conn=conn, acl_id="2021.acl-long.285", force=True)
    # Force re-runs the pipeline; still exactly one row.
    assert conn.execute(
        "SELECT COUNT(*) FROM papers WHERE arxiv_id = ?",
        ("acl:2021.acl-long.285",),
    ).fetchone()[0] == 1
