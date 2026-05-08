"""End-to-end tests for ingest_post orchestrator + delete_post_cascade.

We mock fetch_post / convert_post / classify / extract / index at module
boundaries so the test exercises orchestrator wiring without touching
LLMs or GLiNER. Heavy-stage tests live with their respective unit
modules.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from _system.db.cascade import delete_post_cascade
from _system.schemas.post_metadata import PostMetadata, PostStatus
from _system.scripts import ingest as ingest_mod
from _system.scripts.ingest import ingest_post


# ---------------------------------------------------------------------------
# Stub stages
# ---------------------------------------------------------------------------


def _stub_fetch_post(*, conn, url, force=False, domain_override=None, **_):
    # Insert a row at status='fetched' and return the PostMetadata.
    pm = PostMetadata(
        post_name="stub_post_2024",
        source_url=url,
        canonical_url=url,
        title="Stub Post",
        author="Tester",
        site_name="testsite",
        date="2024-02-01",
        abstract="stub abstract",
        domain=None,
        collection=None,
        status=PostStatus.FETCHED,
        markdown=None,
        raw_html="<p>stub raw html body</p>",
        content_hash="0" * 64,
        needs_review=False,
        ingested_at="2026-05-07T00:00:00",
    )
    conn.execute(
        """
        INSERT INTO posts (
            post_name, source_url, canonical_url, title, author, site_name,
            date, abstract, raw_html, content_hash, ingested_at, status,
            needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pm.post_name, pm.source_url, pm.canonical_url, pm.title,
            pm.author, pm.site_name, pm.date, pm.abstract, pm.raw_html,
            pm.content_hash, pm.ingested_at, "fetched", 0,
        ),
    )
    conn.commit()
    return pm


def _stub_convert(*, post_name, conn, force=False):
    conn.execute(
        "UPDATE posts SET markdown = ?, raw_html = NULL, status = 'converted' "
        "WHERE post_name = ?",
        ("# Stub\n\nbody " * 30, post_name),
    )
    conn.commit()
    from _system.scripts.convert_post import ConvertPostResult
    return ConvertPostResult(
        post_name=post_name,
        status="converted",
        markdown_chars=300,
        references=0,
        references_resolved_forward=0,
    )


def _stub_classify(*, paper_name, conn, force=False, domain_override=None, **_):
    # Pretend the LLM picked a domain + collection.
    conn.execute(
        "INSERT OR IGNORE INTO domains (name) VALUES (?)", ("retrieval",)
    )
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name) VALUES (?, ?)",
        ("retrieval", "rag"),
    )
    conn.execute(
        "UPDATE posts SET domain = ?, collection = ?, status = 'classified' "
        "WHERE post_name = ?",
        ("retrieval", "rag", paper_name),
    )
    conn.execute(
        """
        INSERT INTO collections (target_kind, target_id, domain, collection, is_primary)
        SELECT 'post', id, ?, ?, 1 FROM posts WHERE post_name = ?
        """,
        ("retrieval", "rag", paper_name),
    )
    # Add a topic via the unified topics table.
    conn.execute(
        """
        INSERT INTO topics (target_kind, target_id, domain, topic)
        SELECT 'post', id, 'retrieval', 'memory'
          FROM posts WHERE post_name = ?
        """,
        (paper_name,),
    )
    conn.commit()
    from _system.scripts.classify_paper import ClassifyResult
    return ClassifyResult(
        paper_name=paper_name,
        domain="retrieval",
        collections=("rag",),
        topics=("memory",),
        needs_review=False,
        status="classified",
    )


def _stub_extract(*, paper_name, conn, force=False, **_):
    conn.execute(
        "UPDATE posts SET entity_count = 3, status = 'extracted' "
        "WHERE post_name = ?",
        (paper_name,),
    )
    conn.commit()
    from _system.scripts.extract_entities import ExtractResult
    return ExtractResult(paper_name=paper_name, entity_count=3, status="extracted")


def _stub_index(*, paper_name, conn, force=False, **_):
    conn.execute(
        "UPDATE posts SET section_count = 5, status = 'indexed' "
        "WHERE post_name = ?",
        (paper_name,),
    )
    conn.commit()
    from _system.scripts.index_paper import IndexResult
    return IndexResult(paper_name=paper_name, section_count=5, status="indexed")


@pytest.fixture
def patched_pipeline(monkeypatch):
    monkeypatch.setattr(ingest_mod, "fetch_post_stage", _stub_fetch_post)
    monkeypatch.setattr(ingest_mod, "convert_post_stage", _stub_convert)
    monkeypatch.setattr(ingest_mod, "classify_paper_stage", _stub_classify)
    monkeypatch.setattr(ingest_mod, "extract_stage", _stub_extract)
    monkeypatch.setattr(ingest_mod, "index_stage", _stub_index)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class TestIngestPost:
    def test_full_pipeline_runs(self, conn, patched_pipeline):
        url = "https://example.com/post"
        summary = ingest_post(conn=conn, url=url)
        assert summary["kind"] == "post"
        assert summary["status"] == "indexed"
        assert summary["section_count"] == 5
        assert summary["entity_count"] == 3
        assert summary["domain"] == "retrieval"
        assert summary["collection"] == "rag"

    def test_progress_ticks_emitted(self, conn, patched_pipeline):
        url = "https://example.com/p"
        events: list[tuple[str, int, int]] = []

        def cb(msg, done, total):
            events.append((msg, done, total))

        ingest_post(conn=conn, url=url, progress=cb)
        # 5 stages → 5 starting ticks + 1 complete tick
        assert any(e[0] == "complete" for e in events)
        assert sum(1 for e in events if e[0].startswith("starting")) == 5

    def test_terminal_resume_skips_pipeline(self, conn, patched_pipeline):
        url = "https://example.com/already_done"
        # Seed a post already at INDEXED.
        conn.execute(
            "INSERT OR IGNORE INTO domains (name) VALUES ('retrieval')"
        )
        conn.execute(
            """
            INSERT INTO posts (
                post_name, source_url, canonical_url, title, date, abstract,
                domain, collection, ingested_at, status, needs_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "done_2024", url, url, "done", "2024-01-01", "stub",
                "retrieval", "rag",
                "2026-01-01T00:00:00", "indexed", 0,
            ),
        )
        conn.commit()

        events: list[tuple] = []
        ingest_post(conn=conn, url=url, progress=lambda *a: events.append(a))
        assert events[0] == ("already complete", 0, 0)

    def test_force_cascades_existing_post(self, conn, patched_pipeline):
        url = "https://example.com/redo"
        ingest_post(conn=conn, url=url)
        cnt_before = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE source_url = ?", (url,)
        ).fetchone()[0]
        assert cnt_before == 1

        ingest_post(conn=conn, url=url, force=True)
        cnt_after = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE source_url = ?", (url,)
        ).fetchone()[0]
        # force re-runs the whole pipeline; one row remains.
        assert cnt_after == 1

    def test_failed_fetch_halts_pipeline(self, conn, monkeypatch, patched_pipeline):
        def failing_fetch(*, conn, url, force=False, domain_override=None, **_):
            conn.execute(
                """
                INSERT INTO posts (
                    post_name, source_url, canonical_url, title, date,
                    abstract, ingested_at, status, needs_review
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "fail_2024", url, url, "fail", "2024-01-01", "fail",
                    "2026-01-01T00:00:00", "failed_fetch", 1,
                ),
            )
            conn.commit()
            return PostMetadata(
                post_name="fail_2024", source_url=url, canonical_url=url,
                title="fail", date="2024-01-01", abstract="fail",
                status=PostStatus.FAILED_FETCH,
                ingested_at="2026-01-01T00:00:00",
                needs_review=True,
            )

        monkeypatch.setattr(ingest_mod, "fetch_post_stage", failing_fetch)
        url = "https://example.com/dead"
        summary = ingest_post(conn=conn, url=url)
        assert summary["status"] == "failed_fetch"
        # Convert / classify must NOT have run.
        row = conn.execute(
            "SELECT markdown, status FROM posts WHERE source_url = ?", (url,)
        ).fetchone()
        assert row[0] is None
        assert row[1] == "failed_fetch"


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------


class TestDeletePostCascade:
    def _seed_full_post(self, conn, post_name="cascade_2024", url="https://example.com/c"):
        conn.execute(
            "INSERT OR IGNORE INTO domains (name) VALUES ('retrieval')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO collection_definitions (domain, name) VALUES ('retrieval', 'rag')"
        )
        cur = conn.execute(
            """
            INSERT INTO posts (
                post_name, source_url, canonical_url, title, date, abstract,
                domain, collection, ingested_at, status, needs_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                post_name, url, url, "T", "2024-01-01", "stub",
                "retrieval", "rag",
                "2026-01-01T00:00:00", "classified", 0,
            ),
        )
        post_id = cur.lastrowid
        conn.execute(
            "INSERT INTO collections "
            " (target_kind, target_id, domain, collection, is_primary) "
            "VALUES ('post', ?, ?, ?, 1)",
            (post_id, "retrieval", "rag"),
        )
        conn.execute(
            "INSERT INTO post_references (post_id, cited_arxiv_id, raw_text) "
            "VALUES (?, ?, ?)",
            (post_id, "2303.11366", "ref"),
        )
        # Topics + a canonical topic + alias.
        cur2 = conn.execute(
            "INSERT INTO canonical_terms (domain, term_type, canonical_name, first_seen_in) "
            "VALUES (?, ?, ?, ?)",
            ("retrieval", "topic", "memory", post_name),
        )
        topic_id = cur2.lastrowid
        conn.execute(
            "INSERT INTO topics (target_kind, target_id, domain, topic) "
            "VALUES (?, ?, ?, ?)",
            ("post", post_id, "retrieval", "memory"),
        )
        conn.execute(
            "INSERT INTO term_aliases (term_id, alias, source_paper) "
            "VALUES (?, ?, ?)",
            (topic_id, "memori", post_name),
        )
        # A sections row keyed by the post slug.
        conn.execute(
            """
            INSERT INTO sections (
                paper_id, domain, paper_name, section_title, section_level, body
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (post_id, "retrieval", post_name, "Memory", "2", "body"),
        )
        conn.commit()
        return post_id, topic_id

    def test_cascade_removes_all_per_post_rows(self, conn):
        post_id, topic_id = self._seed_full_post(conn)
        delete_post_cascade(conn, post_id=post_id)
        # All per-post rows gone.
        assert conn.execute(
            "SELECT COUNT(*) FROM posts WHERE id = ?", (post_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM post_references WHERE post_id = ?", (post_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM collections "
            " WHERE target_kind = 'post' AND target_id = ?", (post_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM topics WHERE target_kind = 'post' AND target_id = ?",
            (post_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM term_aliases WHERE source_paper = 'cascade_2024'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM sections WHERE paper_name = 'cascade_2024'"
        ).fetchone()[0] == 0

    def test_cascade_gcs_orphan_topic(self, conn):
        post_id, topic_id = self._seed_full_post(conn)
        delete_post_cascade(conn, post_id=post_id)
        # The topic canonical had only this one binding, so it's GC'd.
        assert conn.execute(
            "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (topic_id,)
        ).fetchone()[0] == 0

    def test_cascade_preserves_collections_registry(self, conn):
        post_id, _ = self._seed_full_post(conn)
        delete_post_cascade(conn, post_id=post_id)
        # Collection registry survives.
        assert conn.execute(
            "SELECT COUNT(*) FROM collection_definitions WHERE name = 'rag'"
        ).fetchone()[0] == 1
