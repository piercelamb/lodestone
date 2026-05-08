"""Unit tests for _system/scripts/convert_post.py — markdown extraction +
arxiv reference resolution.
"""
from __future__ import annotations

import pytest

from _system.scripts.convert_post import (
    StageNotAllowed,
    convert,
    PostNotFound,
    RawHtmlMissing,
)


_HTML_WITH_REFS = """<!doctype html>
<html><head><title>Memory in Agents</title></head>
<body>
  <article>
    <h1>Memory in Agents</h1>
    <p>Posted 2024-02-01.</p>
    <h2>Background</h2>
    <p>Reflexion (<a href="https://arxiv.org/abs/2303.11366">arXiv:2303.11366</a>)
    proposes self-critique as a memory primitive. Tree-of-thoughts
    (arXiv:2305.10601) builds on this with a richer planning loop.</p>
    <h2>Discussion</h2>
    <p>We expect future work to extend these ideas with longer horizons
    and stronger external memory. The reliability of these systems is
    an open question that we plan to investigate further.</p>
  </article>
</body></html>
"""


def _seed_post(conn, *, post_name="memory_2024", raw_html=_HTML_WITH_REFS,
               status="fetched", domain=None, collection=None):
    conn.execute(
        """
        INSERT INTO posts (
            post_name, source_url, canonical_url, title, date, abstract,
            domain, collection, raw_html, ingested_at, status, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            post_name,
            "https://example.com/memory",
            "https://example.com/memory",
            "Memory in Agents",
            "2024-02-01",
            "agents need memory",
            domain,
            collection,
            raw_html,
            "2026-01-01T00:00:00",
            status,
            0,
        ),
    )
    conn.commit()


class TestConvertPostHappy:
    def test_converts_html_to_markdown(self, conn):
        _seed_post(conn)
        result = convert(post_name="memory_2024", conn=conn)
        assert result.status == "converted"
        assert result.markdown_chars > 100

        row = conn.execute(
            "SELECT status, raw_html, markdown FROM posts WHERE post_name = ?",
            ("memory_2024",),
        ).fetchone()
        assert row[0] == "converted"
        assert row[1] is None  # raw_html cleared on success
        assert row[2] is not None
        assert "Reflexion" in row[2] or "Memory" in row[2]

    def test_extracts_arxiv_references(self, conn):
        _seed_post(conn)
        result = convert(post_name="memory_2024", conn=conn)
        assert result.references >= 2  # both arxiv ids appear

        cited_ids = [
            r[0] for r in conn.execute(
                "SELECT cited_arxiv_id FROM post_references"
            ).fetchall()
        ]
        assert "2303.11366" in cited_ids
        assert "2305.10601" in cited_ids

    def test_forward_resolves_when_paper_present(self, conn):
        # Seed a paper that the post references.
        conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, paper_name, title, authors, date, abstract,
                pdf_url, ingested_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2303.11366", "reflexion_2023", "Reflexion",
                "[]", "2023-03-20", "stub", "https://arxiv.org/pdf/2303.11366",
                "2026-01-01T00:00:00", "fetched",
            ),
        )
        conn.commit()
        _seed_post(conn)
        result = convert(post_name="memory_2024", conn=conn)
        assert result.references_resolved_forward == 1

        ref = conn.execute(
            """
            SELECT cited_arxiv_id, cited_paper_id
              FROM post_references WHERE cited_arxiv_id = ?
            """,
            ("2303.11366",),
        ).fetchone()
        assert ref[1] is not None  # linked


class TestConvertPostFailures:
    def test_raises_when_not_found(self, conn):
        with pytest.raises(PostNotFound):
            convert(post_name="missing", conn=conn)

    def test_raises_when_raw_html_null(self, conn):
        _seed_post(conn, raw_html=None, status="converted")
        # Already-converted status; still raw_html is NULL.
        # status=converted skips can_run_from anyway, so make it fetched
        # and NULL raw_html together.
        conn.execute(
            "UPDATE posts SET raw_html = NULL, status = 'fetched' "
            "WHERE post_name = 'memory_2024'"
        )
        conn.commit()
        with pytest.raises(RawHtmlMissing):
            convert(post_name="memory_2024", conn=conn)

    def test_terminal_status_blocks_convert(self, conn):
        _seed_post(conn, status="failed_fetch")
        with pytest.raises(StageNotAllowed):
            convert(post_name="memory_2024", conn=conn)

    def test_short_extraction_marks_failed_parse(self, conn):
        # Skeleton-DOM-style HTML produces near-empty markdown.
        _seed_post(
            conn,
            raw_html="<html><body><div></div></body></html>",
        )
        result = convert(post_name="memory_2024", conn=conn)
        assert result.status == "failed_parse"

        row = conn.execute(
            "SELECT status, needs_review FROM posts WHERE post_name = ?",
            ("memory_2024",),
        ).fetchone()
        assert row[0] == "failed_parse"
        assert row[1] == 1


class TestBackwardResolution:
    def test_paper_convert_picks_up_post_references(self, conn):
        """Post ingests first; paper ingests second; paper's CONVERT
        should backward-resolve the post_references row."""
        from _system.scripts import convert_paper

        # Stage 1: post is converted with a dangling reference.
        _seed_post(conn)
        convert(post_name="memory_2024", conn=conn)
        ref = conn.execute(
            "SELECT cited_paper_id FROM post_references WHERE cited_arxiv_id = ?",
            ("2305.10601",),
        ).fetchone()
        assert ref[0] is None  # not yet linked

        # Stage 2: paper for that arxiv id ingests + converts. The minimal
        # row needs raw_html with the LATEX_LOCAL sentinel so convert
        # has SOMETHING to walk; here we go straight to a manual paper
        # row + invoke just the backward-resolve helper.
        from _system.utils.citation_resolution import resolve_arxiv_citations
        from _system.utils.source_resolution import SourceKind

        cur = conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, paper_name, title, authors, date, abstract,
                pdf_url, ingested_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2305.10601", "tot_2023", "Tree of Thoughts",
                "[]", "2023-05-17", "stub", "https://arxiv.org/pdf/2305.10601",
                "2026-01-01T00:00:00", "fetched",
            ),
        )
        new_paper_id = cur.lastrowid
        conn.commit()

        forward, backward = resolve_arxiv_citations(
            conn,
            kind=SourceKind.PAPER,
            source_id=new_paper_id,
            source_arxiv_id="2305.10601",
        )
        assert backward == 1

        ref = conn.execute(
            "SELECT cited_paper_id FROM post_references WHERE cited_arxiv_id = ?",
            ("2305.10601",),
        ).fetchone()
        assert ref[0] == new_paper_id
