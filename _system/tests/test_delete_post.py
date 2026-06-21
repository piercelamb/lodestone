"""Unit tests for _system/scripts/delete_post.py.

Mirrors the cascade seed in test_ingest_post.py's TestDeletePostCascade,
but drives the deletion through the CLI-level ``delete_post`` /
``_resolve_post`` helpers so the summary + resolution paths are covered.
"""
from __future__ import annotations

import pytest

from _system.scripts.delete_post import _resolve_post, delete_post


def _seed_full_post(conn, *, post_name="cascade_2024", url="https://example.com/c"):
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES ('retrieval')")
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name) "
        "VALUES ('retrieval', 'rag')"
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
            "retrieval", "rag", "2026-01-01T00:00:00", "classified", 0,
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


class TestResolvePost:
    def test_resolve_by_slug(self, conn):
        _seed_full_post(conn)
        post = _resolve_post(conn, slug="cascade_2024", url=None)
        assert post is not None
        assert post.post_name == "cascade_2024"
        assert post.canonical_url == "https://example.com/c"

    def test_resolve_by_url(self, conn):
        _seed_full_post(conn)
        post = _resolve_post(conn, slug=None, url="https://example.com/c")
        assert post is not None
        assert post.post_name == "cascade_2024"

    def test_resolve_missing_returns_none(self, conn):
        assert _resolve_post(conn, slug="ghost", url=None) is None
        assert _resolve_post(conn, slug=None, url="https://nope.example") is None


class TestDeletePost:
    def test_deletes_all_per_post_rows(self, conn):
        post_id, topic_id = _seed_full_post(conn)
        post = _resolve_post(conn, slug="cascade_2024", url=None)

        summary = delete_post(conn=conn, post=post)

        assert summary["post_name"] == "cascade_2024"
        assert summary["deleted"]["post_row"] == 1
        assert summary["deleted"]["post_references"] == 1
        assert summary["deleted"]["sections"] == 1
        assert summary["deleted"]["post_topics"] == 1
        assert summary["deleted"]["collection_memberships"] == 1
        assert summary["deleted"]["term_aliases"] == 1

        # Every per-post row gone.
        assert conn.execute(
            "SELECT COUNT(*) FROM posts WHERE id = ?", (post_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM post_references WHERE post_id = ?", (post_id,)
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM sections WHERE paper_name = 'cascade_2024'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM topics WHERE target_kind = 'post' AND target_id = ?",
            (post_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM term_aliases WHERE source_paper = 'cascade_2024'"
        ).fetchone()[0] == 0

    def test_preserves_curated_catalog(self, conn):
        _seed_full_post(conn)
        post = _resolve_post(conn, slug="cascade_2024", url=None)
        delete_post(conn=conn, post=post)
        # Curated catalog survives the deletion of its last member.
        assert conn.execute(
            "SELECT COUNT(*) FROM collection_definitions WHERE name = 'rag'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM domains WHERE name = 'retrieval'"
        ).fetchone()[0] == 1

    def test_orphan_topic_canonical_gcd(self, conn):
        post_id, topic_id = _seed_full_post(conn)
        post = _resolve_post(conn, slug="cascade_2024", url=None)
        delete_post(conn=conn, post=post)
        # The topic had only this one binding → GC'd by the cascade.
        assert conn.execute(
            "SELECT COUNT(*) FROM canonical_terms WHERE id = ?", (topic_id,)
        ).fetchone()[0] == 0
