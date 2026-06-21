"""Unit tests for _system/scripts/fetch_post.py.

Network is fully mocked via ``httpx.MockTransport``. No test touches real
internet.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from _system.scripts.fetch_post import fetch, fetch_from_file
from _system.utils.http import USER_AGENT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LIL_HTML = """<!doctype html>
<html lang="en">
<head>
  <title>Lil'Log: A Survey on Agents</title>
  <meta property="og:site_name" content="Lil'Log"/>
  <meta property="og:url" content="https://lilianweng.github.io/posts/2023-06-23-agent/"/>
  <meta property="og:description" content="A long-form post about agents."/>
  <link rel="canonical" href="https://lilianweng.github.io/posts/2023-06-23-agent/"/>
</head>
<body>
  <article>
    <h1>A Survey on Agents</h1>
    <p>Posted on <time datetime="2023-06-23">June 23, 2023</time> by Lilian Weng.</p>
    <h2>Memory</h2>
    <p>Agents need memory to operate over long horizons. The reflection
    framework of <a href="https://arxiv.org/abs/2303.11366">Reflexion</a>
    grounds this in language-model self-critique. We also discuss
    <a href="https://github.com/owner/repo">an example repo</a>.</p>
    <h2>Planning</h2>
    <p>Agents plan via tree search. See arXiv:2305.10601 for tree-of-thoughts.</p>
    <h2>Tool Use</h2>
    <p>Agents wield tools through Toolformer-style self-supervised
    learning. The introduction of GPT-4 catalyzed this work.</p>
    <h2>Conclusion</h2>
    <p>Agents are powerful but their reliability remains an open question.
    We invite further work, especially around evaluation and safety.</p>
  </article>
</body>
</html>
"""


def _client_returning(html: str, *, status: int = 200, headers: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        h = {"content-type": "text/html; charset=utf-8"}
        if headers:
            h.update(headers)
        return httpx.Response(status, content=html.encode("utf-8"), headers=h)
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": USER_AGENT},
        timeout=5.0,
        follow_redirects=True,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestFetchHappyPath:
    def test_fetches_and_persists_post(self, conn):
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        client = _client_returning(_LIL_HTML)

        pm = fetch(conn=conn, url=url, client=client)

        assert pm.status == "fetched"
        assert pm.title.startswith("A Survey on Agents") or "Survey" in pm.title
        assert pm.canonical_url == url
        assert pm.date.startswith("2023")
        assert pm.raw_html is not None and len(pm.raw_html) > 100
        assert pm.content_hash is not None and len(pm.content_hash) == 64
        assert pm.post_name and pm.post_name.replace("_", "").isalnum()

        row = conn.execute(
            "SELECT post_name, source_url, canonical_url, status FROM posts"
        ).fetchone()
        assert row is not None
        assert row[1] == url
        assert row[2] == url
        assert row[3] == "fetched"

    def test_idempotent_skip_when_already_present(self, conn):
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        client1 = _client_returning(_LIL_HTML)
        first = fetch(conn=conn, url=url, client=client1)

        # Second call without --force: the source_url pre-check should
        # short-circuit before any network call. A client that would
        # raise on a request proves we never hit the wire.
        def explode(request: httpx.Request) -> httpx.Response:
            raise AssertionError("network should not be invoked for an existing post")

        explode_client = httpx.Client(
            transport=httpx.MockTransport(explode),
            headers={"User-Agent": USER_AGENT},
            timeout=5.0,
            follow_redirects=True,
        )
        try:
            second = fetch(conn=conn, url=url, client=explode_client)
        finally:
            explode_client.close()
        assert second.post_name == first.post_name
        rows = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE canonical_url = ?", (url,)
        ).fetchone()
        assert rows[0] == 1

    def test_extracts_repo_url(self, conn):
        url = "https://example.com/post"
        client = _client_returning(_LIL_HTML)
        pm = fetch(conn=conn, url=url, client=client)
        # The HTML has a github.com link
        assert pm.code_repo == "https://github.com/owner/repo"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFetchFailures:
    def test_404_marks_failed_fetch(self, conn):
        url = "https://example.com/missing"
        client = _client_returning("<html>not found</html>", status=404)
        pm = fetch(conn=conn, url=url, client=client)
        assert pm.status == "failed_fetch"
        row = conn.execute(
            "SELECT status, needs_review FROM posts WHERE source_url = ?", (url,)
        ).fetchone()
        assert row[0] == "failed_fetch"
        assert row[1] == 1

    def test_unparseable_html_marks_failed_fetch(self, conn):
        # An HTML body trafilatura cannot recognize as an article.
        url = "https://example.com/skeleton"
        client = _client_returning("<html><body><script>alert(1)</script></body></html>")
        pm = fetch(conn=conn, url=url, client=client)
        assert pm.status == "failed_fetch"


# ---------------------------------------------------------------------------
# Slug namespace
# ---------------------------------------------------------------------------


class TestCanonicalUrlResolution:
    _ROOT_RELATIVE_HTML = """<!doctype html>
<html><head>
  <title>Semantic Search Without Embeddings</title>
  <link rel="canonical" href="/blog/2026/01/08/semantic-search-without-embeddings.html"/>
</head><body><article>
<h1>Semantic Search Without Embeddings</h1>
<p>Posted on <time datetime="2026-01-08">Jan 8</time>.</p>
<h2>Intro</h2>
<p>This is the body of a sample post that needs enough text to satisfy
trafilatura's article extractor — at least a couple of paragraphs so
the bare extraction recognizes a readable article rather than a
skeleton DOM that the favor_precision pass throws away. We talk about
semantic search, taxonomies, and the alternatives to dense embeddings.
We continue with more text to make sure the body is well above the
minimum-length threshold.</p>
<h2>Method</h2>
<p>The method section discusses keyword matching, BM25 scoring, and
how taxonomies scale better than embeddings on long-tail queries.
We expand on this point with a worked example and concrete numbers
that grade-school readers could follow.</p>
</article></body></html>
"""

    _PROTOCOL_RELATIVE_HTML = _ROOT_RELATIVE_HTML.replace(
        'href="/blog/',
        'href="//softwaredoug.com/blog/',
    )

    def test_root_relative_canonical_is_absolutized(self, conn):
        url = "https://softwaredoug.com/blog/2026/01/08/semantic-search-without-embeddings"
        client = _client_returning(self._ROOT_RELATIVE_HTML)
        pm = fetch(conn=conn, url=url, client=client)
        assert pm.canonical_url == (
            "https://softwaredoug.com/blog/2026/01/08/"
            "semantic-search-without-embeddings.html"
        )

    def test_protocol_relative_canonical_is_absolutized(self, conn):
        url = "https://softwaredoug.com/blog/2026/01/08/semantic-search-without-embeddings"
        client = _client_returning(self._PROTOCOL_RELATIVE_HTML)
        pm = fetch(conn=conn, url=url, client=client)
        assert pm.canonical_url.startswith("https://softwaredoug.com/blog/")

    def test_absolute_canonical_passes_through(self, conn):
        absolute_html = self._ROOT_RELATIVE_HTML.replace(
            'href="/blog/',
            'href="https://example.com/blog/',
        )
        url = "https://softwaredoug.com/some-redirect"
        client = _client_returning(absolute_html)
        pm = fetch(conn=conn, url=url, client=client)
        assert pm.canonical_url.startswith("https://example.com/blog/")


class TestSlugNamespace:
    def test_post_slug_unique_against_papers(self, conn):
        # Seed a paper with paper_name='post_2023' so a colliding post
        # slug must use the URL-hash tiebreaker.
        conn.execute(
            """
            INSERT INTO papers (
                arxiv_id, paper_name, title, authors, date, abstract,
                pdf_url, ingested_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2099.99999", "a_survey_2023", "Stub Paper",
                "[]", "2023-06-23", "stub", "https://arxiv.org/pdf/2099.99999",
                "2026-01-01T00:00:00", "fetched",
            ),
        )
        conn.commit()
        url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        client = _client_returning(_LIL_HTML)
        pm = fetch(conn=conn, url=url, client=client)
        # Slug must NOT equal the existing paper's slug.
        assert pm.post_name != "a_survey_2023"


# ---------------------------------------------------------------------------
# Local-file front-end (fetch_from_file) — no network, no httpx
# ---------------------------------------------------------------------------


_LIL_CANONICAL = "https://lilianweng.github.io/posts/2023-06-23-agent/"

# An article-shaped page with NO <link rel=canonical> and NO og:url, so
# the file front-end has to fall back to the file:// sentinel. Enough
# prose for trafilatura's favor_precision article extractor to keep it.
_NO_CANONICAL_HTML = """<!doctype html>
<html lang="en"><head>
  <title>A Post Without Any Canonical Link</title>
</head><body><article>
<h1>A Post Without Any Canonical Link</h1>
<p>Posted on <time datetime="2025-03-10">March 10, 2025</time>.</p>
<h2>Intro</h2>
<p>This post deliberately omits both link rel=canonical and og:url so the
file front-end has to fall back to the file:// sentinel. The body needs
enough prose for trafilatura's favor_precision article extractor to keep
it — several sentences across a couple of paragraphs covering the topic
at hand with concrete detail rather than a skeleton DOM that the
precision pass would discard.</p>
<h2>Body</h2>
<p>We continue with more text to make sure the body is comfortably above
the minimum-length threshold the convert stage enforces, discussing the
subject with a worked example and a plain explanation a grade-schooler
could follow.</p>
</article></body></html>
"""


class TestFetchFromFile:
    def _write(self, tmp_path: Path, html: str, name: str = "post.html") -> Path:
        p = tmp_path / name
        p.write_text(html, encoding="utf-8")
        return p

    def test_reads_and_persists_from_file(self, conn, tmp_path):
        path = self._write(tmp_path, _LIL_HTML)
        pm = fetch_from_file(conn=conn, html_path=path)

        assert pm.status == "fetched"
        # Canonical comes from the file's <link rel=canonical>, not the path.
        assert pm.canonical_url == _LIL_CANONICAL
        # File lane: source_url == canonical_url, no transport metadata.
        assert pm.source_url == pm.canonical_url
        assert pm.etag is None
        assert pm.last_modified is None
        assert pm.content_hash is not None and len(pm.content_hash) == 64
        assert pm.code_repo == "https://github.com/owner/repo"

        row = conn.execute(
            "SELECT status, source_url, canonical_url, needs_review FROM posts"
        ).fetchone()
        assert row[0] == "fetched"
        assert row[1] == row[2] == _LIL_CANONICAL
        assert row[3] == 0

    def test_no_canonical_falls_back_to_file_uri(self, conn, tmp_path):
        path = self._write(tmp_path, _NO_CANONICAL_HTML)
        pm = fetch_from_file(conn=conn, html_path=path)
        assert pm.status == "fetched"
        assert pm.canonical_url.startswith("file://")
        assert pm.canonical_url.endswith("post.html")
        assert pm.source_url == pm.canonical_url

    def test_missing_file_raises(self, conn, tmp_path):
        with pytest.raises(FileNotFoundError):
            fetch_from_file(conn=conn, html_path=tmp_path / "nope.html")

    def test_unparseable_html_marks_failed_fetch(self, conn, tmp_path):
        path = self._write(
            tmp_path, "<html><body><script>alert(1)</script></body></html>"
        )
        pm = fetch_from_file(conn=conn, html_path=path)
        assert pm.status == "failed_fetch"
        row = conn.execute(
            "SELECT status, needs_review FROM posts"
        ).fetchone()
        assert row[0] == "failed_fetch"
        assert row[1] == 1

    def test_idempotent_without_force(self, conn, tmp_path):
        path = self._write(tmp_path, _LIL_HTML)
        first = fetch_from_file(conn=conn, html_path=path)
        second = fetch_from_file(conn=conn, html_path=path)
        assert second.post_name == first.post_name
        cnt = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE canonical_url = ?",
            (first.canonical_url,),
        ).fetchone()[0]
        assert cnt == 1

    def test_force_overwrites_stub_preserving_slug(self, conn, tmp_path):
        # Seed a paywall stub at the SAME canonical the file resolves to.
        conn.execute(
            """
            INSERT INTO posts (
                post_name, source_url, canonical_url, title, date, abstract,
                ingested_at, status, needs_review
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "post_2026", "https://levelup.gitconnected.com/x?sk=abc",
                _LIL_CANONICAL, "(stub)", "2026-01-01", "(stub)",
                "2026-01-01T00:00:00", "failed_fetch", 1,
            ),
        )
        conn.commit()

        path = self._write(tmp_path, _LIL_HTML)
        pm = fetch_from_file(conn=conn, html_path=path, force=True)

        assert pm.post_name == "post_2026"  # slug preserved across overwrite
        assert pm.status == "fetched"
        cnt = conn.execute(
            "SELECT COUNT(*) FROM posts WHERE canonical_url = ?", (_LIL_CANONICAL,)
        ).fetchone()[0]
        assert cnt == 1
        # ingested_at carried over from the stub row.
        row = conn.execute(
            "SELECT ingested_at FROM posts WHERE post_name = 'post_2026'"
        ).fetchone()
        assert row[0] == "2026-01-01T00:00:00"
