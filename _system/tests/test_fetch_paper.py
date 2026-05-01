"""Unit tests for _system/scripts/fetch_paper.py.

Network is fully mocked via ``httpx.MockTransport`` and the arxiv library
is replaced with a direct-metadata callable. No test touches real internet.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import httpx
import pytest
from PIL import Image

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.scripts import fetch_paper as fp
from _system.scripts.fetch_paper import (
    USER_AGENT,
    _ArxivMetadata,
    _normalize_repo_url,
    _process_figure_image,
    fetch,
)
from _system.utils.arxiv_urls import parse_arxiv_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_meta(**overrides) -> _ArxivMetadata:
    base = dict(
        title="A Paper About Things",
        authors=["Alice Author", "Bob Buthor"],
        abstract="We do things.",
        published="2024-05-11",
        comment=None,
        summary=None,
        pdf_url="https://arxiv.org/pdf/2301.12345",
    )
    base.update(overrides)
    return _ArxivMetadata(**base)


def _png_bytes(width: int, height: int, color=(200, 50, 50)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(width: int, height: int, color=(50, 200, 50)) -> bytes:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _resp_html(body: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        content=body.encode("utf-8"),
        headers={"content-type": "text/html; charset=utf-8"},
    )


def _minimal_html_with_one_figure(src: str = "fig1.png") -> str:
    return f"""<!doctype html>
<html><body>
  <figure class="ltx_figure">
    <img src="{src}"/>
    <figcaption class="ltx_caption">Figure 1: overview.</figcaption>
  </figure>
</body></html>
"""


class _Recorder:
    """Captures every request a MockTransport sees."""

    def __init__(self, responder):
        self.calls: list[httpx.Request] = []
        self._responder = responder

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self._responder(request)


def _client_with(handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": USER_AGENT},
        timeout=5.0,
        follow_redirects=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "lodestone.db"
    conn = get_conn(p)
    init_db(conn)
    conn.close()
    return p


@pytest.fixture
def conn(db_path: Path):
    """Open a migrated connection for each test; closed at teardown."""
    c = get_conn(db_path)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def fast_sleep(monkeypatch):
    """Silence the PwC rate-limit sleep so tests run instantly."""
    monkeypatch.setattr(fp, "_sleep", lambda s: None)


# ---------------------------------------------------------------------------
# Early dedup
# ---------------------------------------------------------------------------


def test_second_fetch_dedups_early_no_network(conn):
    """A second fetch for the same arxiv_id must skip every HTTP call."""
    arxiv_id = "2301.12345"
    meta = _make_meta()

    # Feed the parser a data: URI so no figure download is needed.
    data_b64_png = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlE"
        "QVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    def html_ok_handler(req):
        url = str(req.url)
        if "/html/" in url:
            return _resp_html(_minimal_html_with_one_figure(data_b64_png))
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.4\n%%EOF\n")
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    first_recorder = _Recorder(html_ok_handler)

    with _client_with(first_recorder.handler) as client:
        fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=client,
            arxiv_lookup=lambda _id: meta,
        )
    assert len(first_recorder.calls) > 0

    # Second fetch: MockTransport returns 500 on every URL, so if any HTTP
    # happens the test will blow up.
    second_recorder = _Recorder(lambda r: httpx.Response(500))
    with _client_with(second_recorder.handler) as client:
        returned = fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=client,
            arxiv_lookup=lambda _id: pytest.fail("arxiv lookup ran on dedup"),
        )
    assert second_recorder.calls == []
    assert returned.arxiv_id == arxiv_id


# ---------------------------------------------------------------------------
# HTML source discovery
# ---------------------------------------------------------------------------


def test_html_source_fallback_arxiv_404_ar5iv_200(conn, fast_sleep):
    arxiv_id = "2301.12345"
    meta = _make_meta()

    def handler(req):
        host = req.url.host
        path = req.url.path
        if host == "arxiv.org" and path.startswith("/html"):
            return httpx.Response(404)
        if host == "ar5iv.labs.arxiv.org" and path.startswith("/html"):
            return _resp_html(_minimal_html_with_one_figure("figA.png"))
        if path.startswith("/pdf"):
            return httpx.Response(200, content=b"%PDF-1.4")
        if "figA.png" in path:
            return httpx.Response(200, content=_png_bytes(100, 80), headers={"content-type": "image/png"})
        if host == "paperswithcode.com":
            return httpx.Response(404)
        return httpx.Response(404)

    recorder = _Recorder(handler)
    with _client_with(recorder.handler) as client:
        pm = fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=client,
            arxiv_lookup=lambda _id: meta,
        )
    assert pm.html_source == "ar5iv"


def test_both_html_sources_fail_persists_failed_html_stub(conn, fast_sleep):
    arxiv_id = "2301.99999"
    meta = _make_meta()

    def handler(req):
        url = str(req.url)
        if "/html/" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    with _client_with(_Recorder(handler).handler) as client:
        pm = fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=client,
            arxiv_lookup=lambda _id: meta,
        )
    assert pm.status == "failed_html"
    assert pm.raw_html is None
    assert pm.html_source is None

    figures = conn.execute("SELECT COUNT(*) FROM figures").fetchone()[0]
    papers = conn.execute("SELECT COUNT(*) FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()[0]
    assert figures == 0
    assert papers == 1


def test_raw_html_is_persisted_on_success(conn, fast_sleep):
    arxiv_id = "2301.00001"
    body = _minimal_html_with_one_figure("data:image/png;base64,iVBORw0KGgoAAAA=")

    def handler(req):
        url = str(req.url)
        if "arxiv.org/html" in url:
            return _resp_html(body)
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.4\n%%EOF\n")
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    with _client_with(_Recorder(handler).handler) as client:
        fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=client,
            arxiv_lookup=lambda _id: _make_meta(),
        )

    raw = conn.execute("SELECT raw_html FROM papers WHERE arxiv_id = ?", (arxiv_id,)).fetchone()[0]
    assert raw == body


# ---------------------------------------------------------------------------
# Arxiv id identity
# ---------------------------------------------------------------------------


def test_version_suffix_preserved(conn, fast_sleep):
    arxiv_id = "2301.12345v2"
    captured: list[str] = []

    def handler(req):
        url = str(req.url)
        captured.append(url)
        if "arxiv.org/html" in url:
            return _resp_html(_minimal_html_with_one_figure("data:image/png;base64,AA=="))
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.4")
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    with _client_with(_Recorder(handler).handler) as client:
        pm = fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=client,
            arxiv_lookup=lambda _id: _make_meta(),
        )
    assert pm.arxiv_id == "2301.12345v2"
    # Versioned id should appear in the html URL as-is.
    assert any("2301.12345v2" in u for u in captured)


def test_parse_arxiv_id_from_url_preserves_version():
    assert parse_arxiv_id("https://arxiv.org/abs/2301.12345v3") == "2301.12345v3"
    assert parse_arxiv_id("2301.12345") == "2301.12345"


# ---------------------------------------------------------------------------
# HTTP hygiene
# ---------------------------------------------------------------------------


def test_default_client_factory_sets_user_agent():
    """Unit test: `_make_default_client()` stamps the Lodestone UA on the
    client. Outbound requests that copy it from there carry the right
    header — this is the real invariant."""
    client = fp._make_default_client()
    try:
        assert client.headers.get("User-Agent") == USER_AGENT
    finally:
        client.close()


def test_user_agent_header_present_on_every_request(conn, fast_sleep):
    """End-to-end: the UA installed on the client rides every request the
    pipeline emits."""
    arxiv_id = "2301.00002"
    data_uri = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42"
        "mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )

    seen: list[httpx.Request] = []

    def handler(req):
        seen.append(req)
        host = req.url.host
        path = req.url.path
        if host == "arxiv.org" and path.startswith("/html"):
            return _resp_html(_minimal_html_with_one_figure(data_uri))
        if path.startswith("/pdf"):
            return httpx.Response(200, content=b"%PDF-1.4")
        if host == "paperswithcode.com":
            return httpx.Response(404)
        return httpx.Response(404)

    with _client_with(handler) as client:
        fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=client,
            arxiv_lookup=lambda _id: _make_meta(),
        )

    assert seen, "expected at least one mocked request"
    for req in seen:
        assert req.headers.get("User-Agent") == USER_AGENT


# ---------------------------------------------------------------------------
# Figure handling
# ---------------------------------------------------------------------------


def test_figure_downscaling_preserves_aspect_3000x2000_to_1280():
    raw = _png_bytes(3000, 2000)
    out, mime = _process_figure_image(raw, "image/png")
    img = Image.open(io.BytesIO(out))
    assert img.width == 1280
    # Aspect ratio 3:2 → height should be ~853.
    assert 852 <= img.height <= 854


def test_figure_jpeg_reencoded_quality_85():
    big = _jpeg_bytes(2400, 1600)
    out, mime = _process_figure_image(big, "image/jpeg")
    assert mime == "image/jpeg"
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG"
    assert img.width == 1280


def test_figure_png_without_alpha_reencoded_as_jpeg():
    raw = _png_bytes(800, 600)
    out, mime = _process_figure_image(raw, "image/png")
    assert mime == "image/jpeg"
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG"
    assert img.width == 800
    assert img.height == 600


def test_figure_png_with_real_alpha_kept_as_png():
    rgba = Image.new("RGBA", (400, 300), color=(200, 50, 50, 128))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    raw = buf.getvalue()
    out, mime = _process_figure_image(raw, "image/png")
    assert mime == "image/png"
    img = Image.open(io.BytesIO(out))
    assert img.format == "PNG"
    assert img.mode in ("RGBA", "LA", "P")


def test_figure_rgba_with_opaque_alpha_treated_as_jpeg():
    rgba = Image.new("RGBA", (400, 300), color=(200, 50, 50, 255))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    raw = buf.getvalue()
    out, mime = _process_figure_image(raw, "image/png")
    assert mime == "image/jpeg"
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG"
    assert img.mode == "RGB"


def test_data_uri_figure_persisted_without_any_network(conn, fast_sleep):
    arxiv_id = "2301.00010"
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    body = _minimal_html_with_one_figure(f"data:image/png;base64,{tiny_png_b64}")

    seen_image_calls = []

    def handler(req):
        url = str(req.url)
        if "arxiv.org/html" in url:
            return _resp_html(body)
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.4")
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        # If we get here for what looks like an image URL, record it.
        seen_image_calls.append(url)
        return httpx.Response(404)

    with _client_with(handler) as client:
        fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=client,
            arxiv_lookup=lambda _id: _make_meta(),
        )

    assert seen_image_calls == [], f"unexpected image HTTP calls: {seen_image_calls}"

    count = conn.execute("SELECT COUNT(*) FROM figures").fetchone()[0]
    assert count == 1


def test_decompression_bomb_figure_skipped_others_succeed(monkeypatch):
    good_png = _png_bytes(100, 80)

    # Patch Image.open to raise on the first call, then defer to the real impl.
    real_open = Image.open
    call = {"n": 0}

    def patched(stream, *args, **kwargs):
        call["n"] += 1
        if call["n"] == 1:
            raise Image.DecompressionBombError("boom")
        return real_open(stream, *args, **kwargs)

    monkeypatch.setattr(Image, "open", patched)

    assert _process_figure_image(b"doesnt matter", "image/png") is None
    # Second call returns fine since the real open succeeds.
    result = _process_figure_image(good_png, "image/png")
    assert result is not None


def test_oversize_content_length_aborts_figure_and_continues():
    def handler(req):
        if "toobig" in str(req.url):
            return httpx.Response(
                200,
                content=b"junk",
                headers={"content-length": str(50 * 1024 * 1024), "content-type": "image/png"},
            )
        return httpx.Response(200, content=_png_bytes(10, 10), headers={"content-type": "image/png"})

    client = _client_with(handler)
    try:
        assert fp._download_figure(client, "http://example.com/toobig.png") is None
        ok = fp._download_figure(client, "http://example.com/ok.png")
        assert ok is not None
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Code repo discovery + URL normalization
# ---------------------------------------------------------------------------


def test_normalize_rejects_issues_url():
    assert _normalize_repo_url("https://github.com/foo/bar/issues/42") is None


def test_normalize_keeps_owner_repo_root():
    assert _normalize_repo_url("https://github.com/foo/bar") == "https://github.com/foo/bar"
    assert _normalize_repo_url("https://github.com/foo/bar.git") == "https://github.com/foo/bar"


def test_normalize_strips_trailing_punctuation():
    assert _normalize_repo_url("https://github.com/foo/bar.") == "https://github.com/foo/bar"
    assert _normalize_repo_url("https://github.com/foo/bar)") == "https://github.com/foo/bar"


def test_layer_priority_pwc_wins_over_html_scan(conn, fast_sleep):
    """PwC returns an official repo → layer 2 and 3 are skipped."""

    def handler(req):
        url = str(req.url)
        if "arxiv.org/html" in url:
            return _resp_html(
                "<html><body>see https://github.com/htmlowner/htmlrepo</body></html>"
            )
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.4")
        if "paperswithcode.com/api/v1/papers/?arxiv_id=" in url:
            return httpx.Response(200, json={"results": [{"id": "some-paper-slug"}]})
        if "paperswithcode.com/api/v1/papers/some-paper-slug/repositories" in url:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://github.com/pwcowner/pwcrepo",
                            "is_official": True,
                            "stars": 100,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    with _client_with(handler) as client:
        pm = fetch(
            conn=conn,
            arxiv_id="2301.00055",
            client=client,
            arxiv_lookup=lambda _id: _make_meta(),
        )
    assert pm.code_repo == "https://github.com/pwcowner/pwcrepo"


def test_layer_3_discovers_repo_in_arxiv_comment(conn, fast_sleep):
    """PwC and HTML scan return nothing; arxiv .comment has the URL."""
    meta = _make_meta(comment="Code: https://github.com/commenter/fromcomment. See paper.")

    def handler(req):
        url = str(req.url)
        if "arxiv.org/html" in url:
            return _resp_html("<html><body>no repo here</body></html>")
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.4")
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    with _client_with(handler) as client:
        pm = fetch(
            conn=conn,
            arxiv_id="2301.00066",
            client=client,
            arxiv_lookup=lambda _id: meta,
        )
    assert pm.code_repo == "https://github.com/commenter/fromcomment"


# ---------------------------------------------------------------------------
# Soft dedup
# ---------------------------------------------------------------------------


def test_soft_dedup_warning_logged_when_same_content_hash_different_id(
    conn, fast_sleep, caplog
):
    pdf_bytes = b"%PDF-1.4\nimportant content\n"

    def handler(req):
        url = str(req.url)
        if "arxiv.org/html" in url:
            return _resp_html(_minimal_html_with_one_figure("data:image/png;base64,AA=="))
        if "/pdf/" in url:
            return httpx.Response(200, content=pdf_bytes)
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    # First paper
    with _client_with(handler) as client:
        fetch(
            conn=conn,
            arxiv_id="2301.00088",
            client=client,
            arxiv_lookup=lambda _id: _make_meta(title="Paper A"),
        )

    # The `lodestone.*` logger has propagate=False, so caplog's root handler
    # never sees records unless we attach to the module logger directly.
    logger = logging.getLogger("lodestone.scripts.fetch_paper")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="lodestone.scripts.fetch_paper"):
            with _client_with(handler) as client:
                fetch(
                    conn=conn,
                    arxiv_id="2301.00089",
                    client=client,
                    arxiv_lookup=lambda _id: _make_meta(title="Paper B"),
                        )
    finally:
        logger.removeHandler(caplog.handler)
    assert any("soft-dedup" in r.message for r in caplog.records), (
        f"expected soft-dedup warning; got {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Force re-fetch
# ---------------------------------------------------------------------------


def test_force_refetch_preserves_slug_and_clears_children(conn, fast_sleep):
    """force=True re-runs the pipeline but keeps paper_name + ingested_at,
    and clears dependent rows (figures, entities, paper_topics, FTS) so
    the paper DELETE doesn't trip FK constraints."""
    arxiv_id = "2301.45678"
    data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

    def handler(req):
        host = req.url.host
        path = req.url.path
        if host == "arxiv.org" and path.startswith("/html"):
            return _resp_html(_minimal_html_with_one_figure(data_uri))
        if path.startswith("/pdf"):
            return httpx.Response(200, content=b"%PDF-1.4 first")
        if host == "paperswithcode.com":
            return httpx.Response(404)
        return httpx.Response(404)

    with _client_with(handler) as client:
        first = fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            client=client,
            arxiv_lookup=lambda _id: _make_meta(),
        )

    # Simulate downstream pipeline stages populating children. A naive
    # re-fetch with FK=ON would trip FOREIGN KEY on the paper DELETE.
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", ("rag",))
    paper_id = conn.execute(
        "SELECT id FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO canonical_terms (domain, term_type, entity_type, "
        " canonical_name, first_seen_in) "
        "VALUES (?, 'entity', 'method', 'Some Entity', ?)",
        ("rag", first.paper_name),
    )
    seeded_term_id = conn.execute(
        "SELECT id FROM canonical_terms WHERE canonical_name = 'Some Entity'"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO term_aliases "
        " (term_id, alias, source_paper, match_tier) "
        "VALUES (?, 'some_entity_alt', ?, 2)",
        (seeded_term_id, first.paper_name),
    )
    conn.execute(
        "INSERT INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
        (paper_id, "rag", "tree retrieval"),
    )

    def handler2(req):
        host = req.url.host
        path = req.url.path
        if host == "arxiv.org" and path.startswith("/html"):
            return _resp_html(_minimal_html_with_one_figure(data_uri))
        if path.startswith("/pdf"):
            return httpx.Response(200, content=b"%PDF-1.4 second different")
        if host == "paperswithcode.com":
            return httpx.Response(404)
        return httpx.Response(404)

    with _client_with(handler2) as client:
        refetched = fetch(
            conn=conn,
            arxiv_id=arxiv_id,
            force=True,
            client=client,
            arxiv_lookup=lambda _id: _make_meta(),
        )

    # Slug + ingested_at preserved
    assert refetched.paper_name == first.paper_name
    assert refetched.ingested_at == first.ingested_at

    # Children cleared + re-seeded from phase 2.
    paper_id = conn.execute(
        "SELECT id FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()[0]
    paper_name = conn.execute(
        "SELECT paper_name FROM papers WHERE id = ?", (paper_id,)
    ).fetchone()[0]
    ents = conn.execute(
        "SELECT COUNT(*) FROM term_aliases WHERE source_paper = ?",
        (paper_name,),
    ).fetchone()[0]
    topics = conn.execute(
        "SELECT COUNT(*) FROM paper_topics WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0]
    assert ents == 0
    assert topics == 0


# ---------------------------------------------------------------------------
# Two-phase: no transaction during network
# ---------------------------------------------------------------------------


def test_no_db_transaction_open_during_network(conn, fast_sleep, monkeypatch):
    """Spy on `transaction()` and on the MockTransport so we can assert all
    HTTP calls landed before the single phase-2 BEGIN."""
    events: list[str] = []

    original_transaction = fp.transaction

    def spy_transaction(c):
        events.append("TXN_ENTER")
        return original_transaction(c)

    monkeypatch.setattr(fp, "transaction", spy_transaction)

    def handler(req):
        events.append(f"HTTP {req.url.host}{req.url.path}")
        url = str(req.url)
        if "arxiv.org/html" in url:
            return _resp_html(_minimal_html_with_one_figure("data:image/png;base64,AA=="))
        if "/pdf/" in url:
            return httpx.Response(200, content=b"%PDF-1.4")
        if "paperswithcode.com" in url:
            return httpx.Response(404)
        return httpx.Response(404)

    with _client_with(handler) as client:
        fetch(
            conn=conn,
            arxiv_id="2301.00100",
            client=client,
            arxiv_lookup=lambda _id: _make_meta(),
        )

    http_indices = [i for i, e in enumerate(events) if e.startswith("HTTP")]
    txn_indices = [i for i, e in enumerate(events) if e == "TXN_ENTER"]
    assert http_indices, "expected HTTP calls"
    assert txn_indices, "expected exactly one transaction enter"
    assert len(txn_indices) == 1, f"expected 1 txn enter; saw {len(txn_indices)}"
    assert max(http_indices) < txn_indices[0], (
        f"HTTP happened after transaction opened; events={events}"
    )
