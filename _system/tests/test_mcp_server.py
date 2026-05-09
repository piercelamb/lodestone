"""Tests for the lodestone stdio MCP server.

Two layers:

1. **Unit tests** call dispatch helpers directly with the seeded sqlite
   fixture and assert content-block shape (text + image blocks, the
   labeled markers, the size/count caps, and fail-silent behavior).
2. **Protocol smoke test** spawns ``lodestone-mcp`` as a subprocess, drives
   the JSON-RPC handshake, and verifies the wire envelope — handshake
   fields, tools/list shape, error codes, and the issue-#35287 mitigation
   (initialize succeeds even when the DB is missing).

Reuses the heavy ``seeded_db`` corpus from ``test_search.py`` to keep
fixture maintenance in one place.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import sqlite3
from pathlib import Path

import pytest

from _system.scripts import mcp_server

# Re-use the seeded_db fixture and PNG constant from test_search by
# importing the helpers. pytest will pick the fixture up via conftest
# discovery — but since seeded_db is defined in test_search.py we re-seed
# locally here to avoid cross-module fixture coupling.
from _system.tests.test_search import (
    _PNG_1x1,
    _insert_figure,
    _insert_paper,
    _insert_paper_reference,
    _insert_post_for_citations,
    _insert_post_reference,
    _insert_sections_for_md,
    _seed_domain,
    seeded_citations_db,  # noqa: F401 — re-exported as a pytest fixture
)


# ---------------------------------------------------------------------------
# Local fixtures: a minimal seeded DB tailored to the figure-attach paths
# ---------------------------------------------------------------------------


_MD_WITH_FIGS = (
    "# Abstract\n\n"
    "Tree-of-Thoughts solves the Game of 24 with deliberate search.\n\n"
    "# Method\n\n"
    "We use a tree expansion (figure:1) with a value heuristic.\n\n"
    "# Experiments\n\n"
    "Results visualized as (figure:2). Game of 24 outperforms baselines.\n"
)

_MD_NO_FIGS = (
    "# Abstract\n\n"
    "A paper without any figure references at all.\n\n"
    "# Method\n\n"
    "Just words and equations.\n"
)


@pytest.fixture
def fig_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Seeded DB with two papers: one with figure refs + 2 figures, one
    plain. Independent of the larger seeded_db so figure-attach tests stay
    focused.
    """
    _seed_domain(conn, "rag")
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES ('rag', 'reasoning', NULL)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES ('rag', 'misc', NULL)"
    )
    p_id = _insert_paper(
        conn,
        arxiv_id="2305.10601",
        paper_name="tot_2023",
        title="Tree of Thoughts",
        abstract="Deliberate search with LLMs.",
        markdown=_MD_WITH_FIGS,
        domain="rag",
        collection="reasoning",
        needs_review=0,
        ingested_at="2023-05-17T00:00:00+00:00",
    )
    _insert_sections_for_md(
        conn,
        paper_id=p_id,
        domain="rag",
        paper_name="tot_2023",
        markdown=_MD_WITH_FIGS,
    )
    _insert_figure(
        conn,
        paper_id=p_id,
        figure_number=1,
        display_number="Figure 1",
        figure_id="F1",
        caption="Figure 1: Tree expansion in ToT.",
        section_context="Method",
        image=_PNG_1x1,
    )
    _insert_figure(
        conn,
        paper_id=p_id,
        figure_number=2,
        display_number="Figure 2",
        figure_id="F2",
        caption="Figure 2: ToT in a Game of 24.",
        section_context="Experiments",
        image=_PNG_1x1,
    )

    plain_id = _insert_paper(
        conn,
        arxiv_id="2401.99999",
        paper_name="plain_2024",
        title="Plain Paper",
        abstract="No figures here.",
        markdown=_MD_NO_FIGS,
        domain="rag",
        collection="misc",
        needs_review=0,
        ingested_at="2024-01-01T00:00:00+00:00",
    )
    _insert_sections_for_md(
        conn,
        paper_id=plain_id,
        domain="rag",
        paper_name="plain_2024",
        markdown=_MD_NO_FIGS,
    )
    return conn


# ---------------------------------------------------------------------------
# Helpers: drive the dispatcher from the inside (no subprocess)
# ---------------------------------------------------------------------------


def _make_state(conn: sqlite3.Connection) -> mcp_server._ServerState:
    """Build a _ServerState already wired to a sqlite connection — bypasses
    the env-var resolution path used by real startup.
    """
    state = mcp_server._ServerState()
    state.conn = conn
    state.db_path = Path(":memory:")
    state.startup_error = None
    return state


def _call(state: mcp_server._ServerState, tool: str, args: dict, msg_id: int = 1) -> dict:
    return mcp_server._handle_tools_call(
        state,
        {"jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
         "params": {"name": tool, "arguments": args}},
    )


def _content(resp: dict) -> list[dict]:
    return resp["result"]["content"]


def _is_error(resp: dict) -> bool:
    # Image-bearing results omit isError per the test_mcp_image envelope —
    # absent counts as not-an-error.
    return bool(resp["result"].get("isError"))


def _has_image(resp: dict) -> bool:
    return any(b.get("type") == "image" for b in resp["result"].get("content", []))


def _image_blocks(content: list[dict]) -> list[dict]:
    return [b for b in content if b.get("type") == "image"]


def _text_blocks(content: list[dict]) -> list[dict]:
    return [b for b in content if b.get("type") == "text"]


# ===========================================================================
# Figure ref extraction
# ===========================================================================


class TestExtractFigureRefs:
    def test_finds_refs_in_top_level_text(self):
        payload = {"text": "see (figure:1) and also (figure:3)"}
        assert mcp_server._extract_figure_refs(payload) == [1, 3]

    def test_finds_refs_in_nested_lists_and_dicts(self):
        payload = {
            "results": [
                {"snippet": "first (figure:5) snippet"},
                {"snippet": "no refs here"},
                {"sections": [{"snippet": "and (figure:2) here"}]},
            ],
        }
        assert mcp_server._extract_figure_refs(payload) == [5, 2]

    def test_dedupes_in_order(self):
        payload = {"text": "(figure:1) (figure:2) (figure:1) (figure:3) (figure:2)"}
        assert mcp_server._extract_figure_refs(payload) == [1, 2, 3]

    def test_no_refs_returns_empty(self):
        assert mcp_server._extract_figure_refs({"text": "no refs"}) == []


# ===========================================================================
# Pack result — content block shape + structuredContent
# ===========================================================================


class TestPackResult:
    def test_text_block_carries_payload_json(self, fig_db):
        payload = {"mode": "toc", "slug": "tot_2023", "toc": []}
        out = mcp_server._pack_result(payload, fig_db)
        text = _text_blocks(out["content"])[0]["text"]
        assert json.loads(text) == payload
        assert out["isError"] is False

    def test_structured_content_matches_text_block_when_no_image(self, fig_db):
        payload = {"mode": "toc", "slug": "tot_2023", "toc": [{"level": 1, "title": "X"}]}
        out = mcp_server._pack_result(payload, fig_db)
        text = _text_blocks(out["content"])[0]["text"]
        assert out["structuredContent"] == json.loads(text)

    def test_image_response_uses_minimal_envelope(self, fig_db):
        # Image-bearing responses mirror the test_mcp_image.py shape exactly:
        # only `content`, no `structuredContent`, no `isError` — that's the
        # envelope under which Claude Code is observed to surface image
        # blocks as multimodal input.
        payload = {
            "mode": "read", "status": "ok", "slug": "tot_2023",
            "section": None, "text": _MD_WITH_FIGS,
        }
        out = mcp_server._pack_result(payload, fig_db)
        assert _image_blocks(out["content"]), "expected image blocks for this fixture"
        assert "structuredContent" not in out
        assert "isError" not in out

    def test_no_image_blocks_when_no_refs(self, fig_db):
        payload = {
            "mode": "read", "status": "ok", "slug": "plain_2024",
            "section": None, "text": _MD_NO_FIGS,
        }
        out = mcp_server._pack_result(payload, fig_db)
        assert _image_blocks(out["content"]) == []


# ===========================================================================
# Read tool — figure attach
# ===========================================================================


class TestReadAttachesFigures:
    def test_read_attaches_image_blocks_for_figure_refs(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "read", {"slug": "tot_2023"})
        assert not _is_error(resp), resp
        content = _content(resp)
        # 1 JSON text block + (text marker + image) per figure × 2.
        imgs = _image_blocks(content)
        assert len(imgs) == 2
        for img in imgs:
            assert img["mimeType"] == "image/png"
            # Round-trip the b64 to confirm valid bytes.
            assert base64.b64decode(img["data"]) == _PNG_1x1
        # Markers contain the contract tokens for the model.
        all_text = "\n".join(b["text"] for b in _text_blocks(content))
        assert "figure=1" in all_text
        assert "figure=2" in all_text
        assert "paper_id=" in all_text

    def test_read_no_figure_refs_no_image_blocks(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "read", {"slug": "plain_2024"})
        assert not _is_error(resp), resp
        assert _image_blocks(_content(resp)) == []

    def test_read_section_slice_only_attaches_refs_in_slice(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "read", {"slug": "tot_2023", "section": "Method"})
        content = _content(resp)
        # Method section only has (figure:1).
        imgs = _image_blocks(content)
        assert len(imgs) == 1


# ===========================================================================
# Figure tool — direct image fetch
# ===========================================================================


class TestFigureTool:
    def test_figure_returns_one_image_block(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "figure", {"paper": "tot_2023", "n": "1"})
        assert not _is_error(resp), resp
        imgs = _image_blocks(_content(resp))
        assert len(imgs) == 1
        assert imgs[0]["mimeType"] == "image/png"

    def test_figure_missing_raises_isError(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "figure", {"paper": "tot_2023", "n": "999"})
        assert _is_error(resp)


# ===========================================================================
# BM25 tool — best-effort attach
# ===========================================================================


class TestBm25Attach:
    def test_bm25_attaches_figures_landed_in_snippet(self, fig_db):
        # Seed a section whose body explicitly contains a figure ref so a
        # BM25 hit's snippet window catches it.
        p_id = fig_db.execute(
            "SELECT id FROM papers WHERE paper_name = ?", ("tot_2023",)
        ).fetchone()[0]
        fig_db.execute(
            """
            INSERT INTO sections
                (paper_id, domain, paper_name, section_title, section_level, body)
            VALUES (?, 'rag', 'tot_2023', 'Snippet With Ref', '2',
                    'this snippet mentions tree expansion (figure:1) directly')
            """,
            (p_id,),
        )
        state = _make_state(fig_db)
        resp = _call(state, "bm25", {"query": "tree expansion"})
        content = _content(resp)
        imgs = _image_blocks(content)
        # At least one image block should be attached for figure 1.
        assert len(imgs) >= 1

    def test_bm25_no_attach_when_no_refs_in_snippet(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "bm25", {"query": "deliberate"})
        # The Abstract snippet doesn't carry a figure ref, so there should be
        # no image blocks even though the paper has figures in other sections.
        imgs = _image_blocks(_content(resp))
        assert imgs == []


# ===========================================================================
# Failure-silence guarantees
# ===========================================================================


class TestSilentFailures:
    def test_per_figure_failure_is_silent(self, fig_db, monkeypatch):
        # Make base64 encoding raise on every call — each per-figure
        # encode fails, but the dispatcher must still produce a healthy
        # tool result (text + structuredContent), no image blocks.
        def boom(_data):
            raise ValueError("simulated b64 encode failure")

        monkeypatch.setattr(mcp_server.base64, "b64encode", boom)
        state = _make_state(fig_db)
        resp = _call(state, "read", {"slug": "tot_2023"})
        assert not _is_error(resp)
        assert _image_blocks(_content(resp)) == []

    def test_oversize_blob_is_skipped_with_marker(self, fig_db, monkeypatch):
        # Force the per-blob limit to 1 byte so our 1×1 PNG is "oversize".
        monkeypatch.setenv("LODESTONE_MAX_FIGURE_BYTES", "1")
        state = _make_state(fig_db)
        resp = _call(state, "read", {"slug": "tot_2023"})
        assert not _is_error(resp)
        content = _content(resp)
        # No image blocks (all skipped). Markers explain the skip.
        assert _image_blocks(content) == []
        text_lump = "\n".join(b["text"] for b in _text_blocks(content))
        assert "skipped" in text_lump
        assert "exceeds" in text_lump

    def test_response_caps_image_block_count(self, fig_db, monkeypatch):
        # Cap to 1 figure per response — paper has 2 refs, so 1 image block
        # plus an overflow marker for figure 2.
        monkeypatch.setenv("LODESTONE_MAX_FIGURES_PER_RESPONSE", "1")
        state = _make_state(fig_db)
        resp = _call(state, "read", {"slug": "tot_2023"})
        assert not _is_error(resp)
        content = _content(resp)
        imgs = _image_blocks(content)
        assert len(imgs) == 1
        text_lump = "\n".join(b["text"] for b in _text_blocks(content))
        assert "omitted_figures" in text_lump


# ===========================================================================
# Soft-failure passthrough
# ===========================================================================


class TestSoftFailures:
    def test_section_not_found_is_not_isError(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "read", {"slug": "tot_2023", "section": "NoSuchSection"})
        assert not _is_error(resp)
        text = _text_blocks(_content(resp))[0]["text"]
        payload = json.loads(text)
        assert payload["status"] == "section_not_found"
        # No image blocks on a soft-fail, so structuredContent is included.
        assert resp["result"]["structuredContent"]["status"] == "section_not_found"

    def test_unknown_slug_raises_isError(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "read", {"slug": "no_such_paper"})
        assert _is_error(resp)


# ===========================================================================
# Tool-name and arg validation
# ===========================================================================


class TestProtocolValidation:
    def test_unknown_tool_returns_isError(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "no_such_tool", {})
        assert _is_error(resp)
        assert "unknown tool" in _text_blocks(_content(resp))[0]["text"]

    def test_missing_required_arg_returns_invalid_params(self, fig_db):
        state = _make_state(fig_db)
        resp = mcp_server._handle_tools_call(
            state,
            {"jsonrpc": "2.0", "id": 9, "method": "tools/call",
             "params": {"name": "read", "arguments": {}}},
        )
        assert "error" in resp
        assert resp["error"]["code"] == mcp_server._ERR_INVALID_PARAMS


# ===========================================================================
# Inode pin guard — catches "file replaced under us" / ghost-DB bug
# ===========================================================================


class TestDbInodePinGuard:
    """The ``opened_inode`` pin compares the inode we opened against the
    inode the path resolves to right now. If they diverge, another process
    swapped the file underneath this server (the "ghost DB" footgun) and
    every write is landing on an orphaned inode.
    """

    def _build_state_at(self, db_path: Path) -> mcp_server._ServerState:
        """Replicate the relevant bits of configure() against a real file
        path so we can exercise the inode comparison.
        """
        from _system.db.connection import get_conn
        from _system.db.migrations import init_db

        state = mcp_server._ServerState()
        state.db_path = db_path
        state.conn = get_conn(db_path)
        init_db(state.conn)
        state.opened_inode = db_path.stat().st_ino
        return state

    def test_no_op_when_opened_inode_is_none(self, fig_db):
        # _make_state leaves opened_inode=None and uses :memory: — the
        # full existing test suite implicitly relies on this skipping the
        # check, but make it explicit.
        state = _make_state(fig_db)
        assert state.opened_inode is None
        assert mcp_server._check_db_inode_pinned(state) is None

    def test_no_op_for_memory_db_even_with_inode_set(self, fig_db):
        state = _make_state(fig_db)
        state.opened_inode = 12345
        assert mcp_server._check_db_inode_pinned(state) is None

    def test_passes_when_inode_unchanged(self, tmp_path):
        state = self._build_state_at(tmp_path / "lodestone.db")
        try:
            assert mcp_server._check_db_inode_pinned(state) is None
            # Drive a tools/call to confirm the per-call hook doesn't
            # spuriously flag a swap when nothing has moved. The tool
            # itself may or may not error on an empty DB; we only care
            # that no swap-detection text leaks through.
            resp = _call(state, "overview", {})
            content = _text_blocks(_content(resp))
            for block in content:
                assert "was replaced underneath" not in block["text"]
                assert "no longer accessible" not in block["text"]
        finally:
            state.close()

    def test_detects_replace(self, tmp_path):
        db_path = tmp_path / "lodestone.db"
        state = self._build_state_at(db_path)
        try:
            # Atomically replace the file at the path. ``state.conn`` keeps
            # the original (now-orphaned) inode; the path resolves to the
            # new one.
            db_path.unlink()
            replacement = tmp_path / "replacement.db"
            replacement.write_bytes(b"")
            os.replace(replacement, db_path)
            assert db_path.stat().st_ino != state.opened_inode

            err = mcp_server._check_db_inode_pinned(state)
            assert err is not None
            assert "was replaced underneath" in err

            # And it surfaces through tools/call as a tool error.
            resp = _call(state, "browse", {"which": "domains"})
            assert _is_error(resp)
            text = _text_blocks(_content(resp))[0]["text"]
            assert "was replaced underneath" in text
        finally:
            state.close()

    def test_detects_unlink_without_recreate(self, tmp_path):
        db_path = tmp_path / "lodestone.db"
        state = self._build_state_at(db_path)
        try:
            db_path.unlink()
            err = mcp_server._check_db_inode_pinned(state)
            assert err is not None
            assert "no longer accessible" in err
        finally:
            state.close()


# ===========================================================================
# Tools list — schema integrity
# ===========================================================================


class TestToolsList:
    def test_tools_list_returns_twenty_tools(self):
        out = mcp_server._handle_tools_list({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = out["result"]["tools"]
        assert len(tools) == 20
        names = {t["name"] for t in tools}
        assert names == {
            "search", "bm25", "lookup", "browse", "overview", "collection",
            "toc", "toc_many", "read", "figure", "repo_tree", "read_code",
            "repo", "citations", "tables", "schema", "query",
            "ingest_paper", "ingest_repo", "ingest_post",
        }
        for t in tools:
            assert isinstance(t["description"], str) and t["description"]
            schema = t["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_no_double_underscore_in_tool_names(self):
        # Claude Code splits on `__` to form mcp__lodestone__<tool>; embedded
        # `__` would corrupt that prefix.
        for t in mcp_server.TOOLS:
            assert "__" not in t["name"], t["name"]


# ===========================================================================
# citations tool — outbound buckets, inbound pagination, soft statuses
# ===========================================================================


class TestCitationsTool:
    def test_outbound_dispatch_returns_buckets(self, seeded_citations_db):
        state = _make_state(seeded_citations_db)
        resp = _call(state, "citations", {"slug": "citing_2024"})
        assert not _is_error(resp), resp
        payload = resp["result"]["structuredContent"]
        assert payload["mode"] == "citations"
        assert payload["status"] == "ok"
        assert payload["direction"] == "outbound"
        assert payload["resolved_count"] == 2
        assert payload["missing_count"] == 1
        assert payload["unresolvable_count"] == 1

    def test_inbound_dispatch_returns_paginated_results(
        self, seeded_citations_db
    ):
        state = _make_state(seeded_citations_db)
        resp = _call(
            state,
            "citations",
            {"slug": "cited_2023", "direction": "inbound", "limit": 1},
        )
        assert not _is_error(resp), resp
        payload = resp["result"]["structuredContent"]
        assert payload["status"] == "ok"
        assert payload["direction"] == "inbound"
        assert len(payload["results"]) == 1
        assert payload["total_hits"] == 2
        assert payload["has_more"] is True

    def test_invalid_direction_returns_isError(self, seeded_citations_db):
        # Garbage direction → ValueError → MCP boundary marks isError=true
        # (hard error, not a recoverable soft-fail).
        state = _make_state(seeded_citations_db)
        resp = _call(
            state,
            "citations",
            {"slug": "citing_2024", "direction": "sideways"},
        )
        assert _is_error(resp)

    def test_unsupported_direction_marks_isError_false(
        self, seeded_citations_db
    ):
        # Post + inbound → unsupported_direction soft-status, which must
        # pass through as isError=false so the agent can recover from
        # the diagnostic instead of seeing a hard tool error.
        state = _make_state(seeded_citations_db)
        resp = _call(
            state,
            "citations",
            {"slug": "post_2024", "direction": "inbound"},
        )
        assert not _is_error(resp), resp
        payload = resp["result"]["structuredContent"]
        assert payload["status"] == "unsupported_direction"

    def test_unknown_slug_marks_isError_false(self, seeded_citations_db):
        # Verifies `not_found` is now in _SOFT_FAILURE_STATUSES — without
        # the registration this would land as isError=true.
        state = _make_state(seeded_citations_db)
        resp = _call(state, "citations", {"slug": "no_such_slug"})
        assert not _is_error(resp), resp
        payload = resp["result"]["structuredContent"]
        assert payload["status"] == "not_found"


# ===========================================================================
# search tool — first-pass composite
# ===========================================================================


class TestSearchTool:
    """The `search` tool is text-only by design (AttachMode.NONE) — it
    fans out to taxonomy + sections-bm25 + readmes-bm25 and never inlines
    figures, even when section snippets contain (figure:N) refs."""

    def test_search_returns_three_buckets(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "search", {"query": "Tree of Thoughts"})
        assert not _is_error(resp), resp
        assert not _has_image(resp), "search must not attach figure images"

        # Text block is now markdown (token-cheap orientation digest).
        text_blocks = _text_blocks(_content(resp))
        assert len(text_blocks) == 1
        text = text_blocks[0]["text"]
        assert text.startswith("# search")
        assert "## taxonomy" in text
        assert "## sections" in text
        assert "## readmes" in text

        # Full JSON payload still rides on structuredContent for callers
        # that want it (CLI, structured pipelines).
        payload = resp["result"]["structuredContent"]
        assert payload["mode"] == "search"
        assert "taxonomy" in payload
        assert "sections" in payload
        assert "readmes" in payload

    def test_search_no_image_blocks_even_with_figure_refs(self, fig_db):
        # _MD_WITH_FIGS has (figure:1) in the Method section. A bm25-scope
        # call would attach the image; search must NOT.
        state = _make_state(fig_db)
        resp = _call(state, "search", {"query": "tree expansion"})
        assert not _has_image(resp), (
            "search response carried image blocks despite AttachMode.NONE"
        )

    def test_search_empty_query_soft_fail(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "search", {"query": ""})
        # Soft-failure: payload carries the diagnostic, isError stays false.
        assert not _is_error(resp)
        # Markdown text surfaces the soft-fail header for the agent.
        text = _text_blocks(_content(resp))[0]["text"]
        assert text.startswith("# search (empty query)")
        # Structured payload still carries the machine-readable status.
        assert resp["result"]["structuredContent"]["status"] == "empty_query"

    def test_search_missing_query_argument(self, fig_db):
        # Missing required arg surfaces as JSON-RPC InvalidParams (code -32602)
        # via the dispatcher's KeyError path.
        state = _make_state(fig_db)
        resp = mcp_server._handle_tools_call(
            state,
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "search", "arguments": {}}},
        )
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_search_domain_filter_passes_through(self, fig_db):
        state = _make_state(fig_db)
        # 'rag' has the seeded paper; 'bogus' should narrow everything to empty.
        resp = _call(state, "search", {"query": "Tree", "domain": "bogus"})
        payload = resp["result"]["structuredContent"]
        assert payload["domain"] == "bogus"
        assert payload["taxonomy"] == []
        assert payload["sections"] == []
        assert payload["readmes"] == []

    def test_search_operator_query(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "search", {"query": "Tree OR thoughts"})
        assert not _is_error(resp)
        payload = resp["result"]["structuredContent"]
        assert payload["mode"] == "search"
        # OR should at least match the seeded tot_2023 paper.
        names = {g["paper_name"] for g in payload["sections"]}
        assert "tot_2023" in names

    def test_search_paper_qualifier(self, fig_db):
        state = _make_state(fig_db)
        # paper: qualifier narrows the bm25 sections subquery to one paper.
        resp = _call(state, "search", {"query": "paper:tot_2023 tree"})
        payload = resp["result"]["structuredContent"]
        names = {g["paper_name"] for g in payload["sections"]}
        # Either no hits, or only tot_2023.
        assert names <= {"tot_2023"}, payload

    def test_search_malformed_query_is_soft_fail(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "search", {"query": '"unclosed'})
        assert not _is_error(resp)
        text = _text_blocks(_content(resp))[0]["text"]
        assert text.startswith("# search (malformed query)")
        assert resp["result"]["structuredContent"]["status"] == "malformed_query"

    def test_search_query_array_fans_out(self, fig_db):
        # Pass query as an array: each string runs as its own mode_search,
        # results are concatenated under the multi envelope.
        state = _make_state(fig_db)
        resp = _call(state, "search", {"query": ["Tree", "Thoughts"]})
        assert not _is_error(resp)
        assert not _has_image(resp), "search must not attach figure images"
        payload = resp["result"]["structuredContent"]
        assert payload["mode"] == "search"
        assert payload["multi"] is True
        assert payload["queries"] == ["Tree", "Thoughts"]
        assert len(payload["results"]) == 2
        # Each sub-payload is a full mode_search envelope.
        for sub in payload["results"]:
            assert sub["mode"] == "search"
            assert "taxonomy" in sub
            assert "sections" in sub
            assert "readmes" in sub
        # Markdown text reflects the multi envelope.
        text = _text_blocks(_content(resp))[0]["text"]
        assert text.startswith("# search (multi: 2 queries)")
        assert "## query 1: 'Tree'" in text
        assert "## query 2: 'Thoughts'" in text

    def test_search_query_array_with_per_query_qualifiers(self, fig_db):
        # Per-query qualifiers route independently — surface:taxonomy on
        # one, surface:sections on the other.
        state = _make_state(fig_db)
        resp = _call(state, "search", {"query": [
            "surface:taxonomy Tree",
            "surface:sections tree",
        ]})
        assert not _is_error(resp)
        payload = resp["result"]["structuredContent"]
        a, b = payload["results"]
        assert "taxonomy" in a and "sections" not in a
        assert "sections" in b and "taxonomy" not in b

    def test_search_query_array_mixed_failure(self, fig_db):
        # One good query + one malformed: envelope itself stays OK
        # (isError=false), the malformed one surfaces inside its
        # per-query payload.
        state = _make_state(fig_db)
        resp = _call(state, "search", {"query": ["Tree", '"unclosed']})
        assert not _is_error(resp)
        payload = resp["result"]["structuredContent"]
        assert payload["multi"] is True
        good, bad = payload["results"]
        assert good.get("status") is None
        assert bad["status"] == "malformed_query"

    def test_search_query_array_too_many_soft_fails(self, fig_db):
        state = _make_state(fig_db)
        # Cap is 8 (also enforced by the JSON Schema maxItems, but the
        # mode-level guard is the real one — schema is advisory at this
        # MCP transport).
        resp = _call(state, "search", {"query": ["x"] * 9})
        assert not _is_error(resp)
        payload = resp["result"]["structuredContent"]
        assert payload["status"] == "malformed_query"
        assert "too many" in payload["error"]


# ===========================================================================
# bm25 tool — query syntax through the MCP boundary
# ===========================================================================


class TestBm25SyntaxAtMcp:
    def test_bm25_paper_qualifier(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "bm25", {"query": "paper:tot_2023 tree"})
        assert not _is_error(resp)
        # bm25 with figure refs in snippet returns the image envelope (no
        # structuredContent), so parse the JSON text block instead.
        text = _text_blocks(_content(resp))[0]["text"]
        payload = json.loads(text)
        names = {g["paper_name"] for g in payload.get("results", [])}
        assert names <= {"tot_2023"}, payload

    def test_bm25_malformed_query_is_soft_fail(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "bm25", {"query": "/regex.*/"})
        assert not _is_error(resp)
        payload = resp["result"]["structuredContent"]
        assert payload["status"] == "malformed_query"
        assert "regex" in payload["error"].lower()


# ===========================================================================
# lookup tool — GH-syntax canonical FTS through the MCP boundary
# ===========================================================================


class TestLookupTool:
    """The `lookup` tool is text-only (AttachMode.NONE) and FTS5-only —
    no embedder loading. It accepts the same GitHub-flavored query syntax
    as `search`/`bm25` plus the `kind:` qualifier."""

    def test_lookup_schema_only_requires_query(self):
        tools = {t["name"]: t for t in mcp_server.TOOLS}
        schema = tools["lookup"]["inputSchema"]
        assert schema["required"] == ["query"]
        assert "query" in schema["properties"]
        assert "domain" in schema["properties"]
        assert "limit" in schema["properties"]
        # The old `kind` arg is gone — kind is now a `kind:` qualifier
        # inside the query string.
        assert "kind" not in schema["properties"]
        # The old `term` arg is gone — replaced by `query`.
        assert "term" not in schema["properties"]

    def test_lookup_returns_lookup_envelope(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "lookup", {"query": "tree"})
        assert not _is_error(resp)
        assert not _has_image(resp), "lookup must not attach figure images"
        payload = resp["result"]["structuredContent"]
        assert payload["mode"] == "lookup"
        assert payload["query"] == "tree"
        assert "hits" in payload
        assert isinstance(payload["hits"], list)

    def test_lookup_missing_query_argument(self, fig_db):
        state = _make_state(fig_db)
        resp = mcp_server._handle_tools_call(
            state,
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "lookup", "arguments": {}}},
        )
        assert "error" in resp
        assert resp["error"]["code"] == -32602

    def test_lookup_empty_query_soft_fail(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "lookup", {"query": ""})
        assert not _is_error(resp)
        payload = resp["result"]["structuredContent"]
        assert payload["status"] == "empty_query"

    def test_lookup_paper_qualifier_rejected(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "lookup", {"query": "paper:tot_2023 tree"})
        assert not _is_error(resp)
        payload = resp["result"]["structuredContent"]
        assert payload["status"] == "malformed_query"
        assert "paper" in payload["error"]

    def test_lookup_kind_qualifier_passes_through(self, fig_db):
        # `kind:` is the canonical narrowing qualifier on lookup. Even with
        # no canonicals seeded in fig_db, the structured envelope should
        # still come back well-formed (empty hits is normal).
        state = _make_state(fig_db)
        resp = _call(state, "lookup", {"query": "kind:entity tree"})
        assert not _is_error(resp)
        payload = resp["result"]["structuredContent"]
        assert payload["mode"] == "lookup"
        assert payload["kind"] == "entity"


# ===========================================================================
# toc_many tool — multi-source TOC through the MCP boundary
# ===========================================================================


class TestTocManyTool:
    def test_toc_many_returns_per_source_results(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(
            state, "toc_many", {"slugs": ["tot_2023", "plain_2024"]},
        )
        assert not _is_error(resp)
        assert not _has_image(resp), "toc_many must not attach figure images"
        payload = resp["result"]["structuredContent"]
        assert payload["mode"] == "toc_many"
        assert payload["slugs"] == ["tot_2023", "plain_2024"]
        assert {r["slug"] for r in payload["results"]} == {
            "tot_2023", "plain_2024",
        }
        assert payload["missing"] == []

    def test_toc_many_collects_missing(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(
            state, "toc_many",
            {"slugs": ["tot_2023", "no_such_paper"]},
        )
        assert not _is_error(resp)
        payload = resp["result"]["structuredContent"]
        assert {r["slug"] for r in payload["results"]} == {"tot_2023"}
        assert payload["missing"] == ["no_such_paper"]

    def test_toc_many_dedupes_duplicates(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(
            state, "toc_many",
            {"slugs": ["tot_2023", "tot_2023", "plain_2024"]},
        )
        payload = resp["result"]["structuredContent"]
        assert payload["slugs"] == ["tot_2023", "plain_2024"]
        assert len(payload["results"]) == 2

    def test_toc_many_missing_args_is_invalid_params(self, fig_db):
        state = _make_state(fig_db)
        resp = mcp_server._handle_tools_call(
            state,
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "toc_many", "arguments": {}}},
        )
        assert "error" in resp
        assert resp["error"]["code"] == -32602


# ===========================================================================
# Subprocess protocol smoke test
# ===========================================================================


def _build_msg(method: str, params: dict | None = None, msg_id: int | None = 1) -> str:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if msg_id is not None:
        msg["id"] = msg_id
    if params is not None:
        msg["params"] = params
    return json.dumps(msg) + "\n"


def _drive_server(input_lines: list[str], env_overrides: dict[str, str] | None = None,
                  timeout: float = 10.0) -> list[dict]:
    """Spawn lodestone-mcp, write input lines, return parsed responses."""
    env = os.environ.copy()
    # Default: clear any inherited LODESTONE_DB so each test sets its own.
    env.pop("LODESTONE_DB", None)
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-m", "_system.scripts.mcp_server"],
        input="".join(input_lines),
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return [json.loads(ln) for ln in out_lines]


@pytest.fixture
def real_db(tmp_path: Path) -> Path:
    """Standalone seeded DB on disk (the in-memory conn fixture won't survive
    the subprocess crossing). Seeds the same fig_db corpus."""
    from _system.db.connection import get_conn
    from _system.db.migrations import init_db

    db = tmp_path / "lodestone.db"
    c = get_conn(db)
    try:
        init_db(c)
        _seed_domain(c, "rag")
        p_id = _insert_paper(
            c, arxiv_id="2305.10601", paper_name="tot_2023",
            title="Tree of Thoughts", abstract="Deliberate search.",
            markdown=_MD_WITH_FIGS, domain="rag", collection="reasoning",
            needs_review=0, ingested_at="2023-05-17T00:00:00+00:00",
        )
        _insert_sections_for_md(
            c, paper_id=p_id, domain="rag", paper_name="tot_2023",
            markdown=_MD_WITH_FIGS,
        )
        _insert_figure(
            c, paper_id=p_id, figure_number=1, display_number="Figure 1",
            figure_id="F1", caption="Figure 1: Tree.",
            section_context="Method", image=_PNG_1x1,
        )
        _insert_figure(
            c, paper_id=p_id, figure_number=2, display_number="Figure 2",
            figure_id="F2", caption="Figure 2: Game.",
            section_context="Experiments", image=_PNG_1x1,
        )
    finally:
        c.close()
    return db


class TestProtocolHandshake:
    def test_initialize_and_tools_list(self, real_db):
        responses = _drive_server(
            [
                _build_msg("initialize", {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                }, msg_id=1),
                _build_msg("notifications/initialized", params={}, msg_id=None),
                _build_msg("tools/list", msg_id=2),
            ],
            env_overrides={"LODESTONE_DB": str(real_db)},
        )
        assert len(responses) == 2  # init reply + tools/list reply (no notif)
        init_reply = responses[0]
        assert init_reply["id"] == 1
        result = init_reply["result"]
        # Tracks PROTOCOL_VERSION in mcp_server.py — currently up-pinned
        # to 2025-06-18 to bisect which envelope-shape changes actually
        # gate Claude Code's image rendering.
        assert result["protocolVersion"] == mcp_server.PROTOCOL_VERSION
        assert result["serverInfo"]["name"] == "lodestone"
        assert "tools" in result["capabilities"]

        list_reply = responses[1]
        assert list_reply["id"] == 2
        tools = list_reply["result"]["tools"]
        assert len(tools) == 20
        for t in tools:
            assert t["name"].replace("_", "").isalnum()

    def test_unknown_method_returns_method_not_found(self, real_db):
        responses = _drive_server(
            [
                _build_msg("initialize", {"protocolVersion": "2024-11-05"}, msg_id=1),
                _build_msg("notifications/initialized", params={}, msg_id=None),
                _build_msg("totally/bogus", msg_id=99),
            ],
            env_overrides={"LODESTONE_DB": str(real_db)},
        )
        bogus_reply = next(r for r in responses if r.get("id") == 99)
        assert bogus_reply["error"]["code"] == mcp_server._ERR_METHOD_NOT_FOUND

    def test_tools_call_round_trip_with_structured_content(self, real_db):
        # `toc` returns no images, so structuredContent is preserved.
        responses = _drive_server(
            [
                _build_msg("initialize", {"protocolVersion": "2024-11-05"}, msg_id=1),
                _build_msg("notifications/initialized", params={}, msg_id=None),
                _build_msg("tools/call", {
                    "name": "toc",
                    "arguments": {"slug": "tot_2023"},
                }, msg_id=3),
            ],
            env_overrides={"LODESTONE_DB": str(real_db)},
        )
        call_reply = next(r for r in responses if r.get("id") == 3)
        result = call_reply["result"]
        assert result["isError"] is False
        text = next(b["text"] for b in result["content"] if b["type"] == "text")
        assert json.loads(text) == result["structuredContent"]
        assert result["structuredContent"]["mode"] == "toc"

    def test_image_response_uses_minimal_envelope(self, real_db):
        # `figure` returns an inline image; the envelope must mirror
        # test_mcp_image.py exactly (content only) so Claude Code surfaces
        # the image — see _finalize_result docstring.
        responses = _drive_server(
            [
                _build_msg("initialize", {"protocolVersion": "2024-11-05"}, msg_id=1),
                _build_msg("notifications/initialized", params={}, msg_id=None),
                _build_msg("tools/call", {
                    "name": "figure",
                    "arguments": {"paper": "tot_2023", "n": 1},
                }, msg_id=5),
            ],
            env_overrides={"LODESTONE_DB": str(real_db)},
        )
        call_reply = next(r for r in responses if r.get("id") == 5)
        result = call_reply["result"]
        assert any(b["type"] == "image" for b in result["content"])
        assert "structuredContent" not in result
        assert "isError" not in result

    def test_missing_db_at_env_path_is_auto_created(self, tmp_path):
        # Fresh-install path: LODESTONE_DB points at a not-yet-existing file
        # under a writable parent dir. The server must materialize an empty
        # schema-initialized DB so subsequent ingest_* / read tools work
        # without an out-of-band mkdir step. ``tables`` is the sentinel —
        # an empty schema is still a schema, so the call succeeds and lists
        # the canonical user tables.
        fresh = tmp_path / "fresh.db"
        responses = _drive_server(
            [
                _build_msg("initialize", {"protocolVersion": "2024-11-05"}, msg_id=1),
                _build_msg("notifications/initialized", params={}, msg_id=None),
                _build_msg("tools/call", {
                    "name": "tables",
                    "arguments": {},
                }, msg_id=4),
            ],
            env_overrides={"LODESTONE_DB": str(fresh)},
        )
        init = next(r for r in responses if r.get("id") == 1)
        assert "result" in init
        assert fresh.is_file(), "DB file should be created on first run"
        call = next(r for r in responses if r.get("id") == 4)
        assert call["result"]["isError"] is False
        # Empty schema still exposes the canonical tables.
        text = next(b["text"] for b in call["result"]["content"] if b["type"] == "text")
        payload = json.loads(text)
        table_names = {t["name"] for t in payload.get("tables", [])}
        assert "papers" in table_names
        assert "domains" in table_names

    def test_uncreatable_db_path_completes_initialize_then_isError(self, tmp_path):
        # Parent path is a regular file, not a dir, so the DB cannot be
        # created. Initialize must still succeed (issue #35287); the error
        # surfaces on the first tools/call.
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("")
        bogus = blocker / "no_such.db"
        responses = _drive_server(
            [
                _build_msg("initialize", {"protocolVersion": "2024-11-05"}, msg_id=1),
                _build_msg("notifications/initialized", params={}, msg_id=None),
                _build_msg("tools/call", {
                    "name": "toc",
                    "arguments": {"slug": "x"},
                }, msg_id=4),
            ],
            env_overrides={"LODESTONE_DB": str(bogus)},
        )
        init = next(r for r in responses if r.get("id") == 1)
        assert "result" in init
        call = next(r for r in responses if r.get("id") == 4)
        assert call["result"]["isError"] is True

    def test_malformed_json_returns_parse_error(self, real_db):
        responses = _drive_server(
            [
                "this is not json\n",
                _build_msg("initialize", {"protocolVersion": "2024-11-05"}, msg_id=1),
            ],
            env_overrides={"LODESTONE_DB": str(real_db)},
        )
        # First message: parse error with id=null. Second: init reply.
        parse_err = next(
            r for r in responses
            if r.get("error", {}).get("code") == mcp_server._ERR_PARSE
        )
        assert parse_err["id"] is None


# ===========================================================================
# HTTP transport smoke test
# ===========================================================================


def _free_port() -> int:
    import socket as _socket
    s = _socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _http_post(url: str, body: dict, timeout: float = 5.0) -> tuple[int, dict | None]:
    """POST one JSON-RPC message; return (status_code, parsed-body-or-None)."""
    import urllib.error
    import urllib.request
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            code = resp.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        code = exc.code
    if not raw:
        return code, None
    try:
        return code, json.loads(raw)
    except json.JSONDecodeError:
        return code, None


@pytest.fixture
def http_server(real_db):
    """Spawn ``lodestone-mcp --http`` against ``real_db`` on a free port and
    yield the base URL. Tear down on exit. Polls the port for ~3s before
    handing off so the first request doesn't race the bind."""
    import socket as _socket
    import time

    port = _free_port()
    # stderr=DEVNULL: the server logs every request through log_message →
    # _log to stderr. A PIPE buffer (~64KB) would fill and block the
    # server's writes mid-test. We don't read the pipe during the run, so
    # discard it; on early-exit we re-spawn briefly to capture the error.
    proc = subprocess.Popen(
        [sys.executable, "-m", "_system.scripts.mcp_server",
         "--http", "--host", "127.0.0.1", "--port", str(port)],
        env={**os.environ, "LODESTONE_DB": str(real_db)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 3.0
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        s = _socket.socket()
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", port))
            ready = True
            break
        except OSError:
            time.sleep(0.05)
        finally:
            s.close()
    if not ready:
        proc.terminate()
        raise RuntimeError(f"server did not start listening on port {port}")
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)


class TestHttpTransport:
    def test_initialize_handshake(self, http_server):
        code, body = _http_post(http_server, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "1"}},
        })
        assert code == 200
        assert body["id"] == 1
        result = body["result"]
        assert result["protocolVersion"] == mcp_server.PROTOCOL_VERSION
        assert result["serverInfo"]["name"] == "lodestone"

    def test_tools_list_returns_full_registry(self, http_server):
        code, body = _http_post(http_server, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/list",
        })
        assert code == 200
        tools = body["result"]["tools"]
        assert len(tools) == 20
        names = {t["name"] for t in tools}
        assert names == set(mcp_server._TOOL_INDEX.keys())

    def test_tools_call_search_returns_text_block(self, http_server):
        code, body = _http_post(http_server, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "tree"}},
        })
        assert code == 200
        result = body["result"]
        assert result.get("isError") in (False, None)
        assert any(b["type"] == "text" for b in result["content"])

    def test_notification_returns_202(self, http_server):
        # Notifications (no `id`) get 202 No Content.
        import urllib.request
        data = json.dumps({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        }).encode("utf-8")
        req = urllib.request.Request(
            http_server, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            assert resp.getcode() == 202

    def test_wrong_path_404(self, http_server):
        url = http_server.rsplit("/", 1)[0] + "/nope"
        code, _ = _http_post(url, {
            "jsonrpc": "2.0", "id": 9, "method": "tools/list",
        })
        assert code == 404

    def test_get_returns_405(self, http_server):
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(http_server, timeout=5.0) as resp:
                code = resp.getcode()
        except urllib.error.HTTPError as exc:
            code = exc.code
        assert code == 405


def _http_post_sse(url: str, body: dict, timeout: float = 10.0) -> tuple[int, str, list[dict]]:
    """POST one JSON-RPC message with ``Accept: text/event-stream``.

    Returns (status_code, content_type, parsed-event-payloads).
    Each SSE frame is decoded back to a dict.
    """
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("Content-Type", ""), []

    events: list[dict] = []
    for chunk in raw.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        for line in chunk.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return code, ctype, events


class TestHttpSse:
    """``Accept: text/event-stream`` upgrades the response to SSE; in-flight
    progress notifications and the final reply both arrive as ``data:``
    frames on the same stream."""

    def test_sse_upgrade_returns_event_stream_content_type(self, http_server):
        # tools/list is a quick call; no progress notifications, but the
        # final reply must arrive as an SSE frame, not application/json.
        code, ctype, events = _http_post_sse(http_server, {
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        })
        assert code == 200
        assert "text/event-stream" in ctype
        assert len(events) == 1
        assert events[0]["id"] == 1
        assert "tools" in events[0]["result"]

    def test_sse_search_call_returns_one_response_frame(self, http_server):
        code, ctype, events = _http_post_sse(http_server, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "search", "arguments": {"query": "tree"}},
        })
        assert code == 200
        assert "text/event-stream" in ctype
        # search emits no progress notifications, so we see exactly the response.
        assert len(events) == 1
        assert events[0]["id"] == 2
        assert events[0]["result"].get("isError") in (False, None)

    def test_plain_json_still_works_when_accept_unset(self, http_server):
        # Existing plain-JSON behavior preserved when client doesn't ask
        # for SSE — the test_tools_call_search_returns_text_block case
        # already covers this; this assertion is the explicit sibling.
        code, body = _http_post(http_server, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/list",
        })
        assert code == 200
        assert body["id"] == 3
        assert "tools" in body["result"]

    def test_sse_dispatch_error_returns_error_frame(self, http_server):
        # Unknown method surfaces as a JSON-RPC error inside the SSE frame
        # rather than as a 5xx — the SSE channel must always wrap the
        # protocol-level outcome in a frame.
        code, ctype, events = _http_post_sse(http_server, {
            "jsonrpc": "2.0", "id": 4, "method": "no/such/method",
        })
        assert code == 200
        assert "text/event-stream" in ctype
        assert len(events) == 1
        assert events[0]["id"] == 4
        assert "error" in events[0]


# ===========================================================================
# overview / collection MCP tools
# ===========================================================================


def _seed_overview_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Tiny corpus used by the overview/collection tool tests: one domain
    with one populated collection plus one paper. Independent of fig_db /
    seeded_db so these tests stay focused.
    """
    _seed_domain(conn, "rag")
    p_id = _insert_paper(
        conn,
        arxiv_id="2401.99001",
        paper_name="ov_paper",
        title="Overview Test Paper",
        abstract="An abstract.",
        markdown=None,
        domain="rag",
        collection="hier_indexing",
        needs_review=0,
        ingested_at="2024-01-01T00:00:00+00:00",
    )
    del p_id
    conn.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES (?, ?, ?)",
        ("rag", "hier_indexing", "multi-level toc"),
    )
    return conn


@pytest.fixture
def overview_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    return _seed_overview_db(conn)


def test_overview_tool_returns_tree_text(overview_db):
    state = _make_state(overview_db)
    resp = _call(state, "overview", {})
    assert not _is_error(resp)
    blocks = _content(resp)
    text = blocks[0]["text"]
    # Tree connectors prove this is rendered as a tree, not raw JSON.
    assert "├──" in text or "└──" in text
    assert "rag" in text
    assert "hier_indexing" in text
    structured = resp["result"].get("structuredContent")
    assert structured is not None
    assert isinstance(structured.get("domains"), list)


def test_overview_tool_domain_filter(overview_db):
    # Add a second domain (with a populated collection so the schema
    # invariant is satisfied) that should NOT appear when filter narrows.
    _seed_domain(overview_db, "agents")
    overview_db.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES ('agents', 'tool_use', NULL)"
    )
    overview_db.execute(
        "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, "
        "abstract, pdf_url, html_source, ingested_at, status, domain, "
        "collection, needs_review) VALUES "
        "(?, 'a_paper', 't', '[]', '2024-01-01', 'a', "
        "'https://x', 'arxiv', '2024-01-01T00:00:00+00:00', 'classified', "
        "'agents', 'tool_use', 0)",
        ("2401.99002",),
    )

    state = _make_state(overview_db)
    resp = _call(state, "overview", {"domain": "rag"})
    assert not _is_error(resp)
    structured = resp["result"]["structuredContent"]
    assert structured["domain"] == "rag"
    names = [d["name"] for d in structured["domains"]]
    assert names == ["rag"]
    text = _content(resp)[0]["text"]
    assert "agents" not in text


def test_collection_tool_string_arg(overview_db):
    state = _make_state(overview_db)
    resp = _call(state, "collection", {"collection": "hier_indexing"})
    assert not _is_error(resp)
    structured = resp["result"]["structuredContent"]
    assert len(structured["collections"]) == 1
    assert structured["collections"][0]["collection"] == "hier_indexing"


def test_collection_tool_array_arg(overview_db):
    overview_db.execute(
        "INSERT OR IGNORE INTO collection_definitions (domain, name, description) "
        "VALUES ('rag', 'hybrid', NULL)"
    )
    overview_db.execute(
        "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, "
        "abstract, pdf_url, html_source, ingested_at, status, domain, "
        "collection, needs_review) VALUES "
        "(?, 'hybrid_paper', 't', '[]', '2024-02-01', 'a', "
        "'https://x', 'arxiv', '2024-02-01T00:00:00+00:00', 'classified', "
        "'rag', 'hybrid', 0)",
        ("2402.99003",),
    )

    state = _make_state(overview_db)
    resp = _call(state, "collection", {
        "collection": ["hier_indexing", "hybrid"],
    })
    assert not _is_error(resp)
    structured = resp["result"]["structuredContent"]
    names = {c["collection"] for c in structured["collections"]}
    assert names == {"hier_indexing", "hybrid"}


def test_collection_tool_text_renders_tree(overview_db):
    state = _make_state(overview_db)
    resp = _call(state, "collection", {"collection": "hier_indexing"})
    text = _content(resp)[0]["text"]
    assert "rag / hier_indexing" in text
    assert "ov_paper" in text


def test_collection_tool_no_image_blocks(overview_db):
    state = _make_state(overview_db)
    resp = _call(state, "collection", {"collection": "hier_indexing"})
    assert not _has_image(resp)
    blocks = _content(resp)
    assert all(b.get("type") == "text" for b in blocks)


def test_collection_tool_missing_required_collection(overview_db):
    state = _make_state(overview_db)
    resp = mcp_server._handle_tools_call(
        state,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "collection", "arguments": {}},
        },
    )
    # Missing required arg surfaces as JSON-RPC error (InvalidParams) per
    # the dispatcher's KeyError translation.
    assert "error" in resp
    assert resp["error"]["code"] == mcp_server._ERR_INVALID_PARAMS


# ===========================================================================
# Pagination — bm25 + lookup tools
# ===========================================================================


def _seed_paged_bm25_corpus(conn: sqlite3.Connection) -> None:
    """Seed 12 sections in fig_db's tot_2023 paper that all match the
    token "pageme" — enough to exercise multi-page paging."""
    p_id = conn.execute(
        "SELECT id FROM papers WHERE paper_name = ?", ("tot_2023",)
    ).fetchone()[0]
    rows = [
        (p_id, "rag", "tot_2023", f"Pageable Section {i:02d}", "1",
         f"pageme content {i} pageme")
        for i in range(12)
    ]
    conn.executemany(
        """
        INSERT INTO sections
            (paper_id, domain, paper_name, section_title, section_level, body)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


class TestBm25Pagination:
    def test_default_offset_is_zero_and_payload_echoes_pagination(
        self, fig_db
    ):
        _seed_paged_bm25_corpus(fig_db)
        state = _make_state(fig_db)
        resp = _call(state, "bm25", {"query": "pageme", "limit": 5})
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["offset"] == 0
        assert sc["limit"] == 5
        assert sc["total_hits"] == 12
        assert sc["has_more"] is True

    def test_offset_walks_forward(self, fig_db):
        _seed_paged_bm25_corpus(fig_db)
        state = _make_state(fig_db)
        resp = _call(state, "bm25", {
            "query": "pageme", "limit": 5, "offset": 5,
        })
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["offset"] == 5
        assert sc["total_hits"] == 12
        assert sc["has_more"] is True

    def test_negative_offset_soft_fails(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "bm25", {"query": "deliberate", "offset": -1})
        # Soft-failure: payload carries the diagnostic, isError stays false.
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["status"] == "invalid_pagination"
        assert sc["offset"] == -1


class TestLookupPagination:
    def _seed_n_canonicals(
        self, conn: sqlite3.Connection, n: int, prefix: str
    ) -> None:
        for i in range(n):
            cur = conn.execute(
                """
                INSERT INTO canonical_terms
                    (domain, term_type, entity_type, canonical_name, first_seen_in)
                VALUES ('rag', 'entity', 'method', ?, 'tot_2023')
                """,
                (f"{prefix} {i:02d}",),
            )
            tid = cur.lastrowid
            conn.execute(
                """
                INSERT INTO terms_fts
                    (term_id, domain, term_type, entity_type, canonical_name, aliases)
                VALUES (?, 'rag', 'entity', 'method', ?, '')
                """,
                (tid, f"{prefix} {i:02d}"),
            )

    def test_default_offset_zero_in_payload(self, fig_db):
        self._seed_n_canonicals(fig_db, 3, "LookupPager")
        state = _make_state(fig_db)
        resp = _call(state, "lookup", {"query": "LookupPager"})
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["offset"] == 0
        assert sc["total_hits"] == 3
        assert sc["has_more"] is False

    def test_offset_walks_forward(self, fig_db):
        self._seed_n_canonicals(fig_db, 8, "LookupPager")
        state = _make_state(fig_db)
        resp = _call(state, "lookup", {
            "query": "LookupPager", "limit": 3, "offset": 3,
        })
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["offset"] == 3
        assert sc["limit"] == 3
        assert sc["total_hits"] == 8
        assert sc["has_more"] is True

    def test_negative_offset_soft_fails(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "lookup", {"query": "RAPTOR", "offset": -1})
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["status"] == "invalid_pagination"


# ===========================================================================
# tables / schema / query — DB introspection + read-only SQL escape hatch
# ===========================================================================


class TestTablesTool:
    def test_tables_lists_user_tables(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "tables", {})
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["mode"] == "tables"
        assert sc["status"] == "ok"
        assert sc["include_internal"] is False
        names = {t["name"] for t in sc["tables"]}
        assert "papers" in names
        assert "sections" in names  # virtual
        # Shadow tables filtered out by default.
        assert not any(n.endswith("_data") for n in names)

    def test_tables_include_internal_passes_through(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "tables", {"include_internal": True})
        sc = resp["result"]["structuredContent"]
        assert sc["include_internal"] is True
        names = {t["name"] for t in sc["tables"]}
        assert any(n.endswith("_data") or n.endswith("_idx") for n in names)


class TestSchemaTool:
    def test_schema_string_arg_auto_wraps(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "schema", {"tables": "papers"})
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["mode"] == "schema"
        assert len(sc["tables"]) == 1
        assert sc["tables"][0]["name"] == "papers"
        assert sc["missing"] == []

    def test_schema_array_arg(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(
            state, "schema",
            {"tables": ["papers", "no_such_table"]},
        )
        sc = resp["result"]["structuredContent"]
        names = [t["name"] for t in sc["tables"]]
        assert names == ["papers"]
        assert sc["missing"] == ["no_such_table"]

    def test_schema_missing_arg(self, fig_db):
        state = _make_state(fig_db)
        resp = mcp_server._handle_tools_call(
            state,
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "schema", "arguments": {}}},
        )
        assert "error" in resp
        assert resp["error"]["code"] == -32602


class TestQueryTool:
    def test_query_select_happy_path(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(
            state, "query",
            {"sql": "SELECT paper_name FROM papers ORDER BY paper_name"},
        )
        assert not _is_error(resp)
        assert not _has_image(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["mode"] == "query"
        assert sc["status"] == "ok"
        assert sc["columns"] == ["paper_name"]
        assert sc["truncated"] is False
        names = [r["paper_name"] for r in sc["rows"]]
        assert "tot_2023" in names

    def test_query_drop_is_read_only_violation(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "query", {"sql": "DROP TABLE papers"})
        # Soft-failures stay isError=false.
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["status"] == "read_only_violation"
        # Original DB still has the table.
        assert fig_db.execute(
            "SELECT count(*) FROM papers"
        ).fetchone()[0] >= 1

    def test_query_multiple_statements_soft_fail(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "query", {"sql": "SELECT 1; SELECT 2"})
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["status"] == "multiple_statements"

    def test_query_syntax_error_soft_fail(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "query", {"sql": "SELEKT 1"})
        assert not _is_error(resp)
        sc = resp["result"]["structuredContent"]
        assert sc["status"] == "query_failed"

    def test_query_missing_sql_arg(self, fig_db):
        state = _make_state(fig_db)
        resp = mcp_server._handle_tools_call(
            state,
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "query", "arguments": {}}},
        )
        assert "error" in resp
        assert resp["error"]["code"] == -32602


# ===========================================================================
# ingest_paper / ingest_repo — progress notification wiring
# ===========================================================================


class TestIngestProgress:
    """The ingest tools accept a ``progressToken`` via the standard MCP
    ``_meta.progressToken`` channel and emit ``notifications/progress``
    messages between stages. These tests stub the underlying ingest
    orchestrators so they exercise the dispatch wiring + notification
    plumbing without touching HF caches or arxiv."""

    def _stub_ingest(self, monkeypatch, *, kind: str, ticks: int):
        """Replace ``ingest`` (kind='paper') or ``ingest_repo_only`` (kind='repo')
        with a stub that fires the progress callback ``ticks`` times then
        returns a fake summary. Returns the captured ``calls`` list so the
        caller can assert kwargs flowed through."""
        from _system.scripts import ingest as ingest_mod

        calls: list[dict] = []

        def _fake(**kwargs):
            calls.append(kwargs)
            cb = kwargs.get("progress")
            for i in range(ticks):
                if cb is not None:
                    cb(f"step-{i}", i, ticks)
            if cb is not None:
                cb("complete", ticks, ticks)
            return {"kind": kind, "status": "INDEXED" if kind == "paper" else "CLASSIFIED"}

        target = "ingest" if kind == "paper" else "ingest_repo_only"
        monkeypatch.setattr(ingest_mod, target, _fake)

        # Stub out check_models so we don't touch the HF cache. Patch on
        # the module the dispatcher imports from.
        from _system.scripts import validate_models
        monkeypatch.setattr(validate_models, "check_models", lambda: None)
        return calls

    def test_ingest_paper_emits_progress(self, fig_db, monkeypatch):
        sent: list[dict] = []
        monkeypatch.setattr(mcp_server, "_send", lambda msg: sent.append(msg))
        calls = self._stub_ingest(monkeypatch, kind="paper", ticks=3)

        state = _make_state(fig_db)
        resp = mcp_server._handle_tools_call(
            state,
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "ingest_paper",
                    "arguments": {"url": "2301.12345"},
                    "_meta": {"progressToken": "tok-1"},
                },
            },
        )

        assert not _is_error(resp), resp
        sc = resp["result"]["structuredContent"]
        assert sc == {"kind": "paper", "status": "INDEXED"}

        # Stub fired 3 staged ticks + a final 'complete' tick. The
        # dispatcher itself emits one upfront 'checking models' tick. So
        # 5 progress notifications total.
        progress_msgs = [
            m for m in sent if m.get("method") == "notifications/progress"
        ]
        assert len(progress_msgs) == 5
        for m in progress_msgs:
            assert m["params"]["progressToken"] == "tok-1"
            assert isinstance(m["params"]["progress"], int)
            assert isinstance(m["params"]["total"], int)
            assert isinstance(m["params"]["message"], str)

        # arxiv parsing happened: the underlying ingest got the bare id.
        assert len(calls) == 1
        assert calls[0]["arxiv_id"] == "2301.12345"
        assert calls[0]["force"] is False

    def test_ingest_paper_no_token_no_notifications(self, fig_db, monkeypatch):
        sent: list[dict] = []
        monkeypatch.setattr(mcp_server, "_send", lambda msg: sent.append(msg))
        self._stub_ingest(monkeypatch, kind="paper", ticks=2)

        state = _make_state(fig_db)
        resp = mcp_server._handle_tools_call(
            state,
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "ingest_paper",
                    "arguments": {"url": "2301.12345"},
                    # No _meta -> no progressToken -> notifications suppressed.
                },
            },
        )
        assert not _is_error(resp)
        progress_msgs = [
            m for m in sent if m.get("method") == "notifications/progress"
        ]
        assert progress_msgs == []

    def test_ingest_repo_dispatch_threads_kwargs(self, fig_db, monkeypatch):
        sent: list[dict] = []
        monkeypatch.setattr(mcp_server, "_send", lambda msg: sent.append(msg))
        calls = self._stub_ingest(monkeypatch, kind="repo", ticks=2)

        state = _make_state(fig_db)
        resp = mcp_server._handle_tools_call(
            state,
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {
                    "name": "ingest_repo",
                    "arguments": {
                        "url": "https://github.com/foo/bar",
                        "force": True,
                        "domain": "agents",
                    },
                    "_meta": {"progressToken": "tok-9"},
                },
            },
        )
        assert not _is_error(resp), resp
        sc = resp["result"]["structuredContent"]
        assert sc == {"kind": "repo", "status": "CLASSIFIED"}

        assert len(calls) == 1
        assert calls[0]["repo_url"] == "https://github.com/foo/bar"
        assert calls[0]["force"] is True
        assert calls[0]["domain"] == "agents"
        # The dispatcher's progress callback must be wired in.
        assert callable(calls[0]["progress"])

        progress_msgs = [
            m for m in sent if m.get("method") == "notifications/progress"
        ]
        # 1 'checking models' + 2 step ticks + 1 'complete' = 4.
        assert len(progress_msgs) == 4
        assert all(
            m["params"]["progressToken"] == "tok-9" for m in progress_msgs
        )

