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
    _insert_sections_for_md,
    _seed_domain,
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
        collection=None,
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
        payload = {"mode": "toc", "paper_name": "tot_2023", "toc": []}
        out = mcp_server._pack_result(payload, fig_db)
        text = _text_blocks(out["content"])[0]["text"]
        assert json.loads(text) == payload
        assert out["isError"] is False

    def test_structured_content_matches_text_block_when_no_image(self, fig_db):
        payload = {"mode": "toc", "paper_name": "tot_2023", "toc": [{"level": 1, "title": "X"}]}
        out = mcp_server._pack_result(payload, fig_db)
        text = _text_blocks(out["content"])[0]["text"]
        assert out["structuredContent"] == json.loads(text)

    def test_image_response_uses_minimal_envelope(self, fig_db):
        # Image-bearing responses mirror the test_mcp_image.py shape exactly:
        # only `content`, no `structuredContent`, no `isError` — that's the
        # envelope under which Claude Code is observed to surface image
        # blocks as multimodal input.
        payload = {
            "mode": "read", "status": "ok", "paper_name": "tot_2023",
            "section": None, "text": _MD_WITH_FIGS,
        }
        out = mcp_server._pack_result(payload, fig_db)
        assert _image_blocks(out["content"]), "expected image blocks for this fixture"
        assert "structuredContent" not in out
        assert "isError" not in out

    def test_no_image_blocks_when_no_refs(self, fig_db):
        payload = {
            "mode": "read", "status": "ok", "paper_name": "plain_2024",
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
        resp = _call(state, "read", {"paper_name": "tot_2023"})
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
        resp = _call(state, "read", {"paper_name": "plain_2024"})
        assert not _is_error(resp), resp
        assert _image_blocks(_content(resp)) == []

    def test_read_section_slice_only_attaches_refs_in_slice(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "read", {"paper_name": "tot_2023", "section": "Method"})
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
        resp = _call(state, "read", {"paper_name": "tot_2023"})
        assert not _is_error(resp)
        assert _image_blocks(_content(resp)) == []

    def test_oversize_blob_is_skipped_with_marker(self, fig_db, monkeypatch):
        # Force the per-blob limit to 1 byte so our 1×1 PNG is "oversize".
        monkeypatch.setenv("LODESTONE_MAX_FIGURE_BYTES", "1")
        state = _make_state(fig_db)
        resp = _call(state, "read", {"paper_name": "tot_2023"})
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
        resp = _call(state, "read", {"paper_name": "tot_2023"})
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
        resp = _call(state, "read", {"paper_name": "tot_2023", "section": "NoSuchSection"})
        assert not _is_error(resp)
        text = _text_blocks(_content(resp))[0]["text"]
        payload = json.loads(text)
        assert payload["status"] == "section_not_found"
        # No image blocks on a soft-fail, so structuredContent is included.
        assert resp["result"]["structuredContent"]["status"] == "section_not_found"

    def test_unknown_paper_raises_isError(self, fig_db):
        state = _make_state(fig_db)
        resp = _call(state, "read", {"paper_name": "no_such_paper"})
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
# Tools list — schema integrity
# ===========================================================================


class TestToolsList:
    def test_tools_list_returns_nine_tools(self):
        out = mcp_server._handle_tools_list({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        tools = out["result"]["tools"]
        assert len(tools) == 9
        names = {t["name"] for t in tools}
        assert names == {
            "search", "bm25", "lookup", "browse", "toc", "read",
            "figure", "repo_tree", "read_code",
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
        assert len(tools) == 9
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
                    "arguments": {"paper_name": "tot_2023"},
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

    def test_bogus_db_completes_initialize_then_isError_on_tools_call(self, tmp_path):
        # No DB at this path — initialize must still succeed (issue #35287).
        bogus = tmp_path / "no_such.db"
        responses = _drive_server(
            [
                _build_msg("initialize", {"protocolVersion": "2024-11-05"}, msg_id=1),
                _build_msg("notifications/initialized", params={}, msg_id=None),
                _build_msg("tools/call", {
                    "name": "toc",
                    "arguments": {"paper_name": "x"},
                }, msg_id=4),
            ],
            env_overrides={"LODESTONE_DB": str(bogus)},
        )
        init = next(r for r in responses if r.get("id") == 1)
        assert "result" in init  # not error — handshake succeeded
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
