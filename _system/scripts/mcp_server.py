"""Lodestone stdio MCP server.

Wraps the read-side ``mode_*`` functions in :mod:`_system.scripts.search`
as MCP tools served over stdio JSON-RPC (spec rev 2025-06-18). Returns the
existing JSON envelope as a ``text`` content block plus inline ``image``
content blocks for any ``(figure:N)`` markdown refs the response carries.

Tools registered (all surface as ``mcp__lodestone__<name>`` in Claude Code):

    bm25, lookup, browse, toc, read, figure, repo_tree, read_code

Transport notes (per spec §Transports):

* Newline-delimited UTF-8 JSON on stdin/stdout. No Content-Length headers.
* Stdout carries only valid MCP messages — never ``print()``, banners, or
  progress bars. Diagnostics go to stderr.

The server completes ``initialize`` even on bad config (missing DB, schema
mismatch) — that's the issue #35287 mitigation. Failures surface as
``isError: true`` from subsequent ``tools/call`` instead.
"""
from __future__ import annotations

import base64
import json
import os
import re
import signal
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any

from _system.db.connection import get_conn
from _system.scripts.search import (
    Scope,
    _SOFT_FAILURE_STATUSES,
    mode_bm25,
    mode_browse,
    mode_figure,
    mode_read,
    mode_read_code,
    mode_repo_tree,
    mode_taxonomy_lookup,
    mode_toc,
)
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.mcp_server")

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

# Empirically verified 2026-05-01 against Claude Code v2.1.126: image
# content blocks reach the model under protocolVersion=2025-06-18 as long
# as the result envelope is minimal — see _finalize_result for the gate.
PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "lodestone"
SERVER_VERSION = "0.1.0"

# Supported protocol versions we'll echo back if the client requests them.
# Anything else falls back to PROTOCOL_VERSION.
_SUPPORTED_PROTOCOLS = {"2024-11-05", "2025-03-26", "2025-06-18"}

# JSON-RPC error codes (spec §JSON-RPC).
_ERR_PARSE = -32700
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INVALID_PARAMS = -32602
_ERR_INTERNAL = -32603

# ---------------------------------------------------------------------------
# Figure attach config (env-overridable)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_FIGURE_BYTES = 1 * 1024 * 1024  # 1 MB raw blob
_DEFAULT_MAX_FIGURES_PER_RESPONSE = 8

# Matches ``(figure:N)`` where N is one or more digits. Tolerant of
# whitespace-trimmed surroundings; the markdown image syntax in the
# ingest pipeline writes exactly this form.
_FIGURE_REF_RE = re.compile(r"\(figure:(\d+)\)")


def _max_figure_bytes() -> int:
    raw = os.environ.get("LODESTONE_MAX_FIGURE_BYTES")
    if not raw:
        return _DEFAULT_MAX_FIGURE_BYTES
    try:
        v = int(raw)
        if v > 0:
            return v
    except ValueError:
        pass
    return _DEFAULT_MAX_FIGURE_BYTES


def _max_figures_per_response() -> int:
    raw = os.environ.get("LODESTONE_MAX_FIGURES_PER_RESPONSE")
    if not raw:
        return _DEFAULT_MAX_FIGURES_PER_RESPONSE
    try:
        v = int(raw)
        if v > 0:
            return v
    except ValueError:
        pass
    return _DEFAULT_MAX_FIGURES_PER_RESPONSE


# ---------------------------------------------------------------------------
# Stdio framing
# ---------------------------------------------------------------------------


def _send(msg: dict) -> None:
    """Write one newline-delimited JSON message to stdout and flush."""
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _log(level: str, s: str) -> None:
    sys.stderr.write(f"[lodestone-mcp][{level}] {s}\n")
    sys.stderr.flush()


def _err(msg_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


# ---------------------------------------------------------------------------
# DB resolution
# ---------------------------------------------------------------------------


def _resolve_db_path() -> Path | None:
    """Locate ``lodestone.db``: env var first, then walk up from CWD looking
    for a sibling ``pyproject.toml`` whose project name is ``lodestone``.
    Returns ``None`` if neither path resolves — caller surfaces this via
    ``isError`` after ``initialize``.
    """
    env = os.environ.get("LODESTONE_DB")
    if env:
        p = Path(env).expanduser()
        return p

    here = Path.cwd().resolve()
    for cand in (here, *here.parents):
        py = cand / "pyproject.toml"
        if not py.is_file():
            continue
        try:
            text = py.read_text(encoding="utf-8")
        except OSError:
            continue
        # Cheap text check — avoids a tomllib import dance for the
        # marker we already control. False-matches a substring of
        # ``name = "lodestone"`` only inside this file shape.
        if 'name = "lodestone"' in text:
            db = cand / "lodestone.db"
            if db.is_file():
                return db
    return None


# ---------------------------------------------------------------------------
# Figure attach helpers
# ---------------------------------------------------------------------------


def _walk_strings(obj: Any):
    """Yield every string value reachable inside a JSON-like dict/list tree."""
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
        return
    if isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)


def _extract_figure_refs(payload: dict) -> list[int]:
    """Return a deduped, ordered list of figure_number ints referenced anywhere
    in the payload via ``(figure:N)`` markdown refs.
    """
    seen: dict[int, None] = {}  # ordered set
    for s in _walk_strings(payload):
        for m in _FIGURE_REF_RE.finditer(s):
            try:
                n = int(m.group(1))
            except ValueError:
                continue
            seen.setdefault(n, None)
    return list(seen.keys())


def _resolve_paper_id_for_payload(
    conn: sqlite3.Connection, payload: dict
) -> int | None:
    """Best-effort: find the paper_id this payload is talking about.

    Top-level ``paper_name`` (read/toc/figure/repo_tree/read_code) wins.
    For BM25 results, fall back to the first hit's paper_name. Returns
    ``None`` when no paper context is determinable.
    """
    name = payload.get("paper_name")
    if not name:
        results = payload.get("results") or []
        if results and isinstance(results[0], dict):
            name = results[0].get("paper_name")
    if not name:
        return None
    row = conn.execute(
        "SELECT id FROM papers WHERE paper_name = ?", (name,)
    ).fetchone()
    if row is None:
        return None
    return int(row[0])


def _attach_figure_blocks(
    conn: sqlite3.Connection,
    paper_id: int,
    figure_numbers: list[int],
    content_blocks: list[dict],
) -> None:
    """Append labeled image blocks for each resolved (paper_id, figure_number).

    Per-blob > LODESTONE_MAX_FIGURE_BYTES is skipped with a text marker.
    Beyond LODESTONE_MAX_FIGURES_PER_RESPONSE images, remaining refs are
    listed in a single trailing text marker.
    """
    if not figure_numbers:
        return

    cap = _max_figures_per_response()
    max_bytes = _max_figure_bytes()
    attached = 0
    overflow: list[int] = []

    for n in figure_numbers:
        if attached >= cap:
            overflow.append(n)
            continue
        try:
            row = conn.execute(
                "SELECT image, mime_type, caption FROM figures "
                " WHERE paper_id = ? AND figure_number = ?",
                (paper_id, n),
            ).fetchone()
        except sqlite3.Error as exc:
            _log("debug", f"figure lookup failed paper_id={paper_id} n={n}: {exc!r}")
            continue
        if row is None:
            _log("debug", f"figure not found paper_id={paper_id} n={n}")
            continue
        image, mime_type, caption = row
        if image is None:
            continue
        size = len(image)
        if size > max_bytes:
            content_blocks.append({
                "type": "text",
                "text": (
                    f"--- paper_id={paper_id} figure={n} skipped: "
                    f"blob {size / 1024 / 1024:.2f} MB exceeds "
                    f"{max_bytes / 1024 / 1024:.2f} MB limit; use the "
                    f"'figure' tool to fetch directly ---"
                ),
            })
            continue
        try:
            b64 = base64.b64encode(image).decode("ascii")
        except (TypeError, ValueError) as exc:
            _log("debug", f"b64 encode failed paper_id={paper_id} n={n}: {exc!r}")
            continue
        cap_text = (caption or "").replace("\n", " ").strip()
        content_blocks.append({
            "type": "text",
            "text": (
                f"--- paper_id={paper_id} figure={n} "
                f"caption='{cap_text}' ---"
            ),
        })
        content_blocks.append({
            "type": "image",
            "data": b64,
            "mimeType": mime_type or "image/png",
        })
        attached += 1

    if overflow:
        content_blocks.append({
            "type": "text",
            "text": (
                f"--- paper_id={paper_id} omitted_figures="
                f"{','.join(str(n) for n in overflow)}; per-response cap "
                f"of {cap} reached. Use the 'figure' tool to fetch any "
                f"of these directly. ---"
            ),
        })


def _finalize_result(blocks: list[dict], payload: dict) -> dict:
    """Build the final tools/call result envelope.

    Empirical Claude Code behavior (v2.1.126, 2026-05-01): when an image
    content block is returned alongside ``structuredContent`` and/or
    ``isError``, the client silently strips the image — JSON text reaches
    the model but pixels don't, even though the server emitted them
    correctly on the wire. The previously-working ``test_mcp_image.py``
    PoC used a minimal envelope (``{"content": [text, image]}`` only) and
    images surfaced fine. Bisect verified that NEITHER
    ``protocolVersion: 2025-06-18`` NOR
    ``capabilities.tools.listChanged: false`` is the gate — images
    surfaced after re-adding each. The individual contributions of
    ``isError`` vs ``structuredContent`` were not bisected separately
    because both are pure spec-2025-06-18 additions removable with zero
    semantic loss when an image is present (the JSON text block already
    carries everything ``structuredContent`` would). Text-only responses
    keep the spec niceties for clients that want structured access.
    """
    has_image = any(b.get("type") == "image" for b in blocks)
    if has_image:
        return {"content": blocks}
    return {
        "content": blocks,
        "structuredContent": payload,
        "isError": False,
    }


def _pack_result(payload: dict, conn: sqlite3.Connection | None) -> dict:
    """Build the ``tools/call`` result envelope from a mode_* payload.

    Always emits a JSON text block + ``structuredContent``. Appends labeled
    image blocks for any ``(figure:N)`` refs the payload carries when a
    sqlite connection is available. Soft-failure statuses pass through as
    ``isError: false``.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    blocks: list[dict] = [{"type": "text", "text": text}]

    if conn is not None:
        try:
            refs = _extract_figure_refs(payload)
            if refs:
                paper_id = _resolve_paper_id_for_payload(conn, payload)
                if paper_id is not None:
                    _attach_figure_blocks(conn, paper_id, refs, blocks)
        except Exception as exc:  # noqa: BLE001 — fail-silent contract
            _log("warning", f"figure attach helper failed: {exc!r}")

    return _finalize_result(blocks, payload)


def _figure_only_result(payload: dict, conn: sqlite3.Connection) -> dict:
    """Variant for the ``figure`` tool: payload from mode_figure points at a
    tempfile path, not a BLOB. We re-fetch the BLOB straight from the DB so
    the MCP channel doesn't go through the filesystem.
    """
    blocks: list[dict] = [{
        "type": "text",
        "text": json.dumps(payload, ensure_ascii=False, indent=2),
    }]
    paper = payload.get("paper_name")
    n = payload.get("figure_number")
    if paper and n is not None:
        row = conn.execute(
            "SELECT id FROM papers WHERE paper_name = ?", (paper,)
        ).fetchone()
        if row is not None:
            paper_id = int(row[0])
            try:
                fn = int(n)
            except (TypeError, ValueError):
                fn = None
            if fn is not None:
                _attach_figure_blocks(conn, paper_id, [fn], blocks)
    return _finalize_result(blocks, payload)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


def _bm25_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    query = args["query"]
    scope_val = args.get("scope") or Scope.SECTIONS.value
    filters: dict[str, Any] = {}
    if args.get("domain"):
        filters["domain"] = args["domain"]
    if args.get("collection"):
        filters["collection"] = args["collection"]
    return mode_bm25(
        conn,
        query=query,
        filters=filters,
        limit=int(args.get("limit", 10)),
        scope=Scope(scope_val),
    )


def _lookup_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_taxonomy_lookup(
        conn,
        term=args["term"],
        kind=args["kind"],
        filters={"domain": args.get("domain")},
    )


def _browse_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    which = args["which"]
    filters: dict[str, Any] = {"domain": args.get("domain")}
    if args.get("entity_type"):
        filters["entity_type"] = args["entity_type"]
    if args.get("aliases_term"):
        filters["aliases_term"] = args["aliases_term"]
    return mode_browse(conn, which=which, filters=filters)


def _toc_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_toc(conn, paper_name=args["paper_name"])


def _read_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_read(
        conn,
        paper_name=args["paper_name"],
        section=args.get("section"),
    )


def _figure_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_figure(conn, paper=args["paper"], n=str(args["n"]))


def _repo_tree_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_repo_tree(conn, paper_name=args["paper_name"])


def _read_code_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_read_code(
        conn,
        paper_name=args["paper_name"],
        path=args["path"],
        lines=args.get("lines"),
    )


# Each entry: (name, description, inputSchema, dispatch_fn, attach_mode)
# attach_mode ∈ {"scan", "figure", "none"} — controls which packer we use.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "bm25",
        "description": (
            "BM25 text search across the lodestone corpus. Returns hits "
            "grouped by paper with topics, entity preview, and figure "
            "counts. Snippets are short windows; any (figure:N) refs that "
            "land inside a snippet are appended as inline image content "
            "blocks following the JSON, each preceded by a "
            "'--- paper_id=X figure=N caption=... ---' text marker."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "BM25 query string"},
                "scope": {
                    "type": "string",
                    "enum": [s.value for s in Scope],
                    "default": Scope.SECTIONS.value,
                    "description": (
                        "sections (default), readmes (paper-anchored "
                        "code-repo READMEs), or both (union)."
                    ),
                },
                "domain": {"type": "string"},
                "collection": {"type": "string"},
                "limit": {"type": "integer", "default": 10, "minimum": 1},
            },
            "required": ["query"],
        },
        "dispatch": _bm25_dispatch,
        "attach": "scan",
    },
    {
        "name": "lookup",
        "description": (
            "Resolve a surface-form term to its canonical taxonomy row. "
            "Tier A: porter-stemmed FTS5 match. Tier B: KNN over sentence "
            "embeddings (cosine ≥ 0.80). Returns canonical metadata, "
            "aliases, and the papers that mention it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "term": {"type": "string"},
                "kind": {
                    "type": "string",
                    "enum": ["entity", "topic", "collection"],
                },
                "domain": {"type": "string"},
            },
            "required": ["term", "kind"],
        },
        "dispatch": _lookup_dispatch,
        "attach": "none",
    },
    {
        "name": "browse",
        "description": (
            "List taxonomy/paper rollups: collections, topics, entities of "
            "a given type, aliases of a canonical name, or papers flagged "
            "for review."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "which": {
                    "type": "string",
                    "enum": [
                        "collections", "topics", "entity_type",
                        "aliases", "needs_review",
                    ],
                },
                "domain": {"type": "string"},
                "entity_type": {
                    "type": "string",
                    "description": "required when which='entity_type'",
                },
                "aliases_term": {
                    "type": "string",
                    "description": "required when which='aliases'",
                },
            },
            "required": ["which"],
        },
        "dispatch": _browse_dispatch,
        "attach": "none",
    },
    {
        "name": "toc",
        "description": (
            "Return the level-1..3 ATX header table of contents for a paper."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"paper_name": {"type": "string"}},
            "required": ["paper_name"],
        },
        "dispatch": _toc_dispatch,
        "attach": "none",
    },
    {
        "name": "read",
        "description": (
            "Read a paper's markdown — full body, or a hierarchical "
            "section slice via 'section' (e.g. 'Method' or "
            "'Method > Setup'). Any (figure:N) refs in the returned "
            "markdown are appended as inline image content blocks "
            "following the JSON; each image is preceded by a "
            "'--- paper_id=X figure=N caption=... ---' text marker."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_name": {"type": "string"},
                "section": {
                    "type": "string",
                    "description": "Optional 'Parent > Child' breadcrumb.",
                },
            },
            "required": ["paper_name"],
        },
        "dispatch": _read_dispatch,
        "attach": "scan",
    },
    {
        "name": "figure",
        "description": (
            "Fetch one figure as an inline image content block. 'n' is "
            "tried as an integer figure_number first, then as a "
            "display_number (caption label like 'Figure 3a')."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper": {"type": "string"},
                "n": {
                    "type": ["string", "integer"],
                    "description": "figure_number (int) or display_number (str)",
                },
            },
            "required": ["paper", "n"],
        },
        "dispatch": _figure_dispatch,
        "attach": "figure",
    },
    {
        "name": "repo_tree",
        "description": (
            "List every code_files path for a paper's anchored code repo. "
            "Soft statuses on missing data: 'no_repo', 'failed_repo'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"paper_name": {"type": "string"}},
            "required": ["paper_name"],
        },
        "dispatch": _repo_tree_dispatch,
        "attach": "none",
    },
    {
        "name": "read_code",
        "description": (
            "Read one code file from a paper's anchored repo, optionally "
            "sliced by 1-based line range A-B."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_name": {"type": "string"},
                "path": {"type": "string"},
                "lines": {
                    "type": "string",
                    "description": "Inclusive 1-based range, e.g. '100-200'.",
                },
            },
            "required": ["paper_name", "path"],
        },
        "dispatch": _read_code_dispatch,
        "attach": "none",
    },
]


def _tool_index() -> dict[str, dict[str, Any]]:
    return {t["name"]: t for t in TOOLS}


def _tools_list_payload() -> list[dict[str, Any]]:
    """Project the registry into the spec's tools/list schema."""
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "inputSchema": t["inputSchema"],
        }
        for t in TOOLS
    ]


# ---------------------------------------------------------------------------
# JSON-RPC handlers
# ---------------------------------------------------------------------------


class _ServerState:
    """Holds the (lazily-opened) sqlite connection + the startup-error status,
    if any. The connection is None until we open it lazily on first call.
    Startup faults are surfaced via ``isError`` from each tools/call.
    """

    def __init__(self) -> None:
        self.db_path: Path | None = None
        self.conn: sqlite3.Connection | None = None
        self.startup_error: str | None = None

    def configure(self) -> None:
        """Resolve DB path and try to open. On failure, capture the error
        and let ``initialize`` proceed regardless (issue #35287 mitigation).
        """
        self.db_path = _resolve_db_path()
        if self.db_path is None:
            self.startup_error = (
                "LODESTONE_DB not set and no lodestone.db found by walking "
                "up from CWD. Set LODESTONE_DB to the absolute db path in "
                ".mcp.json env."
            )
            _log("error", self.startup_error)
            return
        if not self.db_path.is_file():
            self.startup_error = (
                f"lodestone.db not found at {self.db_path}. Set "
                f"LODESTONE_DB to the correct absolute path."
            )
            _log("error", self.startup_error)
            return
        try:
            self.conn = get_conn(self.db_path)
        except Exception as exc:  # noqa: BLE001
            self.startup_error = f"failed to open {self.db_path}: {exc!r}"
            _log("error", self.startup_error)
            self.conn = None

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except sqlite3.Error as exc:
                _log("warning", f"sqlite close failed: {exc!r}")
            self.conn = None


def _handle_initialize(state: _ServerState, msg: dict) -> dict:
    params = msg.get("params") or {}
    requested = params.get("protocolVersion")
    proto = (
        requested
        if requested in _SUPPORTED_PROTOCOLS
        else PROTOCOL_VERSION
    )
    return {
        "jsonrpc": "2.0",
        "id": msg.get("id"),
        "result": {
            "protocolVersion": proto,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        },
    }


def _handle_tools_list(msg: dict) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": msg.get("id"),
        "result": {"tools": _tools_list_payload()},
    }


def _handle_tools_call(state: _ServerState, msg: dict) -> dict:
    msg_id = msg.get("id")
    params = msg.get("params") or {}
    name = params.get("name")
    args = params.get("arguments") or {}

    if not isinstance(name, str) or not name:
        return _err(msg_id, _ERR_INVALID_PARAMS, "tools/call requires 'name'")

    tool = _tool_index().get(name)
    if tool is None:
        # Per spec, unknown tools surface as a tool error result rather than
        # a JSON-RPC method-not-found error (the method 'tools/call' is
        # known; the tool name is the parameter that's invalid).
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": f"unknown tool: {name}"}],
                "isError": True,
            },
        }

    if state.startup_error or state.conn is None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": (
                        "lodestone-mcp startup error: "
                        f"{state.startup_error or 'sqlite connection unavailable'}"
                    ),
                }],
                "isError": True,
            },
        }

    try:
        payload = tool["dispatch"](state.conn, args)
    except KeyError as exc:
        # Missing required arg surfaces as KeyError from the dispatcher's
        # ``args[...]`` lookups — translate to InvalidParams at the
        # protocol layer. Tool runtime errors take the isError path below.
        return _err(
            msg_id,
            _ERR_INVALID_PARAMS,
            f"missing required argument: {exc}",
        )
    except (TypeError, ValueError) as exc:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": repr(exc)}],
                "isError": True,
            },
        }
    except Exception as exc:  # noqa: BLE001
        _log("error", f"tool '{name}' raised: {exc!r}")
        traceback.print_exc(file=sys.stderr)
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": repr(exc)}],
                "isError": True,
            },
        }

    attach = tool["attach"]
    if attach == "figure":
        result = _figure_only_result(payload, state.conn)
    elif attach == "scan":
        result = _pack_result(payload, state.conn)
    else:
        result = _pack_result(payload, None)

    # Soft-failure payloads pass through as normal results (isError stays
    # false). The structured content carries the diagnostic.
    if payload.get("status") in _SOFT_FAILURE_STATUSES:
        result["isError"] = False

    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _dispatch(state: _ServerState, msg: dict) -> dict | None:
    """Returns the response dict, or None for notifications."""
    method = msg.get("method")
    msg_id = msg.get("id")

    # Notifications have no `id`. Per spec they get no response.
    is_notification = "id" not in msg

    if method == "initialize":
        return _handle_initialize(state, msg)
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return _handle_tools_list(msg)
    if method == "tools/call":
        return _handle_tools_call(state, msg)
    if method == "notifications/cancelled":
        # Optional per spec; we don't have long-running tools.
        return None

    if is_notification:
        _log("debug", f"ignoring unknown notification: {method}")
        return None
    return _err(msg_id, _ERR_METHOD_NOT_FOUND, f"unknown method: {method}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(stdin=None, stdout=None) -> int:
    if stdin is not None:
        sys.stdin = stdin
    if stdout is not None:
        sys.stdout = stdout

    state = _ServerState()
    state.configure()

    def _on_signal(signum: int, frame: Any) -> None:  # noqa: ARG001
        _log("info", f"signal {signum} received; shutting down")
        state.close()
        # Re-raise as SystemExit so the loop exits cleanly.
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    _log("info", f"starting (db={state.db_path}, startup_error={state.startup_error!r})")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                _log("warning", f"parse error: {exc}: {line[:200]}")
                _send(_err(None, _ERR_PARSE, f"parse error: {exc}"))
                continue
            if not isinstance(msg, dict):
                _send(_err(None, _ERR_INVALID_REQUEST,
                           "request must be a JSON object"))
                continue
            try:
                resp = _dispatch(state, msg)
            except Exception as exc:  # noqa: BLE001
                _log("error", f"dispatch failed: {exc!r}")
                traceback.print_exc(file=sys.stderr)
                if "id" in msg:
                    _send(_err(msg.get("id"), _ERR_INTERNAL, repr(exc)))
                continue
            if resp is not None:
                _send(resp)
    except SystemExit:
        return 0
    finally:
        state.close()

    _log("info", "stdin closed; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
