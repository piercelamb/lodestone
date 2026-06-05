"""Lodestone stdio MCP server.

Wraps the read-side ``mode_*`` functions in :mod:`_system.scripts.search`
as MCP tools served over stdio JSON-RPC (spec rev 2025-06-18). Returns the
existing JSON envelope as a ``text`` content block plus inline ``image``
content blocks for any ``(figure:N)`` markdown refs the response carries.

Tools registered (all surface as ``mcp__lodestone__<name>`` in Claude Code):

    search, bm25, lookup, browse, overview, collection, toc, toc_many,
    read, figure, repo_tree, read_code

Transport notes (per spec §Transports):

* Newline-delimited UTF-8 JSON on stdin/stdout. No Content-Length headers.
* Stdout carries only valid MCP messages — never ``print()``, banners, or
  progress bars. Diagnostics go to stderr.

The server completes ``initialize`` even on bad config (missing DB, schema
mismatch) — that's the issue #35287 mitigation. Failures surface as
``isError: true`` from subsequent ``tools/call`` instead.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import signal
import sqlite3
import sys
import traceback
from contextvars import ContextVar
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

from _system.db.connection import get_conn
from _system.scripts.search import (
    CitationDirection,
    Scope,
    _SOFT_FAILURE_STATUSES,
    format_collection_text,
    format_ingest_paper,
    format_ingest_post,
    format_ingest_repo,
    format_overview_tree,
    format_search_markdown,
    mode_bm25,
    mode_browse,
    mode_citations,
    mode_collection,
    mode_coverage,
    mode_figure,
    mode_overview,
    mode_query,
    mode_read,
    mode_read_code,
    mode_repo,
    mode_repo_tree,
    mode_schema,
    mode_search,
    mode_search_multi,
    mode_tables,
    mode_taxonomy_lookup,
    mode_toc,
    mode_toc_many,
)
from _system.utils.http import reset_progress_hook, set_progress_hook
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
SERVER_VERSION = "0.3.0"

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


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return v if v > 0 else default


def _max_figure_bytes() -> int:
    return _env_positive_int("LODESTONE_MAX_FIGURE_BYTES", _DEFAULT_MAX_FIGURE_BYTES)


def _max_figures_per_response() -> int:
    return _env_positive_int(
        "LODESTONE_MAX_FIGURES_PER_RESPONSE", _DEFAULT_MAX_FIGURES_PER_RESPONSE
    )


# ---------------------------------------------------------------------------
# Stdio framing
# ---------------------------------------------------------------------------


def _send(msg: dict) -> None:
    """Write one newline-delimited JSON message to stdout and flush."""
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


# Per-request SSE frame writer. ``do_POST`` installs this for HTTP requests
# whose ``Accept`` header includes ``text/event-stream``; cleared after the
# request completes. While set, ``_send_progress`` emits frames into the
# response stream instead of stdout.
_SseWriter = Callable[[dict], None]
_sse_writer: ContextVar[Optional[_SseWriter]] = ContextVar(
    "lodestone_mcp_sse_writer", default=None,
)


def _build_progress_payload(
    token: Any, message: str, progress: int, total: int,
) -> dict:
    return {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": token,
            "progress": progress,
            "total": total,
            "message": message,
        },
    }


def _send_progress(token: Any, message: str, progress: int, total: int) -> None:
    """Emit one ``notifications/progress`` message keyed off the client's token.

    Routing:
    - **stdio** transport: write to stdout alongside the response envelope.
    - **http** transport with an SSE writer installed (``Accept: text/event-stream``):
      emit as one SSE frame on the in-flight response.
    - **http** transport without SSE (plain JSON client): no-op — there's no
      out-of-band channel to deliver the notification, and the final response
      will arrive normally.

    No-op when ``token`` is None — clients that don't request progress on
    a tools/call receive nothing here, and the eventual result envelope
    still arrives normally.
    """
    if token is None:
        return
    payload = _build_progress_payload(token, message, progress, total)
    sse = _sse_writer.get()
    if sse is not None:
        sse(payload)
        return
    # Falls through to stdout for stdio transport. HTTP-without-SSE has no
    # writer installed and no stdout sink; the call lands here and is dropped,
    # which matches the prior plain-JSON behavior.
    _send(payload)


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

    Top-level ``slug`` (read/toc) or ``paper_name`` (figure/repo_tree/
    read_code) wins. For BM25 results, fall back to the first hit. Returns
    ``None`` when no paper context is determinable.
    """
    name = payload.get("slug") or payload.get("paper_name")
    if not name:
        results = payload.get("results") or []
        if results and isinstance(results[0], dict):
            name = results[0].get("slug") or results[0].get("paper_name")
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


def _pack_result(
    payload: dict,
    conn: sqlite3.Connection | None,
    *,
    text_format: Any = None,
) -> dict:
    """Build the ``tools/call`` result envelope from a mode_* payload.

    Emits a single text block (markdown if the tool supplied a
    ``text_format`` callable, otherwise the raw JSON dump) plus
    ``structuredContent`` carrying the full JSON shape for callers that
    want it. Appends labeled image blocks for any ``(figure:N)`` refs the
    payload carries when a sqlite connection is available. Soft-failure
    statuses pass through as ``isError: false``.
    """
    if text_format is not None:
        text = text_format(payload)
    else:
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


_DEFAULT_RECENCY_BOOST = 0.2


def _recency_args(args: dict) -> tuple[float, str | None]:
    """Read ``recency_boost`` / ``since`` from a tool-call payload.

    Single-sources the default so the tool schema's documented default
    (0.2) and the dispatch fallback can't drift.
    """
    recency_boost = (
        float(args["recency_boost"])
        if "recency_boost" in args
        else _DEFAULT_RECENCY_BOOST
    )
    since = args.get("since") or None
    return recency_boost, since


def _bm25_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    query = args["query"]
    scope_val = args.get("scope") or Scope.SECTIONS.value
    filters: dict[str, Any] = {}
    if args.get("domain"):
        filters["domain"] = args["domain"]
    if args.get("collection"):
        filters["collection"] = args["collection"]
    recency_boost, since = _recency_args(args)
    return mode_bm25(
        conn,
        query=query,
        filters=filters,
        limit=int(args.get("limit", 15)),
        offset=int(args.get("offset", 0)),
        scope=Scope(scope_val),
        recency_boost=recency_boost,
        since=since,
    )


def _search_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    query = args["query"]
    limit = int(args.get("limit", 5))
    filters: dict[str, Any] = (
        {"domain": args["domain"]} if args.get("domain") else {}
    )
    union = bool(args.get("union", False))
    recency_boost, since = _recency_args(args)
    if isinstance(query, list):
        return mode_search_multi(
            conn,
            queries=[str(q) for q in query],
            filters=filters,
            limit=limit,
            union=union,
            recency_boost=recency_boost,
            since=since,
        )
    return mode_search(
        conn, query=query, filters=filters, limit=limit,
        recency_boost=recency_boost, since=since,
    )


def _lookup_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_taxonomy_lookup(
        conn,
        query=args["query"],
        filters={"domain": args.get("domain")},
        limit=int(args.get("limit") or 50),
        offset=int(args.get("offset", 0)),
    )


def _browse_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    which = args["which"]
    filters: dict[str, Any] = {"domain": args.get("domain")}
    if args.get("entity_type"):
        filters["entity_type"] = args["entity_type"]
    if args.get("aliases_term"):
        filters["aliases_term"] = args["aliases_term"]
    if args.get("collection"):
        filters["collection"] = args["collection"]
    return mode_browse(conn, which=which, filters=filters)


def _overview_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_overview(conn, filters={"domain": args.get("domain")})


def _collection_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    raw = args["collection"]
    names = [raw] if isinstance(raw, str) else [str(x) for x in raw]

    include_abstracts_raw = args.get("include_abstracts")
    if include_abstracts_raw is None:
        include_abstracts = len(names) <= 1
        auto_trimmed = not include_abstracts
    else:
        include_abstracts = bool(include_abstracts_raw)
        auto_trimmed = False

    return mode_collection(
        conn,
        collection_names=names,
        filters={"domain": args.get("domain")},
        include_abstracts=include_abstracts,
        include_topics=bool(args.get("include_topics", True)),
        limit=int(args.get("limit") or 20),
        auto_trimmed=auto_trimmed,
    )


def _coverage_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_coverage(
        conn,
        topic=args["topic"],
        domain=args.get("domain") or None,
    )


def _toc_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_toc(conn, slug=args["slug"])


def _toc_many_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_toc_many(conn, slugs=list(args["slugs"]))


def _read_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_read(
        conn,
        slug=args["slug"],
        section=args.get("section"),
    )


def _figure_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_figure(conn, paper=args["paper"], n=str(args["n"]))


def _repo_tree_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_repo_tree(
        conn,
        paper_name=args.get("paper_name"),
        repo=args.get("repo"),
    )


def _read_code_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_read_code(
        conn,
        paper_name=args.get("paper_name"),
        repo=args.get("repo"),
        path=args["path"],
        lines=args.get("lines"),
    )


def _repo_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_repo(conn, repo=args["repo"])


def _citations_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_citations(
        conn,
        slug=args["slug"],
        direction=args.get("direction", CitationDirection.OUTBOUND.value),
        limit=int(args.get("limit", 50)),
        offset=int(args.get("offset", 0)),
    )


def _tables_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_tables(
        conn, include_internal=bool(args.get("include_internal", False))
    )


def _schema_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    raw = args["tables"]
    names = [raw] if isinstance(raw, str) else [str(x) for x in raw]
    return mode_schema(conn, table_names=names)


def _query_dispatch(conn: sqlite3.Connection, args: dict) -> dict:
    return mode_query(conn, sql=args["sql"])


def _prefetch_lodestone_models(progress) -> None:
    """Pre-warm both HF model caches with one consolidated progress stream.

    Otherwise bge (resolver) and gliner (entity extraction) trickle in
    lazily at different pipeline stages, surfacing as two separate
    multi-minute hangs to the user. Both are pulled before the pipeline
    starts so the actual ingest stages observe a warm cache and their
    own model-load calls become no-ops.
    """
    from _system.scripts.validate_models import (
        ModelId,
        _CumulativeProgress,
        ensure_model_cached,
    )

    cp = _CumulativeProgress([
        (ModelId.BGE,     "bge-small-en-v1.5"),
        (ModelId.GLINER2, "gliner2-large-v1"),
    ])
    if progress is not None:
        progress("preparing lodestone models", 0, cp.total)
    for model_id, label in cp.stages_with_labels:
        with cp.stage(model_id, label):
            ensure_model_cached(model_id)


def _sanitized_domain_arg(args: dict) -> str | None:
    """Canonicalize ``domain`` to its slug form before it reaches the pipeline.

    Fetch stages persist the override verbatim onto rows *before* classify
    runs, so the operator-supplied value has to be canonical by the time
    it crosses the dispatcher boundary. Raises ``ValueError`` (surfaced as
    a tool-level ``isError``) if the supplied value is the wrong type or
    collapses to empty after sanitization. ``domain`` absent / null is the
    no-override signal; any other shape is an operator mistake we report.
    """
    from _system.utils.slug import sanitize_domain

    if "domain" not in args:
        return None
    raw = args["domain"]
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(
            f"domain must be a string, got {type(raw).__name__}={raw!r}"
        )
    sanitized = sanitize_domain(raw)
    if not sanitized:
        raise ValueError(
            f"domain={raw!r} sanitizes to empty string; "
            "use letters, digits, '_' or '-'"
        )
    return sanitized


def _ingest_paper_dispatch(
    conn: sqlite3.Connection, args: dict, progress=None,
) -> dict:
    # Imports kept local: ingest pulls in HF model validation and the full
    # pipeline graph, which is dead weight for the read-side tools.
    from _system.db.migrations import init_db
    from _system.scripts.ingest import ingest, ingest_acl
    from _system.scripts.validate_models import check_models
    from _system.utils.acl_urls import parse_acl_id
    from _system.utils.arxiv_urls import parse_arxiv_id

    # Sanitize before any expensive work (model checks / preload) so a
    # bad --domain fails fast.
    domain = _sanitized_domain_arg(args)

    if progress is not None:
        progress("checking models", 0, 1)
    check_models()
    _prefetch_lodestone_models(progress)
    init_db(conn)

    # Try ACL first: parse_acl_id is strict (modern YYYY.venue.N or legacy
    # [A-Z]\d{2}-\d{4} after URL stripping) and arxiv ids never match its
    # regex, so there's no false-positive collision. parse_arxiv_id is more
    # permissive and would happily mangle a malformed ACL input.
    try:
        acl_id = parse_acl_id(args["url"])
    except ValueError:
        pass
    else:
        return ingest_acl(
            conn=conn,
            acl_id=acl_id,
            force=bool(args.get("force", False)),
            domain=domain,
            progress=progress,
        )

    arxiv_id = parse_arxiv_id(args["url"])
    return ingest(
        conn=conn,
        arxiv_id=arxiv_id,
        force=bool(args.get("force", False)),
        domain=domain,
        progress=progress,
    )


def _ingest_repo_dispatch(
    conn: sqlite3.Connection, args: dict, progress=None,
) -> dict:
    from _system.db.migrations import init_db
    from _system.scripts.ingest import ingest_repo_only
    from _system.scripts.validate_models import check_models

    domain = _sanitized_domain_arg(args)

    if progress is not None:
        progress("checking models", 0, 1)
    check_models()
    _prefetch_lodestone_models(progress)
    init_db(conn)
    return ingest_repo_only(
        conn=conn,
        repo_url=args["url"],
        force=bool(args.get("force", False)),
        domain=domain,
        progress=progress,
    )


def _ingest_post_dispatch(
    conn: sqlite3.Connection, args: dict, progress=None,
) -> dict:
    from _system.db.migrations import init_db
    from _system.scripts.ingest import ingest_post
    from _system.scripts.validate_models import check_models

    domain = _sanitized_domain_arg(args)

    if progress is not None:
        progress("checking models", 0, 1)
    check_models()
    _prefetch_lodestone_models(progress)
    init_db(conn)
    return ingest_post(
        conn=conn,
        url=args["url"],
        force=bool(args.get("force", False)),
        domain=domain,
        progress=progress,
    )


class AttachMode(StrEnum):
    """Controls which figure-attach packer wraps a tool's payload."""

    SCAN = "scan"      # walk the payload for (figure:N) refs and inline them
    FIGURE = "figure"  # payload is itself a single figure — fetch the BLOB
    NONE = "none"      # text-only, no figure attachment


class Transport(StrEnum):
    """Wire transport for the MCP server. HTTP is the issue #51736 workaround."""

    STDIO = "stdio"
    HTTP = "http"


TOOLS: list[dict[str, Any]] = [
    {
        "name": "search",
        "description": (
            "First-pass exploratory search across the lodestone corpus. "
            "Returns three buckets in one call: (1) taxonomy — canonical "
            "term hits across entity/topic/collection mixed, each tagged "
            "with its kind; (2) sections — BM25 hits over paper section "
            "text; (3) readmes — BM25 hits over paper-anchored code-repo "
            "READMEs. Use this FIRST when you don't yet know what's in the "
            "corpus; then drill in with 'lookup', 'read'/'toc', or 'bm25'. "
            "No images — keep this cheap for orientation.\n\n"
            "Query syntax (GitHub-code-search subset; not regex):\n"
            "  - bare words = implicit AND, punctuation auto-defanged\n"
            "    so 'tree-sitter' searches the literal phrase, not 'tree NOT sitter'\n"
            "  - \"chain of thought\" = exact phrase\n"
            "  - reasoning OR planning = alternation (uppercase operators)\n"
            "  - reasoning NOT supervised = exclusion\n"
            "  - (monte carlo) tree* = grouping + prefix\n"
            "  - paper:NAME, domain:NAME, collection:NAME = metadata filter\n"
            "  - surface:sections|readmes|taxonomy = only that bucket\n"
            "    surface:both = sections + readmes (legacy alias, no taxonomy)\n"
            "    omitted = all three buckets\n"
            "  - kind:entity|topic|collection = narrow taxonomy bucket\n"
            "  - /regex/ is NOT supported (FTS5 is token-based)\n"
            "Examples:\n"
            "  'chain of thought' OR reasoning\n"
            "  paper:treeofthoughts_2023 deliberate\n"
            "  tree* NOT supervised\n"
            "  kind:entity reasoning\n"
            "  surface:readmes embedding\n\n"
            "Multi-query: pass an array of up to 8 query strings as "
            "'query' to fan out independent searches in one call. Each "
            "query is parsed and executed on its own (own qualifiers, own "
            "operators, own soft-failure status); results are concatenated "
            "in one response under per-query H2 sections. Use this when "
            "you want to orient across several disjoint angles at once "
            "without paying multiple round-trips:\n"
            "  query=[\"chain of thought\", \"tree of thoughts\", "
            "\"self-consistency\"]\n"
            "Per-query 'limit' is shared across the fan-out — keep it "
            "small (default 5)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 8,
                        },
                    ],
                    "description": (
                        "Free-text query, OR an array of 1-8 query "
                        "strings. Each string in an array is parsed and "
                        "executed independently; results are concatenated "
                        "in one response."
                    ),
                },
                "domain": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Max rows per bucket (default 5).",
                },
                "union": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Only meaningful when 'query' is an array. Fuses "
                        "per-query section/readme buckets into a single "
                        "ranked list via Reciprocal Rank Fusion (k=60); "
                        "each hit is tagged with 'matched_queries' (the "
                        "indices of queries that matched it). Use when you "
                        "want 'any doc matching any of these N concepts' "
                        "without merging by hand."
                    ),
                },
                "recency_boost": {
                    "type": "number",
                    "default": 0.2,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Soft tilt toward recent papers. Multiplies BM25 "
                        "rank by (1 + boost * exp(-age_days/730)) on the "
                        "sections surface, so a paper from today gets "
                        "(1+boost) and a 2-year-old paper gets ~(1 + "
                        "boost/e). Default 0.2 tilts lexical ties toward "
                        "recent without overwhelming strong-old hits. Pass "
                        "0.0 to disable. Has no effect on README hits."
                    ),
                },
                "since": {
                    "type": "string",
                    "description": (
                        "Hard floor on publication date. Use for explicit "
                        "slices like 'only papers from 2025-10 onward'. "
                        "Accepts YYYY (normalized to YYYY-01-01) or "
                        "YYYY-MM-DD. Filters readme hits via paper-link "
                        "only — standalone repos pass through."
                    ),
                },
            },
            "required": ["query"],
        },
        "dispatch": _search_dispatch,
        "attach": AttachMode.NONE,
        "text_format": format_search_markdown,
    },
    {
        "name": "bm25",
        "description": (
            "BM25 text search across the lodestone corpus. Returns hits "
            "grouped by paper with topics, entity preview, and figure "
            "counts. Snippets are short windows; any (figure:N) refs that "
            "land inside a snippet are appended as inline image content "
            "blocks following the JSON, each preceded by a "
            "'--- paper_id=X figure=N caption=... ---' text marker.\n\n"
            "Query syntax (GitHub-code-search subset; not regex):\n"
            "  - bare words = implicit AND, punctuation auto-defanged\n"
            "    so 'tree-sitter' searches the literal phrase\n"
            "  - \"exact phrase\" = phrase query\n"
            "  - a OR b, a NOT b = alternation / exclusion (uppercase ops)\n"
            "  - (group) and term* = parens + prefix\n"
            "  - paper:NAME, domain:NAME, collection:NAME = metadata filter\n"
            "  - surface:sections|readmes|both = pick the surface\n"
            "    (overrides/agrees with the 'scope' kwarg)\n"
            "  - kind: NOT supported here — use 'search' for the taxonomy bucket\n"
            "  - /regex/ is NOT supported\n"
            "Examples:\n"
            "  paper:bookrag_2024 indexing\n"
            "  \"hierarchical retrieval\" OR HiRe\n"
            "  tree* NOT supervised\n"
            "  surface:readmes embedding\n\n"
            "Pagination: pass `offset` (default 0) to skip ranked hits. "
            "The response carries `total_hits` (total matches for this "
            "query) and `has_more` (whether `offset + len(results) < "
            "total_hits`); raise `offset` by `limit` to walk forward. "
            "For scope=both each surface paginates independently with "
            "the same offset/limit, and `total_hits` is the sum across "
            "both surfaces."
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
                "limit": {"type": "integer", "default": 15, "minimum": 1},
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": (
                        "Skip this many ranked hits before returning. "
                        "Pair with `limit` to page deeper into a query: "
                        "offset=10, limit=10 returns hits 11-20. The "
                        "response carries `total_hits` and `has_more` "
                        "so you know whether another page exists."
                    ),
                },
                "recency_boost": {
                    "type": "number",
                    "default": 0.2,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Soft tilt toward recent papers. Multiplies BM25 "
                        "rank by (1 + boost * exp(-age_days/730)) on the "
                        "sections surface, so a paper from today gets "
                        "(1+boost) and a 2-year-old paper gets ~(1 + "
                        "boost/e). Default 0.2 tilts lexical ties toward "
                        "recent without overwhelming strong-old hits. Pass "
                        "0.0 to disable. Has no effect on README hits."
                    ),
                },
                "since": {
                    "type": "string",
                    "description": (
                        "Hard floor on publication date. Use for explicit "
                        "slices like 'only papers from 2025-10 onward'. "
                        "Accepts YYYY (normalized to YYYY-01-01) or "
                        "YYYY-MM-DD. Filters readme hits via paper-link "
                        "only — standalone repos pass through."
                    ),
                },
            },
            "required": ["query"],
        },
        "dispatch": _bm25_dispatch,
        "attach": AttachMode.SCAN,
    },
    {
        "name": "lookup",
        "description": (
            "Canonical-term FTS5 search across the taxonomy (entities, "
            "topics, collections), with aliases inlined per hit. Use this "
            "when 'search' has surfaced a canonical row you want to drill "
            "into, or when you want to enumerate every alias for a term.\n"
            "\n"
            "Accepts the same GitHub-flavored query syntax as 'search' / "
            "'bm25': bare tokens (implicit AND), \"phrase\", AND/OR/NOT "
            "(uppercase), parens, term* prefix. Two qualifiers apply:\n"
            "  - kind:entity|topic|collection — narrow the term_type\n"
            "  - domain:NAME — restrict to one domain\n"
            "paper:, collection:, surface: are rejected (no meaning here).\n"
            "\n"
            "Returns up to `limit` ranked hits. Each hit carries "
            "canonical_name, kind, type/entity_type, domain, an aliases "
            "array ([{alias, source_paper}]), and papers "
            "([{paper_name, code_repo}]). For topic/collection hits a "
            "`papers_count` field is also included; entity hits omit it "
            "because the underlying papers list comes from a synonym "
            "index that misses canonical-surface mentions, so any count "
            "would underreport. FTS5-only — no semantic fallback; use "
            "'search' for a wider sweep.\n\n"
            "Pagination: pass `offset` (default 0) to skip ranked hits. "
            "The response carries `total_hits` (total canonical-term "
            "matches for this query) and `has_more` (whether "
            "`offset + len(hits) < total_hits`); raise `offset` by "
            "`limit` to walk forward."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "domain": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 50,
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": (
                        "Skip this many ranked hits before returning. "
                        "Pair with `limit` to page deeper. The response "
                        "carries `total_hits` and `has_more` to drive "
                        "the next call."
                    ),
                },
            },
            "required": ["query"],
        },
        "dispatch": _lookup_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "browse",
        "description": (
            "List taxonomy/paper rollups: collections, topics, entities of "
            "a given type, aliases of a canonical name, or papers flagged "
            "for review. For which='topics', pass 'collection' to scope the "
            "topic rollup to one collection (papers, posts, and repos all "
            "flow through the polymorphic `collections` junction); pass "
            "'domain' to disambiguate collection names that exist in "
            "multiple domains."
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
                "collection": {
                    "type": "string",
                    "description": (
                        "optional when which='topics': scope the topic "
                        "rollup to one collection"
                    ),
                },
            },
            "required": ["which"],
        },
        "dispatch": _browse_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "overview",
        "description": (
            "Top-down corpus map. Domains are broad research areas; each "
            "domain contains collections that subdivide it by approach or "
            "technique. This tool returns the nested domains → collections "
            "tree with per-kind counts (paper_count, post_count, "
            "repo_count). Use it FIRST when you want to navigate by "
            "structure rather than keywords (the complement to "
            "'search'). Then drill into one or more collections via "
            "'collection' to see papers/posts with abstracts/topics, and "
            "feed slugs into 'toc_many' to inspect structures "
            "side-by-side. The tree drops empty domains/collections "
            "(zero rows)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Optional: restrict the tree to one domain.",
                },
            },
        },
        "dispatch": _overview_dispatch,
        "attach": AttachMode.NONE,
        "text_format": format_overview_tree,
    },
    {
        "name": "collection",
        "description": (
            "Drill into one or more collections (typically picked from "
            "'overview') and return their papers as a light tree. "
            "include_abstracts defaults to True for a single collection "
            "and False for multiple, so a fan-out call stays slim; pass "
            "include_abstracts=true explicitly to force abstracts on a "
            "multi-collection call. The response carries 'include_abstracts' "
            "(effective value) and 'auto_trimmed' (whether the slim default "
            "was applied). Set include_topics=false to drop topics. Pass "
            "'collection' as a string or array of up to 16 names; missing "
            "names land in 'missing' rather than raising. If a collection "
            "name exists under multiple domains and 'domain' is not set, "
            "all matches are returned."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "collection": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 16,
                        },
                    ],
                    "description": (
                        "Collection name, or an array of 1-16 collection "
                        "names to bundle in one call."
                    ),
                },
                "domain": {"type": "string"},
                "include_abstracts": {
                    "type": "boolean",
                    "description": (
                        "Default: True for a single collection, False for "
                        "multiple. Pass explicitly to override."
                    ),
                },
                "include_topics": {"type": "boolean", "default": True},
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Max papers per collection.",
                },
            },
            "required": ["collection"],
        },
        "dispatch": _collection_dispatch,
        "attach": AttachMode.NONE,
        "text_format": format_collection_text,
    },
    {
        "name": "coverage",
        "description": (
            "Defensible 'does lodestone cover X?' probe. Combines lexical "
            "hits (FTS over sections + readmes), exact taxonomy matches, "
            "and fuzzy nearest-neighbors (rapidfuzz >= 70) across "
            "collections, canonical entities/topics, and aliases. Returns "
            "structured counts + similarity scores — no heuristic "
            "high/medium/low synthesis. Use when you need to back a "
            "negative claim ('lodestone has no first-class coverage of X') "
            "with a single citable lookup."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The concept name to probe.",
                },
                "domain": {
                    "type": "string",
                    "description": "Optional: restrict the probe to one domain.",
                },
            },
            "required": ["topic"],
        },
        "dispatch": _coverage_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "toc",
        "description": (
            "Return the level-1..3 ATX header table of contents for a "
            "paper or blog post. 'slug' accepts either kind — the slug "
            "namespace is shared."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"],
        },
        "dispatch": _toc_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "toc_many",
        "description": (
            "Return the level-1..3 ATX header table of contents for "
            "multiple sources in one call. 'slugs' may mix paper and "
            "post slugs freely. Use this after 'search' / 'bm25' / "
            "'lookup' surfaces several candidates and you want to scan "
            "their structures side-by-side before deciding where to "
            "'read'. Slugs that don't resolve are reported in 'missing' "
            "instead of raising — a typo in one doesn't abandon the rest."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "slugs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
            "required": ["slugs"],
        },
        "dispatch": _toc_many_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "read",
        "description": (
            "Read a source's markdown — full body, or a hierarchical "
            "section slice via 'section' (e.g. 'Method' or "
            "'Method > Setup'). 'slug' accepts either a paper or post "
            "slug. Any (figure:N) refs in the returned markdown are "
            "appended as inline image content blocks following the JSON; "
            "each image is preceded by a '--- paper_id=X figure=N "
            "caption=... ---' text marker. (Posts don't carry figure "
            "refs in v1.)"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "section": {
                    "type": "string",
                    "description": "Optional 'Parent > Child' breadcrumb.",
                },
            },
            "required": ["slug"],
        },
        "dispatch": _read_dispatch,
        "attach": AttachMode.SCAN,
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
        "attach": AttachMode.FIGURE,
    },
    {
        "name": "repo_tree",
        "description": (
            "List every code_files path for a repo. Identify by exactly "
            "one of `paper_name` (the paper's anchored repo) or `repo` "
            "(repo_slug, works for standalone repos). Soft statuses on "
            "missing data: 'no_repo', 'failed_repo'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_name": {"type": "string"},
                "repo": {
                    "type": "string",
                    "description": "repo_slug (e.g. gh-owner-name)",
                },
            },
        },
        "dispatch": _repo_tree_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "read_code",
        "description": (
            "Read one code file from a repo, optionally sliced by 1-based "
            "line range A-B. Identify by exactly one of `paper_name` or "
            "`repo` (repo_slug)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "paper_name": {"type": "string"},
                "repo": {
                    "type": "string",
                    "description": "repo_slug (e.g. gh-owner-name)",
                },
                "path": {"type": "string"},
                "lines": {
                    "type": "string",
                    "description": "Inclusive 1-based range, e.g. '100-200'.",
                },
            },
            "required": ["path"],
        },
        "dispatch": _read_code_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "repo",
        "description": (
            "Return one repo's metadata, topics, and (if any) linked paper. "
            "Single SELECT — cheap; useful as a 'tell me about this repo' "
            "step before drilling into repo_tree / read_code."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "repo_slug (e.g. gh-owner-name)",
                },
            },
            "required": ["repo"],
        },
        "dispatch": _repo_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "citations",
        "description": (
            "Outbound or inbound citation graph for a paper or post slug.\n\n"
            "Outbound (direction='outbound', default): the references this "
            "source cites, bucketed by resolution state:\n"
            "  - resolved — cited paper is already in the corpus "
            "(carries slug, title, arxiv_id, raw_text, cited_status); pivot "
            "to it via `read` / `toc`.\n"
            "  - missing — cited an arxiv id we haven't ingested "
            "(carries arxiv_id, raw_text, ingest_hint with the exact "
            "`ingest_paper` call to run).\n"
            "  - unresolvable — reference has no arxiv id at all "
            "(non-arxiv venues / hand-typed bibitems with missing eprint).\n"
            "Outbound returns ALL refs (no pagination); response is capped "
            "at 500 rows with truncated=true if exceeded — paper "
            "bibliographies are O(50-200) refs in practice.\n\n"
            "Inbound (direction='inbound'): papers + posts in the corpus "
            "that cite this paper, unioned and ordered by date DESC. "
            "Paginated by `limit` / `offset` — response carries "
            "`total_hits` and `has_more`. Inbound is paper-only — passing a "
            "post slug returns soft-status `unsupported_direction` because "
            "the schema's cited_paper_id only points at papers.\n\n"
            "Soft statuses: `not_found` (slug missed all three "
            "namespaces), `unsupported_direction` (post + inbound, repo + "
            "anything), `invalid_pagination` (negative limit/offset)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "description": "paper_name or post_name.",
                },
                "direction": {
                    "type": "string",
                    "enum": [d.value for d in CitationDirection],
                    "default": CitationDirection.OUTBOUND.value,
                    "description": (
                        "outbound (what this source cites) or inbound "
                        "(what cites this paper)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "description": "inbound only — page size (default 50).",
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": (
                        "inbound only — skip this many ranked rows. Pair "
                        "with `limit` to walk forward."
                    ),
                },
            },
            "required": ["slug"],
        },
        "dispatch": _citations_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "tables",
        "description": (
            "List every user table / view / virtual table in lodestone.db. "
            "Use this as the first step of the SQL escape hatch ('I don't "
            "know what's in here yet'); follow up with 'schema' on the "
            "tables that look interesting, then 'query' to read rows. "
            "Prefer the curated tools (bm25, lookup, read, toc, "
            "collection, etc.) for everything they cover — this triple is "
            "for cases none of those fit.\n\n"
            "Virtual tables (FTS5 / vec0) are tagged 'virtual' so they're "
            "easy to spot. FTS5 / vec0 shadow tables (suffixed _data, "
            "_idx, _content, _docsize, _config) are filtered out by "
            "default; pass include_internal=true to see them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_internal": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Also return FTS5 / vec0 shadow tables. Off by "
                        "default — they're noise for most questions."
                    ),
                },
            },
        },
        "dispatch": _tables_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "schema",
        "description": (
            "Return CREATE DDL + columns + indexes for one or more tables. "
            "Pass 'tables' as a string or array of names. Names that don't "
            "resolve land in 'missing' rather than raising. If you don't "
            "know which tables exist, call 'tables' first.\n\n"
            "Each entry carries: name, type (table/virtual/view), sql "
            "(the CREATE statement), columns (cid/name/type/notnull/"
            "dflt_value/pk), and indexes (name/unique/origin/partial). "
            "This is the second step of the SQL escape hatch — feed the "
            "DDL into 'query' to write a precise SELECT."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tables": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    ],
                    "description": (
                        "Table name, or an array of table names to "
                        "fetch in one call."
                    ),
                },
            },
            "required": ["tables"],
        },
        "dispatch": _schema_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "query",
        "description": (
            "Run a single read-only SQL statement against lodestone.db. "
            "This is the SQL escape hatch — reach for the curated tools "
            "FIRST (bm25, lookup, read, toc, collection, overview, "
            "repo_tree, read_code). Use 'query' only when none of those "
            "fit your question. Call 'tables' / 'schema' first if you "
            "don't already know the column layout.\n\n"
            "Contract:\n"
            "  - read-only is engine-enforced (mode=ro URI). Any DML / "
            "    DDL (INSERT/UPDATE/DELETE/CREATE/DROP/etc.) returns a "
            "    'read_only_violation' soft-fail.\n"
            "  - exactly one statement per call. Multiple statements "
            "    return 'multiple_statements'.\n"
            "  - hard ceiling of 1000 rows. Larger result sets surface "
            "    truncated=true; paginate by writing LIMIT N OFFSET M "
            "    with a stable ORDER BY in YOUR OWN SQL.\n"
            "  - 5s wall-clock timeout. Slow queries return "
            "    'query_timeout'.\n"
            "  - BLOB columns are summarized as "
            "    {'_blob': true, 'size_bytes': N} — use the 'figure' or "
            "    'read_code' tools to fetch real binary content.\n\n"
            "Example pagination (page through papers ordered by "
            "ingested_at):\n"
            "  page 1: SELECT paper_name, title FROM papers "
            "ORDER BY ingested_at DESC LIMIT 50 OFFSET 0\n"
            "  page 2: SELECT paper_name, title FROM papers "
            "ORDER BY ingested_at DESC LIMIT 50 OFFSET 50\n"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": (
                        "A single read-only SQL statement. No trailing "
                        "';' separator needed."
                    ),
                },
            },
            "required": ["sql"],
        },
        "dispatch": _query_dispatch,
        "attach": AttachMode.NONE,
    },
    {
        "name": "ingest_paper",
        "description": (
            "Ingest an arXiv or ACL Anthology paper into the lodestone DB. "
            "Runs the full pipeline: fetch → convert → classify → extract → "
            "index. Routes by input shape: ACL Anthology ids (e.g. "
            "'2021.acl-long.285', 'P19-1001') and aclanthology.org URLs go "
            "through the ACL pipeline (MODS metadata + PDF body, no repo "
            "discovery); everything else is treated as arXiv. Resumable — "
            "re-running on a paper that partially ingested picks up at the "
            "last completed stage; pass force=true to wipe and re-ingest "
            "from scratch (preserves global taxonomy). Emits MCP progress "
            "notifications between stages so the client can render a "
            "progress bar; total ticks = number of stages remaining for "
            "this run. If an arxiv paper ships a code repo URL, the linked "
            "repo is registered and cloned as a follow-up after the "
            "'complete' tick (no progress events for that step); ACL "
            "papers have no repo-discovery step."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "arXiv URL or bare ID (version suffix preserved "
                        "verbatim — '2301.12345v1' and '2301.12345v2' are "
                        "different rows), OR an ACL Anthology id / URL "
                        "(e.g. '2021.acl-long.285', 'P19-1001', "
                        "'https://aclanthology.org/2021.acl-long.285/', "
                        "or the '.pdf' / '.xml' / '.bib' asset URL)."
                    ),
                },
                "force": {"type": "boolean", "default": False},
                "domain": {
                    "type": "string",
                    "description": "Optional: override the classifier's domain choice.",
                },
            },
            "required": ["url"],
        },
        "dispatch": _ingest_paper_dispatch,
        "attach": AttachMode.NONE,
        "accepts_progress": True,
        "text_format": format_ingest_paper,
    },
    {
        "name": "ingest_repo",
        "description": (
            "Ingest a standalone code repo into the lodestone DB. Runs "
            "resolve → fetch → classify. Use this for repos with no "
            "associated paper; for paper-linked repos, ingest the paper "
            "via 'ingest_paper' (the repo follow-up runs automatically). "
            "Emits MCP progress notifications between stages."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "github/gitlab/bitbucket URL",
                },
                "force": {"type": "boolean", "default": False},
                "domain": {"type": "string"},
            },
            "required": ["url"],
        },
        "dispatch": _ingest_repo_dispatch,
        "attach": AttachMode.NONE,
        "accepts_progress": True,
        "text_format": format_ingest_repo,
    },
    {
        "name": "ingest_post",
        "description": (
            "Ingest a blog post URL into lodestone. Runs fetch → convert "
            "(trafilatura HTML→markdown) → classify → extract → index. "
            "Resumable — re-running on a post that partially ingested "
            "picks up at the last completed stage; pass force=true to "
            "wipe and re-ingest from scratch (preserves global taxonomy). "
            "Outbound arxiv-id citations from the post body are pulled "
            "into post_references and forward-resolved against the "
            "papers table; a future paper that gets ingested will "
            "backward-resolve the same row. If the post links a github "
            "repo, the repo is registered via the standalone path "
            "(no post→repo linkage in v1)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Post URL (e.g. https://lilianweng.github.io/posts/...).",
                },
                "force": {"type": "boolean", "default": False},
                "domain": {
                    "type": "string",
                    "description": "Optional: override the classifier's domain choice.",
                },
            },
            "required": ["url"],
        },
        "dispatch": _ingest_post_dispatch,
        "attach": AttachMode.NONE,
        "accepts_progress": True,
        "text_format": format_ingest_post,
    },
]


_TOOL_INDEX: dict[str, dict[str, Any]] = {t["name"]: t for t in TOOLS}

_TOOLS_LIST_PAYLOAD: list[dict[str, Any]] = [
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
        self.transport: Transport = Transport.STDIO
        # Inode of ``db_path`` at open time. Compared against ``stat`` on
        # each tools/call so a path-vs-inode divergence (file unlinked and
        # recreated under us — see "ghost DB" footgun) is caught with a
        # loud error instead of silently serving a stale view.
        self.opened_inode: int | None = None

    def configure(self) -> None:
        """Resolve DB path and try to open. On failure, capture the error
        and let ``initialize`` proceed regardless (issue #35287 mitigation).

        When ``LODESTONE_DB`` is explicitly set and the file is missing,
        the parent dir is created and an empty schema is initialized so
        a fresh install becomes usable on the first ``ingest_*`` call —
        the README promises this. The walk-up fallback (no env var) keeps
        its strict "must exist" semantics so devs running from inside the
        repo don't accidentally seed a stray DB in an unrelated parent.
        """
        from _system.db.migrations import init_db

        self.db_path = _resolve_db_path()
        if self.db_path is None:
            self.startup_error = (
                "LODESTONE_DB not set and no lodestone.db found by walking "
                "up from CWD. Set LODESTONE_DB to the absolute db path in "
                ".mcp.json env."
            )
            _log("error", self.startup_error)
            return
        env_set = os.environ.get("LODESTONE_DB") is not None
        if not self.db_path.is_file() and not env_set:
            self.startup_error = (
                f"lodestone.db not found at {self.db_path}. Set "
                f"LODESTONE_DB to the correct absolute path."
            )
            _log("error", self.startup_error)
            return
        try:
            if not self.db_path.is_file():
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = get_conn(self.db_path)
            init_db(self.conn)
            try:
                self.opened_inode = self.db_path.stat().st_ino
            except OSError:
                # Path resolution issues here are cosmetic — the conn is
                # already bound. Leave opened_inode None so the per-call
                # check is a no-op rather than a false alarm.
                self.opened_inode = None
        except Exception as exc:  # noqa: BLE001
            self.startup_error = f"failed to open {self.db_path}: {exc!r}"
            _log("error", self.startup_error)
            if self.conn is not None:
                try:
                    self.conn.close()
                except sqlite3.Error:
                    pass
            self.conn = None
            self.opened_inode = None

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.close()
            except sqlite3.Error as exc:
                _log("warning", f"sqlite close failed: {exc!r}")
            self.conn = None
        self.opened_inode = None


def _check_db_inode_pinned(state: _ServerState) -> str | None:
    """Verify ``state.db_path`` still resolves to the inode we opened.

    Catches the "ghost DB" footgun: another process unlinks the canonical
    DB and recreates a fresh file at the same path while this server is
    holding the original inode open. SQLite happily keeps writing to the
    orphaned inode — clients get stale results, and the data dies with
    this process. Cheap (`os.stat`), runs once per ``tools/call``.

    Returns ``None`` when nothing is wrong (or the check is not
    applicable — ``:memory:`` DBs, missing ``opened_inode``). Returns an
    error message string when the path was replaced; the caller surfaces
    it as a tool error so the user notices immediately.
    """
    if state.opened_inode is None or state.db_path is None:
        return None
    if str(state.db_path) == ":memory:":
        return None
    try:
        current_inode = state.db_path.stat().st_ino
    except OSError as exc:
        return (
            f"lodestone-mcp: db path {state.db_path} is no longer accessible "
            f"({exc!r}); restart this MCP server."
        )
    if current_inode == state.opened_inode:
        return None
    return (
        f"lodestone-mcp: db path {state.db_path} was replaced underneath "
        f"this server (inode {state.opened_inode} -> {current_inode}). "
        "Another process recreated the file, so writes here are landing in "
        "an orphaned inode. Restart this MCP server (and any other "
        "lodestone-mcp processes) to re-pin to the live file."
    )


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
        "result": {"tools": _TOOLS_LIST_PAYLOAD},
    }


def _tool_error(msg_id: Any, text: str) -> dict:
    """Wrap a free-text error as a ``tools/call`` result with ``isError: true``.

    Per spec, tool-level failures stay inside ``result`` (with ``isError``)
    rather than becoming JSON-RPC errors — that's what the MCP client looks
    at to decide whether to surface the failure to the model.
    """
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": True,
        },
    }


def _handle_tools_call(state: _ServerState, msg: dict) -> dict:
    msg_id = msg.get("id")
    params = msg.get("params") or {}
    name = params.get("name")
    args = params.get("arguments") or {}

    if not isinstance(name, str) or not name:
        return _err(msg_id, _ERR_INVALID_PARAMS, "tools/call requires 'name'")

    tool = _TOOL_INDEX.get(name)
    if tool is None:
        return _tool_error(msg_id, f"unknown tool: {name}")

    if state.startup_error or state.conn is None:
        return _tool_error(
            msg_id,
            "lodestone-mcp startup error: "
            f"{state.startup_error or 'sqlite connection unavailable'}",
        )

    swap_err = _check_db_inode_pinned(state)
    if swap_err is not None:
        return _tool_error(msg_id, swap_err)

    meta = params.get("_meta") or {}
    progress_token = meta.get("progressToken")

    try:
        if tool.get("accepts_progress"):
            # Progress notifications need an out-of-band channel:
            # - stdio transport always has one (stdout itself).
            # - http transport has one only if the client opted into
            #   ``Accept: text/event-stream`` and ``do_POST`` installed
            #   an SSE writer in the contextvar.
            # Plain-JSON HTTP clients get a no-op cb — there's nowhere
            # to deliver progress until the final response.
            has_progress_channel = (
                state.transport is Transport.STDIO
                or _sse_writer.get() is not None
            )
            if has_progress_channel:
                def _progress_cb(message: str, p: int, total: int) -> None:
                    _send_progress(progress_token, message, p, total)
            else:
                def _progress_cb(message: str, p: int, total: int) -> None:
                    return
            # Also surface HTTP retries (429 backoff, content-fetch
            # transient blips) through the same channel via the
            # contextvar hook in _system.utils.http. Cleared in finally.
            hook_token = set_progress_hook(_progress_cb)
            try:
                payload = tool["dispatch"](state.conn, args, _progress_cb)
            finally:
                reset_progress_hook(hook_token)
        else:
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
        return _tool_error(msg_id, repr(exc))
    except Exception as exc:  # noqa: BLE001
        _log("error", f"tool '{name}' raised: {exc!r}")
        traceback.print_exc(file=sys.stderr)
        return _tool_error(msg_id, repr(exc))

    attach = tool["attach"]
    text_format = tool.get("text_format")
    if attach is AttachMode.FIGURE:
        result = _figure_only_result(payload, state.conn)
    elif attach is AttachMode.SCAN:
        result = _pack_result(payload, state.conn, text_format=text_format)
    else:
        result = _pack_result(payload, None, text_format=text_format)

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


def _install_signal_handlers(state: _ServerState) -> None:
    def _on_signal(signum: int, frame: Any) -> None:  # noqa: ARG001
        _log("info", f"signal {signum} received; shutting down")
        state.close()
        # Re-raise as SystemExit so the transport loop exits cleanly.
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)


def _run_stdio(state: _ServerState) -> int:
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

    _log("info", "stdin closed; exiting")
    return 0


def _run_http(state: _ServerState, host: str, port: int) -> int:
    """Single-threaded HTTP transport.

    ``POST /mcp`` carries one JSON-RPC message per request. The response
    shape depends on the request's ``Accept`` header (per the MCP
    Streamable HTTP spec):

    - ``application/json`` (default): single 200 JSON response, or 202
      No Content for JSON-RPC notifications. No progress notifications —
      there's no channel to deliver them in plain-JSON mode.
    - ``text/event-stream``: the response is an SSE stream. Each
      ``notifications/progress`` emitted during dispatch becomes one
      ``data: <json>\\n\\n`` frame; the final JSON-RPC reply is emitted
      as the last frame, then the stream is closed. This is how
      non-Claude-Code MCP clients (Cursor, Claude Desktop HTTP, etc.)
      receive in-flight progress for long-running tools like ingest.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 — stdlib API
            if self.path.rstrip("/") != "/mcp":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                self._json(400, _err(None, _ERR_PARSE, f"parse error: {exc}"))
                return
            if not isinstance(msg, dict):
                self._json(400, _err(None, _ERR_INVALID_REQUEST,
                                     "request must be a JSON object"))
                return

            if self._client_accepts_sse():
                self._handle_sse(msg)
                return

            try:
                resp = _dispatch(state, msg)
            except Exception as exc:  # noqa: BLE001
                _log("error", f"dispatch failed: {exc!r}")
                traceback.print_exc(file=sys.stderr)
                if "id" in msg:
                    self._json(200, _err(msg["id"], _ERR_INTERNAL, repr(exc)))
                    return
                self.send_response(202)
                self.end_headers()
                return
            if resp is None:  # notification — no reply per JSON-RPC
                self.send_response(202)
                self.end_headers()
                return
            self._json(200, resp)

        def do_GET(self):  # noqa: N802 — stdlib API
            # Spec-optional SSE-on-GET channel for server-initiated
            # streams; lodestone has no server-initiated traffic.
            self.send_error(405)

        def log_message(self, fmt: str, *args: Any) -> None:
            _log("debug", fmt % args)

        def _client_accepts_sse(self) -> bool:
            accept = self.headers.get("Accept", "") or ""
            # Coarse but spec-compliant — clients that opt into SSE
            # name the type explicitly; a bare ``*/*`` falls back to
            # JSON to preserve backwards compat for older clients.
            return "text/event-stream" in accept.lower()

        def _json(self, code: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle_sse(self, msg: dict) -> None:
            """Run dispatch, streaming progress notifications + the final
            response as Server-Sent Events on the open response stream.
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def _write_frame(payload: dict) -> None:
                try:
                    data = json.dumps(payload, ensure_ascii=False)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # Client disconnected mid-stream. Nothing to do —
                    # subsequent writes will also fail; we let the
                    # dispatch finish (sqlite work is still useful).
                    _log("info", "sse client disconnected mid-stream")

            writer_token = _sse_writer.set(_write_frame)
            try:
                try:
                    resp = _dispatch(state, msg)
                except Exception as exc:  # noqa: BLE001
                    _log("error", f"dispatch failed: {exc!r}")
                    traceback.print_exc(file=sys.stderr)
                    resp = _err(msg.get("id"), _ERR_INTERNAL, repr(exc))
                if resp is not None:
                    _write_frame(resp)
            finally:
                _sse_writer.reset(writer_token)

    srv = HTTPServer((host, port), Handler)
    _log("info", f"listening on http://{host}:{port}/mcp")
    try:
        srv.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        srv.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lodestone-mcp")
    p.add_argument(
        "--http",
        action="store_true",
        help="Serve over HTTP instead of stdio. Workaround for Claude Code "
             "issue #51736 (stdio MCP tools silently dropped on 2.1.116+).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)

    state = _ServerState()
    state.configure()
    state.transport = Transport.HTTP if args.http else Transport.STDIO
    _install_signal_handlers(state)

    _log(
        "info",
        f"starting (db={state.db_path}, transport={state.transport}, "
        f"startup_error={state.startup_error!r})",
    )

    try:
        if args.http:
            return _run_http(state, args.host, args.port)
        return _run_stdio(state)
    finally:
        state.close()


if __name__ == "__main__":
    raise SystemExit(main())
