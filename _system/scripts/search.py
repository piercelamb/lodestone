"""Lodestone search CLI — the single entry point for all retrieval.

Five modes dispatched by argparse:

1. **BM25** — positional ``QUERY`` runs against the ``sections`` FTS5 table
   (paper abstracts ride along as the ``# Abstract`` chunk). Hits are
   grouped by ``paper_name`` and enriched with an entity preview, figure
   count, and topics. User input is sanitized into a quoted-token phrase
   query so punctuation (``-``, ``/``, ``:``, parens, etc.) cannot escape
   into FTS5 operator territory.
2. **Taxonomy lookup** — ``--entity`` / ``--topic`` / bare ``--collection``
   (without a positional query) resolves a term against ``terms_fts`` with
   a vec0 KNN fallback at cosine ≥ 0.80.
3. **Browse** — ``--collections`` / ``--topics`` / ``--entity-type`` /
   ``--aliases`` / ``--needs-review`` pure SQL list queries.
4. **ToC** — ``--toc PAPER`` parses level-1..3 ATX headers from the stored
   markdown (skipping fenced code blocks).
5. **Content extraction** — ``--read PAPER [--section S]`` emits markdown
   (optionally sliced by :func:`find_hierarchical_section`);
   ``--figure PAPER N`` writes the figure BLOB to a
   :func:`tempfile.mkstemp` path and returns the path.

JSON is emitted to stdout by default. ``--human`` renders a short plaintext
per mode. All logging goes to stderr via the shared :mod:`_system.utils.logging`
logger. ``--help`` must finish in under 300 ms — no ML library
(``sentence_transformers`` / ``gliner2`` / ``torch``) is imported anywhere
in this module. Search across all modes is FTS5-only; semantic broadening
lives upstream in the resolver / ingest path.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# NB: import only cheap stdlib + the cheap internal modules here. Anything
# that pulls torch / sentence_transformers / gliner must live inside the
# function that needs it.
from _system.db.connection import PathLike, get_conn, get_readonly_conn
from _system.schemas.repo_metadata import RepoStatus
from _system.scripts.taxonomy_tree import (
    CollectionNode,
    DomainNode,
    TaxonomyTreeStyle,
    load_taxonomy,
    render_taxonomy_tree,
)
from _system.utils.logging import get_logger
from _system.utils.sections import (
    SectionQueryError,
    find_hierarchical_section,
    split_sections,
)
from _system.utils.slug import _SLUG_RE
from _system.utils.source_resolution import SourceKind, lookup_slug

_LOG = get_logger("scripts.search")

# BM25 enrichment size cap. Keeping each follow-up query small bounds the
# JSON payload size even on queries that return many hits.
_ENTITY_PREVIEW_LIMIT = 5


def _read_source_markdown(
    conn: sqlite3.Connection, slug: str,
) -> str | None:
    """Return ``markdown`` for a slug that may live in either ``papers`` or
    ``posts``. Returns None when neither table holds the slug. Empty
    markdown coerces to ''.
    """
    resolved = lookup_slug(conn, slug)
    if resolved is None:
        return None
    table = "papers" if resolved.kind is SourceKind.PAPER else "posts"
    row = conn.execute(
        f"SELECT markdown FROM {table} WHERE id = ?", (resolved.id,),
    ).fetchone()
    if row is None:
        return None
    return row[0] or ""


class TaxonomyKind(StrEnum):
    ENTITY = "entity"
    TOPIC = "topic"
    COLLECTION = "collection"


class BrowseView(StrEnum):
    COLLECTIONS = "collections"
    TOPICS = "topics"
    ENTITY_TYPE = "entity_type"
    ALIASES = "aliases"
    NEEDS_REVIEW = "needs_review"


class Scope(StrEnum):
    SECTIONS = "sections"
    READMES = "readmes"
    BOTH = "both"


# ---------------------------------------------------------------------------
# GitHub-code-search-style query parser
# ---------------------------------------------------------------------------
# Surface: bare tokens (implicit AND, each defanged into a phrase so
# ``-``/``/``/``:`` don't escape into operator syntax), ``"phrase"``,
# ``AND``/``OR``/``NOT`` (uppercase), parens, ``term*`` prefix, and
# ``key:value`` qualifiers (``paper`` / ``domain`` / ``collection`` /
# ``surface`` / ``kind``). ``/regex/`` is rejected — FTS5 is token-based.


# Matches a single Unicode word character — used to drop tokens that hold no
# searchable content (e.g. a bare ``-`` or ``/``) after the per-token quote
# wrap. Without this filter, FTS5 raises a syntax error on the empty phrase.
_FTS5_TOKEN_HAS_WORD_CHAR = re.compile(r"\w", re.UNICODE)

# A qualifier key prefix at the current scan position, e.g. ``paper:`` /
# ``domain:``. Only lowercase + underscore — uppercase ``BAAI/bge-small`` is
# never a qualifier, matching GitHub.
_QUALIFIER_KEY_RE = re.compile(r"[a-z_]+")

_OPERATORS: frozenset[str] = frozenset({"AND", "OR", "NOT"})

_SUPPORTED_QUALIFIERS: frozenset[str] = frozenset(
    {"paper", "domain", "collection", "surface", "kind"}
)

_VALID_KIND_VALUES: frozenset[str] = frozenset({"entity", "topic", "collection"})
_VALID_SURFACE_VALUES: frozenset[str] = frozenset(
    {"sections", "readmes", "both", "taxonomy"}
)


class GitHubQueryError(ValueError):
    """Base for parser-rejected queries — caller surfaces as soft-fail."""


class EmptyQueryError(GitHubQueryError):
    """Query has no searchable tokens (whitespace-only, punctuation-only,
    or qualifiers-only on a surface that requires text)."""


class UnclosedQuoteError(GitHubQueryError):
    """Odd number of unescaped ``"`` characters."""


class UnmatchedParenError(GitHubQueryError):
    """``(`` / ``)`` mismatch."""


class DanglingOperatorError(GitHubQueryError):
    """``AND`` / ``OR`` / ``NOT`` at the start, end, or adjacent to another."""


class RegexNotSupportedError(GitHubQueryError):
    """``/.../`` regex form — not supported."""


class UnknownQualifierError(GitHubQueryError):
    """Qualifier key not in :data:`_SUPPORTED_QUALIFIERS`."""


class InvalidQualifierValueError(GitHubQueryError):
    """``kind:`` or ``surface:`` value outside its allowed set, or a
    qualifier key with no value attached."""


class ConflictingFilterError(GitHubQueryError):
    """Same qualifier supplied twice in one query, or qualifier value
    conflicts with the kwarg-supplied value at the dispatch boundary."""


@dataclass(frozen=True)
class ParsedQuery:
    """Result of :func:`_parse_github_query`.

    ``fts_expression`` is a valid FTS5 MATCH input (or ``""`` if the query
    held only qualifiers). ``qualifiers`` is the extracted ``key:value`` map;
    each key appears at most once (duplicates raise
    :class:`ConflictingFilterError`).
    """

    fts_expression: str
    qualifiers: dict[str, str] = field(default_factory=dict)


def _read_quoted(query: str, i: int) -> tuple[str, int]:
    """Lex a ``"..."`` phrase starting at ``query[i] == '"'``.

    Returns ``(unescaped_phrase, end_index)`` where ``end_index`` is the
    position past the closing quote. Honors ``\\"`` and ``\\\\``. Raises
    :class:`UnclosedQuoteError` if no closing quote is found.
    """
    assert query[i] == '"'
    n = len(query)
    j = i + 1
    buf: list[str] = []
    while j < n:
        c = query[j]
        if c == "\\" and j + 1 < n and query[j + 1] in ('"', "\\"):
            buf.append(query[j + 1])
            j += 2
            continue
        if c == '"':
            return "".join(buf), j + 1
        buf.append(c)
        j += 1
    raise UnclosedQuoteError(
        f"unclosed quote in query: {query!r}"
    )


def _validate_operator_placement(tokens: list[str], *, raw: str) -> None:
    """Reject leading/trailing/adjacent operators and dangling parens.

    Walks tokens with an "expects-operand" / "expects-operator" state
    machine. Operands are everything that isn't ``AND``/``OR``/``NOT`` or a
    paren. ``(`` resets to expects-operand; ``)`` requires expects-operator.
    Paren matching is already validated by :func:`_parse_github_query`.
    """
    expects_operand = True
    saw_operand = False
    for tok in tokens:
        if tok == "(":
            if not expects_operand:
                raise DanglingOperatorError(
                    f"missing operator before '(' in query: {raw!r}"
                )
            expects_operand = True
            continue
        if tok == ")":
            if expects_operand:
                raise DanglingOperatorError(
                    f"empty group or operator before ')' in query: {raw!r}"
                )
            expects_operand = False
            continue
        if tok in _OPERATORS:
            if expects_operand:
                raise DanglingOperatorError(
                    f"operator {tok!r} with no left-hand operand in query: {raw!r}"
                )
            expects_operand = True
            continue
        saw_operand = True
        if not expects_operand:
            continue
        expects_operand = False
    if expects_operand and saw_operand:
        raise DanglingOperatorError(
            f"trailing operator in query: {raw!r}"
        )


def _classify_bare_token(run: str) -> str | None:
    """Convert a bare run into its FTS5-token form.

    Returns ``None`` to signal the run held no word characters and should
    be dropped (matches the legacy ``_sanitize_fts5_match`` defang
    behavior for stray punctuation like ``---``).
    """
    if not _FTS5_TOKEN_HAS_WORD_CHAR.search(run):
        return None
    if len(run) > 1 and run.endswith("*") and _FTS5_TOKEN_HAS_WORD_CHAR.search(run[:-1]):
        stem = run[:-1]
        return '"' + stem.replace('"', '""') + '"*'
    return '"' + run.replace('"', '""') + '"'


def _collect_qualifier(
    qualifiers: dict[str, str], key: str, value: str
) -> None:
    """Validate ``key`` / ``value`` and store. Raises typed errors on the
    fast path so the dispatch layer can convert them to soft-fails."""
    if key not in _SUPPORTED_QUALIFIERS:
        raise UnknownQualifierError(
            f"unknown qualifier {key!r}; supported: "
            f"{', '.join(sorted(_SUPPORTED_QUALIFIERS))}"
        )
    if not value:
        raise InvalidQualifierValueError(
            f"qualifier {key!r} has no value"
        )
    if key == "kind" and value not in _VALID_KIND_VALUES:
        raise InvalidQualifierValueError(
            f"kind:{value!r} not allowed; expected one of "
            f"{sorted(_VALID_KIND_VALUES)}"
        )
    if key == "surface" and value not in _VALID_SURFACE_VALUES:
        raise InvalidQualifierValueError(
            f"surface:{value!r} not allowed; expected one of "
            f"{sorted(_VALID_SURFACE_VALUES)}"
        )
    if key in qualifiers and qualifiers[key] != value:
        raise ConflictingFilterError(
            f"qualifier {key!r} given twice with conflicting values: "
            f"{qualifiers[key]!r} vs {value!r}"
        )
    qualifiers[key] = value


def _parse_github_query(query: str) -> ParsedQuery:
    """Parse a GitHub-code-search-style query into an FTS5 expression +
    qualifier map.

    Single-pass character scanner. Recognized tokens (in priority order):

      1. whitespace — separator
      2. ``(`` / ``)`` — paren tokens
      3. ``"..."`` — exact phrase (escapes ``\\"`` ``\\\\``)
      4. ``key:value`` or ``key:"quoted"`` — qualifier (key matches ``[a-z_]+``)
      5. ``AND`` / ``OR`` / ``NOT`` exact bare runs — operator passthrough
      6. ``/regex/`` bare runs — rejected (out of scope for FTS5)
      7. anything else — bare token; defanged into a phrase, with a
         trailing ``*`` lifted as an FTS5 prefix marker
    """
    if not query or not query.strip():
        raise EmptyQueryError(f"empty query: {query!r}")

    tokens: list[str] = []
    qualifiers: dict[str, str] = {}

    n = len(query)
    i = 0
    paren_depth = 0

    while i < n:
        c = query[i]
        if c.isspace():
            i += 1
            continue

        if c == "(":
            tokens.append("(")
            paren_depth += 1
            i += 1
            continue

        if c == ")":
            if paren_depth == 0:
                raise UnmatchedParenError(
                    f"unmatched ')' in query: {query!r}"
                )
            tokens.append(")")
            paren_depth -= 1
            i += 1
            continue

        if c == '"':
            phrase, end = _read_quoted(query, i)
            if not phrase:
                # `""` empty phrase — drop, no FTS token
                i = end
                continue
            tokens.append('"' + phrase.replace('"', '""') + '"')
            i = end
            continue

        # Qualifier? key matches [a-z_]+ followed immediately by ':'.
        m = _QUALIFIER_KEY_RE.match(query, i)
        if m and m.end() < n and query[m.end()] == ":":
            key = m.group(0)
            j = m.end() + 1  # past ':'
            if j >= n or query[j].isspace():
                raise InvalidQualifierValueError(
                    f"qualifier {key!r} has no value in query: {query!r}"
                )
            if query[j] == '"':
                value, j = _read_quoted(query, j)
            else:
                start = j
                while j < n and not query[j].isspace() and query[j] not in '()"':
                    j += 1
                value = query[start:j]
            _collect_qualifier(qualifiers, key, value)
            i = j
            continue

        # Bare run — until whitespace, paren, or quote.
        start = i
        while i < n and not query[i].isspace() and query[i] not in '()"':
            i += 1
        run = query[start:i]

        if run in _OPERATORS:
            tokens.append(run)
            continue

        if len(run) >= 2 and run.startswith("/") and run.endswith("/"):
            raise RegexNotSupportedError(
                f"regex {run!r} not supported — FTS5 is token-based. "
                f"Use prefix ('term*'), phrase ('\"...\"'), or boolean "
                f"operators (AND/OR/NOT) instead."
            )

        emitted = _classify_bare_token(run)
        if emitted is not None:
            tokens.append(emitted)

    if paren_depth != 0:
        raise UnmatchedParenError(
            f"unmatched '(' in query: {query!r}"
        )

    _validate_operator_placement(tokens, raw=query)

    fts_expression = _join_tokens_for_fts5(tokens)
    return ParsedQuery(
        fts_expression=fts_expression,
        qualifiers=qualifiers,
    )


def _join_tokens_for_fts5(tokens: list[str]) -> str:
    """Reassemble parsed tokens into a valid FTS5 MATCH expression.

    FTS5 accepts juxtaposition as implicit AND for *most* token sequences,
    but the grammar refuses to parse a paren-group immediately followed by
    a phrase or prefix marker — ``( "a" ) "b"*`` raises ``fts5: syntax
    error near ""b""`` even though the same expression with explicit
    ``AND`` works. To stay compatible across operator/group/prefix
    combinations we emit ``AND`` between every pair of adjacent operands
    (an "operand" being either a phrase token or a closing paren). This
    is functionally identical to implicit AND for FTS5 but parses cleanly
    in every case.
    """
    out: list[str] = []
    for tok in tokens:
        if out:
            prev = out[-1]
            prev_ends_operand = prev not in _OPERATORS and prev != "("
            next_starts_operand = tok not in _OPERATORS and tok != ")"
            if prev_ends_operand and next_starts_operand:
                out.append("AND")
        out.append(tok)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Mode 1 — BM25
# ---------------------------------------------------------------------------


_BM25_SYNTAX_HINT = (
    "Supported syntax: bare words (implicit AND, defanged), \"phrase\", "
    "AND/OR/NOT (uppercase), parens, term*, and qualifiers "
    "paper:NAME / domain:X / collection:NAME / surface:sections|readmes|both."
)


_EMPTY_QUERY_HINT = (
    "Query had no searchable tokens after parsing. Pass at least "
    "one word; queries that contain only qualifiers (e.g. "
    "'paper:foo') match nothing on this surface."
)


_INVALID_PAGINATION_HINT = (
    "Pagination uses non-negative integers: offset >= 0, limit >= 1. "
    "Page forward by raising `offset` by `limit` once `has_more` "
    "is true; the response carries `total_hits` and `offset`."
)


_SOFT_FAIL_HINTS: dict[str, str] = {
    "empty_query": _EMPTY_QUERY_HINT,
    "invalid_pagination": _INVALID_PAGINATION_HINT,
}


def _soft_fail_payload(
    *, mode: str, status: str, query: str, error: str, extra_hint: str = ""
) -> dict[str, Any]:
    hint = _SOFT_FAIL_HINTS.get(status, _BM25_SYNTAX_HINT)
    if extra_hint:
        hint = f"{extra_hint} {hint}"
    return {
        "mode": mode,
        "status": status,
        "query": query,
        "error": error,
        "hint": hint,
    }


def _malformed_query_payload(
    *, mode: str, query: str, error: str, extra_hint: str = ""
) -> dict[str, Any]:
    return _soft_fail_payload(
        mode=mode, status="malformed_query", query=query,
        error=error, extra_hint=extra_hint,
    )


def _empty_query_payload(*, mode: str, query: str, error: str) -> dict[str, Any]:
    return _soft_fail_payload(
        mode=mode, status="empty_query", query=query, error=error,
    )


def _invalid_pagination_payload(
    *, mode: str, query: str, offset: int, limit: int, error: str,
) -> dict[str, Any]:
    """Soft-fail envelope for negative offset / limit values.

    Echoes the bad params for debug; pagination response keys
    (``total_hits`` / ``has_more``) are intentionally absent — the
    request never executed.
    """
    payload = _soft_fail_payload(
        mode=mode, status="invalid_pagination", query=query, error=error,
    )
    payload["offset"] = offset
    payload["limit"] = limit
    return payload


def _merge_qualifier_with_kwarg(
    qualifiers: dict[str, str], kwarg_value: Any, key: str
) -> Any:
    """Intersect a parsed qualifier with the corresponding kwarg-supplied
    value. Returns the unified value or raises :class:`ConflictingFilterError`.
    """
    qual = qualifiers.get(key)
    if qual is None:
        return kwarg_value
    if kwarg_value in (None, "") or kwarg_value == qual:
        return qual
    raise ConflictingFilterError(
        f"qualifier {key}:{qual!r} conflicts with kwarg {key}={kwarg_value!r}"
    )


def mode_bm25(
    conn: sqlite3.Connection,
    *,
    query: str,
    filters: dict[str, Any],
    limit: int,
    offset: int = 0,
    scope: Scope = Scope.SECTIONS,
    snippet_tokens: int = 256,
) -> dict[str, Any]:
    """BM25 text search across ``sections`` and/or ``readmes_fts``.

    ``query`` is a GitHub-code-search-style string parsed by
    :func:`_parse_github_query`: bare tokens (implicit AND), ``"phrase"``,
    ``AND``/``OR``/``NOT``, parens, prefix ``term*``, and qualifiers
    ``paper:`` / ``domain:`` / ``collection:`` / ``surface:``. The
    ``kind:`` qualifier is rejected here (only meaningful for the search
    tool's taxonomy bucket).

    ``scope`` selects the default surface(s): SECTIONS (current
    behavior), READMES, or BOTH. ``surface:`` qualifier wins (and
    must agree if both are non-default).

    ``snippet_tokens`` controls FTS5 ``snippet()`` window width.

    ``offset`` (default 0) skips that many ranked hits before returning;
    the response carries ``total_hits`` and ``has_more`` so callers can
    walk forward by raising ``offset`` by ``limit``. For ``scope=BOTH``
    each surface paginates independently with the same offset/limit;
    ``total_hits`` is the cross-surface sum.

    Soft failures (no exception raised; soft-status payload):

    * ``empty_query`` — punctuation-only or qualifier-only query
    * ``malformed_query`` — quote/paren/operator mismatch, unknown
      qualifier, ``/regex/`` form, or qualifier↔kwarg conflict
    * ``invalid_pagination`` — ``offset`` < 0
    """
    if offset < 0:
        return _invalid_pagination_payload(
            mode="sections", query=query, offset=offset, limit=limit,
            error="offset must be >= 0",
        )

    try:
        parsed = _parse_github_query(query)
    except EmptyQueryError as e:
        return _empty_query_payload(mode="sections", query=query, error=str(e))
    except GitHubQueryError as e:
        return _malformed_query_payload(
            mode="sections", query=query, error=str(e),
        )

    if "kind" in parsed.qualifiers:
        return _malformed_query_payload(
            mode="sections", query=query,
            error="kind: qualifier is only valid on the 'search' tool's "
                  "taxonomy bucket; bm25 hits sections/readmes which have "
                  "no term_type column.",
        )

    try:
        domain = _merge_qualifier_with_kwarg(
            parsed.qualifiers, filters.get("domain"), "domain"
        )
        collection = _merge_qualifier_with_kwarg(
            parsed.qualifiers, filters.get("collection"), "collection"
        )
    except ConflictingFilterError as e:
        return _malformed_query_payload(
            mode="sections", query=query, error=str(e),
        )

    paper_name = parsed.qualifiers.get("paper")

    surface_qual = parsed.qualifiers.get("surface")
    if surface_qual is not None:
        if surface_qual == "taxonomy":
            return _malformed_query_payload(
                mode="sections", query=query,
                error="surface:taxonomy is only valid on the 'search' tool; "
                      "bm25 has no taxonomy surface (use the 'search' tool "
                      "or the 'lookup' tool for canonical terms).",
            )
        try:
            scope_from_qual = Scope(surface_qual)
        except ValueError:
            return _malformed_query_payload(
                mode="sections", query=query,
                error=f"surface:{surface_qual!r} not a valid scope",
            )
        if scope is not Scope.SECTIONS and scope is not scope_from_qual:
            return _malformed_query_payload(
                mode="sections", query=query,
                error=f"surface:{surface_qual} conflicts with scope={scope.value}",
            )
        scope = scope_from_qual

    if not parsed.fts_expression:
        return _empty_query_payload(
            mode="sections", query=query,
            error="query held only qualifiers, no FTS body",
        )

    if scope is Scope.SECTIONS:
        result = _bm25_sections(
            conn, fts_expression=parsed.fts_expression,
            domain=domain, collection=collection, paper_name=paper_name,
            limit=limit, offset=offset, snippet_tokens=snippet_tokens,
        )
    elif scope is Scope.READMES:
        result = _bm25_readmes(
            conn, fts_expression=parsed.fts_expression,
            domain=domain, collection=collection, paper_name=paper_name,
            limit=limit, offset=offset, snippet_tokens=snippet_tokens,
        )
    else:
        result = _bm25_both(
            conn, fts_expression=parsed.fts_expression,
            domain=domain, collection=collection, paper_name=paper_name,
            limit=limit, offset=offset, snippet_tokens=snippet_tokens,
        )
    result["query"] = query
    return result


def _bm25_sections(
    conn: sqlite3.Connection,
    *,
    fts_expression: str,
    domain: str | None,
    collection: str | None,
    paper_name: str | None,
    limit: int,
    offset: int = 0,
    snippet_tokens: int = 10,
    enrich: bool = True,
) -> dict[str, Any]:
    # sections columns: (paper_id, domain, paper_name, section_title, section_level, body)
    # snippet() against 'body' = column index 5.
    # Breadcrumb is prepended to body at index time as `breadcrumb\n\n<raw>`,
    # so first line of body == breadcrumb. We extract it explicitly so the
    # caller doesn't depend on whether snippet()'s token window happened to
    # land near the start of body.
    join_sql = ""
    wheres = ["sections MATCH ?"]
    where_params: list[Any] = [fts_expression]
    if domain:
        wheres.append("s.domain = ?")
        where_params.append(domain)
    if paper_name:
        wheres.append("s.paper_name = ?")
        where_params.append(paper_name)
    if collection:
        # Sections does not carry collection in FTS5 (by design; the
        # source owns the collection). Match against `paper_collections`
        # / `post_collections` via the slug — sections.paper_id is
        # ambiguous across kinds (a post_id may equal a paper_id), but
        # the slug is globally unique. Secondary memberships filter
        # through too.
        wheres.append(
            "s.paper_name IN ("
            "  SELECT p.paper_name FROM papers p "
            "    JOIN paper_collections pc ON pc.paper_id = p.id "
            "   WHERE pc.collection = ? "
            "  UNION ALL "
            "  SELECT po.post_name FROM posts po "
            "    JOIN post_collections poc ON poc.post_id = po.id "
            "   WHERE poc.collection = ? "
            ")"
        )
        where_params.append(collection)
        where_params.append(collection)
    where_clause = " WHERE " + " AND ".join(wheres)

    sql = (
        "SELECT s.paper_id, s.domain, s.paper_name, s.section_title, "
        "       s.section_level, "
        f"       snippet(sections, 5, '[', ']', '…', {int(snippet_tokens)}) AS snip, "
        "       substr(s.body, 1, instr(s.body || char(10), char(10)) - 1) AS breadcrumb "
        "  FROM sections s"
        + join_sql
        + where_clause
        + " ORDER BY rank LIMIT ? OFFSET ?"
    )
    rows = conn.execute(sql, [*where_params, limit, offset]).fetchall()
    total_hits = _bm25_total_hits(
        conn,
        count_sql=(
            "SELECT COUNT(*) FROM sections s" + join_sql + where_clause
        ),
        count_params=where_params,
        offset=offset,
        limit=limit,
        page_size=len(rows),
    )

    # One bucket walk to build groups; enrichment then batched across all
    # distinct paper_ids in three queries regardless of result count.
    grouped: dict[str, dict[str, Any]] = {}
    pid_order: list[int] = []
    for paper_id, dom, paper_name, section_title, section_level, snip, breadcrumb in rows:
        group = grouped.get(paper_name)
        if group is None:
            group = {
                "paper_name": paper_name,
                "domain": dom,
                "_paper_id": paper_id,
                "hit_count": 0,
                "sections": [],
            }
            grouped[paper_name] = group
            pid_order.append(paper_id)
        group["hit_count"] += 1
        group["sections"].append({
            "section_title": section_title,
            "section_level": section_level,
            "breadcrumb": breadcrumb,
            "snippet": snip,
        })

    if enrich:
        _attach_bm25_enrichment(conn, grouped, pid_order)
    else:
        for group in grouped.values():
            group.pop("_paper_id", None)

    return {
        "mode": "sections",
        "results": list(grouped.values()),
        "offset": offset,
        "limit": limit,
        "total_hits": total_hits,
        "has_more": offset + len(rows) < total_hits,
    }


def _bm25_readmes(
    conn: sqlite3.Connection,
    *,
    fts_expression: str,
    domain: str | None,
    collection: str | None,
    paper_name: str | None,
    limit: int,
    offset: int = 0,
    snippet_tokens: int = 10,
    enrich: bool = True,
) -> dict[str, Any]:
    """BM25 against ``readmes_fts``. Each result is keyed by ``repo_slug``
    (the repo is the searchable unit). When the repo is paper-linked the
    envelope also carries ``paper_name`` + paper title so callers can
    pivot back to the prose surfaces."""
    join_sql = " JOIN repos rr ON rr.id = r.repo_id"
    wheres = ["readmes_fts MATCH ?"]
    where_params: list[Any] = [fts_expression]
    if domain:
        wheres.append("r.domain = ?")
        where_params.append(domain)
    if paper_name:
        # Restrict to a paper-linked repo for this paper.
        join_sql += " JOIN papers pp ON pp.id = rr.paper_id"
        wheres.append("pp.paper_name = ?")
        where_params.append(paper_name)
    if collection:
        wheres.append("rr.collection = ?")
        where_params.append(collection)
    where_clause = " WHERE " + " AND ".join(wheres)

    sql = (
        "SELECT r.repo_id, r.repo_slug, r.domain, r.path, "
        f"       snippet(readmes_fts, 4, '[', ']', '…', {int(snippet_tokens)}) AS snip "
        "  FROM readmes_fts r"
        + join_sql
        + where_clause
        + " ORDER BY rank LIMIT ? OFFSET ?"
    )

    rows = conn.execute(sql, [*where_params, limit, offset]).fetchall()
    total_hits = _bm25_total_hits(
        conn,
        count_sql=(
            "SELECT COUNT(*) FROM readmes_fts r" + join_sql + where_clause
        ),
        count_params=where_params,
        offset=offset,
        limit=limit,
        page_size=len(rows),
    )

    grouped: dict[str, dict[str, Any]] = {}
    repo_id_order: list[int] = []
    for repo_id, repo_slug, dom, path, snip in rows:
        group = grouped.get(repo_slug)
        if group is None:
            group = {
                "repo_slug": repo_slug,
                "domain": dom,
                "_repo_id": repo_id,
                "hit_count": 0,
                "sections": [],
                "readme_hit": None,
            }
            grouped[repo_slug] = group
            repo_id_order.append(repo_id)
        group["hit_count"] += 1
        group["readme_hit"] = {"path": path, "snippet": snip}

    if enrich:
        _attach_repo_bm25_enrichment(conn, grouped, repo_id_order)
    else:
        for group in grouped.values():
            group.pop("_repo_id", None)

    return {
        "mode": "sections",
        "scope": Scope.READMES.value,
        "results": list(grouped.values()),
        "offset": offset,
        "limit": limit,
        "total_hits": total_hits,
        "has_more": offset + len(rows) < total_hits,
    }


def _attach_repo_bm25_enrichment(
    conn: sqlite3.Connection,
    grouped: dict[str, dict[str, Any]],
    repo_id_order: list[int],
) -> None:
    """Attach repo / paper-link envelopes to each readme BM25 hit."""
    if not repo_id_order:
        for group in grouped.values():
            group.pop("_repo_id", None)
        return

    placeholders = ",".join("?" * len(repo_id_order))
    rows = conn.execute(
        f"""
        SELECT r.id, r.repo_slug, r.url, r.status, r.paper_id,
               r.file_count, r.has_readme, r.domain, r.collection,
               p.paper_name, p.title
          FROM repos r
          LEFT JOIN papers p ON p.id = r.paper_id
         WHERE r.id IN ({placeholders})
        """,
        repo_id_order,
    ).fetchall()
    by_id: dict[int, dict[str, Any]] = {}
    for (
        rid, slug, url, status, paper_id, file_count, has_readme,
        dom, coll, paper_name, paper_title,
    ) in rows:
        by_id[int(rid)] = {
            "repo_slug": slug,
            "url": url,
            "status": status,
            "paper_id": paper_id,
            "file_count": int(file_count or 0),
            "has_readme": bool(has_readme),
            "domain": dom,
            "collection": coll,
            "paper_name": paper_name,
            "paper_title": paper_title,
        }

    repo_topics = _topics_batch_repo(conn, repo_id_order)

    for group in grouped.values():
        rid = group.pop("_repo_id")
        info = by_id.get(rid)
        if info is None:
            continue
        group["url"] = info["url"]
        group["repo_status"] = info["status"]
        group["file_count"] = info["file_count"]
        group["has_readme"] = info["has_readme"]
        group["collection"] = info["collection"]
        if info["paper_name"]:
            group["paper_name"] = info["paper_name"]
            group["paper_title"] = info["paper_title"]
        group["topics"] = repo_topics.get(rid, [])


def _bm25_both(
    conn: sqlite3.Connection,
    *,
    fts_expression: str,
    domain: str | None,
    collection: str | None,
    paper_name: str | None,
    limit: int,
    offset: int = 0,
    snippet_tokens: int = 10,
    enrich: bool = True,
) -> dict[str, Any]:
    """Union of sections + READMES hits.

    Sections hits are paper-keyed; readme hits are repo-keyed. The two
    are surfaced as separate buckets (``results`` for sections,
    ``repo_results`` for readmes) since they describe different first-class
    entities now. Each surface paginates independently with the same
    ``offset`` / ``limit``; ``total_hits`` is the cross-surface sum and
    ``has_more`` is true when either surface still has unread rows.
    """
    sec = _bm25_sections(
        conn, fts_expression=fts_expression,
        domain=domain, collection=collection, paper_name=paper_name,
        limit=limit, offset=offset,
        snippet_tokens=snippet_tokens, enrich=enrich,
    )
    rdm = _bm25_readmes(
        conn, fts_expression=fts_expression,
        domain=domain, collection=collection, paper_name=paper_name,
        limit=limit, offset=offset,
        snippet_tokens=snippet_tokens, enrich=enrich,
    )

    total_hits = int(sec.get("total_hits", 0)) + int(rdm.get("total_hits", 0))
    has_more = bool(sec.get("has_more")) or bool(rdm.get("has_more"))
    return {
        "mode": "sections",
        "scope": Scope.BOTH.value,
        "results": list(sec.get("results", [])),
        "repo_results": list(rdm.get("results", [])),
        "offset": offset,
        "limit": limit,
        "total_hits": total_hits,
        "has_more": has_more,
    }


def _bm25_total_hits(
    conn: sqlite3.Connection,
    *,
    count_sql: str,
    count_params: list[Any],
    offset: int,
    limit: int,
    page_size: int,
) -> int:
    """Return the total matching-row count for a BM25 query.

    Optimization: when we asked for the first page (``offset == 0``)
    and the page came back smaller than ``limit``, we already know the
    total is exactly ``page_size`` — no COUNT(*) needed. This saves a
    query on the typical "narrow query, fits in one page" case.
    """
    if offset == 0 and page_size < limit:
        return page_size
    row = conn.execute(count_sql, count_params).fetchone()
    return int(row[0]) if row else 0


def _attach_bm25_enrichment(
    conn: sqlite3.Connection,
    grouped: dict[str, dict[str, Any]],
    pid_order: list[int],
) -> None:
    """Run the four enrichment SELECTs and stamp results onto each group.

    Pops the private ``_paper_id`` field from each group as a side effect.
    Used by ``_bm25_sections`` and ``_bm25_readmes`` for the full envelope.
    Skipped when ``enrich=False`` (the slim ``mode_search`` path).
    """
    topics_by_pid = _topics_batch(conn, pid_order)
    entities_by_pid = _entities_preview_batch(conn, pid_order)
    figures_by_pid = _figures_preview_batch(conn, pid_order)
    code_repo_by_pid = _code_repo_envelope_batch(conn, pid_order)
    for group in grouped.values():
        pid = group.pop("_paper_id")
        group["topics"] = topics_by_pid.get(pid, [])
        group["entities_preview"] = entities_by_pid.get(pid, [])
        group["figures"] = figures_by_pid.get(
            pid, {"count": 0, "first_caption": None}
        )
        group["code_repo"] = code_repo_by_pid.get(pid)


def _code_repo_envelope_batch(
    conn: sqlite3.Connection, paper_ids: list[int]
) -> dict[int, dict[str, Any] | None]:
    """One small SELECT per BM25 hit batch — never a fan-out per result.

    Joins ``repos`` to derive the linked repo (if any) for each paper.
    """
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"""
        SELECT r.paper_id, r.repo_slug, r.url, r.status, r.file_count
          FROM repos r
         WHERE r.paper_id IN ({placeholders})
        """,
        paper_ids,
    ).fetchall()
    result: dict[int, dict[str, Any] | None] = {pid: None for pid in paper_ids}
    for pid, repo_slug, url, status, file_count in rows:
        result[int(pid)] = {
            "repo_slug": repo_slug,
            "url": url,
            "status": status,
            "file_count": int(file_count or 0),
        }
    return result


def _topics_batch(
    conn: sqlite3.Connection, paper_ids: list[int]
) -> dict[int, list[str]]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"SELECT target_id, topic FROM topics "
        f" WHERE target_kind = 'paper' AND target_id IN ({placeholders}) "
        f" ORDER BY target_id, topic",
        paper_ids,
    ).fetchall()
    result: dict[int, list[str]] = {pid: [] for pid in paper_ids}
    for pid, topic in rows:
        result[pid].append(topic)
    return result


def _topics_batch_repo(
    conn: sqlite3.Connection, repo_ids: list[int]
) -> dict[int, list[str]]:
    """Sibling of ``_topics_batch`` for ``target_kind='repo'`` rows."""
    if not repo_ids:
        return {}
    placeholders = ",".join("?" * len(repo_ids))
    rows = conn.execute(
        f"SELECT target_id, topic FROM topics "
        f" WHERE target_kind = 'repo' AND target_id IN ({placeholders}) "
        f" ORDER BY target_id, topic",
        repo_ids,
    ).fetchall()
    result: dict[int, list[str]] = {rid: [] for rid in repo_ids}
    for rid, topic in rows:
        result[rid].append(topic)
    return result


def _topics_batch_post(
    conn: sqlite3.Connection, post_ids: list[int]
) -> dict[int, list[str]]:
    """Sibling of ``_topics_batch`` for ``target_kind='post'`` rows."""
    if not post_ids:
        return {}
    placeholders = ",".join("?" * len(post_ids))
    rows = conn.execute(
        f"SELECT target_id, topic FROM topics "
        f" WHERE target_kind = 'post' AND target_id IN ({placeholders}) "
        f" ORDER BY target_id, topic",
        post_ids,
    ).fetchall()
    result: dict[int, list[str]] = {pid: [] for pid in post_ids}
    for pid, topic in rows:
        result[pid].append(topic)
    return result


def _entities_preview_batch(
    conn: sqlite3.Connection, paper_ids: list[int]
) -> dict[int, list[dict[str, str]]]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    # term_aliases is a synonym index keyed by paper_name; join through
    # papers to map back to paper_id, then ROW_NUMBER caps the per-paper
    # preview at _ENTITY_PREVIEW_LIMIT in a single pass. Caveat: the
    # preview shows entities whose synonyms appeared in this paper, so
    # canonicals that only ever surface as their canonical name (no
    # synonym row) are missed. The BM25 hit's snippet is the
    # authoritative signal — this preview is just a hint.
    rows = conn.execute(
        f"""
        SELECT paper_id, canonical_name, entity_type FROM (
            SELECT paper_id, canonical_name, entity_type,
                   ROW_NUMBER() OVER (
                       PARTITION BY paper_id
                       ORDER BY entity_type, canonical_name
                   ) AS rn
              FROM (
                  SELECT p.id AS paper_id,
                         ct.canonical_name,
                         ct.entity_type
                    FROM papers p
                    JOIN term_aliases ta ON ta.source_paper = p.paper_name
                    JOIN canonical_terms ct ON ct.id = ta.term_id
                   WHERE p.id IN ({placeholders})
                     AND ct.term_type = 'entity'
                   GROUP BY p.id, ct.id
              )
        )
         WHERE rn <= ?
         ORDER BY paper_id, entity_type, canonical_name
        """,
        (*paper_ids, _ENTITY_PREVIEW_LIMIT),
    ).fetchall()
    result: dict[int, list[dict[str, str]]] = {pid: [] for pid in paper_ids}
    for pid, name, etype in rows:
        result[pid].append({"name": name, "type": etype})
    return result


def _figures_preview_batch(
    conn: sqlite3.Connection, paper_ids: list[int]
) -> dict[int, dict[str, Any]]:
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"""
        SELECT paper_id, cnt, caption FROM (
            SELECT paper_id,
                   COUNT(*) OVER (PARTITION BY paper_id) AS cnt,
                   caption,
                   ROW_NUMBER() OVER (
                       PARTITION BY paper_id ORDER BY figure_number
                   ) AS rn
              FROM figures
             WHERE paper_id IN ({placeholders})
        )
         WHERE rn = 1
        """,
        paper_ids,
    ).fetchall()
    result: dict[int, dict[str, Any]] = {
        pid: {"count": 0, "first_caption": None} for pid in paper_ids
    }
    for pid, cnt, caption in rows:
        result[pid] = {"count": int(cnt or 0), "first_caption": caption}
    return result


# ---------------------------------------------------------------------------
# Mode 2 — Taxonomy lookup
# ---------------------------------------------------------------------------


def mode_taxonomy_lookup(
    conn: sqlite3.Connection,
    *,
    query: str,
    filters: dict[str, Any],
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Canonical-term FTS5 search with aliases inlined per hit.

    ``query`` is a GitHub-code-search-style string parsed by
    :func:`_parse_github_query`: bare tokens (implicit AND), ``"phrase"``,
    ``AND``/``OR``/``NOT``, parens, prefix ``term*``. Two qualifiers are
    honored: ``kind:entity|topic|collection`` (narrows the term_type
    bucket) and ``domain:NAME``. ``paper:`` / ``collection:`` /
    ``surface:`` are rejected — they have no meaning against canonical
    terms.

    Returns up to ``limit`` ranked hits starting at ``offset`` (default
    0). The response carries ``total_hits`` and ``has_more`` so callers
    can walk forward by raising ``offset`` by ``limit``.

    Each hit carries the canonical metadata, every alias for the term
    (with its source paper), and the list of papers that mention it
    (papers per kind: aliases-source for entities, paper_topics for
    topics, papers.collection for collections).

    No KNN / vector fallback — this is the precise canonical-search path.
    Semantic broadening lives in the resolver / ingest pipeline; the
    ``search`` tool's taxonomy bucket gives a wider FTS sweep at lower
    detail.

    Soft failures (no exception raised; soft-status payload):

    * ``empty_query`` — punctuation-only or qualifier-only query
    * ``malformed_query`` — quote/paren/operator mismatch, unknown
      qualifier (incl. ``paper:`` / ``collection:`` / ``surface:``),
      ``kind:`` value outside the allowed set, or qualifier↔kwarg
      conflict
    * ``invalid_pagination`` — ``offset`` < 0
    """
    if offset < 0:
        return _invalid_pagination_payload(
            mode="lookup", query=query, offset=offset, limit=limit,
            error="offset must be >= 0",
        )

    if not query or not query.strip():
        return _empty_query_payload(
            mode="lookup", query=query, error="empty query",
        )

    try:
        parsed = _parse_github_query(query)
    except EmptyQueryError as e:
        return _empty_query_payload(mode="lookup", query=query, error=str(e))
    except GitHubQueryError as e:
        return _malformed_query_payload(
            mode="lookup", query=query, error=str(e),
        )

    for bad in ("paper", "collection", "surface"):
        if bad in parsed.qualifiers:
            return _malformed_query_payload(
                mode="lookup", query=query,
                error=f"{bad}: qualifier is not valid on lookup; "
                      f"supported qualifiers are kind:, domain:.",
            )

    try:
        domain = _merge_qualifier_with_kwarg(
            parsed.qualifiers, filters.get("domain"), "domain"
        )
    except ConflictingFilterError as e:
        return _malformed_query_payload(
            mode="lookup", query=query, error=str(e),
        )

    kind_filter = parsed.qualifiers.get("kind")

    if not parsed.fts_expression:
        return _empty_query_payload(
            mode="lookup", query=query,
            error="query held only qualifiers, no FTS body",
        )

    where_sql = " WHERE terms_fts MATCH ? "
    where_params: list[Any] = [parsed.fts_expression]
    if kind_filter:
        where_sql += " AND term_type = ? "
        where_params.append(kind_filter)
    if domain:
        where_sql += " AND domain = ? "
        where_params.append(domain)

    sql = (
        "SELECT term_id, domain, term_type, entity_type, canonical_name "
        "  FROM terms_fts "
        + where_sql
        + " ORDER BY rank LIMIT ? OFFSET ?"
    )

    try:
        rows = conn.execute(
            sql, [*where_params, limit, offset]
        ).fetchall()
    except sqlite3.OperationalError as exc:
        # Degenerate MATCH (e.g. all-punctuation after defang) raises
        # "fts5: syntax error near ..."; treat as zero hits. Anything
        # else (missing table, IO) is a real failure.
        if "fts5" not in str(exc).lower():
            raise
        _LOG.warning(
            "terms_fts MATCH syntax error for query=%r: %s", query, exc,
        )
        rows = []
        total_hits = 0
    else:
        total_hits = _bm25_total_hits(
            conn,
            count_sql="SELECT COUNT(*) FROM terms_fts " + where_sql,
            count_params=where_params,
            offset=offset,
            limit=limit,
            page_size=len(rows),
        )

    hits: list[dict[str, Any]] = []
    seen: set[int] = set()
    for term_id, dom, term_type, entity_type, canonical_name in rows:
        if term_id in seen:
            continue
        seen.add(term_id)

        aliases_rows = conn.execute(
            "SELECT DISTINCT alias, source_paper FROM term_aliases "
            " WHERE term_id = ? ORDER BY alias, source_paper",
            (term_id,),
        ).fetchall()
        aliases = [
            {"alias": a, "source_paper": s} for a, s in aliases_rows
        ]

        if term_type == TaxonomyKind.ENTITY.value:
            prows = conn.execute(
                "SELECT DISTINCT source_paper FROM term_aliases "
                " WHERE term_id = ? ORDER BY source_paper",
                (term_id,),
            ).fetchall()
        elif term_type == TaxonomyKind.TOPIC.value:
            prows = conn.execute(
                "SELECT DISTINCT p.paper_name "
                "  FROM topics t "
                "  JOIN papers p ON p.id = t.target_id "
                " WHERE t.target_kind = 'paper' "
                "   AND t.topic = ? AND t.domain = ? "
                "UNION ALL "
                "SELECT DISTINCT po.post_name "
                "  FROM topics t "
                "  JOIN posts po ON po.id = t.target_id "
                " WHERE t.target_kind = 'post' "
                "   AND t.topic = ? AND t.domain = ? "
                " ORDER BY 1",
                (canonical_name, dom, canonical_name, dom),
            ).fetchall()
        else:  # collection
            # Slug-namespace union: collections can contain both papers
            # and posts. The lookup payload renames the column to
            # ``paper_name`` for backward compat — callers see slugs.
            prows = conn.execute(
                "SELECT paper_name FROM papers "
                " WHERE collection = ? AND domain = ? "
                "UNION ALL "
                "SELECT post_name FROM posts "
                " WHERE collection = ? AND domain = ? "
                "ORDER BY 1",
                (canonical_name, dom, canonical_name, dom),
            ).fetchall()
        papers = [{"paper_name": r[0]} for r in prows]
        _attach_code_repo_to_papers(conn, papers)

        type_label = (
            entity_type
            if term_type == TaxonomyKind.ENTITY.value
            else term_type
        )

        hit_payload: dict[str, Any] = {
            "canonical_name": canonical_name,
            "kind": term_type,
            "type": type_label,
            "entity_type": entity_type,
            "domain": dom,
            "aliases": aliases,
            "papers": papers,
        }
        # papers_count is only honest for topics/collections, whose
        # papers list is derived from a complete per-paper binding
        # (paper_topics / papers.collection). Entity papers come from
        # term_aliases — a synonym index that misses tier-1 mentions
        # written under the canonical surface form. Publishing a count
        # there would lie, so we omit the field for entities; callers
        # who want a lower bound can len(papers) themselves and treat
        # it as such.
        if term_type != TaxonomyKind.ENTITY.value:
            hit_payload["papers_count"] = len(papers)
        hits.append(hit_payload)

    return {
        "mode": "lookup",
        "query": query,
        "domain": domain or None,
        "kind": kind_filter,
        "hits": hits,
        "offset": offset,
        "limit": limit,
        "total_hits": total_hits,
        "has_more": offset + len(rows) < total_hits,
    }


def _attach_code_repo_to_papers(
    conn: sqlite3.Connection, papers: list[dict[str, Any]]
) -> None:
    """Decorate each paper entry with a small ``code_repo`` envelope.

    Mirrors the BM25 enrichment so an agent who lands on a taxonomy hit
    has the same "you can ground this in code" signal without an extra
    follow-up. The envelope now identifies the linked repo by
    ``repo_slug`` (the canonical id) alongside its URL.
    """
    if not papers:
        return
    names = [p["paper_name"] for p in papers if p.get("paper_name")]
    if not names:
        return
    placeholders = ",".join("?" * len(names))
    rows = conn.execute(
        f"""
        SELECT p.paper_name, r.repo_slug, r.url, r.status, r.file_count
          FROM papers p
          JOIN repos r ON r.paper_id = p.id
         WHERE p.paper_name IN ({placeholders})
        """,
        names,
    ).fetchall()
    by_name: dict[str, dict[str, Any] | None] = {}
    for name, repo_slug, url, status, file_count in rows:
        by_name[name] = {
            "repo_slug": repo_slug,
            "url": url,
            "status": status,
            "file_count": int(file_count or 0),
        }
    for entry in papers:
        entry["code_repo"] = by_name.get(entry.get("paper_name"))


# ---------------------------------------------------------------------------
# Mode 2.5 — Search (first-pass composite)
# ---------------------------------------------------------------------------


# (taxonomy, sections, readmes) bucket gate keyed on the parsed surface:
# qualifier (None == default == all three). Values were validated by the
# parser so the lookup is exhaustive.
_SEARCH_SURFACE_BUCKETS: dict[str | None, tuple[bool, bool, bool]] = {
    None: (True, True, True),
    "sections": (False, True, False),
    "readmes": (False, False, True),
    "taxonomy": (True, False, False),
    "both": (False, True, True),
}


def mode_search(
    conn: sqlite3.Connection,
    *,
    query: str,
    filters: dict[str, Any],
    limit: int = 5,
) -> dict[str, Any]:
    """First-pass exploratory search across the corpus.

    Fans out three subqueries and returns labeled buckets the caller can
    use to orient before drilling in:

    * ``taxonomy``: canonical_terms hits via a single ``terms_fts`` MATCH
      with no ``term_type`` predicate so entity/topic/collection rows mix
      in one ranking. Pure FTS — no KNN fallback. Each row carries
      ``canonical_name``, ``kind``, ``entity_type``, ``domain``.
    * ``sections``: slim summary of BM25 section hits — one row per
      paper with ``hit_count`` and ``top_sections`` (the matching
      section titles, no snippets, no enrichment).
    * ``readmes``: slim summary of BM25 README hits — one row per paper
      with ``hit_count``, ``path``, and the matched ``snippet``.

    Quick-glance shape by design. Caller drills into ``lookup`` (canonical
    metadata + papers), ``bm25`` (full snippets + figures), or
    ``read``/``toc`` (full paper text) once it knows what to ask for.

    No figure attachment by design — the MCP wrapper registers this tool
    with ``AttachMode.NONE`` so the response stays text-only.

    Empty/whitespace ``query`` → ``status='empty_query'`` (soft failure,
    agent-recoverable). Parser-rejected query → ``status='malformed_query'``.

    Beyond the bm25 surface, ``mode_search`` honors two extra qualifiers:

    * ``surface:sections|readmes|both`` — restricts which buckets are
      populated. Default keeps both.
    * ``kind:entity|topic|collection`` — narrows the taxonomy bucket to
      the named term_type.
    """
    if not query or not query.strip():
        return {
            "mode": "search",
            "status": "empty_query",
            "query": query,
            "error": "empty query",
            "hint": (
                "search needs at least one word; pass a non-empty query."
            ),
        }

    try:
        parsed = _parse_github_query(query)
    except EmptyQueryError as e:
        return _empty_query_payload(mode="search", query=query, error=str(e))
    except GitHubQueryError as e:
        return _malformed_query_payload(
            mode="search", query=query, error=str(e),
        )

    try:
        domain = _merge_qualifier_with_kwarg(
            parsed.qualifiers, filters.get("domain"), "domain"
        )
        collection = _merge_qualifier_with_kwarg(
            parsed.qualifiers, filters.get("collection"), "collection"
        )
    except ConflictingFilterError as e:
        return _malformed_query_payload(
            mode="search", query=query, error=str(e),
        )

    paper_name = parsed.qualifiers.get("paper")
    kind_filter = parsed.qualifiers.get("kind")

    want_taxonomy, want_sections, want_readmes = _SEARCH_SURFACE_BUCKETS[
        parsed.qualifiers.get("surface")
    ]

    if not parsed.fts_expression:
        return _empty_query_payload(
            mode="search", query=query,
            error="query held only qualifiers, no FTS body",
        )

    # Unqueried buckets are OMITTED from the payload (vs emitted as []) so
    # the agent reads "skipped" rather than "searched and found nothing."
    taxonomy: list[dict[str, Any]] | None = None
    if want_taxonomy:
        taxonomy = _search_taxonomy(
            conn,
            fts_expression=parsed.fts_expression,
            domain=domain,
            kind=kind_filter,
            limit=limit,
        )

    sections_slim: list[dict[str, Any]] | None = None
    if want_sections:
        sections_payload = _bm25_sections(
            conn,
            fts_expression=parsed.fts_expression,
            domain=domain,
            collection=collection,
            paper_name=paper_name,
            limit=limit,
            snippet_tokens=64,
            enrich=False,
        )
        sections_slim = [
            {
                "paper_name": g["paper_name"],
                "hit_count": g.get("hit_count", 0),
                "hits": [
                    {
                        "section_title": s.get("section_title", ""),
                        "breadcrumb": s.get("breadcrumb", ""),
                        "snippet": s.get("snippet", ""),
                    }
                    for s in g.get("sections", [])
                    if s.get("section_title")
                ],
            }
            for g in sections_payload.get("results", [])
        ]

    readmes_slim: list[dict[str, Any]] | None = None
    if want_readmes:
        readmes_slim = []
        readmes_payload = _bm25_readmes(
            conn,
            fts_expression=parsed.fts_expression,
            domain=domain,
            collection=collection,
            paper_name=paper_name,
            limit=limit,
            snippet_tokens=64,
            enrich=False,
        )
        for g in readmes_payload.get("results", []):
            rh = g.get("readme_hit") or {}
            readmes_slim.append({
                "repo_slug": g["repo_slug"],
                "paper_name": g.get("paper_name"),
                "hit_count": g.get("hit_count", 0),
                "path": rh.get("path"),
                "snippet": rh.get("snippet"),
            })

    payload: dict[str, Any] = {
        "mode": "search",
        "query": query,
        "domain": domain or None,
    }
    if taxonomy is not None:
        payload["taxonomy"] = taxonomy
    if sections_slim is not None:
        payload["sections"] = sections_slim
    if readmes_slim is not None:
        payload["readmes"] = readmes_slim
    return payload


_MAX_SEARCH_MULTI_QUERIES = 8


def mode_search_multi(
    conn: sqlite3.Connection,
    *,
    queries: list[str],
    filters: dict[str, Any],
    limit: int = 5,
) -> dict[str, Any]:
    """Run multiple ``mode_search`` queries independently and concatenate
    their per-query payloads into one envelope.

    Each query is parsed and executed via :func:`mode_search` on its own,
    so per-query qualifiers (``paper:``, ``surface:``, ``kind:``…),
    operators, and soft-failure statuses (``empty_query`` /
    ``malformed_query``) are preserved on each sub-result. Filters
    supplied as kwargs apply uniformly to every query.

    The envelope shape is::

        {
            "mode": "search",
            "multi": True,
            "queries": [...],
            "domain": <kwarg domain or None>,
            "results": [<full mode_search payload>, ...],
        }

    Capped at ``_MAX_SEARCH_MULTI_QUERIES`` to keep one MCP response
    bounded; over-cap calls return a top-level ``malformed_query`` payload
    rather than silently truncating.
    """
    if not queries:
        return _empty_query_payload(
            mode="search", query="",
            error="search_multi called with no queries",
        )
    if len(queries) > _MAX_SEARCH_MULTI_QUERIES:
        return _malformed_query_payload(
            mode="search",
            query=", ".join(queries),
            error=(
                f"too many queries: {len(queries)} "
                f"(max {_MAX_SEARCH_MULTI_QUERIES})"
            ),
        )
    results = [
        mode_search(conn, query=q, filters=filters, limit=limit)
        for q in queries
    ]
    return {
        "mode": "search",
        "multi": True,
        "queries": list(queries),
        "domain": filters.get("domain") or None,
        "results": results,
    }


def _search_taxonomy(
    conn: sqlite3.Connection,
    *,
    fts_expression: str,
    domain: str | None,
    kind: str | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    """Multi-kind canonical term search via FTS5 only.

    Single ``terms_fts MATCH`` with the parsed FTS expression. No
    ``term_type`` predicate by default so all three kinds (entity / topic
    / collection) compete in one bm25 ranking. ``kind`` (from a
    ``kind:`` qualifier) narrows the bucket to one term_type. No KNN
    fallback — keep this orientation path cheap and predictable; callers
    that want semantic broadening drill into ``lookup`` (Tier-B KNN).
    """
    fts_sql = (
        "SELECT term_id, domain, term_type, entity_type, canonical_name "
        "  FROM terms_fts "
        " WHERE terms_fts MATCH ? "
    )
    fts_params: list[Any] = [fts_expression]
    if kind:
        fts_sql += " AND term_type = ? "
        fts_params.append(kind)
    if domain:
        fts_sql += " AND domain = ? "
        fts_params.append(domain)
    fts_sql += " ORDER BY rank LIMIT ?"
    fts_params.append(limit)

    try:
        fts_rows = conn.execute(fts_sql, fts_params).fetchall()
    except sqlite3.OperationalError as exc:
        # Degenerate MATCH (e.g. all-punctuation after quoting) raises
        # "fts5: syntax error near ..."; treat as zero hits. Anything
        # else (missing table, IO) is a real failure.
        if "fts5" not in str(exc).lower():
            raise
        _LOG.warning(
            "terms_fts MATCH syntax error for fts=%r: %s", fts_expression, exc,
        )
        return []

    seen_ids: set[int] = set()
    out: list[dict[str, Any]] = []
    for term_id, dom, term_type, entity_type, canonical_name in fts_rows:
        if term_id in seen_ids:
            continue
        seen_ids.add(term_id)
        out.append({
            "canonical_name": canonical_name,
            "kind": term_type,
            "entity_type": entity_type,
            "domain": dom,
        })
    return out


# ---------------------------------------------------------------------------
# Mode 3 — Browse
# ---------------------------------------------------------------------------


def mode_browse(
    conn: sqlite3.Connection,
    *,
    which: str,
    filters: dict[str, Any],
) -> dict[str, Any]:
    """Pure-SQL list queries. ``which`` selects the view (a :class:`BrowseView`)."""
    try:
        view = BrowseView(which)
    except ValueError as exc:
        raise ValueError(f"unknown browse view: {which!r}") from exc
    domain = filters.get("domain")
    if view is BrowseView.COLLECTIONS:
        # *_collections counts a paper or post once per primary/secondary
        # membership it carries. UNION ALL across paper + post tables so
        # the count reflects everything in the collection regardless of
        # source kind.
        params: list[Any] = []
        domain_filter = ""
        if domain:
            domain_filter = " WHERE domain = ?"
            params.append(domain)
        sql = (
            "WITH all_bindings AS ("
            "  SELECT collection, domain FROM paper_collections "
            "  UNION ALL "
            "  SELECT collection, domain FROM post_collections "
            ") "
            "SELECT collection, COUNT(*) AS n "
            "  FROM all_bindings "
            f" {domain_filter}"
            " GROUP BY collection ORDER BY n DESC, collection"
        )
        rows = conn.execute(sql, params).fetchall()
        return {
            "mode": view,
            "results": [{"collection": r[0], "count": r[1]} for r in rows],
        }

    if view is BrowseView.TOPICS:
        collection = filters.get("collection")
        # Aggregate paper + post + repo bindings — all three kinds count.
        if collection:
            # Scope to one collection: papers via paper_collections,
            # posts via post_collections, repos via the scalar
            # repos.collection. `domain` disambiguates collection names
            # that exist in multiple domains.
            paper_clauses = ["pc.collection = ?"]
            paper_params: list[Any] = [collection]
            post_clauses = ["poc.collection = ?"]
            post_params: list[Any] = [collection]
            repo_clauses = ["r.collection = ?"]
            repo_params: list[Any] = [collection]
            if domain:
                paper_clauses.append("pc.domain = ?")
                paper_params.append(domain)
                post_clauses.append("poc.domain = ?")
                post_params.append(domain)
                repo_clauses.append("r.domain = ?")
                repo_params.append(domain)
            sql = (
                "WITH bindings AS ( "
                "  SELECT t.topic, t.target_kind "
                "    FROM topics t "
                "    JOIN paper_collections pc "
                "      ON t.target_kind = 'paper' "
                "     AND t.target_id = pc.paper_id "
                f"    WHERE {' AND '.join(paper_clauses)} "
                "  UNION ALL "
                "  SELECT t.topic, t.target_kind "
                "    FROM topics t "
                "    JOIN post_collections poc "
                "      ON t.target_kind = 'post' "
                "     AND t.target_id = poc.post_id "
                f"    WHERE {' AND '.join(post_clauses)} "
                "  UNION ALL "
                "  SELECT t.topic, t.target_kind "
                "    FROM topics t "
                "    JOIN repos r "
                "      ON t.target_kind = 'repo' "
                "     AND t.target_id = r.id "
                f"    WHERE {' AND '.join(repo_clauses)} "
                ") "
                "SELECT topic, "
                "       COUNT(*) AS n, "
                "       SUM(CASE WHEN target_kind='paper' THEN 1 ELSE 0 END) AS paper_n, "
                "       SUM(CASE WHEN target_kind='post'  THEN 1 ELSE 0 END) AS post_n, "
                "       SUM(CASE WHEN target_kind='repo'  THEN 1 ELSE 0 END) AS repo_n "
                "  FROM bindings "
                " GROUP BY topic "
                " ORDER BY n DESC, topic"
            )
            params = paper_params + post_params + repo_params
        else:
            sql = (
                "SELECT topic, "
                "       COUNT(*) AS n, "
                "       SUM(CASE WHEN target_kind='paper' THEN 1 ELSE 0 END) AS paper_n, "
                "       SUM(CASE WHEN target_kind='post'  THEN 1 ELSE 0 END) AS post_n, "
                "       SUM(CASE WHEN target_kind='repo'  THEN 1 ELSE 0 END) AS repo_n "
                "  FROM topics "
            )
            params = []
            if domain:
                sql += " WHERE domain = ? "
                params.append(domain)
            sql += " GROUP BY topic ORDER BY n DESC, topic"
        rows = conn.execute(sql, params).fetchall()
        return {
            "mode": view,
            "results": [
                {
                    "topic": r[0],
                    "count": int(r[1] or 0),
                    "paper_count": int(r[2] or 0),
                    "post_count": int(r[3] or 0),
                    "repo_count": int(r[4] or 0),
                }
                for r in rows
            ],
        }

    if view is BrowseView.ENTITY_TYPE:
        entity_type = filters.get("entity_type")
        if not entity_type:
            raise ValueError("mode_browse(which='entity_type') requires "
                             "filters['entity_type']")
        # Pure list of canonicals in the type. Per-paper drilldown
        # happens via `--entity NAME` once the user picks one.
        sql = (
            "SELECT canonical_name FROM canonical_terms "
            " WHERE term_type = 'entity' "
            "   AND entity_type = ? "
        )
        params = [entity_type]
        if domain:
            sql += " AND domain = ? "
            params.append(domain)
        sql += " ORDER BY canonical_name"
        rows = conn.execute(sql, params).fetchall()
        return {
            "mode": view,
            "entity_type": entity_type,
            "results": [{"entity_name": r[0]} for r in rows],
        }

    if view is BrowseView.ALIASES:
        term = filters.get("aliases_term")
        if not term:
            raise ValueError("mode_browse(which='aliases') requires "
                             "filters['aliases_term']")
        # The CLI exposes a single string — if ambiguous across scopes, a
        # future caller can narrow with --domain. For now we return aliases
        # across all matching canonical rows.
        rows = conn.execute(
            "SELECT ta.alias, ta.source_paper, ta.match_tier "
            "  FROM term_aliases ta "
            "  JOIN canonical_terms ct ON ct.id = ta.term_id "
            " WHERE ct.canonical_name = ? "
            " ORDER BY ta.source_paper, ta.alias",
            (term,),
        ).fetchall()
        return {
            "mode": view,
            "term": term,
            "results": [
                {
                    "alias": r[0],
                    "source_paper": r[1],
                    "match_tier": r[2],
                }
                for r in rows
            ],
        }

    # BrowseView.NEEDS_REVIEW
    rows = conn.execute(
        "SELECT paper_name, domain, ingested_at "
        "  FROM papers WHERE needs_review = 1 "
        " ORDER BY ingested_at"
    ).fetchall()
    return {
        "mode": view,
        "results": [
            {
                "paper_name": r[0],
                "domain": r[1],
                "ingested_at": r[2],
            }
            for r in rows
        ],
    }


# ---------------------------------------------------------------------------
# Top-down: overview + collection drill-down
# ---------------------------------------------------------------------------


def _serialize_collection(node: CollectionNode) -> dict[str, Any]:
    return {
        "name": node.name,
        "description": node.description,
        "paper_count": node.paper_count,
        "repo_count": node.repo_count,
    }


def _serialize_domain(node: DomainNode) -> dict[str, Any]:
    return {
        "name": node.name,
        "description": node.description,
        "paper_count": node.paper_count,
        "repo_count": node.repo_count,
        "collection_count": len(node.collections),
        "collections": [_serialize_collection(c) for c in node.collections],
    }


def _domain_node_from_dict(d: dict[str, Any]) -> DomainNode:
    """Adapter: rebuild a ``DomainNode`` from its serialized JSON form.

    Used by ``format_overview_tree`` so the formatter can run the shared
    renderer against the same payload that's shipped as ``structuredContent``
    — no DB connection threaded into the format step.
    """
    colls = [
        CollectionNode(
            name=c["name"],
            description=c.get("description"),
            paper_count=int(c.get("paper_count") or 0),
            repo_count=int(c.get("repo_count") or 0),
        )
        for c in (d.get("collections") or [])
    ]
    return DomainNode(
        name=d["name"],
        description=d.get("description"),
        paper_count=int(d.get("paper_count") or 0),
        collections=tuple(colls),
        overflow=int(d.get("overflow") or 0),
        repo_count=int(d.get("repo_count") or 0),
    )


def mode_overview(
    conn: sqlite3.Connection,
    *,
    filters: dict[str, Any],
) -> dict[str, Any]:
    """Top-down corpus map. Returns the nested ``domains → collections``
    tree with paper counts; empty rows are dropped so the response
    reflects content, not registry.
    """
    domain_filter = filters.get("domain")
    domains = load_taxonomy(
        conn,
        domain=domain_filter,
        include_empty_collections=False,
        include_empty_domains=False,
        collections_per_domain_limit=None,
    )
    return {
        "mode": "overview",
        "domain": domain_filter,
        "domains": [_serialize_domain(d) for d in domains],
    }


def _resolve_collection_targets(
    conn: sqlite3.Connection,
    *,
    names: list[str],
    domain_filter: str | None,
) -> tuple[list[tuple[str, str, str | None]], list[str]]:
    """Resolve each collection name to ``(domain, name, description)`` rows.

    Returns ``(targets, missing)``. Each target is a unique
    ``(domain, name, description)`` triple in input order. A collection
    name found in multiple domains (PK is ``(domain, name)``) without a
    ``domain_filter`` returns one target per matching domain.

    Names are also looked up in ``papers.collection`` so legacy rows
    that aren't registered in ``collections`` still resolve. A name that
    is found in neither is appended to ``missing``.
    """
    seen_pairs: set[tuple[str, str]] = set()
    targets: list[tuple[str, str, str | None]] = []
    missing: list[str] = []

    for name in names:
        if not name:
            missing.append(name)
            continue
        if domain_filter:
            row = conn.execute(
                "SELECT domain, name, description FROM collections "
                " WHERE domain = ? AND name = ?",
                (domain_filter, name),
            ).fetchone()
            if row is None:
                # Fall back to paper_collections — legacy rows may not
                # be registered in the curated `collections` table.
                fallback = conn.execute(
                    "SELECT 1 FROM paper_collections "
                    " WHERE domain = ? AND collection = ? LIMIT 1",
                    (domain_filter, name),
                ).fetchone()
                if fallback is None:
                    missing.append(name)
                    continue
                pair = (domain_filter, name)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                targets.append((domain_filter, name, None))
                continue
            pair = (row[0], row[1])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            targets.append((row[0], row[1], row[2]))
            continue

        rows = conn.execute(
            "SELECT domain, name, description FROM collections WHERE name = ? "
            " ORDER BY domain",
            (name,),
        ).fetchall()
        if not rows:
            # Fallback to paper_collections — covers legacy rows that
            # aren't registered in the curated `collections` table.
            paper_rows = conn.execute(
                "SELECT DISTINCT domain FROM paper_collections "
                " WHERE collection = ? "
                " ORDER BY domain",
                (name,),
            ).fetchall()
            if not paper_rows:
                missing.append(name)
                continue
            for (d,) in paper_rows:
                pair = (d, name)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                targets.append((d, name, None))
            continue
        for d, n, desc in rows:
            pair = (d, n)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            targets.append((d, n, desc))

    return targets, missing


def mode_collection(
    conn: sqlite3.Connection,
    *,
    collection_names: list[str],
    filters: dict[str, Any],
    include_abstracts: bool = True,
    include_topics: bool = True,
    limit: int = 20,
) -> dict[str, Any]:
    """Drill into one or more collections; return their papers (with
    abstracts/topics by default).

    Cross-domain name collisions return one entry per matching
    ``(domain, name)`` pair — Claude can re-call with ``domain`` to
    narrow. Names that don't resolve land in ``missing`` rather than
    raising. Empty input is a caller bug — raises ``ValueError``.
    """
    if not collection_names:
        raise ValueError("collection_names must contain at least one name")

    domain_filter = filters.get("domain")
    targets, missing = _resolve_collection_targets(
        conn, names=collection_names, domain_filter=domain_filter
    )

    abstract_col = ", abstract" if include_abstracts else ""
    entries: list[dict[str, Any]] = []
    for d_name, c_name, c_desc in targets:
        total_row = conn.execute(
            "SELECT COUNT(pc.paper_id) "
            "  FROM paper_collections pc "
            " WHERE pc.domain = ? AND pc.collection = ?",
            (d_name, c_name),
        ).fetchone()
        total_papers = int(total_row[0] or 0) if total_row else 0

        post_total_row = conn.execute(
            "SELECT COUNT(poc.post_id) "
            "  FROM post_collections poc "
            " WHERE poc.domain = ? AND poc.collection = ?",
            (d_name, c_name),
        ).fetchone()
        total_posts = int(post_total_row[0] or 0) if post_total_row else 0

        repo_total_row = conn.execute(
            "SELECT COUNT(*) FROM repos "
            " WHERE paper_id IS NULL AND domain = ? AND collection = ?",
            (d_name, c_name),
        ).fetchone()
        total_repos = int(repo_total_row[0] or 0) if repo_total_row else 0

        rows = conn.execute(
            f"""
            SELECT p.id, p.paper_name, p.title, p.authors, p.date,
                   p.section_count, p.figure_count{abstract_col}
              FROM papers p
              JOIN paper_collections pc ON pc.paper_id = p.id
             WHERE pc.domain = ? AND pc.collection = ?
             ORDER BY p.date DESC, p.paper_name
             LIMIT ?
            """,
            (d_name, c_name, limit),
        ).fetchall()

        paper_ids = [int(r[0]) for r in rows]
        topics_by_id: dict[int, list[str]] = {pid: [] for pid in paper_ids}
        if include_topics and paper_ids:
            topics_by_id.update(_topics_batch(conn, paper_ids))

        # Linked-repo lookup for has_repo / repo_slug stamps on papers[].
        repo_by_paper_id: dict[int, tuple[str, str]] = {}
        if paper_ids:
            placeholders = ",".join("?" for _ in paper_ids)
            for pid, slug, url in conn.execute(
                f"SELECT paper_id, repo_slug, url FROM repos "
                f" WHERE paper_id IN ({placeholders})",
                paper_ids,
            ).fetchall():
                repo_by_paper_id[int(pid)] = (slug, url)

        papers: list[dict[str, Any]] = []
        for r in rows:
            pid = int(r[0])
            repo_pair = repo_by_paper_id.get(pid)
            paper: dict[str, Any] = {
                "paper_name": r[1],
                "title": r[2],
                "authors": r[3],
                "date": r[4],
                "section_count": int(r[5] or 0),
                "figure_count": int(r[6] or 0),
                "has_repo": repo_pair is not None,
                "repo_slug": repo_pair[0] if repo_pair else None,
            }
            if include_abstracts:
                paper["abstract"] = r[7]
            if include_topics:
                paper["topics"] = topics_by_id.get(pid, [])
            papers.append(paper)

        # Standalone repos in this (domain, collection).
        repo_rows = conn.execute(
            """
            SELECT id, repo_slug, url, owner, name, description,
                   has_readme, file_count, status
              FROM repos
             WHERE paper_id IS NULL
               AND domain = ? AND collection = ?
             ORDER BY repo_slug
             LIMIT ?
            """,
            (d_name, c_name, limit),
        ).fetchall()
        repo_ids = [int(rr[0]) for rr in repo_rows]
        repo_topics_by_id: dict[int, list[str]] = {rid: [] for rid in repo_ids}
        if include_topics and repo_ids:
            repo_topics_by_id.update(_topics_batch_repo(conn, repo_ids))

        repos: list[dict[str, Any]] = []
        for rr in repo_rows:
            rid = int(rr[0])
            repos.append({
                "repo_slug": rr[1],
                "url": rr[2],
                "owner": rr[3],
                "name": rr[4],
                "description": rr[5],
                "has_readme": bool(rr[6]),
                "file_count": int(rr[7] or 0),
                "status": rr[8],
                **(
                    {"topics": repo_topics_by_id.get(rid, [])}
                    if include_topics else {}
                ),
            })

        # Posts in this collection. Slim shape — title + slug + author +
        # site_name + topics — keeps the response bounded while letting
        # callers pivot to read/toc/bm25 with the slug.
        post_abstract_col = ", abstract" if include_abstracts else ""
        post_rows = conn.execute(
            f"""
            SELECT po.id, po.post_name, po.title, po.author, po.site_name,
                   po.date, po.section_count{post_abstract_col}
              FROM posts po
              JOIN post_collections poc ON poc.post_id = po.id
             WHERE poc.domain = ? AND poc.collection = ?
             ORDER BY po.date DESC, po.post_name
             LIMIT ?
            """,
            (d_name, c_name, limit),
        ).fetchall()

        post_ids = [int(r[0]) for r in post_rows]
        post_topics_by_id: dict[int, list[str]] = {pid: [] for pid in post_ids}
        if include_topics and post_ids:
            post_topics_by_id.update(_topics_batch_post(conn, post_ids))

        posts: list[dict[str, Any]] = []
        for r in post_rows:
            pid = int(r[0])
            entry: dict[str, Any] = {
                "post_name": r[1],
                "title": r[2],
                "author": r[3],
                "site_name": r[4],
                "date": r[5],
                "section_count": int(r[6] or 0),
            }
            if include_abstracts:
                entry["abstract"] = r[7]
            if include_topics:
                entry["topics"] = post_topics_by_id.get(pid, [])
            posts.append(entry)

        entries.append({
            "domain": d_name,
            "collection": c_name,
            "description": c_desc,
            "paper_count": total_papers,
            "post_count": total_posts,
            "repo_count": total_repos,
            "papers_truncated": total_papers > len(papers),
            "posts_truncated": total_posts > len(posts),
            "repos_truncated": total_repos > len(repos),
            "papers": papers,
            "posts": posts,
            "repos": repos,
        })

    return {
        "mode": "collection",
        "domain": domain_filter,
        "collections": entries,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Mode 4 — ToC
# ---------------------------------------------------------------------------


def mode_toc(conn: sqlite3.Connection, *, paper_name: str) -> dict[str, Any]:
    """Flatten ``papers.markdown`` (or ``posts.markdown``) into level-1..3
    headers via the shared :func:`split_sections` walker. The slug
    namespace is shared, so a ``paper_name`` argument may resolve to
    either a paper or a post — both share the same ToC contract.
    """
    markdown = _read_source_markdown(conn, paper_name)
    if markdown is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")

    toc = [
        {"level": chunk.level, "title": chunk.title}
        for chunk in split_sections(markdown)
    ]
    return {"mode": "toc", "paper_name": paper_name, "toc": toc}


def mode_toc_many(
    conn: sqlite3.Connection, *, paper_names: list[str]
) -> dict[str, Any]:
    """Multi-paper ToC. Resolves each name independently; missing names are
    reported in ``missing`` rather than raising, so a typo in one name doesn't
    abandon the rest. Empty list raises — that's a caller bug.
    """
    if not paper_names:
        raise ValueError("paper_names must contain at least one paper name")

    seen: set[str] = set()
    ordered: list[str] = []
    for name in paper_names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in ordered:
        try:
            results.append(mode_toc(conn, paper_name=name))
        except ValueError:
            missing.append(name)
    return {
        "mode": "toc_many",
        "paper_names": ordered,
        "results": results,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Mode 5a — Read
# ---------------------------------------------------------------------------


def mode_read(
    conn: sqlite3.Connection,
    *,
    paper_name: str,
    section: str | None,
) -> dict[str, Any]:
    """Return the full markdown or a hierarchical section slice.

    Failures of the section slice are NOT raised — they're emitted as
    structured payloads so the agent can recover. ``status`` is one of:

    - missing: ``"section_not_found"`` — well-formed query, no matching
      header in this paper. Payload includes the actual top-level section
      titles + a hint pointing at ``--toc`` and the whole-paper fallback.
    - malformed: ``"malformed_section_query"`` — Claude's query violates
      breadcrumb syntax. Payload echoes the bad query and the rule message.

    A non-existent paper still raises ``ValueError`` — that's a hard error
    (the caller picked the wrong arg), not a recoverable agent miss.
    """
    markdown = _read_source_markdown(conn, paper_name)
    if markdown is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")

    if section is None:
        return {
            "mode": "read",
            "status": "ok",
            "paper_name": paper_name,
            "section": None,
            "text": markdown,
        }

    try:
        text = find_hierarchical_section(markdown, section)
    except SectionQueryError as e:
        return {
            "mode": "read",
            "status": "malformed_section_query",
            "paper_name": paper_name,
            "requested_section": section,
            "error": str(e),
            "hint": (
                "Use a single title like 'Method' or a breadcrumb like "
                "'Parent > Child' with non-empty segments separated by ' > '."
            ),
        }

    if text is None:
        available = [
            c.title for c in split_sections(markdown) if c.level <= 2
        ]
        return {
            "mode": "read",
            "status": "section_not_found",
            "paper_name": paper_name,
            "requested_section": section,
            "available_top_level_sections": available,
            "hint": (
                f"Run --toc {paper_name} for the full hierarchy, or drop "
                f"--section to read the whole paper."
            ),
        }

    return {
        "mode": "read",
        "status": "ok",
        "paper_name": paper_name,
        "section": section,
        "text": text,
    }


# ---------------------------------------------------------------------------
# Mode 6 — Repo tree / read code file
# ---------------------------------------------------------------------------


_LINES_RE = re.compile(r"^(\d+)-(\d+)$")


def mode_repo(
    conn: sqlite3.Connection, *, repo: str
) -> dict[str, Any]:
    """Single-repo metadata + topics + linked paper.

    Cheap "tell me about this repo" lookup. Returns the repo's full row,
    its topic list, and (if paper-linked) the paper_name + title so the
    caller can pivot to the prose surfaces.
    """
    row = conn.execute(
        """
        SELECT r.id, r.repo_slug, r.url, r.host, r.owner, r.name,
               r.description, r.default_branch, r.commit_sha, r.fetched_at,
               r.ingested_at, r.domain, r.collection, r.status,
               r.needs_review, r.file_count, r.has_readme,
               p.paper_name, p.title
          FROM repos r
          LEFT JOIN papers p ON p.id = r.paper_id
         WHERE r.repo_slug = ?
        """,
        (repo,),
    ).fetchone()
    if row is None:
        return {
            "mode": "repo",
            "status": "not_found",
            "repo_slug": repo,
            "hint": (
                f"no repo with repo_slug={repo!r}. Try `mode_collection` "
                f"to browse known repos by domain/collection."
            ),
        }

    repo_id = int(row[0])
    topics_by_id = _topics_batch_repo(conn, [repo_id])
    return {
        "mode": "repo",
        "status": "ok",
        "repo_slug": row[1],
        "url": row[2],
        "host": row[3],
        "owner": row[4],
        "name": row[5],
        "description": row[6],
        "default_branch": row[7],
        "commit_sha": row[8],
        "fetched_at": row[9],
        "ingested_at": row[10],
        "domain": row[11],
        "collection": row[12],
        "repo_status": row[13],
        "needs_review": bool(row[14]),
        "file_count": int(row[15] or 0),
        "has_readme": bool(row[16]),
        "paper_name": row[17],
        "paper_title": row[18],
        "topics": topics_by_id.get(repo_id, []),
    }


def _resolve_repo_target(
    conn: sqlite3.Connection,
    *,
    paper_name: str | None,
    repo: str | None,
) -> tuple[int, str, str | None] | dict[str, Any]:
    """Resolve a repo target from either ``paper_name`` (paper-linked
    repo) or ``repo`` (repo_slug) into ``(repo_id, repo_slug, paper_name)``.

    Returns either the resolved tuple or a soft-status payload that
    callers should return verbatim. Exactly one of the two args must be
    set — callers enforce that at the dispatch boundary.
    """
    if repo:
        row = conn.execute(
            "SELECT r.id, r.repo_slug, p.paper_name "
            "  FROM repos r LEFT JOIN papers p ON p.id = r.paper_id "
            " WHERE r.repo_slug = ?",
            (repo,),
        ).fetchone()
        if row is None:
            raise ValueError(f"repo not found: repo_slug={repo!r}")
        return int(row[0]), row[1], row[2]

    assert paper_name is not None
    resolved = lookup_slug(conn, paper_name)
    if resolved is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")
    if resolved.kind is SourceKind.POST:
        # Posts have no linked repo in v1, so surface that as a soft
        # no_repo status rather than raising.
        return {
            "mode": "repo_tree",
            "status": "no_repo",
            "paper_name": paper_name,
            "hint": (
                f"slug {paper_name!r} resolves to a post; post→repo "
                "linkage is not implemented in v1."
            ),
        }
    paper_id = resolved.id
    repo_row = conn.execute(
        "SELECT id, repo_slug FROM repos WHERE paper_id = ?", (paper_id,),
    ).fetchone()
    if repo_row is None:
        return {
            "mode": "repo_tree",
            "status": "no_repo",
            "paper_name": paper_name,
            "hint": (
                f"no repo is linked to paper {paper_name}. Either no repo "
                f"was discovered during fetch, or fetch hasn't run yet."
            ),
        }
    return int(repo_row[0]), repo_row[1], paper_name


def mode_repo_tree(
    conn: sqlite3.Connection,
    *,
    paper_name: str | None = None,
    repo: str | None = None,
) -> dict[str, Any]:
    """List every ``code_files`` path under a repo.

    Identify the target by either ``paper_name`` (the paper's linked
    repo) or ``repo`` (a repo_slug — works for standalone repos too).
    Exactly one must be set.

    Soft statuses on missing data:
    - ``no_repo`` — paper has no linked repo row.
    - ``failed_repo`` — clone failed previously; URL kept for reference.
    """
    if (paper_name is None) == (repo is None):
        raise ValueError("mode_repo_tree requires exactly one of paper_name / repo")

    resolved = _resolve_repo_target(conn, paper_name=paper_name, repo=repo)
    if isinstance(resolved, dict):
        return resolved
    repo_id, repo_slug, linked_paper = resolved

    repo_meta = conn.execute(
        "SELECT url, commit_sha, fetched_at, status FROM repos WHERE id = ?",
        (repo_id,),
    ).fetchone()
    url, commit, fetched_at, status = repo_meta

    if status == RepoStatus.FAILED_REPO.value:
        return {
            "mode": "repo_tree",
            "status": "failed_repo",
            "repo_slug": repo_slug,
            "paper_name": linked_paper,
            "url": url,
            "hint": (
                f"git clone {url} failed during ingest. Re-run "
                f"`ingest --repo {url} --force` (standalone) or "
                f"`ingest --url <id> --force` (paper-linked) to retry."
            ),
        }

    file_rows = conn.execute(
        "SELECT path, language, size_bytes FROM code_files "
        " WHERE repo_id = ? ORDER BY path",
        (repo_id,),
    ).fetchall()

    files = [
        {"path": p, "language": lang, "size_bytes": int(sz)}
        for p, lang, sz in file_rows
    ]
    total = sum(f["size_bytes"] for f in files)

    return {
        "mode": "repo_tree",
        "status": "ok",
        "repo_slug": repo_slug,
        "paper_name": linked_paper,
        "url": url,
        "commit": commit,
        "fetched_at": fetched_at,
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }


def mode_read_code(
    conn: sqlite3.Connection,
    *,
    paper_name: str | None = None,
    repo: str | None = None,
    path: str,
    lines: str | None = None,
) -> dict[str, Any]:
    """Read one ``code_files`` row, optionally sliced by 1-based line range.

    Identify the repo by either ``paper_name`` or ``repo`` (repo_slug).
    Soft statuses (mirror ``mode_read``):
    - ``file_not_found``
    - ``malformed_lines``
    - ``no_repo`` (when called via paper_name and the paper has no repo)
    """
    if (paper_name is None) == (repo is None):
        raise ValueError("mode_read_code requires exactly one of paper_name / repo")

    resolved = _resolve_repo_target(conn, paper_name=paper_name, repo=repo)
    if isinstance(resolved, dict):
        # _resolve_repo_target only returns no_repo soft-status; rebrand
        # it into mode_read_code shape.
        resolved["mode"] = "read_code"
        resolved["path"] = path
        return resolved
    repo_id, repo_slug, linked_paper = resolved

    file_row = conn.execute(
        "SELECT path, language, size_bytes, content "
        "  FROM code_files WHERE repo_id = ? AND path = ?",
        (repo_id, path),
    ).fetchone()
    if file_row is None:
        return {
            "mode": "read_code",
            "status": "file_not_found",
            "repo_slug": repo_slug,
            "paper_name": linked_paper,
            "path": path,
            "hint": f"Run --repo-tree --repo {repo_slug} for the available paths.",
        }

    stored_path, language, size_bytes, content = file_row

    if lines is None:
        return {
            "mode": "read_code",
            "status": "ok",
            "repo_slug": repo_slug,
            "paper_name": linked_paper,
            "path": stored_path,
            "language": language,
            "size_bytes": int(size_bytes),
            "content": content,
        }

    m = _LINES_RE.match(lines)
    if m is None:
        return {
            "mode": "read_code",
            "status": "malformed_lines",
            "paper_name": paper_name,
            "path": stored_path,
            "requested_lines": lines,
            "error": f"--lines must be A-B with 1-based positive ints; got {lines!r}",
            "hint": "Example: --lines 100-200 reads lines 100..200 inclusive.",
        }
    a, b = int(m.group(1)), int(m.group(2))
    if a < 1 or b < a:
        return {
            "mode": "read_code",
            "status": "malformed_lines",
            "paper_name": paper_name,
            "path": stored_path,
            "requested_lines": lines,
            "error": f"--lines requires 1 <= A <= B; got A={a}, B={b}",
            "hint": "Example: --lines 100-200 reads lines 100..200 inclusive.",
        }

    sliced = "".join(content.splitlines(keepends=True)[a - 1:b])
    return {
        "mode": "read_code",
        "status": "ok",
        "paper_name": paper_name,
        "path": stored_path,
        "language": language,
        "size_bytes": int(size_bytes),
        "lines": [a, b],
        "content": sliced,
    }


# ---------------------------------------------------------------------------
# Mode 5b — Figure BLOB extraction
# ---------------------------------------------------------------------------


def _assert_safe_paper_name(paper: str) -> None:
    if not _SLUG_RE.fullmatch(paper):
        raise ValueError(
            f"paper_name {paper!r} does not match ^[a-z0-9_]+$ — refusing "
            f"to interpolate into a tempfile prefix"
        )


def _safe_n_for_filename(n: Any) -> str:
    """Sanitize the figure identifier for use in a tempfile prefix.

    Paper names are already slug-validated, but ``n`` can legitimately be a
    caption label like ``"Figure 3a"`` with spaces. Collapse anything not in
    ``[A-Za-z0-9]`` to an underscore so mkstemp's prefix argument stays well
    behaved across platforms.
    """
    s = str(n)
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return cleaned or "x"


def _lookup_paper_id(conn: sqlite3.Connection, paper: str) -> int:
    """Resolve a slug to a ``papers.id``.

    Posts are rejected with :class:`ValueError` — figures are paper-only
    in v1 (blog posts don't yet ingest inline images, see
    ``planning/blog-posts.md`` for the v2 plan).
    """
    _assert_safe_paper_name(paper)
    resolved = lookup_slug(conn, paper)
    if resolved is None:
        raise ValueError(f"paper not found: paper_name={paper!r}")
    if resolved.kind is SourceKind.POST:
        raise ValueError(
            f"figures unavailable for posts: slug={paper!r} resolves to a "
            "post, but blog-post figure extraction is not implemented in v1"
        )
    return resolved.id


def _write_blob_tempfile(image: bytes, *, prefix: str, suffix: str = ".png") -> str:
    """Atomically spill ``image`` bytes to a new mkstemp path and return it.

    The BaseException guard covers the narrow window where ``os.fdopen``
    raises after mkstemp handed us the fd but before the ``with`` block
    would have closed it.
    """
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(image)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return path


def mode_figure(
    conn: sqlite3.Connection,
    *,
    paper: str,
    n: str,
) -> dict[str, Any]:
    """Extract a figure BLOB to a ``tempfile.mkstemp`` path.

    ``n`` is tried first as an integer against ``figures.figure_number``
    (DOM ordinal), then as a string against ``figures.display_number``
    (caption label like ``"Figure 3a"``). Raises :class:`ValueError` if
    neither lookup hits.
    """
    paper_id = _lookup_paper_id(conn, paper)

    frow = None
    try:
        fn = int(n)
    except (TypeError, ValueError):
        fn = None
    if fn is not None:
        frow = conn.execute(
            "SELECT image, mime_type FROM figures "
            " WHERE paper_id = ? AND figure_number = ?",
            (paper_id, fn),
        ).fetchone()
    if frow is None:
        frow = conn.execute(
            "SELECT image, mime_type FROM figures "
            " WHERE paper_id = ? AND display_number = ?",
            (paper_id, str(n)),
        ).fetchone()
    if frow is None:
        raise ValueError(
            f"figure not found: paper={paper!r} n={n!r}"
        )

    image, mime_type = frow
    path = _write_blob_tempfile(
        image, prefix=f"lodestone_{paper}_fig{_safe_n_for_filename(n)}_"
    )
    return {
        "mode": "figure",
        "paper_name": paper,
        "figure_number": n,
        "path": path,
        "mime_type": mime_type or "image/png",
    }


# ---------------------------------------------------------------------------
# DB introspection + read-only SQL escape hatch
# ---------------------------------------------------------------------------
# Three modes for the rare 5% case where none of the curated mode_* fits the
# agent's question: enumerate the schema, inspect one or many tables, and run
# an arbitrary read-only SELECT. Read-only is enforced by opening a fresh
# ``mode=ro`` URI connection per query call (see ``get_readonly_conn``); a
# 5 s wall-clock budget is enforced via ``set_progress_handler``; row output
# is hard-capped to ``_QUERY_MAX_ROWS``.

# Internal FTS5 / vec0 shadow-table suffix patterns. Filtered out of
# mode_tables() unless include_internal=True.
_INTERNAL_TABLE_SUFFIXES: tuple[str, ...] = (
    "_data", "_idx", "_content", "_docsize", "_config",
)

# Hard ceiling on rows returned by mode_query. Agents paginate via
# LIMIT/OFFSET in their own SQL.
_QUERY_MAX_ROWS = 1000

# Wall-clock budget for a single mode_query call, in seconds.
_QUERY_TIMEOUT_SECONDS = 5.0

# How often the SQLite engine asks the progress handler whether to keep
# going. Smaller = finer interrupt granularity; this is only the upper
# bound on how many VDBE steps run between checks.
_QUERY_PROGRESS_STEPS = 1_000_000


def _is_internal_shadow_name(name: str) -> bool:
    return any(name.endswith(suf) for suf in _INTERNAL_TABLE_SUFFIXES)


def _classify_table_kind(typ: str, sql: str | None) -> str:
    if typ == "table" and sql and sql.lstrip().upper().startswith(
        "CREATE VIRTUAL TABLE"
    ):
        return "virtual"
    return typ


def mode_tables(
    conn: sqlite3.Connection,
    *,
    include_internal: bool = False,
) -> dict[str, Any]:
    """List every user table / view / virtual table in the DB.

    Virtual tables (FTS5, vec0) are tagged ``virtual`` so callers can tell
    them apart from regular tables. FTS5 / vec0 shadow tables (``%_data``,
    ``%_idx``, ``%_content``, ``%_docsize``, ``%_config``) are filtered
    out by default; pass ``include_internal=True`` to surface them.
    """
    rows = conn.execute(
        """
        SELECT name, type, sql
          FROM sqlite_master
         WHERE type IN ('table', 'view')
         ORDER BY type, name
        """
    ).fetchall()

    tables: list[dict[str, Any]] = []
    for name, typ, sql in rows:
        if not include_internal and _is_internal_shadow_name(name):
            continue
        tables.append({"name": name, "type": _classify_table_kind(typ, sql)})

    return {
        "mode": "tables",
        "status": "ok",
        "include_internal": bool(include_internal),
        "tables": tables,
    }


def mode_schema(
    conn: sqlite3.Connection,
    *,
    table_names: list[str],
) -> dict[str, Any]:
    """Return DDL + column metadata + index metadata for each named table.

    Names that don't resolve land in ``missing`` (mirrors ``mode_toc_many``
    / ``mode_collection`` rather than raising on a single typo). Empty
    input raises :class:`ValueError` — that's a caller bug.
    """
    if not table_names:
        raise ValueError("table_names must contain at least one name")

    ordered = list(dict.fromkeys(table_names))

    tables: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in ordered:
        master = conn.execute(
            "SELECT type, sql FROM sqlite_master "
            " WHERE name = ? AND type IN ('table', 'view')",
            (name,),
        ).fetchone()
        if master is None:
            missing.append(name)
            continue
        typ, sql = master
        # Bind table name into pragma_table_info / pragma_index_list rather
        # than interpolating it into a PRAGMA statement.
        columns = [
            {
                "cid": int(r[0]),
                "name": r[1],
                "type": r[2],
                "notnull": int(r[3]),
                "dflt_value": r[4],
                "pk": int(r[5]),
            }
            for r in conn.execute(
                "SELECT cid, name, type, [notnull], dflt_value, pk "
                "  FROM pragma_table_info(?)",
                (name,),
            ).fetchall()
        ]
        indexes = [
            {
                "name": r[1],
                "unique": int(r[2]),
                "origin": r[3],
                "partial": int(r[4]),
            }
            for r in conn.execute(
                "SELECT seq, name, [unique], origin, partial "
                "  FROM pragma_index_list(?)",
                (name,),
            ).fetchall()
        ]
        tables.append({
            "name": name,
            "type": _classify_table_kind(typ, sql),
            "sql": sql,
            "columns": columns,
            "indexes": indexes,
        })

    return {
        "mode": "schema",
        "status": "ok",
        "tables": tables,
        "missing": missing,
    }


def _serialize_row(description: tuple, row: tuple) -> dict[str, Any]:
    """Turn a positional sqlite row + ``cur.description`` into a column-keyed
    dict. ``bytes`` columns are summarized as ``{"_blob": True, "size_bytes":
    N}`` so multi-MB figure BLOBs don't bloat the JSON envelope.
    """
    out: dict[str, Any] = {}
    for col, value in zip(description, row):
        col_name = col[0]
        if isinstance(value, memoryview):
            out[col_name] = {"_blob": True, "size_bytes": value.nbytes}
        elif isinstance(value, (bytes, bytearray)):
            out[col_name] = {"_blob": True, "size_bytes": len(value)}
        else:
            out[col_name] = value
    return out


def _count_statements(sql: str) -> int:
    """Count distinct SQL statements separated by ``;`` terminators. A
    non-empty trailing fragment after the last terminator counts as one
    additional statement (so ``"SELECT 1; SELECT 2"`` — where the second
    has no trailing ``;`` — counts as 2).

    ``sqlite3.complete_statement`` only meaningfully changes verdict at a
    ``;`` boundary (it tracks string/comment state inside the statement),
    so we slice on ``;`` rather than rebuilding a prefix per character.
    """
    count = 0
    start = 0
    for i, ch in enumerate(sql):
        if ch != ";":
            continue
        piece = sql[start:i + 1]
        if sqlite3.complete_statement(piece) and piece.strip(" \t\r\n;"):
            count += 1
            start = i + 1
    if sql[start:].strip():
        count += 1
    return count


def mode_query(
    conn: sqlite3.Connection,
    *,
    sql: str,
    db_path: PathLike | None = None,
    timeout_seconds: float = _QUERY_TIMEOUT_SECONDS,
    max_rows: int = _QUERY_MAX_ROWS,
) -> dict[str, Any]:
    """Run an arbitrary read-only SQL statement against the DB.

    Read-only is enforced by opening a fresh ``mode=ro`` URI connection
    (DML/DDL surfaces as ``SQLITE_READONLY`` from the engine). A
    progress-handler-driven wall-clock timeout caps runtime. Output rows
    are hard-capped at ``max_rows``; agents paginate by writing
    ``LIMIT N OFFSET M`` (with a stable ``ORDER BY``) in their own SQL.

    ``conn`` is the existing writable handle — used only to pull the
    DB path if ``db_path`` isn't supplied. The actual execution runs on
    a separate read-only connection that is opened and closed inside
    this call.

    Soft-fail statuses (no exception raised):

    - ``multiple_statements`` — input contained more than one terminated
      SQL statement.
    - ``read_only_violation`` — engine rejected as not-read-only
      (DML / DDL / write-pragma).
    - ``query_timeout`` — exceeded ``timeout_seconds``.
    - ``query_failed`` — any other engine error (syntax, unknown table,
      type mismatch, etc.).
    """
    n_stmts = _count_statements(sql)
    if n_stmts > 1:
        return _query_soft_fail(
            sql=sql,
            status="multiple_statements",
            error=(
                f"input contains {n_stmts} statements; mode_query accepts "
                f"exactly one. Drop trailing ';' separators or split into "
                f"separate calls."
            ),
        )

    if db_path is None:
        # sqlite3.Connection has no public "filename" attribute, but
        # ``database_list`` returns it for the main schema.
        row = conn.execute("PRAGMA database_list").fetchone()
        if row is None or not row[2]:
            return _query_soft_fail(
                sql=sql,
                status="query_failed",
                error="cannot resolve db path from connection",
            )
        db_path = row[2]

    ro = get_readonly_conn(db_path)

    deadline = time.monotonic() + float(timeout_seconds)

    def _on_progress() -> int:
        return 1 if time.monotonic() >= deadline else 0

    ro.set_progress_handler(_on_progress, _QUERY_PROGRESS_STEPS)

    try:
        try:
            cur = ro.execute(sql)
            rows = cur.fetchmany(max_rows + 1)
        except sqlite3.OperationalError as exc:
            return _classify_query_error(sql, exc)
        except sqlite3.DatabaseError as exc:
            return _query_soft_fail(
                sql=sql, status="query_failed", error=str(exc),
            )

        truncated = len(rows) > max_rows
        if truncated:
            rows = rows[:max_rows]

        description = cur.description or ()
        columns = [c[0] for c in description]
        serialized = [_serialize_row(description, r) for r in rows]
    finally:
        ro.set_progress_handler(None, 0)
        ro.close()

    return {
        "mode": "query",
        "status": "ok",
        "sql": sql,
        "columns": columns,
        "row_count": len(serialized),
        "truncated": truncated,
        "rows": serialized,
    }


def _query_soft_fail(
    *, sql: str, status: str, error: str,
) -> dict[str, Any]:
    return {
        "mode": "query",
        "status": status,
        "sql": sql,
        "error": error,
    }


def _classify_query_error(
    sql: str, exc: sqlite3.OperationalError,
) -> dict[str, Any]:
    """Map an OperationalError raised by the engine onto the right
    soft-fail bucket. The engine surfaces read-only refusals as
    ``"attempt to write a readonly database"`` and timeouts (interrupted
    by the progress handler) as ``"interrupted"``.
    """
    msg = str(exc)
    low = msg.lower()
    if "readonly" in low or "read-only" in low or "attempt to write" in low:
        return _query_soft_fail(
            sql=sql, status="read_only_violation", error=msg,
        )
    if "interrupted" in low:
        return _query_soft_fail(
            sql=sql, status="query_timeout",
            error=(
                f"query exceeded {_QUERY_TIMEOUT_SECONDS:.0f}s wall-clock "
                f"budget and was interrupted."
            ),
        )
    return _query_soft_fail(sql=sql, status="query_failed", error=msg)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def to_json(payload: dict[str, Any]) -> str:
    """Serialize ``payload`` to a single JSON string with UTF-8 passthrough."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _clean_breadcrumb_for_display(breadcrumb: str) -> str:
    """Strip per-segment ``#`` level markers from a section breadcrumb.

    ``split_sections`` builds breadcrumbs as ``# A > ## B > ### C`` where
    the leading ``#`` count encodes the markdown header level. The level
    info is redundant once we have the path, and the bare ``#`` symbols
    look like markdown headers when rendered inside our own markdown
    output. ``# A > ## B`` → ``A > B``.
    """
    if not breadcrumb:
        return ""
    parts = [seg.lstrip("# ").strip() for seg in breadcrumb.split(" > ")]
    return " > ".join(p for p in parts if p)


def format_search_markdown(
    payload: dict[str, Any], *, header_level: int = 1
) -> str:
    """Render a ``mode_search`` payload as a compact markdown digest.

    Used at the MCP boundary so Claude reads a token-cheap orientation
    summary instead of the raw JSON. The full JSON shape stays available
    via ``structuredContent``. Multi-query envelopes route to a sectioned
    renderer that nests sub-results one level deeper.

    ``header_level`` is the level of the top-level header (1 == ``#``).
    The multi renderer recurses with level 3 so its per-query H2 sits
    above the sub-payload's bucket headers.
    """
    if payload.get("multi"):
        return _format_search_multi_markdown(payload)

    h_top = "#" * header_level
    h_bucket = "#" * (header_level + 1)
    h_sub = "#" * (header_level + 2)

    status = payload.get("status")
    if status in ("empty_query", "malformed_query"):
        label = "empty query" if status == "empty_query" else "malformed query"
        lines = [f"{h_top} search ({label})"]
        if status == "malformed_query":
            lines.append(f"query: {payload.get('query', '')!r}")
        err = payload.get("error")
        if err and status == "malformed_query":
            lines.append(f"error: {err}")
        hint = payload.get("hint")
        if hint:
            lines.append(hint)
        return "\n".join(lines)

    query = payload.get("query", "")
    domain = payload.get("domain")
    header = f"{h_top} search {query!r}"
    if domain:
        header += f"  [domain={domain}]"
    lines: list[str] = [header, ""]

    if "taxonomy" in payload:
        taxonomy = payload.get("taxonomy") or []
        lines.append(f"{h_bucket} taxonomy ({len(taxonomy)})")
        if taxonomy:
            by_kind: dict[str, list[dict[str, Any]]] = {}
            for row in taxonomy:
                by_kind.setdefault(row.get("kind", "?"), []).append(row)
            for kind in ("entity", "topic", "collection"):
                rows = by_kind.get(kind)
                if not rows:
                    continue
                lines.append(f"{h_sub} {kind} ({len(rows)})")
                for row in rows:
                    lines.append(f"- {row.get('canonical_name', '?')}")
        else:
            lines.append(
                "(none — try a different surface form, or skip to bm25)"
            )
        lines.append("")

    if "sections" in payload:
        sections = payload.get("sections") or []
        sec_total = sum(int(g.get("hit_count", 0)) for g in sections)
        lines.append(
            f"{h_bucket} sections ({sec_total} hits across {len(sections)} "
            f"paper{'' if len(sections) == 1 else 's'})"
        )
        if sections:
            for g in sections:
                lines.append(
                    f"- {g.get('paper_name', '?')} ({g.get('hit_count', 0)} hits)"
                )
                for i, hit in enumerate(g.get("hits", []), start=1):
                    crumb = _clean_breadcrumb_for_display(hit.get("breadcrumb", ""))
                    heading = crumb or hit.get("section_title", "?")
                    lines.append(f"  {i}. **{heading}**")
                    raw = (hit.get("snippet") or "").strip()
                    snip = " ".join(raw.split())
                    if snip:
                        lines.append(f"     {snip}")
        else:
            lines.append("(none)")
        lines.append("")

    if "readmes" in payload:
        readmes = payload.get("readmes") or []
        rdm_total = sum(int(g.get("hit_count", 0)) for g in readmes)
        lines.append(
            f"{h_bucket} readmes ({rdm_total} hits across {len(readmes)} "
            f"paper{'' if len(readmes) == 1 else 's'})"
        )
        if readmes:
            for g in readmes:
                path = g.get("path") or ""
                raw = (g.get("snippet") or "").strip()
                ident = g.get("repo_slug") or g.get("paper_name") or "?"
                snip = " ".join(raw.split())
                lines.append(
                    f"- {ident}: {path} — {snip}"
                )
        else:
            lines.append("(none)")

    return "\n".join(lines)


def _format_search_multi_markdown(payload: dict[str, Any]) -> str:
    """Render a multi-query envelope as one stitched markdown document.

    H1 is the multi header; per-query H2 ("## query N: 'q'") sits above
    each sub-payload's H3 bucket headers. Soft-failure sub-results
    surface inline so the agent doesn't need to crack open
    ``structuredContent``.
    """
    queries = payload.get("queries") or []
    domain = payload.get("domain")
    n = len(queries)
    header = f"# search (multi: {n} {'query' if n == 1 else 'queries'})"
    if domain:
        header += f"  [domain={domain}]"
    parts: list[str] = [header, ""]

    for i, sub in enumerate(payload.get("results") or [], start=1):
        q = sub.get("query", "")
        status = sub.get("status")
        tag = f" ({status.replace('_', ' ')})" if status else ""
        parts.append(f"## query {i}{tag}: {q!r}")
        # Render the sub at level 2 so its buckets land at H3 ("###
        # taxonomy") under the per-query H2. The sub's own H2 top-header
        # is redundant with our "## query i" line, so strip it.
        sub_md = format_search_markdown(sub, header_level=2)
        sub_lines = sub_md.split("\n")
        if sub_lines and sub_lines[0].startswith("## search"):
            sub_lines = sub_lines[1:]
            if sub_lines and sub_lines[0] == "":
                sub_lines = sub_lines[1:]
        parts.extend(sub_lines)
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def format_overview_tree(payload: dict[str, Any]) -> str:
    """Render a ``mode_overview`` payload as count-style tree text.

    Adapter pulls the JSON payload back into ``DomainNode``s and runs
    the shared :func:`render_taxonomy_tree`. Tree text is the primary
    surface for ``overview`` — JSON wrapping bloats the per-row token
    count 3-4x; tree connectors carry the same structure for free. JSON
    stays in ``structuredContent`` for programmatic consumers.
    """
    domain_filter = payload.get("domain")
    domain_dicts = payload.get("domains") or []
    nodes = [_domain_node_from_dict(d) for d in domain_dicts]

    header = "# overview"
    if domain_filter:
        header += f"  [domain={domain_filter}]"
    legend = (
        "_Domains = broad research areas. Collections = subdivisions of a "
        "domain by approach or technique. Drill in via the 'collection' "
        "tool, then inspect papers with 'toc_many' / 'read'._"
    )
    if not nodes:
        scope = (
            f" under domain={domain_filter!r}" if domain_filter else ""
        )
        return f"{header}\n\n{legend}\n\n(no domains with content{scope})"

    body = render_taxonomy_tree(nodes, style=TaxonomyTreeStyle.COUNT)
    return f"{header}\n\n{legend}\n\n{body}"


def _format_count_label(n: int, *, label: str) -> str:
    return f"{n} {label}" if n == 1 else f"{n} {label}s"


def format_collection_text(payload: dict[str, Any]) -> str:
    """Render a ``mode_collection`` payload as a light tree.

    Each (domain, collection) bundle is a header line with paper rows
    hung off it via ``├──``/``└──`` connectors. Continuation lines (date
    / authors / topics / abstract) use ``│   ``/``    `` indents so the
    tree shape stays legible. Missing names are listed under a trailing
    section so a typo doesn't sink the response.
    """
    domain_filter = payload.get("domain")
    entries = payload.get("collections") or []
    missing = payload.get("missing") or []

    header = "# collection"
    if domain_filter:
        header += f"  [domain={domain_filter}]"
    lines: list[str] = [header, ""]

    if not entries:
        lines.append("(no matching collections)")
    for entry in entries:
        d = entry.get("domain", "?")
        c = entry.get("collection", "?")
        desc = entry.get("description")
        n_total = int(entry.get("paper_count") or 0)
        count_label = _format_count_label(n_total, label="paper")
        head = f"{d} / {c}"
        if desc:
            head += f" — {desc}"
        head += f"  ({count_label})"
        lines.append(head)

        papers = entry.get("papers") or []
        truncated = bool(entry.get("papers_truncated"))
        n_leaves = len(papers) + (1 if truncated else 0)

        for i, p in enumerate(papers):
            is_last = (i == n_leaves - 1)
            top = "└──" if is_last else "├──"
            cont = "    " if is_last else "│   "
            paper_name = p.get("paper_name", "?")
            title = p.get("title") or ""
            head_line = f"{top} {paper_name}"
            if title:
                head_line += f" — {title}"
            lines.append(head_line)

            meta_bits: list[str] = []
            date = p.get("date")
            if date:
                meta_bits.append(str(date))
            authors = p.get("authors")
            if authors:
                meta_bits.append(str(authors))
            if p.get("has_repo"):
                slug = p.get("repo_slug")
                if slug:
                    meta_bits.append(f"repo:{slug}")
            sec_n = int(p.get("section_count") or 0)
            fig_n = int(p.get("figure_count") or 0)
            counts = []
            if sec_n:
                counts.append(f"{sec_n}§")
            if fig_n:
                counts.append(
                    f"{fig_n} fig" if fig_n == 1 else f"{fig_n} figs"
                )
            if counts:
                meta_bits.append(" ".join(counts))
            if meta_bits:
                lines.append(f"{cont}{' · '.join(meta_bits)}")

            topics = p.get("topics")
            if topics:
                lines.append(f"{cont}topics: {', '.join(topics)}")

            abstract = p.get("abstract")
            if abstract:
                snippet = " ".join(str(abstract).split())
                lines.append(f"{cont}abstract: {snippet}")

        if truncated:
            hidden = n_total - len(papers)
            lines.append(
                f"└── (+ {hidden} more not shown; raise limit or call again)"
            )

        lines.append("")

    if missing:
        lines.append(f"## missing ({len(missing)})")
        for name in missing:
            lines.append(f"- {name}")

    return "\n".join(lines).rstrip() + "\n"


def _paging_footer(payload: dict[str, Any], *, page_size: int) -> str | None:
    """Render the "showing hits N-M of T" footer for paged results.

    ``page_size`` is the number of result *rows* (not groups) on this
    page — the caller computes it because the unit varies between
    surfaces (BM25 sums hit_count across paper groups; lookup uses the
    hit count directly).
    """
    if "total_hits" not in payload:
        return None
    total = int(payload.get("total_hits") or 0)
    offset = int(payload.get("offset") or 0)
    limit = int(payload.get("limit") or 0)
    if total == 0 and page_size == 0:
        return "-- no hits --"
    first = offset + 1 if page_size else offset
    last = offset + page_size
    if payload.get("has_more") and limit > 0:
        return (
            f"-- showing hits {first}-{last} of {total} (offset={offset}). "
            f"Re-call with --offset {offset + limit} for next page. --"
        )
    return f"-- showing hits {first}-{last} of {total} (end of results). --"


def to_human(payload: dict[str, Any]) -> str:
    """Short plaintext rendering per mode. Designed for terminal eyeballing,
    not programmatic reuse — pipelines should consume ``to_json``.
    """
    mode = payload.get("mode", "?")
    if mode == "search":
        return format_search_markdown(payload)
    if mode == "overview":
        return format_overview_tree(payload)
    if mode == "collection":
        return format_collection_text(payload)

    lines: list[str] = []

    if mode == "sections":
        if payload.get("status") == "empty_query":
            lines.append(
                f"empty BM25 query: {payload.get('query')!r}"
            )
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif payload.get("status") == "malformed_query":
            lines.append(
                f"malformed BM25 query: {payload.get('query')!r}"
            )
            err = payload.get("error")
            if err:
                lines.append(err)
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif payload.get("status") == "invalid_pagination":
            lines.append(
                f"invalid pagination on BM25 query: {payload.get('query')!r}"
            )
            err = payload.get("error")
            if err:
                lines.append(err)
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        else:
            scope_label = payload.get("scope") or "sections"
            lines.append(
                f"== BM25 {scope_label}: {payload.get('query')!r} =="
            )
            row_count = sum(
                int(h.get("hit_count", 0))
                for h in payload.get("results", [])
            ) + sum(
                int(h.get("hit_count", 0))
                for h in payload.get("repo_results", [])
            )
            for hit in payload.get("results", []):
                ident = hit.get("paper_name") or hit.get("repo_slug") or "?"
                lines.append(
                    f"- {ident} (hits={hit['hit_count']})"
                )
                for s in hit.get("sections", []):
                    lines.append(
                        f"    §{s['section_level']} {s['section_title']}: {s.get('snippet', '')}"
                    )
                rh = hit.get("readme_hit")
                if rh:
                    lines.append(
                        f"    via README: {rh.get('path')}: {rh.get('snippet', '')}"
                    )
                cr = hit.get("code_repo")
                if cr:
                    lines.append(
                        f"    code_repo: {cr.get('url')} "
                        f"(status={cr.get('status')}, files={cr.get('file_count')})"
                    )
            for hit in payload.get("repo_results", []):
                slug = hit.get("repo_slug", "?")
                lines.append(
                    f"- repo:{slug} (hits={hit.get('hit_count', 0)})"
                )
                rh = hit.get("readme_hit")
                if rh:
                    lines.append(
                        f"    README: {rh.get('path')}: {rh.get('snippet', '')}"
                    )
                if hit.get("paper_name"):
                    lines.append(
                        f"    linked paper: {hit['paper_name']}"
                    )
            footer = _paging_footer(payload, page_size=row_count)
            if footer:
                lines.append(footer)

    elif mode == "lookup":
        status = payload.get("status")
        if status == "empty_query":
            lines.append(f"empty lookup query: {payload.get('query')!r}")
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif status == "malformed_query":
            lines.append(f"malformed lookup query: {payload.get('query')!r}")
            err = payload.get("error")
            if err:
                lines.append(err)
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif status == "invalid_pagination":
            lines.append(
                f"invalid pagination on lookup query: {payload.get('query')!r}"
            )
            err = payload.get("error")
            if err:
                lines.append(err)
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        else:
            hits = payload.get("hits") or []
            kind_tag = payload.get("kind") or "any"
            lines.append(
                f"== lookup: {payload.get('query')!r} "
                f"(kind={kind_tag}, {len(hits)} hit"
                f"{'' if len(hits) == 1 else 's'}) =="
            )
            if not hits:
                lines.append("  (no canonical terms matched)")
            # Show domain on the line only if hits straddle multiple domains;
            # otherwise it's a constant we can drop (and surface once in the
            # header instead, when relevant).
            domains = {h.get("domain") for h in hits}
            multi_domain = len(domains) > 1
            for h in hits:
                kind = h.get("kind")
                # entity rows have a meaningful entity_type; topic/collection
                # rows have term_type==entity_type==<kind> which is redundant.
                if kind == "entity":
                    label = f"entity:{h.get('entity_type') or '?'}"
                else:
                    label = kind or "?"
                bits = [f"- {h['canonical_name']} ({label})"]
                if multi_domain:
                    bits.append(f"[{h.get('domain')}]")
                papers_count = int(h.get("papers_count") or 0)
                if papers_count:
                    bits.append(f"papers={papers_count}")
                lines.append(" ".join(bits))
                aliases = h.get("aliases") or []
                if aliases:
                    alias_names = sorted({a["alias"] for a in aliases})
                    lines.append(f"  aliases: {', '.join(alias_names)}")
                papers = h.get("papers") or []
                if papers:
                    lines.append(
                        "  papers: "
                        + ", ".join(p["paper_name"] for p in papers)
                    )
            footer = _paging_footer(payload, page_size=len(hits))
            if footer:
                lines.append(footer)

    elif mode == "collections":
        lines.append("== collections ==")
        for row in payload.get("results", []):
            lines.append(f"  {row['collection']}  ({row['count']})")

    elif mode == "topics":
        lines.append("== topics ==")
        for row in payload.get("results", []):
            lines.append(f"  {row['topic']}  ({row['count']})")

    elif mode == "entity_type":
        lines.append(f"== entity_type: {payload.get('entity_type')!r} ==")
        for row in payload.get("results", []):
            lines.append(f"  {row['entity_name']}")

    elif mode == "aliases":
        lines.append(f"== aliases of {payload.get('term')!r} ==")
        for row in payload.get("results", []):
            lines.append(
                f"  {row['alias']}  (from {row['source_paper']}, "
                f"tier={row['match_tier']})"
            )

    elif mode == "needs_review":
        lines.append("== papers flagged for review ==")
        for row in payload.get("results", []):
            lines.append(
                f"  {row['paper_name']} [{row.get('domain')}] "
                f"ingested={row.get('ingested_at')}"
            )

    elif mode == "toc":
        lines.append(f"== ToC {payload.get('paper_name')} ==")
        for entry in payload.get("toc", []):
            indent = "  " * (entry["level"] - 1)
            lines.append(f"{indent}{'#' * entry['level']} {entry['title']}")

    elif mode == "toc_many":
        for sub in payload.get("results", []):
            lines.append(f"== ToC {sub.get('paper_name')} ==")
            for entry in sub.get("toc", []):
                indent = "  " * (entry["level"] - 1)
                lines.append(
                    f"{indent}{'#' * entry['level']} {entry['title']}"
                )
        missing = payload.get("missing") or []
        if missing:
            lines.append("")
            lines.append(
                f"== missing ({len(missing)}) =="
            )
            for name in missing:
                lines.append(f"  - {name}")

    elif mode == "read":
        status = payload.get("status", "ok")
        paper = payload.get("paper_name")
        if status == "section_not_found":
            lines.append(
                f"section not found in {paper}: "
                f"{payload.get('requested_section')!r}"
            )
            avail = payload.get("available_top_level_sections") or []
            if avail:
                lines.append(f"available top-level sections: {', '.join(avail)}")
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif status == "malformed_section_query":
            lines.append(
                f"malformed --section query for {paper}: "
                f"{payload.get('requested_section')!r}"
            )
            err = payload.get("error")
            if err:
                lines.append(err)
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        else:
            sec = payload.get("section")
            header = f"== {paper}"
            if sec:
                header += f" § {sec}"
            header += " =="
            lines.append(header)
            lines.append(payload.get("text", ""))

    elif mode == "figure":
        lines.append(
            f"figure {payload.get('figure_number')} of "
            f"{payload.get('paper_name')} → {payload.get('path')} "
            f"({payload.get('mime_type')})"
        )

    elif mode == "repo_tree":
        status = payload.get("status", "ok")
        target = payload.get("repo_slug") or payload.get("paper_name") or "?"
        if status == "no_repo":
            lines.append(f"no repo for {target}")
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif status == "failed_repo":
            lines.append(
                f"clone failed for {target}: {payload.get('url')}"
            )
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        else:
            lines.append(
                f"== {target} repo: {payload.get('url')} "
                f"({payload.get('file_count')} files, "
                f"{payload.get('total_bytes')} bytes) =="
            )
            for f in payload.get("files", []):
                lang = f.get("language") or "?"
                lines.append(
                    f"  {f['path']}  [{lang}]  ({f['size_bytes']} B)"
                )

    elif mode == "read_code":
        status = payload.get("status", "ok")
        target = payload.get("repo_slug") or payload.get("paper_name") or "?"
        path = payload.get("path")
        if status == "file_not_found":
            lines.append(f"file not found in {target}: {path!r}")
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif status == "no_repo":
            lines.append(f"no repo for {target}: {path!r}")
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif status == "malformed_lines":
            lines.append(
                f"malformed --lines for {target} {path!r}: "
                f"{payload.get('requested_lines')!r}"
            )
            err = payload.get("error")
            if err:
                lines.append(err)
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        else:
            header = f"== {target} :: {path}"
            ln = payload.get("lines")
            if ln:
                header += f" [lines {ln[0]}-{ln[1]}]"
            header += " =="
            lines.append(header)
            lines.append(payload.get("content", ""))

    elif mode == "tables":
        tables = payload.get("tables") or []
        lines.append(f"== tables ({len(tables)}) ==")
        # Group by type (table / virtual / view) for legibility.
        by_type: dict[str, list[str]] = {}
        for t in tables:
            by_type.setdefault(t.get("type", "?"), []).append(t.get("name", "?"))
        for typ in sorted(by_type):
            names = sorted(by_type[typ])
            lines.append(f"  [{typ}] ({len(names)})")
            for n in names:
                lines.append(f"    {n}")

    elif mode == "schema":
        for t in payload.get("tables") or []:
            lines.append(
                f"== {t.get('name')} ({t.get('type')}) =="
            )
            sql = t.get("sql")
            if sql:
                lines.append(sql)
            cols = t.get("columns") or []
            if cols:
                lines.append("  columns:")
                for c in cols:
                    bits = [
                        f"#{c.get('cid')}",
                        c.get("name", "?"),
                        c.get("type") or "",
                    ]
                    if c.get("notnull"):
                        bits.append("NOT NULL")
                    if c.get("pk"):
                        bits.append(f"PK={c['pk']}")
                    if c.get("dflt_value") is not None:
                        bits.append(f"DEFAULT {c['dflt_value']}")
                    lines.append("    " + " ".join(b for b in bits if b))
            idx = t.get("indexes") or []
            if idx:
                lines.append("  indexes:")
                for ix in idx:
                    bits = [ix.get("name", "?")]
                    if ix.get("unique"):
                        bits.append("UNIQUE")
                    bits.append(f"origin={ix.get('origin')}")
                    if ix.get("partial"):
                        bits.append("partial")
                    lines.append("    " + " ".join(bits))
            lines.append("")
        missing = payload.get("missing") or []
        if missing:
            lines.append(f"== missing ({len(missing)}) ==")
            for n in missing:
                lines.append(f"  - {n}")

    elif mode == "query":
        status = payload.get("status")
        sql = payload.get("sql", "")
        if status != "ok":
            lines.append(f"query {status}: {sql}")
            err = payload.get("error")
            if err:
                lines.append(err)
        else:
            cols = payload.get("columns") or []
            rows = payload.get("rows") or []
            row_count = int(payload.get("row_count") or 0)
            truncated = bool(payload.get("truncated"))
            lines.append(
                f"== query ({row_count} row{'' if row_count == 1 else 's'}"
                + (", truncated" if truncated else "")
                + ") =="
            )
            if cols:
                lines.append("  " + " | ".join(cols))
            for r in rows:
                vals: list[str] = []
                for c in cols:
                    v = r.get(c)
                    if isinstance(v, dict) and v.get("_blob"):
                        vals.append(f"<blob {v.get('size_bytes')} B>")
                    else:
                        vals.append("" if v is None else str(v))
                lines.append("  " + " | ".join(vals))
            if truncated:
                lines.append(
                    f"-- truncated at {row_count} rows; paginate with "
                    f"LIMIT/OFFSET in your SQL --"
                )

    else:
        lines.append(f"(unknown mode: {mode!r})")
        lines.append(json.dumps(payload, indent=2))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI / dispatch
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="search.py",
        description=(
            "Lodestone search CLI. Five modes routed via argparse:\n"
            "  1. BM25 (positional QUERY; searches the sections FTS5 index)\n"
            "  2. Taxonomy lookup (--entity/--topic/--collection without QUERY)\n"
            "  3. Browse (--collections/--topics/--entity-type/--aliases/--needs-review)\n"
            "  4. ToC (--toc PAPER)\n"
            "  5. Read / figure (--read / --figure)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("query", nargs="?", default=None,
                   help="BM25 query string (mode 1)")
    p.add_argument("--domain", default=None, help="filter by papers.domain")
    p.add_argument("--collection", default=None,
                   help="collection name — filter when QUERY is set, "
                        "topic-scope when --topics is set, "
                        "otherwise Mode 2 lookup term")
    p.add_argument(
        "--limit", type=int, default=None,
        help=(
            "max hits per response. Default is mode-specific to mirror "
            "the MCP defaults: 15 for BM25 (positional QUERY), 50 for "
            "--lookup, 5 for --search, 20 for --collection-name. Pass "
            "--limit explicitly to override."
        ),
    )
    p.add_argument(
        "--offset", type=int, default=0,
        help=(
            "skip this many ranked hits (default 0). Pair with --limit "
            "to page deeper into a query. Applies to BM25 and --lookup; "
            "ignored elsewhere. Negative values surface an "
            "`invalid_pagination` soft-fail."
        ),
    )

    p.add_argument("--search", default=None, action="append",
                   help="generic first-pass search: returns three buckets "
                        "(taxonomy / sections / readmes) for QUERY in one call. "
                        "Mirrors the `search` MCP tool. Use --domain to filter; "
                        "--limit caps each bucket (default 5). Repeat the flag "
                        "to fan out multiple queries (max 8); each runs "
                        "independently and the per-query payloads are "
                        "concatenated under one envelope.")

    p.add_argument(
        "--lookup", default=None,
        help=(
            "canonical-term FTS search. Accepts the same GitHub-flavored "
            "syntax as --search/QUERY (bare tokens, \"phrase\", AND/OR/NOT, "
            "parens, term*); honors kind:entity|topic|collection and "
            "domain:NAME qualifiers. Returns ranked hits with aliases "
            "inlined per hit."
        ),
    )
    p.add_argument(
        "--entity", default=None,
        help=(
            "shorthand for --lookup with kind:entity prepended. "
            "Surface form term — extra GitHub syntax is supported."
        ),
    )
    p.add_argument(
        "--topic", default=None,
        help=(
            "shorthand for --lookup with kind:topic prepended. "
            "Surface form term — extra GitHub syntax is supported."
        ),
    )

    p.add_argument(
        "--overview", action="store_true",
        help=(
            "top-down corpus map: nested domains → collections with paper "
            "counts. Use --domain to narrow. Empty domains/collections are "
            "dropped."
        ),
    )
    p.add_argument(
        "--collection-name", dest="collection_name", default=None,
        action="append",
        help=(
            "drill into one or more collections by name. Repeat the flag "
            "to bundle multiple collections in one call."
        ),
    )
    p.add_argument(
        "--no-abstracts", dest="no_abstracts", action="store_true",
        help="for --collection-name: omit paper abstracts.",
    )
    p.add_argument(
        "--no-topics", dest="no_topics", action="store_true",
        help="for --collection-name: omit per-paper topics.",
    )
    p.add_argument(
        "--collection-limit", dest="collection_limit", type=int, default=20,
        help="for --collection-name: max papers per collection (default 20).",
    )

    p.add_argument("--collections", action="store_true",
                   help="browse: list distinct collections")
    p.add_argument("--topics", action="store_true",
                   help="browse: list distinct topics")
    p.add_argument("--entity-type", dest="entity_type", default=None,
                   help="browse: list entity_names of this type")
    p.add_argument("--aliases", default=None,
                   help="browse: list aliases of a canonical_name")
    p.add_argument("--needs-review", dest="needs_review",
                   action="store_true", help="browse: papers flagged for review")

    p.add_argument(
        "--toc", default=None, action="append",
        help=(
            "ToC of PAPER (paper_name). Repeat the flag to fetch multiple "
            "papers in one call: --toc paper_a --toc paper_b. Single use "
            "returns a flat envelope; repeated use returns a toc_many "
            "envelope with per-paper results plus a 'missing' list for "
            "names that didn't resolve."
        ),
    )

    p.add_argument("--read", default=None,
                   help="read full markdown of PAPER (paper_name)")
    p.add_argument("--section", default=None,
                   help="when --read is set, slice to this section "
                        "(supports 'Parent > Child' breadcrumb)")

    p.add_argument("--figure", nargs=2, metavar=("PAPER", "N"), default=None,
                   help="extract figure N from PAPER to a tempfile")

    p.add_argument(
        "--scope",
        default=Scope.SECTIONS.value,
        choices=tuple(s.value for s in Scope),
        help=("BM25 scope for the positional QUERY: sections (default), "
              "readmes, or both."),
    )
    p.add_argument("--repo-tree", dest="repo_tree", default=None,
                   help="list paths in PAPER's code repo (paper_name)")
    p.add_argument("--repo-tree-slug", dest="repo_tree_slug", default=None,
                   help="list paths in REPO_SLUG (standalone or paper-linked)")
    p.add_argument("--read-code", dest="read_code", default=None,
                   help="read a file from PAPER's code repo (paper_name)")
    p.add_argument("--read-code-slug", dest="read_code_slug", default=None,
                   help="read a file from REPO_SLUG (standalone or paper-linked)")
    p.add_argument("--path", default=None,
                   help="repo-relative file path for --read-code")
    p.add_argument("--lines", default=None,
                   help="line range A-B (1-based, inclusive) for --read-code")

    p.add_argument(
        "--tables", action="store_true",
        help=(
            "list every user table / view / virtual table in the DB. "
            "Pair with --include-internal to also show FTS5 / vec0 "
            "shadow tables."
        ),
    )
    p.add_argument(
        "--include-internal", dest="include_internal", action="store_true",
        help=(
            "with --tables: also show FTS5 / vec0 shadow tables "
            "(suffixed _data/_idx/_content/_docsize/_config)."
        ),
    )
    p.add_argument(
        "--schema", default=None, action="append",
        help=(
            "print DDL + columns + indexes for TABLE. Repeat the flag to "
            "fetch many tables in one call. Names that don't resolve land "
            "in 'missing' rather than raising."
        ),
    )
    p.add_argument(
        "--sql", default=None,
        help=(
            "run a single read-only SQL statement against the DB. "
            "Read-only is engine-enforced (mode=ro URI), single-statement "
            "only, capped at 1000 rows, 5s wall-clock timeout. Paginate "
            "with LIMIT N OFFSET M + ORDER BY in your own SQL."
        ),
    )

    p.add_argument("--human", action="store_true",
                   help="emit plaintext instead of JSON")
    p.add_argument("--db", default="lodestone.db", help="sqlite db path")
    return p


def _check_mode_conflicts(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject mutually exclusive mode flag combinations up front.

    The plan's routing table is "first match wins," but silent first-match
    resolution hides user mistakes (``QUERY --entity FOO`` would drop the
    positional query without warning). We enforce exclusivity explicitly:
    exactly one mode must be selected. Mode-1 filters (``--domain``,
    ``--limit``, and ``--collection`` when QUERY is present) are NOT
    counted as a mode.
    """
    modes: list[str] = []
    if args.query is not None:
        modes.append("QUERY (Mode 1 BM25)")
    if args.search is not None:
        modes.append("--search")
    # Mode 2 lookups — --collection is dual-use: only counts as Mode 2 when
    # there is no positional query.
    if args.lookup is not None:
        modes.append("--lookup")
    if args.entity is not None:
        modes.append("--entity")
    if args.topic is not None:
        modes.append("--topic")
    if (
        args.collection is not None
        and args.query is None
        and not args.topics
    ):
        modes.append("--collection (lookup)")
    # Mode 3 browse flags
    if args.collections:
        modes.append("--collections")
    if args.topics:
        modes.append("--topics")
    if args.entity_type:
        modes.append("--entity-type")
    if args.aliases:
        modes.append("--aliases")
    if args.needs_review:
        modes.append("--needs-review")
    # Mode 4 / 5 / 6
    if args.toc is not None:
        modes.append("--toc")
    if args.read is not None:
        modes.append("--read")
    if args.figure is not None:
        modes.append("--figure")
    if args.repo_tree is not None:
        modes.append("--repo-tree")
    if args.repo_tree_slug is not None:
        modes.append("--repo-tree-slug")
    if args.read_code is not None:
        modes.append("--read-code")
    if args.read_code_slug is not None:
        modes.append("--read-code-slug")
    if args.overview:
        modes.append("--overview")
    if args.collection_name is not None:
        modes.append("--collection-name")
    if args.tables:
        modes.append("--tables")
    if args.schema is not None:
        modes.append("--schema")
    if args.sql is not None:
        modes.append("--sql")

    if len(modes) > 1:
        parser.error(
            f"mutually exclusive modes selected: {', '.join(modes)}. "
            f"Pick exactly one."
        )

    # `--scope` is a Mode-1 modifier, not a mode. It MUST NOT count above,
    # but a non-default value without a positional QUERY is a user
    # mistake — fail fast rather than silently dropping the flag.
    if args.scope != Scope.SECTIONS.value and args.query is None:
        parser.error(
            "--scope requires a positional QUERY (Mode 1 BM25). "
            "It is ignored by every other mode."
        )

    # `--read-code` / `--read-code-slug` requires `--path`; `--lines`
    # is only meaningful with one of them.
    using_read_code = args.read_code is not None or args.read_code_slug is not None
    if using_read_code and not args.path:
        parser.error("--read-code/--read-code-slug requires --path REPO_RELATIVE_PATH.")
    if args.lines is not None and not using_read_code:
        parser.error("--lines is only valid with --read-code/--read-code-slug.")

    # --include-internal is a --tables modifier; reject it elsewhere so a
    # stray flag doesn't get silently dropped.
    if args.include_internal and not args.tables:
        parser.error("--include-internal is only valid with --tables.")


def _dispatch(args: argparse.Namespace, conn: sqlite3.Connection) -> dict[str, Any]:
    if args.tables:
        return mode_tables(conn, include_internal=args.include_internal)
    if args.schema is not None:
        return mode_schema(conn, table_names=list(args.schema))
    if args.sql is not None:
        return mode_query(conn, sql=args.sql, db_path=args.db)

    # Top-down: overview / collection
    if args.overview:
        return mode_overview(conn, filters={"domain": args.domain})
    if args.collection_name is not None:
        return mode_collection(
            conn,
            collection_names=list(args.collection_name),
            filters={"domain": args.domain},
            include_abstracts=not args.no_abstracts,
            include_topics=not args.no_topics,
            limit=args.collection_limit,
        )

    # Mode 6: repo tree / read code
    if args.repo_tree is not None:
        return mode_repo_tree(conn, paper_name=args.repo_tree)
    if args.repo_tree_slug is not None:
        return mode_repo_tree(conn, repo=args.repo_tree_slug)
    if args.read_code is not None:
        return mode_read_code(
            conn,
            paper_name=args.read_code,
            path=args.path,
            lines=args.lines,
        )
    if args.read_code_slug is not None:
        return mode_read_code(
            conn,
            repo=args.read_code_slug,
            path=args.path,
            lines=args.lines,
        )

    # Mode 5b: figure
    if args.figure is not None:
        paper, n = args.figure
        return mode_figure(conn, paper=paper, n=n)

    # Mode 5a: read
    if args.read is not None:
        return mode_read(conn, paper_name=args.read, section=args.section)

    # Mode 4: toc. action='append' gives ['name'] for one --toc and
    # ['a', 'b', ...] for repeated use. Single → flat mode_toc envelope;
    # multi → mode_toc_many envelope.
    if args.toc:
        if len(args.toc) == 1:
            return mode_toc(conn, paper_name=args.toc[0])
        return mode_toc_many(conn, paper_names=args.toc)

    # Mode 3: browse
    domain_filter: dict[str, Any] = {"domain": args.domain}
    if args.needs_review:
        return mode_browse(conn, which=BrowseView.NEEDS_REVIEW, filters={})
    if args.collections:
        return mode_browse(conn, which=BrowseView.COLLECTIONS, filters=domain_filter)
    if args.topics:
        topics_filters: dict[str, Any] = {**domain_filter}
        if args.collection is not None:
            topics_filters["collection"] = args.collection
        return mode_browse(conn, which=BrowseView.TOPICS, filters=topics_filters)
    if args.entity_type:
        return mode_browse(
            conn,
            which=BrowseView.ENTITY_TYPE,
            filters={**domain_filter, "entity_type": args.entity_type},
        )
    if args.aliases:
        return mode_browse(
            conn,
            which=BrowseView.ALIASES,
            filters={"aliases_term": args.aliases},
        )

    # Mode 2.5: generic first-pass search (composite — three buckets).
    # ``--search`` uses ``action='append'`` so a single use gives ['q'] and
    # repeated uses give ['q1', 'q2', ...]. Multi fans out via
    # mode_search_multi; the single case still goes through mode_search so
    # legacy callers keep the flat envelope.
    if args.search is not None:
        search_filters: dict[str, Any] = {}
        if args.domain:
            search_filters["domain"] = args.domain
        search_limit = 5 if args.limit is None else args.limit
        if len(args.search) == 1:
            return mode_search(
                conn,
                query=args.search[0],
                filters=search_filters,
                limit=search_limit,
            )
        return mode_search_multi(
            conn,
            queries=args.search,
            filters=search_filters,
            limit=search_limit,
        )

    # Mode 2: taxonomy lookup. --lookup takes a GH-flavored query directly;
    # --entity / --topic / --collection are shorthand that prepend the
    # corresponding `kind:` qualifier to the query. --collection is DUAL-USE
    # (Mode 1 filter when a positional query is set, lookup shorthand
    # otherwise).
    lookup_query: str | None = None
    if args.lookup is not None:
        lookup_query = args.lookup
    elif args.entity is not None:
        lookup_query = f"kind:entity {args.entity}"
    elif args.topic is not None:
        lookup_query = f"kind:topic {args.topic}"
    elif args.collection is not None and args.query is None:
        lookup_query = f"kind:collection {args.collection}"
    if lookup_query is not None:
        return mode_taxonomy_lookup(
            conn,
            query=lookup_query,
            filters=domain_filter,
            limit=50 if args.limit is None else args.limit,
            offset=args.offset,
        )

    # Mode 1: BM25 — defaults to `sections`; `--scope` switches to
    # `readmes` or `both`.
    if args.query is not None:
        filters: dict[str, Any] = {}
        if args.domain:
            filters["domain"] = args.domain
        if args.collection:
            filters["collection"] = args.collection
        return mode_bm25(
            conn,
            query=args.query,
            filters=filters,
            limit=15 if args.limit is None else args.limit,
            offset=args.offset,
            scope=Scope(args.scope),
        )

    raise SystemExit(
        "no action selected — pass a positional QUERY or one of the mode "
        "flags (--search/--lookup/--entity/--topic/--collection/--collections/"
        "--topics/--entity-type/--aliases/--needs-review/--toc/--read/"
        "--figure). Run with --help for details."
    )


_SOFT_FAILURE_STATUSES = frozenset({
    "section_not_found",
    "malformed_section_query",
    "empty_query",
    "malformed_query",
    "invalid_pagination",
    "no_repo",
    "failed_repo",
    "file_not_found",
    "malformed_lines",
    "multiple_statements",
    "read_only_violation",
    "query_timeout",
    "query_failed",
})


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _check_mode_conflicts(parser, args)

    # DB connect is deferred to AFTER parse_args so `--help` never touches
    # the filesystem — argparse's auto-help short-circuits before we land
    # here.
    conn = get_conn(Path(args.db))
    try:
        result = _dispatch(args, conn)
    finally:
        conn.close()

    # Soft failures: agent-recoverable misses (e.g. --section not found,
    # malformed breadcrumb). Payload still carries the diagnostic; exit
    # code 2 signals "you asked for something that didn't work" without
    # the caller having to parse the JSON to find out.
    is_soft_failure = result.get("status") in _SOFT_FAILURE_STATUSES

    if args.human:
        # Keep stdout empty on soft failure so shell pipes don't see partial
        # output. The diagnostic still goes to the user via stderr.
        if is_soft_failure:
            sys.stderr.write(to_human(result) + "\n")
        else:
            sys.stdout.write(to_human(result) + "\n")
    else:
        # JSON mode: payload always lands on stdout (the agent reads it to
        # recover), regardless of success/failure.
        sys.stdout.write(to_json(result) + "\n")

    return 2 if is_soft_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
