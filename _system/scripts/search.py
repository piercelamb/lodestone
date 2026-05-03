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
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# NB: import only cheap stdlib + the cheap internal modules here. Anything
# that pulls torch / sentence_transformers / gliner must live inside the
# function that needs it.
from _system.db.connection import get_conn
from _system.schemas.paper_metadata import PaperStatus
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

_LOG = get_logger("scripts.search")

# BM25 enrichment size cap. Keeping each follow-up query small bounds the
# JSON payload size even on queries that return many hits.
_ENTITY_PREVIEW_LIMIT = 5


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


def _soft_fail_payload(
    *, mode: str, status: str, query: str, error: str, extra_hint: str = ""
) -> dict[str, Any]:
    hint = _EMPTY_QUERY_HINT if status == "empty_query" else _BM25_SYNTAX_HINT
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
    scope: Scope = Scope.SECTIONS,
    snippet_tokens: int = 10,
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

    Soft failures (no exception raised; soft-status payload):

    * ``empty_query`` — punctuation-only or qualifier-only query
    * ``malformed_query`` — quote/paren/operator mismatch, unknown
      qualifier, ``/regex/`` form, or qualifier↔kwarg conflict
    """
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
            limit=limit, snippet_tokens=snippet_tokens,
        )
    elif scope is Scope.READMES:
        result = _bm25_readmes(
            conn, fts_expression=parsed.fts_expression,
            domain=domain, collection=collection, paper_name=paper_name,
            limit=limit, snippet_tokens=snippet_tokens,
        )
    else:
        result = _bm25_both(
            conn, fts_expression=parsed.fts_expression,
            domain=domain, collection=collection, paper_name=paper_name,
            limit=limit, snippet_tokens=snippet_tokens,
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
    snippet_tokens: int = 10,
    enrich: bool = True,
) -> dict[str, Any]:
    # sections columns: (paper_id, domain, paper_name, section_title, section_level, body)
    # snippet() against 'body' = column index 5.
    # Breadcrumb is prepended to body at index time as `breadcrumb\n\n<raw>`,
    # so first line of body == breadcrumb. We extract it explicitly so the
    # caller doesn't depend on whether snippet()'s token window happened to
    # land near the start of body.
    sql = (
        "SELECT s.paper_id, s.domain, s.paper_name, s.section_title, "
        "       s.section_level, "
        f"       snippet(sections, 5, '[', ']', '…', {int(snippet_tokens)}) AS snip, "
        "       substr(s.body, 1, instr(s.body || char(10), char(10)) - 1) AS breadcrumb "
        "  FROM sections s"
    )
    wheres = ["sections MATCH ?"]
    params: list[Any] = [fts_expression]
    if domain:
        wheres.append("s.domain = ?")
        params.append(domain)
    if paper_name:
        wheres.append("s.paper_name = ?")
        params.append(paper_name)
    if collection:
        # Join papers for the collection filter — sections does not carry
        # collection in FTS5 (by design; the paper owns the collection).
        sql += " JOIN papers p ON p.id = s.paper_id"
        wheres.append("p.collection = ?")
        params.append(collection)
    sql += " WHERE " + " AND ".join(wheres)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()

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
    }


def _bm25_readmes(
    conn: sqlite3.Connection,
    *,
    fts_expression: str,
    domain: str | None,
    collection: str | None,
    paper_name: str | None,
    limit: int,
    snippet_tokens: int = 10,
    enrich: bool = True,
) -> dict[str, Any]:
    """BM25 against ``readmes_fts``. Same envelope shape as ``_bm25_sections``
    so ``to_human`` / enrichment branches treat both uniformly."""
    sql = (
        "SELECT r.paper_id, r.domain, r.paper_name, r.path, "
        f"       snippet(readmes_fts, 4, '[', ']', '…', {int(snippet_tokens)}) AS snip "
        "  FROM readmes_fts r"
    )
    wheres = ["readmes_fts MATCH ?"]
    params: list[Any] = [fts_expression]
    if domain:
        wheres.append("r.domain = ?")
        params.append(domain)
    if paper_name:
        wheres.append("r.paper_name = ?")
        params.append(paper_name)
    if collection:
        sql += " JOIN papers p ON p.id = r.paper_id"
        wheres.append("p.collection = ?")
        params.append(collection)
    sql += " WHERE " + " AND ".join(wheres)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, params).fetchall()

    grouped: dict[str, dict[str, Any]] = {}
    pid_order: list[int] = []
    for paper_id, dom, paper_name, path, snip in rows:
        # readmes_fts has at most one row per paper, but we still group
        # to preserve envelope symmetry with `_bm25_sections`.
        group = grouped.get(paper_name)
        if group is None:
            group = {
                "paper_name": paper_name,
                "domain": dom,
                "_paper_id": paper_id,
                "hit_count": 0,
                "sections": [],
                "readme_hit": None,
            }
            grouped[paper_name] = group
            pid_order.append(paper_id)
        group["hit_count"] += 1
        group["readme_hit"] = {"path": path, "snippet": snip}

    if enrich:
        _attach_bm25_enrichment(conn, grouped, pid_order)
    else:
        for group in grouped.values():
            group.pop("_paper_id", None)

    return {
        "mode": "sections",
        "scope": Scope.READMES.value,
        "results": list(grouped.values()),
    }


def _bm25_both(
    conn: sqlite3.Connection,
    *,
    fts_expression: str,
    domain: str | None,
    collection: str | None,
    paper_name: str | None,
    limit: int,
    snippet_tokens: int = 10,
    enrich: bool = True,
) -> dict[str, Any]:
    """Union of sections + READMES hits, merged by paper_name."""
    sec = _bm25_sections(
        conn, fts_expression=fts_expression,
        domain=domain, collection=collection, paper_name=paper_name,
        limit=limit, snippet_tokens=snippet_tokens, enrich=enrich,
    )
    rdm = _bm25_readmes(
        conn, fts_expression=fts_expression,
        domain=domain, collection=collection, paper_name=paper_name,
        limit=limit, snippet_tokens=snippet_tokens, enrich=enrich,
    )

    by_name: dict[str, dict[str, Any]] = {}
    for group in sec.get("results", []):
        by_name[group["paper_name"]] = group
        group.setdefault("readme_hit", None)

    for group in rdm.get("results", []):
        existing = by_name.get(group["paper_name"])
        if existing is None:
            by_name[group["paper_name"]] = group
            continue
        existing["hit_count"] = existing.get("hit_count", 0) + group.get("hit_count", 0)
        if group.get("readme_hit") is not None:
            existing["readme_hit"] = group["readme_hit"]

    return {
        "mode": "sections",
        "scope": Scope.BOTH.value,
        "results": list(by_name.values()),
    }


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
    """One small SELECT per BM25 hit batch — never a fan-out per result."""
    if not paper_ids:
        return {}
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"""
        SELECT p.id, p.code_repo, p.status,
               (SELECT COUNT(*) FROM code_files cf WHERE cf.paper_id = p.id) AS file_count
          FROM papers p
         WHERE p.id IN ({placeholders})
        """,
        paper_ids,
    ).fetchall()
    result: dict[int, dict[str, Any] | None] = {}
    for pid, code_repo, status, file_count in rows:
        if not code_repo:
            result[pid] = None
            continue
        result[pid] = {
            "url": code_repo,
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
        f"SELECT paper_id, topic FROM paper_topics "
        f" WHERE paper_id IN ({placeholders}) "
        f" ORDER BY paper_id, topic",
        paper_ids,
    ).fetchall()
    result: dict[int, list[str]] = {pid: [] for pid in paper_ids}
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
    limit: int = 10,
) -> dict[str, Any]:
    """Canonical-term FTS5 search with aliases inlined per hit.

    ``query`` is a GitHub-code-search-style string parsed by
    :func:`_parse_github_query`: bare tokens (implicit AND), ``"phrase"``,
    ``AND``/``OR``/``NOT``, parens, prefix ``term*``. Two qualifiers are
    honored: ``kind:entity|topic|collection`` (narrows the term_type
    bucket) and ``domain:NAME``. ``paper:`` / ``collection:`` /
    ``surface:`` are rejected — they have no meaning against canonical
    terms.

    Returns up to ``limit`` ranked hits. Each hit carries the canonical
    metadata, every alias for the term (with its source paper), and the
    list of papers that mention it (papers per kind: aliases-source for
    entities, paper_topics for topics, papers.collection for collections).

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
    """
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

    sql = (
        "SELECT term_id, domain, term_type, entity_type, canonical_name "
        "  FROM terms_fts "
        " WHERE terms_fts MATCH ? "
    )
    params: list[Any] = [parsed.fts_expression]
    if kind_filter:
        sql += " AND term_type = ? "
        params.append(kind_filter)
    if domain:
        sql += " AND domain = ? "
        params.append(domain)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    try:
        rows = conn.execute(sql, params).fetchall()
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
                "  FROM paper_topics pt "
                "  JOIN papers p ON p.id = pt.paper_id "
                " WHERE pt.topic = ? AND pt.domain = ? "
                " ORDER BY p.paper_name",
                (canonical_name, dom),
            ).fetchall()
        else:  # collection
            prows = conn.execute(
                "SELECT paper_name FROM papers "
                " WHERE collection = ? AND domain = ? "
                " ORDER BY paper_name",
                (canonical_name, dom),
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
    }


def _attach_code_repo_to_papers(
    conn: sqlite3.Connection, papers: list[dict[str, Any]]
) -> None:
    """Decorate each paper entry with a small ``code_repo`` envelope.

    Mirrors the BM25 enrichment so an agent who lands on a taxonomy hit
    has the same "you can ground this in code" signal without an extra
    follow-up.
    """
    if not papers:
        return
    names = [p["paper_name"] for p in papers if p.get("paper_name")]
    if not names:
        return
    placeholders = ",".join("?" * len(names))
    rows = conn.execute(
        f"""
        SELECT p.paper_name, p.code_repo, p.status,
               (SELECT COUNT(*) FROM code_files cf WHERE cf.paper_id = p.id) AS file_count
          FROM papers p
         WHERE p.paper_name IN ({placeholders})
        """,
        names,
    ).fetchall()
    by_name: dict[str, dict[str, Any] | None] = {}
    for name, code_repo, status, file_count in rows:
        if not code_repo:
            by_name[name] = None
            continue
        by_name[name] = {
            "url": code_repo,
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
                "paper_name": g["paper_name"],
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
        sql = (
            "SELECT collection, COUNT(*) AS n FROM papers "
            " WHERE collection IS NOT NULL "
        )
        params: list[Any] = []
        if domain:
            sql += " AND domain = ? "
            params.append(domain)
        sql += " GROUP BY collection ORDER BY n DESC, collection"
        rows = conn.execute(sql, params).fetchall()
        return {
            "mode": view,
            "results": [{"collection": r[0], "count": r[1]} for r in rows],
        }

    if view is BrowseView.TOPICS:
        sql = (
            "SELECT topic, COUNT(DISTINCT paper_id) AS n FROM paper_topics "
        )
        params = []
        if domain:
            sql += " WHERE domain = ? "
            params.append(domain)
        sql += " GROUP BY topic ORDER BY n DESC, topic"
        rows = conn.execute(sql, params).fetchall()
        return {
            "mode": view,
            "results": [{"topic": r[0], "count": r[1]} for r in rows],
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
    }


def _serialize_domain(node: DomainNode) -> dict[str, Any]:
    return {
        "name": node.name,
        "description": node.description,
        "paper_count": node.paper_count,
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
        )
        for c in (d.get("collections") or [])
    ]
    return DomainNode(
        name=d["name"],
        description=d.get("description"),
        paper_count=int(d.get("paper_count") or 0),
        collections=tuple(colls),
        overflow=int(d.get("overflow") or 0),
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
                # Fall back to papers — legacy rows may not be registered.
                fallback = conn.execute(
                    "SELECT 1 FROM papers WHERE domain = ? AND collection = ? LIMIT 1",
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
            # Fallback to legacy paper-side rows.
            paper_rows = conn.execute(
                "SELECT DISTINCT domain FROM papers "
                " WHERE collection = ? AND domain IS NOT NULL "
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
            "SELECT COUNT(*) FROM papers WHERE domain = ? AND collection = ?",
            (d_name, c_name),
        ).fetchone()
        total = int(total_row[0] or 0) if total_row else 0

        rows = conn.execute(
            f"""
            SELECT id, paper_name, title, authors, date, code_repo,
                   section_count, figure_count{abstract_col}
              FROM papers
             WHERE domain = ? AND collection = ?
             ORDER BY date DESC, paper_name
             LIMIT ?
            """,
            (d_name, c_name, limit),
        ).fetchall()

        paper_ids = [int(r[0]) for r in rows]
        topics_by_id: dict[int, list[str]] = {pid: [] for pid in paper_ids}
        if include_topics and paper_ids:
            placeholders = ",".join("?" for _ in paper_ids)
            for pid, topic in conn.execute(
                f"SELECT paper_id, topic FROM paper_topics "
                f" WHERE paper_id IN ({placeholders}) "
                f" ORDER BY topic",
                paper_ids,
            ).fetchall():
                topics_by_id.setdefault(int(pid), []).append(topic)

        papers: list[dict[str, Any]] = []
        for r in rows:
            pid = int(r[0])
            paper: dict[str, Any] = {
                "paper_name": r[1],
                "title": r[2],
                "authors": r[3],
                "date": r[4],
                "code_repo": r[5],
                "section_count": int(r[6] or 0),
                "figure_count": int(r[7] or 0),
            }
            if include_abstracts:
                paper["abstract"] = r[8]
            if include_topics:
                paper["topics"] = topics_by_id.get(pid, [])
            papers.append(paper)

        entries.append({
            "domain": d_name,
            "collection": c_name,
            "description": c_desc,
            "paper_count": total,
            "papers_truncated": total > len(papers),
            "papers": papers,
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
    """Flatten ``papers.markdown`` into level-1..3 headers via the shared
    :func:`split_sections` walker — keeps ToC boundaries in lockstep with
    how the indexer chunked the paper, including fenced-code suppression.
    """
    row = conn.execute(
        "SELECT markdown FROM papers WHERE paper_name = ?", (paper_name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")
    markdown = row[0] or ""

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
    row = conn.execute(
        "SELECT markdown FROM papers WHERE paper_name = ?", (paper_name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")
    markdown = row[0] or ""

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


def mode_repo_tree(
    conn: sqlite3.Connection, *, paper_name: str
) -> dict[str, Any]:
    """List every ``code_files`` path for ``paper_name``.

    Soft statuses on missing data:
    - ``no_repo`` — paper has no ``code_repo`` URL.
    - ``failed_repo`` — clone failed previously; URL kept for reference.
    """
    row = conn.execute(
        "SELECT id, code_repo, code_repo_commit, code_repo_fetched_at, status "
        "  FROM papers WHERE paper_name = ?",
        (paper_name,),
    ).fetchone()
    if row is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")
    paper_id, code_repo, commit, fetched_at, status = row

    if not code_repo:
        return {
            "mode": "repo_tree",
            "status": "no_repo",
            "paper_name": paper_name,
            "hint": (
                f"papers.code_repo is NULL for {paper_name}. No repo "
                f"discovery hit during fetch — nothing to list."
            ),
        }

    if status == PaperStatus.FAILED_REPO.value:
        return {
            "mode": "repo_tree",
            "status": "failed_repo",
            "paper_name": paper_name,
            "code_repo": code_repo,
            "hint": (
                f"git clone {code_repo} failed during ingest. Re-run "
                f"`ingest --url <id> --force` to retry."
            ),
        }

    file_rows = conn.execute(
        "SELECT path, language, size_bytes FROM code_files "
        " WHERE paper_id = ? ORDER BY path",
        (paper_id,),
    ).fetchall()

    files = [
        {"path": p, "language": lang, "size_bytes": int(sz)}
        for p, lang, sz in file_rows
    ]
    total = sum(f["size_bytes"] for f in files)

    return {
        "mode": "repo_tree",
        "status": "ok",
        "paper_name": paper_name,
        "code_repo": code_repo,
        "commit": commit,
        "fetched_at": fetched_at,
        "file_count": len(files),
        "total_bytes": total,
        "files": files,
    }


def mode_read_code(
    conn: sqlite3.Connection,
    *,
    paper_name: str,
    path: str,
    lines: str | None = None,
) -> dict[str, Any]:
    """Read one ``code_files`` row, optionally sliced by 1-based line range.

    Soft statuses (mirror ``mode_read``):
    - ``file_not_found``
    - ``malformed_lines``
    """
    paper_row = conn.execute(
        "SELECT id FROM papers WHERE paper_name = ?", (paper_name,)
    ).fetchone()
    if paper_row is None:
        raise ValueError(f"paper not found: paper_name={paper_name!r}")
    paper_id = paper_row[0]

    file_row = conn.execute(
        "SELECT path, language, size_bytes, content "
        "  FROM code_files WHERE paper_id = ? AND path = ?",
        (paper_id, path),
    ).fetchone()
    if file_row is None:
        return {
            "mode": "read_code",
            "status": "file_not_found",
            "paper_name": paper_name,
            "path": path,
            "hint": f"Run --repo-tree {paper_name} for the available paths.",
        }

    stored_path, language, size_bytes, content = file_row

    if lines is None:
        return {
            "mode": "read_code",
            "status": "ok",
            "paper_name": paper_name,
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
    _assert_safe_paper_name(paper)
    prow = conn.execute(
        "SELECT id FROM papers WHERE paper_name = ?", (paper,)
    ).fetchone()
    if prow is None:
        raise ValueError(f"paper not found: paper_name={paper!r}")
    return prow[0]


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
                snip = " ".join(raw.split())
                lines.append(
                    f"- {g.get('paper_name', '?')}: {path} — {snip}"
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
            code_repo = p.get("code_repo")
            if code_repo:
                meta_bits.append(str(code_repo))
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
        else:
            scope_label = payload.get("scope") or "sections"
            lines.append(
                f"== BM25 {scope_label}: {payload.get('query')!r} =="
            )
            for hit in payload.get("results", []):
                lines.append(
                    f"- {hit['paper_name']} (hits={hit['hit_count']})"
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
        paper = payload.get("paper_name")
        if status == "no_repo":
            lines.append(f"no code_repo for {paper}")
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif status == "failed_repo":
            lines.append(
                f"clone failed for {paper}: {payload.get('code_repo')}"
            )
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        else:
            lines.append(
                f"== {paper} repo: {payload.get('code_repo')} "
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
        paper = payload.get("paper_name")
        path = payload.get("path")
        if status == "file_not_found":
            lines.append(f"file not found in {paper}: {path!r}")
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        elif status == "malformed_lines":
            lines.append(
                f"malformed --lines for {paper} {path!r}: "
                f"{payload.get('requested_lines')!r}"
            )
            err = payload.get("error")
            if err:
                lines.append(err)
            hint = payload.get("hint")
            if hint:
                lines.append(hint)
        else:
            header = f"== {paper} :: {path}"
            ln = payload.get("lines")
            if ln:
                header += f" [lines {ln[0]}-{ln[1]}]"
            header += " =="
            lines.append(header)
            lines.append(payload.get("content", ""))

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
                        "otherwise Mode 2 lookup term")
    p.add_argument("--limit", type=int, default=10,
                   help="max BM25 hits (default: 10)")

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
                   help="list paths in PAPER's code repo")
    p.add_argument("--read-code", dest="read_code", default=None,
                   help="read a file from PAPER's code repo")
    p.add_argument("--path", default=None,
                   help="repo-relative file path for --read-code")
    p.add_argument("--lines", default=None,
                   help="line range A-B (1-based, inclusive) for --read-code")

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
    if args.collection is not None and args.query is None:
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
    if args.read_code is not None:
        modes.append("--read-code")
    if args.overview:
        modes.append("--overview")
    if args.collection_name is not None:
        modes.append("--collection-name")

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

    # `--read-code` requires `--path`; `--lines` is only meaningful with
    # `--read-code`.
    if args.read_code is not None and not args.path:
        parser.error("--read-code requires --path REPO_RELATIVE_PATH.")
    if args.lines is not None and args.read_code is None:
        parser.error("--lines is only valid with --read-code.")


def _dispatch(args: argparse.Namespace, conn: sqlite3.Connection) -> dict[str, Any]:
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
    if args.read_code is not None:
        return mode_read_code(
            conn,
            paper_name=args.read_code,
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
        return mode_browse(conn, which=BrowseView.TOPICS, filters=domain_filter)
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
        if len(args.search) == 1:
            return mode_search(
                conn,
                query=args.search[0],
                filters=search_filters,
                limit=args.limit,
            )
        return mode_search_multi(
            conn,
            queries=args.search,
            filters=search_filters,
            limit=args.limit,
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
            limit=args.limit,
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
            limit=args.limit,
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
    "no_repo",
    "failed_repo",
    "file_not_found",
    "malformed_lines",
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
