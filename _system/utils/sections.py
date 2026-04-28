"""Shared markdown section-walking utilities.

Pure stdlib (``re``) — imports must not trigger any ML model load. Consumed by
the LaTeXML parser, entity extractor, indexer, and search layers; all must
agree on the same header walker so "what got extracted" matches "what search
returns".
"""
from __future__ import annotations

import re
from typing import Callable, NamedTuple, Optional

_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_BREADCRUMB_LEAD_RE = re.compile(r"^#{1,3}\s+\S")
_WHITESPACE_TOKEN_RE = re.compile(r"\S+")

_SECTION_QUERY_MAX_LEN = 500


class SectionQueryError(ValueError):
    """Raised when a ``--section`` query is syntactically malformed.

    Distinguishes "Claude typed something wrong" (raise) from "Claude typed
    something well-formed that doesn't match this paper" (returns ``None``).
    The CLI surface translates this into a structured ``malformed_section_query``
    payload so the agent has an actionable recovery path.
    """


class SectionChunk(NamedTuple):
    level: int
    title: str
    title_path: tuple[str, ...]
    breadcrumb: str
    body: str
    start_offset: int  # character offset (not bytes) of the header line in the source markdown


def split_sections(markdown: str) -> list[SectionChunk]:
    """Walk ATX-headered markdown, returning one SectionChunk per header.

    - Fenced (``` or ~~~) code blocks are skipped — '#' lines inside them are
      never treated as headers.
    - Non-whitespace content before the first header emits a synthetic
      ``# Abstract`` chunk at index 0 (start_offset=0).
    - A chunk's body ends before the very next header of any level. Each
      chunk owns only its own paragraphs; descendant content lives in the
      descendants' own chunks. The hierarchy is recoverable at read time
      via ``find_hierarchical_section``, which slices the source markdown
      using ``start_offset`` — it does not depend on parent bodies
      containing descendant text. ``body`` has the breadcrumb prepended:
      ``breadcrumb\\n\\n<raw>``.
    """
    lines = markdown.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    header_events: list[tuple[int, int, str]] = []
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADER_RE.match(line)
        if m:
            header_events.append((i, len(m.group(1)), m.group(2).strip()))

    chunks: list[SectionChunk] = []

    first_header_line = header_events[0][0] if header_events else len(lines)
    preamble = "".join(lines[:first_header_line])
    if preamble.strip():
        chunks.append(
            SectionChunk(
                level=1,
                title="Abstract",
                title_path=("Abstract",),
                breadcrumb="# Abstract",
                body=f"# Abstract\n\n{preamble}",
                start_offset=0,
            )
        )

    stack: list[tuple[int, str]] = []
    for idx, (line_idx, level, title) in enumerate(header_events):
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        title_path = tuple(t for _, t in stack)
        breadcrumb = " > ".join(f"{'#' * lvl} {t}" for lvl, t in stack)

        end_line_idx = (
            header_events[idx + 1][0]
            if idx + 1 < len(header_events)
            else len(lines)
        )

        raw_body = "".join(lines[line_idx + 1 : end_line_idx])
        chunks.append(
            SectionChunk(
                level=level,
                title=title,
                title_path=title_path,
                breadcrumb=breadcrumb,
                body=f"{breadcrumb}\n\n{raw_body}",
                start_offset=offsets[line_idx],
            )
        )

    return chunks


def strip_breadcrumb(body: str) -> str:
    """Remove the breadcrumb line and following blank line from a ``SectionChunk.body``.

    Only meant to be called on the output of :func:`split_sections`, where the
    first line is known to be a breadcrumb (either ``# A > ## B`` form or a
    bare ``# Abstract`` synthetic line). If the first line doesn't look like a
    breadcrumb, body is returned unchanged.
    """
    if not body.startswith("#"):
        return body
    first_line, sep, rest = body.partition("\n")
    if " > " in first_line or _BREADCRUMB_LEAD_RE.match(first_line):
        if rest.startswith("\n"):
            return rest[1:]
        return rest
    return body


def sub_chunk(
    text: str,
    max_tokens: int = 350,
    overlap_tokens: int = 20,
    tokenizer_cb: Optional[Callable[[str], list[tuple[int, int]]]] = None,
) -> list[str]:
    """Token-based sliding window that preserves original whitespace.

    ``tokenizer_cb`` returns ``(start, end)`` character offsets of each token
    in ``text``. Chunks are produced by slicing the source string at the
    first/last token of each window — ``text[window[0][0] : window[-1][1]]``
    — so original whitespace, casing, and punctuation are preserved verbatim
    regardless of which tokenizer (whitespace / BPE / WordPiece /
    SentencePiece) is plugged in. This is load-bearing downstream: entity
    extractors return span text as a slice of the input they receive, so
    corrupting the input (e.g. ``" ".join(subword_tokens)``) corrupts every
    extracted entity name.

    The default callback matches non-whitespace runs, giving the same
    tokenization as ``str.split`` but as offsets. Production in
    ``extract_entities.py`` plugs in GLiNER2's fast tokenizer via
    ``return_offsets_mapping=True`` so sub-chunks respect the model's 384-
    token ceiling.

    Returns ``[text]`` unchanged if the token count fits in one window.
    Otherwise slides a window of ``max_tokens`` with step
    ``max_tokens - overlap_tokens`` so adjacent chunks share exactly
    ``overlap_tokens`` tokens.
    """
    if overlap_tokens >= max_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})"
        )

    if tokenizer_cb is not None:
        offsets = tokenizer_cb(text)
    else:
        offsets = [(m.start(), m.end()) for m in _WHITESPACE_TOKEN_RE.finditer(text)]

    if len(offsets) <= max_tokens:
        return [text]

    step = max_tokens - overlap_tokens
    chunks: list[str] = []
    i = 0
    n = len(offsets)
    while i < n:
        window = offsets[i : i + max_tokens]
        chunks.append(text[window[0][0] : window[-1][1]])
        if i + max_tokens >= n:
            break
        i += step
    return chunks


def _validate_section_query(section_query: str) -> tuple[str, ...]:
    """Validate a raw ``--section`` query and return its lowercased parts.

    Raises :class:`SectionQueryError` for malformed input — empty/whitespace
    string, segments with no content (catches ``">"``, ``"A >> B"``,
    ``"> B"``, ``"A >"``), embedded newlines, or excessive length. Well-
    formed queries that simply don't match any section in a given paper
    are NOT this function's concern; the caller signals that with ``None``.
    """
    if not isinstance(section_query, str):
        raise SectionQueryError(
            f"section query must be a string, got {type(section_query).__name__}"
        )

    if "\n" in section_query or "\r" in section_query:
        raise SectionQueryError(
            f"section query contains a newline: {section_query!r}. "
            f"Pass a single-line title or 'Parent > Child' breadcrumb."
        )

    if len(section_query) > _SECTION_QUERY_MAX_LEN:
        raise SectionQueryError(
            f"section query exceeds {_SECTION_QUERY_MAX_LEN} characters "
            f"(got {len(section_query)}). Check for accidental input."
        )

    if not section_query.strip():
        raise SectionQueryError(
            "section query is empty. "
            "Pass a title like 'Method' or a breadcrumb like 'Experiments > Setup'."
        )

    raw_parts = section_query.split(">")
    parts: list[str] = []
    for raw in raw_parts:
        stripped = raw.strip()
        if not stripped:
            raise SectionQueryError(
                f"empty segment in breadcrumb {section_query!r}. "
                f"Use 'Parent > Child' with non-empty segments separated by ' > '."
            )
        parts.append(stripped.lower())
    return tuple(parts)


def find_hierarchical_section(markdown: str, section_query: str) -> Optional[str]:
    """Return the raw markdown slice for the section matching ``section_query``.

    The query is a title (``"Method"``) or a breadcrumb fragment
    (``"Experiments > Setup"``). A chunk matches when its breadcrumb titles
    (lowercased) *end with* the query parts (lowercased). The returned slice
    starts at the matched header line and ends before the next chunk of
    same-or-higher level — so descendants are included.

    Raises :class:`SectionQueryError` if ``section_query`` is syntactically
    malformed. Returns ``None`` for a well-formed query that finds no match.
    """
    query_parts = _validate_section_query(section_query)
    q_len = len(query_parts)

    chunks = split_sections(markdown)
    if not chunks:
        return None

    for idx, chunk in enumerate(chunks):
        if len(chunk.title_path) < q_len:
            continue
        if tuple(t.lower() for t in chunk.title_path[-q_len:]) != query_parts:
            continue

        end_offset = len(markdown)
        for later in chunks[idx + 1 :]:
            if later.level <= chunk.level:
                end_offset = later.start_offset
                break
        return markdown[chunk.start_offset : end_offset]

    return None
