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
    - A chunk's body ends before the next header whose level is <= its own.
      Deeper children remain inside the parent's body (and also emit their
      own chunks). ``body`` has the breadcrumb prepended: ``breadcrumb\\n\\n<raw>``.
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

        end_line_idx = len(lines)
        for later_idx in range(idx + 1, len(header_events)):
            if header_events[later_idx][1] <= level:
                end_line_idx = header_events[later_idx][0]
                break

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
    tokenizer_cb: Optional[Callable[[str], list[str]]] = None,
) -> list[str]:
    """Token-based sliding window.

    Returns ``[text]`` unchanged if it fits in one window. Otherwise slides a
    window of ``max_tokens`` with step ``max_tokens - overlap_tokens`` so that
    adjacent chunks share exactly ``overlap_tokens`` tokens.

    The default tokenizer is ``str.split``; the production caller in
    ``extract_entities.py`` passes GLiNER2's tokenizer via ``tokenizer_cb``.
    Note that chunks are reconstructed via ``" ".join(window)``, so original
    whitespace is not preserved — this is acceptable because chunks are never
    written back to disk (GLiNER2 re-tokenizes internally).
    """
    if overlap_tokens >= max_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < max_tokens ({max_tokens})"
        )

    tokenize = tokenizer_cb if tokenizer_cb is not None else str.split
    tokens = tokenize(text)

    if len(tokens) <= max_tokens:
        return [text]

    step = max_tokens - overlap_tokens
    chunks: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        chunks.append(" ".join(tokens[i : i + max_tokens]))
        if i + max_tokens >= n:
            break
        i += step
    return chunks


def find_hierarchical_section(markdown: str, section_query: str) -> Optional[str]:
    """Return the raw markdown slice for the section matching ``section_query``.

    The query is a title (``"Method"``) or a breadcrumb fragment
    (``"Experiments > Setup"``). A chunk matches when its breadcrumb titles
    (lowercased) *end with* the query parts (lowercased). The returned slice
    starts at the matched header line and ends before the next chunk of
    same-or-higher level — so descendants are included.
    """
    chunks = split_sections(markdown)
    if not chunks:
        return None

    query_parts = tuple(p.strip().lower() for p in section_query.split(">"))
    q_len = len(query_parts)

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
