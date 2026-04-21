"""Tests for _system/utils/sections.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from _system.utils.sections import (
    SectionChunk,
    find_hierarchical_section,
    split_sections,
    strip_breadcrumb,
    sub_chunk,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_three_level.md"


@pytest.fixture(scope="module")
def sample_markdown() -> str:
    return FIXTURE.read_text()


@pytest.fixture(scope="module")
def sample_chunks(sample_markdown: str) -> list[SectionChunk]:
    return split_sections(sample_markdown)


# --- split_sections ---


def test_split_sections_produces_ordered_chunks(sample_chunks: list[SectionChunk]) -> None:
    """3-level markdown yields synthetic Abstract + each header as a chunk in DOM order."""
    titles = [c.title for c in sample_chunks]
    assert titles == [
        "Abstract",
        "Introduction",
        "Method",
        "Gradient-based Entity Resolution",
        "Optimization",
        "Scaling",
        "Experiments",
        "Setup",
        "Results",
        "Discussion",
        "Setup",
    ]


def test_breadcrumb_full_ancestry_with_marker_prefixes(sample_chunks: list[SectionChunk]) -> None:
    """Breadcrumb joins the ancestry with ' > ' and prefixes each entry with its '#'s."""
    by_title = {c.breadcrumb: c for c in sample_chunks if c.title == "Optimization"}
    # Only one Optimization in the fixture.
    opt = next(iter(by_title.values()))
    assert opt.breadcrumb == "# Method > ## Gradient-based Entity Resolution > ### Optimization"


def test_breadcrumb_prepended_to_body(sample_chunks: list[SectionChunk]) -> None:
    """First line of body is the breadcrumb, followed by a blank line, then original text."""
    scaling = next(c for c in sample_chunks if c.title == "Scaling")
    assert scaling.body.startswith(f"{scaling.breadcrumb}\n\n")
    assert "Scaling discussion." in scaling.body


def test_body_ends_before_next_same_or_higher_header(sample_chunks: list[SectionChunk]) -> None:
    """A chunk's body stops at the next header of same-or-higher level; deeper headers stay inside."""
    method = next(c for c in sample_chunks if c.title == "Method" and c.level == 1)
    # Method body includes its children because they are deeper.
    assert "## Gradient-based Entity Resolution" in method.body
    assert "## Scaling" in method.body
    # Body must NOT reach the next top-level section.
    assert "# Experiments" not in method.body
    assert "# Discussion" not in method.body


def test_fenced_backtick_code_blocks_ignored(sample_chunks: list[SectionChunk]) -> None:
    """The '# Not a real header' inside a ``` block does not emit a chunk."""
    titles = [c.title for c in sample_chunks]
    assert "Not a real header" not in titles


def test_fenced_tilde_code_blocks_ignored(sample_chunks: list[SectionChunk]) -> None:
    """Headers inside ~~~ fences are ignored."""
    titles = [c.title for c in sample_chunks]
    assert "Also not a real header" not in titles


def test_synthetic_abstract_emitted_for_preamble(sample_chunks: list[SectionChunk]) -> None:
    """Non-whitespace content before the first header triggers a synthetic '# Abstract' chunk at index 0."""
    abstract = sample_chunks[0]
    assert abstract.title == "Abstract"
    assert abstract.breadcrumb == "# Abstract"
    assert abstract.start_offset == 0
    assert "preamble paragraph" in abstract.body
    # The breadcrumb and body are separated by exactly '\n\n' (strip_breadcrumb relies on this).
    assert abstract.body.startswith("# Abstract\n\n")


def test_no_synthetic_abstract_when_no_preamble() -> None:
    """Markdown that starts directly with a header gets no synthetic Abstract."""
    md = "# Title\n\nBody.\n"
    chunks = split_sections(md)
    assert [c.title for c in chunks] == ["Title"]


def test_split_sections_empty_returns_empty() -> None:
    """Empty markdown produces no chunks."""
    assert split_sections("") == []
    assert split_sections("   \n\n  \n") == []


def test_split_sections_only_preamble_emits_abstract_only() -> None:
    """Markdown with content but no headers emits only the synthetic Abstract chunk."""
    md = "Just some prose.\nNo headers at all.\n"
    chunks = split_sections(md)
    assert len(chunks) == 1
    assert chunks[0].title == "Abstract"
    assert "Just some prose." in chunks[0].body


def test_start_offset_points_to_header_line(sample_markdown: str, sample_chunks: list[SectionChunk]) -> None:
    """start_offset is the byte offset of the header line in the original markdown."""
    for c in sample_chunks:
        if c.title == "Abstract" and c.breadcrumb == "# Abstract":
            # Synthetic chunk — offset is 0, source begins with preamble text, not a '#'.
            assert c.start_offset == 0
            continue
        hashes = "#" * c.level
        assert sample_markdown[c.start_offset:].startswith(f"{hashes} {c.title}")


# --- strip_breadcrumb ---


def test_strip_breadcrumb_removes_breadcrumb_first_line() -> None:
    """A 'a > b > c' breadcrumb line plus its blank follower is removed."""
    body = "# Method > ## Setup\n\nreal body text\nmore text\n"
    assert strip_breadcrumb(body) == "real body text\nmore text\n"


def test_strip_breadcrumb_leaves_plain_body_unchanged() -> None:
    """First line is plain text — nothing changes."""
    body = "no leading hash here\nsecond line\n"
    assert strip_breadcrumb(body) == body


def test_strip_breadcrumb_strips_single_level_synthetic() -> None:
    """'# Abstract\\n\\n...' (single-header synthetic breadcrumb) is also stripped."""
    body = "# Abstract\n\nsome abstract text\n"
    assert strip_breadcrumb(body) == "some abstract text\n"


# --- sub_chunk ---


def test_sub_chunk_respects_max_tokens() -> None:
    """Every chunk has at most max_tokens tokens (default 350)."""
    text = " ".join(f"t{i}" for i in range(1200))
    chunks = sub_chunk(text, max_tokens=100, overlap_tokens=10)
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert len(c.split()) <= 100


def test_sub_chunk_produces_overlap() -> None:
    """Last overlap_tokens tokens of chunk[i] equal first overlap_tokens tokens of chunk[i+1]."""
    text = " ".join(f"t{i}" for i in range(500))
    chunks = sub_chunk(text, max_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        tail = chunks[i].split()[-20:]
        head = chunks[i + 1].split()[:20]
        assert tail == head


def test_sub_chunk_short_text_returned_unchanged() -> None:
    """Text shorter than max_tokens is returned as a single-element list."""
    text = "only a few tokens"
    assert sub_chunk(text, max_tokens=100, overlap_tokens=10) == [text]


def test_sub_chunk_uses_custom_tokenizer_cb() -> None:
    """A caller-supplied tokenizer is preferred over the default."""
    calls: list[str] = []

    def my_tok(s: str) -> list[str]:
        calls.append(s)
        return list(s)  # one-char tokens

    text = "x" * 50
    chunks = sub_chunk(text, max_tokens=10, overlap_tokens=2, tokenizer_cb=my_tok)
    assert calls, "tokenizer_cb was not called"
    for c in chunks:
        assert len(c.split()) <= 10


def test_sub_chunk_pathological_overlap_raises() -> None:
    """overlap_tokens >= max_tokens raises ValueError."""
    with pytest.raises(ValueError):
        sub_chunk("a b c", max_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        sub_chunk("a b c", max_tokens=10, overlap_tokens=20)


def test_sub_chunk_zero_overlap() -> None:
    """overlap_tokens=0 produces chunks with no shared tokens between adjacent chunks."""
    text = " ".join(f"t{i}" for i in range(50))
    chunks = sub_chunk(text, max_tokens=10, overlap_tokens=0)
    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        tail = chunks[i].split()[-1:]
        head = chunks[i + 1].split()[:1]
        assert tail != head


# --- find_hierarchical_section ---


def test_find_hierarchical_section_by_title(sample_markdown: str) -> None:
    """find_hierarchical_section('Method') returns Method through its last descendant, stopping at Experiments."""
    slice_ = find_hierarchical_section(sample_markdown, "Method")
    assert slice_ is not None
    assert slice_.startswith("# Method")
    assert "## Gradient-based Entity Resolution" in slice_
    assert "### Optimization" in slice_
    assert "## Scaling" in slice_
    assert "# Experiments" not in slice_


def test_find_hierarchical_section_disambiguates_duplicate_child(sample_markdown: str) -> None:
    """'Experiments > Setup' matches only the Setup under Experiments, not under Discussion."""
    slice_ = find_hierarchical_section(sample_markdown, "Experiments > Setup")
    assert slice_ is not None
    assert slice_.startswith("## Setup")
    assert "Experiments setup description." in slice_
    assert "Discussion setup description." not in slice_


def test_find_hierarchical_section_missing_returns_none(sample_markdown: str) -> None:
    """Non-existent section returns None."""
    assert find_hierarchical_section(sample_markdown, "ZZZ Nope") is None


def test_find_hierarchical_section_case_insensitive(sample_markdown: str) -> None:
    """Title query is matched case-insensitively."""
    slice_ = find_hierarchical_section(sample_markdown, "method")
    assert slice_ is not None
    assert slice_.startswith("# Method")


# --- Cross-cutting ---


def test_strip_breadcrumb_on_split_output_removes_ancestry(sample_chunks: list[SectionChunk]) -> None:
    """Feeding SectionChunk.body through strip_breadcrumb removes the ' > ' ancestry line."""
    for c in sample_chunks:
        stripped = strip_breadcrumb(c.body)
        # The breadcrumb itself (with its ' > ' chain, if any) must not appear as the first line.
        if " > " in c.breadcrumb:
            first_line = stripped.split("\n", 1)[0]
            assert " > " not in first_line


def test_module_does_not_pull_heavy_deps() -> None:
    """Importing _system.utils.sections must not pull gliner2, torch, or sqlite3."""
    import sys

    # Module is already imported at the top; just verify no heavy deps came with it.
    for mod in ("gliner2", "torch", "sentence_transformers"):
        assert mod not in sys.modules, f"{mod} leaked into import path"
