"""Tests for _system/html/latexml_parser.py.

Pure offline: every test parses from strings or the checked-in fixture — no
network, no DB.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import pytest

from _system.html.latexml_parser import FigureDescriptor, ParsedPaper, parse

FIXTURE = Path(__file__).parent / "fixtures" / "latexml_small.html"

# Real ar5iv paper URLs typically have a trailing slash on the per-paper
# directory — important because ``urljoin(base, "x.png")`` without a trailing
# slash replaces the last path segment (dropping the paper id).
BASE_URL = "https://ar5iv.labs.arxiv.org/html/1234.5678/"


@pytest.fixture(scope="module")
def latexml_sample_html() -> str:
    return FIXTURE.read_text()


# --- fixture-based structure ---


def test_fixture_produces_nested_markdown_headers(latexml_sample_html: str) -> None:
    """ltx_section -> '# ', ltx_subsection -> '## ' with stripped numbering."""
    paper = parse(latexml_sample_html, base_url=BASE_URL)
    assert "# Method" in paper.markdown
    assert "## BookIndex Construction" in paper.markdown
    # ltx_tag numbering ("1.") must be stripped from the title.
    assert "# 1. Method" not in paper.markdown


def test_fixture_markdown_usable_by_split_sections(latexml_sample_html: str) -> None:
    """Emitted headers satisfy ``^#{1,3}\\s+`` (consumable by section 05 splitter)."""
    from _system.utils.sections import split_sections

    paper = parse(latexml_sample_html, base_url=BASE_URL)
    chunks = split_sections(paper.markdown)
    titles = [c.title for c in chunks]
    assert "Method" in titles
    assert "BookIndex Construction" in titles


def test_fixture_figure_descriptor_basics(latexml_sample_html: str) -> None:
    """A normal <figure class='ltx_figure'> yields one FigureDescriptor with
    caption, figure_id, and section_context from the enclosing section."""
    paper = parse(latexml_sample_html, base_url=BASE_URL)
    assert len(paper.figures) == 1
    fig = paper.figures[0]
    assert isinstance(fig, FigureDescriptor)
    assert fig.figure_number == 1
    assert fig.figure_id == "S3.F1"
    assert fig.section_context == "Method"
    assert "Overview" in fig.caption
    assert fig.src_url is not None and fig.src_url.endswith("x1.png")
    assert fig.inline_data is None
    assert fig.inline_mime is None


def test_ltx_table_excluded_from_figures_list(latexml_sample_html: str) -> None:
    """<figure class='ltx_table'> is text, not an image — no FigureDescriptor
    and the simple table renders as a GFM markdown table."""
    paper = parse(latexml_sample_html, base_url=BASE_URL)
    assert len(paper.figures) == 1  # the ltx_table is not counted
    assert "| Name | Value |" in paper.markdown
    assert "| --- | --- |" in paper.markdown
    assert "| Foo | 1 |" in paper.markdown


# --- figure image handling ---


def test_empty_src_emits_placeholder_comment() -> None:
    """Empty <img src> → markdown placeholder comment and null URL fields."""
    html = (
        '<html><body><figure class="ltx_figure" id="F1">'
        '<img src=""><figcaption>Figure 1: Alone</figcaption>'
        "</figure></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert '<!-- Figure 1: "Figure 1: Alone" — no image in HTML' in paper.markdown
    assert len(paper.figures) == 1
    fig = paper.figures[0]
    assert fig.src_url is None
    assert fig.inline_data is None
    assert fig.inline_mime is None


def test_missing_img_emits_placeholder_comment() -> None:
    """Figure with no <img> descendant → placeholder, all url fields None."""
    html = (
        '<html><body><figure class="ltx_figure" id="F1">'
        "<figcaption>Figure 1: Bare</figcaption>"
        "</figure></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert "<!-- Figure 1:" in paper.markdown
    fig = paper.figures[0]
    assert fig.src_url is None and fig.inline_data is None


def test_data_uri_figure_decodes_inline_bytes() -> None:
    """<img src='data:image/png;base64,...'> -> inline_data set, src_url None."""
    payload = base64.b64encode(b"hello-bytes").decode()
    html = (
        '<html><body><figure class="ltx_figure" id="F1">'
        f'<img src="data:image/png;base64,{payload}">'
        "<figcaption>Figure 1: Data</figcaption>"
        "</figure></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    fig = paper.figures[0]
    assert fig.src_url is None
    assert fig.inline_data == b"hello-bytes"
    assert fig.inline_mime == "image/png"
    assert "![Figure 1: Figure 1: Data](figure:1)" in paper.markdown


def test_data_uri_non_base64_falls_through_to_placeholder() -> None:
    """Non-base64 data URIs are treated as a decode failure (placeholder)."""
    html = (
        '<html><body><figure class="ltx_figure" id="F1">'
        '<img src="data:text/plain,justsometext">'
        "<figcaption>Figure 1: Raw data</figcaption>"
        "</figure></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    fig = paper.figures[0]
    assert fig.src_url is None
    assert fig.inline_data is None
    assert fig.inline_mime is None
    assert "<!-- Figure 1:" in paper.markdown


def test_normal_src_is_urljoined_to_base() -> None:
    html = (
        '<html><body><figure class="ltx_figure" id="F1">'
        '<img src="assets/figA.png"><figcaption>Figure 1: A</figcaption>'
        "</figure></body></html>"
    )
    paper = parse(html, base_url="https://ar5iv.labs.arxiv.org/html/2512.03413/")
    fig = paper.figures[0]
    assert fig.src_url == "https://ar5iv.labs.arxiv.org/html/2512.03413/assets/figA.png"


def test_urljoin_without_trailing_slash_drops_last_segment() -> None:
    """Documented gotcha: a base_url whose path does NOT end with '/' causes
    urljoin to treat the last segment as a filename and replace it. Section 08
    must pass trailing-slash URLs for ar5iv paper directories."""
    html = (
        '<html><body><figure class="ltx_figure" id="F1">'
        '<img src="assets/figA.png"><figcaption>Figure 1: A</figcaption>'
        "</figure></body></html>"
    )
    paper = parse(html, base_url="https://ar5iv.labs.arxiv.org/html/2512.03413")
    # The '2512.03413' segment is dropped — this is stdlib urljoin behavior.
    assert paper.figures[0].src_url == "https://ar5iv.labs.arxiv.org/html/assets/figA.png"


# --- math handling ---


def test_inline_math_uses_single_dollar() -> None:
    html = '<html><body><p>use <math alttext="\\alpha">&#945;</math> here</p></body></html>'
    paper = parse(html, base_url="https://example.com/")
    assert r"$\alpha$" in paper.markdown


def test_display_math_uses_double_dollar() -> None:
    html = '<html><body><math display="block" alttext="E = mc^2">E = mc</math></body></html>'
    paper = parse(html, base_url="https://example.com/")
    assert "$$E = mc^2$$" in paper.markdown


def test_math_without_alttext_escapes_dollars_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing alttext -> emit visible text with $ escaped as \\$; log warning."""
    html = "<html><body><p><math>$100 total</math></p></body></html>"
    # ``lodestone`` is the Lodestone logger namespace and has ``propagate=False``,
    # so caplog's root handler never sees records unless we attach to it directly.
    logger = logging.getLogger("lodestone.html.latexml_parser")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="lodestone.html.latexml_parser"):
            paper = parse(html, base_url="https://example.com/")
    finally:
        logger.removeHandler(caplog.handler)
    assert r"\$100 total" in paper.markdown
    assert any("alttext" in rec.getMessage() for rec in caplog.records)


# --- captions / display_number ---


def test_display_number_parsed_from_caption() -> None:
    """'Figure 3a: ...' -> display_number == '3a'."""
    html = (
        '<html><body><figure class="ltx_figure" id="F1">'
        '<img src="x.png"><figcaption>Figure 3a: BookIndex overview</figcaption>'
        "</figure></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert paper.figures[0].display_number == "3a"


def test_display_number_none_when_no_leading_label() -> None:
    """A plain caption without a trailing delimiter yields display_number=None."""
    html = (
        '<html><body><figure class="ltx_figure" id="F1">'
        '<img src="x.png"><figcaption>Overview</figcaption>'
        "</figure></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert paper.figures[0].display_number is None


# --- non-simple tables ---


def test_non_simple_table_preserved_as_html() -> None:
    """A table with rowspan is not convertable to GFM -> preserve HTML."""
    html = (
        '<html><body><figure class="ltx_table"><table class="ltx_tabular">'
        '<tr><th>A</th><th rowspan="2">B</th></tr>'
        "<tr><td>1</td></tr>"
        "</table></figure></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    # Raw HTML preserved
    assert "<table" in paper.markdown
    assert 'rowspan="2"' in paper.markdown
    # Never converted to GFM header divider
    assert "| --- | --- |" not in paper.markdown


# --- parser hardening ---


def test_parser_hardened_against_xxe() -> None:
    """XXE payload must not resolve — no /etc/passwd in the output."""
    html = (
        '<!DOCTYPE x [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
        "<html><body><p>&xxe;</p></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    # parse() returns without raising
    assert isinstance(paper, ParsedPaper)
    # The entity must not have been resolved: no file content leaked.
    assert "/etc/passwd" not in paper.markdown
    assert "root:" not in paper.markdown


def test_parser_flags_set_on_hardened_parser() -> None:
    """Regression: the parse function instantiates lxml's HTMLParser with the
    hardening flags. If this test starts failing, the XXE test above may be a
    silent no-op.

    Note: ``resolve_entities`` is an XMLParser-only kwarg; lxml's HTMLParser
    rejects it. HTML mode does not resolve external entities, so ``no_network``
    + ``huge_tree`` cover the XXE/billion-laughs surface.
    """
    import inspect

    from _system.html import latexml_parser

    src = inspect.getsource(latexml_parser.parse)
    assert "no_network=True" in src
    assert "huge_tree=False" in src


# --- sub-figure flat numbering ---


def test_subfigure_panels_get_flat_figure_numbers() -> None:
    """Two adjacent ltx_figure_panel siblings get ordinals 1 and 2 — flat."""
    html = (
        "<html><body>"
        '<figure class="ltx_figure ltx_figure_panel" id="F1a">'
        '<img src="a.png"><figcaption>Figure 1a</figcaption></figure>'
        '<figure class="ltx_figure ltx_figure_panel" id="F1b">'
        '<img src="b.png"><figcaption>Figure 1b</figcaption></figure>'
        "</body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert [f.figure_number for f in paper.figures] == [1, 2]
    assert [f.figure_id for f in paper.figures] == ["F1a", "F1b"]
    # Plan explicitly specifies flat numeric ordinals here, not "1a"/"1b".
    # The ``display_number`` field is best-effort from the plan-specified regex
    # and its value on EOS-terminated captions is not part of the contract.


# --- API shape ---


def test_return_type_is_named_tuple_with_two_fields() -> None:
    """ParsedPaper and FigureDescriptor are NamedTuples (hashable-adjacent)."""
    paper = parse(
        '<html><body><p>hi</p></body></html>', base_url="https://example.com/"
    )
    assert isinstance(paper, tuple)
    assert paper._fields == ("markdown", "figures")


def test_figure_descriptor_is_named_tuple() -> None:
    """FigureDescriptor is a NamedTuple with the specified field layout."""
    html = (
        '<html><body><figure class="ltx_figure" id="F1">'
        '<img src="x.png"><figcaption>Figure 1: X</figcaption>'
        "</figure></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    fd = paper.figures[0]
    assert isinstance(fd, tuple)
    assert fd._fields == (
        "figure_number",
        "display_number",
        "figure_id",
        "caption",
        "section_context",
        "src_url",
        "inline_data",
        "inline_mime",
    )


# --- section_context nesting ---


def test_section_context_joined_with_space_arrow_space() -> None:
    """A figure inside a nested subsection gets 'Parent > Child' context."""
    html = (
        "<html><body>"
        '<section class="ltx_section">'
        '<h2 class="ltx_title">Method</h2>'
        '<section class="ltx_subsection">'
        '<h3 class="ltx_title">BookIndex Construction</h3>'
        '<figure class="ltx_figure" id="NF">'
        '<img src="n.png"><figcaption>Figure 1: Nested</figcaption></figure>'
        "</section></section></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    fig = paper.figures[0]
    assert fig.section_context == "Method > BookIndex Construction"


def test_preamble_figure_section_context_is_empty_string() -> None:
    """Figure before any section header -> section_context == ''."""
    html = (
        "<html><body>"
        '<figure class="ltx_figure" id="PRE">'
        '<img src="p.png"><figcaption>Figure 1: Pre</figcaption></figure>'
        "</body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert paper.figures[0].section_context == ""


# --- inline text-only elements (cite, a.ltx_ref) ---


def test_cite_emits_text_only() -> None:
    """<cite> drops the anchor target, keeping only the visible text."""
    html = (
        '<html><body><p>See <cite>Smith et al. 2024</cite> '
        "for details.</p></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert "See Smith et al. 2024 for details." in paper.markdown
    assert "<cite>" not in paper.markdown


def test_a_ltx_ref_emits_text_only() -> None:
    """<a class='ltx_ref' href='...'> keeps the link text, drops the href."""
    html = (
        '<html><body><p>Refer to '
        '<a class="ltx_ref" href="#sec3">Section 3</a> above.</p></body></html>'
    )
    paper = parse(html, base_url="https://example.com/")
    assert "Refer to Section 3 above." in paper.markdown
    assert "href" not in paper.markdown
    assert "#sec3" not in paper.markdown


# --- code / pre ---


def test_code_inline_wrapped_in_backticks() -> None:
    html = "<html><body><p>Call <code>foo()</code> first.</p></body></html>"
    paper = parse(html, base_url="https://example.com/")
    assert "Call `foo()` first." in paper.markdown


def test_pre_wrapped_in_triple_backtick_fence() -> None:
    html = (
        "<html><body><pre>line one\nline two</pre></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert "```\nline one\nline two\n```" in paper.markdown


# --- lists ---


def test_ul_renders_as_bullet_markdown() -> None:
    html = (
        "<html><body><ul><li>alpha</li><li>beta</li><li>gamma</li></ul></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert "- alpha" in paper.markdown
    assert "- beta" in paper.markdown
    assert "- gamma" in paper.markdown


def test_ol_renders_as_numbered_markdown() -> None:
    html = (
        "<html><body><ol><li>first</li><li>second</li><li>third</li></ol></body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    assert "1. first" in paper.markdown
    assert "2. second" in paper.markdown
    assert "3. third" in paper.markdown


# --- caption bleed regression ---


def test_outer_figure_without_caption_does_not_steal_sub_panel_caption() -> None:
    """H1 regression: a parent <figure class='ltx_figure'> with no direct
    figcaption must NOT take the first descendant figcaption belonging to a
    nested sub-panel."""
    html = (
        "<html><body>"
        '<figure class="ltx_figure" id="OUTER">'
        '<img src="outer.png">'
        '<figure class="ltx_figure ltx_figure_panel" id="INNER">'
        '<figcaption>Inner panel caption</figcaption>'
        "</figure>"
        "</figure>"
        "</body></html>"
    )
    paper = parse(html, base_url="https://example.com/")
    outer = next(f for f in paper.figures if f.figure_id == "OUTER")
    assert outer.caption == ""


# --- module import hygiene ---


def test_module_does_not_import_heavy_deps() -> None:
    """Parser is pure: no ML, no DB, no HTTP clients.

    sqlite3 / httpx / sentence_transformers / gliner2 / torch may already be
    in ``sys.modules`` from other tests in the session; we can't reliably
    check post-hoc. A source-level grep is a pragmatic smoke test that the
    module itself does not reach for them.
    """
    import inspect

    from _system.html import latexml_parser

    src = inspect.getsource(latexml_parser)
    assert "import httpx" not in src
    assert "import sqlite3" not in src
    assert "import subprocess" not in src
    assert "sentence_transformers" not in src
    assert "gliner2" not in src
