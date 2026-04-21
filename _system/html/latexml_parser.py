"""LaTeXML / ar5iv HTML -> markdown + figure descriptors.

Pure / offline. No HTTP, no DB, no subprocess. The only "network-ish" op is
``urljoin(base_url, src)``, which is pure string work. Any download of
``src_url`` happens downstream in ``fetch_paper.py`` (section 08).
"""
from __future__ import annotations

import base64
import binascii
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import NamedTuple, Optional
from urllib.parse import urljoin

import lxml.html

_logger = logging.getLogger(__name__)


class LtxClass(StrEnum):
    """LaTeXML / ar5iv CSS class names recognized by the parser."""

    SECTION = "ltx_section"
    SUBSECTION = "ltx_subsection"
    SUBSUBSECTION = "ltx_subsubsection"
    FIGURE = "ltx_figure"
    TABLE = "ltx_table"
    TABULAR = "ltx_tabular"
    TITLE = "ltx_title"
    CAPTION = "ltx_caption"
    TAG = "ltx_tag"
    REF = "ltx_ref"


class FigureDescriptor(NamedTuple):
    figure_number: int
    display_number: Optional[str]
    figure_id: str
    caption: str
    section_context: str
    src_url: Optional[str]
    inline_data: Optional[bytes]
    inline_mime: Optional[str]


class ParsedPaper(NamedTuple):
    markdown: str
    figures: list[FigureDescriptor]


_SECTION_DEPTHS: dict[str, int] = {
    LtxClass.SECTION: 1,
    LtxClass.SUBSECTION: 2,
    LtxClass.SUBSUBSECTION: 3,
}

_DISPLAY_NUMBER_RE = re.compile(
    # Plan-specified shape. Best-effort; imperfect in the usual ways — e.g.
    # captions without a trailing delimiter do not match, "Overview diagram"
    # captures "Overview", and "Appendix A:" captures "Appendix".
    r"^\s*(?:Figure|Fig\.?|Table)?\s*(\w+)[:\.\s]",
    re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")
_BLANK_RUN_RE = re.compile(r"\n{3,}")

_HEADER_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = {"head", "script", "style", "noscript"}


def parse(html: str, base_url: str) -> ParsedPaper:
    """Parse LaTeXML / ar5iv HTML into markdown + figure descriptors. Pure / offline.

    The parser is instantiated with ``no_network=True`` and ``huge_tree=False``,
    which together block external DTD / entity network fetches and billion-laughs
    -style entity expansion. lxml's HTML parser does not accept a
    ``resolve_entities`` kwarg (that's for the XML parser), and it does not
    resolve external entities in HTML mode anyway — so these two flags are the
    full XXE/DoS hardening surface for this module.
    """
    parser = lxml.html.HTMLParser(no_network=True, huge_tree=False)
    root = lxml.html.fromstring(html, parser=parser)
    state = _State(base_url=base_url)
    bodies = root.xpath(".//body")
    target = bodies[0] if bodies else root
    markdown = _convert(target, state)
    markdown = _postprocess(markdown)
    return ParsedPaper(markdown=markdown, figures=state.figures)


@dataclass
class _State:
    base_url: str
    figures: list[FigureDescriptor] = field(default_factory=list)
    figure_counter: int = 0
    section_stack: list[str] = field(default_factory=list)


def _classes(elem) -> set[str]:
    return set((elem.get("class") or "").split())


def _convert(elem, state: _State) -> str:
    tag = elem.tag
    if not isinstance(tag, str):
        return ""

    if tag in _SKIP_TAGS:
        return ""

    cls = _classes(elem)

    if tag == "section":
        for cls_key, depth in _SECTION_DEPTHS.items():
            if cls_key in cls:
                return _convert_section(elem, state, depth)

    if tag == "figure":
        if LtxClass.TABLE in cls:
            return _convert_table_figure(elem, state)
        if LtxClass.FIGURE in cls:
            return _convert_figure(elem, state)

    if tag == "table" and LtxClass.TABULAR in cls:
        return _convert_table(elem)

    if tag == "math":
        return _convert_math(elem)

    if tag == "cite":
        return _inline_text_only(elem)
    if tag == "a" and LtxClass.REF in cls:
        return _inline_text_only(elem)

    if tag == "code":
        return _convert_code(elem)
    if tag == "pre":
        return _convert_pre(elem)

    if tag == "ul":
        return _convert_list(elem, state, ordered=False)
    if tag == "ol":
        return _convert_list(elem, state, ordered=True)

    if tag == "p":
        return _convert_paragraph(elem, state)

    if tag in _HEADER_TAGS:
        # Stray <hN> outside a recognized section — skip; the section walker
        # handles the real ones. The plan says to trust <section> nesting over
        # <hN> tag names.
        return ""

    return _default_children(elem, state)


def _default_children(elem, state: _State) -> str:
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_convert(child, state))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _convert_section(elem, state: _State, depth: int) -> str:
    title_elem = _find_section_title(elem)
    title = _extract_title_text(title_elem) if title_elem is not None else ""

    header = f"{'#' * depth} {title}\n\n" if title else ""

    state.section_stack.append(title)
    try:
        body_parts: list[str] = [header]
        if elem.text:
            body_parts.append(elem.text)
        for child in elem:
            if child is title_elem:
                if child.tail:
                    body_parts.append(child.tail)
                continue
            body_parts.append(_convert(child, state))
            if child.tail:
                body_parts.append(child.tail)
        return "\n\n" + "".join(body_parts) + "\n\n"
    finally:
        state.section_stack.pop()


def _find_section_title(section_elem):
    """Return the first direct <hN class='ltx_title'> child of the section."""
    for child in section_elem:
        if not isinstance(child.tag, str):
            continue
        if child.tag.lower() in _HEADER_TAGS and LtxClass.TITLE in _classes(child):
            return child
    return None


def _extract_title_text(title_elem) -> str:
    """Extract the title prose, stripping <span class='ltx_tag'> numbering."""
    parts: list[str] = []
    if title_elem.text:
        parts.append(title_elem.text)
    for child in title_elem:
        if isinstance(child.tag, str) and LtxClass.TAG in _classes(child):
            # Skip numbering content; keep tail
            pass
        elif isinstance(child.tag, str):
            parts.append(child.text_content())
        if child.tail:
            parts.append(child.tail)
    return _WS_RE.sub(" ", "".join(parts)).strip()


def _convert_figure(elem, state: _State) -> str:
    state.figure_counter += 1
    fig_num = state.figure_counter
    figure_id = elem.get("id") or ""
    caption = _figure_caption(elem)
    display_number = _parse_display_number(caption)

    img = elem.find(".//img")
    src = (img.get("src") or "") if img is not None else ""

    section_context = " > ".join(t for t in state.section_stack if t)

    src_url: Optional[str] = None
    inline_data: Optional[bytes] = None
    inline_mime: Optional[str] = None

    placeholder = (
        f'<!-- Figure {fig_num}: "{caption}" '
        f"— no image in HTML, see page images -->"
    )
    image_ref = f"![Figure {fig_num}: {caption}](figure:{fig_num})"

    if not src:
        markdown = placeholder
    elif src.startswith("data:"):
        parsed = _parse_data_uri(src)
        if parsed is None:
            _logger.warning(
                "figure %d: unrecognized or non-base64 data URI; "
                "emitting placeholder",
                fig_num,
            )
            markdown = placeholder
        else:
            mime, payload = parsed
            try:
                inline_data = base64.b64decode(payload, validate=True)
                inline_mime = mime
                markdown = image_ref
            except (binascii.Error, ValueError) as exc:
                _logger.warning(
                    "figure %d: base64 decode failed (%s); "
                    "emitting placeholder",
                    fig_num,
                    exc,
                )
                inline_data = None
                inline_mime = None
                markdown = placeholder
    else:
        src_url = urljoin(state.base_url, src)
        markdown = image_ref

    state.figures.append(
        FigureDescriptor(
            figure_number=fig_num,
            display_number=display_number,
            figure_id=figure_id,
            caption=caption,
            section_context=section_context,
            src_url=src_url,
            inline_data=inline_data,
            inline_mime=inline_mime,
        )
    )

    return "\n\n" + markdown + "\n\n"


def _parse_data_uri(src: str) -> Optional[tuple[Optional[str], str]]:
    """Parse a ``data:[<mime>][;base64],<payload>`` URI.

    Returns ``(mime_or_none, payload)`` if the URI is base64-encoded, else None.
    """
    if not src.startswith("data:"):
        return None
    body = src[5:]
    if "," not in body:
        return None
    header, _, payload = body.partition(",")
    parts = header.split(";")
    mime = parts[0] if parts[0] else None
    if "base64" not in parts[1:]:
        return None
    return mime, payload


def _figure_caption(elem) -> str:
    """Prefer direct-child <figcaption class='ltx_caption'>, then any direct
    <figcaption>, else ''.

    Restricted to DIRECT children so a nested sub-panel's figcaption cannot
    bleed up to become the parent figure's caption.
    """
    ltx_caption = None
    any_caption = None
    for child in elem:
        if not isinstance(child.tag, str):
            continue
        if child.tag != "figcaption":
            continue
        if any_caption is None:
            any_caption = child
        if LtxClass.CAPTION in _classes(child) and ltx_caption is None:
            ltx_caption = child
    caption_elem = ltx_caption if ltx_caption is not None else any_caption
    if caption_elem is None:
        return ""
    return _WS_RE.sub(" ", caption_elem.text_content()).strip()


def _parse_display_number(caption: str) -> Optional[str]:
    if not caption:
        return None
    m = _DISPLAY_NUMBER_RE.match(caption)
    if not m:
        return None
    return m.group(1)


def _convert_table_figure(elem, state: _State) -> str:
    """Render an <figure class='ltx_table'>: tabular + caption, no FigureDescriptor."""
    parts: list[str] = []
    caption_text = ""
    for child in elem:
        if not isinstance(child.tag, str):
            continue
        if child.tag == "figcaption":
            caption_text = _WS_RE.sub(" ", child.text_content()).strip()
            continue
        parts.append(_convert(child, state))
    body = "".join(parts).strip()
    if caption_text and body:
        out = f"{body}\n\n**{caption_text}**"
    elif caption_text:
        out = f"**{caption_text}**"
    else:
        out = body
    return "\n\n" + out + "\n\n"


def _convert_table(elem) -> str:
    rows = _extract_rows(elem)
    if not rows:
        return ""
    if _is_simple(rows):
        return "\n\n" + _render_markdown_table(rows) + "\n\n"
    return "\n\n" + _render_html_table(elem) + "\n\n"


def _extract_rows(table) -> list[list]:
    rows: list[list] = []
    for tr in table.iter("tr"):
        cells = [c for c in tr if isinstance(c.tag, str) and c.tag in {"td", "th"}]
        rows.append(cells)
    return rows


def _is_simple(rows: list[list]) -> bool:
    if not rows:
        return False
    width = len(rows[0])
    if width == 0:
        return False
    for row in rows:
        if len(row) != width:
            return False
        for cell in row:
            rs = cell.get("rowspan")
            cs = cell.get("colspan")
            if rs not in (None, "", "1"):
                return False
            if cs not in (None, "", "1"):
                return False
            if cell.find(".//table") is not None:
                return False
            if cell.find(".//img") is not None:
                return False
    return True


def _render_markdown_table(rows: list[list]) -> str:
    def cell_text(c) -> str:
        return _WS_RE.sub(" ", c.text_content()).strip().replace("|", "\\|")

    lines: list[str] = []
    header = [cell_text(c) for c in rows[0]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell_text(c) for c in row) + " |")
    return "\n".join(lines)


def _render_html_table(elem) -> str:
    return lxml.html.tostring(elem, encoding="unicode", method="html")


def _convert_math(elem) -> str:
    alttext = elem.get("alttext")
    display = elem.get("display") == "block"
    delim = "$$" if display else "$"
    if alttext is not None and alttext != "":
        return f"{delim}{alttext}{delim}"
    _logger.warning("math element missing alttext; falling back to visible text")
    visible = elem.text_content() or ""
    return visible.replace("$", r"\$")


def _inline_text_only(elem) -> str:
    """<cite>, <a class='ltx_ref'> — emit visible text, drop link target."""
    return elem.text_content() or ""


def _convert_code(elem) -> str:
    text = elem.text_content() or ""
    return f"`{text}`"


def _convert_pre(elem) -> str:
    text = elem.text_content() or ""
    return f"\n\n```\n{text}\n```\n\n"


def _convert_list(elem, state: _State, ordered: bool) -> str:
    lines: list[str] = []
    i = 1
    for child in elem:
        if not isinstance(child.tag, str) or child.tag != "li":
            continue
        content = _default_children(child, state)
        content = _WS_RE.sub(" ", content).strip()
        marker = f"{i}." if ordered else "-"
        lines.append(f"{marker} {content}")
        i += 1
    return "\n\n" + "\n".join(lines) + "\n\n"


def _convert_paragraph(elem, state: _State) -> str:
    content = _default_children(elem, state)
    content = _WS_RE.sub(" ", content).strip()
    if not content:
        return ""
    return "\n\n" + content + "\n\n"


def _postprocess(md: str) -> str:
    md = _BLANK_RUN_RE.sub("\n\n", md)
    md = md.strip()
    return md + "\n" if md else ""
