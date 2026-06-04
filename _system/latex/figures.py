"""Figure discovery from assembled LaTeX source.

Walks the parsed node tree for ``\\begin{figure}...\\end{figure}`` envs,
pulls the first ``\\includegraphics{path}`` plus the figure's caption
and label, and resolves the path against the tarball root.

PDF / EPS / SVG figures are flagged as missing — pillow can't render
them without librsvg / poppler, and we want the fallback to stay pure
Python. They surface as placeholders in the markdown so the gap is
visible to readers; the convert pipeline logs the skip counters via
``_LOG.warning`` (``papers.needs_review`` is no longer toggled — that
column is now reserved for classify's brand-new-taxonomy review queue).

TikZ-only figures (``\\begin{figure}\\begin{tikzpicture}...``) get a
descriptor with ``local_path = None`` so the walker can emit an
"included as PDF" placeholder at the right ordinal.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import NamedTuple, Optional

from pylatexenc.latexwalker import (
    LatexEnvironmentNode,
    LatexMacroNode,
    LatexWalker,
)

from _system.latex.context import build_context
from _system.utils.logging import get_logger

_LOG = get_logger("latex.figures")

# Extension search order when \includegraphics omits the suffix. Raster
# formats first because they're what pillow can decode; PDF/EPS/SVG come
# last and become placeholders during read_figure_bytes.
_FIGURE_EXTS = (".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg", "")

_FIGURE_ENVS = frozenset({"figure", "figure*", "SCfigure", "wrapfigure"})


class LatexFigureDescriptor(NamedTuple):
    figure_number: int
    display_number: Optional[str]
    figure_id: str           # \label key (or empty string)
    caption: str
    section_context: str
    local_path: Optional[Path]  # None for TikZ-only / unrenderable; fetch-only
    # True iff bytes are available for rendering. fetch sets this when the
    # raster file is on disk; convert sets it from DB membership. The walker
    # consults only this field, never local_path.
    has_image: bool = False


def discover_figures(
    assembled_tex: str, tex_root: Path
) -> list[LatexFigureDescriptor]:
    """Walk the assembled source and return one descriptor per figure env.

    Numbered in document order. Section context is approximated by
    tracking the most recent ``\\section`` / ``\\subsection`` titles.
    """
    walker = LatexWalker(
        assembled_tex,
        latex_context=build_context(),
        tolerant_parsing=True,
    )
    nodes, _, _ = walker.get_latex_nodes()
    state = _DiscoverState(tex_root=tex_root)
    _walk(nodes, state)
    return state.figures


class _DiscoverState:
    def __init__(self, tex_root: Path):
        self.tex_root = tex_root
        self.figures: list[LatexFigureDescriptor] = []
        self.section_stack: list[str] = []

    def section_context(self) -> str:
        return " > ".join(t for t in self.section_stack if t)


def _walk(nodes, state: _DiscoverState) -> None:
    for node in nodes:
        if node is None:
            continue
        if isinstance(node, LatexMacroNode):
            _track_section(node, state)
            # Macros can carry nested groups; recurse so a figure inside
            # a custom wrapper macro still gets discovered.
            for arg in _arg_nodelists(node):
                _walk(arg, state)
            continue
        if isinstance(node, LatexEnvironmentNode):
            if node.environmentname in _FIGURE_ENVS:
                _emit_figure(node, state)
                # Skip recursion into the figure body — we already
                # captured what we need and don't want a sub-figure's
                # \includegraphics to double-count.
                continue
            inner = node.nodelist or []
            _walk(inner, state)
            continue
        # Group / chars / math / comment — recurse if they have children.
        inner = getattr(node, "nodelist", None)
        if inner:
            _walk(inner, state)


def _arg_nodelists(macro: LatexMacroNode):
    if not macro.nodeargd or not macro.nodeargd.argnlist:
        return
    for arg in macro.nodeargd.argnlist:
        if arg is None:
            continue
        nl = getattr(arg, "nodelist", None)
        if nl:
            yield nl


def _track_section(node: LatexMacroNode, state: _DiscoverState) -> None:
    name = node.macroname
    depth = {"section": 1, "subsection": 2, "subsubsection": 3}.get(name)
    if depth is None:
        return
    title = _extract_text(node)
    while len(state.section_stack) >= depth:
        state.section_stack.pop()
    while len(state.section_stack) < depth - 1:
        state.section_stack.append("")
    state.section_stack.append(title)


def _emit_figure(env: LatexEnvironmentNode, state: _DiscoverState) -> None:
    figure_number = len(state.figures) + 1
    body = env.nodelist or []

    includegraphics_paths: list[str] = []
    caption_parts: list[str] = []
    label = ""
    has_tikz = False

    for child in _iter_descendants(body):
        if isinstance(child, LatexEnvironmentNode):
            if child.environmentname in {"tikzpicture", "pspicture", "pgfpicture"}:
                has_tikz = True
            continue
        if isinstance(child, LatexMacroNode):
            if child.macroname == "includegraphics":
                p = _last_arg_text(child)
                if p:
                    includegraphics_paths.append(p)
            elif child.macroname in {"caption", "captionof"}:
                caption_parts.append(_extract_text(child))
            elif child.macroname == "label":
                label = _last_arg_text(child) or label

    caption = " ".join(c.strip() for c in caption_parts if c.strip()).strip()

    local_path: Optional[Path] = None
    if includegraphics_paths:
        # Last \includegraphics wins — multi-panel figures often build
        # the layout incrementally, with the final \includegraphics being
        # the one carrying the full subfloat. Sub-panel discovery would
        # need a richer model; this matches the HTML path's "first <img>
        # by document order is the figure" simplification.
        local_path = _resolve_figure_path(includegraphics_paths[0], state.tex_root)
        if local_path is None:
            _LOG.info(
                "figure %d: \\includegraphics{%s} not found under %s",
                figure_number, includegraphics_paths[0], state.tex_root,
            )
    elif has_tikz:
        local_path = None  # placeholder; emits "TikZ — see PDF" comment
    else:
        # Layout-only figure with no graphics: still register so the
        # walker emits a placeholder at the right ordinal.
        local_path = None

    state.figures.append(
        LatexFigureDescriptor(
            figure_number=figure_number,
            display_number=str(figure_number),
            figure_id=label,
            caption=caption,
            section_context=state.section_context(),
            local_path=local_path,
            has_image=local_path is not None,
        )
    )


def _iter_descendants(nodes):
    """Depth-first iteration over every descendant node. Walks both
    nodelist children and macro args so a \\includegraphics nested inside
    a wrapper (\\centering group, minipage, sub-figure) is still found."""
    stack: deque = deque(nodes)
    while stack:
        node = stack.popleft()
        if node is None:
            continue
        yield node
        children: list = []
        inner = getattr(node, "nodelist", None)
        if inner:
            children.extend(inner)
        if isinstance(node, LatexMacroNode) and node.nodeargd:
            for arg in node.nodeargd.argnlist or []:
                if arg is None:
                    continue
                arg_nl = getattr(arg, "nodelist", None)
                if arg_nl:
                    children.extend(arg_nl)
        stack.extendleft(reversed(children))


def _last_arg_text(macro: LatexMacroNode) -> str:
    if not macro.nodeargd or not macro.nodeargd.argnlist:
        return ""
    for arg in reversed(macro.nodeargd.argnlist):
        if arg is None:
            continue
        text = _flatten_text(getattr(arg, "nodelist", None) or [])
        if text:
            return text.strip()
    return ""


def _extract_text(macro_or_env) -> str:
    if isinstance(macro_or_env, LatexMacroNode):
        return _last_arg_text(macro_or_env)
    if isinstance(macro_or_env, LatexEnvironmentNode):
        return _flatten_text(macro_or_env.nodelist or []).strip()
    return ""


def _flatten_text(nodes) -> str:
    parts: list[str] = []
    for n in nodes:
        if n is None:
            continue
        chars = getattr(n, "chars", None)
        if chars is not None:
            parts.append(chars)
            continue
        inner = getattr(n, "nodelist", None)
        if inner:
            parts.append(_flatten_text(inner))
    return "".join(parts)


def _resolve_figure_path(raw: str, tex_root: Path) -> Optional[Path]:
    raw = raw.strip().strip("\"")
    if not raw:
        return None
    candidate = tex_root / raw
    # Try the raw path first (may already include extension).
    if candidate.is_file():
        return candidate
    # Then try common extensions.
    for ext in _FIGURE_EXTS:
        if not ext:
            continue
        with_ext = candidate.with_suffix(ext) if candidate.suffix else Path(str(candidate) + ext)
        if with_ext.is_file():
            return with_ext
    return None


def read_figure_bytes(desc: LatexFigureDescriptor) -> tuple[bytes, str] | None:
    """Read a figure's bytes from disk and return (data, content_type_hint).

    Pure file IO; no rendering. PDF/EPS/SVG figures return None — the
    fallback explicitly does not invoke poppler / librsvg / latex
    rendering. Caller (fetch_paper) downscales raster output through the
    existing ``_process_figure_image`` helper.
    """
    if desc.local_path is None:
        return None
    suffix = desc.local_path.suffix.lower()
    if suffix in {".pdf", ".eps", ".svg", ".ps"}:
        _LOG.info(
            "figure %d: %s is %s, emitting placeholder (no pure-Python renderer)",
            desc.figure_number, desc.local_path.name, suffix,
        )
        return None
    if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}:
        _LOG.info(
            "figure %d: %s extension %r not recognized, skipping",
            desc.figure_number, desc.local_path.name, suffix,
        )
        return None
    try:
        data = desc.local_path.read_bytes()
    except OSError as exc:
        _LOG.warning("figure %d: read failed (%s)", desc.figure_number, exc)
        return None
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }[suffix]
    return data, mime
