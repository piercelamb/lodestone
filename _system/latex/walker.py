"""LaTeX → markdown walker for the LaTeX-source fallback.

Single entry point: ``tex_to_markdown(assembled_tex, figures)``. Emits
markdown that mirrors what ``latexml_parser`` produces from HTML — same
heading levels, same ``![Figure N: caption](figure:N)`` figure refs,
same fenced code blocks, same ``$$...$$`` math.

Architecture: dispatch dict ``macro_name -> handler`` and
``env_name -> handler`` (no class hierarchy). Unknown macros emit their
children's text and bump a skip counter. A per-top-level-child circuit
breaker catches emit-time exceptions so one bad node doesn't lose the
whole paper.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexNode,
    LatexSpecialsNode,
    LatexWalker,
)

from _system.html.latexml_parser import ReferenceDescriptor
from _system.latex.context import build_context
from _system.latex.figures import LatexFigureDescriptor
from _system.utils.arxiv_urls import extract_arxiv_id_from_text
from _system.utils.logging import get_logger

_LOG = get_logger("latex.walker")

_BLANK_RUN_RE = re.compile(r"\n{3,}")
_WS_RE = re.compile(r"[ \t]+")

_INLINE_STYLE_PASSTHROUGH = {
    "textsc": "{}",
    "textsl": "*{}*",
    "textsf": "{}",
    "textrm": "{}",
    "textnormal": "{}",
    "underline": "{}",
    "uline": "{}",
    "smash": "{}",
    "mbox": "{}",
    "hbox": "{}",
}

_HEADING_DEPTHS = {
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
    "paragraph": 4,
    "subparagraph": 5,
}

# Math envs whose body we pass straight through inside `$$...$$`.
_DISPLAY_MATH_ENVS = {
    "equation", "equation*",
    "align", "align*",
    "gather", "gather*",
    "multline", "multline*",
    "eqnarray", "eqnarray*",
    "displaymath",
}

# Diagram envs that have no pure-Python renderer.
_DIAGRAM_ENVS = {"tikzpicture", "pspicture", "pgfpicture"}

# Code envs.
_CODE_ENVS = {"verbatim", "verbatim*", "lstlisting", "minted", "Verbatim"}

# Macros we deliberately swallow with no output.
_SILENT_MACROS = frozenset({
    "label", "maketitle", "noindent", "indent", "linebreak", "newpage",
    "clearpage", "newline", "thispagestyle", "pagestyle", "vspace", "hspace",
    "bigskip", "medskip", "smallskip", "centering", "raggedright",
    "raggedleft", "tableofcontents", "listoffigures", "listoftables",
    "bibliographystyle", "bibliography", "input", "include",
    "newcommand", "renewcommand", "providecommand", "DeclareMathOperator",
    "newtheorem", "setlength", "setcounter", "addtocounter",
    "captionsetup", "graphicspath", "usepackage", "documentclass",
    "RequirePackage", "PassOptionsToPackage", "ProvidesPackage",
    "@input",
})


class ConversionResult(NamedTuple):
    markdown: str
    references: list[ReferenceDescriptor]
    skipped_macros: dict[str, int]
    skipped_envs: dict[str, int]
    failed_sections: list[str]
    parse_errors: int


@dataclass
class _Ctx:
    figures: list[LatexFigureDescriptor]
    figure_index: int = 0  # next figure descriptor to consume
    skipped_macros: dict[str, int] = field(default_factory=dict)
    skipped_envs: dict[str, int] = field(default_factory=dict)
    failed_sections: list[str] = field(default_factory=list)
    references: list[ReferenceDescriptor] = field(default_factory=list)
    section_stack: list[str] = field(default_factory=list)
    parse_errors: int = 0
    current_section_title: str = ""

    def bump_macro(self, name: str) -> None:
        self.skipped_macros[name] = self.skipped_macros.get(name, 0) + 1

    def bump_env(self, name: str) -> None:
        self.skipped_envs[name] = self.skipped_envs.get(name, 0) + 1


def tex_to_markdown(
    assembled_tex: str, figures: list[LatexFigureDescriptor]
) -> ConversionResult:
    """Convert assembled LaTeX source to markdown.

    ``figures`` is the descriptor list produced by ``figures.discover_figures``.
    The walker consumes them positionally — the Nth ``\\begin{figure}`` env
    pulls the Nth descriptor.
    """
    ctx = _Ctx(figures=figures)

    walker = LatexWalker(
        assembled_tex,
        latex_context=build_context(),
        tolerant_parsing=True,
    )
    nodes, _, _ = walker.get_latex_nodes()

    body_nodes = _find_document_body(nodes)
    md = _emit_top_level(body_nodes, ctx)
    md = _postprocess(md)

    return ConversionResult(
        markdown=md,
        references=ctx.references,
        skipped_macros=ctx.skipped_macros,
        skipped_envs=ctx.skipped_envs,
        failed_sections=ctx.failed_sections,
        parse_errors=ctx.parse_errors,
    )


def _find_document_body(nodes: list[LatexNode]) -> list[LatexNode]:
    """Return the nodelist inside ``\\begin{document}...\\end{document}``.

    If the source has no document env (a fragment, or a malformed paper)
    we walk the whole top-level list — ``\\documentclass`` and other
    preamble macros are silenced so they don't bleed into the markdown.
    """
    for n in nodes:
        if isinstance(n, LatexEnvironmentNode) and n.environmentname == "document":
            return list(n.nodelist or [])
    return list(nodes)


def _emit_top_level(nodes: list[LatexNode], ctx: _Ctx) -> str:
    """Walk the document body's children with a per-child circuit breaker.

    A single emit-time exception (a malformed env, an unhandled node type,
    a deeply nested macro that pylatexenc misparsed) is caught here,
    recorded against the current section title, and recovery continues
    with the next sibling.
    """
    out: list[str] = []
    for child in nodes:
        try:
            out.append(_emit(child, ctx))
        except Exception as exc:  # noqa: BLE001 — circuit breaker by design
            title = ctx.current_section_title or "<preamble>"
            ctx.failed_sections.append(title)
            ctx.parse_errors += 1
            _LOG.warning(
                "section %r: emit failed on %s — %s",
                title, type(child).__name__, exc,
            )
            out.append(
                f'\n\n<!-- lodestone: section "{title}" partially failed: '
                f'{type(exc).__name__} -->\n\n'
            )
    return "".join(out)


# ----------------------------------------------------------------------
# Core dispatch
# ----------------------------------------------------------------------


def _emit(node: LatexNode, ctx: _Ctx) -> str:
    if node is None:
        return ""
    if isinstance(node, LatexCharsNode):
        return _emit_chars(node.chars)
    if isinstance(node, LatexCommentNode):
        return ""  # comments stripped at assemble-time, but be defensive
    if isinstance(node, LatexGroupNode):
        return _emit_nodelist(node.nodelist or [], ctx)
    if isinstance(node, LatexMathNode):
        return _emit_math(node)
    if isinstance(node, LatexSpecialsNode):
        return getattr(node, "specials_chars", "") or ""
    if isinstance(node, LatexMacroNode):
        return _emit_macro(node, ctx)
    if isinstance(node, LatexEnvironmentNode):
        return _emit_env(node, ctx)
    return ""


def _emit_nodelist(nodes, ctx: _Ctx) -> str:
    return "".join(_emit(n, ctx) for n in nodes if n is not None)


def _emit_chars(chars: str) -> str:
    """Pass through plain text. Convert literal LaTeX escapes into prose
    forms that markdown won't choke on."""
    if not chars:
        return ""
    # Resolve common LaTeX escapes back to literal characters.
    out = (
        chars
        .replace("\\%", "%")
        .replace("\\&", "&")
        .replace("\\$", "$")
        .replace("\\_", "_")
        .replace("\\#", "#")
        .replace("~", " ")
        .replace("---", "—")
        .replace("--", "–")
        .replace("``", '"')
        .replace("''", '"')
    )
    return out


def _emit_math(node: LatexMathNode) -> str:
    """Pass math through verbatim — never try to re-parse the body.

    ``displaytype`` is ``'inline'`` for ``$...$`` / ``\\(...\\)`` and
    ``'display'`` for ``$$...$$`` / ``\\[...\\]``. We use the original
    delimiters by reading ``latex_verbatim()`` so custom math macros
    survive untouched.
    """
    raw = node.latex_verbatim()
    # Inline `$...$` and display `$$...$$` already wrap themselves in
    # markdown-compatible delimiters. `\(...\)` and `\[...\]` need
    # rewriting since CommonMark renderers don't recognize them.
    if raw.startswith("\\("):
        inner = raw[2:-2] if raw.endswith("\\)") else raw[2:]
        return f"${inner}$"
    if raw.startswith("\\["):
        inner = raw[2:-2] if raw.endswith("\\]") else raw[2:]
        return f"$${inner}$$"
    return raw


# ----------------------------------------------------------------------
# Macro handlers
# ----------------------------------------------------------------------


def _emit_macro(node: LatexMacroNode, ctx: _Ctx) -> str:
    name = node.macroname

    if name in _SILENT_MACROS:
        return ""

    handler = _MACRO_HANDLERS.get(name)
    if handler is not None:
        return handler(node, ctx)

    if name in _HEADING_DEPTHS:
        return _emit_heading(node, ctx, _HEADING_DEPTHS[name])

    if name in _INLINE_STYLE_PASSTHROUGH:
        wrap = _INLINE_STYLE_PASSTHROUGH[name]
        inner = _last_arg_md(node, ctx)
        return wrap.replace("{}", inner)

    # Unknown macro — emit children's text, count it.
    ctx.bump_macro(name)
    return _last_arg_md(node, ctx)


def _emit_heading(node: LatexMacroNode, ctx: _Ctx, depth: int) -> str:
    title = _last_arg_md(node, ctx).strip()
    title = _strip_heading_numbers(title)
    while len(ctx.section_stack) >= depth:
        ctx.section_stack.pop()
    while len(ctx.section_stack) < depth - 1:
        ctx.section_stack.append("")
    ctx.section_stack.append(title)
    if depth == 1:
        ctx.current_section_title = title
    prefix = "#" * depth
    if not title:
        return ""
    return f"\n\n{prefix} {title}\n\n"


_HEADING_NUMBER_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")


def _strip_heading_numbers(title: str) -> str:
    return _HEADING_NUMBER_RE.sub("", title).strip()


def _macro_textbf(node: LatexMacroNode, ctx: _Ctx) -> str:
    return f"**{_last_arg_md(node, ctx)}**"


def _macro_textit(node: LatexMacroNode, ctx: _Ctx) -> str:
    return f"*{_last_arg_md(node, ctx)}*"


def _macro_emph(node: LatexMacroNode, ctx: _Ctx) -> str:
    return f"*{_last_arg_md(node, ctx)}*"


def _macro_texttt(node: LatexMacroNode, ctx: _Ctx) -> str:
    inner = _last_arg_md(node, ctx)
    if not inner:
        return ""
    return f"`{inner}`"


def _macro_url(node: LatexMacroNode, ctx: _Ctx) -> str:
    inner = _last_arg_md(node, ctx).strip()
    if not inner:
        return ""
    return f"<{inner}>"


def _macro_href(node: LatexMacroNode, ctx: _Ctx) -> str:
    args = _all_arg_md(node, ctx)
    if len(args) >= 2:
        url, label = args[0].strip(), args[1].strip()
        if url and label:
            return f"[{label}]({url})"
        return label or url
    return args[0] if args else ""


def _macro_cite(node: LatexMacroNode, ctx: _Ctx) -> str:
    text = _last_arg_md(node, ctx).strip()
    if not text:
        return ""
    keys = [k.strip() for k in text.split(",") if k.strip()]
    return "[" + ", ".join(keys) + "]"


def _macro_ref_like(node: LatexMacroNode, ctx: _Ctx) -> str:
    label = _last_arg_md(node, ctx).strip()
    return label


def _macro_footnote(node: LatexMacroNode, ctx: _Ctx) -> str:
    text = _last_arg_md(node, ctx).strip()
    if not text:
        return ""
    return f" ({text})"


def _macro_title(node: LatexMacroNode, ctx: _Ctx) -> str:
    # Title comes from the arxiv API at the persistence layer; emitting
    # it in the body would duplicate it on top of the abstract.
    return ""


def _macro_includegraphics(node: LatexMacroNode, ctx: _Ctx) -> str:
    # Outside a figure env, a stray \includegraphics shouldn't render
    # anything — we still bump the env counter for visibility.
    ctx.bump_macro("includegraphics-loose")
    return ""


def _macro_caption(node: LatexMacroNode, ctx: _Ctx) -> str:
    # Captions are consumed by the surrounding figure / table env. A
    # standalone \caption (e.g. inside a \captionof in a non-figure
    # context) becomes a bolded line so the reader still gets the prose.
    inner = _last_arg_md(node, ctx).strip()
    if not inner:
        return ""
    return f"\n\n**{inner}**\n\n"


_MACRO_HANDLERS: dict[str, Callable[[LatexMacroNode, _Ctx], str]] = {
    "textbf": _macro_textbf,
    "bf": _macro_textbf,
    "textit": _macro_textit,
    "it": _macro_textit,
    "emph": _macro_emph,
    "texttt": _macro_texttt,
    "tt": _macro_texttt,
    "url": _macro_url,
    "nolinkurl": _macro_url,
    "href": _macro_href,
    "cite": _macro_cite,
    "citep": _macro_cite,
    "citet": _macro_cite,
    "citeauthor": _macro_cite,
    "citeyear": _macro_cite,
    "citealp": _macro_cite,
    "ref": _macro_ref_like,
    "eqref": _macro_ref_like,
    "autoref": _macro_ref_like,
    "nameref": _macro_ref_like,
    "Cref": _macro_ref_like,
    "cref": _macro_ref_like,
    "pageref": _macro_ref_like,
    "footnote": _macro_footnote,
    "footnotetext": _macro_footnote,
    "title": _macro_title,
    "author": _macro_title,
    "date": _macro_title,
    "thanks": _macro_title,
    "address": _macro_title,
    "affil": _macro_title,
    "affiliation": _macro_title,
    "email": _macro_title,
    "includegraphics": _macro_includegraphics,
    "caption": _macro_caption,
    "captionof": _macro_caption,
    "abstract": _macro_title,  # in case \abstract appears as a macro
    "and": lambda n, c: " and ",
    "TeX": lambda n, c: "TeX",
    "LaTeX": lambda n, c: "LaTeX",
    "ldots": lambda n, c: "…",
    "dots": lambda n, c: "…",
    "cdots": lambda n, c: "…",
    "vdots": lambda n, c: "…",
    "ddots": lambda n, c: "…",
    "quad": lambda n, c: "  ",
    "qquad": lambda n, c: "    ",
    "textbackslash": lambda n, c: "\\",
    "textasciitilde": lambda n, c: "~",
    "textunderscore": lambda n, c: "_",
}


# ----------------------------------------------------------------------
# Environment handlers
# ----------------------------------------------------------------------


def _emit_env(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    name = env.environmentname

    if name == "document":
        return _emit_nodelist(env.nodelist or [], ctx)

    if name in _DIAGRAM_ENVS:
        ctx.bump_env(name)
        return ""

    if name in _DISPLAY_MATH_ENVS:
        return _emit_math_env(env)

    if name in _CODE_ENVS:
        return _emit_code_env(env)

    handler = _ENV_HANDLERS.get(name)
    if handler is not None:
        return handler(env, ctx)

    # Unknown env — recurse into body, wrap in HTML comment so the
    # markdown is auditable.
    ctx.bump_env(name)
    body = _emit_nodelist(env.nodelist or [], ctx)
    if not body.strip():
        return ""
    return f"\n\n<!-- env: {name} -->\n{body}\n"


def _emit_math_env(env: LatexEnvironmentNode) -> str:
    raw = env.latex_verbatim()
    return f"\n\n$$\n{raw}\n$$\n\n"


def _emit_code_env(env: LatexEnvironmentNode) -> str:
    body_text = _verbatim_body(env)
    lang = _code_env_lang(env)
    fence = "```" + (lang or "")
    return f"\n\n{fence}\n{body_text}\n```\n\n"


def _verbatim_body(env: LatexEnvironmentNode) -> str:
    raw = env.latex_verbatim()
    name = env.environmentname
    begin = f"\\begin{{{name}}}"
    end = f"\\end{{{name}}}"
    start = raw.find(begin)
    if start != -1:
        start += len(begin)
        # Skip an optional bracket arg (e.g. ``\begin{lstlisting}[language=Python]``).
        if start < len(raw) and raw[start] == "[":
            close = raw.find("]", start)
            if close != -1:
                start = close + 1
    else:
        start = 0
    stop = raw.rfind(end)
    if stop == -1:
        stop = len(raw)
    body = raw[start:stop]
    return body.strip("\n")


def _code_env_lang(env: LatexEnvironmentNode) -> str:
    """Look for ``language=<lang>`` in the env's first ``[...]`` arg.

    pylatexenc's default env DB doesn't bind args on lstlisting/minted, so
    the bracket arg shows up only inside the verbatim form. We read it
    out of the prefix string before the body line.
    """
    raw = env.latex_verbatim()
    name = env.environmentname
    begin = f"\\begin{{{name}}}"
    start = raw.find(begin)
    if start == -1:
        return ""
    after = raw[start + len(begin):]
    if not after.startswith("["):
        return ""
    close = after.find("]")
    if close == -1:
        return ""
    bracket = after[1:close]
    m = re.search(r"language\s*=\s*([A-Za-z0-9_+\-]+)", bracket)
    if m:
        return m.group(1).lower()
    # minted's first positional arg is the language: \begin{minted}{python}
    if name in {"minted", "Verbatim"} and re.fullmatch(r"[A-Za-z0-9_+\-]+", bracket):
        return bracket.lower()
    return ""


def _env_figure(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    fig_idx = ctx.figure_index
    ctx.figure_index += 1
    if fig_idx >= len(ctx.figures):
        # The figures discovery pass missed this env (unusual: probably
        # nested inside a wrapper macro the discovery walker didn't enter).
        # Emit a placeholder so the markdown is still readable.
        return '\n\n<!-- lodestone: figure missing descriptor -->\n\n'
    desc = ctx.figures[fig_idx]
    n = desc.figure_number
    caption = desc.caption or ""
    if not desc.has_image:
        kind = "diagram"
        if any(_iter_macros(env, "includegraphics")):
            kind = "PDF/EPS/SVG figure"
        return (
            f"\n\n<!-- Figure {n}: \"{caption}\" — {kind}, see PDF -->\n\n"
        )
    return f"\n\n![Figure {n}: {caption}](figure:{n})\n\n"


def _env_table(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    caption = ""
    body_parts: list[str] = []
    for child in env.nodelist or []:
        if isinstance(child, LatexMacroNode) and child.macroname in {"caption", "captionof"}:
            caption = _last_arg_md(child, ctx).strip()
            continue
        if isinstance(child, LatexMacroNode) and child.macroname == "label":
            continue
        body_parts.append(_emit(child, ctx))
    body = "".join(body_parts).strip()
    out = body
    if caption:
        out = (out + "\n\n" if out else "") + f"**{caption}**"
    return f"\n\n{out}\n\n"


def _env_tabular(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    rows = _split_tabular_rows(env, ctx)
    if not rows:
        return ""
    if _is_simple_table(rows):
        return "\n\n" + _render_md_table(rows) + "\n\n"
    return "\n\n" + _render_html_table(rows) + "\n\n"


def _split_tabular_rows(env: LatexEnvironmentNode, ctx: _Ctx) -> list[list[str]]:
    """Split a tabular's body into rows of cell-text by walking the
    nodelist and segmenting on ``\\\\`` and ``&``.

    pylatexenc emits ``\\\\`` as a ``LatexMacroNode`` with macroname
    ``\\`` (escaped); we detect by name. ``&`` arrives as a
    ``LatexSpecialsNode`` with ``specials_chars == "&"``.
    """
    rows: list[list[str]] = []
    current: list[list[str]] = [[]]
    for n in env.nodelist or []:
        if isinstance(n, LatexMacroNode) and n.macroname == "\\":
            # End of row. Flush the buffer.
            row = ["".join(parts).strip() for parts in current]
            rows.append(row)
            current = [[]]
            continue
        if isinstance(n, LatexSpecialsNode) and getattr(n, "specials_chars", "") == "&":
            current.append([])
            continue
        if isinstance(n, LatexMacroNode) and n.macroname in {
            "hline", "toprule", "midrule", "bottomrule", "cline", "cmidrule",
        }:
            continue
        current[-1].append(_emit(n, ctx))
    # Flush trailing row if it has content.
    last = ["".join(parts).strip() for parts in current]
    if any(c for c in last):
        rows.append(last)
    return [r for r in rows if r]


def _is_simple_table(rows: list[list[str]]) -> bool:
    if not rows:
        return False
    width = len(rows[0])
    if width == 0:
        return False
    for r in rows:
        if len(r) != width:
            return False
    return True


def _render_md_table(rows: list[list[str]]) -> str:
    def _cell(s: str) -> str:
        return s.replace("\n", " ").replace("|", "\\|")

    header = [_cell(c) for c in rows[0]]
    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join("---" for _ in header) + " |")
    for r in rows[1:]:
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    return "\n".join(out)


def _render_html_table(rows: list[list[str]]) -> str:
    parts = ["<table>"]
    if rows:
        parts.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr></thead>")
        parts.append("<tbody>")
        for r in rows[1:]:
            parts.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
        parts.append("</tbody>")
    parts.append("</table>")
    return "".join(parts)


def _env_itemize(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    return _render_list(env, ctx, ordered=False)


def _env_enumerate(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    return _render_list(env, ctx, ordered=True)


def _env_description(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    items = _split_items(env)
    if not items:
        return ""
    lines: list[str] = []
    for opt, item_nodes in items:
        term = _emit_nodelist(opt or [], ctx).strip() if opt else ""
        body = _emit_nodelist(item_nodes, ctx).strip()
        body = _WS_RE.sub(" ", body)
        if term:
            lines.append(f"**{term}**: {body}")
        else:
            lines.append(f"- {body}")
    return "\n\n" + "\n".join(lines) + "\n\n"


def _render_list(env: LatexEnvironmentNode, ctx: _Ctx, ordered: bool) -> str:
    items = _split_items(env)
    if not items:
        return ""
    lines: list[str] = []
    for i, (_opt, item_nodes) in enumerate(items, start=1):
        body = _emit_nodelist(item_nodes, ctx).strip()
        body = _WS_RE.sub(" ", body)
        marker = f"{i}." if ordered else "-"
        lines.append(f"{marker} {body}")
    return "\n\n" + "\n".join(lines) + "\n\n"


def _split_items(env: LatexEnvironmentNode) -> list[tuple[Optional[list], list]]:
    """Split a list env's body into (optional_arg, content_nodes) per ``\\item``."""
    items: list[tuple[Optional[list], list]] = []
    current: list = []
    current_opt: Optional[list] = None
    in_item = False
    for n in env.nodelist or []:
        if isinstance(n, LatexMacroNode) and n.macroname == "item":
            if in_item:
                items.append((current_opt, current))
            current = []
            current_opt = None
            in_item = True
            # \item may carry an optional [term] arg (description envs).
            if n.nodeargd and n.nodeargd.argnlist:
                for a in n.nodeargd.argnlist:
                    if a is not None and getattr(a, "delimiters", (None,))[0] == "[":
                        current_opt = list(getattr(a, "nodelist", None) or [])
                        break
            continue
        if in_item:
            current.append(n)
    if in_item:
        items.append((current_opt, current))
    return items


def _env_quote(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    body = _emit_nodelist(env.nodelist or [], ctx).strip()
    if not body:
        return ""
    quoted = "\n".join(f"> {ln}" if ln.strip() else ">" for ln in body.split("\n"))
    return "\n\n" + quoted + "\n\n"


def _env_abstract(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    body = _emit_nodelist(env.nodelist or [], ctx).strip()
    if not body:
        return ""
    return f"\n\n## Abstract\n\n{body}\n\n"


def _env_thebibliography(env: LatexEnvironmentNode, ctx: _Ctx) -> str:
    """Walk ``\\bibitem`` siblings and append ReferenceDescriptors.

    Each ``\\bibitem`` macro is followed by free-text + nested macros
    (``\\newblock``, citation styling, etc.) up to the next bibitem. We
    emit a `## References` heading + a numbered list in markdown so the
    text path is still readable. Cross-paper resolution happens in
    convert_paper from ``ctx.references``.
    """
    children = list(env.nodelist or [])
    entries: list[tuple[str, list[LatexNode]]] = []  # (key, body_nodes)
    current_key: Optional[str] = None
    current_body: list[LatexNode] = []
    for n in children:
        if isinstance(n, LatexMacroNode) and n.macroname == "bibitem":
            if current_key is not None:
                entries.append((current_key, current_body))
            current_key = _last_arg_md(n, ctx).strip()
            current_body = []
            continue
        if current_key is None:
            continue
        current_body.append(n)
    if current_key is not None:
        entries.append((current_key, current_body))

    lines: list[str] = ["## References", ""]
    for idx, (key, body_nodes) in enumerate(entries, start=1):
        raw_text = _emit_nodelist(body_nodes, ctx)
        raw_text = _WS_RE.sub(" ", raw_text).replace("\n", " ").strip()
        if not raw_text:
            continue
        ctx.references.append(
            ReferenceDescriptor(
                bibitem_id=key or None,
                ref_number=idx,
                raw_text=raw_text,
                cited_arxiv_id=extract_arxiv_id_from_text(raw_text),
            )
        )
        lines.append(f"{idx}. {raw_text}")
    return "\n\n" + "\n".join(lines) + "\n\n"


_ENV_HANDLERS: dict[str, Callable[[LatexEnvironmentNode, _Ctx], str]] = {
    "figure": _env_figure,
    "figure*": _env_figure,
    "SCfigure": _env_figure,
    "wrapfigure": _env_figure,
    "table": _env_table,
    "table*": _env_table,
    "tabular": _env_tabular,
    "tabular*": _env_tabular,
    "tabularx": _env_tabular,
    "tabulary": _env_tabular,
    "array": _env_tabular,
    "itemize": _env_itemize,
    "compactitem": _env_itemize,
    "enumerate": _env_enumerate,
    "compactenum": _env_enumerate,
    "description": _env_description,
    "quote": _env_quote,
    "quotation": _env_quote,
    "abstract": _env_abstract,
    "thebibliography": _env_thebibliography,
    "center": lambda env, ctx: _emit_nodelist(env.nodelist or [], ctx),
    "flushleft": lambda env, ctx: _emit_nodelist(env.nodelist or [], ctx),
    "flushright": lambda env, ctx: _emit_nodelist(env.nodelist or [], ctx),
    "small": lambda env, ctx: _emit_nodelist(env.nodelist or [], ctx),
    "footnotesize": lambda env, ctx: _emit_nodelist(env.nodelist or [], ctx),
    "scriptsize": lambda env, ctx: _emit_nodelist(env.nodelist or [], ctx),
    "minipage": lambda env, ctx: _emit_nodelist(env.nodelist or [], ctx),
    "subfigure": lambda env, ctx: _emit_nodelist(env.nodelist or [], ctx),
    "subfloat": lambda env, ctx: _emit_nodelist(env.nodelist or [], ctx),
    "titlepage": lambda env, ctx: "",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _iter_macros(node: LatexNode, name: str):
    if isinstance(node, LatexMacroNode) and node.macroname == name:
        yield node
        return
    inner = getattr(node, "nodelist", None)
    if inner:
        for ch in inner:
            if ch is None:
                continue
            yield from _iter_macros(ch, name)
    if isinstance(node, LatexMacroNode) and node.nodeargd:
        for a in node.nodeargd.argnlist or []:
            if a is None:
                continue
            yield from _iter_macros(a, name)


def _last_arg_md(node: LatexMacroNode, ctx: _Ctx) -> str:
    """Render the macro's last (mandatory) argument as markdown.

    Falls back to scanning the immediate next sibling group when
    pylatexenc didn't bind any args (happens for macros not registered
    in the context db). Without the spec the {...} just shows up as a
    sibling LatexGroupNode, so a macro like an unknown ``\\foo{x}`` is
    treated as ``\\foo`` followed by the standalone ``{x}`` group.
    """
    if not node.nodeargd or not node.nodeargd.argnlist:
        return ""
    for arg in reversed(node.nodeargd.argnlist):
        if arg is None:
            continue
        nl = getattr(arg, "nodelist", None) or []
        return _emit_nodelist(nl, ctx)
    return ""


def _all_arg_md(node: LatexMacroNode, ctx: _Ctx) -> list[str]:
    if not node.nodeargd or not node.nodeargd.argnlist:
        return []
    out: list[str] = []
    for arg in node.nodeargd.argnlist:
        if arg is None:
            continue
        nl = getattr(arg, "nodelist", None) or []
        out.append(_emit_nodelist(nl, ctx))
    return out


def _flatten_chars(nodes) -> str:
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
            parts.append(_flatten_chars(inner))
    return "".join(parts)


def _postprocess(md: str) -> str:
    md = _BLANK_RUN_RE.sub("\n\n", md)
    # Collapse runs of horizontal whitespace inside a single line.
    lines = []
    for ln in md.split("\n"):
        if ln.startswith("    ") or ln.startswith("\t"):
            lines.append(ln)
        else:
            lines.append(_WS_RE.sub(" ", ln).rstrip())
    md = "\n".join(lines).strip()
    return md + "\n" if md else ""
