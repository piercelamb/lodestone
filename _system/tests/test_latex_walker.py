"""Unit tests for `_system/latex/walker.py`."""
from __future__ import annotations

from pathlib import Path

from _system.latex.figures import LatexFigureDescriptor
from _system.latex.walker import tex_to_markdown


def _convert(src: str, figs=None):
    return tex_to_markdown(src, figs or [])


def test_section_headings_emitted_with_correct_depth():
    src = r"""\section{Intro} body \subsection{M} more \subsubsection{D} deep"""
    out = _convert(src).markdown
    assert "# Intro" in out
    assert "## M" in out
    assert "### D" in out


def test_inline_styles_pass_through():
    src = r"\textbf{bold} and \textit{ital} and \emph{em} and \texttt{code}"
    md = _convert(src).markdown
    assert "**bold**" in md
    assert "*ital*" in md
    assert "*em*" in md
    assert "`code`" in md


def test_href_and_url_render_as_markdown_links():
    src = r"\href{https://x.com}{x} and \url{https://y.com}"
    md = _convert(src).markdown
    assert "[x](https://x.com)" in md
    assert "<https://y.com>" in md


def test_cite_emits_bracketed_keys():
    src = r"See \cite{a, b , c}."
    md = _convert(src).markdown
    assert "[a, b, c]" in md


def test_ref_eqref_autoref_emit_label_text():
    src = r"See \ref{fig:one} and \eqref{eq:two} and \autoref{sec:three}."
    md = _convert(src).markdown
    assert "fig:one" in md
    assert "eq:two" in md
    assert "sec:three" in md


def test_label_swallowed():
    md = _convert(r"\label{fig:hidden} body").markdown
    assert "fig:hidden" not in md


def test_footnote_emits_inline_parens():
    md = _convert(r"text\footnote{a side note}").markdown
    assert "(a side note)" in md


def test_unknown_macro_keeps_children_and_counts_skip():
    result = _convert(r"\fancyhighlight{important text}")
    assert "important text" in result.markdown
    assert result.skipped_macros.get("fancyhighlight") == 1


def test_tikzpicture_env_emits_nothing_and_bumps_skip():
    src = r"\begin{tikzpicture}\draw (0,0)--(1,1);\end{tikzpicture}"
    result = _convert(src)
    assert "draw" not in result.markdown
    assert "tikzpicture" in result.skipped_envs


def test_math_passthrough_inline_and_display():
    src = r"inline $x^2 + y^2$ then \[ z = ax + b \]"
    md = _convert(src).markdown
    assert "$x^2 + y^2$" in md
    assert "$$" in md
    assert "z = ax + b" in md


def test_equation_env_passthrough():
    src = r"\begin{equation} E = mc^2 \end{equation}"
    md = _convert(src).markdown
    assert "$$" in md
    assert "E = mc^2" in md


def test_figure_with_image_emits_image_ref():
    descriptor = LatexFigureDescriptor(
        figure_number=1, display_number="1", figure_id="fig:a",
        caption="A.", section_context="",
        local_path=Path("/tmp/x.png"), has_image=True,
    )
    src = r"\begin{figure}\includegraphics{x.png}\caption{A.}\label{fig:a}\end{figure}"
    md = _convert(src, [descriptor]).markdown
    assert "![Figure 1: A.](figure:1)" in md


def test_figure_without_image_emits_placeholder():
    descriptor = LatexFigureDescriptor(
        figure_number=1, display_number="1", figure_id="",
        caption="X.", section_context="", local_path=None, has_image=False,
    )
    src = r"\begin{figure}\begin{tikzpicture}\draw(0,0);\end{tikzpicture}\caption{X.}\end{figure}"
    md = _convert(src, [descriptor]).markdown
    assert "<!-- Figure 1:" in md
    assert "see PDF" in md


def test_itemize_renders_bullets():
    src = r"\begin{itemize}\item One \item Two\end{itemize}"
    md = _convert(src).markdown
    assert "- One" in md
    assert "- Two" in md


def test_enumerate_renders_numbered():
    src = r"\begin{enumerate}\item A \item B\end{enumerate}"
    md = _convert(src).markdown
    assert "1. A" in md
    assert "2. B" in md


def test_verbatim_renders_fenced_code():
    src = "\\begin{verbatim}\nprint('hi')\n\\end{verbatim}"
    md = _convert(src).markdown
    assert "```" in md
    assert "print('hi')" in md


def test_lstlisting_extracts_language_hint():
    src = "\\begin{lstlisting}[language=Python]\nx = 1\n\\end{lstlisting}"
    md = _convert(src).markdown
    assert "```python" in md.lower()
    assert "x = 1" in md


def test_thebibliography_extracts_references_with_arxiv_id():
    src = r"""
    \begin{thebibliography}{1}
    \bibitem{key1} A. Author. arXiv:2310.08560, 2023.
    \bibitem{key2} B. Other. ICML 2024.
    \end{thebibliography}
    """
    result = _convert(src)
    assert len(result.references) == 2
    r1, r2 = result.references
    assert r1.bibitem_id == "key1"
    assert r1.cited_arxiv_id == "2310.08560"
    assert r2.bibitem_id == "key2"
    assert r2.cited_arxiv_id is None


def test_abstract_env_emits_heading():
    src = r"\begin{abstract} we did things. \end{abstract}"
    md = _convert(src).markdown
    assert "## Abstract" in md
    assert "we did things" in md.lower()


def test_simple_tabular_renders_md_table():
    src = r"""
    \begin{tabular}{l c r}
    A & B & C \\
    1 & 2 & 3 \\
    \end{tabular}
    """
    md = _convert(src).markdown
    assert "| A | B | C |" in md
    assert "| 1 | 2 | 3 |" in md


def test_section_heading_numbering_stripped():
    md = _convert(r"\section{1.2 Introduction}").markdown
    assert "# Introduction" in md
    assert "1.2" not in md.split("\n")[0] if "Introduction" in md else True


def test_per_section_circuit_breaker_records_failed_section(monkeypatch):
    """A handler raising mid-emit must record the section title and continue."""
    from _system.latex import walker as walker_mod

    original = walker_mod._env_table

    def boom(env, ctx):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(walker_mod, "_ENV_HANDLERS", {**walker_mod._ENV_HANDLERS, "table": boom})

    src = r"""\section{Sec One} body
    \begin{table}\begin{tabular}{l}A\\\end{tabular}\caption{C}\end{table}
    after the broken table.
    """
    result = walker_mod.tex_to_markdown(src, [])
    assert "Sec One" in result.failed_sections
    assert "after the broken table" in result.markdown
    assert "partially failed" in result.markdown


def test_documentclass_and_preamble_ignored():
    src = r"""\documentclass{article}
    \title{Hidden}
    \author{A B}
    \begin{document}
    \maketitle
    \section{Body}
    text.
    \end{document}"""
    md = _convert(src).markdown
    assert "Hidden" not in md  # \title swallowed
    assert "Body" in md
    assert "text." in md
