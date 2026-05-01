"""Unit tests for `_system/latex/assemble.py`."""
from __future__ import annotations

from pathlib import Path

from _system.latex.assemble import (
    assemble_source,
    find_main_tex,
    _strip_comments,
)


def test_find_main_tex_prefers_main_dot_tex(tmp_path: Path):
    (tmp_path / "extra.tex").write_text(r"\documentclass{article}\begin{document}Other.\end{document}")
    (tmp_path / "main.tex").write_text(r"\documentclass{article}\begin{document}Main.\end{document}")
    chosen = find_main_tex(tmp_path)
    assert chosen is not None
    assert chosen.name == "main.tex"


def test_find_main_tex_returns_none_when_no_documentclass(tmp_path: Path):
    (tmp_path / "foo.tex").write_text("just a fragment, no documentclass.")
    assert find_main_tex(tmp_path) is None


def test_find_main_tex_falls_back_to_paper_then_lex(tmp_path: Path):
    (tmp_path / "zeta.tex").write_text(r"\documentclass{article}")
    (tmp_path / "paper.tex").write_text(r"\documentclass{article}")
    chosen = find_main_tex(tmp_path)
    assert chosen.name == "paper.tex"


def test_assemble_inlines_input_and_include(tmp_path: Path):
    (tmp_path / "secs").mkdir()
    (tmp_path / "secs" / "intro.tex").write_text(r"\section{Intro}\nContent.")
    (tmp_path / "main.tex").write_text(
        r"\documentclass{article}\begin{document}\input{secs/intro}\end{document}"
    )
    out = assemble_source(tmp_path / "main.tex")
    assert r"\section{Intro}" in out
    assert "Content." in out
    assert r"\input{" not in out


def test_assemble_breaks_input_cycle(tmp_path: Path):
    (tmp_path / "a.tex").write_text(r"\documentclass{article}\input{b}")
    (tmp_path / "b.tex").write_text(r"\input{a}")
    out = assemble_source(tmp_path / "a.tex")
    assert "cycle" in out.lower()


def test_assemble_handles_missing_input_with_placeholder(tmp_path: Path):
    # The placeholder is emitted as a `%` comment so pylatexenc treats
    # whatever follows on the same line as a LatexCommentNode and the
    # walker silently drops it.
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\input{missing/file}\n"
        "\\end{document}\n"
    )
    out = assemble_source(tmp_path / "main.tex")
    assert "lodestone: missing" in out
    # Find the line carrying the residual directive — it must be a comment
    # so pylatexenc swallows it.
    residual_lines = [ln for ln in out.splitlines() if "\\input{missing/file}" in ln]
    assert residual_lines
    for line in residual_lines:
        assert line.lstrip().startswith("%")


def test_assemble_adds_tex_extension(tmp_path: Path):
    (tmp_path / "frag.tex").write_text("PIECE")
    (tmp_path / "main.tex").write_text(
        r"\documentclass{article}\begin{document}\input{frag}\end{document}"
    )
    out = assemble_source(tmp_path / "main.tex")
    assert "PIECE" in out


def test_strip_comments_preserves_escaped_percent():
    src = "100\\% true % a comment\nnext line"
    out = _strip_comments(src)
    assert "100\\% true " in out
    assert "a comment" not in out
    assert "next line" in out


def test_assemble_appends_companion_bbl(tmp_path: Path):
    (tmp_path / "main.tex").write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\bibliography{refs}\n"
        "\\end{document}\n"
    )
    (tmp_path / "main.bbl").write_text(
        "\\begin{thebibliography}{1}\n"
        "\\bibitem{a} A. Author. arXiv:1234.5678, 2024.\n"
        "\\end{thebibliography}\n"
    )
    out = assemble_source(tmp_path / "main.tex")
    assert "\\begin{thebibliography}" in out
    assert "\\bibitem{a}" in out


def test_strip_comments_drops_full_line_comments():
    src = "real text\n% drop me\nalso real"
    out = _strip_comments(src)
    assert "drop me" not in out
    assert "real text" in out
    assert "also real" in out
