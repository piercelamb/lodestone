"""Unit tests for `_system/latex/figures.py`."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from _system.latex.figures import (
    LatexFigureDescriptor,
    discover_figures,
    read_figure_bytes,
)


def _png(path: Path) -> None:
    img = Image.new("RGB", (16, 12), color=(255, 0, 128))
    img.save(path, format="PNG")


def test_discover_finds_figure_with_caption_and_label(tmp_path: Path):
    _png(tmp_path / "f.png")
    src = r"""
    \begin{figure}
    \includegraphics{f.png}
    \caption{First figure.}
    \label{fig:first}
    \end{figure}
    """
    figs = discover_figures(src, tmp_path)
    assert len(figs) == 1
    f = figs[0]
    assert f.figure_number == 1
    assert f.caption == "First figure."
    assert f.figure_id == "fig:first"
    assert f.local_path is not None
    assert f.local_path.name == "f.png"


def test_discover_resolves_extension_when_missing(tmp_path: Path):
    _png(tmp_path / "diagram.png")
    src = r"""
    \begin{figure}
    \includegraphics{diagram}
    \caption{No extension.}
    \end{figure}
    """
    figs = discover_figures(src, tmp_path)
    assert figs[0].local_path is not None
    assert figs[0].local_path.suffix == ".png"


def test_discover_tikz_only_figure_has_no_local_path(tmp_path: Path):
    src = r"""
    \begin{figure}
    \begin{tikzpicture}\draw (0,0) -- (1,1);\end{tikzpicture}
    \caption{TikZ only.}
    \end{figure}
    """
    figs = discover_figures(src, tmp_path)
    assert len(figs) == 1
    assert figs[0].local_path is None
    assert figs[0].caption == "TikZ only."


def test_discover_numbers_in_document_order(tmp_path: Path):
    _png(tmp_path / "a.png")
    _png(tmp_path / "b.png")
    src = r"""
    \begin{figure}\includegraphics{a.png}\caption{A.}\end{figure}
    Some prose.
    \begin{figure}\includegraphics{b.png}\caption{B.}\end{figure}
    """
    figs = discover_figures(src, tmp_path)
    assert [f.figure_number for f in figs] == [1, 2]
    assert figs[0].local_path.name == "a.png"
    assert figs[1].local_path.name == "b.png"


def test_read_figure_bytes_returns_png(tmp_path: Path):
    _png(tmp_path / "f.png")
    desc = LatexFigureDescriptor(
        figure_number=1, display_number="1", figure_id="",
        caption="x", section_context="",
        local_path=tmp_path / "f.png",
    )
    result = read_figure_bytes(desc)
    assert result is not None
    data, mime = result
    assert mime == "image/png"
    Image.open(io.BytesIO(data))  # round-trip decode


def test_read_figure_bytes_rejects_pdf(tmp_path: Path):
    p = tmp_path / "f.pdf"
    p.write_bytes(b"%PDF-1.4")
    desc = LatexFigureDescriptor(
        figure_number=1, display_number="1", figure_id="",
        caption="x", section_context="",
        local_path=p,
    )
    assert read_figure_bytes(desc) is None


def test_read_figure_bytes_rejects_svg(tmp_path: Path):
    p = tmp_path / "f.svg"
    p.write_text("<svg/>")
    desc = LatexFigureDescriptor(
        figure_number=1, display_number="1", figure_id="",
        caption="x", section_context="",
        local_path=p,
    )
    assert read_figure_bytes(desc) is None


def test_read_figure_bytes_handles_missing_local_path():
    desc = LatexFigureDescriptor(
        figure_number=1, display_number="1", figure_id="",
        caption="x", section_context="",
        local_path=None,
    )
    assert read_figure_bytes(desc) is None
