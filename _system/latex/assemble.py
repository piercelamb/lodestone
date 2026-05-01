"""Locate the main LaTeX file and inline ``\\input`` / ``\\include`` directives.

arxiv tarballs are not standardized: a paper may be one ``main.tex`` or a
tree with ``intro.tex``, ``method.tex``, etc. pulled in via ``\\input``.
We need a single LaTeX string before walking, so this module finds the
top-level ``\\documentclass``-bearing file and recursively inlines the
referenced parts.
"""
from __future__ import annotations

import re
from pathlib import Path

from _system.utils.logging import get_logger

_LOG = get_logger("latex.assemble")

# Maximum recursion depth for \input chains. Real papers rarely go past
# 2-3 levels; 32 leaves headroom while still tripping on a misbehaving
# cycle that the visited-set guard somehow misses.
_MAX_INPUT_DEPTH = 32

_DOCUMENTCLASS_RE = re.compile(r"\\documentclass\b")

_PREFERRED_NAMES = ("main.tex", "paper.tex", "ms.tex")

# Match \input{path} and \include{path}. Allow whitespace around the
# braces because authors write `\input { foo }`. We deliberately do NOT
# match starred forms or \subimport — those are uncommon in arxiv.
_INPUT_RE = re.compile(
    r"\\(?P<cmd>input|include)\s*\{\s*(?P<path>[^}]+?)\s*\}"
)

# Strip line comments. % to end-of-line UNLESS the % is preceded by an
# unescaped backslash. Implementation: match a non-backslash (or start)
# before the %. The raw `%` itself is never escaped LaTeX-side as `\%`
# is escaped via the leading backslash; we preserve literal `\%` by
# requiring the preceding char to NOT be `\`.
_COMMENT_RE = re.compile(r"(?<!\\)%[^\n]*")


def find_main_tex(root: Path) -> Path | None:
    """Locate the top-level `.tex` file in ``root``.

    Heuristic: any ``*.tex`` containing ``\\documentclass``. Multiple
    matches are ordered by preferred filename (``main.tex``,
    ``paper.tex``, ``ms.tex``) then by lexical order. Returns ``None``
    when no candidate exists — caller falls through to ``failed_html``.
    """
    candidates: list[Path] = []
    for path in sorted(root.rglob("*.tex")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            _LOG.warning("failed to read %s: %s", path, exc)
            continue
        if _DOCUMENTCLASS_RE.search(text):
            candidates.append(path)

    if not candidates:
        return None

    def _rank(p: Path) -> tuple[int, str]:
        try:
            return (_PREFERRED_NAMES.index(p.name), str(p))
        except ValueError:
            return (len(_PREFERRED_NAMES), str(p))

    candidates.sort(key=_rank)
    return candidates[0]


def _strip_comments(text: str) -> str:
    """Drop ``%``-comments while preserving escaped ``\\%`` literals.

    Comments otherwise leak into the walker as text (commented-out broken
    math, draft annotations) and produce noise in the markdown.
    """
    return _COMMENT_RE.sub("", text)


def _read_text(path: Path) -> str:
    # arxiv tarballs sometimes mix latin-1 and utf-8 in older papers.
    # ``errors='replace'`` keeps us alive — the walker tolerates U+FFFD.
    return path.read_text(encoding="utf-8", errors="replace")


def _resolve_input_path(parent: Path, raw: str) -> Path | None:
    """Resolve a relative ``\\input{path}`` against the file containing it.

    Adds a ``.tex`` extension if missing. Returns ``None`` when the
    referenced file does not exist on disk — the caller logs and replaces
    the directive with a placeholder comment.
    """
    candidate = parent.parent / raw
    if candidate.exists():
        return candidate
    if not candidate.suffix:
        with_ext = candidate.with_suffix(".tex")
        if with_ext.exists():
            return with_ext
    return None


def assemble_source(main: Path) -> str:
    """Read ``main`` and recursively inline ``\\input``/``\\include`` directives.

    Cycle detection: each visited absolute path goes into a set; a second
    visit emits a placeholder comment in place of the directive and
    continues. Comments are stripped *before* inlining so commented-out
    ``\\input`` directives do not pull in extra files.

    If the paper uses BibTeX (``\\bibliography{custom}``) and the e-print
    ships a precomputed ``.bbl`` file in the tarball, the .bbl content is
    appended at the end so the walker's ``thebibliography`` handler can
    recover references. Papers that ship only ``.bib`` (raw BibTeX, not
    yet processed by bibtex) lose references — parsing BibTeX source is
    out of scope for the fallback.

    Returns a single LaTeX string ready to feed to the walker.
    """
    visited: set[Path] = set()
    body = _inline(main, visited, depth=0)
    bbl_text = _read_companion_bbl(main)
    if bbl_text:
        body = body + "\n\n" + bbl_text + "\n"
    return body


def _read_companion_bbl(main: Path) -> str:
    """If a ``*.bbl`` lives alongside the main .tex, return its contents.

    Convention: arxiv builds expect bibtex to have run; the ``.bbl``
    output (which contains the rendered ``\\begin{thebibliography}``
    block) sits next to ``main.tex``. If multiple .bbl files are present
    we concatenate them, but in practice there's exactly one.
    """
    parent = main.parent
    bbl_files = sorted(parent.glob("*.bbl"))
    if not bbl_files:
        return ""
    parts: list[str] = []
    for p in bbl_files:
        try:
            parts.append(_strip_comments(_read_text(p)))
        except OSError as exc:
            _LOG.warning("could not read %s: %s", p, exc)
    return "\n\n".join(parts)


def _inline(path: Path, visited: set[Path], depth: int) -> str:
    abs_path = path.resolve()
    if abs_path in visited:
        _LOG.warning("\\input cycle detected at %s, breaking", abs_path)
        return f"% lodestone: cycle on \\input{{{path.name}}}\n"
    if depth > _MAX_INPUT_DEPTH:
        _LOG.warning("\\input recursion past depth %d at %s", _MAX_INPUT_DEPTH, abs_path)
        return f"% lodestone: depth-cap on \\input{{{path.name}}}\n"

    visited.add(abs_path)
    text = _strip_comments(_read_text(path))

    def _replace(match: re.Match[str]) -> str:
        rel = match.group("path").strip()
        resolved = _resolve_input_path(path, rel)
        if resolved is None:
            _LOG.warning(
                "could not resolve \\%s{%s} relative to %s",
                match.group("cmd"), rel, path,
            )
            return f"% lodestone: missing \\{match.group('cmd')}{{{rel}}}\n"
        return _inline(resolved, visited, depth + 1)

    return _INPUT_RE.sub(_replace, text)
