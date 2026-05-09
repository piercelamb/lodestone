"""Heading-level normalizer for pymupdf4llm PDF-fallback markdown.

pymupdf4llm infers ATX heading levels from font-size clustering, which
collapses parent and child sections to the same level on academic PDFs
where they share a font size (e.g., ``## 6. Results`` and ``## 6.1
Layout Detection`` end up siblings instead of parent/child). This module
post-processes that markdown to re-derive heading levels from the title
text using two rules:

- **Numeric / appendix prefix is authoritative.** A title like
  ``1. Introduction`` is forced to H2; ``1.1 Foo`` and deeper are forced
  to H3 (the splitter caps at H3).
- **Canonical name is graceful.** Common section names like ``Conclusion``
  or ``Related Work`` are *promoted* to H2 if they're at H3 or deeper, but
  never demoted and never touched if already at H1/H2.

Headings inside fenced code blocks are left alone. The function operates
on markdown only — no model, no fonts, no layout — and is invoked exactly
once from ``convert_paper`` on the PDF-fallback path. HTML/LaTeX paths
produce trustworthy heading levels and bypass this step.
"""
from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_WS_RE = re.compile(r"\s+")

# Numeric-prefix patterns. Order matters: the deepest must be tested first
# because the shallower patterns also match deeper prefixes after backtracking.
_NUM_DEEP_RE = re.compile(r"^\d+(?:\.\d+){2,}\.?\s")
_NUM_TWO_RE = re.compile(r"^\d+\.\d+\.?\s")
_NUM_ONE_RE = re.compile(r"^\d+\.?\s")
_APPENDIX_PREFIX_RE = re.compile(r"^Appendix\s+[A-Z]\b")
_APPENDIX_NUM_RE = re.compile(r"^[A-Z]\.\d+(?:\.\d+)*\b")
_APPENDIX_LOOSE_RE = re.compile(r"^appendix(?:\s|:|$)", re.IGNORECASE)

_CANONICAL_TITLES = frozenset({
    "abstract",
    "introduction",
    "background",
    "related work",
    "prior work",
    "methods",
    "methodology",
    "method",
    "approach",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "acknowledgments",
    "acknowledgements",
})


def _strip_bold(title: str) -> str:
    """Strip surrounding ``**`` / ``*`` wrappers (common pymupdf4llm artifact)."""
    s = title.strip()
    while True:
        if len(s) >= 4 and s.startswith("**") and s.endswith("**"):
            s = s[2:-2].strip()
            continue
        if len(s) >= 2 and s.startswith("*") and s.endswith("*"):
            s = s[1:-1].strip()
            continue
        break
    return s


def _classify_numeric(title: str) -> int | None:
    """Return target heading level for numeric/appendix titles, else None."""
    if _NUM_DEEP_RE.match(title):
        return 3
    if _NUM_TWO_RE.match(title):
        return 3
    if _NUM_ONE_RE.match(title):
        return 2
    if _APPENDIX_PREFIX_RE.match(title):
        return 2
    if _APPENDIX_NUM_RE.match(title):
        return 3
    return None


def _classify_canonical(title: str) -> int | None:
    """Return 2 if title matches the canonical-name allowlist, else None."""
    norm = _WS_RE.sub(" ", title.strip().lower())
    if norm in _CANONICAL_TITLES:
        return 2
    if _APPENDIX_LOOSE_RE.match(title):
        return 2
    return None


def normalize_pdf_headings(markdown: str) -> str:
    """Reassign ATX heading levels in pymupdf4llm output.

    Numeric/appendix prefix is authoritative; canonical names gracefully
    promote H3+ to H2 (never demote, never touch H1/H2). Fenced code
    blocks are passed through unchanged.
    """
    out: list[str] = []
    in_fence = False
    fence_marker: str | None = None

    for line in markdown.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        fence_match = _FENCE_RE.match(body)
        if fence_match:
            marker = fence_match.group(1)
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
            out.append(line)
            continue

        if in_fence:
            out.append(line)
            continue

        m = _HEADING_RE.match(body)
        if not m:
            out.append(line)
            continue

        current_level = len(m.group(1))
        title = m.group(2)
        bare = _strip_bold(title)

        target = _classify_numeric(bare)
        if target is None:
            canonical = _classify_canonical(bare)
            if canonical is not None and current_level >= 3:
                target = canonical

        if target is None or target == current_level:
            out.append(line)
            continue

        if line.endswith("\r\n"):
            newline = "\r\n"
        elif line.endswith("\n"):
            newline = "\n"
        else:
            newline = ""
        out.append(f"{'#' * target} {title}{newline}")

    return "".join(out)
