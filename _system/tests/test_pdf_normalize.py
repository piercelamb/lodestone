"""Tests for the PDF-fallback heading-level normalizer."""
from __future__ import annotations

import pytest

from _system.pdf.normalize import normalize_pdf_headings


# --- Numeric / appendix prefix is authoritative ----------------------------


@pytest.mark.parametrize(
    "src, expected",
    [
        ("### 1. Introduction", "## 1. Introduction"),
        ("## 6. Results", "## 6. Results"),
        ("## 6.1 Layout Detection", "### 6.1 Layout Detection"),
        ("## 6.1.2 Foo", "### 6.1.2 Foo"),
        ("# 1. Introduction", "## 1. Introduction"),
        ("#### 2.1. Setup", "### 2.1. Setup"),
        ("### 12. Discussion", "## 12. Discussion"),
        ("### Appendix A: Walkthrough", "## Appendix A: Walkthrough"),
        ("## A.1 Input", "### A.1 Input"),
        ("## A.5.2 Subitem", "### A.5.2 Subitem"),
    ],
)
def test_numeric_and_appendix_authoritative(src: str, expected: str) -> None:
    assert normalize_pdf_headings(src) == expected


# --- Canonical allowlist is graceful ---------------------------------------


@pytest.mark.parametrize(
    "src, expected",
    [
        ("### Conclusion", "## Conclusion"),
        ("## Conclusion", "## Conclusion"),
        ("# Conclusion", "# Conclusion"),  # NEVER demoted
        ("### **Related Work**", "## **Related Work**"),  # bold preserved in output
        ("### Methods", "## Methods"),
        ("### Methodology", "## Methodology"),
        ("### method", "## method"),
        ("### Experimental Setup", "## Experimental Setup"),
        ("### References", "## References"),
        ("### Acknowledgements", "## Acknowledgements"),
        ("### Appendix", "## Appendix"),
        ("### Appendix: Notation", "## Appendix: Notation"),
    ],
)
def test_canonical_promotion(src: str, expected: str) -> None:
    assert normalize_pdf_headings(src) == expected


# --- Non-matching headings are left alone ----------------------------------


@pytest.mark.parametrize(
    "src",
    [
        "### The Story So Far",
        "### Pritesh Jha",
        "### priteshjha2711@gmail.com",
        "## What works well and what does not",
        "### Image fidelity:",
    ],
)
def test_non_matching_untouched(src: str) -> None:
    assert normalize_pdf_headings(src) == src


# --- Code-fence safety -----------------------------------------------------


def test_code_fence_lines_are_not_modified() -> None:
    src = (
        "## Real\n"
        "```\n"
        "## fake (inside fence — must NOT be modified)\n"
        "### 7. also fake\n"
        "```\n"
        "## 2. Real\n"
    )
    expected = (
        "## Real\n"
        "```\n"
        "## fake (inside fence — must NOT be modified)\n"
        "### 7. also fake\n"
        "```\n"
        "## 2. Real\n"
    )
    assert normalize_pdf_headings(src) == expected


def test_tilde_fence_is_respected() -> None:
    src = (
        "### Conclusion\n"
        "~~~\n"
        "### Conclusion\n"  # inside fence — must NOT be promoted
        "~~~\n"
        "### Conclusion\n"
    )
    expected = (
        "## Conclusion\n"
        "~~~\n"
        "### Conclusion\n"
        "~~~\n"
        "## Conclusion\n"
    )
    assert normalize_pdf_headings(src) == expected


# --- Newline preservation --------------------------------------------------


def test_no_trailing_newline_preserved() -> None:
    # Last line without a trailing newline must round-trip cleanly.
    assert normalize_pdf_headings("### 1. Intro") == "## 1. Intro"


def test_crlf_line_endings_preserved() -> None:
    src = "### 1. Intro\r\n## 1.1 Detail\r\n"
    expected = "## 1. Intro\r\n### 1.1 Detail\r\n"
    assert normalize_pdf_headings(src) == expected


def test_non_heading_lines_passthrough() -> None:
    src = (
        "### 1. Introduction\n"
        "\n"
        "Body paragraph that should be passed through.\n"
        "Another line.\n"
        "\n"
        "## 1.1 Detail\n"
    )
    expected = (
        "## 1. Introduction\n"
        "\n"
        "Body paragraph that should be passed through.\n"
        "Another line.\n"
        "\n"
        "### 1.1 Detail\n"
    )
    assert normalize_pdf_headings(src) == expected


# --- Full-document smoke test ----------------------------------------------


def test_full_document_smoke() -> None:
    src = (
        "# Title of Paper\n"
        "\n"
        "## Abstract\n"
        "abstract body\n"
        "\n"
        "### 1. Introduction\n"
        "intro body\n"
        "\n"
        "## 1.1 Contributions\n"
        "contributions body\n"
        "\n"
        "### 2. Related Work\n"
        "related body\n"
        "\n"
        "## 6. Results\n"
        "\n"
        "## 6.1 Layout Detection\n"
        "layout body\n"
        "\n"
        "## 6.1.2 Sub-detail\n"
        "sub body\n"
        "\n"
        "### **Related Work**\n"
        "\n"
        "### Conclusion\n"
        "\n"
        "### Pritesh Jha\n"
        "author byline\n"
        "\n"
        "### Appendix A: Walkthrough\n"
        "\n"
        "## A.1 Input Document\n"
    )
    expected = (
        "# Title of Paper\n"
        "\n"
        "## Abstract\n"
        "abstract body\n"
        "\n"
        "## 1. Introduction\n"
        "intro body\n"
        "\n"
        "### 1.1 Contributions\n"
        "contributions body\n"
        "\n"
        "## 2. Related Work\n"
        "related body\n"
        "\n"
        "## 6. Results\n"
        "\n"
        "### 6.1 Layout Detection\n"
        "layout body\n"
        "\n"
        "### 6.1.2 Sub-detail\n"
        "sub body\n"
        "\n"
        "## **Related Work**\n"
        "\n"
        "## Conclusion\n"
        "\n"
        "### Pritesh Jha\n"
        "author byline\n"
        "\n"
        "## Appendix A: Walkthrough\n"
        "\n"
        "### A.1 Input Document\n"
    )
    assert normalize_pdf_headings(src) == expected
