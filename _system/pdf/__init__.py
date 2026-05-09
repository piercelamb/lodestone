"""PDF-source fallback package.

Used when arxiv.org/html, ar5iv.labs.arxiv.org/html, AND the e-print LaTeX
source all fail for an arxiv paper. We download arxiv.org/pdf/{id} and feed
it through pymupdf4llm to produce best-effort markdown. No ML weights, no
GPU — pure-Python plus MuPDF C bindings.

pymupdf4llm/PyMuPDF are AGPL-3.0; bundled as a default dep so the fallback
is frictionless out of the box.
"""

# Prefixes the markdown we stash in `papers.raw_html` when the PDF
# fallback fires. convert_paper strips it before exposing the markdown.
# Mirrors LATEX_SENTINEL_PREFIX so the same dispatch shape works.
PDF_SENTINEL_PREFIX = "<lodestone:pdf>\n"
