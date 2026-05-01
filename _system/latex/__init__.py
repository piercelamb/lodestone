"""LaTeX-source fallback package.

Used when both arxiv.org/html/{id} and ar5iv.labs.arxiv.org/html/{id} fail
for an arxiv paper. We download arxiv.org/e-print/{id}, parse the LaTeX
source with pylatexenc, and emit markdown directly. No subprocess, no
Perl, no Docker — pure Python.
"""

# Prefixes the LaTeX source we stash in `papers.raw_html` when the
# fallback fires. convert_paper strips it before handing the body to the
# walker; latexml_parser would silently mis-parse the LaTeX otherwise.
LATEX_SENTINEL_PREFIX = "<lodestone:latex>\n"
