"""Shared pylatexenc context database for the LaTeX-source fallback.

The default macro DB pylatexenc ships with covers ~80% of what arxiv
papers use, but the load-bearing arg-binding for ``\\caption``,
``\\href``, and the inline-style macros is missing. Without explicit
specs the walker swallows the macro but leaves the next ``{...}`` group
floating as a sibling node — the figure handler can't see its caption
and the walker can't render text styling.

We register the missing specs in a single shared context so figures.py
and walker.py see the same node tree.
"""
from __future__ import annotations

from pylatexenc.latexwalker import get_default_latex_context_db
from pylatexenc.macrospec import std_macro

# pylatexenc 2.x exposes LatexContextDb under macrospec; we don't actually
# need to type-annotate it, just construct + return the default one with
# additions. Keeping the import light avoids version-specific breakage.


def build_context():
    ctx = get_default_latex_context_db()
    ctx.add_context_category(
        "lodestone-fallback",
        macros=[
            # Captions and labels (figure / table envelope).
            std_macro("caption", "*[{"),
            std_macro("captionof", "{{"),
            # Inline text styling not in the default DB.
            std_macro("texttt", "{"),
            std_macro("textsc", "{"),
            std_macro("textsl", "{"),
            std_macro("textsf", "{"),
            std_macro("textrm", "{"),
            std_macro("textnormal", "{"),
            std_macro("underline", "{"),
            std_macro("uline", "{"),
            std_macro("smash", "{"),
            std_macro("mbox", "{"),
            std_macro("hbox", "{"),
            # Linking.
            std_macro("href", "{{"),
            std_macro("url", "{"),
            std_macro("nolinkurl", "{"),
            # Footnotes — emit inline as `(text)`.
            std_macro("footnote", "[{"),
            std_macro("footnotetext", "[{"),
            # Cross-references — emit literal label/key.
            # (default DB has \ref/\eqref/\autoref already; \nameref needs a spec)
            std_macro("nameref", "{"),
            std_macro("Cref", "{"),
            std_macro("cref", "{"),
            # Definitions and theorems we will swallow.
            std_macro("newtheorem", "{[{["),
            std_macro("renewcommand", "*{[[{"),
            std_macro("newcommand", "*{[[{"),
            std_macro("providecommand", "*{[[{"),
            std_macro("DeclareMathOperator", "*{{"),
            # Bibliography directives.
            std_macro("bibliography", "{"),
            std_macro("bibliographystyle", "{"),
            # \bibitem[opt-label]{key} — the default DB only binds the
            # mandatory arg, missing the optional [label] which then bleeds
            # into the entry text. Explicit spec gives us both.
            std_macro("bibitem", "[{"),
            std_macro("newblock", ""),
        ],
        environments=[
            # All environments we care about (figure, table, tabular,
            # itemize, equation, etc.) are already in the default DB or
            # are handled implicitly by env-name dispatch in the walker.
        ],
        prepend=True,
    )
    return ctx
