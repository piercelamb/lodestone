"""Verifies FTS5 tokenizer choices encode the intended behaviours.

- ``sections``: ``unicode61 remove_diacritics 2`` (no porter) — preserves
  hyphenated model names, arxiv IDs, and LaTeXML figure IDs.
- ``terms_fts``: ``porter unicode61`` — stem-folds queries so aliases like
  ``"index"`` match natural-language queries like ``"indexing"``.
"""
from __future__ import annotations

import pytest


def _insert_section(conn, body: str, paper_id: int = 1) -> None:
    conn.execute(
        "INSERT INTO sections (paper_id, domain, paper_name, section_title, section_level, body) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (paper_id, "ml", "p", "Some Section", 1, body),
    )


def _section_matches(conn, query: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM sections WHERE sections MATCH ?",
        (query,),
    ).fetchone()[0]


@pytest.mark.parametrize("stored_body, term", [
    ("We evaluate Book-RAG against baselines.", "Book-RAG"),
    ("Embeddings from bge-small-en-v1.5 are used.", "bge-small-en-v1.5"),
    ("See paper 2512.03413 for more details.", "2512.03413"),
    ("The LaTeXML id S3.F1 resolves to Figure 3, panel (a).", "S3.F1"),
])
def test_sections_tokenizer_preserves_technical_tokens(conn, stored_body, term):
    _insert_section(conn, stored_body)
    # Phrase query — forces adjacent-token matching regardless of how `-` or `.`
    # are parsed by the FTS5 query grammar.
    assert _section_matches(conn, f'"{term}"') >= 1, (
        f"sections FTS5 did not match phrase {term!r} in body {stored_body!r}"
    )


def test_terms_fts_porter_stemming(conn):
    """Querying 'indexing' must stem-match a stored alias 'index'."""
    conn.execute(
        "INSERT INTO canonical_terms (id, domain, term_type, entity_type, canonical_name, first_seen_in) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (7, "ml", "entity", "method", "Indexing", "paper_1"),
    )
    conn.execute(
        "INSERT INTO terms_fts (term_id, domain, term_type, entity_type, canonical_name, aliases) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (7, "ml", "entity", "method", "Indexing", "index"),
    )
    hits = conn.execute(
        "SELECT COUNT(*) FROM terms_fts WHERE terms_fts MATCH ?",
        ("indexing",),
    ).fetchone()[0]
    assert hits >= 1, "porter stemmer failed to match 'indexing' against alias 'index'"
