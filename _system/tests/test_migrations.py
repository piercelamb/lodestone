"""Tests for idempotent schema bootstrap and per-table invariants."""
from __future__ import annotations

import sqlite3

import pytest
from sqlite_vec import serialize_float32

from _system.db.migrations import init_db

EXPECTED_TABLES = {
    "domains",
    "collections",
    "papers",
    "repos",
    "figures",
    "paper_references",
    "sections",
    "terms_fts",
    "canonical_terms",
    "term_aliases",
    "term_embeddings",
    "topics",
    "code_files",
    "readmes_fts",
}

# Virtual tables (FTS5, vec0) create auxiliary shadow tables; filter by prefix.
_SHADOW_PREFIXES = (
    "sections_", "terms_fts_", "term_embeddings_", "readmes_fts_",
)


def _user_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table','virtual') AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {
        r[0] for r in rows
        if not any(r[0].startswith(p) for p in _SHADOW_PREFIXES)
    }


# Plain (non-virtual) tables — support row-count queries directly.
_PLAIN_TABLES = {
    "domains", "collections", "papers", "repos", "figures",
    "paper_references",
    "canonical_terms", "term_aliases", "topics",
    "code_files",
}


def _plain_row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in _PLAIN_TABLES
    }


def _seed_paper(conn: sqlite3.Connection, paper_id: int = 1) -> None:
    conn.execute("INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
                 ("ml", "Machine learning"))
    conn.execute(
        """
        INSERT INTO papers (
            id, arxiv_id, paper_name, title, authors, date, abstract, domain,
            content_hash, pdf_url, ingested_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (paper_id, f"2512.{paper_id:05d}", f"paper_{paper_id}", "t", "[]",
         "2026-04-20", "abstract text", "ml", "deadbeef", "http://x", "2026-04-20T00:00:00", "fetched"),
    )


def test_init_db_creates_all_expected_tables(conn):
    assert _user_tables(conn) == EXPECTED_TABLES


def test_init_db_is_idempotent(conn):
    first_tables = _user_tables(conn)

    # Seed one row per user (non-virtual) table so we can check "zero new
    # rows" on repeat init_db calls. Only plain tables accept these inserts
    # — skip the virtual ones (FTS5 / vec0) since they have required columns.
    conn.execute("INSERT INTO domains (name) VALUES ('ml')")
    conn.execute(
        "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, abstract, "
        "domain, content_hash, pdf_url, ingested_at, status) "
        "VALUES ('x', 'p', 't', '[]', '2026-04-20', 'a', 'ml', 'h', 'u', 'd', 'fetched')"
    )
    baseline_counts = _plain_row_counts(conn)

    init_db(conn)

    assert _user_tables(conn) == first_tables == EXPECTED_TABLES
    # Zero new rows, zero errors on repeat init_db.
    assert _plain_row_counts(conn) == baseline_counts


def test_vec0_existence_guard(conn):
    """sqlite-vec vec0 does not support IF NOT EXISTS; idempotency must work anyway."""
    # The conn fixture already ran init_db once; a naive second run without
    # the sqlite_master guard would raise here.
    init_db(conn)
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='term_embeddings'"
    ).fetchone()
    assert exists is not None


def test_canonical_terms_uniqueness(conn):
    conn.execute(
        "INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in) "
        "VALUES (?, ?, ?, ?, ?)",
        ("ml", "entity", "method", "Transformer", "paper_1"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in) "
            "VALUES (?, ?, ?, ?, ?)",
            ("ml", "entity", "method", "Transformer", "paper_2"),
        )


def test_term_aliases_uniqueness(conn):
    """``term_aliases`` PK is ``(term_id, alias, source_paper)``. Same
    triple is a duplicate; differing on any column is a new row."""
    _seed_paper(conn, paper_id=1)
    _seed_paper(conn, paper_id=2)
    conn.execute(
        "INSERT INTO canonical_terms "
        " (id, domain, term_type, entity_type, canonical_name, first_seen_in) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (1, "ml", "entity", "method", "Transformer", "paper_1"),
    )
    conn.execute(
        "INSERT INTO term_aliases "
        " (term_id, alias, source_paper, match_tier) "
        "VALUES (?, ?, ?, ?)",
        (1, "transformer-block", "paper_1", 2),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO term_aliases "
            " (term_id, alias, source_paper, match_tier) "
            "VALUES (?, ?, ?, ?)",
            (1, "transformer-block", "paper_1", 3),
        )
    # Different source_paper: must succeed.
    conn.execute(
        "INSERT INTO term_aliases "
        " (term_id, alias, source_paper, match_tier) "
        "VALUES (?, ?, ?, ?)",
        (1, "transformer-block", "paper_2", 2),
    )


def test_entities_table_is_dropped(conn):
    """Legacy ``entities`` table must not exist after the merge."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entities'"
    ).fetchone()
    assert row is None


def test_init_db_migrates_legacy_entities_into_term_aliases(tmp_path):
    """Pre-PR-#20 DB shape (3-column term_aliases PK + sibling entities
    table) folds into the synonym-index shape on first init_db run.
    Synonym entity rows survive; entities rows whose entity_name matches
    the canonical are filtered out (canonicals are never synonyms of
    themselves under Option C)."""
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    try:
        # Only the columns we touch — enough to exercise the migration path.
        legacy.executescript(
            """
            CREATE TABLE canonical_terms (
                id INTEGER PRIMARY KEY,
                domain TEXT NOT NULL,
                term_type TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT '',
                entity_type_score REAL NOT NULL DEFAULT 0.0,
                canonical_name TEXT NOT NULL,
                first_seen_in TEXT NOT NULL,
                UNIQUE(domain, term_type, canonical_name)
            );
            CREATE TABLE term_aliases (
                term_id INTEGER NOT NULL REFERENCES canonical_terms(id),
                alias TEXT NOT NULL,
                source_paper TEXT NOT NULL,
                match_tier INTEGER,
                PRIMARY KEY(term_id, alias, source_paper)
            );
            CREATE TABLE entities (
                id INTEGER PRIMARY KEY,
                paper_id INTEGER NOT NULL,
                domain TEXT NOT NULL,
                paper_name TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                source_breadcrumb TEXT NOT NULL,
                description TEXT
            );
            """
        )
        legacy.execute(
            "INSERT INTO canonical_terms (id, domain, term_type, entity_type, "
            " canonical_name, first_seen_in) "
            "VALUES (1, 'rag', 'entity', 'method', 'BookRAG', 'p1')"
        )
        # A real synonym in the legacy alias table — must survive.
        legacy.execute(
            "INSERT INTO term_aliases (term_id, alias, source_paper, match_tier) "
            "VALUES (1, 'Book-RAG', 'p1', 3)"
        )
        # entities rows: one matches the canonical (must be filtered),
        # one is a real synonym (must survive).
        legacy.execute(
            "INSERT INTO entities (paper_id, domain, paper_name, entity_name, "
            " entity_type, source_breadcrumb) "
            "VALUES (7, 'rag', 'p1', 'BookRAG', 'method', '# Intro')"
        )
        legacy.execute(
            "INSERT INTO entities (paper_id, domain, paper_name, entity_name, "
            " entity_type, source_breadcrumb) "
            "VALUES (7, 'rag', 'p1', 'Book-RAG', 'method', '# Method')"
        )
        legacy.commit()
    finally:
        legacy.close()

    from _system.db.connection import get_conn
    conn = get_conn(db_path)
    try:
        init_db(conn)
        rows = conn.execute(
            "SELECT term_id, alias, source_paper, match_tier "
            "  FROM term_aliases ORDER BY alias"
        ).fetchall()
        # Pre-existing synonym row + entities synonym row collapse via
        # INSERT OR IGNORE on the new 3-col PK; the canonical-as-alias
        # entities row is filtered out entirely.
        assert (1, "Book-RAG", "p1", 3) in rows
        assert all(r[1] != "BookRAG" for r in rows), rows
        # entities is gone.
        gone = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'entities'"
        ).fetchone()
        assert gone is None
    finally:
        conn.close()


def test_init_db_migrates_pr20_shape_to_synonym_index(tmp_path):
    """Post-PR-#20 appearance-log shape (4-col PK with source_breadcrumb,
    canonical-as-alias rows present) folds into the 3-col synonym-index
    PK: synonym rows survive deduped; canonical-as-alias rows are
    filtered out; the source_breadcrumb column is gone."""
    db_path = tmp_path / "pr20.db"
    legacy = sqlite3.connect(db_path)
    try:
        legacy.executescript(
            """
            CREATE TABLE canonical_terms (
                id INTEGER PRIMARY KEY,
                domain TEXT NOT NULL,
                term_type TEXT NOT NULL,
                entity_type TEXT NOT NULL DEFAULT '',
                entity_type_score REAL NOT NULL DEFAULT 0.0,
                canonical_name TEXT NOT NULL,
                first_seen_in TEXT NOT NULL,
                UNIQUE(domain, term_type, canonical_name)
            );
            CREATE TABLE term_aliases (
                term_id INTEGER NOT NULL REFERENCES canonical_terms(id),
                alias TEXT NOT NULL,
                source_paper TEXT NOT NULL,
                source_breadcrumb TEXT NOT NULL DEFAULT '',
                match_tier INTEGER,
                PRIMARY KEY(term_id, alias, source_paper, source_breadcrumb)
            );
            """
        )
        legacy.execute(
            "INSERT INTO canonical_terms (id, domain, term_type, entity_type, "
            " canonical_name, first_seen_in) "
            "VALUES (1, 'rag', 'entity', 'method', 'BookRAG', 'p1')"
        )
        # Two canonical-as-alias rows in different sections — both must
        # be filtered.
        legacy.execute(
            "INSERT INTO term_aliases (term_id, alias, source_paper, "
            " source_breadcrumb, match_tier) "
            "VALUES (1, 'BookRAG', 'p1', '# Intro', 1)"
        )
        legacy.execute(
            "INSERT INTO term_aliases (term_id, alias, source_paper, "
            " source_breadcrumb, match_tier) "
            "VALUES (1, 'BookRAG', 'p1', '# Method', 1)"
        )
        # Two breadcrumb-variants of a real synonym — must dedupe to ONE
        # row on the new 3-col PK.
        legacy.execute(
            "INSERT INTO term_aliases (term_id, alias, source_paper, "
            " source_breadcrumb, match_tier) "
            "VALUES (1, 'Book-RAG', 'p1', '# Intro', 3)"
        )
        legacy.execute(
            "INSERT INTO term_aliases (term_id, alias, source_paper, "
            " source_breadcrumb, match_tier) "
            "VALUES (1, 'Book-RAG', 'p1', '# Method', 3)"
        )
        legacy.commit()
    finally:
        legacy.close()

    from _system.db.connection import get_conn
    conn = get_conn(db_path)
    try:
        init_db(conn)
        rows = conn.execute(
            "SELECT term_id, alias, source_paper, match_tier "
            "  FROM term_aliases ORDER BY alias"
        ).fetchall()
        assert rows == [(1, "Book-RAG", "p1", 3)], rows
        # source_breadcrumb column is gone.
        cols = [c[1] for c in conn.execute(
            "PRAGMA table_info(term_aliases)"
        ).fetchall()]
        assert "source_breadcrumb" not in cols
    finally:
        conn.close()


def test_init_db_backfills_collections_from_legacy_papers(conn):
    """Papers predating the `collections` table must be registered by `init_db`.

    Simulates an old database: a paper already has a (domain, collection)
    pair but no row in `collections`. `init_db` runs its one-shot backfill
    (``INSERT OR IGNORE INTO collections SELECT DISTINCT ...``) and the
    pair should appear.
    """
    conn.execute(
        "INSERT OR IGNORE INTO domains (name) VALUES (?)", ("rag",)
    )
    conn.execute(
        "INSERT INTO papers (arxiv_id, paper_name, title, authors, date, abstract, "
        "domain, collection, content_hash, pdf_url, ingested_at, status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("2401.00001", "legacy_paper", "t", "[]", "2026-04-20", "a",
         "rag", "hierarchical indexing", "h", "http://x",
         "2026-04-20T00:00:00", "classified"),
    )
    # Drop the backfilled row the fixture already created (if any) to prove
    # init_db is what puts it back.
    conn.execute("DELETE FROM collections")

    init_db(conn)

    row = conn.execute(
        "SELECT domain, name, description FROM collections "
        " WHERE domain = ? AND name = ?",
        ("rag", "hierarchical indexing"),
    ).fetchone()
    assert row is not None
    assert row[2] is None  # legacy rows land with NULL description


def test_init_db_drops_legacy_abstracts_fts5(tmp_path):
    """abstracts FTS5 virtual table was retired (the # Abstract chunk in
    ``sections`` covers the same text). init_db must DROP the virtual
    table on any DB that predates the removal so the schema converges,
    and the implicit DROP must take its shadow tables with it.
    """
    db_path = tmp_path / "legacy_abs.db"
    legacy = sqlite3.connect(db_path)
    try:
        legacy.execute(
            "CREATE VIRTUAL TABLE abstracts USING fts5("
            "  paper_id UNINDEXED, domain, paper_name, collection, title, body,"
            "  tokenize='porter unicode61'"
            ")"
        )
        legacy.execute(
            "INSERT INTO abstracts (paper_id, domain, paper_name, collection, "
            "title, body) VALUES (1, 'rag', 'p1', 'c1', 't', 'body text')"
        )
        legacy.commit()
    finally:
        legacy.close()

    from _system.db.connection import get_conn
    conn = get_conn(db_path)
    try:
        init_db(conn)
        # Virtual table is gone.
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'abstracts'"
        ).fetchone()
        assert row is None
        # Shadow tables drop with the parent virtual table — none should remain.
        shadows = conn.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'abstracts\\_%' ESCAPE '\\'"
        ).fetchall()
        assert shadows == []
    finally:
        conn.close()


def test_init_db_drops_legacy_page_images(conn):
    """page_images was retired (LaTeXML strips page layout — chunks have no
    page index to correlate back to a render). init_db must DROP the table
    on any DB that predates the removal so the schema converges.
    """
    conn.execute(
        "CREATE TABLE page_images ("
        "  paper_id INTEGER NOT NULL,"
        "  page_number INTEGER NOT NULL,"
        "  image BLOB NOT NULL,"
        "  PRIMARY KEY(paper_id, page_number))"
    )

    init_db(conn)

    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'page_images'"
    ).fetchone()
    assert row is None


def test_papers_raw_html_accepts_large_payload(conn):
    """`raw_html` must exist and accept a multi-megabyte string."""
    big = "x" * (2 * 1024 * 1024)  # 2 MB
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES (?)", ("ml",))
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract, domain,
            content_hash, pdf_url, ingested_at, status, raw_html
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("2512.99999", "big_paper", "t", "[]", "2026-04-20", "a", "ml",
         "hash", "http://x", "2026-04-20T00:00:00", "fetched", big),
    )
    row = conn.execute(
        "SELECT length(raw_html) FROM papers WHERE paper_name = 'big_paper'"
    ).fetchone()
    assert row[0] == len(big)


def test_code_files_table_exists_with_unique_constraint(conn):
    """``code_files`` is a plain table with UNIQUE(repo_id, path)."""
    cols = {c[1] for c in conn.execute("PRAGMA table_info(code_files)").fetchall()}
    assert {"repo_id", "path", "language", "size_bytes", "content"} <= cols

    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES ('rag')")
    conn.execute(
        "INSERT INTO repos (repo_slug, url, host, owner, name, status, "
        "  ingested_at) VALUES (?, ?, 'github.com', 'o', 'r', 'resolved', 'd')",
        ("gh-o-r", "https://github.com/o/r"),
    )
    rid = conn.execute("SELECT id FROM repos WHERE repo_slug='gh-o-r'").fetchone()[0]
    conn.execute(
        "INSERT INTO code_files (repo_id, path, language, size_bytes, content) "
        "VALUES (?, ?, ?, ?, ?)",
        (rid, "main.py", "python", 5, "x=1\n"),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO code_files (repo_id, path, language, size_bytes, content) "
            "VALUES (?, ?, ?, ?, ?)",
            (rid, "main.py", "python", 6, "y=2\n"),
        )


def test_readmes_fts_virtual_table_exists(conn):
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'readmes_fts' AND type = 'table'"
    ).fetchone()
    assert row is not None
    # FTS5 must accept inserts and tokenize MATCH queries.
    conn.execute(
        "INSERT INTO readmes_fts (repo_id, repo_slug, domain, path, content) "
        "VALUES (1, 'gh-o-x', 'rag', 'README.md', 'mixture-of-experts training')"
    )
    n = conn.execute(
        "SELECT COUNT(*) FROM readmes_fts WHERE readmes_fts MATCH ?",
        ('"mixture-of-experts"',),
    ).fetchone()[0]
    assert n == 1


def test_papers_no_legacy_code_repo_columns(conn):
    """code_repo / code_repo_commit / code_repo_fetched_at have moved to
    the first-class ``repos`` table; the legacy columns must not exist."""
    cols = {c[1] for c in conn.execute("PRAGMA table_info(papers)").fetchall()}
    assert "code_repo" not in cols
    assert "code_repo_commit" not in cols
    assert "code_repo_fetched_at" not in cols


def test_repos_table_has_expected_columns(conn):
    cols = {c[1] for c in conn.execute("PRAGMA table_info(repos)").fetchall()}
    expected = {
        "id", "repo_slug", "url", "host", "owner", "name", "paper_id",
        "description", "default_branch", "commit_sha", "fetched_at",
        "ingested_at", "domain", "collection", "status", "needs_review",
        "file_count", "has_readme",
    }
    assert expected <= cols


def test_term_embeddings_metadata_filters(conn):
    """vec0 metadata columns (term_type/entity_type/domain) must be usable in WHERE.

    Seeds two rows with disjoint metadata so the filter has something to exclude;
    a silently-ignored filter would leak the second row and fail the test.
    """
    # Two embeddings at the same point in space so both are equally-ranked by KNN.
    vec = serialize_float32([1.0] + [0.0] * 383)
    conn.execute(
        "INSERT INTO canonical_terms (id, domain, term_type, entity_type, canonical_name, first_seen_in) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (42, "ml", "entity", "method", "Transformer", "paper_1"),
    )
    conn.execute(
        "INSERT INTO canonical_terms (id, domain, term_type, entity_type, canonical_name, first_seen_in) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (43, "bio", "collection", "", "Genomics", "paper_2"),
    )
    conn.execute(
        "INSERT INTO term_embeddings (term_id, embedding, term_type, entity_type, domain) "
        "VALUES (?, ?, ?, ?, ?)",
        (42, vec, "entity", "method", "ml"),
    )
    conn.execute(
        "INSERT INTO term_embeddings (term_id, embedding, term_type, entity_type, domain) "
        "VALUES (?, ?, ?, ?, ?)",
        (43, vec, "collection", "", "bio"),
    )

    # Each filter must return ONLY the matching row.
    for col, expected_val, expected_id in [
        ("term_type", "entity", 42),
        ("entity_type", "method", 42),
        ("domain", "ml", 42),
        ("term_type", "collection", 43),
        ("domain", "bio", 43),
    ]:
        rows = conn.execute(
            f"""
            SELECT term_id FROM term_embeddings
            WHERE embedding MATCH ? AND k = 5 AND {col} = ?
            """,
            (vec, expected_val),
        ).fetchall()
        ids = {r[0] for r in rows}
        assert ids == {expected_id}, (
            f"metadata filter on {col}={expected_val!r} returned {ids}, "
            f"expected exactly {{{expected_id}}}"
        )


# ===========================================================================
# papers invariant: classified+ rows must have domain AND collection
# ===========================================================================


def _insert_paper_raw(
    conn: sqlite3.Connection,
    *,
    paper_name: str,
    arxiv_id: str,
    status: str,
    domain: str | None,
    collection: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, ingested_at, status, domain, collection
        ) VALUES (?, ?, 't', '[]', '2024-01-01', 'abs',
                  ?, '2024-01-01T00:00:00+00:00', ?, ?, ?)
        """,
        (
            arxiv_id, paper_name,
            f"https://arxiv.org/pdf/{arxiv_id}",
            status, domain, collection,
        ),
    )


def test_invariant_classified_paper_requires_domain_and_collection_on_insert(conn):
    """A direct INSERT of a CLASSIFIED+ row missing domain or collection
    must be rejected by the schema-level trigger."""
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES ('rag')")
    conn.execute(
        "INSERT OR IGNORE INTO collections (domain, name, description) "
        "VALUES ('rag', 'hier', NULL)"
    )

    with pytest.raises(sqlite3.IntegrityError) as exc:
        _insert_paper_raw(
            conn, paper_name="bad1", arxiv_id="2401.10001",
            status="classified", domain=None, collection="hier",
        )
    assert "invariant" in str(exc.value).lower()

    with pytest.raises(sqlite3.IntegrityError):
        _insert_paper_raw(
            conn, paper_name="bad2", arxiv_id="2401.10002",
            status="classified", domain="rag", collection=None,
        )

    # Indexed-status row with both NULL is rejected too.
    with pytest.raises(sqlite3.IntegrityError):
        _insert_paper_raw(
            conn, paper_name="bad3", arxiv_id="2401.10003",
            status="indexed", domain=None, collection=None,
        )


def test_invariant_pre_classify_paper_can_have_null_domain_collection(conn):
    """FETCHED / CONVERTED rows are allowed to have NULL domain and
    collection — that's the natural state before classify_paper runs."""
    _insert_paper_raw(
        conn, paper_name="ok_fetched", arxiv_id="2401.10010",
        status="fetched", domain=None, collection=None,
    )
    _insert_paper_raw(
        conn, paper_name="ok_converted", arxiv_id="2401.10011",
        status="converted", domain=None, collection=None,
    )


def test_invariant_failed_paper_can_have_null_domain_collection(conn):
    """Terminal FAILED_HTML rows can carry NULLs — they will never be
    classified. (FAILED_REPO is now a repo-side terminal status.)"""
    _insert_paper_raw(
        conn, paper_name="ok_failed_html", arxiv_id="2401.10020",
        status="failed_html", domain=None, collection=None,
    )


def test_invariant_update_to_classified_without_domain_collection_rejected(conn):
    """Promoting a paper to CLASSIFIED via UPDATE without setting domain
    AND collection violates the invariant and is rejected."""
    conn.execute("INSERT OR IGNORE INTO domains (name) VALUES ('rag')")
    _insert_paper_raw(
        conn, paper_name="upd1", arxiv_id="2401.10030",
        status="converted", domain=None, collection=None,
    )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE papers SET status = 'classified' WHERE paper_name = 'upd1'"
        )

    # Setting only domain still leaves collection NULL — must reject.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE papers SET status = 'classified', domain = 'rag' "
            " WHERE paper_name = 'upd1'"
        )

    # Setting both succeeds.
    conn.execute(
        "INSERT OR IGNORE INTO collections (domain, name, description) "
        "VALUES ('rag', 'hier', NULL)"
    )
    conn.execute(
        "UPDATE papers SET status = 'classified', domain = 'rag', "
        "collection = 'hier' WHERE paper_name = 'upd1'"
    )
    row = conn.execute(
        "SELECT status, domain, collection FROM papers WHERE paper_name = 'upd1'"
    ).fetchone()
    assert row == ("classified", "rag", "hier")
