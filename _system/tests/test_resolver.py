"""Tests for the 5-tier term resolver (_system.resolution.resolver)."""
from __future__ import annotations

import sqlite3

import pytest
import sqlite_vec

from _system.resolution import resolver as resolver_mod
from _system.resolution.resolver import ResolvedTerm, resolve


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _seed_canonical(
    conn: sqlite3.Connection,
    *,
    canonical_name: str,
    domain: str = "rag",
    term_type: str = "entity",
    entity_type: str | None = "method",
    first_seen_in: str = "seed_paper",
) -> int:
    """Insert one canonical_terms row and return the new id.

    ``None`` is coerced to ``""`` to match the resolver's convention
    (sqlite-vec vec0 metadata filters require simple equality, and
    SQLite's UNIQUE treats NULL as distinct).
    """
    cur = conn.execute(
        """
        INSERT INTO canonical_terms
            (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES (?, ?, ?, ?, ?)
        """,
        (domain, term_type, entity_type or "", canonical_name, first_seen_in),
    )
    return cur.lastrowid


def _seed_embedding(
    conn: sqlite3.Connection,
    *,
    term_id: int,
    vector: list[float],
    domain: str = "rag",
    term_type: str = "entity",
    entity_type: str | None = "method",
) -> None:
    conn.execute(
        """
        INSERT INTO term_embeddings
            (term_id, embedding, term_type, entity_type, domain)
        VALUES (?, ?, ?, ?, ?)
        """,
        (term_id, sqlite_vec.serialize_float32(vector), term_type, entity_type or "", domain),
    )


def _alias_rows(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT term_id, alias, source_paper, match_tier FROM term_aliases "
        "ORDER BY term_id, alias, source_paper"
    ).fetchall()


def _canonical_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM canonical_terms").fetchone()[0]


def _embedding_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM term_embeddings").fetchone()[0]


class _FakeEmbedder:
    """Deterministic fake embedder for tier-4 tests.

    Stores a manually-assigned vector per input string. Unknown inputs
    produce a default "zero-ish" vector so tier-4 will miss.
    """

    def __init__(self) -> None:
        self._by_text: dict[str, list[float]] = {}
        self.embed_calls: list[str] = []

    def preset(self, text: str, vector: list[float]) -> None:
        self._by_text[text] = vector

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        if text in self._by_text:
            return list(self._by_text[text])
        # Orthogonal default so KNN against seeded vectors returns large distances.
        v = [0.0] * 384
        v[-1] = 1.0
        return v

    def embed_batch(self, texts):  # pragma: no cover - unused in tests
        return [self.embed(t) for t in texts]


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


@pytest.fixture(autouse=True)
def _clear_pending(conn):
    """Reset the module-level pending-rebuild map before every test."""
    resolver_mod.pending_fts_rebuilds(conn).clear()
    yield
    resolver_mod.pending_fts_rebuilds(conn).clear()


# ---------------------------------------------------------------------------
# Tier 1: exact
# ---------------------------------------------------------------------------


class TestTier1Exact:
    def test_hit_no_alias_insert(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        result = resolve(
            conn,
            "BookRAG",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert isinstance(result, ResolvedTerm)
        assert result.term_id == term_id
        assert result.canonical_name == "BookRAG"
        assert result.matched_via == "tier1"
        assert result.created_new is False
        assert _alias_rows(conn) == []
        assert resolver_mod.pending_fts_rebuilds(conn) == set()


# ---------------------------------------------------------------------------
# Tier 2: normalized alias match
# ---------------------------------------------------------------------------


class TestTier2Normalized:
    def test_canonical_normalize_hit_skips_alias_insert(self, conn, fake_embedder):
        """Tier 2 via canonical normalize-match: alias filter rejects the insert
        because ``normalize(alias) == normalize(canonical)`` by definition.
        """
        term_id = _seed_canonical(conn, canonical_name="bookrag")
        # "BookRAG" normalizes to "bookrag", same as the canonical — tier 1
        # misses on case, tier 2 catches it via the normalize pool.
        result = resolve(
            conn,
            "BookRAG",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.term_id == term_id
        assert result.matched_via == "tier2"
        assert result.created_new is False
        assert _alias_rows(conn) == []

    def test_existing_alias_normalize_hit_inserts_new_alias(self, conn, fake_embedder):
        """Tier 2 can also match via a previously-inserted alias row whose
        normalized form differs from the canonical. In that case the new raw
        form is eligible for its own alias insert (filter passes).
        """
        term_id = _seed_canonical(conn, canonical_name="Hierarchical Indexing")
        # Pre-seed an alias whose normalize form ("tree retrieval") differs
        # from the canonical's normalize form ("hierarchical indexing").
        conn.execute(
            "INSERT INTO term_aliases (term_id, alias, source_paper, match_tier) "
            "VALUES (?, ?, ?, ?)",
            (term_id, "tree-retrieval", "pX", 2),
        )
        result = resolve(
            conn,
            "Tree Retrieval",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.term_id == term_id
        assert result.canonical_name == "Hierarchical Indexing"
        assert result.matched_via == "tier2"
        rows = _alias_rows(conn)
        assert (term_id, "tree-retrieval", "pX", 2) in rows
        assert (term_id, "Tree Retrieval", "p1", 2) in rows
        assert term_id in resolver_mod.pending_fts_rebuilds(conn)


# ---------------------------------------------------------------------------
# Tier 3: rapidfuzz fuzzy
# ---------------------------------------------------------------------------


class TestTier3Fuzzy:
    def test_plural_variant_matches(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        result = resolve(
            conn,
            "BookRAGs",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.term_id == term_id
        assert result.matched_via == "tier3"
        rows = _alias_rows(conn)
        assert rows == [(term_id, "BookRAGs", "p1", 3)]

    def test_hyphenated_variant_matches_via_tier3(self, conn, fake_embedder):
        """``Book-RAG`` normalizes to ``book rag`` (not ``bookrag``), so tier 2
        misses on normalize-equality. Rapidfuzz at tier 3 then catches it.
        """
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        result = resolve(
            conn,
            "Book-RAG",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.term_id == term_id
        assert result.matched_via == "tier3"
        rows = _alias_rows(conn)
        assert rows == [(term_id, "Book-RAG", "p1", 3)]

    def test_prefilter_narrows_candidate_pool(self, conn, fake_embedder, monkeypatch):
        """Tier 3 must pre-filter in SQL, not score all 5000 rows in Python.

        Seeds are constructed so that only a tiny fraction can pass the
        first-letter='b' OR ``|len - 7| <= 3`` prefilter. If the resolver
        forgets the prefilter, the Python-side candidate count will equal
        (or approach) 5000.
        """
        # Use a first letter far from 'b' and a length far from 7 so neither
        # prefilter clause will admit the noise rows.
        for i in range(5000):
            full_name = "z" * 20 + f"_{i:05d}"  # length 26+, first letter 'z'
            conn.execute(
                "INSERT INTO canonical_terms "
                "(domain, term_type, entity_type, canonical_name, first_seen_in) "
                "VALUES (?, ?, ?, ?, ?)",
                ("rag", "entity", "method", full_name, "seed"),
            )
        # Target canonical passes both clauses (starts with 'b', length ~14).
        _seed_canonical(conn, canonical_name="bookrag_target")

        # Intercept the resolver's rapidfuzz candidate-loading step. The
        # resolver must call ``_tier3_candidates`` (or equivalent) and pass
        # the result to rapidfuzz; we assert the candidate count is much
        # smaller than 5001.
        original = resolver_mod._tier3_candidates

        captured: dict[str, int] = {}

        def spy(conn, *, domain, term_type, entity_type, raw):
            rows = original(
                conn,
                domain=domain,
                term_type=term_type,
                entity_type=entity_type,
                raw=raw,
            )
            captured["n"] = len(rows)
            return rows

        monkeypatch.setattr(resolver_mod, "_tier3_candidates", spy)

        resolve(
            conn,
            "bookrag",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert "n" in captured, "resolver did not call the candidate loader"
        # Tight bound: SQL prefilter must drop almost everything.
        assert captured["n"] < 50, (
            f"tier-3 prefilter returned {captured['n']} rows out of 5001; "
            "expected a tight SQL-side subset, not a near-full scan"
        )


# ---------------------------------------------------------------------------
# Tier 4: sqlite-vec KNN
# ---------------------------------------------------------------------------


class TestTier4Embedding:
    def test_near_neighbor_above_threshold_hits(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="hierarchical indexing")
        # Unit vector; nearly-identical query vector has cos ~ 1.
        seed_vec = [0.0] * 384
        seed_vec[0] = 1.0
        _seed_embedding(conn, term_id=term_id, vector=seed_vec)

        query_vec = [0.0] * 384
        query_vec[0] = 1.0
        fake_embedder.preset("tree retrieval", query_vec)

        result = resolve(
            conn,
            "tree retrieval",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.term_id == term_id
        assert result.canonical_name == "hierarchical indexing"
        assert result.matched_via == "tier4"
        rows = _alias_rows(conn)
        assert rows == [(term_id, "tree retrieval", "p1", 4)]

    def test_threshold_rejects_below_085_cosine(self, conn, fake_embedder):
        """Query and seed vectors at cos≈0.5 (distance≈1.0) — below 0.85 threshold."""
        import math

        term_id = _seed_canonical(conn, canonical_name="hierarchical indexing")
        seed_vec = [0.0] * 384
        seed_vec[0] = 1.0
        _seed_embedding(conn, term_id=term_id, vector=seed_vec)

        # 60° apart: cos=0.5. d^2 = 2 - 2*0.5 = 1.0, above 0.3 threshold.
        angle = math.radians(60)
        query_vec = [0.0] * 384
        query_vec[0] = math.cos(angle)
        query_vec[1] = math.sin(angle)
        fake_embedder.preset("unrelated thing", query_vec)

        # No tier 1/2/3 hit, tier 4 below threshold → falls through to tier 5.
        result = resolve(
            conn,
            "unrelated thing",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier5"
        assert result.created_new is True

    def test_threshold_accepts_just_above_085_cosine(self, conn, fake_embedder):
        """Query at exactly the seed vector — cos=1.0, distance=0. Always a hit."""
        import math

        term_id = _seed_canonical(conn, canonical_name="hierarchical indexing")
        seed_vec = [0.0] * 384
        seed_vec[0] = 1.0
        _seed_embedding(conn, term_id=term_id, vector=seed_vec)

        # ~30° apart: cos≈0.866, just above threshold. d^2 = 2 - 2*cos ≈ 0.268.
        angle = math.radians(30)
        query_vec = [0.0] * 384
        query_vec[0] = math.cos(angle)
        query_vec[1] = math.sin(angle)
        fake_embedder.preset("close enough", query_vec)

        result = resolve(
            conn,
            "close enough",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier4"
        assert result.term_id == term_id


# ---------------------------------------------------------------------------
# Tier 5: insert new canonical + embedding
# ---------------------------------------------------------------------------


class TestTier5NewCanonical:
    def test_creates_canonical_and_embedding(self, conn, fake_embedder):
        # Preset the embedder so the new canonical gets a stable vector.
        v = [0.0] * 384
        v[5] = 1.0
        fake_embedder.preset("NovelMethodXYZ", v)

        before_canon = _canonical_count(conn)
        before_emb = _embedding_count(conn)

        result = resolve(
            conn,
            "NovelMethodXYZ",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier5"
        assert result.created_new is True
        assert result.canonical_name == "NovelMethodXYZ"

        assert _canonical_count(conn) == before_canon + 1
        assert _embedding_count(conn) == before_emb + 1
        assert _alias_rows(conn) == []
        # Tier 5 must NOT enqueue an FTS rebuild — the row is brand new
        # and will be inserted fresh by index_paper.
        assert resolver_mod.pending_fts_rebuilds(conn) == set()

    def test_integrity_error_falls_back_to_tier1(self, conn, db_path, fake_embedder):
        """Simulate a concurrent writer by committing the sibling row on a
        second connection just before tier 5's INSERT would have succeeded.

        A second connection is required because the resolver wraps the two
        tier-5 inserts in a ``SAVEPOINT``; if the sibling were inserted via
        the same connection it would be rolled back along with the failed
        primary insert.
        """
        from _system.db.connection import get_conn

        class _RaceConn:
            def __init__(self, inner: sqlite3.Connection, db_path) -> None:
                self._inner = inner
                self._db_path = db_path
                self._raised = False

            def execute(self, sql, params=()):
                if not self._raised and "INSERT INTO canonical_terms" in sql:
                    # Commit the concurrent row on a separate connection so
                    # it survives the savepoint rollback on the wrapped conn.
                    other = get_conn(self._db_path)
                    try:
                        other.execute("BEGIN")
                        other.execute(sql, params)
                        other.execute("COMMIT")
                    finally:
                        other.close()
                    self._raised = True
                    raise sqlite3.IntegrityError(
                        "UNIQUE constraint failed: canonical_terms"
                    )
                return self._inner.execute(sql, params)

            def __getattr__(self, item):
                return getattr(self._inner, item)

        v = [0.0] * 384
        v[3] = 1.0
        fake_embedder.preset("RacedTerm", v)

        wrapped = _RaceConn(conn, db_path)
        result = resolve(
            wrapped,
            "RacedTerm",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier1"
        assert result.created_new is False
        assert result.canonical_name == "RacedTerm"
        # Exactly one canonical_terms row exists for this scope.
        rows = conn.execute(
            "SELECT id FROM canonical_terms WHERE canonical_name = 'RacedTerm'"
        ).fetchall()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Alias acceptance filter
# ---------------------------------------------------------------------------


class TestAliasAcceptance:
    def test_rejects_alias_shorter_than_three(self, conn, fake_embedder):
        # Seed a canonical whose normalize form matches "ab".
        term_id = _seed_canonical(conn, canonical_name="AB")
        result = resolve(
            conn,
            "ab",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        # Should hit tier 2 (same normalized form), but alias row must be skipped.
        assert result.term_id == term_id
        assert result.matched_via == "tier2"
        assert _alias_rows(conn) == []

    def test_rejects_alias_identical_to_canonical_after_normalize(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        # Raw is "bookrag" — differs only in case from canonical; normalize collapses both.
        result = resolve(
            conn,
            "bookrag",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.term_id == term_id
        # matched_via is tier2 (normalized match) but filter rejects the alias row.
        assert result.matched_via == "tier2"
        assert _alias_rows(conn) == []


# ---------------------------------------------------------------------------
# Scope isolation
# ---------------------------------------------------------------------------


class TestScopeIsolation:
    def test_different_domains_resolve_independently(self, conn, fake_embedder):
        rag_id = _seed_canonical(conn, canonical_name="X", domain="rag")
        agents_id = _seed_canonical(conn, canonical_name="X", domain="agents")
        r1 = resolve(
            conn,
            "X",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        r2 = resolve(
            conn,
            "X",
            domain="agents",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert r1.term_id == rag_id
        assert r2.term_id == agents_id

    def test_entity_type_null_for_non_entity_scope(self, conn, fake_embedder):
        """``term_type='collection'`` implies ``entity_type IS NULL``."""
        cid = _seed_canonical(
            conn,
            canonical_name="agentic-rag",
            domain="rag",
            term_type="collection",
            entity_type=None,
        )
        r = resolve(
            conn,
            "agentic-rag",
            domain="rag",
            term_type="collection",
            entity_type=None,
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert r.term_id == cid
        assert r.matched_via == "tier1"


# ---------------------------------------------------------------------------
# Multi-paper alias provenance
# ---------------------------------------------------------------------------


class TestMultiPaperProvenance:
    def test_same_alias_different_papers_yields_two_rows(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        r1 = resolve(conn, "Book-RAG", domain="rag", term_type="entity",
                     entity_type="method", source_paper="p1", embedder=fake_embedder)
        r2 = resolve(conn, "Book-RAG", domain="rag", term_type="entity",
                     entity_type="method", source_paper="p2", embedder=fake_embedder)
        # "Book-RAG" first resolves via tier 3 (fuzz). After that alias is
        # persisted, the second call normalizes to "book rag" and matches the
        # existing alias via tier 2, so the second row gets match_tier=2.
        assert r1.matched_via == "tier3"
        assert r2.matched_via == "tier2"
        rows = _alias_rows(conn)
        assert (term_id, "Book-RAG", "p1", 3) in rows
        assert (term_id, "Book-RAG", "p2", 2) in rows
        assert len(rows) == 2

    def test_repeated_same_alias_same_paper_is_noop(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        resolve(conn, "Book-RAG", domain="rag", term_type="entity",
                entity_type="method", source_paper="p1", embedder=fake_embedder)
        resolve(conn, "Book-RAG", domain="rag", term_type="entity",
                entity_type="method", source_paper="p1", embedder=fake_embedder)
        rows = _alias_rows(conn)
        # Second call hits tier 2 via existing alias, but the composite PK
        # (term_id, alias, source_paper) makes INSERT OR IGNORE a no-op.
        assert rows == [(term_id, "Book-RAG", "p1", 3)]


# ---------------------------------------------------------------------------
# ResolvedTerm shape
# ---------------------------------------------------------------------------


class TestResolvedTermShape:
    def test_is_named_tuple_with_expected_fields(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        r = resolve(conn, "BookRAG", domain="rag", term_type="entity",
                    entity_type="method", source_paper="p1", embedder=fake_embedder)
        # NamedTuple: positional and attribute access both work.
        assert r[0] == r.term_id == term_id
        assert r[1] == r.canonical_name == "BookRAG"
        assert r[2] == r.matched_via == "tier1"
        assert r[3] == r.created_new is False
        assert isinstance(r.term_id, int)
        assert isinstance(r.canonical_name, str)
        assert isinstance(r.matched_via, str)
        assert isinstance(r.created_new, bool)


# ---------------------------------------------------------------------------
# Deferred terms_fts rebuild queue
# ---------------------------------------------------------------------------


class TestPendingFtsRebuilds:
    def test_enqueued_on_alias_insert(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        resolve(conn, "Book-RAG", domain="rag", term_type="entity",
                entity_type="method", source_paper="p1", embedder=fake_embedder)
        assert resolver_mod.pending_fts_rebuilds(conn) == {term_id}

    def test_tier1_does_not_enqueue(self, conn, fake_embedder):
        _seed_canonical(conn, canonical_name="BookRAG")
        resolve(conn, "BookRAG", domain="rag", term_type="entity",
                entity_type="method", source_paper="p1", embedder=fake_embedder)
        assert resolver_mod.pending_fts_rebuilds(conn) == set()

    def test_tier5_does_not_enqueue(self, conn, fake_embedder):
        v = [0.0] * 384
        v[2] = 1.0
        fake_embedder.preset("NovelX", v)
        resolve(conn, "NovelX", domain="rag", term_type="entity",
                entity_type="method", source_paper="p1", embedder=fake_embedder)
        assert resolver_mod.pending_fts_rebuilds(conn) == set()

    def test_per_connection_isolation(self, conn, db_path):
        """Different connections have independent pending sets."""
        from _system.db.connection import get_conn
        from _system.db.migrations import init_db

        other = get_conn(db_path)
        try:
            init_db(other)
            resolver_mod.pending_fts_rebuilds(conn).add(99)
            assert 99 in resolver_mod.pending_fts_rebuilds(conn)
            assert 99 not in resolver_mod.pending_fts_rebuilds(other)
        finally:
            other.close()
