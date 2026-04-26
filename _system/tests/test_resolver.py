"""Tests for the 5-tier term resolver (_system.resolution.resolver)."""
from __future__ import annotations

import logging
import math
import sqlite3

import pytest
import sqlite_vec

from _system.resolution import resolver as resolver_mod
from _system.resolution.resolver import ResolvedTerm, resolve


_RESOLVER_LOGGER = "lodestone._system.resolution.resolver"


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
    entity_type_score: float = 0.0,
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
            (domain, term_type, entity_type, entity_type_score,
             canonical_name, first_seen_in)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            domain,
            term_type,
            entity_type or "",
            entity_type_score,
            canonical_name,
            first_seen_in,
        ),
    )
    return cur.lastrowid


def _seed_entity_row(
    conn: sqlite3.Connection,
    *,
    paper_name: str,
    domain: str,
    entity_name: str,
    entity_type: str,
    term_id: int,
) -> int:
    """Insert one ``term_aliases`` appearance row tied to an existing
    canonical and a freshly-created ``papers`` row.

    Used by flip tests to model a historical mention that predates the
    flip. ``entity_type`` is unused (entity_type lives on the canonical
    after the merge) but kept in the signature for caller readability.
    """
    del entity_type  # kept in signature for caller readability; unused
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, html_source, ingested_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"arxiv_{paper_name}",
            paper_name,
            "t",
            "a",
            "2024-01-01",
            "abs",
            f"https://arxiv.org/pdf/arxiv_{paper_name}",
            "arxiv",
            "2024-01-01T00:00:00+00:00",
            "extracted",
        ),
    )
    paper_id = cur.lastrowid
    conn.execute(
        """
        INSERT OR IGNORE INTO term_aliases
            (term_id, alias, source_paper, source_breadcrumb, match_tier)
        VALUES (?, ?, ?, '# Section', 1)
        """,
        (term_id, entity_name, paper_name),
    )
    return paper_id


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
        "SELECT term_id, alias, source_paper, source_breadcrumb, match_tier "
        "  FROM term_aliases "
        " ORDER BY term_id, alias, source_paper, source_breadcrumb"
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
    def test_hit_writes_appearance_alias(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        result = resolve(
            conn,
            "BookRAG",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            source_breadcrumb="# Method",
            embedder=fake_embedder,
        )
        assert isinstance(result, ResolvedTerm)
        assert result.term_id == term_id
        assert result.canonical_name == "BookRAG"
        assert result.matched_via == "tier1"
        assert result.created_new is False
        # Tier-1 hits write an appearance row with alias == canonical_name
        # so term_aliases is the complete per-paper appearance log.
        assert _alias_rows(conn) == [(term_id, "BookRAG", "p1", "# Method", 1)]
        assert resolver_mod.pending_fts_rebuilds(conn) == {term_id}


# ---------------------------------------------------------------------------
# Tier 2: normalized alias match
# ---------------------------------------------------------------------------


class TestTier2Normalized:
    def test_canonical_normalize_hit_writes_appearance_row(self, conn, fake_embedder):
        """Tier 2 via canonical normalize-match writes an appearance row
        carrying the raw form (case preserved) — even when normalize(raw)
        equals normalize(canonical). The appearance log records every
        mention regardless of whether the surface form coincides with the
        canonical.
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
            source_breadcrumb="# Body",
            embedder=fake_embedder,
        )
        assert result.term_id == term_id
        assert result.matched_via == "tier2"
        assert result.created_new is False
        assert _alias_rows(conn) == [(term_id, "BookRAG", "p1", "# Body", 2)]

    def test_existing_alias_normalize_hit_inserts_new_alias(self, conn, fake_embedder):
        """Tier 2 can also match via a previously-inserted alias row whose
        normalized form differs from the canonical. The new raw form lands
        as its own appearance row.
        """
        term_id = _seed_canonical(conn, canonical_name="Hierarchical Indexing")
        # Pre-seed an alias whose normalize form ("tree retrieval") differs
        # from the canonical's normalize form ("hierarchical indexing").
        conn.execute(
            "INSERT INTO term_aliases "
            "  (term_id, alias, source_paper, source_breadcrumb, match_tier) "
            "VALUES (?, ?, ?, ?, ?)",
            (term_id, "tree-retrieval", "pX", "", 2),
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
        assert (term_id, "tree-retrieval", "pX", "", 2) in rows
        assert (term_id, "Tree Retrieval", "p1", "", 2) in rows
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
        assert rows == [(term_id, "BookRAGs", "p1", "", 3)]

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
        assert rows == [(term_id, "Book-RAG", "p1", "", 3)]

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

        def spy(conn, *, domain, term_type, raw):
            rows = original(
                conn,
                domain=domain,
                term_type=term_type,
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
# Tier 4: sqlite-vec top-K + gradient walk
#
# All tests below construct candidate vectors as ``v[0] = cos(θ); v[axis] =
# sin(θ)`` so the dot product with a query at ``v[0] = 1`` is exactly cos(θ).
# Different ``axis`` values per seed keep the seed embeddings linearly
# distinct without affecting their cosine-against-query.
# ---------------------------------------------------------------------------


def _unit_vec_with_dot(dot_with_query: float, axis: int) -> list[float]:
    """Return a 384-dim unit vector whose dot product with the canonical query
    (``v[0]=1``) is ``dot_with_query``. Energy in dim 0 sets the dot; the
    remainder is parked in ``axis`` so distinct seeds stay orthogonal in their
    non-query components.
    """
    if axis == 0:
        raise ValueError("axis must be != 0 to keep the residual orthogonal to query")
    vec = [0.0] * 384
    vec[0] = dot_with_query
    residual = math.sqrt(max(0.0, 1.0 - dot_with_query * dot_with_query))
    vec[axis] = residual
    return vec


class TestTier4Embedding:
    def test_sharp_drop_after_top1_merges(self, conn, fake_embedder):
        """Top-1 cos≈0.985, top-2 cos≈0.643 → ratio 0.653 < 0.88 → 1 survivor."""
        a_id = _seed_canonical(conn, canonical_name="hierarchical indexing")
        b_id = _seed_canonical(conn, canonical_name="distant concept")
        _seed_embedding(conn, term_id=a_id, vector=_unit_vec_with_dot(math.cos(math.radians(10)), axis=1))
        _seed_embedding(conn, term_id=b_id, vector=_unit_vec_with_dot(math.cos(math.radians(50)), axis=2))

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
        assert result.matched_via == "tier4"
        assert result.term_id == a_id
        assert result.canonical_name == "hierarchical indexing"
        rows = _alias_rows(conn)
        assert rows == [(a_id, "tree retrieval", "p1", "", 4)]

    def test_two_close_candidates_fall_through_to_tier5(self, conn, fake_embedder):
        """Top-1 cos≈0.999, top-2 cos≈0.998 → ratio ≈ 1.0 → 2 survivors → tier 5."""
        a_id = _seed_canonical(conn, canonical_name="adaptive RRF")
        b_id = _seed_canonical(conn, canonical_name="adaptive RRF fusion")
        _seed_embedding(conn, term_id=a_id, vector=_unit_vec_with_dot(math.cos(math.radians(2)), axis=1))
        _seed_embedding(conn, term_id=b_id, vector=_unit_vec_with_dot(math.cos(math.radians(3)), axis=2))

        query_vec = [0.0] * 384
        query_vec[0] = 1.0
        fake_embedder.preset("adaptive ranking fusion", query_vec)

        before_canon = _canonical_count(conn)
        result = resolve(
            conn,
            "adaptive ranking fusion",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier5"
        assert result.created_new is True
        assert result.term_id != a_id and result.term_id != b_id
        assert _canonical_count(conn) == before_canon + 1
        # Neither pre-existing canonical absorbed an alias.
        for tid, _alias, _src, _bc, _tier in _alias_rows(conn):
            assert tid not in (a_id, b_id)

    def test_top1_below_floor_falls_through_to_tier5(self, conn, fake_embedder):
        """Top-1 cos≈0.643 < 0.70 floor → no merge, even with no other candidates."""
        a_id = _seed_canonical(conn, canonical_name="hierarchical indexing")
        _seed_embedding(conn, term_id=a_id, vector=_unit_vec_with_dot(1.0, axis=1))

        query_vec = [0.0] * 384
        query_vec[0] = math.cos(math.radians(50))
        query_vec[1] = math.sin(math.radians(50))
        fake_embedder.preset("unrelated thing", query_vec)

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
        # The original canonical should not have absorbed this raw form.
        assert (a_id, "unrelated thing", "p1", "", 4) not in _alias_rows(conn)

    def test_uniform_decline_all_survive_falls_through_to_tier5(self, conn, fake_embedder):
        """K=10 candidates with every consecutive ratio ≥ 0.88 → all survive →
        no sharp drop → tier 5. 'Everything is mildly similar' isn't a merge.
        """
        # Hand-picked ramp where every adjacent ratio ≥ 0.88. Only the first
        # value is above the 0.70 floor — but the floor only gates top-1; the
        # walk continues regardless.
        ramp = [0.99, 0.92, 0.86, 0.80, 0.75, 0.70, 0.66, 0.62, 0.58, 0.55]
        ids: list[int] = []
        for i, c in enumerate(ramp):
            cid = _seed_canonical(conn, canonical_name=f"ramp{i}")
            ids.append(cid)
            _seed_embedding(conn, term_id=cid, vector=_unit_vec_with_dot(c, axis=i + 1))

        query_vec = [0.0] * 384
        query_vec[0] = 1.0
        fake_embedder.preset("ramp probe", query_vec)

        before_canon = _canonical_count(conn)
        result = resolve(
            conn,
            "ramp probe",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier5"
        assert result.created_new is True
        assert _canonical_count(conn) == before_canon + 1
        # No aliases attached to any ramp canonical.
        for tid, _alias, _src, _bc, _tier in _alias_rows(conn):
            assert tid not in ids

    def test_just_above_gradient_single_survivor_merges(self, conn, fake_embedder):
        """top1=0.90, top2=0.79; ratio 0.878 < 0.88 → top2 fails → 1 survivor."""
        a_id = _seed_canonical(conn, canonical_name="winner")
        b_id = _seed_canonical(conn, canonical_name="runner-up")
        _seed_embedding(conn, term_id=a_id, vector=_unit_vec_with_dot(0.90, axis=1))
        _seed_embedding(conn, term_id=b_id, vector=_unit_vec_with_dot(0.79, axis=2))

        query_vec = [0.0] * 384
        query_vec[0] = 1.0
        fake_embedder.preset("close to A", query_vec)

        result = resolve(
            conn,
            "close to A",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier4"
        assert result.term_id == a_id

    def test_just_below_gradient_two_survivors_fall_through(self, conn, fake_embedder):
        """top1=0.90, top2=0.80; ratio 0.889 ≥ 0.88 → top2 passes → 2 survivors."""
        a_id = _seed_canonical(conn, canonical_name="winner")
        b_id = _seed_canonical(conn, canonical_name="runner-up")
        _seed_embedding(conn, term_id=a_id, vector=_unit_vec_with_dot(0.90, axis=1))
        _seed_embedding(conn, term_id=b_id, vector=_unit_vec_with_dot(0.80, axis=2))

        query_vec = [0.0] * 384
        query_vec[0] = 1.0
        fake_embedder.preset("close to both", query_vec)

        result = resolve(
            conn,
            "close to both",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier5"
        assert result.created_new is True
        # Neither pre-existing canonical should have absorbed an alias.
        for tid, _alias, _src, _bc, _tier in _alias_rows(conn):
            assert tid not in (a_id, b_id)

    def test_single_candidate_above_floor_merges(self, conn, fake_embedder):
        """Degenerate case: only one canonical in scope, no runner-up to
        compare against. Falls back to 'top-1 above floor → merge'."""
        a_id = _seed_canonical(conn, canonical_name="lonely canonical")
        _seed_embedding(conn, term_id=a_id, vector=_unit_vec_with_dot(1.0, axis=1))

        # Query at cos ≈ 0.95 → above floor, no runner-up.
        query_vec = [0.0] * 384
        query_vec[0] = math.cos(math.radians(18))
        query_vec[1] = math.sin(math.radians(18))
        fake_embedder.preset("similar lonely", query_vec)

        result = resolve(
            conn,
            "similar lonely",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier4"
        assert result.term_id == a_id


# ---------------------------------------------------------------------------
# Tier 5: insert new canonical + embedding
# ---------------------------------------------------------------------------


class TestTier5NewCanonical:
    def test_creates_canonical_and_embedding(self, conn, fake_embedder):
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
            source_breadcrumb="# Method",
            embedder=fake_embedder,
        )
        assert result.matched_via == "tier5"
        assert result.created_new is True
        assert result.canonical_name == "NovelMethodXYZ"

        assert _canonical_count(conn) == before_canon + 1
        assert _embedding_count(conn) == before_emb + 1
        # Tier 5 mints the canonical AND seeds its first appearance row
        # so a paper that introduces a brand-new term still has a
        # term_aliases entry recording the mention site.
        assert _alias_rows(conn) == [
            (result.term_id, "NovelMethodXYZ", "p1", "# Method", 5)
        ]
        # Enqueued by the alias insert path (the canonical row itself
        # is brand-new and will be picked up by the indexer's first
        # rebuild over this term).
        assert resolver_mod.pending_fts_rebuilds(conn) == {result.term_id}

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

    def test_accepts_alias_identical_to_canonical_after_normalize(self, conn, fake_embedder):
        """Surface form whose normalized shape matches the canonical (e.g.
        case-only difference) still writes an appearance row — the log
        records every mention regardless of whether the surface form
        coincides with the canonical after normalization. The previous
        equality guard was removed when ``term_aliases`` became the
        appearance log.
        """
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        # Raw is "bookrag" — differs only in case from canonical; normalize collapses both.
        result = resolve(
            conn,
            "bookrag",
            domain="rag",
            term_type="entity",
            entity_type="method",
            source_paper="p1",
            source_breadcrumb="# Body",
            embedder=fake_embedder,
        )
        assert result.term_id == term_id
        assert result.matched_via == "tier2"
        assert _alias_rows(conn) == [(term_id, "bookrag", "p1", "# Body", 2)]


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
        assert (term_id, "Book-RAG", "p1", "", 3) in rows
        assert (term_id, "Book-RAG", "p2", "", 2) in rows
        assert len(rows) == 2

    def test_repeated_same_alias_same_paper_is_noop(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        resolve(conn, "Book-RAG", domain="rag", term_type="entity",
                entity_type="method", source_paper="p1", embedder=fake_embedder)
        resolve(conn, "Book-RAG", domain="rag", term_type="entity",
                entity_type="method", source_paper="p1", embedder=fake_embedder)
        rows = _alias_rows(conn)
        # Second call hits tier 2 via existing alias, but the composite PK
        # (term_id, alias, source_paper, source_breadcrumb) makes INSERT
        # OR IGNORE a no-op. Both calls share source_breadcrumb=''.
        assert rows == [(term_id, "Book-RAG", "p1", "", 3)]


# ---------------------------------------------------------------------------
# ResolvedTerm shape
# ---------------------------------------------------------------------------


class TestResolvedTermShape:
    def test_is_named_tuple_with_expected_fields(self, conn, fake_embedder):
        term_id = _seed_canonical(
            conn, canonical_name="BookRAG", entity_type="method"
        )
        r = resolve(conn, "BookRAG", domain="rag", term_type="entity",
                    entity_type="method", source_paper="p1", embedder=fake_embedder)
        # NamedTuple: positional and attribute access both work.
        assert r[0] == r.term_id == term_id
        assert r[1] == r.canonical_name == "BookRAG"
        assert r[2] == r.entity_type == "method"
        assert r[3] == r.entity_type_score == 0.0
        assert r[4] == r.matched_via == "tier1"
        assert r[5] == r.created_new is False
        assert isinstance(r.term_id, int)
        assert isinstance(r.canonical_name, str)
        assert isinstance(r.entity_type, str)
        assert isinstance(r.entity_type_score, float)
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

    def test_tier1_enqueues_via_appearance_alias(self, conn, fake_embedder):
        term_id = _seed_canonical(conn, canonical_name="BookRAG")
        resolve(conn, "BookRAG", domain="rag", term_type="entity",
                entity_type="method", source_paper="p1", embedder=fake_embedder)
        # Tier-1 hits write an alias row (the appearance log) and that
        # path enqueues the FTS rebuild like every other tier.
        assert resolver_mod.pending_fts_rebuilds(conn) == {term_id}

    def test_tier5_enqueues_via_appearance_alias(self, conn, fake_embedder):
        v = [0.0] * 384
        v[2] = 1.0
        fake_embedder.preset("NovelX", v)
        result = resolve(conn, "NovelX", domain="rag", term_type="entity",
                         entity_type="method", source_paper="p1",
                         embedder=fake_embedder)
        # Tier-5 mints the canonical and writes its first appearance row;
        # the alias-insert path enqueues an FTS rebuild for the new term.
        assert resolver_mod.pending_fts_rebuilds(conn) == {result.term_id}

    def test_flip_enqueues_even_when_alias_is_idempotent(
        self, conn, fake_embedder
    ):
        """A tier-1 hit that flips entity_type but writes an alias row
        identical to a pre-existing one (INSERT OR IGNORE no-op) must
        still queue the FTS rebuild — terms_fts.entity_type mirrors
        canonical_terms.entity_type and would otherwise stay stale.
        """
        term_id = _seed_canonical(
            conn, canonical_name="DPR", entity_type="method",
            entity_type_score=0.3,
        )
        # First resolve seeds the appearance row and enqueues normally.
        resolve(conn, "DPR", domain="rag", term_type="entity",
                entity_type="method", entity_type_score=0.3,
                source_paper="p1", source_breadcrumb="# S",
                embedder=fake_embedder)
        resolver_mod.pending_fts_rebuilds(conn).clear()

        # Second resolve repeats the same alias (PK collision → no-op
        # insert) but flips on a higher-score 'software' label.
        resolve(conn, "DPR", domain="rag", term_type="entity",
                entity_type="software", entity_type_score=0.9,
                source_paper="p1", source_breadcrumb="# S",
                embedder=fake_embedder)
        assert term_id in resolver_mod.pending_fts_rebuilds(conn)

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


# ---------------------------------------------------------------------------
# entity_type flip on higher-confidence re-encounter
# ---------------------------------------------------------------------------


def _read_canonical(
    conn: sqlite3.Connection, term_id: int
) -> tuple[str, float]:
    row = conn.execute(
        "SELECT entity_type, entity_type_score FROM canonical_terms WHERE id = ?",
        (term_id,),
    ).fetchone()
    return (row[0], float(row[1]))


class TestEntityTypeFlip:
    def test_tier1_flip_on_higher_score_different_type(
        self, conn, fake_embedder, caplog
    ):
        """Paper 1 inserts at score 0.4 / method; paper 2 resolves the same name
        as software at 0.8 — canonical flips and INFO log fires."""
        term_id = _seed_canonical(
            conn,
            canonical_name="DPR",
            entity_type="method",
            entity_type_score=0.4,
        )
        _seed_entity_row(
            conn,
            paper_name="p_old",
            domain="rag",
            entity_name="DPR",
            entity_type="method",
            term_id=term_id,
        )

        logger = logging.getLogger(_RESOLVER_LOGGER)
        logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger=_RESOLVER_LOGGER):
                r = resolve(
                    conn,
                    "DPR",
                    domain="rag",
                    term_type="entity",
                    entity_type="software",
                    entity_type_score=0.8,
                    source_paper="p2",
                    embedder=fake_embedder,
                )
        finally:
            logger.removeHandler(caplog.handler)

        assert r.term_id == term_id
        assert r.entity_type == "software"
        assert r.entity_type_score == pytest.approx(0.8)
        assert r.matched_via == "tier1"
        assert _read_canonical(conn, term_id) == ("software", pytest.approx(0.8))

        # The historical paper's mention now reads back the post-flip
        # entity_type via the canonical_terms JOIN — no per-row migration
        # is needed because entity_type is no longer duplicated on every
        # appearance.
        entity_type_old_paper = conn.execute(
            """
            SELECT ct.entity_type
              FROM term_aliases ta
              JOIN canonical_terms ct ON ct.id = ta.term_id
             WHERE ta.source_paper = ? AND ct.canonical_name = ?
             LIMIT 1
            """,
            ("p_old", "DPR"),
        ).fetchone()[0]
        assert entity_type_old_paper == "software"

        assert any(
            "entity_type flip" in rec.getMessage()
            and rec.levelname == "INFO"
            for rec in caplog.records
        ), "expected an INFO log entry describing the flip"

    def test_tier1_no_flip_on_lower_score(
        self, conn, fake_embedder, caplog
    ):
        term_id = _seed_canonical(
            conn,
            canonical_name="DPR",
            entity_type="software",
            entity_type_score=0.8,
        )
        logger = logging.getLogger(_RESOLVER_LOGGER)
        logger.addHandler(caplog.handler)
        try:
            with caplog.at_level(logging.INFO, logger=_RESOLVER_LOGGER):
                r = resolve(
                    conn,
                    "DPR",
                    domain="rag",
                    term_type="entity",
                    entity_type="method",
                    entity_type_score=0.4,
                    source_paper="p2",
                    embedder=fake_embedder,
                )
        finally:
            logger.removeHandler(caplog.handler)
        assert r.entity_type == "software"
        assert r.entity_type_score == pytest.approx(0.8)
        assert _read_canonical(conn, term_id) == ("software", pytest.approx(0.8))
        assert not any(
            "entity_type flip" in rec.getMessage() for rec in caplog.records
        )

    def test_tier1_no_flip_on_equal_score(self, conn, fake_embedder):
        """Strictly-greater gate: equal score does not flip."""
        term_id = _seed_canonical(
            conn,
            canonical_name="DPR",
            entity_type="method",
            entity_type_score=0.7,
        )
        r = resolve(
            conn,
            "DPR",
            domain="rag",
            term_type="entity",
            entity_type="software",
            entity_type_score=0.7,
            source_paper="p2",
            embedder=fake_embedder,
        )
        assert r.entity_type == "method"
        assert r.entity_type_score == pytest.approx(0.7)
        assert _read_canonical(conn, term_id) == ("method", pytest.approx(0.7))

    def test_tier1_no_flip_on_same_type(self, conn, fake_embedder):
        """Same type + higher score: score is NOT bumped. The stored score is
        the score at which the *type* was established; types matching means
        no evidence to reconsider, even if a higher-confidence mention lands.
        (Per handoff: 'If paper 2's resolved entity_type matches the stored
        one: no change.')"""
        term_id = _seed_canonical(
            conn,
            canonical_name="DPR",
            entity_type="method",
            entity_type_score=0.5,
        )
        r = resolve(
            conn,
            "DPR",
            domain="rag",
            term_type="entity",
            entity_type="method",
            entity_type_score=0.9,
            source_paper="p2",
            embedder=fake_embedder,
        )
        assert r.entity_type == "method"
        assert r.entity_type_score == pytest.approx(0.5)
        assert _read_canonical(conn, term_id) == ("method", pytest.approx(0.5))

    def test_tier5_persists_score(self, conn, fake_embedder):
        v = [0.0] * 384
        v[7] = 1.0
        fake_embedder.preset("NovelTerm", v)

        r = resolve(
            conn,
            "NovelTerm",
            domain="rag",
            term_type="entity",
            entity_type="method",
            entity_type_score=0.72,
            source_paper="p1",
            embedder=fake_embedder,
        )
        assert r.matched_via == "tier5"
        assert r.entity_type == "method"
        assert r.entity_type_score == pytest.approx(0.72)
        stored = _read_canonical(conn, r.term_id)
        assert stored == ("method", pytest.approx(0.72))

    def test_tier3_flip_overturns_canonical(self, conn, fake_embedder):
        """Fuzzy re-encounter still flips: paper 1 seeds BookRAG/method@0.3;
        paper 2's 'BookRAGs' resolves via tier 3 as software@0.9, overturning
        the canonical. The historical paper's mention now reads back as
        software via the canonical_terms JOIN."""
        term_id = _seed_canonical(
            conn,
            canonical_name="BookRAG",
            entity_type="method",
            entity_type_score=0.3,
        )
        _seed_entity_row(
            conn,
            paper_name="p_old",
            domain="rag",
            entity_name="BookRAG",
            entity_type="method",
            term_id=term_id,
        )

        r = resolve(
            conn,
            "BookRAGs",
            domain="rag",
            term_type="entity",
            entity_type="software",
            entity_type_score=0.9,
            source_paper="p2",
            embedder=fake_embedder,
        )
        assert r.matched_via == "tier3"
        assert r.entity_type == "software"
        assert r.entity_type_score == pytest.approx(0.9)
        assert _read_canonical(conn, term_id) == ("software", pytest.approx(0.9))
        assert conn.execute(
            """
            SELECT ct.entity_type
              FROM term_aliases ta
              JOIN canonical_terms ct ON ct.id = ta.term_id
             WHERE ta.source_paper = ? AND ct.canonical_name = ?
             LIMIT 1
            """,
            ("p_old", "BookRAG"),
        ).fetchone()[0] == "software"

    def test_flip_does_not_touch_other_domain(
        self, conn, fake_embedder
    ):
        """Domain isolation is now a property of canonical identity:
        ``canonical_terms`` UNIQUE on (domain, term_type, canonical_name)
        means rag-domain DPR and vision-domain DPR are distinct term_ids,
        so a flip on one cannot affect the other."""
        conn.execute(
            "INSERT OR IGNORE INTO domains (name) VALUES (?)",
            ("vision",),
        )
        rag_term_id = _seed_canonical(
            conn,
            canonical_name="DPR",
            entity_type="method",
            entity_type_score=0.3,
        )
        vision_term_id = _seed_canonical(
            conn,
            canonical_name="DPR",
            domain="vision",
            entity_type="method",
            entity_type_score=0.3,
        )
        _seed_entity_row(
            conn,
            paper_name="p_old_rag",
            domain="rag",
            entity_name="DPR",
            entity_type="method",
            term_id=rag_term_id,
        )
        _seed_entity_row(
            conn,
            paper_name="p_old_vision",
            domain="vision",
            entity_name="DPR",
            entity_type="method",
            term_id=vision_term_id,
        )
        resolve(
            conn,
            "DPR",
            domain="rag",
            term_type="entity",
            entity_type="software",
            entity_type_score=0.8,
            source_paper="p2",
            embedder=fake_embedder,
        )
        # rag canonical flipped; vision canonical untouched.
        assert _read_canonical(conn, rag_term_id) == ("software", pytest.approx(0.8))
        assert _read_canonical(conn, vision_term_id) == ("method", pytest.approx(0.3))

    def test_non_entity_callers_never_flip(self, conn, fake_embedder):
        """Collections / topics resolves pass entity_type=None and score=0.0.
        The gate on non-empty new type must prevent any flip, even though
        stored entity_type is '' and the comparison is trivially 'different'.
        """
        term_id = _seed_canonical(
            conn,
            canonical_name="retrieval-augmented generation",
            term_type="collection",
            entity_type=None,
            entity_type_score=0.0,
        )
        r = resolve(
            conn,
            "retrieval-augmented generation",
            domain="rag",
            term_type="collection",
            entity_type=None,
            source_paper="p2",
            embedder=fake_embedder,
        )
        assert r.entity_type == ""
        assert r.entity_type_score == 0.0
        assert _read_canonical(conn, term_id) == ("", 0.0)
