"""Shared 5-tier term resolver used by the classification and entity-extraction stages.

``resolve(conn, raw, ...)`` canonicalizes a raw term string against
``canonical_terms`` scoped by ``(domain, term_type, entity_type)`` and returns a
:class:`ResolvedTerm` describing the hit. Callers receive the canonical name
back so no follow-up ``SELECT`` is required.

Tiers, cheapest first, each falling through on miss:

1. SQL equality on ``canonical_name`` — no alias insert.
2. Python filter on the scoped candidate set using
   :func:`_system.resolution.normalize.normalize_term`. Also checks
   ``term_aliases`` normalized.
3. ``rapidfuzz.fuzz.ratio >= 85`` over a SQL-prefiltered candidate pool
   (first-letter match OR ``|len_diff| <= 3``).
4. ``sqlite-vec`` KNN with cosine similarity ``>= 0.85`` computed from the
   extension's L2 distance on unit-norm ``bge-small-en-v1.5`` embeddings.
5. New canonical: insert ``canonical_terms`` + ``term_embeddings`` atomically
   via ``SAVEPOINT``; on ``IntegrityError`` (UNIQUE race) fall back to tier 1
   and return the sibling row.

Note on ``entity_type``: the column is ``NOT NULL DEFAULT ''`` in the schema
(sqlite-vec vec0 metadata filters require simple equality, and SQLite UNIQUE
treats NULL as distinct — both constraints push toward using the empty string
for the non-entity scopes). Callers may pass ``None`` to this module; it is
coerced to ``""`` at the public entry point.
"""
from __future__ import annotations

import math
import sqlite3
from enum import StrEnum
from typing import NamedTuple

import sqlite_vec
from rapidfuzz import fuzz

from _system.resolution.embeddings import Embedder
from _system.resolution.normalize import normalize_term
from _system.utils.logging import get_logger


class MatchTier(StrEnum):
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    TIER4 = "tier4"
    TIER5 = "tier5"


_TIER3_MIN_RATIO = 85
_TIER4_MIN_COSINE = 0.85
# sqlite-vec returns actual L2 distance on unit-norm vectors: d = sqrt(2 - 2*cos).
_TIER4_MAX_DISTANCE = math.sqrt(2.0 - 2.0 * _TIER4_MIN_COSINE)

_MIN_ALIAS_LEN = 3

_log = get_logger(__name__)


class ResolvedTerm(NamedTuple):
    term_id: int
    canonical_name: str
    matched_via: MatchTier
    created_new: bool


# Per-connection set of term_ids whose terms_fts row needs a rebuild after
# the current stage. Keyed by id(conn) because Python 3.14's sqlite3.Connection
# rejects weak references and its heap type is immutable. The orchestrator
# drains and clears this set at stage boundaries, so id() reuse from a GC'd
# connection would only leak into the next stage if the caller holds a stale
# id — not a real failure mode for the single-connection ingestion path.
_pending_fts_rebuilds: dict[int, set[int]] = {}


def pending_fts_rebuilds(conn: sqlite3.Connection) -> set[int]:
    """Return (and lazily create) the per-connection pending-rebuild set."""
    return _pending_fts_rebuilds.setdefault(id(conn), set())


def resolve(
    conn: sqlite3.Connection,
    raw: str,
    *,
    domain: str,
    term_type: str,
    entity_type: str | None = None,
    source_paper: str,
    embedder: Embedder | None = None,
) -> ResolvedTerm:
    """5-tier canonicalization of ``raw`` in scope ``(domain, term_type, entity_type)``.

    Side effects: may insert into ``term_aliases`` (tiers 2/3/4) and into
    ``canonical_terms`` + ``term_embeddings`` (tier 5). Alias inserts enqueue
    a deferred ``terms_fts`` rebuild. All writes happen on the caller-provided
    ``conn`` without calling ``commit()`` — the orchestrator owns transaction
    boundaries.
    """
    entity_type = entity_type or ""

    hit = _tier1(conn, raw, domain=domain, term_type=term_type, entity_type=entity_type)
    if hit is not None:
        _log.debug("resolve tier1 hit: raw=%r term_id=%s", raw, hit.term_id)
        return hit

    norm_query = normalize_term(raw)

    hit2 = _tier2(
        conn,
        norm_query=norm_query,
        domain=domain,
        term_type=term_type,
        entity_type=entity_type,
    )
    if hit2 is not None:
        term_id, canonical_name = hit2
        _maybe_insert_alias(
            conn,
            term_id=term_id,
            canonical_name=canonical_name,
            alias=raw,
            source_paper=source_paper,
            match_tier=2,
        )
        _log.debug("resolve tier2 hit: raw=%r term_id=%s", raw, term_id)
        return ResolvedTerm(term_id, canonical_name, MatchTier.TIER2, False)

    hit3 = _tier3(
        conn,
        raw,
        norm_query=norm_query,
        domain=domain,
        term_type=term_type,
        entity_type=entity_type,
    )
    if hit3 is not None:
        term_id, canonical_name = hit3
        _maybe_insert_alias(
            conn,
            term_id=term_id,
            canonical_name=canonical_name,
            alias=raw,
            source_paper=source_paper,
            match_tier=3,
        )
        _log.debug("resolve tier3 hit: raw=%r term_id=%s", raw, term_id)
        return ResolvedTerm(term_id, canonical_name, MatchTier.TIER3, False)

    if embedder is not None:
        hit4 = _tier4(
            conn,
            raw,
            domain=domain,
            term_type=term_type,
            entity_type=entity_type,
            embedder=embedder,
        )
        if hit4 is not None:
            term_id, canonical_name = hit4
            _maybe_insert_alias(
                conn,
                term_id=term_id,
                canonical_name=canonical_name,
                alias=raw,
                source_paper=source_paper,
                match_tier=4,
            )
            _log.debug("resolve tier4 hit: raw=%r term_id=%s", raw, term_id)
            return ResolvedTerm(term_id, canonical_name, MatchTier.TIER4, False)

    return _tier5(
        conn,
        raw,
        domain=domain,
        term_type=term_type,
        entity_type=entity_type,
        source_paper=source_paper,
        embedder=embedder,
    )


def _tier1(
    conn: sqlite3.Connection,
    raw: str,
    *,
    domain: str,
    term_type: str,
    entity_type: str,
) -> ResolvedTerm | None:
    row = conn.execute(
        """
        SELECT id, canonical_name FROM canonical_terms
        WHERE domain = ? AND term_type = ?
          AND entity_type = ?
          AND canonical_name = ?
        LIMIT 1
        """,
        (domain, term_type, entity_type, raw),
    ).fetchone()
    if row is None:
        return None
    return ResolvedTerm(row[0], row[1], MatchTier.TIER1, False)


def _tier2(
    conn: sqlite3.Connection,
    *,
    norm_query: str,
    domain: str,
    term_type: str,
    entity_type: str,
) -> tuple[int, str] | None:
    """Normalized match against canonical_name or any existing alias in scope."""
    if not norm_query:
        return None

    canonical_rows = conn.execute(
        """
        SELECT id, canonical_name FROM canonical_terms
        WHERE domain = ? AND term_type = ? AND entity_type = ?
        """,
        (domain, term_type, entity_type),
    ).fetchall()
    for term_id, canonical_name in canonical_rows:
        if normalize_term(canonical_name) == norm_query:
            return (term_id, canonical_name)

    alias_rows = conn.execute(
        """
        SELECT ct.id, ct.canonical_name, ta.alias
        FROM term_aliases ta
        JOIN canonical_terms ct ON ct.id = ta.term_id
        WHERE ct.domain = ? AND ct.term_type = ? AND ct.entity_type = ?
        """,
        (domain, term_type, entity_type),
    ).fetchall()
    for term_id, canonical_name, alias in alias_rows:
        if normalize_term(alias) == norm_query:
            return (term_id, canonical_name)

    return None


def _tier3_candidates(
    conn: sqlite3.Connection,
    *,
    domain: str,
    term_type: str,
    entity_type: str,
    raw: str,
) -> list[tuple[int, str]]:
    """SQL-side prefilter for tier 3: share first letter OR ``|len_diff| <= 3``.

    Load-bearing: without this, tier 3 would scan every canonical in the scope
    in Python.
    """
    return conn.execute(
        """
        SELECT id, canonical_name FROM canonical_terms
        WHERE domain = ? AND term_type = ? AND entity_type = ?
          AND (
                substr(lower(canonical_name), 1, 1) = substr(lower(?), 1, 1)
             OR abs(length(canonical_name) - length(?)) <= 3
          )
        """,
        (domain, term_type, entity_type, raw, raw),
    ).fetchall()


def _tier3(
    conn: sqlite3.Connection,
    raw: str,
    *,
    norm_query: str,
    domain: str,
    term_type: str,
    entity_type: str,
) -> tuple[int, str] | None:
    if not norm_query:
        return None
    candidates = _tier3_candidates(
        conn,
        domain=domain,
        term_type=term_type,
        entity_type=entity_type,
        raw=raw,
    )
    # Stable tiebreak: max score, then shortest canonical_name, then lowest id.
    best_score = -1
    best: tuple[int, str] | None = None
    best_key: tuple[int, int] | None = None  # (len(name), term_id)
    for term_id, canonical_name in candidates:
        score = fuzz.ratio(norm_query, normalize_term(canonical_name))
        if score < _TIER3_MIN_RATIO:
            continue
        key = (len(canonical_name), term_id)
        if score > best_score or (score == best_score and key < best_key):
            best_score = score
            best = (term_id, canonical_name)
            best_key = key
            if score == 100:
                break
    return best


def _tier4(
    conn: sqlite3.Connection,
    raw: str,
    *,
    domain: str,
    term_type: str,
    entity_type: str,
    embedder: Embedder,
) -> tuple[int, str] | None:
    qvec = sqlite_vec.serialize_float32(embedder.embed(raw))
    rows = conn.execute(
        """
        SELECT term_id, distance FROM term_embeddings
        WHERE embedding MATCH ?
          AND term_type = ?
          AND domain = ?
          AND entity_type = ?
          AND k = 5
        ORDER BY distance
        """,
        (qvec, term_type, domain, entity_type),
    ).fetchall()
    if not rows:
        return None
    top_term_id, top_distance = rows[0]
    if top_distance > _TIER4_MAX_DISTANCE:
        return None
    name_row = conn.execute(
        "SELECT canonical_name FROM canonical_terms WHERE id = ?",
        (top_term_id,),
    ).fetchone()
    if name_row is None:
        return None
    return (top_term_id, name_row[0])


def _tier5(
    conn: sqlite3.Connection,
    raw: str,
    *,
    domain: str,
    term_type: str,
    entity_type: str,
    source_paper: str,
    embedder: Embedder | None,
) -> ResolvedTerm:
    if embedder is None:
        raise ValueError(
            "tier 5 requires an Embedder to populate term_embeddings; "
            f"resolve({raw!r}, ...) was called with embedder=None"
        )

    # Scope both inserts under one savepoint so a vec0 failure after the
    # canonical row is written cannot leave an orphaned canonical. The outer
    # orchestrator owns the surrounding transaction; SAVEPOINT composes with
    # whatever outer transaction state is active.
    conn.execute("SAVEPOINT tier5")
    try:
        cur = conn.execute(
            """
            INSERT INTO canonical_terms
                (domain, term_type, entity_type, canonical_name, first_seen_in)
            VALUES (?, ?, ?, ?, ?)
            """,
            (domain, term_type, entity_type, raw, source_paper),
        )
        term_id = cur.lastrowid
        if term_id is None:
            raise RuntimeError(
                f"tier 5 insert did not return a lastrowid for raw={raw!r} "
                f"scope=({domain},{term_type},{entity_type!r})"
            )
        vec = embedder.embed(raw)
        conn.execute(
            """
            INSERT INTO term_embeddings
                (term_id, embedding, term_type, entity_type, domain)
            VALUES (?, ?, ?, ?, ?)
            """,
            (term_id, sqlite_vec.serialize_float32(vec), term_type, entity_type, domain),
        )
    except sqlite3.IntegrityError:
        conn.execute("ROLLBACK TO tier5")
        conn.execute("RELEASE tier5")
        fallback = _tier1(
            conn, raw, domain=domain, term_type=term_type, entity_type=entity_type
        )
        if fallback is None:
            raise
        _log.info(
            "tier5 race resolved via tier1: raw=%r term_id=%s",
            raw, fallback.term_id,
        )
        return fallback
    except Exception:
        conn.execute("ROLLBACK TO tier5")
        conn.execute("RELEASE tier5")
        raise

    conn.execute("RELEASE tier5")
    _log.info(
        "tier5 new canonical: raw=%r scope=(%s,%s,%s) term_id=%s",
        raw, domain, term_type, entity_type, term_id,
    )
    return ResolvedTerm(term_id, raw, MatchTier.TIER5, True)


def _maybe_insert_alias(
    conn: sqlite3.Connection,
    *,
    term_id: int,
    canonical_name: str,
    alias: str,
    source_paper: str,
    match_tier: int,
) -> None:
    """Insert a ``term_aliases`` row if it passes the acceptance filter, and
    enqueue a deferred ``terms_fts`` rebuild. Idempotent per composite PK.
    """
    if len(alias) < _MIN_ALIAS_LEN:
        return
    if normalize_term(alias) == normalize_term(canonical_name):
        return
    conn.execute(
        """
        INSERT OR IGNORE INTO term_aliases
            (term_id, alias, source_paper, match_tier)
        VALUES (?, ?, ?, ?)
        """,
        (term_id, alias, source_paper, match_tier),
    )
    pending_fts_rebuilds(conn).add(term_id)
