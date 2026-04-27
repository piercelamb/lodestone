"""Shared 5-tier term resolver used by the classification and entity-extraction stages.

``resolve(conn, raw, ...)`` canonicalizes a raw term string against
``canonical_terms`` scoped by ``(domain, term_type)`` and returns a
:class:`ResolvedTerm` describing the hit. ``term_aliases`` is a
per-(concept, paper) **synonym index**: rows are written iff the
resolved surface form differs from the canonical's name after
normalization. Tier-1 hits and tier-5 mints are silent on the alias
side — the canonical itself is not a synonym of itself, and an
``--entity X`` lookup gets the canonical from ``canonical_terms``
without needing an alias row to confirm. Tier-2/3/4 hits add a row
when the surface form is a real synonym (different normalized shape).

Tiers, cheapest first, each falling through on miss:

1. SQL equality on ``canonical_name``.
2. Python filter on the scoped candidate set using
   :func:`_system.resolution.normalize.normalize_term`. Also checks
   ``term_aliases`` normalized.
3. ``rapidfuzz.fuzz.ratio >= 85`` over a SQL-prefiltered candidate pool
   (first-letter match OR ``|len_diff| <= 3``).
4. ``sqlite-vec`` top-K KNN over unit-norm ``bge-small-en-v1.5`` embeddings,
   then a gradient walk: drop top-1 if cosine < ``_TIER4_MIN_TOP1`` (absolute
   floor on the very-low tail only), include each subsequent candidate iff
   ``cos[i] >= _TIER4_GRADIENT * cos[prev_included]``, stop at the first
   failure. Merge only when **exactly one** candidate survives the walk.
   Two-or-more survivors (ambiguous tie) and an all-K survivor list (no sharp
   drop, uniform decline) both fall through to tier 5 — duplicate canonicals
   are recoverable; bad merges aren't. The gradient self-calibrates: what
   matters is the gap between top-1 and runner-up, not an absolute number.
5. New canonical: insert ``canonical_terms`` + ``term_embeddings`` atomically
   via ``SAVEPOINT``; on ``IntegrityError`` (UNIQUE race) fall back to tier 1
   and return the sibling row.

Note on ``entity_type`` / ``entity_type_score``: the column pair is metadata on
the canonical, NOT part of its identity or lookup scope. GLiNER2's label output
for the same string is noisy across mentions — scoping lookups by entity_type
fragments every popular entity into multiple rows. Tiers 1-4 ignore the
caller's ``entity_type`` for *matching*; tier 5 writes it onto the new
canonical row along with the caller's score.

On a tier-1-to-4 hit, the resolver also runs a flip check: if the caller's
``entity_type`` differs from what's stored AND the caller's
``entity_type_score`` is strictly higher than the stored one, the canonical's
``(entity_type, entity_type_score)`` is overwritten. The flip enqueues a
deferred ``terms_fts`` rebuild for the term so the FTS row stays in sync
with the new label. A flip is logged at INFO. Non-entity callers
(collections/topics) pass ``entity_type=None`` and ``entity_type_score=0.0``;
the flip path is gated on a non-empty new ``entity_type`` so their resolves
can never trigger one.

The resolved :class:`ResolvedTerm` carries the *currently-stored*
``entity_type`` and ``entity_type_score`` back. Callers may pass
``entity_type=None``; it is coerced to ``""``.
"""
from __future__ import annotations

import sqlite3
from enum import StrEnum
from typing import NamedTuple

import sqlite_vec
from rapidfuzz import fuzz

from _system.resolution.embeddings import Embedder
from _system.resolution.normalize import normalize_term
from _system.utils.logging import get_logger


class MatchTier(StrEnum):
    # ACRONYM is pre-resolver: aliases recovered by paper-native
    # Schwartz-Hearst detection before any tier runs. Stored as
    # ``term_aliases.match_tier = 0`` to distinguish from the 1-4
    # range of resolver-discovered aliases.
    ACRONYM = "acronym"
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    TIER4 = "tier4"
    TIER5 = "tier5"


_TIER3_MIN_RATIO = 85

# Tier 4: top-K + gradient walk. See module docstring for the algorithm.
# K is small (single SQL round-trip with k=10 is cheap on sqlite-vec) and the
# gradient does the discrimination work; the floor only gates the very-low tail.
_TIER4_TOP_K = 10
# Raised from 0.70 → 0.82 (2026-04-26): on the live vstash_2026 corpus with
# bge-small-en-v1.5, gradient-walk single-survivor merges in the 0.72–0.80
# range were collapsing distinct concepts (Mem0↔A-MEM, P95↔P99, full-text
# search↔hybrid retrieval). Clean tier-4 hits cluster ≥0.82. See
# .agents/.../handoffs/tier4_overmerge.md for the audit.
_TIER4_MIN_TOP1 = 0.82
_TIER4_GRADIENT = 0.88

_MIN_ALIAS_LEN = 3

_log = get_logger(__name__)


class ResolvedTerm(NamedTuple):
    term_id: int
    canonical_name: str
    entity_type: str
    entity_type_score: float
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
    entity_type_score: float = 0.0,
    source_paper: str,
    embedder: Embedder | None = None,
) -> ResolvedTerm:
    """5-tier canonicalization of ``raw`` in scope ``(domain, term_type)``.

    Lookups ignore ``entity_type`` entirely — it's metadata written onto the
    canonical at tier 5 and returned on the resolved row so callers can keep
    derived tables consistent. A non-default ``entity_type_score`` together
    with a non-empty ``entity_type`` also arms the flip check on tier-1-to-4
    hits: see :func:`_maybe_flip_entity_type`. Side effects: writes a
    ``term_aliases`` row on tier-2/3/4 hits whose normalized surface form
    differs from the canonical (synonym index — no row for tier-1 hits or
    tier-5 mints), may update ``canonical_terms`` on a flip, and inserts
    into ``canonical_terms`` + ``term_embeddings`` on tier 5. Alias inserts
    and flips both enqueue a deferred ``terms_fts`` rebuild. All writes
    happen on the caller-provided ``conn`` without calling ``commit()`` —
    the orchestrator owns transaction boundaries.
    """
    entity_type = entity_type or ""

    hit = _tier1(conn, raw, domain=domain, term_type=term_type)
    if hit is not None:
        return _apply_flip(
            conn,
            hit,
            domain=domain,
            new_entity_type=entity_type,
            new_entity_type_score=entity_type_score,
            tier=MatchTier.TIER1,
            raw=raw,
        )

    norm_query = normalize_term(raw)

    hit2 = _tier2(
        conn,
        norm_query=norm_query,
        domain=domain,
        term_type=term_type,
    )
    if hit2 is not None:
        return _hit(
            conn, hit2, raw=raw, source_paper=source_paper,
            tier=MatchTier.TIER2,
            domain=domain,
            new_entity_type=entity_type,
            new_entity_type_score=entity_type_score,
        )

    hit3 = _tier3(
        conn,
        raw,
        norm_query=norm_query,
        domain=domain,
        term_type=term_type,
    )
    if hit3 is not None:
        return _hit(
            conn, hit3, raw=raw, source_paper=source_paper,
            tier=MatchTier.TIER3,
            domain=domain,
            new_entity_type=entity_type,
            new_entity_type_score=entity_type_score,
        )

    if embedder is not None:
        hit4 = _tier4(
            conn,
            raw,
            domain=domain,
            term_type=term_type,
            embedder=embedder,
        )
        if hit4 is not None:
            return _hit(
                conn, hit4, raw=raw, source_paper=source_paper,
                tier=MatchTier.TIER4,
                domain=domain,
                new_entity_type=entity_type,
                new_entity_type_score=entity_type_score,
            )

    return _tier5(
        conn,
        raw,
        domain=domain,
        term_type=term_type,
        entity_type=entity_type,
        entity_type_score=entity_type_score,
        source_paper=source_paper,
        embedder=embedder,
    )


_Candidate = tuple[int, str, str, float]  # (term_id, canonical_name, entity_type, entity_type_score)


def _tier1(
    conn: sqlite3.Connection,
    raw: str,
    *,
    domain: str,
    term_type: str,
) -> ResolvedTerm | None:
    row = conn.execute(
        """
        SELECT id, canonical_name, entity_type, entity_type_score FROM canonical_terms
        WHERE domain = ? AND term_type = ?
          AND canonical_name = ?
        LIMIT 1
        """,
        (domain, term_type, raw),
    ).fetchone()
    if row is None:
        return None
    return ResolvedTerm(row[0], row[1], row[2], float(row[3]), MatchTier.TIER1, False)


def _tier2(
    conn: sqlite3.Connection,
    *,
    norm_query: str,
    domain: str,
    term_type: str,
) -> _Candidate | None:
    """Normalized match against canonical_name or any existing alias in scope."""
    if not norm_query:
        return None

    canonical_rows = conn.execute(
        """
        SELECT id, canonical_name, entity_type, entity_type_score FROM canonical_terms
        WHERE domain = ? AND term_type = ?
        """,
        (domain, term_type),
    ).fetchall()
    for term_id, canonical_name, entity_type, entity_type_score in canonical_rows:
        if normalize_term(canonical_name) == norm_query:
            return (term_id, canonical_name, entity_type, float(entity_type_score))

    alias_rows = conn.execute(
        """
        SELECT ct.id, ct.canonical_name, ct.entity_type, ct.entity_type_score, ta.alias
        FROM term_aliases ta
        JOIN canonical_terms ct ON ct.id = ta.term_id
        WHERE ct.domain = ? AND ct.term_type = ?
        """,
        (domain, term_type),
    ).fetchall()
    for term_id, canonical_name, entity_type, entity_type_score, alias in alias_rows:
        if normalize_term(alias) == norm_query:
            return (term_id, canonical_name, entity_type, float(entity_type_score))

    return None


def _tier3_candidates(
    conn: sqlite3.Connection,
    *,
    domain: str,
    term_type: str,
    raw: str,
) -> list[_Candidate]:
    """SQL-side prefilter for tier 3: share first letter OR ``|len_diff| <= 3``.

    Load-bearing: without this, tier 3 would scan every canonical in the scope
    in Python.
    """
    rows = conn.execute(
        """
        SELECT id, canonical_name, entity_type, entity_type_score FROM canonical_terms
        WHERE domain = ? AND term_type = ?
          AND (
                substr(lower(canonical_name), 1, 1) = substr(lower(?), 1, 1)
             OR abs(length(canonical_name) - length(?)) <= 3
          )
        """,
        (domain, term_type, raw, raw),
    ).fetchall()
    return [(tid, name, et, float(score)) for tid, name, et, score in rows]


def _tier3(
    conn: sqlite3.Connection,
    raw: str,
    *,
    norm_query: str,
    domain: str,
    term_type: str,
) -> _Candidate | None:
    if not norm_query:
        return None
    candidates = _tier3_candidates(
        conn,
        domain=domain,
        term_type=term_type,
        raw=raw,
    )
    # Stable tiebreak: max score, then shortest canonical_name, then lowest id.
    best_score = -1
    best: _Candidate | None = None
    best_key: tuple[int, int] | None = None  # (len(name), term_id)
    for term_id, canonical_name, entity_type, entity_type_score in candidates:
        score = fuzz.ratio(norm_query, normalize_term(canonical_name))
        if score < _TIER3_MIN_RATIO:
            continue
        key = (len(canonical_name), term_id)
        if score > best_score or (
            score == best_score and best_key is not None and key < best_key
        ):
            best_score = score
            best = (term_id, canonical_name, entity_type, entity_type_score)
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
    embedder: Embedder,
) -> _Candidate | None:
    """Top-K KNN + gradient walk. Merge iff exactly one candidate survives."""
    qvec = sqlite_vec.serialize_float32(embedder.embed(raw))
    rows = conn.execute(
        """
        SELECT te.term_id, te.distance, ct.canonical_name, ct.entity_type, ct.entity_type_score
        FROM term_embeddings te
        JOIN canonical_terms ct ON ct.id = te.term_id
        WHERE te.embedding MATCH ?
          AND te.term_type = ?
          AND te.domain = ?
          AND te.k = ?
        ORDER BY te.distance
        """,
        (qvec, term_type, domain, _TIER4_TOP_K),
    ).fetchall()
    if not rows:
        return None
    # cosine = 1 - d^2/2 for unit-norm vectors (sqlite-vec returns actual L2).
    candidates = [
        (tid, 1.0 - (dist * dist) / 2.0, name, et, float(score))
        for tid, dist, name, et, score in rows
    ]
    top1_score = candidates[0][1]
    if top1_score < _TIER4_MIN_TOP1:
        return None

    survivors = [candidates[0]]
    for cand in candidates[1:]:
        if cand[1] >= _TIER4_GRADIENT * survivors[-1][1]:
            survivors.append(cand)
        else:
            break

    if len(survivors) == 1 and len(candidates) > 1:
        # Sharp drop after top-1 → clear winner.
        tid, _cos, name, et, score = survivors[0]
        return (tid, name, et, score)

    if len(survivors) == 1 and len(candidates) == 1:
        # Only one candidate exists in the whole scope and it cleared the
        # floor. Treat as a clear winner — no runner-up means no ambiguity.
        tid, _cos, name, et, score = survivors[0]
        return (tid, name, et, score)

    # 2+ survivors (ambiguous cluster) OR walk consumed all K (uniform decline,
    # no sharp drop). Both mean "no single canonical stands out" — duplicates
    # are recoverable, bad merges aren't. Defer to tier 5.
    runner_up = candidates[1][1] if len(candidates) > 1 else float("nan")
    _log.debug(
        "tier4 fall-through: raw=%r survivors=%d/%d top1=%.3f top2=%.3f scope=(%s,%s)",
        raw, len(survivors), len(candidates), top1_score, runner_up,
        domain, term_type,
    )
    return None


def _tier5(
    conn: sqlite3.Connection,
    raw: str,
    *,
    domain: str,
    term_type: str,
    entity_type: str,
    entity_type_score: float,
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
                (domain, term_type, entity_type, entity_type_score,
                 canonical_name, first_seen_in)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (domain, term_type, entity_type, entity_type_score, raw, source_paper),
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
    except Exception as exc:
        conn.execute("ROLLBACK TO tier5")
        conn.execute("RELEASE tier5")
        if not isinstance(exc, sqlite3.IntegrityError):
            raise
        fallback = _tier1(
            conn, raw, domain=domain, term_type=term_type
        )
        if fallback is None:
            raise
        _log.info(
            "tier5 race resolved via tier1: raw=%r term_id=%s",
            raw, fallback.term_id,
        )
        # The sibling inserter's entity_type may differ from ours; run the
        # same flip check we'd run on any other tier-1 hit. Under the
        # synonym-index regime, no alias row is written for the tier-1
        # fallback either — the canonical itself is not a synonym.
        return _apply_flip(
            conn,
            fallback,
            domain=domain,
            new_entity_type=entity_type,
            new_entity_type_score=entity_type_score,
            tier=MatchTier.TIER1,
            raw=raw,
        )

    conn.execute("RELEASE tier5")
    _log.debug(
        "tier5 new canonical: raw=%r scope=(%s,%s,%s) term_id=%s score=%.3f",
        raw, domain, term_type, entity_type, term_id, entity_type_score,
    )
    return ResolvedTerm(
        term_id, raw, entity_type, entity_type_score, MatchTier.TIER5, True
    )


def _hit(
    conn: sqlite3.Connection,
    candidate: _Candidate,
    *,
    raw: str,
    source_paper: str,
    tier: MatchTier,
    domain: str,
    new_entity_type: str,
    new_entity_type_score: float,
) -> ResolvedTerm:
    """Common return path for tiers 2/3/4: synonym insert, flip check, return."""
    term_id, canonical_name, entity_type, entity_type_score = candidate
    _maybe_insert_alias(
        conn,
        term_id=term_id,
        canonical_name=canonical_name,
        alias=raw,
        source_paper=source_paper,
        tier=tier,
    )
    _log.debug("resolve %s hit: raw=%r term_id=%s", tier.value, raw, term_id)
    pre_flip = ResolvedTerm(
        term_id, canonical_name, entity_type, entity_type_score, tier, False
    )
    return _apply_flip(
        conn,
        pre_flip,
        domain=domain,
        new_entity_type=new_entity_type,
        new_entity_type_score=new_entity_type_score,
        tier=tier,
        raw=raw,
    )


def _apply_flip(
    conn: sqlite3.Connection,
    hit: ResolvedTerm,
    *,
    domain: str,
    new_entity_type: str,
    new_entity_type_score: float,
    tier: MatchTier,
    raw: str,
) -> ResolvedTerm:
    """Run the entity_type flip check and return the (possibly updated) row.

    See :func:`_maybe_flip_entity_type` for semantics. Pure wiring: when a
    flip lands, rebuild the :class:`ResolvedTerm` with the new
    ``(entity_type, entity_type_score)`` so callers see the post-flip state
    without a re-SELECT.
    """
    if _maybe_flip_entity_type(
        conn,
        term_id=hit.term_id,
        canonical_name=hit.canonical_name,
        domain=domain,
        stored_entity_type=hit.entity_type,
        stored_entity_type_score=hit.entity_type_score,
        new_entity_type=new_entity_type,
        new_entity_type_score=new_entity_type_score,
        tier=tier,
        raw=raw,
    ):
        return hit._replace(
            entity_type=new_entity_type,
            entity_type_score=new_entity_type_score,
        )
    return hit


def _maybe_flip_entity_type(
    conn: sqlite3.Connection,
    *,
    term_id: int,
    canonical_name: str,
    domain: str,
    stored_entity_type: str,
    stored_entity_type_score: float,
    new_entity_type: str,
    new_entity_type_score: float,
    tier: MatchTier,
    raw: str,
) -> bool:
    """Overturn ``canonical_terms.entity_type`` when the caller presents a
    strictly-higher-confidence different label. Returns True iff a flip ran.

    Gate: the new label must be non-empty (so collections/topics resolves
    with ``entity_type=None`` never flip), must differ from the stored one,
    and the new score must be strictly greater than the stored score.
    Enqueues a deferred ``terms_fts`` rebuild because the FTS row mirrors
    ``canonical_terms.entity_type`` and would otherwise stay stale until
    the next alias insert against this term. Logged at INFO.
    """
    if not new_entity_type:
        return False
    if new_entity_type == stored_entity_type:
        return False
    if new_entity_type_score <= stored_entity_type_score:
        return False

    conn.execute(
        """
        UPDATE canonical_terms
           SET entity_type = ?, entity_type_score = ?
         WHERE id = ?
        """,
        (new_entity_type, new_entity_type_score, term_id),
    )
    pending_fts_rebuilds(conn).add(term_id)
    _log.info(
        "entity_type flip: term_id=%s canonical=%r domain=%s %s(%.3f) -> %s(%.3f) "
        "via %s on raw=%r",
        term_id, canonical_name, domain,
        stored_entity_type or "''", stored_entity_type_score,
        new_entity_type, new_entity_type_score,
        tier.value, raw,
    )
    return True


def _maybe_insert_alias(
    conn: sqlite3.Connection,
    *,
    term_id: int,
    canonical_name: str,
    alias: str,
    source_paper: str,
    tier: MatchTier,
) -> None:
    """Insert a ``term_aliases`` synonym row and enqueue a deferred
    ``terms_fts`` rebuild. Idempotent per composite PK
    ``(term_id, alias, source_paper)``.

    ``term_aliases.match_tier`` stores ``0`` for :attr:`MatchTier.ACRONYM`
    (paper-native, pre-resolver) and the numeric part of the tier name
    (1-5) for resolver-discovered rows. **Synonym-index invariant**: rows
    where ``normalize_term(alias) == normalize_term(canonical_name)`` are
    silently dropped — the canonical is not a synonym of itself.
    Aliases shorter than :data:`_MIN_ALIAS_LEN` are also dropped to avoid
    noise from one- or two-letter spans.
    """
    if len(alias) < _MIN_ALIAS_LEN:
        return
    if normalize_term(alias) == normalize_term(canonical_name):
        return
    if tier is MatchTier.ACRONYM:
        match_tier = 0
    else:
        match_tier = int(tier.value.removeprefix("tier"))
    conn.execute(
        """
        INSERT OR IGNORE INTO term_aliases
            (term_id, alias, source_paper, match_tier)
        VALUES (?, ?, ?, ?)
        """,
        (term_id, alias, source_paper, match_tier),
    )
    pending_fts_rebuilds(conn).add(term_id)


def insert_acronym_alias(
    conn: sqlite3.Connection,
    *,
    term_id: int,
    canonical_name: str,
    alias: str,
    source_paper: str,
) -> None:
    """Public: persist a Schwartz-Hearst short form as a ``term_aliases`` row.

    Wrapper around :func:`_maybe_insert_alias` for pre-resolver acronym
    aliases. Short forms under :data:`_MIN_ALIAS_LEN` are silently skipped,
    as are forms that normalize to the canonical (the synonym-index
    invariant).
    """
    _maybe_insert_alias(
        conn,
        term_id=term_id,
        canonical_name=canonical_name,
        alias=alias,
        source_paper=source_paper,
        tier=MatchTier.ACRONYM,
    )
