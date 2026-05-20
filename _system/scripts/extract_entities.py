"""GLiNER2 entity extraction for a single paper.

Fourth pipeline stage (after ``fetch`` → ``convert`` → ``classify``): loads the
paper's markdown, splits into sections via the shared section splitter, sub-
chunks each section under GLiNER2's 384-token ceiling, runs labels-with-
descriptions inference, applies the garbage gate, and resolves every surviving
entity name through the shared 5-tier resolver. The resolver writes a
``term_aliases`` synonym row only for tier-2/3/4 hits whose surface form
differs from the canonical's name; tier-1 hits and tier-5 mints are silent
on the alias side. Sets ``papers.status = 'extracted'`` and updates
``papers.entity_count`` to the number of distinct canonicals this paper
resolved (counted via a Python set, not the alias table). Re-running the
stage on a paper wipes its entity-typed alias rows first; topic/collection
aliases written by the classify stage are preserved.

The GLiNER2 model is loaded once per process via a module-level singleton.
Callers never load GLiNER2 at import time — the model is pulled in lazily from
``_get_model()`` on the first inference call.

Test seam: pass ``run_inference`` to :func:`extract` to substitute canned spans
for the real model. All non-``@pytest.mark.slow`` tests use this seam so no
torch/GLiNER2 weights are ever loaded.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, NamedTuple, TypedDict

from _system.db.connection import get_conn, transaction
from _system.resolution.acronyms import extract_acronym_pairs
from _system.resolution.embeddings import Embedder
from _system.resolution.normalize import normalize_term
from _system.resolution.resolver import (
    insert_acronym_alias,
    pending_fts_rebuilds,
    resolve,
)
from _system.schemas.entities import EntityType
from _system.schemas.paper_metadata import PaperStatus, can_run_from as paper_can_run_from
from _system.schemas.post_metadata import PostStatus, can_run_from as post_can_run_from
from _system.utils.config import load_gliner_config
from _system.utils.logging import get_logger
from _system.utils.sections import split_sections, strip_breadcrumb, sub_chunk
from _system.utils.source_resolution import (
    SlugNotFound,
    SourceKind,
    resolve_slug,
)

_LOG = get_logger("scripts.extract_entities")

_ACRONYM_MIN_LEN = 3
_REJECT_RATE_WARN = 0.5
_REJECT_RATE_WARN_MIN_SAMPLES = 10
_ENTITY_MAX_LEN = 100

_ACRONYM_ALLOWLIST: frozenset[str] = frozenset({"LM", "QA", "NN", "AI", "ML", "NLP"})
_ENTITY_STOPLIST: frozenset[str] = frozenset(
    {
        "table", "figure", "we", "using", "our", "this", "these", "that", "it", "however",
        # Generic concept nouns bleeding in as standalone entities
        "vector", "hybrid", "scoring", "scale", "mean", "median", "distance",
        # File formats the extractor keeps labelling as dataset/system
        "json", "pdf", "html", "docx", "xml", "yaml", "csv",
    }
)
# Derived from EntityType so the two can't drift.
_LABEL_WORDS: frozenset[str] = frozenset(t.value for t in EntityType)

_NUMERIC_RE = re.compile(r"\d+")
# Document-structure references: "Table 2", "Figure 3a", "Appendix C",
# "Section 4.1", "Eq. 7". These are pointers, not entities.
_STRUCTURAL_REF_RE = re.compile(
    r"^(Table|Figure|Fig\.?|Appendix|Section|Chapter|Eq\.?|Equation)\s+\S+$"
)
# Value-delta or raw-quantity spans: "+5.6%", "-10ms", "10K", "10.9 ms",
# "209 docs", "700 chunks per second". Any leading sign or digit followed
# only by more digits, punctuation, and lowercase unit words.
_QUANTITY_RE = re.compile(r"^[+\-]?\d[\d.,]*(\s*[a-zA-Z%]+)*$")
# Dangling punctuation at the end = GLiNER2 span-boundary truncation
# (e.g. "NV-", "foo/", "bar:").
_TRAILING_JUNK = "-/:,@#"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ExtractError(Exception):
    """Base class for extract_entities failures."""


class PaperNotFound(ExtractError):
    """No papers row for the given paper_name."""


class MarkdownMissing(ExtractError):
    """papers.markdown is NULL — convert stage did not run."""


class StatusTooLow(ExtractError):
    """can_run_from rejected current status for EXTRACTED (and --force not set)."""


class UnknownStatusError(ExtractError):
    """papers.status holds a string not recognized by PaperStatus."""


class ExtractResult(NamedTuple):
    paper_name: str
    entity_count: int
    status: str


class Span(TypedDict):
    text: str
    label: str
    score: float
    start: int
    end: int


InferenceFn = Callable[[str, dict[str, str], float], list[Span]]
TokenizeFn = Callable[[str], list[tuple[int, int]]]


# ---------------------------------------------------------------------------
# GLiNER2 singleton + default inference
# ---------------------------------------------------------------------------

_MODEL: Any = None


def _get_model() -> Any:
    """Lazy-load the GLiNER2 model once per process.

    The concrete factory call lives here so tests can monkeypatch this single
    entry point. The GLiNER2 import is deferred so ``import extract_entities``
    stays cheap and does not drag in torch.
    """
    global _MODEL
    if _MODEL is None:
        # Populate the HF cache first; emits byte-level progress through
        # the active _progress_hook when the MCP server has one set.
        # GLiNER2.from_pretrained then finds the snapshot present.
        from _system.scripts.validate_models import ModelId, ensure_model_cached

        ensure_model_cached(ModelId.GLINER2)
        from gliner2 import GLiNER2
        _MODEL = GLiNER2.from_pretrained("fastino/gliner2-large-v1")
    return _MODEL


def _default_inference(
    text: str, label_descriptions: dict[str, str], threshold: float
) -> list[Span]:
    """Production inference seam; mocked in all non-slow tests."""
    model = _get_model()
    raw = model.extract_entities(
        text,
        entity_types=label_descriptions,
        threshold=threshold,
        include_confidence=True,
        include_spans=True,
    )
    return _flatten_gliner_output(raw)


def _default_tokenize(text: str) -> list[tuple[int, int]]:
    """Production tokenizer for :func:`sub_chunk` — returns ``(start, end)``
    character offsets for each subword token GLiNER2 will produce.

    Uses the HF fast tokenizer's ``return_offsets_mapping=True`` so sub-chunks
    respect the model's 384-token ceiling (a word-count approximation would
    exceed the ceiling on dense prose) *and* are reconstructed by slicing the
    source string rather than rejoining subword pieces. The latter matters:
    SentencePiece / WordPiece split ``FiQA`` → ``[▁Fi, QA]`` and ``BGE-small``
    → ``[▁B, GE, -, small]``, and ``" ".join(...)`` would feed GLiNER2 a
    mangled input whose span slices become ``"Fi QA"`` / ``"B GE - small"`` —
    corrupting every entity name, the resolver's canonical_terms, and the
    FTS index. Offsets map back to the original text, keeping entity names
    verbatim.
    """
    enc = _get_model().processor.tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    return [tuple(pair) for pair in enc["offset_mapping"]]


def _flatten_gliner_output(raw: Any) -> list[Span]:
    """Normalize GLiNER2 formatted output into flat span dicts.

    ``format_results=True`` returns ``{"entities": {<label>: [<span>, ...]}}``
    where each span carries ``text``, ``start``, ``end`` and — with
    ``include_confidence=True`` — ``confidence``. We normalize to
    ``{text, label, score, start, end}`` so downstream code never branches on
    GLiNER2's exact key names. An empty ``{}`` — GLiNER2's legitimate
    no-entities output — is handled (returns ``[]``); any other malformed
    shape raises ``ValueError`` so API drift surfaces immediately rather than
    silently dropping entities.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"GLiNER2 output must be a dict; got {type(raw).__name__}: {raw!r}"
        )
    entities = raw.get("entities", {})
    if not isinstance(entities, dict):
        raise ValueError(
            "GLiNER2 output.entities must be a dict keyed by label; got "
            f"{type(entities).__name__}: {entities!r}"
        )
    spans: list[Span] = []
    for label, items in entities.items():
        if not isinstance(items, list):
            raise ValueError(
                f"GLiNER2 output.entities[{label!r}] must be a list; got "
                f"{type(items).__name__}: {items!r}"
            )
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(
                    f"GLiNER2 output.entities[{label!r}] item must be a dict; "
                    f"got {type(item).__name__}: {item!r}"
                )
            spans.append(
                Span(
                    text=item["text"],
                    label=label,
                    score=float(item.get("confidence", item.get("score", 0.0))),
                    start=int(item["start"]),
                    end=int(item["end"]),
                )
            )
    return spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract(
    *,
    paper_name: str,
    conn: sqlite3.Connection,
    force: bool = False,
    run_inference: InferenceFn | None = None,
    tokenize: TokenizeFn | None = None,
    embedder: Embedder | None = None,
    config_path: Path | str | None = None,
) -> ExtractResult:
    """Extract entities for one paper and advance ``status`` to EXTRACTED.

    Replace-all semantics: deletes any prior ``entities`` rows for the paper
    before inserting. Commits in a single transaction.

    ``run_inference`` is the test seam for GLiNER2. ``tokenize`` returns
    ``(start, end)`` char offsets per token and defaults to :func:`sub_chunk`'s
    whitespace offsets; production leaves it None so GLiNER2's fast tokenizer
    is used with ``return_offsets_mapping=True`` (see :func:`_default_tokenize`
    for why offsets — not token strings — are required). ``embedder`` is forwarded
    to the resolver for tier 5 inserts. Fresh entity names miss tiers 1-4 and
    need an Embedder; absent one we lazily construct the real BGE embedder.
    """
    try:
        resolved = resolve_slug(conn, paper_name)
    except SlugNotFound as exc:
        raise PaperNotFound(
            f"slug={paper_name!r} not found in papers or posts"
        ) from exc
    kind = resolved.kind
    source_id = resolved.id
    target_table = "papers" if kind is SourceKind.PAPER else "posts"
    row = conn.execute(
        f"SELECT domain, status, markdown FROM {target_table} WHERE id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise PaperNotFound(
            f"slug={paper_name!r} resolved to id={source_id} but row vanished"
        )
    domain, status_str, markdown = row

    if markdown is None:
        raise MarkdownMissing(f"slug={paper_name!r}: markdown is NULL")

    if kind is SourceKind.PAPER:
        try:
            current = PaperStatus(status_str) if status_str else None
        except ValueError as exc:
            raise UnknownStatusError(
                f"paper_name={paper_name!r}: unrecognized status={status_str!r}"
            ) from exc
        if not force and not paper_can_run_from(current, PaperStatus.EXTRACTED):
            extra = (
                " (FAILED_HTML is terminal — re-fetch required)"
                if current is PaperStatus.FAILED_HTML
                else ""
            )
            raise StatusTooLow(
                f"paper_name={paper_name!r}: cannot run EXTRACTED from status="
                f"{status_str!r}{extra}"
            )
        target_status_value = PaperStatus.EXTRACTED.value
    else:
        try:
            current_post = PostStatus(status_str) if status_str else None
        except ValueError as exc:
            raise UnknownStatusError(
                f"post_name={paper_name!r}: unrecognized status={status_str!r}"
            ) from exc
        if not force and not post_can_run_from(current_post, PostStatus.EXTRACTED):
            extra = (
                " (terminal failure — re-fetch required)"
                if current_post in (PostStatus.FAILED_FETCH, PostStatus.FAILED_PARSE)
                else ""
            )
            raise StatusTooLow(
                f"post_name={paper_name!r}: cannot run EXTRACTED from status="
                f"{status_str!r}{extra}"
            )
        target_status_value = PostStatus.EXTRACTED.value

    cfg = (
        load_gliner_config(config_path)
        if config_path is not None
        else load_gliner_config()
    )

    # GlinerConfig._label_keys_agree guarantees every key round-trips.
    label_to_type = {
        label: EntityType(label.lower()) for label in cfg.label_descriptions
    }
    label_descriptions = dict(cfg.label_descriptions)

    inference = run_inference or _default_inference
    # Only pair the real GLiNER2 tokenizer with the real inference path so
    # tests running canned inference don't trigger a model load via sub_chunk.
    if tokenize is not None:
        tokenize_fn = tokenize
    elif run_inference is None:
        tokenize_fn = _default_tokenize
    else:
        tokenize_fn = None

    # Defer Embedder construction until the first tier-5 insert; all-tier-1
    # papers never need to load sentence-transformers.
    def _get_embedder() -> Embedder:
        nonlocal embedder
        if embedder is None:
            embedder = Embedder()
        return embedder

    # Inference-time floor: use the smallest per-label threshold so GLiNER2
    # returns candidates for every label; per-label filtering below applies
    # the exact threshold per span.
    min_threshold = min([*cfg.per_label.values(), cfg.global_threshold])

    chunks = split_sections(markdown)

    n_rejected = 0
    n_total = 0
    paper_domain = domain or ""

    # ------------------------------------------------------------------
    # Schwartz-Hearst pre-pass: find paper-native ``Long Form (SHORT)``
    # definitions and build a rewrite map so acronym spans funnel into
    # the long-form canonical. Every mention of ``RRF`` in a paper that
    # defines ``Reciprocal Rank Fusion (RRF)`` votes and resolves under
    # the long form. Short forms observed as spans are remembered so we
    # can persist them to ``term_aliases`` after the long-form canonical
    # has a term_id.
    # ------------------------------------------------------------------
    acronym_rewrite: dict[str, str] = {
        normalize_term(short): long
        for short, long in extract_acronym_pairs(markdown)
    }
    # normalized long-form name -> set of raw short forms observed as spans
    observed_short_forms: dict[str, set[str]] = defaultdict(set)

    def _canonicalize(name: str) -> str:
        """Rewrite ``name`` to its paper-native long form if it's a known
        Schwartz-Hearst acronym; otherwise return unchanged.
        """
        return acronym_rewrite.get(normalize_term(name), name)

    # ------------------------------------------------------------------
    # Pass 1 (outside transaction): inference + paper-wide label vote.
    #
    # Two-stage vote:
    #
    # Stage A — argmax-per-span (bug fix). GLiNER2 emits one row per
    # label that clears the threshold, so a single physical span can
    # appear 3x with different labels/scores ("BGE-small embeddings"
    # at one position: model=0.67, dataset=0.41, method=0.31). Without
    # this dedup, one mention would cast three equal votes, which stacks
    # up across sections and dilutes the entity's clear per-mention
    # winner. Grouping by ``(start, end)`` within each sub_chunk and
    # keeping the top-scoring label collapses multi-label rows back to
    # one vote per span. Strict ``>`` on score preserves first-seen
    # tiebreak.
    #
    # Stage B — count-based majority vote across the paper. The argmax
    # winner per span casts a ``+= 1`` vote for its label; the etype
    # with the most votes wins. ``Counter.most_common`` preserves
    # insertion order on ties, so the first-observed winner wins a tie.
    # Majority voting is robust to single high-confidence outliers
    # (one 0.90 mislabel can't outvote 19 0.65 correct mentions).
    #
    # Stage C — ``entity_type_score`` = max score observed for the
    # *winning* label's mentions only. This is what the resolver's
    # cross-paper flip check compares against: a later paper that wins
    # majority on a different label with a strictly-higher peak for
    # THAT label overturns the stored one. Peaks on losing labels are
    # discarded.
    # ------------------------------------------------------------------
    type_votes: dict[str, Counter[EntityType]] = defaultdict(Counter)
    # norm -> {etype -> max score observed for that etype}. Populated
    # for every label seen; only the winning label's score is kept
    # after the paper-wide tally.
    type_scores: dict[str, dict[EntityType, float]] = defaultdict(dict)
    per_chunk: list[tuple[Any, list[str], list[list[Span]]]] = []

    for chunk in chunks:
        stripped = strip_breadcrumb(chunk.body)
        sub_chunks_list = sub_chunk(
            stripped,
            max_tokens=cfg.chunk.max_tokens,
            overlap_tokens=cfg.chunk.overlap_tokens,
            tokenizer_cb=tokenize_fn,
        )
        chunk_valid_spans: list[list[Span]] = []
        for sub in sub_chunks_list:
            raw_spans = inference(sub, label_descriptions, min_threshold)

            # Stage A — argmax-per-span dedup.
            best_by_pos: dict[tuple[int, int], Span] = {}
            for span in raw_spans:
                start = int(span.get("start", 0))
                end = int(span.get("end", 0))
                score = float(span.get("score", 0.0))
                prev = best_by_pos.get((start, end))
                if prev is None or score > float(prev.get("score", 0.0)):
                    best_by_pos[(start, end)] = span
            # Iterate in first-seen order across positions so pass-2
            # dedup and log output stay deterministic.
            deduped_spans: list[Span] = list(best_by_pos.values())

            valid: list[Span] = []
            for span in deduped_spans:
                label = span.get("label", "")
                name = (span.get("text") or "").strip()
                if not name:
                    continue
                score = float(span.get("score", 0.0))
                threshold = cfg.per_label.get(label, cfg.global_threshold)
                if score < threshold:
                    continue
                etype = label_to_type.get(label)
                if etype is None:
                    _LOG.warning(
                        "unknown GLiNER label %r not in EntityType; skipping "
                        "span text=%r in paper_name=%s",
                        label, name, paper_name,
                    )
                    continue
                # Rewrite acronym spans to their long form so votes and
                # dedup collapse both acronym and expanded mentions onto
                # a single canonical.
                canonical_text = _canonicalize(name)
                if canonical_text != name:
                    observed_short_forms[
                        normalize_term(canonical_text)
                    ].add(name)
                norm = normalize_term(canonical_text)
                type_votes[norm][etype] += 1
                prev_score = type_scores[norm].get(etype, 0.0)
                if score > prev_score:
                    type_scores[norm][etype] = score
                valid.append(span)
            chunk_valid_spans.append(valid)
        per_chunk.append((chunk, sub_chunks_list, chunk_valid_spans))

    # Stage B + C — majority winner per name, max-score for winning label.
    winning_type: dict[str, tuple[EntityType, float]] = {}
    for norm, counter in type_votes.items():
        winner = counter.most_common(1)[0][0]
        winner_score = type_scores[norm].get(winner, 0.0)
        winning_type[norm] = (winner, winner_score)

    # ------------------------------------------------------------------
    # Pass 2 (transactional): dedup per section, resolve (which writes a
    # synonym row only for tier-2/3/4 hits with a non-canonical surface
    # form), track every distinct canonical this paper resolves so we
    # can update papers.entity_count and seed the deferred FTS rebuild
    # queue (tier-1/tier-5 paths leave no alias trail under the synonym-
    # index regime, so the rebuild queue is the only signal index_paper
    # has for those terms).
    # ------------------------------------------------------------------
    with transaction(conn):
        # Wipe entity-typed alias rows for this paper so re-extraction
        # doesn't accumulate stale synonym records. Topic/collection
        # aliases (term_type != 'entity') are left untouched — classify
        # owns those and won't re-run as part of this stage.
        conn.execute(
            """
            DELETE FROM term_aliases
             WHERE source_paper = ?
               AND term_id IN (
                   SELECT id FROM canonical_terms WHERE term_type = 'entity'
               )
            """,
            (paper_name,),
        )

        # Track canonicals resolved this run so (a) entity_count counts
        # distinct entity canonicals from this paper without needing a
        # post-hoc COUNT against term_aliases (which under-counts after
        # the synonym-index revert — tier-1 / tier-5 leave no rows), and
        # (b) we can flag every touched term for the deferred terms_fts
        # rebuild that index_paper drains.
        touched_term_ids: set[int] = set()
        # Schwartz-Hearst short form bookkeeping: only insert the alias
        # once per canonical per paper. Alias insertion is itself
        # idempotent per composite PK.
        aliased_term_ids: set[int] = set()

        for chunk, sub_chunks_list, chunk_valid_spans in per_chunk:
            # Per-section first-seen dedup keyed on normalized name only
            # (one concept per section, regardless of label variation).
            seen: dict[str, tuple[EntityType, float, str]] = {}
            for sub, spans in zip(sub_chunks_list, chunk_valid_spans):
                for span in spans:
                    name = (span.get("text") or "").strip()
                    if not name:
                        continue
                    canonical_text = _canonicalize(name)
                    norm = normalize_term(canonical_text)
                    if norm in seen:
                        continue
                    etype, etype_score = winning_type[norm]
                    seen[norm] = (etype, etype_score, canonical_text)

            for norm, (etype, etype_score, raw_name) in seen.items():
                n_total += 1
                if _is_garbage(raw_name):
                    n_rejected += 1
                    _LOG.debug(
                        "garbage gate rejected name=%r etype=%s breadcrumb=%r",
                        raw_name, etype.value, chunk.breadcrumb,
                    )
                    continue

                resolved = resolve(
                    conn,
                    raw_name,
                    domain=paper_domain,
                    term_type="entity",
                    entity_type=etype.value,
                    entity_type_score=etype_score,
                    source_paper=paper_name,
                    embedder=_get_embedder(),
                )
                touched_term_ids.add(resolved.term_id)

                # Persist the Schwartz-Hearst short form(s) as aliases.
                if (
                    resolved.term_id not in aliased_term_ids
                    and norm in observed_short_forms
                ):
                    for short in observed_short_forms[norm]:
                        insert_acronym_alias(
                            conn,
                            term_id=resolved.term_id,
                            canonical_name=resolved.canonical_name,
                            alias=short,
                            source_paper=paper_name,
                        )
                    aliased_term_ids.add(resolved.term_id)

        entity_count = len(touched_term_ids)

        conn.execute(
            f"""
            UPDATE {target_table}
               SET entity_count = ?,
                   status = ?
             WHERE id = ?
            """,
            (entity_count, target_status_value, source_id),
        )

        # Make sure index_paper rebuilds terms_fts for every canonical we
        # touched, including tier-1 / tier-5 hits that wrote no alias row.
        pending_fts_rebuilds(conn).update(touched_term_ids)

    _LOG.info(
        "garbage gate: rejected %d/%d entities for paper_name=%s",
        n_rejected, n_total, paper_name,
    )
    if (
        n_total >= _REJECT_RATE_WARN_MIN_SAMPLES
        and (n_rejected / n_total) > _REJECT_RATE_WARN
    ):
        _LOG.warning(
            "paper_name=%s garbage rejection rate %.1f%% exceeds %.0f%% — "
            "consider raising per-label thresholds",
            paper_name, 100 * n_rejected / n_total, 100 * _REJECT_RATE_WARN,
        )

    if entity_count == 0:
        _LOG.info(
            "paper %s: 0 entities extracted — possibly expected for short papers",
            paper_name,
        )

    _LOG.info(
        "extracted source_id=%s paper_name=%s entity_count=%d",
        source_id, paper_name, entity_count,
    )

    return ExtractResult(
        paper_name=paper_name,
        entity_count=entity_count,
        status=target_status_value,
    )


# ---------------------------------------------------------------------------
# Garbage gate
# ---------------------------------------------------------------------------


def _is_garbage(name: str) -> bool:
    """True if ``name`` should be rejected before the resolver call.

    Rejects, in order of cheapest check first:

    - empty / whitespace-only
    - contains control whitespace (``\\n``, ``\\r``, ``\\t``): the span
      crossed a paragraph, table row, or list-item boundary, so it's two
      unrelated concepts glued together.
    - pure-numeric
    - ends with dangling punctuation (``-/:,@#``): GLiNER2 span-boundary
      truncation (e.g. ``NV-``, ``foo/``).
    - longer than ``_ENTITY_MAX_LEN``: no legitimate entity name is 100+
      chars; long spans are phrase captures that escaped the structural
      boundary check.
    - document-structure reference (``Table 2``, ``Appendix C``, ``Eq. 7``):
      these are pointers into the paper, not entities.
    - quantity / value-delta (``+5.6%``, ``10K``, ``10.9 ms``, ``209 docs``):
      raw measurements the reader sees in result tables, not entity names.
    - stoplist hit (case-insensitive): generic concept nouns (``vector``,
      ``hybrid``), file formats (``json``, ``pdf``), or banned function words.
    - label-word hit after ``normalize_term``: extractor picked up one of
      its own labels as an entity (``Method``, ``Dataset``, ...).
    - shorter than 3 chars and not in the acronym allowlist.
    """
    s = name.strip()
    if not s:
        return True
    if any(c in s for c in "\n\r\t"):
        return True
    if _NUMERIC_RE.fullmatch(s):
        return True
    if s[-1] in _TRAILING_JUNK:
        return True
    if s[0] in "+-":
        return True
    if len(s) > _ENTITY_MAX_LEN:
        return True
    if _STRUCTURAL_REF_RE.match(s):
        return True
    if _QUANTITY_RE.match(s):
        return True
    if s.lower() in _ENTITY_STOPLIST:
        return True
    if normalize_term(s) in _LABEL_WORDS:
        return True
    if len(s) < _ACRONYM_MIN_LEN and s.upper() not in _ACRONYM_ALLOWLIST:
        return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run GLiNER2 entity extraction on one paper."
    )
    parser.add_argument("--paper", required=True, help="papers.paper_name")
    parser.add_argument("--db", default="lodestone.db", help="sqlite db path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="bypass the can_run_from status guard",
    )
    args = parser.parse_args(argv)

    conn = get_conn(Path(args.db))
    try:
        result = extract(
            paper_name=args.paper,
            conn=conn,
            force=args.force,
        )
    finally:
        conn.close()
    print(json.dumps(result._asdict()))


if __name__ == "__main__":
    _main()
