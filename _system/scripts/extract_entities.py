"""GLiNER2 entity extraction for a single paper.

Fourth pipeline stage (after ``fetch`` → ``convert`` → ``classify``): loads the
paper's markdown, splits into sections via the shared section splitter, sub-
chunks each section under GLiNER2's 384-token ceiling, runs labels-with-
descriptions inference, applies the garbage gate, resolves every surviving
entity name through the shared 5-tier resolver, and writes canonical rows into
``entities`` with replace-all semantics per paper. Sets
``papers.status = 'extracted'`` on success.

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
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

from _system.db.connection import get_conn, transaction
from _system.resolution.embeddings import Embedder
from _system.resolution.normalize import normalize_term
from _system.resolution.resolver import resolve
from _system.schemas.entities import EntityType
from _system.schemas.paper_metadata import PaperStatus, can_run_from
from _system.utils.config import load_gliner_config
from _system.utils.logging import get_logger
from _system.utils.sections import split_sections, strip_breadcrumb, sub_chunk

_LOG = get_logger("scripts.extract_entities")

_DESCRIPTION_WINDOW_CHARS = 200
_DESCRIPTION_MAX_CHARS = 240
_ACRONYM_MIN_LEN = 3
_REJECT_RATE_WARN = 0.5
_REJECT_RATE_WARN_MIN_SAMPLES = 10

ACRONYM_ALLOWLIST: frozenset[str] = frozenset({"LM", "QA", "NN", "AI", "ML", "NLP"})
ENTITY_STOPLIST: frozenset[str] = frozenset(
    {"table", "figure", "we", "using", "our", "this", "these", "that", "it", "however"}
)
# Label words the extractor must reject so "Method"/"Dataset" spans never reach
# the resolver. Derived from EntityType so the two can't drift.
_LABEL_WORDS: frozenset[str] = frozenset(t.value for t in EntityType)

_NUMERIC_RE = re.compile(r"\d+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]\s+")


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


# ---------------------------------------------------------------------------
# Span shape: flat dict with keys text/label/score/start/end.
# Signature for the test seam:
#   (text, label_descriptions, threshold) -> list[span_dict]
# ---------------------------------------------------------------------------

Span = dict
InferenceFn = Callable[[str, dict[str, str], float], list[Span]]
TokenizeFn = Callable[[str], list[str]]


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
        from gliner2 import GLiNER2  # heavy import — deferred
        _MODEL = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
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


def _default_tokenize(text: str) -> list[str]:
    """Production tokenizer; GLiNER2's subword tokenizer so sub-chunks respect
    the model's 384-token ceiling (a word-count approximation would exceed
    the ceiling on dense prose).
    """
    return _get_model().processor.tokenizer.tokenize(text)


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
                {
                    "text": item.get("text", ""),
                    "label": label,
                    "score": float(item.get("confidence", item.get("score", 0.0))),
                    "start": int(item.get("start", 0)),
                    "end": int(item.get("end", 0)),
                }
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

    ``run_inference`` is the test seam for GLiNER2. ``tokenize`` defaults to
    the word-split used by :func:`sub_chunk`; production callers typically
    leave it None (see note in :func:`sub_chunk`). ``embedder`` is forwarded
    to the resolver for tier 5 inserts. Fresh entity names miss tiers 1-4 and
    need an Embedder; absent one we lazily construct the real BGE embedder.
    """
    row = conn.execute(
        """
        SELECT id, domain, status, markdown
          FROM papers WHERE paper_name = ?
        """,
        (paper_name,),
    ).fetchone()
    if row is None:
        raise PaperNotFound(f"paper_name={paper_name!r} not found in papers table")
    paper_id, domain, status_str, markdown = row

    if markdown is None:
        raise MarkdownMissing(f"paper_name={paper_name!r}: markdown is NULL")

    try:
        current = PaperStatus(status_str) if status_str else None
    except ValueError as exc:
        raise UnknownStatusError(
            f"paper_name={paper_name!r}: unrecognized status={status_str!r}"
        ) from exc
    if not force and not can_run_from(current, PaperStatus.EXTRACTED):
        extra = (
            " (FAILED_HTML is terminal — re-fetch required)"
            if current is PaperStatus.FAILED_HTML
            else ""
        )
        raise StatusTooLow(
            f"paper_name={paper_name!r}: cannot run EXTRACTED from status="
            f"{status_str!r}{extra}"
        )

    cfg = (
        load_gliner_config(config_path)
        if config_path is not None
        else load_gliner_config()
    )

    label_to_type = _build_label_map(cfg.label_descriptions.keys())

    inference = run_inference or _default_inference
    # Default to GLiNER2's subword tokenizer so sub-chunks honour the
    # 384-token ceiling. Tests that pass a canned `run_inference` typically
    # leave `tokenize=None` and their sections are short enough to fit in
    # one sub-chunk, so tokenize() is never invoked.
    tokenize_fn = tokenize if tokenize is not None else (
        _default_tokenize if run_inference is None else None
    )

    # Fresh spans hit tier 5 (INSERT into canonical_terms) which needs an
    # embedder. Defer construction until the first resolver call that actually
    # needs one — all-tier-1 papers never load sentence-transformers.
    _lazy_embedder_slot: list[Embedder | None] = [embedder]

    def _get_embedder() -> Embedder:
        if _lazy_embedder_slot[0] is None:
            _lazy_embedder_slot[0] = Embedder()
        return _lazy_embedder_slot[0]

    # Inference-time floor: use the smallest per-label threshold so GLiNER2
    # returns candidates for every label; per-label filtering below applies
    # the exact threshold per span.
    min_threshold = min([*cfg.per_label.values(), cfg.global_threshold])

    chunks = split_sections(markdown)

    n_rejected = 0
    n_total = 0
    paper_domain = domain or ""

    with transaction(conn):
        conn.execute("DELETE FROM entities WHERE paper_id = ?", (paper_id,))

        for chunk in chunks:
            stripped = strip_breadcrumb(chunk.body)
            sub_chunks_list = sub_chunk(
                stripped,
                max_tokens=cfg.chunk.max_tokens,
                overlap_tokens=cfg.chunk.overlap_tokens,
                tokenizer_cb=tokenize_fn,
            )
            # Per-section first-seen dedup on (entity_type, normalize_term(name)).
            # Value carries the sub-chunk text + in-chunk offset so we can build
            # a description in the same slice the span was found.
            seen: dict[tuple[EntityType, str], tuple[str, str, int]] = {}
            for sub in sub_chunks_list:
                spans = inference(sub, dict(cfg.label_descriptions), min_threshold)
                for span in spans:
                    label = span.get("label", "")
                    name = (span.get("text") or "").strip()
                    if not name:
                        continue
                    score = float(span.get("score", 0.0))
                    start = int(span.get("start", 0))
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
                    key = (etype, normalize_term(name))
                    if key in seen:
                        continue
                    seen[key] = (name, sub, start)

            for (etype, _norm), (raw_name, sub_text, offset) in seen.items():
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
                    source_paper=paper_name,
                    embedder=_get_embedder(),
                )
                description = _description_for(sub_text, offset)

                conn.execute(
                    """
                    INSERT OR IGNORE INTO entities
                      (paper_id, domain, paper_name, entity_name, entity_type,
                       source_breadcrumb, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        paper_domain,
                        paper_name,
                        resolved.canonical_name,
                        etype.value,
                        chunk.breadcrumb,
                        description,
                    ),
                )

        entity_count = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE paper_id = ?", (paper_id,)
        ).fetchone()[0]

        conn.execute(
            """
            UPDATE papers
               SET entity_count = ?,
                   status = ?
             WHERE id = ?
            """,
            (entity_count, PaperStatus.EXTRACTED.value, paper_id),
        )

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
        "extracted paper_id=%s paper_name=%s entity_count=%d",
        paper_id, paper_name, entity_count,
    )

    return ExtractResult(
        paper_name=paper_name,
        entity_count=entity_count,
        status=PaperStatus.EXTRACTED.value,
    )


def _build_label_map(labels: Iterable[str]) -> dict[str, EntityType]:
    """Map GLiNER label keys (e.g. 'Method') to EntityType members.

    Keys whose lowercased form doesn't correspond to an EntityType value are
    logged as WARNING and omitted from the map — at runtime such spans are
    dropped without writing to the DB.
    """
    out: dict[str, EntityType] = {}
    for label in labels:
        try:
            out[label] = EntityType(label.lower())
        except ValueError:
            _LOG.warning(
                "GLiNER config label %r does not map to any EntityType; ignoring",
                label,
            )
    return out


# ---------------------------------------------------------------------------
# Garbage gate
# ---------------------------------------------------------------------------


def _is_garbage(name: str) -> bool:
    """True if ``name`` should be rejected before the resolver call.

    Rejects: empty, pure-numeric, stoplist hit (case-insensitive), label-word
    hit after ``normalize_term``, and shorter-than-3 names that aren't in the
    acronym allowlist.
    """
    s = name.strip()
    if not s:
        return True
    if _NUMERIC_RE.fullmatch(s):
        return True
    if s.lower() in ENTITY_STOPLIST:
        return True
    if normalize_term(s) in _LABEL_WORDS:
        return True
    if len(s) < _ACRONYM_MIN_LEN and s.upper() not in ACRONYM_ALLOWLIST:
        return True
    return False


# ---------------------------------------------------------------------------
# Description extraction
# ---------------------------------------------------------------------------


def _description_for(sub_chunk_text: str, offset: int) -> str:
    """Return a ≤240-char description centered on ``offset`` within ``sub_chunk_text``.

    Strategy: take a ±200-char window around ``offset``, then pick the
    sentence fragment covering ``offset`` using ``[.!?]\\s+`` as the sentence
    boundary. Falls back to the raw window when no boundary falls inside it.
    Hard-truncates at 240 chars (no ellipsis).
    """
    n = len(sub_chunk_text)
    if n == 0:
        return ""

    lo = max(0, offset - _DESCRIPTION_WINDOW_CHARS)
    hi = min(n, offset + _DESCRIPTION_WINDOW_CHARS)
    window = sub_chunk_text[lo:hi]
    if not window:
        return ""
    local_offset = max(0, min(offset - lo, len(window) - 1))

    frag_start = 0
    frag_end = len(window)
    boundary_seen = False
    for m in _SENTENCE_SPLIT_RE.finditer(window):
        boundary_seen = True
        if local_offset < m.start():
            frag_end = m.start()
            break
        frag_start = m.end()
    else:
        if boundary_seen:
            frag_end = len(window)

    picked = window[frag_start:frag_end].strip() if boundary_seen else window.strip()
    if not picked:
        picked = window.strip()
    if len(picked) > _DESCRIPTION_MAX_CHARS:
        picked = picked[:_DESCRIPTION_MAX_CHARS]
    return picked


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
