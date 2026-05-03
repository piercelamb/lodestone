"""Single-LLM classification pass for one paper.

This is the **only** LLM call in the ingest pipeline. Every other stage is
deterministic code or local inference.

The call goes through :mod:`_system.llm` which dispatches to one of three
provider SDKs (Anthropic, OpenAI, Gemini) based on the user's
``~/.config/lodestone/config.toml`` and environment. Structured output is
enforced provider-side (tool_use, response_format=json_schema,
responseSchema) so the LLM cannot return malformed JSON.

Prompt assets live at ``_system/llm/prompts/classify_paper/``:
``system.md``, ``user.md`` (with ``{EXISTING_TAXONOMY}`` /
``{PAPER_CONTENT}`` placeholders), and ``response.json`` (schema with
a ``DOMAIN_INDEX_ENUM`` runtime-replaced sentinel for the index-replace
pattern).

Domain selection uses *index-replace*: the LLM picks an integer index
into the runtime-supplied ``existing_domains`` list (or ``-1`` to
propose a new one). This is cheaper than regenerating a free-form
string, eliminates typos, and stays strict-mode compatible with all
three providers' structured-output modes. Collection and topics remain
free strings — they depend on the chosen domain / are unbounded, so the
5-tier term resolver still canonicalizes them.

On success we resolve ``collection`` and every ``topic`` through the
shared 5-tier term resolver (Section 4) and write the canonical names.
On a proposed new domain we sanitize the name and auto-insert it with
``needs_review=1`` on the *paper* row.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable, NamedTuple

from _system.db.connection import get_conn, transaction
from _system.db.orphan_gc import gc_orphan_topic_collection_canonicals
from _system.llm import call_structured, load_prompt
from _system.resolution.embeddings import Embedder
from _system.resolution.resolver import pending_fts_rebuilds, resolve
from _system.schemas.paper_metadata import PaperStatus, can_run_from
from _system.schemas.taxonomy import ClassificationLLMOutput, ClassificationOutput
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.classify_paper")

_PAPER_CONTENT_MAX_CHARS = 8000
# Display cap on collections per domain. Purely a prompt-size budget —
# overflow surfaces as a sibling leaf that nudges the LLM to propose new
# rather than guess at hidden entries.
_COLLECTIONS_PER_DOMAIN_LIMIT = 30
_DOMAIN_MAX_LEN = 32

_WS_OR_SLASH_RE = re.compile(r"[\s/]+")
_DOMAIN_ALLOWED_RE = re.compile(r"[^a-z0-9_-]")


class ClassifyError(Exception):
    """Base class for classify_paper failures."""


class ClassifyLLMError(ClassifyError):
    """The LLM call failed (transport, auth, schema, or bad index).

    Wraps :class:`_system.llm.errors.LLMError` subclasses raised from
    inside the dispatch layer, plus index-range validation we perform
    after the call.
    """


class ClassifyStateError(ClassifyError):
    """`can_run_from` rejected the current status for CLASSIFIED."""


class ClassifyDomainNameError(ClassifyError):
    """Proposed domain name sanitizes to an empty string."""


class ClassifyPaperNotFound(ClassifyError):
    """No papers row for the given paper_name."""


class ClassifyResult(NamedTuple):
    paper_name: str
    domain: str
    collection: str
    topics: tuple[str, ...]
    needs_review: bool
    status: str


class _DomainDecision(NamedTuple):
    """Pure-function output of `_choose_domain`; no DB side effects."""

    name: str
    insert_new: bool
    paper_needs_review: bool


def classify(
    *,
    paper_name: str,
    conn: sqlite3.Connection,
    force: bool = False,
    domain_override: str | None = None,
    call_llm=None,
    embedder: Embedder | None = None,
) -> ClassifyResult:
    """Run the single-LLM classification pass for one paper.

    ``call_llm`` is a test seam. Production leaves it ``None`` and the
    module-level :func:`_call_llm_default` dispatches through
    :func:`_system.llm.call_structured`. Its signature is::

        call_llm(system: str, user: str, schema: dict,
                 response_model: type[T]) -> T
    """
    del force  # see docstring on orchestrator parity

    row = conn.execute(
        """
        SELECT id, status, abstract, markdown
          FROM papers
         WHERE paper_name = ?
        """,
        (paper_name,),
    ).fetchone()
    if row is None:
        raise ClassifyPaperNotFound(
            f"paper_name={paper_name!r} not found in papers table"
        )
    paper_id, status_str, abstract, markdown = row

    try:
        current = PaperStatus(status_str) if status_str else None
    except ValueError as exc:
        raise ClassifyStateError(
            f"paper_name={paper_name!r}: unrecognized status={status_str!r}"
        ) from exc
    if not can_run_from(current, PaperStatus.CLASSIFIED):
        extra = (
            " (FAILED_HTML is terminal — re-fetch required)"
            if current is PaperStatus.FAILED_HTML
            else ""
        )
        raise ClassifyStateError(
            f"paper_name={paper_name!r}: cannot run CLASSIFIED from status="
            f"{status_str!r}{extra}"
        )

    paper_content = _head_slice_paper_content(
        markdown=markdown or "", abstract=abstract or ""
    )
    del markdown

    existing_domains = _load_domains(conn)
    existing_domain_names = {d[0] for d in existing_domains}
    collections_by_domain, overflow = _truncate_collections(
        _load_collections_by_domain(conn)
    )
    max_collections = max(
        (len(colls) for colls in collections_by_domain.values()), default=0
    )

    loaded = load_prompt(
        "classify_paper",
        md_context={
            "EXISTING_TAXONOMY": _render_taxonomy_tree(
                existing_domains, collections_by_domain, overflow
            ),
            "PAPER_CONTENT": paper_content,
        },
        schema_replacements={
            "DOMAIN_INDEX_ENUM": [-1, *range(len(existing_domains))],
            "COLLECTION_INDEX_ENUM": [-1, *range(max_collections)],
        },
    )

    # Fresh terms miss tiers 1-4 and need an Embedder for tier 5.
    if embedder is None:
        embedder = Embedder()

    runner = call_llm or _call_llm_default
    raw = runner(loaded.system, loaded.user, loaded.schema, ClassificationLLMOutput)

    output = _resolve_raw(raw, existing_domains, collections_by_domain)

    decision = _choose_domain(
        proposed=output.domain,
        domain_is_new=output.domain_is_new,
        override=domain_override,
        existing_domains=existing_domain_names,
    )

    with transaction(conn):
        if decision.insert_new:
            # LLM description applies only when the LLM actually proposed
            # this domain. With --domain-override the operator picked a
            # different name, so any LLM description is about a different
            # research area — store NULL and let the operator fill it in.
            new_description = (
                output.domain_description
                if output.domain_is_new and domain_override is None
                else None
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO domains (name, description)
                VALUES (?, ?)
                """,
                (decision.name, new_description),
            )

        conn.execute("DELETE FROM paper_topics WHERE paper_id = ?", (paper_id,))

        # Track every canonical this stage resolves so index_paper can
        # rebuild terms_fts for them. Under the synonym-index revert,
        # tier-1 / tier-5 hits leave no alias trail, so the deferred
        # rebuild queue is the only signal for unchanged collection /
        # topic canonicals.
        touched_term_ids: set[int] = set()

        collection_hit = resolve(
            conn,
            output.collection,
            domain=decision.name,
            term_type="collection",
            source_paper=paper_name,
            embedder=embedder,
        )
        touched_term_ids.add(collection_hit.term_id)

        # Register the (domain, collection) pair in the first-class table.
        # INSERT OR IGNORE is a no-op when the resolver canonicalized to an
        # already-registered collection; in that case we keep the existing
        # description rather than overwriting with whatever the LLM wrote
        # for a name it thought was novel. A genuinely new collection
        # lands with the LLM's description attached.
        conn.execute(
            """
            INSERT OR IGNORE INTO collections (domain, name, description)
            VALUES (?, ?, ?)
            """,
            (
                decision.name,
                collection_hit.canonical_name,
                output.collection_description,
            ),
        )

        seen_term_ids: set[int] = set()
        inserted_topic_names: list[str] = []
        for topic_raw in output.topics:
            topic_hit = resolve(
                conn,
                topic_raw,
                domain=decision.name,
                term_type="topic",
                source_paper=paper_name,
                embedder=embedder,
            )
            touched_term_ids.add(topic_hit.term_id)
            if topic_hit.term_id in seen_term_ids:
                continue
            seen_term_ids.add(topic_hit.term_id)
            conn.execute(
                """
                INSERT INTO paper_topics (paper_id, domain, topic)
                VALUES (?, ?, ?)
                """,
                (paper_id, decision.name, topic_hit.canonical_name),
            )
            inserted_topic_names.append(topic_hit.canonical_name)

        conn.execute(
            """
            UPDATE papers
               SET domain = ?,
                   collection = ?,
                   needs_review = ?,
                   status = ?
             WHERE id = ?
            """,
            (
                decision.name,
                collection_hit.canonical_name,
                int(decision.paper_needs_review),
                PaperStatus.CLASSIFIED.value,
                paper_id,
            ),
        )

        pending_fts_rebuilds(conn).update(touched_term_ids)

        # Re-classify deletes the paper's paper_topics rows up front and
        # re-runs the LLM. Topic canonicals from the *previous* run that
        # the new run didn't re-bind are now orphaned in canonical_terms;
        # collection canonicals can also orphan if the only paper that
        # used them just moved off. GC them here, after all new bindings
        # are in place. A GC'd canonical's deferred FTS rebuild becomes a
        # no-op against an absent row (harmless).
        gc_orphan_topic_collection_canonicals(conn)

    _LOG.info(
        "classified paper_id=%s paper_name=%s domain=%s collection=%s "
        "topics=%d needs_review=%s",
        paper_id, paper_name, decision.name, collection_hit.canonical_name,
        len(inserted_topic_names), decision.paper_needs_review,
    )

    return ClassifyResult(
        paper_name=paper_name,
        domain=decision.name,
        collection=collection_hit.canonical_name,
        topics=tuple(inserted_topic_names),
        needs_review=decision.paper_needs_review,
        status=PaperStatus.CLASSIFIED.value,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_llm_default(
    system: str, user: str, schema: dict, response_model
):
    """Production runner: delegate to the dispatch layer.

    Separating this from ``call_structured`` keeps the test seam narrow
    (tests inject any callable with this signature) while leaving the
    dispatch layer free to handle retry + provider resolution.
    """
    return call_structured(
        system=system,
        user=user,
        schema=schema,
        response_model=response_model,
    )


# ---------------------------------------------------------------------------
# Resolve raw → output
# ---------------------------------------------------------------------------


def _resolve_raw(
    raw: ClassificationLLMOutput,
    existing_domains: list[tuple[str, str | None]],
    collections_by_domain: dict[str, list[tuple[str, str | None]]],
) -> ClassificationOutput:
    """Translate the index-based LLM output to a name-based pipeline shape.

    Domain: ``-1`` → propose new; ``0..N-1`` → existing.
    Collection: ``-1`` → propose new; ``0..M-1`` → lookup in the truncated
    collection list for the chosen domain.

    Cross-field rules the schema enum cannot express are enforced here:

    - domain_index out of ``[-1, N-1]`` → :class:`ClassifyLLMError`
    - domain_index == -1 but new_domain_desc empty → raise
    - domain_index >= 0 but new_domain_desc non-empty → raise
      (LLM is writing a description for a domain it didn't propose)
    - collection_index >= 0 while domain_index == -1 (new domain has no
      existing collections) → raise
    - collection_index >= 0 but out of range for the chosen domain's
      collections → raise
    - collection_index == -1 but new_collection empty → raise
    """
    if raw.domain_index == -1:
        domain_name = raw.new_domain
        domain_is_new = True
        domain_description: str | None = raw.new_domain_desc.strip()
        if not domain_description:
            raise ClassifyLLMError(
                "LLM returned domain_index=-1 but "
                "new_domain_desc is empty; new domains "
                "must include a one-sentence description of the research area"
            )
    elif 0 <= raw.domain_index < len(existing_domains):
        domain_name = existing_domains[raw.domain_index][0]
        domain_is_new = False
        if raw.new_domain_desc.strip():
            raise ClassifyLLMError(
                f"LLM picked existing domain_index={raw.domain_index} but "
                f"also set new_domain_desc="
                f"{raw.new_domain_desc!r}; description "
                f"must be empty unless domain_index == -1"
            )
        domain_description = None
    else:
        raise ClassifyLLMError(
            f"LLM returned domain_index={raw.domain_index} outside "
            f"[-1, {len(existing_domains) - 1}]; schema enum was supposed "
            f"to make this impossible"
        )

    if raw.collection_index == -1:
        proposed = raw.new_collection.strip()
        if not proposed:
            raise ClassifyLLMError(
                "LLM returned collection_index=-1 but "
                "new_collection is empty"
            )
        collection_name = proposed
        collection_description: str | None = (
            raw.new_collection_desc.strip()
        )
        if not collection_description:
            raise ClassifyLLMError(
                "LLM returned collection_index=-1 but "
                "new_collection_desc is empty; new "
                "collections must include a one-sentence description"
            )
    elif raw.collection_index >= 0:
        if domain_is_new:
            raise ClassifyLLMError(
                f"LLM proposed new domain={domain_name!r} but set "
                f"collection_index={raw.collection_index}; new domains have "
                f"no existing collections — collection_index must be -1"
            )
        domain_colls = collections_by_domain.get(domain_name, [])
        if raw.collection_index >= len(domain_colls):
            raise ClassifyLLMError(
                f"LLM returned collection_index={raw.collection_index} "
                f"for domain={domain_name!r}, which has "
                f"{len(domain_colls)} collection(s) — index out of range"
            )
        collection_name = domain_colls[raw.collection_index][0]
        if raw.new_collection_desc.strip():
            raise ClassifyLLMError(
                f"LLM picked existing collection_index={raw.collection_index} "
                f"but also set new_collection_desc="
                f"{raw.new_collection_desc!r}; description "
                f"must be empty unless collection_index == -1"
            )
        collection_description = None
    else:
        raise ClassifyLLMError(
            f"LLM returned collection_index={raw.collection_index} outside "
            f"[-1, max); schema enum was supposed to make this impossible"
        )

    return ClassificationOutput(
        domain=domain_name,
        domain_is_new=domain_is_new,
        domain_description=domain_description,
        collection=collection_name,
        collection_description=collection_description,
        topics=raw.topics,
    )


# ---------------------------------------------------------------------------
# Paper content head-slice
# ---------------------------------------------------------------------------


def _head_slice_paper_content(*, markdown: str, abstract: str) -> str:
    """Return up to ~8K chars of paper content for classification.

    The head of the markdown naturally contains title + abstract + start of
    introduction — everything the LLM needs to pick a domain/collection/topics.
    If markdown is missing (stale row, upstream conversion skipped), fall back
    to the abstract column alone.
    """
    stripped = markdown.strip()
    if stripped:
        return stripped[:_PAPER_CONTENT_MAX_CHARS]
    return abstract.strip()


# ---------------------------------------------------------------------------
# Taxonomy context
# ---------------------------------------------------------------------------


def _load_domains(conn: sqlite3.Connection) -> list[tuple[str, str | None]]:
    rows = conn.execute(
        "SELECT name, description FROM domains ORDER BY name"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _load_collections_by_domain(
    conn: sqlite3.Connection,
) -> dict[str, list[tuple[str, str | None]]]:
    """Return ``{domain: [(name, description), ...]}`` ordered by popularity.

    Source of truth is the first-class ``collections`` table; paper count
    is computed via LEFT JOIN so empty collections (registered but with
    no papers yet) still appear — that's the whole point of first-classing
    collections. Most-used collections rank first so the LLM sees the
    heavy hitters at the top of each domain.
    """
    rows = conn.execute(
        """
        SELECT c.domain, c.name, c.description, COUNT(p.id) AS paper_count
          FROM collections c
          LEFT JOIN papers p
            ON p.domain = c.domain AND p.collection = c.name
         GROUP BY c.domain, c.name, c.description
         ORDER BY c.domain, paper_count DESC, c.name
        """
    ).fetchall()
    result: dict[str, list[tuple[str, str | None]]] = {}
    for domain, name, description, _count in rows:
        result.setdefault(domain, []).append((name, description))
    return result


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _truncate_collections(
    raw: dict[str, list[tuple[str, str | None]]],
) -> tuple[dict[str, list[tuple[str, str | None]]], dict[str, int]]:
    """Cap each domain's collections at ``_COLLECTIONS_PER_DOMAIN_LIMIT``.

    Returns ``(truncated, overflow)`` where ``overflow[domain]`` is the
    count of hidden collections for that domain (``0`` when nothing was
    truncated; absent when the domain had no collections at all). The
    truncated view is what both the tree renderer and ``_resolve_raw``
    should use, so index labels and index→name lookup stay consistent.
    """
    truncated: dict[str, list[tuple[str, str | None]]] = {}
    overflow: dict[str, int] = {}
    for domain, colls in raw.items():
        if len(colls) > _COLLECTIONS_PER_DOMAIN_LIMIT:
            truncated[domain] = list(colls[:_COLLECTIONS_PER_DOMAIN_LIMIT])
            overflow[domain] = len(colls) - _COLLECTIONS_PER_DOMAIN_LIMIT
        else:
            truncated[domain] = list(colls)
    return truncated, overflow


def _render_taxonomy_tree(
    domains: list[tuple[str, str | None]],
    collections_by_domain: dict[str, list[tuple[str, str | None]]],
    overflow: dict[str, int] | None = None,
) -> str:
    """Render domains + their integer-indexed collections as a tree.

    Shape::

        0. rag — retrieval augmented generation
           ├── 0: hybrid_search — dense+sparse retrieval fusion
           └── 1: rag_systems — end-to-end retrieval + generation pipelines
        1. agents — multi-agent systems   (no existing collections)
        2. theorem_proving
           ├── 0: saturation_methods
           ├── 1: superposition
           └── (+ 4 more exist; feel free to propose new)

    Collection indices reset per domain. Domain uses ``N.`` suffix;
    collection uses ``N:`` suffix — different punctuation plus the
    indent + tree chars keeps the two levels unambiguous. Truncation
    past ``_COLLECTIONS_PER_DOMAIN_LIMIT`` surfaces as a label-free
    sibling leaf so the LLM sees more exist but can't "pick" them.
    Descriptions for both levels are shown inline when present; NULL
    descriptions (legacy collections backfilled from papers, or manually
    created without one) render as just ``N: name``.
    """
    overflow = overflow or {}
    if not domains:
        return (
            "(taxonomy is empty — propose a new domain by setting "
            "domain_index to -1, and a new collection under it by "
            "setting collection_index to -1)"
        )

    lines: list[str] = []
    for i, (name, description) in enumerate(domains):
        colls = collections_by_domain.get(name, [])
        head = f"{i}. {name} — {description}" if description else f"{i}. {name}"
        if not colls:
            lines.append(f"{head}   (no existing collections)")
            continue

        lines.append(head)
        has_overflow = overflow.get(name, 0) > 0
        n_leaves = len(colls) + (1 if has_overflow else 0)
        for j, (coll_name, coll_description) in enumerate(colls):
            connector = "└──" if j == n_leaves - 1 else "├──"
            leaf = (
                f"{j}: {coll_name} — {coll_description}"
                if coll_description
                else f"{j}: {coll_name}"
            )
            lines.append(f"   {connector} {leaf}")
        if has_overflow:
            more = overflow[name]
            lines.append(
                f"   └── (+ {more} more exist; feel free to propose new)"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Domain handling
# ---------------------------------------------------------------------------


def _sanitize_domain(proposed: str) -> str:
    lowered = proposed.lower()
    collapsed = _WS_OR_SLASH_RE.sub("_", lowered)
    stripped = _DOMAIN_ALLOWED_RE.sub("", collapsed)
    trimmed = stripped[:_DOMAIN_MAX_LEN]
    return trimmed.strip("_-")


def _choose_domain(
    *,
    proposed: str,
    domain_is_new: bool,
    override: str | None,
    existing_domains: set[str],
) -> _DomainDecision:
    if override is not None:
        return _DomainDecision(
            name=override,
            insert_new=override not in existing_domains,
            paper_needs_review=False,
        )

    treat_as_new = domain_is_new or proposed not in existing_domains
    if treat_as_new:
        sanitized = _sanitize_domain(proposed)
        if not sanitized:
            raise ClassifyDomainNameError(
                f"proposed domain {proposed!r} sanitizes to empty string"
            )
        if sanitized in existing_domains:
            return _DomainDecision(sanitized, insert_new=False, paper_needs_review=False)
        return _DomainDecision(sanitized, insert_new=True, paper_needs_review=True)

    return _DomainDecision(proposed, insert_new=False, paper_needs_review=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify one paper via the configured LLM provider."
    )
    parser.add_argument("--paper", required=True, help="papers.paper_name")
    parser.add_argument("--db", default="lodestone.db", help="sqlite db path")
    parser.add_argument(
        "--force",
        action="store_true",
        help="no-op at classify level; forwarded for parity",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="override the LLM's domain choice with this exact name",
    )
    args = parser.parse_args(argv if argv is None else list(argv))

    conn = get_conn(Path(args.db))
    try:
        result = classify(
            paper_name=args.paper,
            conn=conn,
            force=args.force,
            domain_override=args.domain,
        )
    finally:
        conn.close()
    print(json.dumps(result._asdict()))


if __name__ == "__main__":
    _main()
