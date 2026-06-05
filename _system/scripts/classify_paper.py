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
``DOMAIN_INDEX_ENUM`` / ``COLLECTION_INDEX_ENUM`` runtime-replaced
sentinels for the index-replace pattern).

Domain selection uses *index-replace*: the LLM picks an integer index
into the runtime-supplied ``existing_domains`` list (or ``-1`` to
propose a new one). This is cheaper than regenerating a free-form
string, eliminates typos, and stays strict-mode compatible with all
three providers' structured-output modes. Collections use the same
index-replace pattern but per-entry inside an ordered list — index 0 is
the paper's PRIMARY collection, indices 1+ are SECONDARY memberships
within the same domain (max 4 total). Topics remain free strings —
they're unbounded so the 5-tier term resolver still canonicalizes them.

On success we resolve every collection and every ``topic`` through the
shared term resolver (Section 4) and write the canonical names. Topics
flow through the full 5-tier ladder because the LLM never sees the
existing topic set — they're free strings. Collections are gated on
overflow: when the chosen domain's collection list was truncated in the
prompt (``DomainNode.overflow > 0``) the resolver runs the full ladder so
it can catch picks that collide with canonicals the LLM didn't see;
otherwise the LLM had the comparison set in front of it and we run tier
1 → tier 5 only, trusting the LLM's verbatim choice without letting the
fuzzy ladder second-guess it. The denormalized ``papers.collection`` (or
``posts.collection``) scalar always points at the primary collection;
the full set lives in the polymorphic ``collections`` junction keyed by
``(target_kind, target_id)``. On a proposed new domain we sanitize the
name and auto-insert it with ``needs_review=1``; ``needs_review`` is
also set when any picked collection is new.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Iterable, NamedTuple

from _system.db.connection import get_conn, transaction
from _system.db.orphan_gc import gc_orphan_topic_canonicals
from _system.llm import call_structured, load_prompt
from _system.resolution.embeddings import Embedder
from _system.resolution.resolver import pending_fts_rebuilds, resolve
from _system.schemas.paper_metadata import PaperStatus, can_run_from as paper_can_run_from
from _system.schemas.post_metadata import PostStatus, can_run_from as post_can_run_from
from _system.schemas.taxonomy import (
    ClassificationLLMOutput,
    ClassificationOutput,
    ResolvedCollection,
)
from _system.scripts.taxonomy_tree import (
    DomainNode,
    TaxonomyTreeStyle,
    load_taxonomy,
    render_taxonomy_tree,
)
from _system.utils.logging import get_logger
from _system.utils.slug import sanitize_domain
from _system.utils.source_resolution import (
    SlugNotFound,
    SourceKind,
    resolve_slug,
)

_LOG = get_logger("scripts.classify_paper")

_PAPER_CONTENT_MAX_CHARS = 8000
_POST_CONTENT_MAX_CHARS = 10000
# Display cap on collections per domain. Purely a prompt-size budget —
# overflow surfaces as a sibling leaf that nudges the LLM to propose new
# rather than guess at hidden entries.
_COLLECTIONS_PER_DOMAIN_LIMIT = 30


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
    collections: tuple[str, ...]
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

    try:
        resolved = resolve_slug(conn, paper_name)
    except SlugNotFound as exc:
        raise ClassifyPaperNotFound(
            f"slug={paper_name!r} not found in papers or posts"
        ) from exc
    kind = resolved.kind
    source_id = resolved.id

    table = "papers" if kind is SourceKind.PAPER else "posts"
    row = conn.execute(
        f"SELECT status, abstract, markdown FROM {table} WHERE id = ?",
        (source_id,),
    ).fetchone()
    if row is None:
        raise ClassifyPaperNotFound(
            f"slug={paper_name!r} resolved to id={source_id} but row vanished"
        )
    status_str, abstract, markdown = row

    if kind is SourceKind.PAPER:
        try:
            current = PaperStatus(status_str) if status_str else None
        except ValueError as exc:
            raise ClassifyStateError(
                f"paper_name={paper_name!r}: unrecognized status={status_str!r}"
            ) from exc
        if not paper_can_run_from(current, PaperStatus.CLASSIFIED):
            extra = (
                " (FAILED_HTML is terminal — re-fetch required)"
                if current is PaperStatus.FAILED_HTML
                else ""
            )
            raise ClassifyStateError(
                f"paper_name={paper_name!r}: cannot run CLASSIFIED from status="
                f"{status_str!r}{extra}"
            )
    else:
        try:
            current_post = PostStatus(status_str) if status_str else None
        except ValueError as exc:
            raise ClassifyStateError(
                f"post_name={paper_name!r}: unrecognized status={status_str!r}"
            ) from exc
        if not post_can_run_from(current_post, PostStatus.CLASSIFIED):
            extra = (
                " (terminal failure — re-fetch required)"
                if current_post in (PostStatus.FAILED_FETCH, PostStatus.FAILED_PARSE)
                else ""
            )
            raise ClassifyStateError(
                f"post_name={paper_name!r}: cannot run CLASSIFIED from status="
                f"{status_str!r}{extra}"
            )

    max_chars = (
        _PAPER_CONTENT_MAX_CHARS
        if kind is SourceKind.PAPER
        else _POST_CONTENT_MAX_CHARS
    )
    paper_content = _head_slice_paper_content(
        markdown=markdown or "", abstract=abstract or "", max_chars=max_chars,
    )
    del markdown

    if domain_override is not None:
        # Lock the LLM to a single domain: sanitize the operator-supplied
        # name to its canonical slug, then load only that domain's subtree
        # (no truncation — the prompt is already shrunk to one domain).
        # If the domain row doesn't exist yet, synthesize an empty node so
        # the prompt still renders coherently; the row gets created in
        # the success transaction below (no DB write before the LLM call).
        domain_override = sanitize_domain(domain_override)
        if not domain_override:
            raise ClassifyDomainNameError(
                "domain_override sanitizes to empty string"
            )
        existing_domains = load_taxonomy(
            conn,
            domain=domain_override,
            include_empty_collections=True,
            include_empty_domains=True,
            collections_per_domain_limit=None,
        )
        # Track whether the override domain already exists in the DB —
        # only DB-loaded rows go into existing_domain_names, otherwise
        # _choose_domain would compute insert_new=False against the
        # synthesized node and the new domain would never be created.
        domain_row_exists = bool(existing_domains)
        if not existing_domains:
            existing_domains = [
                DomainNode(
                    name=domain_override,
                    description=None,
                    paper_count=0,
                    collections=(),
                    overflow=0,
                    repo_count=0,
                    post_count=0,
                )
            ]
        existing_domain_names = (
            {d.name for d in existing_domains} if domain_row_exists else set()
        )
        # Pin the schema enums: domain is locked to index 0, collection
        # enum spans -1 (propose new in this domain) plus every existing
        # collection in this domain (no 30-cap). Zero-collection domain
        # collapses to [-1], which the schema already exercises.
        domain_enum = [0]
        collection_enum = [-1, *range(len(existing_domains[0].collections))]
    else:
        existing_domains = load_taxonomy(
            conn,
            include_empty_collections=True,
            include_empty_domains=True,
            collections_per_domain_limit=_COLLECTIONS_PER_DOMAIN_LIMIT,
        )
        existing_domain_names = {d.name for d in existing_domains}
        max_collections = max(
            (len(d.collections) for d in existing_domains), default=0
        )
        domain_enum = [-1, *range(len(existing_domains))]
        collection_enum = [-1, *range(max_collections)]

    if kind is SourceKind.PAPER:
        prompt_name = "classify_paper"
        content_placeholder = "PAPER_CONTENT"
    else:
        prompt_name = "classify_post"
        content_placeholder = "POST_CONTENT"
    loaded = load_prompt(
        prompt_name,
        md_context={
            "EXISTING_TAXONOMY": render_taxonomy_tree(
                existing_domains,
                style=TaxonomyTreeStyle.INDEX,
                overflow_message="(+ {n} more exist; feel free to propose new)",
            ),
            content_placeholder: paper_content,
        },
        schema_replacements={
            "DOMAIN_INDEX_ENUM": domain_enum,
            "COLLECTION_INDEX_ENUM": collection_enum,
        },
    )

    # Fresh terms miss tiers 1-4 and need an Embedder for tier 5.
    if embedder is None:
        embedder = Embedder()

    runner = call_llm or _call_llm_default
    raw = runner(loaded.system, loaded.user, loaded.schema, ClassificationLLMOutput)

    output = _resolve_raw(raw, existing_domains)

    decision = _choose_domain(
        proposed=output.domain,
        domain_is_new=output.domain_is_new,
        override=domain_override,
        existing_domains=existing_domain_names,
    )

    # Overflow gate for collection resolution: only run the fuzzy ladder
    # (tiers 2/3/4) when the LLM was rendered a truncated collection list
    # for the chosen domain. With the full list visible the LLM has
    # already had the chance to dedup itself; fuzzy matching can only
    # second-guess that informed decision. A brand-new domain has no
    # existing collections at all → no overflow possible.
    chosen_domain_node = next(
        (d for d in existing_domains if d.name == decision.name), None
    )
    allow_fuzzy_collection = (
        chosen_domain_node is not None and chosen_domain_node.overflow > 0
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

        target_table = "papers" if kind is SourceKind.PAPER else "posts"
        target_status = (
            PaperStatus.CLASSIFIED if kind is SourceKind.PAPER else PostStatus.CLASSIFIED
        )

        conn.execute(
            "DELETE FROM topics WHERE target_kind = ? AND target_id = ?",
            (kind.value, source_id),
        )

        # Track every canonical this stage resolves so index_paper can
        # rebuild terms_fts for them. Under the synonym-index revert,
        # tier-1 / tier-5 hits leave no alias trail, so the deferred
        # rebuild queue is the only signal for unchanged collection /
        # topic canonicals.
        touched_term_ids: set[int] = set()

        resolved: list[ResolvedCollection] = []
        seen_raw_names: set[str] = set()
        seen_canonicals: set[str] = set()
        for i, pick in enumerate(output.collections):
            raw_key = pick.name.casefold()
            if raw_key in seen_raw_names:
                continue
            seen_raw_names.add(raw_key)

            coll_hit = resolve(
                conn,
                pick.name,
                domain=decision.name,
                term_type="collection",
                source_paper=paper_name,
                embedder=embedder,
                allow_fuzzy=allow_fuzzy_collection,
            )
            touched_term_ids.add(coll_hit.term_id)

            if coll_hit.canonical_name in seen_canonicals:
                _LOG.warning(
                    "classify paper_name=%s: collection pick #%d (%r) "
                    "canonicalized to %r — already present; dropping",
                    paper_name, i, pick.name, coll_hit.canonical_name,
                )
                continue
            seen_canonicals.add(coll_hit.canonical_name)

            conn.execute(
                """
                INSERT OR IGNORE INTO collection_definitions (domain, name, description)
                VALUES (?, ?, ?)
                """,
                (decision.name, coll_hit.canonical_name, pick.description),
            )
            resolved.append(
                ResolvedCollection(
                    name=coll_hit.canonical_name,
                    description=pick.description,
                )
            )

        primary_name = resolved[0].name
        any_new_collection = any(r.description is not None for r in resolved)

        # papers/posts UPDATE runs before the collections write so the
        # invariant trigger sees a non-NULL collection in the same
        # transaction, and so the collections intra-domain trigger reads
        # the new domain off the parent row.
        conn.execute(
            f"""
            UPDATE {target_table}
               SET domain = ?,
                   collection = ?,
                   needs_review = ?,
                   status = ?
             WHERE id = ?
            """,
            (
                decision.name,
                primary_name,
                int(decision.paper_needs_review or any_new_collection),
                target_status.value,
                source_id,
            ),
        )

        conn.execute(
            "DELETE FROM collections WHERE target_kind = ? AND target_id = ?",
            (kind.value, source_id),
        )
        conn.executemany(
            """
            INSERT INTO collections
                (target_kind, target_id, domain, collection, is_primary)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (kind.value, source_id, decision.name, r.name, int(i == 0))
                for i, r in enumerate(resolved)
            ],
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
                INSERT INTO topics (target_kind, target_id, domain, topic)
                VALUES (?, ?, ?, ?)
                """,
                (
                    kind.value, source_id,
                    decision.name, topic_hit.canonical_name,
                ),
            )
            inserted_topic_names.append(topic_hit.canonical_name)

        pending_fts_rebuilds(conn).update(touched_term_ids)

        # Re-classify deletes the paper's topics rows up front and
        # re-runs the LLM. Topic canonicals from the *previous* run that
        # the new run didn't re-bind are now orphaned in canonical_terms.
        # GC them here, after all new bindings are in place. A GC'd
        # canonical's deferred FTS rebuild becomes a no-op against an
        # absent row (harmless). Collections are curated categories and
        # survive even if the paper moves off — only humans delete them.
        gc_orphan_topic_canonicals(conn)

    needs_review = bool(decision.paper_needs_review or any_new_collection)
    _LOG.info(
        "classified source_id=%s paper_name=%s domain=%s primary=%s "
        "secondaries=%d topics=%d needs_review=%s",
        source_id, paper_name, decision.name, primary_name,
        len(resolved) - 1, len(inserted_topic_names), needs_review,
    )

    return ClassifyResult(
        paper_name=paper_name,
        domain=decision.name,
        collections=tuple(r.name for r in resolved),
        topics=tuple(inserted_topic_names),
        needs_review=needs_review,
        status=target_status.value,
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
    existing_domains: list[DomainNode],
) -> ClassificationOutput:
    """Translate the index-based LLM output to a name-based pipeline shape.

    Domain: ``-1`` → propose new; ``0..N-1`` → existing.
    Collections: list of picks. Each pick uses ``-1`` to propose a new
    collection name + description, or an index ``0..M-1`` into the
    chosen domain's collection list. Index 0 of the *list* is always the
    PRIMARY collection.

    Cross-field rules the schema enum cannot express are enforced here:

    - domain_index out of ``[-1, N-1]`` → :class:`ClassifyLLMError`
    - domain_index == -1 but new_domain_desc empty → raise
    - domain_index >= 0 but new_domain_desc non-empty → raise
      (LLM is writing a description for a domain it didn't propose)
    - len(collections) == 0 → raise (schema enforces minItems: 1, but
      we double-check for resilience against schema regressions)
    - per pick: index out of ``[-1, M-1]`` for the chosen domain → raise
    - per pick: index == -1 with empty new_name or new_desc → raise
    - per pick: index >= 0 with non-empty new_name or new_desc → raise
    - any pick.index >= 0 while domain_index == -1 → raise (new domain
      has no existing collections)
    """
    if raw.domain_index == -1:
        domain_name = raw.new_domain
        domain_is_new = True
        domain_description: str | None = raw.new_domain_desc.strip()
        domain_node: DomainNode | None = None
        if not domain_description:
            raise ClassifyLLMError(
                "LLM returned domain_index=-1 but "
                "new_domain_desc is empty; new domains "
                "must include a one-sentence description of the research area"
            )
    elif 0 <= raw.domain_index < len(existing_domains):
        domain_node = existing_domains[raw.domain_index]
        domain_name = domain_node.name
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

    if not raw.collections:
        raise ClassifyLLMError(
            "LLM returned an empty collections list; schema minItems was "
            "supposed to make this impossible"
        )

    resolved_picks = [
        _resolve_pick(pick, i, domain_name, domain_is_new, domain_node)
        for i, pick in enumerate(raw.collections)
    ]

    return ClassificationOutput(
        domain=domain_name,
        domain_is_new=domain_is_new,
        domain_description=domain_description,
        collections=resolved_picks,
        topics=raw.topics,
    )


def _resolve_pick(
    pick,
    i: int,
    domain_name: str,
    domain_is_new: bool,
    domain_node: DomainNode | None,
) -> ResolvedCollection:
    """Validate one CollectionPick and return its ResolvedCollection.

    Schema enum already constrains ``pick.index`` to ``[-1, M-1]``, so
    the only out-of-range case left to enforce here is ``index >= len``
    for the *chosen* domain (which the cross-domain enum cannot express).
    """
    if pick.index == -1:
        proposed = pick.new_name.strip()
        if not proposed:
            raise ClassifyLLMError(
                f"LLM collection pick #{i} has index=-1 but new_name is empty"
            )
        description = pick.new_desc.strip()
        if not description:
            raise ClassifyLLMError(
                f"LLM collection pick #{i} has index=-1 but new_desc is "
                f"empty; new collections must include a one-sentence description"
            )
        return ResolvedCollection(name=proposed, description=description)

    if domain_is_new or domain_node is None:
        raise ClassifyLLMError(
            f"LLM proposed new domain={domain_name!r} but pick #{i} has "
            f"index={pick.index}; new domains have no existing collections "
            f"— every pick's index must be -1"
        )
    domain_colls = domain_node.collections
    if pick.index >= len(domain_colls):
        raise ClassifyLLMError(
            f"LLM collection pick #{i} has index={pick.index} for "
            f"domain={domain_name!r}, which has {len(domain_colls)} "
            f"collection(s) — index out of range"
        )
    if pick.new_name.strip():
        raise ClassifyLLMError(
            f"LLM collection pick #{i} picked existing index={pick.index} "
            f"but also set new_name={pick.new_name!r}; new_name must be "
            f"empty when index >= 0"
        )
    if pick.new_desc.strip():
        raise ClassifyLLMError(
            f"LLM collection pick #{i} picked existing index={pick.index} "
            f"but also set new_desc={pick.new_desc!r}; new_desc must be "
            f"empty when index >= 0"
        )
    return ResolvedCollection(
        name=domain_colls[pick.index].name,
        description=None,
    )


# ---------------------------------------------------------------------------
# Paper content head-slice
# ---------------------------------------------------------------------------


def _head_slice_paper_content(
    *, markdown: str, abstract: str, max_chars: int = _PAPER_CONTENT_MAX_CHARS,
) -> str:
    """Return up to ``max_chars`` chars of source content for classification.

    The head of the markdown naturally contains title + abstract + start of
    introduction (papers) or title + lead + opening sections (posts) —
    everything the LLM needs to pick a domain/collection/topics. If markdown
    is missing (stale row, upstream conversion skipped), fall back to the
    abstract column alone.
    """
    stripped = markdown.strip()
    if stripped:
        return stripped[:max_chars]
    return abstract.strip()


# ---------------------------------------------------------------------------
# Domain handling
# ---------------------------------------------------------------------------


def _choose_domain(
    *,
    proposed: str,
    domain_is_new: bool,
    override: str | None,
    existing_domains: set[str],
) -> _DomainDecision:
    if override is not None:
        insert_new = override not in existing_domains
        # An operator-created brand-new domain has no human-authored
        # description (override path forces it NULL) and no review trail;
        # flag for review so it shows up in the same queue as LLM-proposed
        # new domains. Existing-domain overrides stay needs_review=False
        # because a human already curated that domain.
        return _DomainDecision(
            name=override,
            insert_new=insert_new,
            paper_needs_review=insert_new,
        )

    treat_as_new = domain_is_new or proposed not in existing_domains
    if treat_as_new:
        sanitized = sanitize_domain(proposed)
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

    if args.domain is not None:
        sanitized = sanitize_domain(args.domain)
        if not sanitized:
            parser.error(
                f"--domain={args.domain!r} sanitizes to empty string"
            )
        args.domain = sanitized

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
