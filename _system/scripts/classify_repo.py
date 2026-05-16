"""Single-LLM classification pass for one standalone code repository.

Mirrors :mod:`_system.scripts.classify_paper`. Operates on the README
plus optional GitHub metadata; writes ``repos.domain``,
``repos.collection`` (the denormalized primary pointer), polymorphic
``collections`` rows with ``target_kind='repo'``, and ``topics`` rows
with ``target_kind='repo'``. Repos without a usable README are marked
``ORPHANED`` and skip the LLM call entirely — they remain searchable by
name/path/file content.

Prompt assets live under ``_system/llm/prompts/classify_repo/`` —
deliberately separate from ``classify_paper`` because the language and
structural cues differ (README chrome / badges / install instructions
must not leak into topics) — but the response schema is identical to
the paper side: 1 PRIMARY collection (item 0) plus 0..3 SECONDARY
memberships within the same domain.
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
from _system.schemas.repo_metadata import RepoStatus, can_run_from
from _system.utils.source_resolution import SourceKind
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

_LOG = get_logger("scripts.classify_repo")

_README_CONTENT_MAX_CHARS = 10000
_COLLECTIONS_PER_DOMAIN_LIMIT = 30


class ClassifyRepoError(Exception):
    """Base class for classify_repo failures."""


class ClassifyRepoLLMError(ClassifyRepoError):
    """The LLM call failed (transport, auth, schema, or bad index)."""


class ClassifyRepoStateError(ClassifyRepoError):
    """`can_run_from` rejected the current status for CLASSIFIED."""


class ClassifyRepoDomainNameError(ClassifyRepoError):
    """Proposed domain name sanitizes to an empty string."""


class ClassifyRepoNotFound(ClassifyRepoError):
    """No repos row for the given repo_slug."""


class ClassifyRepoResult(NamedTuple):
    repo_slug: str
    status: str
    domain: str | None
    collection: str | None  # primary collection name (matches repos.collection scalar)
    collections: tuple[str, ...]  # primary first, then secondaries
    topics: tuple[str, ...]
    needs_review: bool


class _DomainDecision(NamedTuple):
    name: str
    insert_new: bool
    repo_needs_review: bool


def classify(
    *,
    repo_slug: str,
    conn: sqlite3.Connection,
    force: bool = False,
    domain_override: str | None = None,
    call_llm=None,
    embedder: Embedder | None = None,
) -> ClassifyRepoResult:
    """Run the LLM classification pass for one standalone repo.

    Paper-linked repos inherit their domain/collection from the paper —
    callers must NOT route them through here. ``ingest_repo_only`` only
    schedules CLASSIFY_REPO for the standalone path.
    """
    del force  # parity with other stage callers; idempotency is via DELETE+INSERT.

    row = conn.execute(
        """
        SELECT id, status, has_readme, paper_id, description
          FROM repos
         WHERE repo_slug = ?
        """,
        (repo_slug,),
    ).fetchone()
    if row is None:
        raise ClassifyRepoNotFound(
            f"repo_slug={repo_slug!r} not found in repos table"
        )
    repo_id, status_str, has_readme, paper_id, description = row

    if paper_id is not None:
        raise ClassifyRepoStateError(
            f"repo_slug={repo_slug!r} is paper-linked (paper_id={paper_id}); "
            "paper-linked repos inherit taxonomy and must not be classified directly"
        )

    try:
        current = RepoStatus(status_str) if status_str else None
    except ValueError as exc:
        raise ClassifyRepoStateError(
            f"repo_slug={repo_slug!r}: unrecognized status={status_str!r}"
        ) from exc
    if not can_run_from(current, RepoStatus.CLASSIFIED):
        extra = (
            " (orphaned/failed is terminal — re-fetch required)"
            if current in (RepoStatus.ORPHANED, RepoStatus.FAILED_REPO, RepoStatus.FAILED_RESOLVE)
            else ""
        )
        raise ClassifyRepoStateError(
            f"repo_slug={repo_slug!r}: cannot run CLASSIFIED from status="
            f"{status_str!r}{extra}"
        )

    # README-required gate. Repos that came back from fetch_repo with no
    # top-level README never get LLM-classified — there's no signal to
    # ground a domain/collection in. Mark them ORPHANED and move on; the
    # files remain searchable by path / file content.
    if not has_readme:
        with transaction(conn):
            conn.execute(
                "UPDATE repos SET status = ? WHERE id = ?",
                (RepoStatus.ORPHANED.value, repo_id),
            )
        _LOG.info(
            "repo %s has no README; marking ORPHANED (no domain/collection)", repo_slug
        )
        return ClassifyRepoResult(
            repo_slug=repo_slug,
            status=RepoStatus.ORPHANED.value,
            domain=None,
            collection=None,
            collections=(),
            topics=(),
            needs_review=False,
        )

    readme_row = conn.execute(
        "SELECT content FROM readmes_fts WHERE repo_id = ?",
        (repo_id,),
    ).fetchone()
    if readme_row is None:
        # has_readme=1 but no row → bookkeeping bug. Be loud.
        raise ClassifyRepoStateError(
            f"repo_slug={repo_slug!r} has_readme=1 but no readmes_fts row"
        )
    readme_content = (readme_row[0] or "").strip()
    if not readme_content:
        with transaction(conn):
            conn.execute(
                "UPDATE repos SET status = ? WHERE id = ?",
                (RepoStatus.ORPHANED.value, repo_id),
            )
        _LOG.info(
            "repo %s README is empty; marking ORPHANED", repo_slug
        )
        return ClassifyRepoResult(
            repo_slug=repo_slug,
            status=RepoStatus.ORPHANED.value,
            domain=None,
            collection=None,
            collections=(),
            topics=(),
            needs_review=False,
        )

    readme_truncated = readme_content[:_README_CONTENT_MAX_CHARS]

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

    metadata_block = _build_metadata_block(repo_slug=repo_slug, description=description)

    loaded = load_prompt(
        "classify_repo",
        md_context={
            "EXISTING_TAXONOMY": render_taxonomy_tree(
                existing_domains,
                style=TaxonomyTreeStyle.INDEX,
                overflow_message="(+ {n} more exist; feel free to propose new)",
            ),
            "README_CONTENT": readme_truncated,
            "METADATA_BLOCK": metadata_block,
        },
        schema_replacements={
            "DOMAIN_INDEX_ENUM": [-1, *range(len(existing_domains))],
            "COLLECTION_INDEX_ENUM": [-1, *range(max_collections)],
        },
    )

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

    # Repos use ``repo_slug`` as their ``source_paper`` token in
    # ``term_aliases`` — the synonym index is paper-vs-repo agnostic at
    # the schema level, so any unique identifier suffices. The repo is
    # the "source" the alias was first observed in.
    source_token = repo_slug

    # Overflow gate for collection resolution: only run the fuzzy ladder
    # (tiers 2/3/4) when the LLM was rendered a truncated collection list
    # for the chosen domain. See classify_paper.py for the rationale.
    chosen_domain_node = next(
        (d for d in existing_domains if d.name == decision.name), None
    )
    allow_fuzzy_collection = (
        chosen_domain_node is not None and chosen_domain_node.overflow > 0
    )

    with transaction(conn):
        if decision.insert_new:
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

        conn.execute(
            "DELETE FROM topics WHERE target_kind = ? AND target_id = ?",
            (SourceKind.REPO.value, repo_id),
        )

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
                source_paper=source_token,
                embedder=embedder,
                allow_fuzzy=allow_fuzzy_collection,
            )
            touched_term_ids.add(coll_hit.term_id)

            if coll_hit.canonical_name in seen_canonicals:
                _LOG.warning(
                    "classify repo_slug=%s: collection pick #%d (%r) "
                    "canonicalized to %r — already present; dropping",
                    repo_slug, i, pick.name, coll_hit.canonical_name,
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

        # repos UPDATE runs before the collections write so the
        # invariant trigger sees a non-NULL collection in the same
        # transaction, and so the collections intra-domain trigger reads
        # the new domain off the parent row.
        conn.execute(
            """
            UPDATE repos
               SET domain = ?,
                   collection = ?,
                   needs_review = ?,
                   status = ?
             WHERE id = ?
            """,
            (
                decision.name,
                primary_name,
                int(decision.repo_needs_review or any_new_collection),
                RepoStatus.CLASSIFIED.value,
                repo_id,
            ),
        )

        conn.execute(
            "DELETE FROM collections WHERE target_kind = ? AND target_id = ?",
            (SourceKind.REPO.value, repo_id),
        )
        conn.executemany(
            """
            INSERT INTO collections
                (target_kind, target_id, domain, collection, is_primary)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (SourceKind.REPO.value, repo_id, decision.name, r.name, int(i == 0))
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
                source_paper=source_token,
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
                    SourceKind.REPO.value, repo_id,
                    decision.name, topic_hit.canonical_name,
                ),
            )
            inserted_topic_names.append(topic_hit.canonical_name)

        pending_fts_rebuilds(conn).update(touched_term_ids)
        gc_orphan_topic_canonicals(conn)

    needs_review = bool(decision.repo_needs_review or any_new_collection)
    _LOG.info(
        "classified repo_id=%s repo_slug=%s domain=%s primary=%s "
        "secondaries=%d topics=%d needs_review=%s",
        repo_id, repo_slug, decision.name, primary_name,
        len(resolved) - 1, len(inserted_topic_names), needs_review,
    )

    return ClassifyRepoResult(
        repo_slug=repo_slug,
        status=RepoStatus.CLASSIFIED.value,
        domain=decision.name,
        collection=primary_name,
        collections=tuple(r.name for r in resolved),
        topics=tuple(inserted_topic_names),
        needs_review=needs_review,
    )


# ---------------------------------------------------------------------------
# Metadata block
# ---------------------------------------------------------------------------


def _build_metadata_block(*, repo_slug: str, description: str | None) -> str:
    """Render the optional ``<metadata>`` block for the user prompt.

    Returns an empty string when there's nothing useful to add (no
    description and only the slug we already pass elsewhere) — the
    prompt template renders the empty string as no block at all rather
    than an empty XML element, keeping the LLM from reading "no
    metadata" as a signal.
    """
    pieces: list[str] = []
    if description:
        cleaned = description.strip()
        if cleaned:
            pieces.append(f"<description>{cleaned}</description>")
    pieces.append(f"<slug>{repo_slug}</slug>")

    if not any(p.startswith("<description>") for p in pieces):
        return ""

    inner = "\n".join(pieces)
    return f"<metadata>\n{inner}\n</metadata>"


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_llm_default(system: str, user: str, schema: dict, response_model):
    return call_structured(
        system=system,
        user=user,
        schema=schema,
        response_model=response_model,
    )


# ---------------------------------------------------------------------------
# Resolve raw → output (mirrors classify_paper._resolve_raw)
# ---------------------------------------------------------------------------


def _resolve_raw(
    raw: ClassificationLLMOutput,
    existing_domains: list[DomainNode],
) -> ClassificationOutput:
    """Translate the index-based LLM output to a name-based pipeline shape.

    Mirrors :func:`classify_paper._resolve_raw`. Domain uses the
    index-replace pattern; ``collections`` is a 1..4-entry list with the
    first entry the PRIMARY membership.
    """
    if raw.domain_index == -1:
        domain_name = raw.new_domain
        domain_is_new = True
        domain_description: str | None = raw.new_domain_desc.strip()
        domain_node: DomainNode | None = None
        if not domain_description:
            raise ClassifyRepoLLMError(
                "LLM returned domain_index=-1 but new_domain_desc is empty; "
                "new domains must include a one-sentence description"
            )
    elif 0 <= raw.domain_index < len(existing_domains):
        domain_node = existing_domains[raw.domain_index]
        domain_name = domain_node.name
        domain_is_new = False
        if raw.new_domain_desc.strip():
            raise ClassifyRepoLLMError(
                f"LLM picked existing domain_index={raw.domain_index} but "
                f"also set new_domain_desc={raw.new_domain_desc!r}; "
                "description must be empty unless domain_index == -1"
            )
        domain_description = None
    else:
        raise ClassifyRepoLLMError(
            f"LLM returned domain_index={raw.domain_index} outside "
            f"[-1, {len(existing_domains) - 1}]"
        )

    if not raw.collections:
        raise ClassifyRepoLLMError(
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

    Mirrors :func:`classify_paper._resolve_pick`.
    """
    if pick.index == -1:
        proposed = pick.new_name.strip()
        if not proposed:
            raise ClassifyRepoLLMError(
                f"LLM collection pick #{i} has index=-1 but new_name is empty"
            )
        description = pick.new_desc.strip()
        if not description:
            raise ClassifyRepoLLMError(
                f"LLM collection pick #{i} has index=-1 but new_desc is "
                f"empty; new collections must include a one-sentence description"
            )
        return ResolvedCollection(name=proposed, description=description)

    if domain_is_new or domain_node is None:
        raise ClassifyRepoLLMError(
            f"LLM proposed new domain={domain_name!r} but pick #{i} has "
            f"index={pick.index}; new domains have no existing collections "
            f"— every pick's index must be -1"
        )
    domain_colls = domain_node.collections
    if pick.index >= len(domain_colls):
        raise ClassifyRepoLLMError(
            f"LLM collection pick #{i} has index={pick.index} for "
            f"domain={domain_name!r}, which has {len(domain_colls)} "
            f"collection(s) — index out of range"
        )
    if pick.new_name.strip():
        raise ClassifyRepoLLMError(
            f"LLM collection pick #{i} picked existing index={pick.index} "
            f"but also set new_name={pick.new_name!r}; new_name must be "
            f"empty when index >= 0"
        )
    if pick.new_desc.strip():
        raise ClassifyRepoLLMError(
            f"LLM collection pick #{i} picked existing index={pick.index} "
            f"but also set new_desc={pick.new_desc!r}; new_desc must be "
            f"empty when index >= 0"
        )
    return ResolvedCollection(
        name=domain_colls[pick.index].name,
        description=None,
    )


# ---------------------------------------------------------------------------
# Domain handling (mirrors classify_paper)
# ---------------------------------------------------------------------------


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
            repo_needs_review=False,
        )

    treat_as_new = domain_is_new or proposed not in existing_domains
    if treat_as_new:
        sanitized = sanitize_domain(proposed)
        if not sanitized:
            raise ClassifyRepoDomainNameError(
                f"proposed domain {proposed!r} sanitizes to empty string"
            )
        if sanitized in existing_domains:
            return _DomainDecision(sanitized, insert_new=False, repo_needs_review=False)
        return _DomainDecision(sanitized, insert_new=True, repo_needs_review=True)

    return _DomainDecision(proposed, insert_new=False, repo_needs_review=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify one standalone repo via the configured LLM provider."
    )
    parser.add_argument("--repo", required=True, help="repos.repo_slug")
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
            repo_slug=args.repo,
            conn=conn,
            force=args.force,
            domain_override=args.domain,
        )
    finally:
        conn.close()
    print(json.dumps(result._asdict()))


if __name__ == "__main__":
    _main()
