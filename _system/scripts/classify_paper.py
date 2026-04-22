"""Single-LLM classification pass for one paper.

This is the **only** subprocess/LLM call in the ingest pipeline. Every other
stage is deterministic code or local inference.

Shape of the call:

    subprocess.run(
        ["claude", "-p", "--bare", "--output-format", "json"],
        input=prompt_bytes,
        shell=False,
        timeout=180,
        check=False,
    )

The prompt is built from the arxiv-supplied abstract plus the paper's
introduction (located via the shared section splitter). Existing domains and
per-domain collection usage are included as context so the LLM prefers
reusing a canonical label over inventing a new one.

On success we resolve ``collection`` and every ``topic`` through the shared
5-tier term resolver (Section 4) and write the canonical names. On a proposed
new domain we sanitize the name and auto-insert it with ``needs_review=1`` on
the *paper* row (the ``domains`` table has no per-row review flag — only
``papers.needs_review`` exists in the schema, so that's where the signal
lives until a human reviews via ``search.py --needs-review``).
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Iterable, NamedTuple

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from _system.db.connection import get_conn, transaction
from _system.schemas.paper_metadata import PaperStatus, can_run_from
from _system.schemas.taxonomy import ClassificationOutput
from _system.resolution.resolver import resolve
from _system.resolution.embeddings import Embedder
from _system.utils.logging import get_logger
from _system.utils.sections import split_sections, strip_breadcrumb

_LOG = get_logger("scripts.classify_paper")

CLAUDE_ARGV: tuple[str, ...] = ("claude", "-p", "--bare", "--output-format", "json")

_SUBPROCESS_TIMEOUT_S = 180
_INTRO_MAX_CHARS = 8000
_COLLECTIONS_PER_DOMAIN_LIMIT = 30
_HEAD_BYTES_CAP = 2048
_DOMAIN_MAX_LEN = 32

_INTRO_TITLES: tuple[str, ...] = ("introduction", "overview", "background")

_WS_OR_SLASH_RE = re.compile(r"[\s/]+")
_DOMAIN_ALLOWED_RE = re.compile(r"[^a-z0-9_-]")

_AUTO_DOMAIN_DESCRIPTION = "(auto-created by classify_paper; review and edit)"


class ClassifyError(Exception):
    """Base class for classify_paper failures."""


class ClassifySubprocessError(ClassifyError):
    """`claude` CLI exited non-zero or timed out."""


class ClassifyEnvelopeError(ClassifyError):
    """stdout JSON is missing the `structured_output` key."""


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
    """Pure-function output of `_choose_domain`; no DB side effects.

    ``insert_new`` is True when the decision requires inserting a new row
    into ``domains`` — caller writes it inside the transaction so a later
    failure rolls it back together with the paper_topics / papers writes.
    """

    name: str
    insert_new: bool
    paper_needs_review: bool


def classify(
    *,
    paper_name: str,
    conn: sqlite3.Connection,
    force: bool = False,
    domain_override: str | None = None,
    run_subprocess=None,
    embedder: Embedder | None = None,
) -> ClassifyResult:
    """Run the single-LLM classification pass for one paper.

    Pre-conditions: papers row exists with status >= CONVERTED (and not
    FAILED_HTML). Post-conditions: papers.{domain, collection, needs_review,
    status=CLASSIFIED} updated; paper_topics rebuilt; new domain inserted if
    needed. Raises on all failure modes.

    ``force`` is accepted for orchestrator parity but does not bypass the
    ``can_run_from`` guard — the orchestrator handles force by cascading back
    to earlier stages, which lowers the status before classify ever runs.

    ``run_subprocess`` is a test seam. Production callers leave it as None,
    and the module-level retrying subprocess runner is used.
    """
    del force  # see docstring

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

    intro_text = _extract_intro(markdown or "", paper_name=paper_name)
    del markdown

    existing_domains = _load_domains(conn)
    existing_domain_names = {d[0] for d in existing_domains}
    collections_by_domain = _load_collections_by_domain(conn)

    prompt = _build_prompt(
        abstract=abstract or "",
        intro_text=intro_text,
        domains=existing_domains,
        collections_by_domain=collections_by_domain,
    )

    # Fresh terms miss tiers 1-4 and need an Embedder for tier 5.
    if embedder is None:
        embedder = Embedder()

    runner = run_subprocess or _run_claude_cli
    envelope = runner(prompt)

    structured = envelope.get("structured_output")
    if structured is None:
        raise ClassifyEnvelopeError(
            f"paper_name={paper_name!r}: envelope missing 'structured_output'; "
            f"envelope head={str(envelope)[:_HEAD_BYTES_CAP]!r}"
        )
    output = ClassificationOutput.model_validate(structured)

    decision = _choose_domain(
        proposed=output.domain,
        domain_is_new=output.domain_is_new,
        override=domain_override,
        existing_domains=existing_domain_names,
    )

    with transaction(conn):
        if decision.insert_new:
            conn.execute(
                """
                INSERT OR IGNORE INTO domains (name, description)
                VALUES (?, ?)
                """,
                (decision.name, _AUTO_DOMAIN_DESCRIPTION),
            )

        conn.execute("DELETE FROM paper_topics WHERE paper_id = ?", (paper_id,))

        collection_hit = resolve(
            conn,
            output.collection,
            domain=decision.name,
            term_type="collection",
            source_paper=paper_name,
            embedder=embedder,
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
# Intro extraction
# ---------------------------------------------------------------------------


def _extract_intro(markdown: str, *, paper_name: str) -> str:
    """Pick the best introduction-ish section from ``markdown``.

    Preference order: "Introduction" > "Overview" > "Background" > first
    level-1 section. When the markdown has no headers at all, returns an
    empty string and logs a warning — the caller still builds a prompt from
    the abstract alone.
    """
    if not markdown.strip():
        _LOG.warning(
            "paper_name=%s has empty markdown; falling back to abstract-only prompt",
            paper_name,
        )
        return ""

    chunks = split_sections(markdown)
    if not chunks:
        _LOG.warning(
            "paper_name=%s markdown has no '#' headers; falling back to "
            "abstract-only prompt",
            paper_name,
        )
        return ""

    by_title: dict[str, str] = {}
    first_level_one_body: str | None = None
    for chunk in chunks:
        title_key = chunk.title.strip().lower()
        if title_key in _INTRO_TITLES and title_key not in by_title:
            by_title[title_key] = chunk.body
        if first_level_one_body is None and chunk.level == 1:
            first_level_one_body = chunk.body

    for title in _INTRO_TITLES:
        body = by_title.get(title)
        if body is not None:
            return strip_breadcrumb(body)[:_INTRO_MAX_CHARS]

    if first_level_one_body is not None:
        return strip_breadcrumb(first_level_one_body)[:_INTRO_MAX_CHARS]

    _LOG.warning(
        "paper_name=%s markdown has headers but no level-1 or intro-like "
        "section; falling back to abstract-only prompt",
        paper_name,
    )
    return ""


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
) -> dict[str, list[str]]:
    """Per-domain collections ordered by descending usage count.

    Only collections with at least one paper appear. Domains with *no* used
    collections are simply absent from the dict — the prompt builder handles
    that gracefully.
    """
    rows = conn.execute(
        """
        SELECT domain, collection, COUNT(*) AS c
          FROM papers
         WHERE collection IS NOT NULL AND domain IS NOT NULL
         GROUP BY domain, collection
         ORDER BY domain, c DESC, collection
        """
    ).fetchall()
    result: dict[str, list[str]] = {}
    for domain, collection, _count in rows:
        result.setdefault(domain, []).append(collection)
    return result


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _build_prompt(
    *,
    abstract: str,
    intro_text: str,
    domains: list[tuple[str, str | None]],
    collections_by_domain: dict[str, list[str]],
) -> str:
    """Assemble the classification prompt.

    The JSON schema stub mirrors ``ClassificationOutput`` exactly — any drift
    here risks validation errors that a retry cannot fix.
    """
    schema_block = (
        "Return JSON matching this schema:\n"
        "{\n"
        '  "domain": "...",\n'
        '  "domain_is_new": true|false,\n'
        '  "collection": "...",\n'
        '  "topics": ["...", "..."]\n'
        "}"
    )

    domain_lines: list[str] = []
    if domains:
        for name, description in domains:
            if description:
                domain_lines.append(f"- {name}: {description}")
            else:
                domain_lines.append(f"- {name}")
    else:
        domain_lines.append("(none yet — propose a new domain)")
    domain_block = "Existing domains:\n" + "\n".join(domain_lines)

    collection_lines: list[str] = []
    for name, _desc in domains:
        colls = collections_by_domain.get(name, [])
        if not colls:
            collection_lines.append(f"{name}: []")
            continue
        if len(colls) > _COLLECTIONS_PER_DOMAIN_LIMIT:
            shown = colls[:_COLLECTIONS_PER_DOMAIN_LIMIT]
            more = len(colls) - _COLLECTIONS_PER_DOMAIN_LIMIT
            collection_lines.append(
                f"{name}: [{', '.join(shown)}] "
                f"(+ {more} more; feel free to propose new)"
            )
        else:
            collection_lines.append(f"{name}: [{', '.join(colls)}]")
    collection_block = "Existing collections within each domain:\n" + "\n".join(
        collection_lines
    )

    intro_section = (
        f"Paper introduction:\n{intro_text}"
        if intro_text
        else "Paper introduction:\n(not available; classify from the abstract alone)"
    )

    return (
        "You are a research librarian classifying an arxiv paper. "
        f"{schema_block}\n\n"
        f"{domain_block}\n\n"
        f"{collection_block}\n\n"
        f"Paper abstract:\n{abstract}\n\n"
        f"{intro_section}\n\n"
        "Classify this paper. If no existing domain fits well, propose a "
        "new domain and set domain_is_new=true."
    )


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(
        (
            ClassifySubprocessError,
            json.JSONDecodeError,
            subprocess.TimeoutExpired,
        )
    ),
    reraise=True,
)
def _run_claude_cli(prompt: str) -> dict:
    """Run the `claude` CLI once; return the parsed JSON envelope.

    Tenacity retries on subprocess errors (non-zero exit / process-level
    failures) and JSON parse errors (genuinely flaky CLI output).
    ``ClassifyEnvelopeError`` and ``ValidationError`` are *not* retried —
    a structurally broken envelope is deterministic, and retrying burns
    real LLM cost for a response that will fail identically next call.
    """
    try:
        result = subprocess.run(
            list(CLAUDE_ARGV),
            input=prompt.encode("utf-8"),
            capture_output=True,
            shell=False,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ClassifySubprocessError(
            "`claude` CLI not found on PATH. validate_models.check_models() "
            "should have caught this — is ingest.py invoking it first?"
        ) from exc

    if result.returncode != 0:
        raise ClassifySubprocessError(
            f"claude CLI exited {result.returncode}; "
            f"stderr head={result.stderr[:_HEAD_BYTES_CAP]!r}"
        )

    return json.loads(result.stdout.decode("utf-8", errors="replace"))


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
    """Decide the final domain name; side-effect-free.

    The caller writes any ``INSERT INTO domains`` inside its transaction so
    that a downstream failure rolls back the domain row together with the
    paper writes.
    """
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
            # Already present — LLM was wrong to call it new, but it's a
            # legit reuse. Don't flag the paper for review.
            return _DomainDecision(sanitized, insert_new=False, paper_needs_review=False)
        return _DomainDecision(sanitized, insert_new=True, paper_needs_review=True)

    return _DomainDecision(proposed, insert_new=False, paper_needs_review=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify one paper via the `claude` CLI."
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
