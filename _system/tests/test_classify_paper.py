"""Unit tests for _system/scripts/classify_paper.py.

No real LLM calls — every classify_paper invocation injects a ``call_llm``
stub that returns a pre-built ``ClassificationLLMOutput``. No network, no
ML weight loading (the resolver is called with a fake embedder; tier 4
is not needed for these tests).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.schemas.paper_metadata import PaperStatus
from _system.schemas.taxonomy import ClassificationLLMOutput
from _system.scripts import classify_paper as cp
from _system.scripts.classify_paper import (
    ClassifyDomainNameError,
    ClassifyLLMError,
    ClassifyPaperNotFound,
    ClassifyStateError,
    _COLLECTIONS_PER_DOMAIN_LIMIT,
    _head_slice_paper_content,
    classify,
)
from _system.utils.slug import sanitize_domain


FIXTURES = Path(__file__).parent / "fixtures" / "classify"


# ---------------------------------------------------------------------------
# Fake embedder — fresh topics/collections land in tier 5, which requires
# an embedder. Classify passes `embedder=` straight to the resolver; tests
# inject this deterministic fake so no sentence-transformers model loads.
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """384-dim deterministic vectors keyed off input text."""

    def __init__(self) -> None:
        self.embed_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        v = [0.0] * 384
        v[hash(text) % 384] = 1.0
        return v

    def embed_batch(self, texts):  # pragma: no cover
        return [self.embed(t) for t in texts]


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


@pytest.fixture(autouse=True)
def _patch_embedder_class(monkeypatch):
    monkeypatch.setattr(cp, "Embedder", _FakeEmbedder)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

_DEFAULT_MARKDOWN = (
    "## Abstract\n"
    "\n"
    "We propose hierarchical tree retrieval for long-context QA.\n"
    "\n"
    "## Introduction\n"
    "\n"
    "This paper presents a hierarchical retrieval approach.\n"
    "\n"
    "## Method\n"
    "\n"
    "We build a tree.\n"
)


def _seed_domain(
    conn: sqlite3.Connection,
    name: str = "rag",
    description: str = "retrieval-augmented generation",
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO domains (name, description) VALUES (?, ?)",
        (name, description),
    )


def _seed_paper(
    conn: sqlite3.Connection,
    *,
    paper_name: str = "paper_name_2024",
    arxiv_id: str = "2401.00001",
    status: str = PaperStatus.CONVERTED.value,
    abstract: str = "We propose hierarchical tree retrieval for long-context QA.",
    markdown: str | None = _DEFAULT_MARKDOWN,
    domain: str | None = None,
    collection: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, html_source, ingested_at, status, markdown,
            domain, collection, needs_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id,
            paper_name,
            "A Title",
            '["A. Author"]',
            "2024-01-01",
            abstract,
            f"https://arxiv.org/pdf/{arxiv_id}",
            "arxiv",
            "2024-01-01T00:00:00+00:00",
            status,
            markdown,
            domain,
            collection,
            0,
        ),
    )
    # Mirror the invariant production maintains: a paper with a collection
    # means that (domain, collection) is registered. Keeps seeded fixtures
    # in sync with what classify() builds up on its own.
    if domain is not None and collection is not None:
        conn.execute(
            "INSERT OR IGNORE INTO collections (domain, name, description) "
            "VALUES (?, ?, NULL)",
            (domain, collection),
        )
    return cur.lastrowid


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _fake_runner(fixture: str = "classification_rag_hierarchical.json"):
    """Return a ``call_llm`` stub that yields the parsed fixture payload."""
    payload = _load_fixture(fixture)

    def _runner(system: str, user: str, schema: dict, response_model):
        return response_model.model_validate(payload)

    return _runner


def _runner_from_dict(payload: dict):
    """Return a ``call_llm`` stub that yields the given dict as a model."""

    def _runner(system: str, user: str, schema: dict, response_model):
        return response_model.model_validate(payload)

    return _runner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_with_domain(conn: sqlite3.Connection):
    _seed_domain(conn)
    return conn


@pytest.fixture
def seeded(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    return tmp_db_with_domain


# ===========================================================================
# Paper content head-slice
# ===========================================================================


def test_head_slice_returns_markdown_when_present():
    md = "## Abstract\n\nsome text\n\n## Introduction\n\nbody\n"
    out = _head_slice_paper_content(markdown=md, abstract="ignored")
    assert out == md.strip()


def test_head_slice_caps_at_8000_chars():
    big = "x" * 20000
    out = _head_slice_paper_content(markdown=big, abstract="")
    assert len(out) == 8000


def test_head_slice_falls_back_to_abstract_when_markdown_empty():
    out = _head_slice_paper_content(markdown="", abstract="abs only")
    assert out == "abs only"


def test_head_slice_falls_back_to_abstract_when_markdown_whitespace():
    out = _head_slice_paper_content(markdown="   \n\n  ", abstract="abs only")
    assert out == "abs only"


def test_head_slice_returns_empty_when_both_missing():
    out = _head_slice_paper_content(markdown="", abstract="")
    assert out == ""


# ===========================================================================
# Call seam — contract between classify() and call_structured
# ===========================================================================


def test_classify_passes_resolved_prompt_and_schema_to_call_llm(seeded):
    captured: dict[str, object] = {}

    def _runner(system: str, user: str, schema: dict, response_model):
        captured["system"] = system
        captured["user"] = user
        captured["schema"] = schema
        captured["response_model"] = response_model
        return response_model(
            domain_index=0,
            new_domain="",
            new_domain_desc="",
            collection_index=-1,
            new_collection="hierarchical indexing",
            new_collection_desc="Retrieval methods that build hierarchical indices over long documents.",
            topics=["tree retrieval"],
        )

    classify(paper_name="paper_name_2024", conn=seeded, call_llm=_runner)

    assert "research librarian" in captured["system"]
    assert "<paper_content>" in captured["user"]
    assert "0. rag" in captured["user"]
    assert "hierarchical tree retrieval" in captured["user"]
    schema = captured["schema"]
    assert isinstance(schema, dict)
    assert schema["name"] == "classify_paper"
    # DOMAIN_INDEX_ENUM sentinel must have been replaced with a real list.
    enum_val = schema["schema"]["properties"]["domain_index"]["enum"]
    assert enum_val == [-1, 0]
    # COLLECTION_INDEX_ENUM — rag has no collections yet, so only -1 is valid.
    coll_enum = schema["schema"]["properties"]["collection_index"]["enum"]
    assert coll_enum == [-1]
    assert captured["response_model"] is ClassificationLLMOutput


def test_classify_writes_paper_state_from_llm_output(seeded):
    classify(
        paper_name="paper_name_2024",
        conn=seeded,
        call_llm=_fake_runner(),
    )
    row = seeded.execute(
        "SELECT domain, collection, status, needs_review FROM papers "
        "WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "rag"
    assert row[1] == "hierarchical indexing"
    assert row[2] == PaperStatus.CLASSIFIED.value
    assert row[3] == 0


def test_domain_index_out_of_range_raises_llm_error(seeded):
    def _runner(system, user, schema, response_model):
        return response_model(
            domain_index=99,  # outside [-1, 0]
            new_domain="",
            new_domain_desc="",
            collection_index=-1,
            new_collection="c",
            new_collection_desc="",
            topics=["t"],
        )

    with pytest.raises(ClassifyLLMError):
        classify(paper_name="paper_name_2024", conn=seeded, call_llm=_runner)


def test_collection_index_minus_one_with_empty_proposal_raises(seeded):
    def _runner(system, user, schema, response_model):
        return response_model(
            domain_index=0,
            new_domain="",
            new_domain_desc="",
            collection_index=-1,
            new_collection="",  # invalid: both signal "new" but name is empty
            new_collection_desc="A dummy cluster.",
            topics=["t"],
        )

    with pytest.raises(ClassifyLLMError) as exc_info:
        classify(paper_name="paper_name_2024", conn=seeded, call_llm=_runner)
    assert "new_collection" in str(exc_info.value)


def test_new_domain_with_empty_description_raises(seeded):
    def _runner(system, user, schema, response_model):
        return response_model(
            domain_index=-1,
            new_domain="new_dom",
            new_domain_desc="   ",  # whitespace-only → empty
            collection_index=-1,
            new_collection="c",
            new_collection_desc="A dummy cluster.",
            topics=["t"],
        )

    with pytest.raises(ClassifyLLMError) as exc_info:
        classify(paper_name="paper_name_2024", conn=seeded, call_llm=_runner)
    assert "new_domain_desc" in str(exc_info.value)


def test_existing_domain_with_description_raises(seeded):
    def _runner(system, user, schema, response_model):
        return response_model(
            domain_index=0,
            new_domain="",
            new_domain_desc="should not be set",
            collection_index=-1,
            new_collection="c",
            new_collection_desc="A dummy cluster.",
            topics=["t"],
        )

    with pytest.raises(ClassifyLLMError) as exc_info:
        classify(paper_name="paper_name_2024", conn=seeded, call_llm=_runner)
    assert "new_domain_desc" in str(exc_info.value)


def test_new_collection_with_empty_description_raises(seeded):
    def _runner(system, user, schema, response_model):
        return response_model(
            domain_index=0,
            new_domain="",
            new_domain_desc="",
            collection_index=-1,
            new_collection="some new cluster",
            new_collection_desc="   ",  # whitespace-only → empty
            topics=["t"],
        )

    with pytest.raises(ClassifyLLMError) as exc_info:
        classify(paper_name="paper_name_2024", conn=seeded, call_llm=_runner)
    assert "new_collection_desc" in str(exc_info.value)


def test_existing_collection_with_description_raises(tmp_db_with_domain):
    _seed_paper(
        tmp_db_with_domain,
        paper_name="earlier_2024",
        arxiv_id="2400.00000",
        status=PaperStatus.CLASSIFIED.value,
        domain="rag",
        collection="hybrid search",
    )
    _seed_paper(tmp_db_with_domain)  # paper under test

    def _runner(system, user, schema, response_model):
        return response_model(
            domain_index=0,
            new_domain="",
            new_domain_desc="",
            collection_index=0,  # existing "hybrid search"
            new_collection="",
            new_collection_desc="should not be set",
            topics=["t"],
        )

    with pytest.raises(ClassifyLLMError) as exc_info:
        classify(
            paper_name="paper_name_2024",
            conn=tmp_db_with_domain,
            call_llm=_runner,
        )
    assert "new_collection_desc" in str(exc_info.value)


def test_new_domain_with_existing_collection_index_raises(seeded):
    def _runner(system, user, schema, response_model):
        return response_model(
            domain_index=-1,
            new_domain="new_dom",
            new_domain_desc="A new research area about something.",
            collection_index=0,  # invalid: new domain has no collections
            new_collection="",
            new_collection_desc="",
            topics=["t"],
        )

    with pytest.raises(ClassifyLLMError) as exc_info:
        classify(paper_name="paper_name_2024", conn=seeded, call_llm=_runner)
    assert "new" in str(exc_info.value).lower()
    assert "collection_index" in str(exc_info.value)


def test_existing_collection_index_resolves_to_name(tmp_db_with_domain):
    # Seed a paper with a collection so the 'rag' domain has collection 0.
    _seed_paper(
        tmp_db_with_domain,
        paper_name="earlier_2024",
        arxiv_id="2400.00000",
        status=PaperStatus.CLASSIFIED.value,
        domain="rag",
        collection="hybrid search",
    )
    _seed_paper(tmp_db_with_domain)  # paper under test

    def _runner(system, user, schema, response_model):
        return response_model(
            domain_index=0,
            new_domain="",
            new_domain_desc="",
            collection_index=0,  # → "hybrid search" (only collection under rag)
            new_collection="",
            new_collection_desc="",
            topics=["tree retrieval"],
        )

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_runner,
    )
    row = tmp_db_with_domain.execute(
        "SELECT collection FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "hybrid search"


def test_collection_index_out_of_range_for_chosen_domain_raises(tmp_db_with_domain):
    # rag has zero collections → any non-negative index is out of range.
    _seed_paper(tmp_db_with_domain)

    def _runner(system, user, schema, response_model):
        return response_model(
            domain_index=0,
            new_domain="",
            new_domain_desc="",
            collection_index=1,
            new_collection="",
            new_collection_desc="",
            topics=["t"],
        )

    with pytest.raises(ClassifyLLMError) as exc_info:
        classify(
            paper_name="paper_name_2024",
            conn=tmp_db_with_domain,
            call_llm=_runner,
        )
    assert "out of range" in str(exc_info.value)


# ===========================================================================
# Status / resume guard
# ===========================================================================


def test_status_fetched_blocks_classify(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain, status=PaperStatus.FETCHED.value)
    with pytest.raises(ClassifyStateError):
        classify(
            paper_name="paper_name_2024",
            conn=tmp_db_with_domain,
            call_llm=_fake_runner(),
        )


def test_status_failed_html_blocks_classify(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain, status=PaperStatus.FAILED_HTML.value)
    with pytest.raises(ClassifyStateError) as exc_info:
        classify(
            paper_name="paper_name_2024",
            conn=tmp_db_with_domain,
            call_llm=_fake_runner(),
        )
    assert "failed_html" in str(exc_info.value).lower()


def test_status_converted_proceeds(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain, status=PaperStatus.CONVERTED.value)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner(),
    )


def test_status_classified_allows_rerun(tmp_db_with_domain):
    _seed_paper(
        tmp_db_with_domain,
        status=PaperStatus.CLASSIFIED.value,
        domain="rag",
        collection="hierarchical indexing",
    )
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner(),
    )


def test_paper_not_found_raises(tmp_db_with_domain):
    with pytest.raises(ClassifyPaperNotFound):
        classify(
            paper_name="no_such_paper",
            conn=tmp_db_with_domain,
            call_llm=_fake_runner(),
        )


# ===========================================================================
# Rerun semantics
# ===========================================================================


def test_rerun_deletes_existing_topics_before_insert(tmp_db_with_domain):
    paper_id = _seed_paper(tmp_db_with_domain)
    tmp_db_with_domain.execute(
        "INSERT INTO topics (target_kind, target_id, domain, topic) VALUES ('paper', ?, ?, ?)",
        (paper_id, "rag", "stale topic"),
    )

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner(),
    )
    rows = tmp_db_with_domain.execute(
        "SELECT topic FROM topics WHERE target_kind='paper' AND target_id = ? ORDER BY topic",
        (paper_id,),
    ).fetchall()
    topics = [r[0] for r in rows]
    assert "stale topic" not in topics
    assert len(topics) == 2


class _OrthogonalEmbedder:
    """Per-unique-string orthogonal vectors — guarantees tier-4 cannot
    spuriously merge two distinct strings (cosine = 0). Avoids the
    FakeEmbedder's per-process-randomized ``hash() % 384`` collision risk
    that shows up in tests that exercise the resolver across runs."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}

    def _vec_for(self, text: str) -> list[float]:
        idx = self._index.setdefault(text, len(self._index))
        v = [0.0] * 384
        v[idx % 384] = 1.0
        return v

    def embed(self, text: str) -> list[float]:
        return self._vec_for(text)

    def embed_batch(self, texts):  # pragma: no cover
        return [self.embed(t) for t in texts]


def test_rerun_gcs_orphan_topic_canonicals_from_prior_run(tmp_db_with_domain):
    """Re-classify wipes the paper's prior `topics` rows up front.
    If the second LLM run emits different topic phrasings, the first
    run's topic canonicals are orphaned in `canonical_terms`. The
    end-of-transaction GC must remove them once the new bindings are in
    place. Bound canonicals from a *different* paper survive (they still
    have a topics referent).
    """
    _seed_paper(tmp_db_with_domain)
    embedder = _OrthogonalEmbedder()

    # First run: topics = ["apple zoology", "zebra fishery"].
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_runner_from_dict({
            "domain_index": 0,
            "new_domain": "",
            "new_domain_desc": "",
            "collection_index": -1,
            "new_collection": "first cluster",
            "new_collection_desc": "Initial classification cluster.",
            "topics": ["apple zoology", "zebra fishery"],
        }),
        embedder=embedder,
    )

    # Seed an unrelated paper that pins a distinct topic canonical so we
    # can show GC is precise — the unrelated topic must survive.
    other_paper_id = _seed_paper(
        tmp_db_with_domain,
        paper_name="other_2024",
        arxiv_id="2400.99999",
        domain="rag",
        collection="first cluster",
    )
    tmp_db_with_domain.execute(
        "INSERT INTO topics (target_kind, target_id, domain, topic) VALUES ('paper', ?, ?, ?)",
        (other_paper_id, "rag", "unrelated cantaloupe"),
    )
    tmp_db_with_domain.execute(
        """
        INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES (?, 'topic', '', ?, ?)
        """,
        ("rag", "unrelated cantaloupe", "other_2024"),
    )

    # Sanity: run-1 canonicals exist before run-2.
    pre_topics = {
        r[0] for r in tmp_db_with_domain.execute(
            "SELECT canonical_name FROM canonical_terms WHERE term_type = 'topic'"
        )
    }
    assert "apple zoology" in pre_topics
    assert "zebra fishery" in pre_topics
    assert "unrelated cantaloupe" in pre_topics

    # Second run: completely different topic phrasings.
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_runner_from_dict({
            "domain_index": 0,
            "new_domain": "",
            "new_domain_desc": "",
            "collection_index": 0,  # existing "first cluster"
            "new_collection": "",
            "new_collection_desc": "",
            "topics": ["foobar widget", "qux baz silo"],
        }),
        embedder=embedder,
    )

    post_topics = {
        r[0] for r in tmp_db_with_domain.execute(
            "SELECT canonical_name FROM canonical_terms WHERE term_type = 'topic'"
        )
    }
    # Run-1 orphans gone.
    assert "apple zoology" not in post_topics
    assert "zebra fishery" not in post_topics
    # Run-2 canonicals present.
    assert "foobar widget" in post_topics
    assert "qux baz silo" in post_topics
    # Bound canonical from the other paper survives.
    assert "unrelated cantaloupe" in post_topics


def test_second_run_idempotent_topics_set(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner(),
    )
    first_topics = tmp_db_with_domain.execute(
        "SELECT topic FROM topics ORDER BY topic"
    ).fetchall()

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner(),
    )
    second_topics = tmp_db_with_domain.execute(
        "SELECT topic FROM topics ORDER BY topic"
    ).fetchall()
    assert first_topics == second_topics


# ===========================================================================
# Domain handling
# ===========================================================================


def test_sanitize_domain_lowercases_and_replaces_ws():
    assert sanitize_domain("Multi-Agent Systems") == "multi-agent_systems"


def test_sanitize_domain_strips_illegal_chars():
    assert sanitize_domain("Multi-Agent Systems!!") == "multi-agent_systems"


def test_sanitize_domain_truncates_at_max_len():
    from _system.utils.slug import DOMAIN_MAX_LEN
    long = "a" * (DOMAIN_MAX_LEN * 4)
    out = sanitize_domain(long)
    assert len(out) == DOMAIN_MAX_LEN


def test_sanitize_domain_empty_input_returns_empty():
    assert sanitize_domain("   ") == ""
    assert sanitize_domain("!!!") == ""


def test_new_domain_inserts_into_domains_and_sets_paper_needs_review(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner("classification_new_domain.json"),
    )
    row = tmp_db_with_domain.execute(
        "SELECT domain, needs_review FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "multi-agent_systems"
    assert row[1] == 1

    domain_row = tmp_db_with_domain.execute(
        "SELECT name, description FROM domains WHERE name = ?",
        ("multi-agent_systems",),
    ).fetchone()
    assert domain_row is not None
    # Description is the one the LLM supplied, not a canned sentinel.
    assert domain_row[1] == (
        "Research on coordinating multiple autonomous agents to plan, "
        "reason, and act on shared tasks."
    )

    # Collection description also lands in the first-class collections table.
    coll_row = tmp_db_with_domain.execute(
        "SELECT domain, name, description FROM collections "
        " WHERE domain = ? AND name = ?",
        ("multi-agent_systems", "orchestration patterns"),
    ).fetchone()
    assert coll_row is not None
    assert coll_row[2] == (
        "Design patterns for dividing work between a planner agent "
        "and one or more executor agents."
    )


def test_existing_collection_description_is_not_overwritten(tmp_db_with_domain):
    # Seed a collection with a curated description; the LLM then proposes
    # a new collection whose name collides with it (after resolver
    # canonicalization would, in principle, pick up this name). The
    # INSERT OR IGNORE must leave the curated description intact.
    tmp_db_with_domain.execute(
        "INSERT INTO collections (domain, name, description) VALUES (?, ?, ?)",
        ("rag", "hierarchical indexing", "curated by a human"),
    )
    _seed_paper(tmp_db_with_domain)

    # Default fixture proposes "hierarchical indexing" with a different
    # description; resolver will return the same canonical name.
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner(),
    )
    row = tmp_db_with_domain.execute(
        "SELECT description FROM collections "
        " WHERE domain = ? AND name = ?",
        ("rag", "hierarchical indexing"),
    ).fetchone()
    assert row is not None
    assert row[0] == "curated by a human"


def test_llm_returns_dirty_domain_name_gets_sanitized(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner("classification_bad_name.json"),
    )
    row = tmp_db_with_domain.execute(
        "SELECT domain FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "multi-agent_systems"


def test_proposed_domain_sanitizing_to_empty_raises(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    runner = _runner_from_dict({
        "domain_index": -1,
        "new_domain": "!!! ???",
        "new_domain_desc": "A dummy area.",
        "collection_index": -1,
        "new_collection": "c",
        "new_collection_desc": "A dummy cluster.",
        "topics": ["t1"],
    })

    with pytest.raises(ClassifyDomainNameError):
        classify(
            paper_name="paper_name_2024",
            conn=tmp_db_with_domain,
            call_llm=runner,
        )


def test_domain_override_bypasses_llm_choice_and_does_not_flag_review(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner("classification_new_domain.json"),
        domain_override="rag",
    )
    row = tmp_db_with_domain.execute(
        "SELECT domain, needs_review FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "rag"
    assert row[1] == 0


def test_domain_override_inserts_new_domain_needs_review_false(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner("classification_new_domain.json"),
        domain_override="theorem_proving",
    )
    row = tmp_db_with_domain.execute(
        "SELECT domain, needs_review FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "theorem_proving"
    assert row[1] == 0

    # Override forces a name the LLM didn't propose — we must not attach
    # the LLM's description (which was about the LLM's proposed domain).
    domain_row = tmp_db_with_domain.execute(
        "SELECT description FROM domains WHERE name = 'theorem_proving'"
    ).fetchone()
    assert domain_row is not None
    assert domain_row[0] is None


# ===========================================================================
# Resolver wiring
# ===========================================================================


def test_collection_and_topics_resolved_into_canonical_terms(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner(),
    )
    term_rows = tmp_db_with_domain.execute(
        """
        SELECT domain, term_type, canonical_name FROM canonical_terms
         ORDER BY term_type, canonical_name
        """
    ).fetchall()
    assert ("rag", "collection") in [(d, tt) for d, tt, _ in term_rows]
    assert ("rag", "topic") in [(d, tt) for d, tt, _ in term_rows]


def test_papers_collection_is_canonical_name_not_raw_llm(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    tmp_db_with_domain.execute(
        """
        INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES ('rag', 'collection', '', 'hierarchical indexing', 'seed')
        """
    )

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner(),
    )
    collection = tmp_db_with_domain.execute(
        "SELECT collection FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()[0]
    assert collection == "hierarchical indexing"


def test_topics_uses_resolver_returned_canonical_names(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner(),
    )
    topics = {
        r[0]
        for r in tmp_db_with_domain.execute(
            "SELECT topic FROM topics WHERE domain = 'rag'"
        )
    }
    assert topics == {"tree retrieval", "long-context qa"}


def test_duplicate_topics_dedupe_by_term_id(tmp_db_with_domain):
    paper_id = _seed_paper(tmp_db_with_domain)
    tmp_db_with_domain.execute(
        """
        INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES ('rag', 'topic', '', 'tree retrieval', 'seed')
        """
    )
    runner = _runner_from_dict({
        "domain_index": 0,
        "new_domain": "",
        "new_domain_desc": "",
        "collection_index": -1,
        "new_collection": "hierarchical indexing",
        "new_collection_desc": "Hierarchical retrieval over long documents.",
        "topics": ["tree retrieval", "tree retrieval"],
    })

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=runner,
    )
    rows = tmp_db_with_domain.execute(
        "SELECT COUNT(*) FROM topics WHERE target_kind='paper' AND target_id = ?", (paper_id,)
    ).fetchone()[0]
    assert rows == 1


# ===========================================================================
# Final state
# ===========================================================================


def test_status_is_classified_after_success(seeded):
    classify(
        paper_name="paper_name_2024",
        conn=seeded,
        call_llm=_fake_runner(),
    )
    row = seeded.execute(
        "SELECT status FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == PaperStatus.CLASSIFIED.value


def test_needs_review_is_zero_on_existing_domain(seeded):
    classify(
        paper_name="paper_name_2024",
        conn=seeded,
        call_llm=_fake_runner(),
    )
    row = seeded.execute(
        "SELECT needs_review FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == 0


def test_needs_review_is_one_on_new_domain_auto_create(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        call_llm=_fake_runner("classification_new_domain.json"),
    )
    row = tmp_db_with_domain.execute(
        "SELECT needs_review FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == 1


# ===========================================================================
# CLI
# ===========================================================================


def test_cli_prints_json_summary(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "lodestone.db"
    c = get_conn(db_path)
    init_db(c)
    _seed_domain(c)
    _seed_paper(c)
    c.close()

    def _fake_default(system, user, schema, response_model):
        return response_model.model_validate(
            _load_fixture("classification_rag_hierarchical.json")
        )

    monkeypatch.setattr(cp, "_call_llm_default", _fake_default)
    cp._main(["--paper", "paper_name_2024", "--db", str(db_path)])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["paper_name"] == "paper_name_2024"
    assert payload["domain"] == "rag"
    assert payload["status"] == PaperStatus.CLASSIFIED.value
