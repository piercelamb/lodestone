"""Unit tests for _system/scripts/classify_paper.py.

No real `claude` CLI invocation — every subprocess call is either short-circuited
through the `run_subprocess` test seam or monkeypatched at module level. No
network, no ML weight loading (the resolver is called with ``embedder=None``;
tier 4 is not needed for these tests).
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import pytest

from _system.db.connection import get_conn
from _system.db.migrations import init_db
from _system.schemas.paper_metadata import PaperStatus
from _system.scripts import classify_paper as cp
from _system.scripts.classify_paper import (
    CLAUDE_ARGV,
    ClassifyDomainNameError,
    ClassifyEnvelopeError,
    ClassifyPaperNotFound,
    ClassifyStateError,
    ClassifySubprocessError,
    _build_prompt,
    _extract_intro,
    _sanitize_domain,
    classify,
)


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
        # Trivial text fingerprint — we never need semantic similarity in
        # these tests, only a non-null, shape-correct vector for vec0 writes.
        v[hash(text) % 384] = 1.0
        return v

    def embed_batch(self, texts):  # pragma: no cover
        return [self.embed(t) for t in texts]


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


@pytest.fixture(autouse=True)
def _patch_embedder_class(monkeypatch):
    """Swap the real sentence-transformers Embedder for the fake.

    classify() falls through to ``Embedder()`` when no embedder is passed;
    loading bge-small-en-v1.5 every test would be seconds of overhead for
    nothing — we never test semantic similarity here.
    """
    monkeypatch.setattr(cp, "Embedder", _FakeEmbedder)


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

_DEFAULT_MARKDOWN = (
    "# Introduction\n"
    "\n"
    "This paper presents a hierarchical retrieval approach.\n"
    "\n"
    "# Method\n"
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
    return cur.lastrowid


def _load_fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _fake_runner(envelope_file: str = "envelope_rag_hierarchical.json"):
    """Return a run_subprocess stub that yields the parsed fixture envelope."""
    envelope = json.loads(_load_fixture(envelope_file).decode("utf-8"))

    def _runner(prompt: str) -> dict:
        return envelope

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
# Prompt builder
# ===========================================================================


def test_build_prompt_contains_abstract_intro_and_existing_taxonomy():
    prompt = _build_prompt(
        abstract="abstract body",
        intro_text="intro body",
        domains=[("rag", "retrieval-augmented generation")],
        collections_by_domain={"rag": ["hierarchical indexing", "graph retrieval"]},
    )
    assert "abstract body" in prompt
    assert "intro body" in prompt
    assert "rag: retrieval-augmented generation" in prompt
    assert "hierarchical indexing" in prompt


def test_build_prompt_truncates_long_collection_list_with_n_more_suffix():
    colls = [f"coll_{i:02d}" for i in range(45)]
    prompt = _build_prompt(
        abstract="a",
        intro_text="i",
        domains=[("rag", None)],
        collections_by_domain={"rag": colls},
    )
    assert "coll_00" in prompt
    assert "coll_29" in prompt
    assert "coll_30" not in prompt
    assert "(+ 15 more; feel free to propose new)" in prompt


def test_build_prompt_includes_json_schema_stub():
    prompt = _build_prompt(
        abstract="a", intro_text="i", domains=[], collections_by_domain={}
    )
    assert '"domain": "..."' in prompt
    assert '"domain_is_new": true|false' in prompt
    assert '"collection": "..."' in prompt
    assert '"topics"' in prompt


def test_build_prompt_handles_no_existing_domains():
    prompt = _build_prompt(
        abstract="a", intro_text="i", domains=[], collections_by_domain={}
    )
    assert "(none yet" in prompt


def test_build_prompt_handles_missing_intro():
    prompt = _build_prompt(
        abstract="abstract only",
        intro_text="",
        domains=[("rag", None)],
        collections_by_domain={},
    )
    assert "abstract only" in prompt
    assert "not available" in prompt


# ===========================================================================
# Intro extraction
# ===========================================================================


def test_intro_prefers_introduction_case_insensitive():
    md = "# INTRODUCTION\nalpha intro\n# Background\nbeta bg\n"
    assert "alpha intro" in _extract_intro(md, paper_name="p")
    assert "beta bg" not in _extract_intro(md, paper_name="p")


def test_intro_falls_back_to_overview_when_no_introduction():
    md = "# Overview\noverview body\n# Conclusion\nconc body\n"
    assert "overview body" in _extract_intro(md, paper_name="p")


def test_intro_falls_back_to_background_when_no_intro_or_overview():
    md = "# Background\nbackground body\n# Method\nmethod body\n"
    assert "background body" in _extract_intro(md, paper_name="p")


def test_intro_falls_back_to_first_level_one_when_no_titled_match():
    md = "# Something Else\nfirst body\n# Method\nmethod body\n"
    assert "first body" in _extract_intro(md, paper_name="p")


def test_intro_empty_markdown_returns_empty_and_logs_warning(caplog):
    # lodestone loggers set propagate=False; attach caplog's handler directly.
    import logging as _logging
    logger = _logging.getLogger("lodestone.scripts.classify_paper")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(_logging.WARNING, logger="lodestone.scripts.classify_paper"):
            assert _extract_intro("", paper_name="p") == ""
            assert _extract_intro("   \n\n  ", paper_name="p") == ""
    finally:
        logger.removeHandler(caplog.handler)
    assert any(
        "empty markdown" in r.getMessage() for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_intro_no_headers_only_prose_falls_back_to_synthetic_abstract():
    # split_sections synthesizes a `# Abstract` chunk when there are no real
    # headers. That chunk's title is "Abstract", which is NOT in the intro
    # whitelist; the splitter marks level=1 so the "first level-1" fallback
    # kicks in and returns that prose. Either outcome is acceptable as long
    # as it does not raise and gives us *something* to prompt with.
    md = "Just some raw prose with no headers at all.\n"
    out = _extract_intro(md, paper_name="p")
    assert "raw prose" in out or out == ""


def test_intro_strips_breadcrumb_prefix():
    md = "# Introduction\nbody text here\n"
    out = _extract_intro(md, paper_name="p")
    # breadcrumb is "# Introduction" — it should not appear in the output
    assert not out.startswith("# Introduction")
    assert "body text here" in out


def test_intro_caps_length_at_8000_chars():
    big = "x" * 20000
    md = f"# Introduction\n{big}\n"
    out = _extract_intro(md, paper_name="p")
    assert len(out) <= 8000


# ===========================================================================
# Subprocess invariants
# ===========================================================================


def test_subprocess_argv_is_exact_and_shell_false(seeded, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["shell"] = kwargs.get("shell", True)
        captured["has_input"] = "input" in kwargs and kwargs["input"] is not None
        captured["timeout"] = kwargs.get("timeout")
        return subprocess.CompletedProcess(
            argv, 0,
            stdout=_load_fixture("envelope_rag_hierarchical.json"),
            stderr=b"",
        )

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    classify(paper_name="paper_name_2024", conn=seeded)

    assert captured["argv"] == list(CLAUDE_ARGV)
    assert captured["argv"] == ["claude", "-p", "--bare", "--output-format", "json"]
    assert captured["shell"] is False
    assert captured["has_input"] is True
    assert captured["timeout"] == 180


def test_subprocess_prompt_passed_via_stdin_not_argv(seeded, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = list(argv)
        captured["input"] = kwargs.get("input")
        return subprocess.CompletedProcess(
            argv, 0,
            stdout=_load_fixture("envelope_rag_hierarchical.json"),
            stderr=b"",
        )

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    classify(paper_name="paper_name_2024", conn=seeded)

    # Prompt bytes must NOT appear in argv.
    for arg in captured["argv"]:
        assert "Classify this paper" not in arg
    # Prompt must appear in input stream.
    assert captured["input"] is not None
    assert b"Classify this paper" in captured["input"]


def test_subprocess_nonzero_exit_retries_three_times_then_raises(seeded, monkeypatch):
    call_count = {"n": 0}

    def fake_run(argv, **kwargs):
        call_count["n"] += 1
        return subprocess.CompletedProcess(
            argv, 7, stdout=b"", stderr=b"boom"
        )

    # Kill tenacity sleeps so the test stays fast.
    monkeypatch.setattr(cp, "subprocess", cp.subprocess)
    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    _neuter_tenacity_sleep(monkeypatch)

    with pytest.raises(ClassifySubprocessError):
        classify(paper_name="paper_name_2024", conn=seeded)
    assert call_count["n"] == 3


def test_subprocess_json_parse_failure_retries(seeded, monkeypatch):
    call_count = {"n": 0}

    def fake_run(argv, **kwargs):
        call_count["n"] += 1
        return subprocess.CompletedProcess(
            argv, 0, stdout=b"not json at all", stderr=b""
        )

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    _neuter_tenacity_sleep(monkeypatch)

    with pytest.raises(json.JSONDecodeError):
        classify(paper_name="paper_name_2024", conn=seeded)
    assert call_count["n"] == 3


def test_missing_structured_output_key_raises_without_retry(seeded, monkeypatch):
    call_count = {"n": 0}
    odd = b'{"cost_usd": 0.0, "duration_ms": 0, "oops": "no structured_output here"}'

    def fake_run(argv, **kwargs):
        call_count["n"] += 1
        return subprocess.CompletedProcess(argv, 0, stdout=odd, stderr=b"")

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    _neuter_tenacity_sleep(monkeypatch)

    with pytest.raises(ClassifyEnvelopeError) as exc_info:
        classify(paper_name="paper_name_2024", conn=seeded)
    # Envelope head is embedded for debugging
    assert "oops" in str(exc_info.value) or "no structured_output" in str(exc_info.value)
    # Envelope-shape errors are deterministic — no retries.
    assert call_count["n"] == 1


def test_stderr_head_truncated_in_subprocess_error(seeded, monkeypatch):
    flood = b"X" * 100000  # much larger than 2KB cap

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=flood)

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    _neuter_tenacity_sleep(monkeypatch)

    with pytest.raises(ClassifySubprocessError) as exc_info:
        classify(paper_name="paper_name_2024", conn=seeded)
    msg = str(exc_info.value)
    # Message should contain some X's but not all 100k of them.
    assert "X" in msg
    assert len(msg) < 10000


# ===========================================================================
# Envelope parsing + validation
# ===========================================================================


def test_classify_parses_envelope_and_writes_paper_state(seeded):
    classify(
        paper_name="paper_name_2024",
        conn=seeded,
        run_subprocess=_fake_runner(),
    )
    row = seeded.execute(
        "SELECT domain, collection, status, needs_review FROM papers "
        "WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "rag"
    assert row[1] == "hierarchical indexing"
    assert row[2] == PaperStatus.CLASSIFIED.value
    assert row[3] == 0  # existing domain → no review


def test_schema_validation_error_is_not_retried(seeded, monkeypatch):
    call_count = {"n": 0}
    malformed = json.dumps(
        {
            "structured_output": {
                # missing required 'domain'
                "domain_is_new": False,
                "collection": "c",
                "topics": [],
            }
        }
    ).encode()

    def fake_run(argv, **kwargs):
        call_count["n"] += 1
        return subprocess.CompletedProcess(argv, 0, stdout=malformed, stderr=b"")

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    _neuter_tenacity_sleep(monkeypatch)

    with pytest.raises(Exception) as exc_info:
        classify(paper_name="paper_name_2024", conn=seeded)
    # Pydantic ValidationError is a plain ValueError subclass — just check
    # we didn't retry.
    assert call_count["n"] == 1
    assert "domain" in str(exc_info.value).lower()


# ===========================================================================
# Status / resume guard
# ===========================================================================


def test_status_fetched_blocks_classify(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain, status=PaperStatus.FETCHED.value)
    with pytest.raises(ClassifyStateError):
        classify(
            paper_name="paper_name_2024",
            conn=tmp_db_with_domain,
            run_subprocess=_fake_runner(),
        )


def test_status_failed_html_blocks_classify(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain, status=PaperStatus.FAILED_HTML.value)
    with pytest.raises(ClassifyStateError) as exc_info:
        classify(
            paper_name="paper_name_2024",
            conn=tmp_db_with_domain,
            run_subprocess=_fake_runner(),
        )
    assert "failed_html" in str(exc_info.value).lower()


def test_status_converted_proceeds(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain, status=PaperStatus.CONVERTED.value)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner(),
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
        run_subprocess=_fake_runner(),
    )


def test_paper_not_found_raises(tmp_db_with_domain):
    with pytest.raises(ClassifyPaperNotFound):
        classify(
            paper_name="no_such_paper",
            conn=tmp_db_with_domain,
            run_subprocess=_fake_runner(),
        )


# ===========================================================================
# Rerun semantics
# ===========================================================================


def test_rerun_deletes_existing_paper_topics_before_insert(tmp_db_with_domain):
    paper_id = _seed_paper(tmp_db_with_domain)
    tmp_db_with_domain.execute(
        "INSERT INTO paper_topics (paper_id, domain, topic) VALUES (?, ?, ?)",
        (paper_id, "rag", "stale topic"),
    )

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner(),
    )
    rows = tmp_db_with_domain.execute(
        "SELECT topic FROM paper_topics WHERE paper_id = ? ORDER BY topic",
        (paper_id,),
    ).fetchall()
    topics = [r[0] for r in rows]
    assert "stale topic" not in topics
    assert len(topics) == 2


def test_rerun_does_not_delete_canonical_terms_or_aliases(tmp_db_with_domain):
    paper_id = _seed_paper(tmp_db_with_domain)
    tmp_db_with_domain.execute(
        """
        INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES (?, ?, '', ?, ?)
        """,
        ("rag", "topic", "other topic", "some_other_paper"),
    )
    tmp_db_with_domain.execute(
        """
        INSERT INTO term_aliases (term_id, alias, source_paper, match_tier)
        VALUES (1, 'an alias', 'some_other_paper', 2)
        """
    )
    pre_terms = tmp_db_with_domain.execute(
        "SELECT COUNT(*) FROM canonical_terms"
    ).fetchone()[0]
    pre_aliases = tmp_db_with_domain.execute(
        "SELECT COUNT(*) FROM term_aliases"
    ).fetchone()[0]

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner(),
    )

    post_terms = tmp_db_with_domain.execute(
        "SELECT COUNT(*) FROM canonical_terms"
    ).fetchone()[0]
    post_aliases = tmp_db_with_domain.execute(
        "SELECT COUNT(*) FROM term_aliases"
    ).fetchone()[0]
    # Post >= pre: the classify run creates new canonicals but must not delete existing ones.
    assert post_terms >= pre_terms
    assert post_aliases >= pre_aliases


def test_second_run_idempotent_paper_topics_set(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner(),
    )
    first_topics = tmp_db_with_domain.execute(
        "SELECT topic FROM paper_topics ORDER BY topic"
    ).fetchall()

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner(),
    )
    second_topics = tmp_db_with_domain.execute(
        "SELECT topic FROM paper_topics ORDER BY topic"
    ).fetchall()
    assert first_topics == second_topics


# ===========================================================================
# Domain handling
# ===========================================================================


def test_sanitize_domain_lowercases_and_replaces_ws():
    assert _sanitize_domain("Multi-Agent Systems") == "multi-agent_systems"


def test_sanitize_domain_strips_illegal_chars():
    assert _sanitize_domain("Multi-Agent Systems!!") == "multi-agent_systems"


def test_sanitize_domain_truncates_at_32():
    long = "a" * 100
    out = _sanitize_domain(long)
    assert len(out) == 32


def test_sanitize_domain_empty_input_returns_empty():
    assert _sanitize_domain("   ") == ""
    assert _sanitize_domain("!!!") == ""


def test_new_domain_inserts_into_domains_and_sets_paper_needs_review(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner("envelope_new_domain.json"),
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
    assert "auto-created" in (domain_row[1] or "")


def test_llm_returns_dirty_domain_name_gets_sanitized(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner("envelope_bad_name.json"),
    )
    row = tmp_db_with_domain.execute(
        "SELECT domain FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "multi-agent_systems"


def test_proposed_domain_sanitizing_to_empty_raises(tmp_db_with_domain, monkeypatch):
    _seed_paper(tmp_db_with_domain)
    bad = {
        "structured_output": {
            "domain": "!!! ???",
            "domain_is_new": True,
            "collection": "c",
            "topics": ["t1"],
        }
    }

    def runner(prompt: str):
        return bad

    with pytest.raises(ClassifyDomainNameError):
        classify(
            paper_name="paper_name_2024",
            conn=tmp_db_with_domain,
            run_subprocess=runner,
        )


def test_llm_says_not_new_but_name_absent_treated_as_new(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    payload = {
        "structured_output": {
            "domain": "agents",  # not in the domains table (only 'rag' is seeded)
            "domain_is_new": False,
            "collection": "orchestration",
            "topics": ["planner"],
        }
    }

    def runner(prompt: str):
        return payload

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=runner,
    )
    row = tmp_db_with_domain.execute(
        "SELECT domain, needs_review FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "agents"
    assert row[1] == 1  # auto-created → flagged for review

    assert tmp_db_with_domain.execute(
        "SELECT 1 FROM domains WHERE name = 'agents'"
    ).fetchone() is not None


def test_domain_override_bypasses_llm_choice_and_does_not_flag_review(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner("envelope_new_domain.json"),
        domain_override="rag",  # override existing domain
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
        run_subprocess=_fake_runner("envelope_new_domain.json"),
        domain_override="theorem_proving",
    )
    row = tmp_db_with_domain.execute(
        "SELECT domain, needs_review FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()
    assert row[0] == "theorem_proving"
    assert row[1] == 0

    assert tmp_db_with_domain.execute(
        "SELECT 1 FROM domains WHERE name = 'theorem_proving'"
    ).fetchone() is not None


# ===========================================================================
# Resolver wiring
# ===========================================================================


def test_collection_and_topics_resolved_into_canonical_terms(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner(),
    )
    term_rows = tmp_db_with_domain.execute(
        """
        SELECT domain, term_type, canonical_name FROM canonical_terms
         ORDER BY term_type, canonical_name
        """
    ).fetchall()
    term_dict = {(d, tt): cn for d, tt, cn in term_rows}
    # New canonical_terms for rag/collection + rag/topic entries created.
    assert ("rag", "collection") in [(d, tt) for d, tt, _ in term_rows]
    assert ("rag", "topic") in [(d, tt) for d, tt, _ in term_rows]


def test_papers_collection_is_canonical_name_not_raw_llm(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    # Seed an existing canonical collection so the resolver hits tier 1.
    tmp_db_with_domain.execute(
        """
        INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES ('rag', 'collection', '', 'hierarchical indexing', 'seed')
        """
    )

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner(),
    )
    # The LLM said "hierarchical indexing"; the resolver confirms the existing
    # canonical row; papers.collection is written as the canonical_name.
    collection = tmp_db_with_domain.execute(
        "SELECT collection FROM papers WHERE paper_name = ?",
        ("paper_name_2024",),
    ).fetchone()[0]
    assert collection == "hierarchical indexing"


def test_paper_topics_uses_resolver_returned_canonical_names(tmp_db_with_domain):
    _seed_paper(tmp_db_with_domain)
    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=_fake_runner(),
    )
    topics = {
        r[0]
        for r in tmp_db_with_domain.execute(
            "SELECT topic FROM paper_topics WHERE domain = 'rag'"
        )
    }
    assert topics == {"tree retrieval", "long-context qa"}


def test_duplicate_topics_dedupe_by_term_id(tmp_db_with_domain):
    paper_id = _seed_paper(tmp_db_with_domain)
    # Seed a canonical topic so both incoming topic strings resolve to it.
    tmp_db_with_domain.execute(
        """
        INSERT INTO canonical_terms (domain, term_type, entity_type, canonical_name, first_seen_in)
        VALUES ('rag', 'topic', '', 'tree retrieval', 'seed')
        """
    )
    payload = {
        "structured_output": {
            "domain": "rag",
            "domain_is_new": False,
            "collection": "hierarchical indexing",
            "topics": ["tree retrieval", "tree retrieval"],
        }
    }

    def runner(prompt: str):
        return payload

    classify(
        paper_name="paper_name_2024",
        conn=tmp_db_with_domain,
        run_subprocess=runner,
    )
    rows = tmp_db_with_domain.execute(
        "SELECT COUNT(*) FROM paper_topics WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0]
    assert rows == 1


# ===========================================================================
# Final state
# ===========================================================================


def test_status_is_classified_after_success(seeded):
    classify(
        paper_name="paper_name_2024",
        conn=seeded,
        run_subprocess=_fake_runner(),
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
        run_subprocess=_fake_runner(),
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
        run_subprocess=_fake_runner("envelope_new_domain.json"),
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

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(
            argv, 0,
            stdout=_load_fixture("envelope_rag_hierarchical.json"),
            stderr=b"",
        )

    monkeypatch.setattr(cp.subprocess, "run", fake_run)
    cp._main(["--paper", "paper_name_2024", "--db", str(db_path)])
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["paper_name"] == "paper_name_2024"
    assert payload["domain"] == "rag"
    assert payload["status"] == PaperStatus.CLASSIFIED.value


# ===========================================================================
# Internal helpers
# ===========================================================================


def _neuter_tenacity_sleep(monkeypatch):
    """Make tenacity retries instant so retry tests run fast."""
    # tenacity's Retrying calls `sleep` through the attached wait strategy;
    # the simplest way to make it instant is to patch time.sleep on the
    # module's imported reference.
    import tenacity.nap
    monkeypatch.setattr(tenacity.nap.time, "sleep", lambda _s: None)
