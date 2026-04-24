# Lodestone — agent navigation contract

Audience: a future Claude Code skill (or other automation agent) opening
this repo for the first time. Read this before touching the DB.

## Overview

Everything lives in `lodestone.db`. To read it, use `search.py`. Do not
hand-edit rows, do not query the DB from arbitrary tooling, and do not
shell out to `sqlite3` for content — `search.py` is the contract.

Two CLIs write to the DB: `ingest.py` (full paper pipeline) and
`create_domain.py` (manual taxonomy registration). Every other access is
read-only via `search.py`.

## `search.py` is the only read entry point

Five modes, all emit JSON by default. Add `--human` for pretty text.

1. **BM25 free-text** — `search.py "query"` runs BM25 over
   `abstracts` and `sections`. Filter with `--domain <slug>` or
   `--collection <name>`. Returns paper + section hits ranked by rank.
2. **Taxonomy lookup** — `--entity <name>`, `--topic <name>`,
   `--collection-lookup <name>` resolve through FTS5 first, then fall
   back to sqlite-vec KNN when no FTS match is found. Tier info and
   provenance (source paper, match tier) flow through on every hit.
3. **Browse** — `--collections`, `--topics`, `--entity-type <t>`,
   `--aliases <term>`, `--needs-review`. Use these to audit taxonomy.
4. **Table of contents** — `--toc <paper>` emits the section tree
   extracted during ingest (levels 1–3).
5. **Content extraction** — `--read <paper> [--section "Method"]`,
   `--figure <paper> <N>`, `--page <paper> <N>`. BLOB data (figures,
   page renders) is written to a temp file via `tempfile.mkstemp`; the
   returned path is ephemeral. Copy if you need to keep it.

### Key flags

- `--needs-review` surfaces papers whose domain was auto-created by the
  classifier's LLM pass. Operator decides: rename, merge, or keep.
- Aliases always carry provenance: `source_paper` and `match_tier` are
  recorded per alias in `term_aliases`, enabling per-paper BM25 scoping
  through `terms_fts`.

## Ingestion (`ingest.py`)

```
uv run _system/scripts/ingest.py --url <arxiv_url_or_id> [--force] [--domain <slug>]
```

- **Resumable** (by design — see below). Rerunning on the same
  `arxiv_id` picks up at the last completed stage, driven by
  `papers.status`. No duplicate work. `ingest.py` is fully resumable.
- `--force` cascade-deletes the paper (abstracts, sections, entities,
  paper_topics, figures, page_images, papers) inside one transaction,
  then re-ingests. Preserves global taxonomy (`canonical_terms`,
  `term_aliases`, `term_embeddings`) so reuse across papers isn't lost.
- `--domain <slug>` overrides the classifier's domain choice and
  threads through to both fetch and classify.

Pipeline stages, in order: `fetch → convert → classify → extract →
index`. Pre-flight (`validate_models.check_models`) runs first —
unresolved LLM provider or uncached HF models fail fast before any DB
write.

### Provider configuration

Classify calls one of three LLM providers directly via its SDK:
Anthropic, OpenAI, or Gemini. Structured output is enforced
provider-side (tool_use / `response_format=json_schema` /
`responseSchema`), so the model cannot return malformed JSON.

Provider selection is resolved by `_system.llm.resolve_provider()` in
this order:

1. `~/.config/lodestone/config.toml` (XDG user config dir):
   ```toml
   [llm]
   provider = "anthropic"       # one of: anthropic | openai | gemini
   model = "claude-opus-4-7"    # optional; per-provider default applies
   ```
   If the config selects a provider whose env var is not set, pre-flight
   raises `ProviderKeyMissing` naming the var.
2. No config — inspect env vars:
   - Exactly one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
     `GEMINI_API_KEY` set → use it and persist the choice.
   - Multiple set in a TTY → interactive numbered-menu prompt, persist
     the pick.
   - Multiple set in non-TTY → raise `ProviderAmbiguous`.
   - None set → raise `ProviderUnconfigured`.

## Key semantics

- `arxiv_id` preserves its version suffix verbatim. `2301.12345v1` and
  `2301.12345v2` are **different rows** — the identity policy is
  version-sensitive by design. Do not normalize.
- `FAILED_HTML` is a terminal status without `--force`. A paper with
  that status had both arxiv.org/html and ar5iv 404 on the first pass;
  the stub row remains so `search.py --needs-review` can surface it.
- Canonical terms, aliases, and embeddings persist across `--force`
  deletes. This is intentional: cross-paper reuse would evaporate
  otherwise. A phase-2 "gardening" pass will reconcile orphans (terms
  whose sole producer was a since-deleted paper).

## Manual domain registration (`create_domain.py`)

```
uv run _system/scripts/create_domain.py --name <slug> --description "..."
```

- `--name` must match `[a-z0-9_-]{1,32}` (same charset the classifier's
  auto-sanitizer uses). Idempotent: re-running with the same name emits
  `{"created": false}` and exits 0.
- `needs_review=0` on the row (manual registration is explicit).

## What NOT to do

- Do not hand-edit `lodestone.db` rows. The FTS5 / vec0 tables are
  derived; direct writes desync them from the source-of-truth tables.
- Do not bypass `search.py` for content reads. Filesystem export is
  the contract — ephemeral tempfiles via the CLI.
- Do not shell out to `sqlite3 lodestone.db` for queries in scripts;
  that path doesn't load the sqlite-vec extension or apply the project
  pragmas (FK on, WAL, busy_timeout).
- Do not `rm -rf ~/.cache/huggingface/hub` without realizing that
  `validate_models.py` will re-download `BAAI/bge-small-en-v1.5` and
  `fastino/gliner2-base-v1` on the next ingest run. Expect latency.
- Do not bypass `ingest.py`'s `--force` and manually DELETE rows — the
  cascade order matters (FTS5 has no FK cascade; getting the order
  wrong trips `FOREIGN KEY constraint failed`).
- Do not hand-edit `~/.config/lodestone/config.toml` to a provider
  whose API key is not exported — pre-flight raises rather than
  silently falling back. Either export the key or delete the file to
  re-trigger env-based selection.
