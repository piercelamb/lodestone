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

1. **BM25 free-text** — `search.py "query"` runs BM25 over the
   `sections` FTS5 index (paper abstracts ride along as the
   `# Abstract` chunk). Filter with `--domain <slug>` or
   `--collection <name>`. Hits are grouped by paper. The query is
   sanitized into a quoted-token phrase query, so punctuation in
   queries (`tree-sitter`, `BAAI/bge-small`, `O(1)`) is treated as
   phrase content rather than as FTS5 operators.
2. **Taxonomy lookup** — `--entity <name>`, `--topic <name>`,
   `--collection-lookup <name>` resolve through FTS5 first, then fall
   back to sqlite-vec KNN when no FTS match is found. Tier info and
   provenance (source paper, match tier) flow through on every hit.
3. **Browse** — `--collections`, `--topics`, `--entity-type <t>`,
   `--aliases <term>`, `--needs-review`. Use these to audit taxonomy.
4. **Table of contents** — `--toc <paper>` emits the section tree
   extracted during ingest (levels 1–3).
5. **Content extraction** — `--read <paper> [--section "Method"]`,
   `--figure <paper> <N>`. Figure BLOBs are written to a temp file via
   `tempfile.mkstemp`; the returned path is ephemeral. Copy if you need
   to keep it.

### Key flags

- `--needs-review` surfaces papers whose domain was auto-created by the
  classifier's LLM pass. Operator decides: rename, merge, or keep.
- Aliases always carry provenance: `source_paper` and `match_tier` are
  recorded per alias in `term_aliases`, enabling per-paper BM25 scoping
  through `terms_fts`.

## Ingestion (`ingest.py`)

```
uv run _system/scripts/ingest.py --url <arxiv_url_or_id> [--force] [--domain <slug>]
uv run _system/scripts/ingest.py --repo <github_url>      [--force] [--domain <slug>]
uv run _system/scripts/ingest.py --post <post_url>        [--force] [--domain <slug>]
uv run _system/scripts/ingest.py --pdf <local_path> [--no-split] [--force] [--domain <slug>]
uv run _system/scripts/ingest.py --pdf <chapter_path> --book-slug <slug> \
       --chapter-index <N> [--chapter-title <title>] [--force] [--domain <slug>]
```

- **Resumable** (by design — see below). Rerunning on the same
  `arxiv_id` picks up at the last completed stage, driven by
  `papers.status`. No duplicate work. `ingest.py` is fully resumable.
- `--force` cascade-deletes the paper (sections, term_aliases,
  paper_topics, figures, paper_references, papers) inside one
  transaction, then re-ingests. Preserves global taxonomy
  (`canonical_terms`, `term_embeddings`) so reuse across papers isn't
  lost.
- `--domain <slug>` overrides the classifier's domain choice and
  threads through to both fetch and classify.
- `--pdf <local_path>` ingests a local PDF book. Outline-split by
  default: one `papers` row per level-1 TOC entry. Chapters share the
  book's `content_hash`; `arxiv_id` is synthetic
  (`pdf:<sha256[:12]>:ch<NN>`); `paper_name` is
  `<book_slug>__ch<NN>_<chapter_slug>` (zero-padded so
  `ORDER BY paper_name` returns chapters in TOC order). When level-1
  yields fewer than 3 entries (books partitioned into "volumes" or
  "parts" with the real chapters at level-2), `discover_chapters`
  auto-falls back to level-2 if it has ≥3 entries; level-2 chapter
  boundaries are clipped at level-1 part headers so chapters don't
  bleed across parts. Consecutive TOC entries pointing at the same
  page are deduped (e.g. "Bibliography" + "Subject Index" registered
  at the same start page). PDFs whose embedded outline still can't be
  split fail fast — re-run with `--no-split` to ingest the whole PDF
  as one row (bare `<book_slug>` / `pdf:<sha256[:12]>`), or pre-slice
  manually. `--force` cascades all rows whose `arxiv_id` matches
  `pdf:<sha256[:12]>%` in one transaction. **Limitation**: bibliography
  back-resolution in `search.py` builds `arxiv.org/abs/...` hints from
  `arxiv_id`; for `pdf:` ids the hint is bogus (book chapters are very
  unlikely to be cited by `pdf:` id anyway).
- `--pdf <chapter_path> --book-slug <slug> --chapter-index <N>` is the
  hand-sliced variant: ingest one chapter PDF at a time and declare its
  position in a shared book namespace. `paper_name` follows the same
  `<book_slug>__ch<NN>_<chapter_slug>` convention, so
  `SELECT paper_name FROM papers WHERE paper_name LIKE 'foo__%'
  ORDER BY paper_name` lists the assembled chapters in TOC order
  exactly like the auto-split path. Per-chapter `content_hash` is the
  individual file's sha256 (siblings do NOT share an `arxiv_id`
  prefix); `--force` cascades by `paper_name LIKE '<book_slug>__chNN_%'`
  instead, so swapping in a different file for the same slot Just
  Works. `--chapter-title` is optional (defaults to the chapter PDF's
  title metadata). `--book-slug` must match `^[a-z0-9_]+$` and must
  not contain `__` (reserved as the book/chapter separator). Mutex
  with `--no-split`.

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
  `fastino/gliner2-large-v1` on the next ingest run. Expect latency.
- Do not bypass `ingest.py`'s `--force` and manually DELETE rows — the
  cascade order matters (FTS5 has no FK cascade; getting the order
  wrong trips `FOREIGN KEY constraint failed`).
- Do not hand-edit `~/.config/lodestone/config.toml` to a provider
  whose API key is not exported — pre-flight raises rather than
  silently falling back. Either export the key or delete the file to
  re-trigger env-based selection.

## SQL escape hatch (rare)

When none of the curated modes above fits the question, three extra
modes expose the DB directly. Reach for the curated tools FIRST — they
exist because they're the right answer 95% of the time.

- `--tables` — list every user table / view / virtual table.
  `--include-internal` adds FTS5 / vec0 shadow tables.
- `--schema TABLE` — print DDL + columns + indexes. Repeat the flag
  for many tables at once; missing names land in `missing` rather than
  raising.
- `--sql 'SELECT ...'` — run one read-only statement. Read-only is
  engine-enforced (`mode=ro` URI); writes return a `read_only_violation`
  soft-fail. Single statement only, hard ceiling of 1000 rows, 5 s
  wall-clock timeout. Paginate with `LIMIT N OFFSET M` + a stable
  `ORDER BY` in your own SQL.

Same three modes are exposed over MCP as `tables`, `schema`, and
`query`. Treat `query` as the last resort: it bypasses the curated
output shapes and gives you raw rows.
