![lodestone hero](assets/hero.jpeg)

# Lodestone: Research Retrieval for Claude Code

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-beta-orange)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Claude Code](https://img.shields.io/badge/Claude%20Code-Plugin-purple)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

`lodestone` is a single-file SQLite database and MCP server that ingests and retrieves over arXiv research papers, code repos, and blog posts. It's intended to be utilized by the agentic search mechanisms of coding agents like Claude Code and others to ground plans and implementations in the latest state-of-the-art research. It is best used when paired with a Claude Code plugin called [`/deep-sota`](https://github.com/piercelamb/deep-sota) (*deep state of the art*) — a sibling plugin to the [Deep Trilogy](https://github.com/piercelamb/deep-plan).

Where the Deep Trilogy is built for *writing* software, lodestone is built for *grounding* it: the assistant can cite a section, view an inline figure, or peek at the original training code instead of hallucinating.

> **Recommended pairing:** install [`/deep-sota`](https://github.com/piercelamb/deep-sota) alongside lodestone and drive the corpus through it. `/deep-sota` is the intended front-end — pass it a URL to ingest, or a research question to investigate, and it orchestrates the 21 MCP tools below with coverage checks, citation traversal, figure inlining, and code-repo reads. You *can* call lodestone's tools directly from any MCP client (Claude Code, Claude Desktop, Cursor, Cline, Zed — and the rest of this README documents how), but `/deep-sota` is what the system was designed around.

## TL;DR

Requires [`uv`](https://docs.astral.sh/uv/) on `PATH`.

```
/plugin marketplace add piercelamb/lodestone
/plugin install lodestone
/plugin install deep-sota
/reload-plugins
/lodestone:doctor
/mcp -> find lodestone -> enable or reconnect
```
Then **fully quit and relaunch Claude Code** (not `/reload-plugins`) — once. Done.

Why this sequence: `/lodestone:doctor` runs the one-time `uv sync` (~30–90s) and pre-seeds the venv *before* you restart, so when Claude Code launches the lodestone MCP server on the next startup it finds a populated venv on the first attempt. If you skip the doctor step and just restart, Claude Code launches the MCP server in parallel with the SessionStart prewarm hook — the MCP server hits an empty venv, exits 1, and Claude Code doesn't retry it mid-session, so you'd need a *second* restart for tools to appear. Doctor preempts that race.

Two CPU-only HuggingFace models (~400 MB total) download lazily on the first ingest with live progress streamed through `notifications/progress`. Subsequent sessions skip the prewarm entirely (it's hash-gated against `pyproject.toml` + `uv.lock`).

Hand [`/deep-sota`](https://github.com/piercelamb/deep-sota) an arXiv / GitHub / blog URL to ingest, or a research question to investigate — it picks the right tools (search, read, figures, citations, code repos) and surfaces coverage gaps honestly instead of hallucinating.

**Optional — pre-seed the taxonomy** (domains and collections) before your first ingest. Either use [my taxonomy](./taxonomy.json) or write your own in the same shape, then point Claude at [`seed_taxonomy.py`](./_system/scripts/seed_taxonomy.py). Skip seeding entirely and the classify step grows the taxonomy from scratch as you ingest — both paths are fully supported. See [Seeding the Taxonomy](#seeding-the-taxonomy).

Prefer to drive the tools yourself? Call `mcp__lodestone__ingest_paper` with an arXiv URL to seed your corpus, and `mcp__lodestone__search` / `read` / `figure` to query it directly.

## Table of Contents

- [Overview](#overview)
- [The Plugin Family](#the-plugin-family)
- [Why lodestone?](#why-lodestone)
- [When to Use](#when-to-use)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [MCP Tools](#mcp-tools)
- [Installation](#installation)
- [Standalone MCP Server (non-Claude-Code clients)](#standalone-mcp-server-non-claude-code-clients)
- [Usage](#usage)
- [CLI Ingestion](#cli-ingestion)
- [Seeding the Taxonomy](#seeding-the-taxonomy)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [Requirements](#requirements)
- [Best Practices](#best-practices)
- [Security & Privacy](#security--privacy)
- [Troubleshooting](#troubleshooting)
- [Developing on lodestone](#developing-on-lodestone)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Overview

**lodestone** is a corpus-builder and read-surface pair. The pipeline turns raw research artifacts into a structured, queryable knowledge base:

```
Fetch → Convert → Split → Classify → Index → Resolve Citations → Fetch Repo
```

Each ingested paper produces:
- **Markdown body**: LaTeXML-converted from arxiv.org/html or ar5iv, with LaTeX-source and PDF fallbacks for the long tail.
- **Flat section tree**: each chunk owns only its own paragraphs (no nested duplication), addressable by hierarchical slug. Section breadcrumbs are prepended to every BM25 chunk so keyword hits carry their hierarchical context.
- **Inline figures**: downscaled to 1280px max and re-encoded to JPEG q=85 (unless they have real transparency) so they fit in MCP image responses; image-less figures render as captioned text.
- **Domain + collection labels**: LLM-authored taxonomy with 1 primary collection + up to 3 secondaries per paper, plus extracted entities (GLiNER2 + 5-tier resolver) and topics.
- **BM25 + vector indexes**: sections, READMEs, taxonomy, and code files indexed separately so search can pick the right surface. READMEs are not chunked — each gets its own BM25 row.
- **Citation graph**: bibliography parsed (LaTeXML or heading-fallback), arxiv-id citations resolved forward (this → corpus) and backward (corpus → this) so a re-ingest can re-link earlier dangling references.
- **Code repo**: the paper's linked GitHub repo is discovered via the PapersWithCode API first, then by scraping the abstract / early footnotes if needed. Cloned and filtered (Linguist + ML-research-specific rules) with per-extension size caps (50 KB for JSON/CSV/TSV, 1 MB otherwise), and per-file rows indexed for `repo_tree` / `read_code`.

**One LLM call per ingest.** The whole pipeline — fetch, convert, split, entity extraction, indexing, citation resolution, repo fetch — is deterministic. Only the *classify* step hits an LLM (capped at the first ~8 KB of an arXiv paper or 10 KB of a blog/README). Two CPU-only models do the rest of the heavy lifting locally: `bge-small-en-v1.5` for the resolver's nearest-neighbor tier and `gliner2-large-v1` for entity extraction.

Blog posts ride the same `collections` table as papers; standalone repos ingest without a paper anchor. Post citations are forward-only (post cites paper) — the backward pass is paper-only.

## The Plugin Family

`lodestone` and [`/deep-sota`](https://github.com/piercelamb/deep-sota) ship from the same `piercelamb-plugins` marketplace as the Deep Trilogy. The research half pairs as cleanly as the build half:

```
┌───────────────────────────────────────────────────────────────────┐
│                     THE PIERCELAMB PLUGINS                        │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│   Research half:                  Build half:                     │
│                                                                   │
│   ┌──────────────┐                ┌──────────────┐                │
│   │  lodestone   │                │ /deep-project│                │
│   │  (corpus +   │                │  (decompose) │                │
│   │   MCP tools) │                └──────┬───────┘                │
│   └──────┬───────┘                       ▼                        │
│          │                        ┌──────────────┐                │
│          ▼                        │  /deep-plan  │                │
│   ┌──────────────┐                │    (plan)    │                │
│   │  /deep-sota  │                └──────┬───────┘                │
│   │ (orchestrate │                       ▼                        │
│   │   research)  │                ┌──────────────┐                │
│   └──────────────┘                │/deep-implement│                │
│                                   │   (build)    │                │
│                                   └──────────────┘                │
│                                                                   │
│   "what's the SOTA on X?"         "build me a SaaS platform"      │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

| Plugin | Purpose | Pairs with |
|--------|---------|------------|
| [`lodestone`](https://github.com/piercelamb/lodestone) | Build & query a personal research corpus (arXiv, blogs, repos) | `/deep-sota`, any MCP client |
| [`/deep-sota`](https://github.com/piercelamb/deep-sota) | Orchestrate paper- and repo-grounded research over lodestone | `lodestone` |
| [`/deep-project`](https://github.com/piercelamb/deep-project) | Decompose vague requirements into focused spec files | `/deep-plan` |
| [`/deep-plan`](https://github.com/piercelamb/deep-plan) | AI-assisted deep planning with research, review, and TDD | `/deep-implement` |
| [`/deep-implement`](https://github.com/piercelamb/deep-implement) | Implement section files with TDD and integrated code review | — |

## Why lodestone?

### Without lodestone
```
You: "What's the state of the art on long-context retrieval?"
Claude: *web searches, hallucinates citations, can't see figures or code*
Result: Surface-level answers, broken arxiv links, confident wrong claims
```

### With lodestone (driven by `/deep-sota`)
```
You: /deep-sota https://arxiv.org/abs/2305.10601   # ingest first
You: /deep-sota "long-context retrieval"           # then research
/deep-sota: coverage check → top-down taxonomy / bottom-up search →
            read sections → follow citations → view figures → read training code
Result: Grounded answers with section-level citations, inline figures, real code
        excerpts — and an honest "lodestone doesn't have X" when it doesn't.
```

**One-time cost**: a few minutes per ingest (LaTeXML conversion, classification, embedding). Ingest is explicit — hand [`/deep-sota`](https://github.com/piercelamb/deep-sota) a URL whenever you want to grow the corpus; it won't silently fetch papers behind your back.
**Payoff**: every future query against those papers grounds in *your* corpus, no API calls.

## When to Use

**Use lodestone when:**
- You want Claude to ground answers in specific papers, not generic web search.
- You read enough arXiv to benefit from a queryable local index.
- You want figure-level grounding (the assistant *sees* the architecture diagram, not just the caption).
- You want cross-paper citation following without third-party APIs.
- You want a portable, single-file corpus you control end-to-end.

**Skip lodestone when:**
- One-off questions that web search handles.
- You don't have arXiv-style sources to ingest.
- You need a multi-user knowledge base — lodestone is a personal, local-first single-file DB.

## Quick Start

> **TL;DR**: Install lodestone + [`/deep-sota`](https://github.com/piercelamb/deep-sota), restart, (optionally) seed the taxonomy, ingest at least one paper, then ask `/deep-sota` a research question. An empty corpus can't answer anything — `/deep-sota` won't auto-fetch papers behind your back.

**1. Install both plugins (canonical sequence):**
```
/plugin marketplace add piercelamb/lodestone
/plugin install lodestone
/plugin install deep-sota
/reload-plugins
/lodestone:doctor
```

Then **fully quit and relaunch Claude Code** (not `/reload-plugins`).

The `/reload-plugins` step makes the new slash commands (including `/lodestone:doctor`) available in the current session. `/lodestone:doctor` does the one-time `uv sync` (~30–90s) so the venv is populated *before* you restart. On the next launch, Claude Code finds the venv ready on its first MCP-server attempt and `mcp__lodestone__*` tools register immediately.

If you skip the doctor step and just restart, Claude Code launches the lodestone MCP server in parallel with the SessionStart prewarm hook. The server hits an empty venv, exits 1, and isn't retried until you restart again — so you'd need *two* restarts for tools to appear. The doctor-first flow makes one restart sufficient.

**2. Drive the corpus through `/deep-sota` (recommended):**

`/deep-sota` is invoked for one of three reasons — and the URL form is how you seed:

```
# (a) Ingest — when you have a URL you want in the corpus
/deep-sota https://arxiv.org/abs/2305.10601
/deep-sota https://github.com/owner/repo
/deep-sota https://lilianweng.github.io/posts/2023-06-23-agent/

# (b) Research a question — once the corpus has relevant material
/deep-sota "long-context retrieval"

# (c) Ground an implementation plan in research
/deep-sota "what's the SOTA on RAG architectures with KV-cache offloading?"
```

`/deep-sota` runs a coverage check first, picks the right read tools (taxonomy walk, BM25, section reads, citation traversal, figure inlining, code-repo reads), and tells you honestly when lodestone doesn't have something instead of hallucinating. It does **not** auto-ingest on research questions — if the corpus lacks coverage, it'll surface the gap and (for arXiv-id citations) hand you ready-to-run `ingest_paper` calls. Two CPU-only HuggingFace models (`bge-small-en-v1.5` for embeddings, `gliner2-large-v1` for entity extraction — ~400 MB total) download lazily on the first ingest.

**Or, drive lodestone directly (secondary path):**

If you'd rather call tools yourself, ask Claude to ingest a paper and search:
```
Use mcp__lodestone__ingest_paper on https://arxiv.org/abs/2305.10601
Then search lodestone for "tree of thoughts" and read the methodology section.
```

Either way, your DB lives at `~/.lodestone/lodestone.db` and survives `/plugin update` cache refreshes.

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                       lodestone pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ingest_paper(arxiv_url)                                       │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐  arxiv.org/html → ar5iv → LaTeX src → PDF    │
│   │    Fetch     │  (4-tier fallback; arxiv ToS-rate-limited)   │
│   └──────┬───────┘                                              │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │   Convert    │  LaTeXML → markdown + figures + references   │
│   └──────┬───────┘                                              │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │    Split     │  Flat section tree, each owning its prose    │
│   └──────┬───────┘                                              │
│          ▼                                                      │
│   ┌──────────────┐  Domain + 1 primary / 0-3 secondary          │
│   │   Classify   │  collections + topics — the ONE LLM call     │
│   │  (1 LLM call)│  per ingest (capped at 8-10k chars of input) │
│   └──────┬───────┘                                              │
│          ▼                                                      │
│   ┌──────────────┐  GLiNER2 + Schwartz-Hearst acronyms +        │
│   │   Entities   │  5-tier resolver (exact → normalized →       │
│   │              │  fuzzy → embedding → mint canonical)         │
│   └──────┬───────┘                                              │
│          ▼                                                      │
│   ┌──────────────┐  BM25 (sections, readmes, taxonomy) +        │
│   │    Index     │  sqlite-vec embeddings + term aliases        │
│   └──────┬───────┘                                              │
│          ▼                                                      │
│   ┌──────────────┐  Forward (this → corpus) + backward          │
│   │  Resolve     │  (corpus → this) arxiv-id resolution         │
│   │  Citations   │                                              │
│   └──────┬───────┘                                              │
│          ▼                                                      │
│   ┌──────────────┐  Clone paper's GitHub repo if present;       │
│   │  Fetch Repo  │  filter to real source (.py, .md, .ipynb)    │
│   └──────┬───────┘                                              │
│          ▼                                                      │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    lodestone.db                          │  │
│   │   sections · figures · code_files · collections ·        │  │
│   │   paper_references · readmes_fts · vec_sections · …      │  │
│   └──────────────────────────────────────────────────────────┘  │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │  MCP server  │  21 read/write tools (stdio or HTTP)         │
│   └──────────────┘                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## MCP Tools

All 21 tools surface as `mcp__lodestone__<name>` once the plugin is installed.

### Discovery & search

| Tool | Purpose |
|------|---------|
| `search` | Composite first-pass: taxonomy + sections + READMEs returned as three labeled buckets. GitHub-style query syntax (AND/OR/NOT, `"phrase"`, `term*`, `paper:` / `domain:` / `collection:` / `surface:` qualifiers). Multi-query fan-out via array `query` arg; `union=true` fuses with Reciprocal Rank Fusion. |
| `bm25` | Direct BM25 with `scope=sections|readmes|both`, offset pagination, `recency_boost` soft-tilt, and `since=YYYY[-MM-DD]` hard floor. |
| `lookup` | Taxonomy lookup over domains, collections, entities, topics, and aliases. |
| `coverage` | "Do I have anything on X?" — combines FTS hits, exact taxonomy matches, and rapidfuzz nearest-neighbors across collections/entities/topics/aliases with similarity scores. |
| `browse` | Enumerate taxonomy (domains, collections, topics, entities, papers, posts) with filters. |
| `overview` | Top-down corpus summary (totals, domains, recent ingests). |
| `collection` | Drill into one or more collections with abstracts; auto-trims when ≥2 names. |

### Read

| Tool | Purpose |
|------|---------|
| `toc` / `toc_many` | Section tree for one or many papers/posts, by slug. |
| `read` | Read a section by slug, or the full document. Slug-unified across papers and posts. |
| `figure` | Inline figure image (returns an MCP `image` block — pixels reach the model). |
| `citations` | Forward citations (this paper cites X, Y) and backward citations (X, Y cite this paper). |
| `repo_tree` | File listing for a paper's linked code repo. |
| `read_code` | Read a specific path (and optional line range) from a paper's repo. |
| `repo` | Read surface for standalone (non-paper-anchored) repos. |

### Schema escape hatches

| Tool | Purpose |
|------|---------|
| `tables` | List database tables. |
| `schema` | Get the schema for one or more tables. |
| `query` | Read-only SQL escape hatch when the curated tools don't fit. |

### Ingest

| Tool | Purpose |
|------|---------|
| `ingest_paper` | arXiv URL or ID → full pipeline. Emits `notifications/progress` per stage. |
| `ingest_post` | HTML blog post URL → trafilatura-converted, classified, indexed. |
| `ingest_repo` | Standalone GitHub repo URL → cloned, filtered, classified. |

Long ingests stream progress to the client. HTTP transport supports SSE for non-Claude-Code clients (`Accept: text/event-stream`).

## Installation

### Prerequisites

- [Claude Code](https://claude.ai/code), or any MCP-speaking client — see [Manual Setup](#manual-setup-other-mcp-clients) below.
- [uv](https://docs.astral.sh/uv/) on `PATH`
- Python 3.11+
- (Optional) LLM API keys for classification — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`. Lodestone picks whichever is configured.

### Install via Marketplace (Recommended)

**Canonical install sequence** — install both plugins, reload, run the doctor (which performs the one-time `uv sync` proactively), then restart Claude Code **once**:

```
/plugin marketplace add piercelamb/lodestone
/plugin install lodestone
/plugin install deep-sota
/reload-plugins
/lodestone:doctor
```

Then **fully quit and relaunch Claude Code** (not `/reload-plugins`).

What each step does:

| Step | Purpose |
|---|---|
| `/plugin marketplace add piercelamb/lodestone` | Adds the `piercelamb-plugins` marketplace. Skip if you already have it from `/deep-project`, `/deep-plan`, or `/deep-implement`. |
| `/plugin install lodestone` + `/plugin install deep-sota` | Installs both plugins. `/deep-sota` is the [recommended front-end](https://github.com/piercelamb/deep-sota) for the 21 MCP tools. |
| `/reload-plugins` | Makes the new slash commands (including `/lodestone:doctor`) available in the current session. |
| `/lodestone:doctor` | Runs the one-time `uv sync` (~30–90s) and pre-seeds the venv. Idempotent — also doubles as a diagnostic. |
| Full quit + relaunch | Registers the lodestone MCP server against the now-populated venv. |

> **Why doctor *before* restart?** Claude Code launches MCP servers in parallel with `SessionStart` prewarm hooks ([per its hook docs](https://code.claude.com/docs/en/hooks)) and won't retry a server that fails mid-session. If you skip `/lodestone:doctor` and just restart, the lodestone MCP server attempts to launch against an empty venv, exits 1, and stays unavailable until you restart *a second time*. Doctor-first preempts the race.

Embedding weights (~400 MB total) download lazily on the first `ingest_*` call with live progress.

If `mcp__lodestone__*` tools don't appear after this sequence, re-run `/lodestone:doctor` — it diagnoses `uv` presence, venv state, MCP registration, and DB writability in one shot.

> **Already installed `/deep-project`, `/deep-plan`, or `/deep-implement`?** All five plugins share the `piercelamb-plugins` marketplace. Skip the `marketplace add` step.

> **Prefer the UI for enable/disable?** Run `/plugins` after install to scroll to "Installed" and manage each plugin individually.

### Standalone MCP Server (non-Claude-Code clients)

> ⚠️ **Untested:** the Claude Code plugin path above is what I run daily and the only configuration I've personally verified end-to-end. The standalone instructions below are written from the code, not from a working install on each listed client. Treat them as a starting point — exact config keys, file paths, and reload behavior vary by client and version. Please open an issue or PR if you get a client working so the recipe can be confirmed.

The `lodestone-mcp` binary is a standalone Python MCP server. It has no Claude Code dependencies and runs against any client that speaks the [Model Context Protocol](https://modelcontextprotocol.io/) — Claude Desktop, Cursor, Cline, Zed, Continue, custom clients, MCP inspector, etc.

#### 1. Clone and install

```sh
git clone https://github.com/piercelamb/lodestone.git
cd lodestone
uv sync
```

`uv sync` materializes a venv at `./.venv/` and installs `lodestone-mcp` as a console script at `./.venv/bin/lodestone-mcp`. You can either:
- Activate the venv (`source .venv/bin/activate`) so `lodestone-mcp` is on `PATH`, or
- Reference the binary by absolute path in your client config (recommended — it's robust to PATH changes).

Verify the install:
```sh
.venv/bin/lodestone-mcp --help
```

#### 2. Pick a database location

The server reads `LODESTONE_DB` (an absolute path to a SQLite file) at startup. The directory must exist; the file is auto-created on first run if missing.

```sh
mkdir -p ~/.lodestone
export LODESTONE_DB=$HOME/.lodestone/lodestone.db
```

You can use the same `~/.lodestone/lodestone.db` the Claude Code plugin uses if you want a shared corpus across clients — but **don't run both the plugin and a standalone server against the same DB concurrently**. Each server pins the DB inode at startup; if one server's pipeline writes while another is mid-query, you'll get inconsistent reads. Run one at a time, or use separate DB files.

#### 3. Seed the corpus from the CLI before connecting

The DB is empty after install. MCP tools will return cleanly but with no results until you ingest something. Bootstrap from the CLI *first* — not just because it's easier to debug, but because the **first CLI run is what configures the LLM provider** for the classify step. The MCP server can't prompt for provider/model (no TTY), so it relies on a config file the CLI's first-run picker writes.

```sh
# Configure an LLM provider for the classifier (export at least one):
export ANTHROPIC_API_KEY=sk-ant-...
# and/or: export OPENAI_API_KEY=sk-...
# and/or: export GEMINI_API_KEY=...

# Ingest a paper — triggers the first-run picker if no config exists yet:
uv run python -m _system.scripts.ingest --url https://arxiv.org/abs/2305.10601
```

**First-run picker.** On a TTY, the first CLI invocation walks you through:

1. **Provider** — if multiple API keys are set, a numbered menu lists the available providers (`anthropic`, `openai`, `gemini`) and you pick one. If only one key is set, that provider is picked silently.
2. **Model** — a numbered menu of known-good models for the chosen provider, with the first entry as the default. Press Enter for the default, type a number, or paste a custom model ID.

Both choices are persisted to `~/.config/lodestone/config.toml` (XDG-compliant — uses `$XDG_CONFIG_HOME/lodestone/config.toml` if set, `%APPDATA%\lodestone\config.toml` on Windows). Every subsequent CLI *and MCP* call reads from that file silently. Edit the file directly to switch provider or model later.

**Headless / CI?** Without a TTY, the picker can't run. Either set only one provider key (single-key path silently picks that provider + default model and writes the config), or write `~/.config/lodestone/config.toml` by hand:

```toml
[llm]
provider = "anthropic"  # or "openai", "gemini"
model = "claude-opus-4-7"  # see _system/llm/<provider>_adapter.py for the catalog
temperature = 0.2
```

The first ingest call also downloads two CPU-only HuggingFace models — `bge-small-en-v1.5` (embeddings) and `gliner2-large-v1` (entity extraction), ~400 MB total — to `~/.cache/huggingface/`. From the MCP server, the download streams live `notifications/progress` to the client so it's visible, not an opaque hang; subsequent ingests reuse the cached weights. It also clones the paper's GitHub repo if one is linked.

Verify the DB is populated:
```sh
uv run python -m _system.scripts.search --search "tree of thoughts"
```

#### 4. Choose a transport

`lodestone-mcp` supports two transports. Both expose the same 21 tools.

**Stdio (default)** — the client spawns the server as a subprocess and talks JSON-RPC over its stdin/stdout. This is the standard MCP transport and works on every spec-compliant client.

**HTTP (opt-in)** — `lodestone-mcp --http` runs a long-lived HTTP server. The client connects via a URL. Useful as a workaround when stdio isn't an option, or when you want one server process shared by multiple clients. Supports SSE (`Accept: text/event-stream`) so long-running ingest tools can stream `notifications/progress` to clients that handle it.

#### 5a. Stdio configuration

Most MCP clients use a JSON config that maps a server name to a launch command. The lodestone entry looks like:

```jsonc
{
  "mcpServers": {
    "lodestone": {
      "command": "/abs/path/to/lodestone/.venv/bin/lodestone-mcp",
      "args": [],
      "env": {
        "LODESTONE_DB": "/abs/path/to/lodestone.db",
        "MCP_TIMEOUT": "30000",
        "MCP_TOOL_TIMEOUT": "1800000"
      }
    }
  }
}
```

Notes:
- Use an **absolute path** for `command` — clients launch with an unpredictable working directory and PATH.
- `LODESTONE_DB` must be absolute; the file is auto-created if missing.
- `MCP_TIMEOUT` (server startup, ms) and `MCP_TOOL_TIMEOUT` (per-tool-call, ms) are read by some MCP clients. Long ingests stream `notifications/progress` to keep the call alive (per the MCP spec, clients should treat progress as activity). If your client still kills mid-ingest, set its equivalent timeout key generously — the example block above uses 30 minutes for `ingest_paper`.
- If your LLM provider key for classification isn't already in the launching shell, add it to `env`.

**Client-specific config paths** (verify against current docs — these change):

| Client | Config file |
|--------|-------------|
| Claude Desktop | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) |
| Cursor | `~/.cursor/mcp.json` or workspace `.cursor/mcp.json` |
| Cline (VS Code) | Cline settings UI → MCP servers (writes to its own JSON) |
| Zed | `~/.config/zed/settings.json` → `context_servers` |
| Continue | `~/.continue/config.yaml` → `mcpServers` |
| MCP Inspector | `npx @modelcontextprotocol/inspector /abs/path/to/lodestone-mcp` |

Most clients hot-reload the config; some (Claude Desktop) require a full restart.

#### 5b. HTTP configuration

Run the server in a dedicated terminal or as a background service:

```sh
LODESTONE_DB=/abs/path/to/lodestone.db \
ANTHROPIC_API_KEY=sk-ant-... \
/abs/path/to/lodestone/.venv/bin/lodestone-mcp \
  --http --host 127.0.0.1 --port 8765
```

Flags:
- `--http` — enable HTTP transport (without this, the server runs as stdio).
- `--host` — bind address, default `127.0.0.1`.
- `--port` — port, default `8765`.

Point the client at the URL:

```jsonc
{
  "mcpServers": {
    "lodestone": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

The HTTP endpoint is `/mcp`. SSE upgrade happens automatically when the client sends `Accept: text/event-stream` on a tool call — there's no separate URL.

**Security:** the HTTP transport ships with **no authentication and no TLS**. It binds to loopback by default for that reason. Do not bind to a public interface (`--host 0.0.0.0`) or expose the port through a router without putting a reverse proxy with auth in front. The MCP server has `query` and `ingest_*` tools that can read arbitrary rows and clone arbitrary repos — anyone who can reach the port can drive them.

For a hands-off background process, wrap the command in `launchd` (macOS), `systemd --user` (Linux), or `pm2`. Restart on failure; don't auto-restart on exit code 0 (that's a clean shutdown).

#### 6. Verify the server is reachable

**Stdio:** restart the client and check its MCP/tools panel. Tools should appear as `lodestone__search`, `lodestone__read`, etc. (Exact prefix varies by client; some show the server name, some don't.) Try `overview` first — it works against an empty DB and confirms the round-trip.

**HTTP:** check the server is up:
```sh
curl -s http://127.0.0.1:8765/mcp -X POST \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json,text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | head -c 500
```
You should see a JSON-RPC response listing all 21 tools.

#### 7. Updating

```sh
cd /abs/path/to/lodestone
git pull
uv sync
```
Then restart the server (stdio: restart the client; HTTP: restart the long-lived process).

#### Known standalone gotchas

- **The server doesn't manage its own DB lifecycle beyond inode pinning.** If you delete `lodestone.db` while the server is running, the next `tools/call` returns a loud error — restart the server.
- **Progress notifications require client support.** `ingest_paper` emits `notifications/progress` keyed on the client's `_meta.progressToken`. Clients that don't send a token see a silent ingest until the final response. Clients that send a token but don't render progress events see one too — the notifications just aren't displayed.
- **The classifier requires an LLM API key.** Without `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY`, ingest fails at the classify stage. The other stages (fetch, convert, split, index, citations, repo) work offline.
- **The MCP server can't run the first-run provider picker.** If `~/.config/lodestone/config.toml` doesn't exist and multiple API keys are set, the MCP server raises `ProviderAmbiguous` because it can't prompt for a TTY-less process. Either run a CLI ingest first to write the config, set only one provider key, or write the TOML by hand. See [LLM provider and model](#llm-provider-and-model).
- **`uv` must be on PATH for the CLI scripts**, but not for the standalone server itself (it runs from the materialized venv binary).

## Usage

### Database location

By default the SQLite file lives at `~/.lodestone/lodestone.db` so it survives `/plugin update` cache refreshes. Override with the `LODESTONE_DB` environment variable when running manually — but **don't `export LODESTONE_DB` in your shell rc**: the plugin entrypoint explicitly unsets inherited values to prevent the prod DB from being redirected to a dev clone.

The MCP server pins the DB inode at startup and re-checks it on every `tools/call`. If the file at the path ever gets unlinked-and-recreated under a running server, the next call returns a loud error telling you to restart instead of silently writing to an orphaned inode.

### Recommended: drive lodestone with `/deep-sota`

The intended way to use lodestone from Claude Code is through [`/deep-sota`](https://github.com/piercelamb/deep-sota). It serves three distinct invocations:

**Ingest (URL → corpus)** — give it an arXiv, GitHub, or blog URL:
```
/deep-sota https://arxiv.org/abs/2305.10601
/deep-sota https://github.com/owner/repo
/deep-sota https://example.com/some-blog-post
```

**Research a question** — natural-language query over the existing corpus:
```
/deep-sota "long-context retrieval"
/deep-sota "what's the SOTA on agentic web search?"
/deep-sota "compare RAG architectures with KV-cache offloading"
```

**Ground an implementation plan in research** — same query shape, different intent. Use before `/deep-plan` for research-heavy specs.

For research questions, `/deep-sota` knows the right tool sequence (top-down taxonomy walk → bottom-up keyword search → section reads → citation traversal → figure inlining → code-repo reads) and runs a `coverage` check before asserting that lodestone doesn't have something. It does **not** silently ingest new papers when the corpus lacks coverage — instead it surfaces the gap (and, for arXiv-id citations, hands you exact `ingest_paper` calls to run).

Treat `/deep-sota` as the default entry point. The rest of this section documents the lower-level tool calls underneath, which you can also drive by hand for fine control.

### Driving the tools directly (secondary path)

You can also bypass `/deep-sota` and ask Claude Code to call lodestone tools individually. Useful when you want a specific operation rather than a full research workflow:

- *"Ingest [arxiv URL]"* → `ingest_paper`
- *"Search lodestone for X"* → `search` or `bm25`
- *"Show me figure 3 of paper Y"* → `figure`
- *"Read the methodology section of paper Y"* → `toc` then `read`
- *"What papers cite Z?"* → `citations`
- *"Do I have anything on long-context retrieval?"* → `coverage`

For non-Claude-Code clients (Claude Desktop, Cursor, Cline, Zed, …) `/deep-sota` is unavailable, so direct tool calls are the only path. See [Standalone MCP Server](#standalone-mcp-server-non-claude-code-clients) for setup.

## CLI Ingestion

For bulk ingest, or if you've cloned the repo for development:

```sh
uv sync
uv run python -m _system.scripts.ingest --url https://arxiv.org/abs/2305.10601
uv run python -m _system.scripts.ingest --post https://example.com/some-blog-post
uv run python -m _system.scripts.ingest --repo https://github.com/owner/name
```

Additional CLI tools:
- `python -m _system.scripts.search --search "your query"` — mirrors the `search` MCP tool
- `python -m _system.scripts.seed_taxonomy` — bulk-seed domains and collections from `taxonomy.json` (see [Seeding the Taxonomy](#seeding-the-taxonomy))
- `python -m _system.scripts.taxonomy_tree` — print the current taxonomy as a tree
- `python -m _system.scripts.reset_db` — nuke the dev DB (will refuse to touch the canonical path)
- `python -m _system.scripts.sync_plugin` — sync the working tree into a local plugin install for dev testing

## Seeding the Taxonomy

> **Optional but recommended.** The classifier works fine against an empty taxonomy — it'll mint new domains and collections as it goes. But seeding it with a curated starting set gives the first dozen ingests a stable, sensible scaffold to attach to, and avoids the early-corpus problem where two related papers land in two almost-but-not-quite-identical fresh collections.

This repo ships my personal taxonomy — the same one I use to drive [`/deep-sota`](https://github.com/piercelamb/deep-sota) daily. It covers an AI-research-heavy slice (retrieval & RAG, LLM training & inference, agents, evaluation, multimodal, etc.) and lives in two files at the repo root:

- **[`taxonomy.json`](./taxonomy.json)** — machine-readable, what `seed_taxonomy.py` reads.
- **[`taxonomy_tree.md`](./taxonomy_tree.md)** — human-readable rendering of the same data. Browse this first to see what you'd be importing.

### Seed via Claude Code (plugin install)

The cleanest way is to point Claude at the seed file:

```
Use mcp__lodestone__query to check if the domains table is empty,
then run `python -m _system.scripts.seed_taxonomy` to bulk-seed
from my taxonomy.json. The script is at:
https://github.com/piercelamb/lodestone/blob/main/_system/scripts/seed_taxonomy.py
```

Or download my `taxonomy.json` and pass its path explicitly:

```sh
curl -O https://raw.githubusercontent.com/piercelamb/lodestone/main/taxonomy.json
uv run --project ~/.claude/plugins/cache/piercelamb-plugins/lodestone \
  python -m _system.scripts.seed_taxonomy --file ./taxonomy.json
```

(Plugin install path varies by Claude Code version; substitute as needed. The script reads `LODESTONE_DB` for the target DB.)

### Seed via dev clone (CLI)

If you've cloned the repo:

```sh
uv sync
uv run python -m _system.scripts.seed_taxonomy
```

By default this reads `./taxonomy.json` and writes to whatever `LODESTONE_DB` points at (or `./lodestone.db` in a dev workspace). Pass `--file /abs/path/to/your-taxonomy.json` to seed from a different file.

### What happens during ingest

The classifier receives the existing domain/collection taxonomy on every classify call and prefers to slot new artifacts into existing buckets. It's also allowed to extend the taxonomy when nothing fits — so the seed is a starting scaffold, not a hard schema. Over time the taxonomy grows in whichever directions your ingests pull it.

### Bring your own taxonomy

If my taxonomy is too AI-research-flavored for what you're studying, write your own. `taxonomy.json` has a simple shape (look at the file or `taxonomy_tree.md` for the structure) — replace it, run `seed_taxonomy`, and the classifier will respect your domains and collections going forward.

## Configuration

### LLM provider and model

The classifier is the **only** LLM call in the ingest pipeline. Input is capped (~8k of paper prose, ~10k chars of blog or README) so per-ingest token cost is predictable and small.

Provider and model selection is persisted in **`~/.config/lodestone/config.toml`** (XDG-compliant; `%APPDATA%\lodestone\config.toml` on Windows). Resolution order on every CLI / MCP call:

1. **Config file exists?** Read provider + model from it. Done.
2. **No config, only one provider API key set?** Silently pick that provider and the catalog's default model, write the config.
3. **No config, multiple provider API keys set, TTY available?** Prompt the user (provider menu, then model menu), write the config.
4. **No config, multiple keys set, no TTY?** Raise `ProviderAmbiguous` with instructions to write the config by hand. The MCP server hits this path because it can't prompt.

This means the first CLI ingest is what establishes the config for everything downstream. From plugin users, the bundled `/deep-sota` `validate-env.sh` walks you through the same picker before the first MCP call. From a bare standalone install, run any CLI ingest first to seed the config — or write the TOML directly.

To change provider or model later, edit `~/.config/lodestone/config.toml`:

```toml
[llm]
provider = "anthropic"   # "anthropic" | "openai" | "gemini"
model = "claude-opus-4-7"
temperature = 0.2
```

Model catalogs (with descriptions and the default at index 0) live in `_system/llm/<provider>_adapter.py`. Custom model IDs not in the catalog are accepted — the picker will take whatever you paste.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `LODESTONE_DB` | Path to the SQLite file. Plugin always uses `~/.lodestone/lodestone.db` regardless of this var. |
| `ANTHROPIC_API_KEY` | Provider key for Claude classification. |
| `OPENAI_API_KEY` | Provider key for OpenAI classification. |
| `GEMINI_API_KEY` | Provider key for Gemini classification. |
| `XDG_CONFIG_HOME` | Overrides the config directory root (default `~/.config`). |
| `MCP_TIMEOUT` | MCP server startup timeout (default 30 s). |
| `MCP_TOOL_TIMEOUT` | Per-tool-call timeout (ms). Read by some MCP clients. Ingest streams `notifications/progress` to signal activity — set generously (e.g. 1800000 = 30 min) if your client kills long calls. |

Provider selection itself is **not** driven by env-var precedence — it's driven by `~/.config/lodestone/config.toml`. Set whatever provider keys you want; the first CLI ingest (or `/deep-sota` `validate-env.sh`) picks one and persists the choice. See [LLM provider and model](#llm-provider-and-model) above for the full resolution order.

LLM adapter and httpx timeouts are capped server-side so a wedged downstream request can't hang the MCP server post-sleep.

## Requirements

- Claude Code (or any MCP client)
- Python ≥ 3.11
- `uv` package manager
- `git` on PATH (for repo ingestion)
- (Optional) An LLM API key for classification

### Python Dependencies

Managed via `pyproject.toml`:
- `arxiv`, `httpx`, `beautifulsoup4`, `lxml`, `Pillow` — fetch and convert
- `pymupdf4llm`, `pylatexenc`, `trafilatura`, `htmldate` — PDF/LaTeX/HTML extraction
- `sentence-transformers`, `gliner2`, `sqlite-vec`, `rapidfuzz` — embeddings, NER, vector search, fuzzy matching
- `pydantic>=2`, `tenacity`, `pyyaml`, `tomli-w` — modeling, retries, config
- `anthropic`, `openai`, `google-genai` — LLM SDKs

## Best Practices

1. **Seed the taxonomy before your first ingest.** Run `seed_taxonomy` against the shipped `taxonomy.json` (or your own) so the classifier has a stable scaffold to slot the first dozen papers into. Skipping this is fine — the classifier will mint domains and collections from scratch — but you'll spend the early corpus correcting near-duplicates. See [Seeding the Taxonomy](#seeding-the-taxonomy).

2. **Ingest one or two papers in your area before asking anything.** An empty corpus can't answer research questions, and [`/deep-sota`](https://github.com/piercelamb/deep-sota) won't auto-fetch papers behind your back. Hand it a couple of relevant arXiv URLs first, then start asking — the corpus grows naturally from there as `/deep-sota` surfaces gaps and hands you `ingest_paper` calls.

3. **Drive everything through `/deep-sota`.** It's the intended front-end for both ingest (`/deep-sota <url>`) and research (`/deep-sota "<question>"`). It already knows the right tool sequence, the right order, and when to stop. Reaching past it into individual MCP tools is a fallback for the rare case where `/deep-sota` doesn't fit (e.g. headless scripting or non-Claude-Code clients), not the default.

4. **Back up `~/.lodestone/lodestone.db`.** It's a single file. `cp lodestone.db lodestone.db.bak` before destructive operations is cheap insurance.

5. **Re-ingest with `force` only when you must.** It re-runs the whole pipeline including the LLM classify call — costs tokens and rewrites the citation graph.

## Security & Privacy

### What lodestone does with your data

- **Ingested papers, posts, and repos** are stored locally in `~/.lodestone/lodestone.db`. Nothing leaves your machine except:
- **LLM classification calls** during ingest, which send paper abstracts (and small section snippets) to whichever LLM provider you configured.
- **arXiv, ar5iv, and GitHub fetches** during ingest, subject to those sites' rate limits.

### API key handling

API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) are:
- Read from the environment by the relevant SDK adapter only
- Never logged or transmitted by lodestone code
- Passed directly to the official SDKs

### Rate limiting

The arxiv metadata API is gated behind a file-locked 3.1 s process-wide throttle (arxiv's ToS minimum) with a 429-only retry policy that honors `Retry-After`. Transport errors are not retried — they signal IP-level escalation, and retrying deepens the offense.

## Troubleshooting

### "uv not found"

**Issue**: The plugin entrypoint can't locate `uv`.
**Solution**: Install `uv` from [astral.sh/uv](https://astral.sh/uv) and ensure it's on `PATH`. The plugin won't fall back to system Python.

### "DB inode changed since startup"

**Issue**: The DB file at the canonical path was unlinked and recreated under a running server (usually because `reset_db` ran while the plugin was attached).
**Solution**: Restart Claude Code. The plugin re-pins the new inode on startup.

### Ingest hangs after laptop sleep

**Issue**: A pre-cap LLM adapter or httpx call was wedged across the suspend.
**Solution**: Already fixed — LLM adapter and httpx timeouts are capped server-side. If you see this on an older install, `/plugin update lodestone` and restart Claude Code.

### `mcp__lodestone__*` tools don't appear after install

**Issue**: The plugin shows as installed but `mcp__lodestone__*` tools never reach the session.

**Solution**: Run **`/lodestone:doctor`** — it checks the common causes and prints a fix line per failure. Most common causes:
- Prewarm still running on the first session — wait 60–90s for the `[lodestone] Dependency install complete.` line, then `/reload-plugins`.
- `uv` missing from `PATH` — install from [astral.sh/uv](https://astral.sh/uv) and restart Claude Code.

### "PDF fallback triggered" on every paper

**Issue**: arxiv.org/html and ar5iv are throttling or returning 5xxs.
**Solution**: Wait a few minutes. The 4-tier fallback (html → ar5iv → LaTeX source → PDF) means ingest still succeeds, just slower.

### Search returns "fallback: OR"

**Issue**: Your bare-token query produced no AND hits, so the parser auto-retried with OR-joining.
**Solution**: This is informational, not an error. The hits are still relevant — the flag tells you the query was loosened.

## Developing on lodestone

If you're hacking on this repo (not just using the installed plugin), keep your dev experiments out of the canonical database. There are two deliberately distinct setups, and **you should never have both active in the same Claude Code session**:

| | Plugin (prod) | Project (dev) |
|---|---|---|
| Source | `/plugin install lodestone` | clone of this repo |
| MCP server name | `lodestone` | `lodestone-dev` |
| Tool prefix | `mcp__lodestone__*` | `mcp__lodestone-dev__*` |
| DB path | `$HOME/.lodestone/lodestone.db` | `./lodestone.db` (repo-local) |
| Code | last installed plugin version | live working tree |

**Setup for dev work:**

1. Clone the repo and `uv sync`.
2. `cp .mcp.dev.json.example .mcp.dev.json` and replace `<REPO_ROOT>` with the absolute path to your clone. (`.mcp.dev.json` is gitignored — it carries absolute paths and shouldn't be shared.)
3. Launch Claude Code in the dev workspace with `claude --mcp-config .mcp.dev.json --strict-mcp-config`. `--strict-mcp-config` suppresses the tracked `.mcp.json` at the repo root (which is the plugin's entrypoint and would otherwise also try to register a `lodestone` server in the workspace). With `--mcp-config .mcp.dev.json`, only `lodestone-dev` registers.
4. Also **disable the `lodestone` marketplace plugin for this workspace** (`/plugin` → disable). Otherwise the plugin's prewarm + MCP server still run for this workspace, and you can get `lodestone` and `lodestone-dev` both registered at once.
5. Use `mcp__lodestone-dev__*` tools or `uv run python -m _system.scripts.ingest` for everything in the dev workspace. They both write to `./lodestone.db` only.
6. Open a *different* terminal/workspace (e.g. `cd ~ && claude`) when you want to query the prod DB; that workspace has the plugin enabled and no `.mcp.dev.json`, so `mcp__lodestone__*` is the only lodestone surface and it always points at `~/.lodestone/lodestone.db`.

## Testing

```sh
uv run pytest                           # full suite
uv run pytest -n 8                      # parallel
uv run pytest -m "not slow"             # skip tests that load heavy ML models
uv run pytest _system/tests/test_search.py -v
```

## Project Structure

```
lodestone/
├── .claude-plugin/
│   ├── plugin.json              # Plugin metadata
│   └── marketplace.json         # piercelamb-plugins marketplace listing
├── bin/
│   ├── lodestone-mcp-plugin.sh  # Plugin MCP entrypoint (exec only — fast startup)
│   └── lodestone-prewarm.sh     # SessionStart hook: hash-gated `uv sync`
├── _system/
│   ├── config/                  # Pipeline config
│   ├── db/                      # Schema, migrations, cascade, orphan GC
│   ├── html/                    # LaTeXML / ar5iv parsing
│   ├── latex/                   # LaTeX-source fallback
│   ├── pdf/                     # PDF fallback (pymupdf4llm)
│   ├── llm/                     # Multi-provider LLM adapters + prompts
│   ├── resolution/              # 5-tier term canonicalization + embeddings
│   ├── schemas/                 # Pydantic models
│   ├── scripts/
│   │   ├── ingest.py            # Pipeline orchestrator
│   │   ├── mcp_server.py        # 21-tool MCP server (stdio + HTTP)
│   │   ├── search.py            # Search modes (BM25, multi, taxonomy)
│   │   ├── fetch_paper.py       # arxiv fetch + 4-tier fallback
│   │   ├── convert_paper.py     # LaTeXML → markdown + references
│   │   ├── classify_paper.py    # LLM domain/collection classification
│   │   ├── classify_repo.py     # Repo classification
│   │   ├── fetch_repo.py        # Repo clone + filter
│   │   ├── index_paper.py       # BM25 + sqlite-vec indexing
│   │   ├── extract_entities.py  # GLiNER NER
│   │   ├── seed_taxonomy.py     # Bulk-seed taxonomy.json
│   │   ├── reset_db.py          # Dev DB nuke
│   │   └── sync_plugin.py       # Dev → local plugin install sync
│   ├── tests/                   # ~1300 tests
│   └── utils/                   # Shared helpers
├── lodestone/
│   └── AGENTS.md                # Subagent docs
├── taxonomy.json                # Seed taxonomy
├── taxonomy_tree.md             # Rendered taxonomy reference
├── pyproject.toml               # Python dependencies
├── .mcp.json                    # Plugin MCP entrypoint (uses ${CLAUDE_PLUGIN_ROOT})
├── .mcp.dev.json.example        # Dev MCP config template (copy → .mcp.dev.json)
├── hooks/
│   └── hooks.json               # SessionStart prewarm hook
├── skills/
│   └── doctor/
│       ├── SKILL.md             # /lodestone:doctor diagnostic
│       └── scripts/             # bundled venv/install-hash checks
├── LICENSE                      # Apache-2.0
├── CHANGELOG.md
└── README.md                    # This file
```

## Contributing

Contributions welcome! Please:

1. Clone the repository and set up the dev workflow (see [Developing on lodestone](#developing-on-lodestone))
2. Create a feature branch
3. Run tests: `uv run pytest -n 8`
4. Submit a pull request

## License

[Apache-2.0](./LICENSE)

## Author

Pierce Lamb

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Version

0.1.0
