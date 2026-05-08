# Lodestone

A personal arXiv research knowledge base in a single SQLite file. Ingest
papers (and HTML blog posts, and GitHub repos), then search/lookup/read
them — and the figures inside them — over MCP from Claude Code, Claude
Desktop, Cursor, Cline, or anything else that speaks the protocol.

## What you get

- A SQLite file (`lodestone.db`) containing markdown-converted papers,
  extracted figures (as inline image blobs), per-paper section trees,
  per-paper code-repo trees, classifier-derived domain/collection
  taxonomy, BM25 + vector indexes, and forward/backward arXiv citations.
- An MCP server (`lodestone-mcp`) exposing 20 tools — `search`, `bm25`,
  `lookup`, `browse`, `overview`, `collection`, `toc`, `toc_many`,
  `read`, `figure`, `repo_tree`, `read_code`, `citations`, `tables`,
  `schema`, `taxonomy_lookup`, `search_multi`, `repo`, plus
  `ingest_paper` / `ingest_post` / `ingest_repo`.

## Quickstart (Claude Code)

Lodestone ships as a [Claude Code plugin][cc-plugins]. Two commands and
you're done:

```
/plugin marketplace add piercelamb/deep-plan
/plugin install lodestone
```

Restart Claude Code. The `mcp__lodestone__*` tools are now available
(`search`, `read`, `ingest_paper`, …).

**Requirements:** [`uv`](https://astral.sh/uv) on `PATH`. On first
invocation the plugin runs `uv sync` to materialize its venv (~30s–2min
once; near-instant after that). HuggingFace embedding weights download
lazily on the first `ingest_*` call, not on startup.

**Database location:** by default the SQLite file lives at
`~/.lodestone/lodestone.db` so it survives `/plugin update` cache
refreshes. Override with the `LODESTONE_DB` env var if you need
something else.

**First steps:** the database is empty after a fresh install. Call
`mcp__lodestone__ingest_paper` (or `ingest_post` / `ingest_repo`) before
search/read tools will return anything useful.

## Build the database from the CLI

For bulk ingest, or if you've cloned the repo for development:

```sh
uv sync
uv run python -m _system.scripts.ingest --url https://arxiv.org/abs/2305.10601
uv run python -m _system.scripts.ingest --post https://example.com/some-blog-post
uv run python -m _system.scripts.ingest --repo https://github.com/owner/name
```

## Manual setup (other MCP clients)

If you're not on Claude Code — Claude Desktop, Cursor, Cline, Zed,
custom clients — you can wire `lodestone-mcp` up directly. Clone the
repo and `uv sync` first so `lodestone-mcp` is on the venv's PATH.

`lodestone-mcp` supports two transports. Stdio is the default and works
out of the box on every MCP client *except* Claude Code 2.1.116+, which
silently drops tools from user-configured stdio servers
([#51736][cc51736]). The plugin install above sidesteps that regression
because plugin-managed stdio servers are unaffected. Other clients can
use stdio without issue. HTTP is opt-in (`--http`) and remains useful as
a workaround when stdio isn't an option.

### Stdio

```jsonc
{
  "mcpServers": {
    "lodestone": {
      "command": "lodestone-mcp",
      "env": { "LODESTONE_DB": "/abs/path/to/lodestone.db" }
    }
  }
}
```

### HTTP

Run the server in its own terminal:

```sh
LODESTONE_DB=/abs/path/to/lodestone.db lodestone-mcp --http --port 8765
```

Then point the client at it:

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

The HTTP transport binds to `127.0.0.1` by default. There is no auth or
TLS — don't expose it publicly without putting something in front.

## Known issues

- **Claude Code 2.1.116+ silently drops stdio MCP tools** —
  [anthropics/claude-code#51736][cc51736]. The server connects, the
  JSON-RPC handshake succeeds, `tools/list` returns the full registry,
  but tools never appear in the assistant's deferred-tool registry. Use
  the HTTP transport on affected versions.

[cc-plugins]: https://docs.claude.com/en/docs/claude-code/plugins
[cc51736]: https://github.com/anthropics/claude-code/issues/51736
