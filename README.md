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

## Install

```sh
git clone <this-repo> lodestone
cd lodestone
uv sync
```

`lodestone-mcp` is then available on your PATH inside the venv (e.g.
`./.venv/bin/lodestone-mcp`).

## Build the database

The repo ships with a prebuilt `lodestone.db` you can use directly. To
ingest your own:

```sh
uv run python -m _system.scripts.ingest --url https://arxiv.org/abs/2305.10601
uv run python -m _system.scripts.ingest --post https://example.com/some-blog-post
uv run python -m _system.scripts.ingest --repo https://github.com/owner/name
```

## Connect an MCP client

`lodestone-mcp` supports two transports. Stdio is the default and works
out of the box on every MCP client. HTTP is opt-in (`--http`) and exists
mainly as a workaround for [Claude Code #51736][cc51736] — a regression
that silently drops tools from user-configured stdio MCP servers on
Claude Code 2.1.116+. HTTP-transport servers are unaffected.

### Stdio (Claude Desktop, Cursor, Cline, Zed, Claude Code ≤ 2.1.115)

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

### HTTP (Claude Code 2.1.116+)

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

[cc51736]: https://github.com/anthropics/claude-code/issues/51736
