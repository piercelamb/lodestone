---
description: Diagnose lodestone install issues — Claude Code version, uv presence, venv state, MCP registration, DB writability, prewarm status
allowed-tools: Bash(uv:*) Bash(claude:*) Bash(command:*) Bash(ls:*) Bash(stat:*) Bash(cat:*) Bash(mkdir:*) Bash(test:*)
---

# /lodestone:doctor

Diagnose the lodestone install. Each section below runs a check; interpret the output and print one PASS/FAIL line per check, then a single fix line per FAIL. Be terse — no preamble, no recap. Finish with a one-line overall verdict.

## Diagnostics

### Claude Code version

!`claude --version 2>&1 || echo "claude CLI not on PATH"`

Required: ≥ 2.1.144 (older versions hit `.mcp.json` regressions and paginated `tools/list` bugs that block lodestone). If older, fix line is: `upgrade with 'npm i -g @anthropic-ai/claude-code' and restart Claude Code`.

### `uv` on PATH

!`command -v uv 2>&1 || echo "uv not on PATH"`

Required for the prewarm hook and venv. If missing, fix line is: `install uv from https://astral.sh/uv and restart Claude Code`.

### Plugin venv

!`ls -la "$CLAUDE_PLUGIN_ROOT/.venv/bin/lodestone-mcp" 2>&1 || echo "venv binary missing"`

If missing, the prewarm hook hasn't completed. Fix line: `wait 60-90s for the SessionStart prewarm hook to finish, or run '/reload-plugins' and watch for the '[lodestone] Dependency install complete.' message`.

### Install hash

!`cat "$CLAUDE_PLUGIN_DATA/install.hash" 2>/dev/null || echo "no install hash yet — prewarm has not completed a successful sync"`

Informational. Presence of a hash means a `uv sync` has succeeded for the current pyproject/lock.

### Canonical DB directory

!`mkdir -p "$HOME/.lodestone" && stat -L "$HOME/.lodestone/lodestone.db" 2>&1 || echo "no DB yet (created on first ingest)"`

Informational. Empty DB is fine — `mcp__lodestone__overview` works against an empty corpus.

### MCP server registration

!`claude mcp list 2>&1 | grep -i lodestone || echo "lodestone not in 'claude mcp list' output"`

If absent: the plugin's root `.mcp.json` didn't register. Fix line: `restart Claude Code; if still absent, confirm '/plugin install lodestone' completed and the plugin is enabled (/plugin)`.

## Verdict

After printing PASS/FAIL per check and a fix line per FAIL, print one final line:

- All PASS → `lodestone looks healthy — if mcp__lodestone__* tools still don't appear, restart Claude Code once more`.
- Any FAIL → `fix the FAILs above (top to bottom), then re-run /lodestone:doctor`.
