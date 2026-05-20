---
description: Diagnose lodestone install issues — Claude Code version, uv presence, venv state, MCP registration, DB writability, prewarm status, HF model cache
allowed-tools: Bash(uv:*) Bash(claude:*) Bash(command:*) Bash(ls:*) Bash(stat:*) Bash(cat:*) Bash(mkdir:*) Bash(test:*) Bash(bash:*)
---

# /lodestone:doctor

Diagnose the lodestone install. Each section below runs a check; interpret the output and print one PASS/FAIL line per check, then a single fix line per FAIL. Be terse — no preamble, no recap. After diagnostics, if the venv check FAILed, run the remediation step. Finish with a one-line overall verdict.

## Diagnostics

### Claude Code version

!`claude --version 2>&1 || echo "claude CLI not on PATH"`

Required: ≥ 2.1.144 (older versions hit `.mcp.json` regressions and paginated `tools/list` bugs that block lodestone). If older, fix line is: `upgrade with 'npm i -g @anthropic-ai/claude-code' and restart Claude Code`.

### `uv` on PATH

!`command -v uv 2>&1 || echo "uv not on PATH"`

Required for the prewarm hook and venv. If missing, fix line is: `install uv from https://astral.sh/uv and restart Claude Code`.

### Plugin venv

!`ls -la "$CLAUDE_PLUGIN_ROOT/.venv/bin/lodestone-mcp" 2>&1 || echo "venv binary missing"`

If missing, the prewarm hook hasn't completed. Most common cause: `/plugin install lodestone` was run mid-session and `/reload-plugins` doesn't fire SessionStart hooks. Fix line: `running prewarm now (see remediation below); after it completes run '/reload-plugins'`.

### Install hash

!`cat "$CLAUDE_PLUGIN_DATA/install.hash" 2>/dev/null || echo "no install hash yet — prewarm has not completed a successful sync"`

Informational. Presence of a hash means a `uv sync` has succeeded for the current pyproject/lock.

### Canonical DB directory

!`mkdir -p "$HOME/.lodestone" && stat -L "$HOME/.lodestone/lodestone.db" 2>&1 || echo "no DB yet (created on first ingest)"`

Informational. Empty DB is fine — `mcp__lodestone__overview` works against an empty corpus.

### MCP server registration

!`claude mcp list 2>&1 | grep -i lodestone || echo "lodestone not in 'claude mcp list' output"`

If absent: the plugin's root `.mcp.json` didn't register. Fix line: `restart Claude Code; if still absent, confirm '/plugin install lodestone' completed and the plugin is enabled (/plugin)`.

### HuggingFace model cache

!`test -d "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5" && echo "bge cached" || echo "bge NOT cached"`
!`test -d "$HOME/.cache/huggingface/hub/models--fastino--gliner2-large-v1" && echo "gliner cached" || echo "gliner NOT cached"`

Informational — NOT a FAIL. The two CPU-only HuggingFace models (`BAAI/bge-small-en-v1.5` ~133 MB for embeddings, `fastino/gliner2-large-v1` ~285 MB for entity extraction) download on the first ingest with live `notifications/progress` streaming, so this isn't blocking. If you'd rather eat the ~400 MB download upfront (e.g. on faster network now), run: `uv run --directory "$CLAUDE_PLUGIN_ROOT" --no-sync python -m _system.scripts.validate_models`.

## Remediation

If the **Plugin venv** check FAILed, run the prewarm hook directly to materialize the venv (the hook normally runs at session start, but `/reload-plugins` does not re-fire SessionStart hooks):

!`bash "$CLAUDE_PLUGIN_ROOT/bin/lodestone-prewarm.sh"`

After the prewarm finishes (look for `[lodestone] Dependency install complete.`), tell the user to run `/reload-plugins` to pick up the freshly-installed venv. If prewarm exits non-zero, surface the error and stop — likely `uv` is not on PATH or `pyproject.toml`/`uv.lock` is missing.

## Verdict

After printing PASS/FAIL per check, the fix line per FAIL, and (if applicable) the remediation result, print one final line:

- All PASS → `lodestone looks healthy — if mcp__lodestone__* tools still don't appear, restart Claude Code once more`.
- Venv was FAILed and remediation succeeded → `prewarm completed — run '/reload-plugins', then mcp__lodestone__* tools should appear`.
- Any other FAIL → `fix the FAILs above (top to bottom), then re-run /lodestone:doctor`.
