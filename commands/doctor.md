---
description: Diagnose lodestone install issues — Claude Code version, uv presence, venv state, MCP registration, DB writability, prewarm status, HF model cache
allowed-tools: Bash(uv:*) Bash(claude:*) Bash(command:*) Bash(ls:*) Bash(stat:*) Bash(cat:*) Bash(mkdir:*) Bash(test:*) Bash(bash:*) Bash(grep:*) Bash(sed:*)
---

# /lodestone:doctor

Diagnose the lodestone install. Each section below runs a check; interpret the output and print one PASS/FAIL line per check, then a single fix line per FAIL. Be terse — no preamble, no recap. After diagnostics, if the **Plugin venv** check FAILed, follow the remediation section. Finish with a one-line overall verdict.

> **Note on `$CLAUDE_PLUGIN_ROOT`.** Claude Code exports this for hooks but **not reliably for slash-command `!`-injection shells**. Every `!` injection below that needs the plugin root derives it locally via the same `_lpr()` shell helper: prefer `$CLAUDE_PLUGIN_ROOT`, else parse `installed_plugins.json` to find the installPath. Do not rely on `$CLAUDE_PLUGIN_ROOT` being set in `!` injections.

## Diagnostics

### Claude Code version

!`claude --version 2>&1 || echo "claude CLI not on PATH"`

Required: ≥ 2.1.144 (older versions hit `.mcp.json` regressions and paginated `tools/list` bugs that block lodestone). If older, fix line is: `upgrade with 'npm i -g @anthropic-ai/claude-code' and restart Claude Code`. **This is the most common reason `mcp__lodestone__*` tools never appear even when everything else looks fine.**

### `uv` on PATH

!`command -v uv 2>&1 || echo "uv not on PATH"`

Required for the prewarm hook and venv. If missing, fix line is: `install uv from https://astral.sh/uv and restart Claude Code`.

### Plugin venv

!`_lpr() { if [ -n "${CLAUDE_PLUGIN_ROOT:-}" ]; then echo "$CLAUDE_PLUGIN_ROOT"; return; fi; grep -A2 '"lodestone@piercelamb-plugins"' "$HOME/.claude/plugins/installed_plugins.json" 2>/dev/null | grep '"installPath"' | tail -1 | sed -E 's/.*"installPath"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/'; }; PR="$(_lpr)"; if [ -z "$PR" ]; then echo "could not resolve plugin root (no CLAUDE_PLUGIN_ROOT, no installed_plugins.json entry)"; else ls -la "$PR/.venv/bin/lodestone-mcp" 2>&1 || echo "venv binary missing at $PR/.venv/bin/lodestone-mcp"; fi`

If missing, the prewarm hook hasn't completed. Most common cause: `/plugin install lodestone` was run mid-session and `/reload-plugins` doesn't fire SessionStart hooks. Fix line: `Claude will run the prewarm hook now — see Remediation below; after it completes run '/reload-plugins'`.

### Install hash

!`_lpd() { if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then echo "$CLAUDE_PLUGIN_DATA"; return; fi; for d in "$HOME/.claude/plugins/data/lodestone-piercelamb-plugins" "$HOME/.claude/plugins/data/piercelamb-plugins/lodestone"; do [ -d "$d" ] && { echo "$d"; return; }; done; }; PD="$(_lpd)"; if [ -z "$PD" ]; then echo "no plugin data dir located"; else cat "$PD/install.hash" 2>/dev/null || echo "no install hash yet — prewarm has not completed a successful sync"; fi`

Informational. Presence of a hash means a `uv sync` has succeeded for the current pyproject/lock.

### Canonical DB directory

!`mkdir -p "$HOME/.lodestone" && stat -L "$HOME/.lodestone/lodestone.db" 2>&1 || echo "no DB yet (created on first ingest)"`

Informational. Empty DB is fine — `mcp__lodestone__overview` works against an empty corpus.

### MCP server registration

!`claude mcp list 2>&1 | grep -i lodestone || echo "lodestone not in 'claude mcp list' output"`

If absent: the plugin's root `.mcp.json` didn't register. Most common cause is Claude Code < 2.1.144 (see version check above). Fix line: `upgrade Claude Code to ≥ 2.1.144 and restart; if still absent, confirm '/plugin install lodestone' completed and the plugin is enabled (/plugin)`.

### HuggingFace model cache

!`test -d "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5" && echo "bge cached" || echo "bge NOT cached"`
!`test -d "$HOME/.cache/huggingface/hub/models--fastino--gliner2-large-v1" && echo "gliner cached" || echo "gliner NOT cached"`

Informational — NOT a FAIL. The two CPU-only HuggingFace models (`BAAI/bge-small-en-v1.5` ~133 MB for embeddings, `fastino/gliner2-large-v1` ~285 MB for entity extraction) download on the first ingest with live `notifications/progress` streaming, so this isn't blocking. If you'd rather eat the ~400 MB download upfront, ask Claude to run the prefetch via the Bash tool with the discovered plugin root: `uv run --directory "$PR" --no-sync python -m _system.scripts.validate_models` (substitute `$PR` with the venv path from the Plugin venv check).

## Remediation

**Skip this section unless the Plugin venv check above FAILed.** Otherwise no action is needed.

Claude — to fix a missing venv, invoke the **Bash tool** (not the `!` injection mechanism, which doesn't get `$CLAUDE_PLUGIN_ROOT`) with the following command. It derives the plugin root the same way the diagnostics do, then runs the prewarm script:

```sh
PR="${CLAUDE_PLUGIN_ROOT:-$(grep -A2 '"lodestone@piercelamb-plugins"' "$HOME/.claude/plugins/installed_plugins.json" 2>/dev/null | grep '"installPath"' | tail -1 | sed -E 's/.*"installPath"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')}"
if [ -z "$PR" ] || [ ! -x "$PR/bin/lodestone-prewarm.sh" ]; then
  echo "could not locate prewarm script (PR='$PR')"; exit 1
fi
bash "$PR/bin/lodestone-prewarm.sh"
```

After the prewarm finishes (look for `[lodestone] Dependency install complete.`), tell the user to run `/reload-plugins` to pick up the freshly-installed venv. If prewarm exits non-zero, surface the error and stop — likely `uv` is not on PATH or `pyproject.toml`/`uv.lock` is missing.

## Verdict

After printing PASS/FAIL per check, the fix line per FAIL, and (if applicable) the remediation result, print one final line:

- All PASS → `lodestone looks healthy — if mcp__lodestone__* tools still don't appear, restart Claude Code once more (full quit, not /reload-plugins)`.
- Venv was FAILed and remediation succeeded → `prewarm completed — **fully quit and relaunch Claude Code** (not /reload-plugins). On the next launch the lodestone MCP server will find the populated venv and mcp__lodestone__* tools will register`.
- Claude Code version FAILed → `upgrade Claude Code to ≥ 2.1.144 first, then re-run /lodestone:doctor — version is the most common blocker`.
- Any other FAIL → `fix the FAILs above (top to bottom), then re-run /lodestone:doctor`.
