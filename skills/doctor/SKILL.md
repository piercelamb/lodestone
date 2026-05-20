---
description: Diagnose lodestone install issues — uv presence, venv state, MCP registration, DB writability, prewarm status, HF model cache. Use when mcp__lodestone__* tools are missing, when /lodestone:doctor is invoked, or after a fresh /plugin install lodestone.
disable-model-invocation: true
allowed-tools:
  - Bash(command:*)
  - Bash(bash:*)
  - Bash(mkdir:*)
  - Bash(stat:*)
  - Bash(claude:*)
  - Bash(grep:*)
  - Bash(test:*)
---

# /lodestone:doctor

Diagnose the lodestone install. Each section below runs a check; interpret the output and print one PASS/FAIL line per check, then a single fix line per FAIL. Be terse — no preamble, no recap. After diagnostics, if the **Plugin venv** check FAILed, follow the remediation section. Finish with a one-line overall verdict.

## Diagnostics

### `uv` on PATH

!`command -v uv 2>&1 || echo "uv not on PATH"`

Required for the prewarm hook and venv. If missing, fix line is: `install uv from https://astral.sh/uv and restart Claude Code`.

### Plugin venv

!`bash "${CLAUDE_SKILL_DIR}/scripts/check-venv.sh"`

If missing, the prewarm hook hasn't completed. Most common cause: `/plugin install lodestone` was run mid-session and `/reload-plugins` doesn't fire SessionStart hooks. Fix line: `Claude will run the prewarm hook now — see Remediation below; after it completes run '/reload-plugins'`.

### Install hash

!`bash "${CLAUDE_SKILL_DIR}/scripts/check-install-hash.sh"`

Informational. Presence of a hash means a `uv sync` has succeeded for the current pyproject/lock.

### Canonical DB directory

!`mkdir -p "$HOME/.lodestone" && stat -L "$HOME/.lodestone/lodestone.db" 2>&1 || echo "no DB yet (created on first ingest)"`

Informational. Empty DB is fine — `mcp__lodestone__overview` works against an empty corpus.

### MCP server registration

!`claude mcp list 2>&1 | grep -i lodestone || echo "lodestone not in 'claude mcp list' output"`

If absent: the plugin's root `.mcp.json` didn't register. Fix line: `confirm '/plugin install lodestone' completed and the plugin is enabled (/plugin), then restart Claude Code`.

### HuggingFace model cache

!`test -d "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5" && echo "bge cached" || echo "bge NOT cached"`
!`test -d "$HOME/.cache/huggingface/hub/models--fastino--gliner2-large-v1" && echo "gliner cached" || echo "gliner NOT cached"`

Informational — NOT a FAIL. The two CPU-only HuggingFace models (`BAAI/bge-small-en-v1.5` ~133 MB for embeddings, `fastino/gliner2-large-v1` ~285 MB for entity extraction) download on the first ingest with live `notifications/progress` streaming, so this isn't blocking. If you'd rather eat the ~400 MB download upfront, ask Claude to run the prefetch via the Bash tool: `uv run --directory "${CLAUDE_SKILL_DIR}/../.." --no-sync python -m _system.scripts.validate_models`.

## Remediation

**Skip this section unless the Plugin venv check above FAILed.** Otherwise no action is needed.

Claude — to fix a missing venv, invoke the **Bash tool** (not the `!` injection mechanism) with:

```sh
bash "${CLAUDE_SKILL_DIR}/../../bin/lodestone-prewarm.sh"
```

The prewarm script self-discovers its plugin root from its own location, so no env-var setup is needed. If it exits non-zero, surface the error and stop — likely `uv` is not on PATH or `pyproject.toml`/`uv.lock` is missing.

After the prewarm finishes (look for `[lodestone] Dependency install complete.`), tell the user to fully quit and relaunch Claude Code (not `/reload-plugins`) so the MCP server registers against the populated venv.

## Verdict

After printing PASS/FAIL per check, the fix line per FAIL, and (if applicable) the remediation result, print one final line:

- All PASS → `lodestone looks healthy — if mcp__lodestone__* tools still don't appear, restart Claude Code once more (full quit, not /reload-plugins)`.
- Venv was FAILed and remediation succeeded → `prewarm completed — **fully quit and relaunch Claude Code** (not /reload-plugins). On the next launch the lodestone MCP server will find the populated venv and mcp__lodestone__* tools will register`.
- Any other FAIL → `fix the FAILs above (top to bottom), then re-run /lodestone:doctor`.
