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
  - Bash(uv run --no-sync --directory:*)
---

# /lodestone:doctor

Diagnose the lodestone install. Each section below runs a check; interpret the output and print one PASS/FAIL line per check, then a single fix line per FAIL. Be terse — no preamble, no recap. After diagnostics, run any applicable remediation sections (venv first if FAILed, then models). Finish with a one-line overall verdict.

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

If absent: the plugin's root `.mcp.json` didn't register. Fix line: `open /mcp -> find lodestone -> enable or reconnect; if not listed, confirm '/plugin install lodestone' completed and the plugin is enabled (/plugin), then restart Claude Code`.

### HuggingFace model cache

!`test -d "$HOME/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5" && echo "bge cached" || echo "bge NOT cached"`
!`test -d "$HOME/.cache/huggingface/hub/models--fastino--gliner2-large-v1" && echo "gliner cached" || echo "gliner NOT cached"`

Two CPU-only HuggingFace models are required at ingest time: `BAAI/bge-small-en-v1.5` (~133 MB, embeddings) and `fastino/gliner2-large-v1` (~285 MB, entity extraction). If both are cached, PASS. If either reports "NOT cached", FAIL — fix line: `Claude will download the missing model(s) now — see Model download remediation below`.

## Remediation — Plugin venv

**Skip this section unless the Plugin venv check above FAILed.**

Claude — to fix a missing venv, invoke the **Bash tool** (not the `!` injection mechanism) with:

```sh
bash "${CLAUDE_SKILL_DIR}/../../bin/lodestone-prewarm.sh"
```

The prewarm script self-discovers its plugin root from its own location, so no env-var setup is needed. If it exits non-zero, surface the error and stop — likely `uv` is not on PATH or `pyproject.toml`/`uv.lock` is missing.

After the prewarm finishes (look for `[lodestone] Dependency install complete.`), tell the user to open `/mcp -> find lodestone -> enable or reconnect` so the MCP server retries against the now-populated venv. If that doesn't surface `mcp__lodestone__*` tools, fully quit and relaunch Claude Code (not `/reload-plugins`).

## Remediation — Model download

**Skip this section unless the HuggingFace model cache check above FAILed.**

Before invoking the Bash tool, tell the user verbatim: `Downloading lodestone models (~400 MB total: BAAI/bge-small-en-v1.5 + fastino/gliner2-large-v1). This can take a few minutes depending on your connection — progress isn't visible here, but the download is running.`

Then invoke the **Bash tool** (not the `!` injection mechanism) with:

```sh
uv run --no-sync --directory "${CLAUDE_SKILL_DIR}/../.." python -m _system.scripts.validate_models
```

The script is idempotent — any already-cached model is a no-op, so a partial cache resumes safely. The Plugin venv check must have PASSed (or its remediation must have already run) for this to work; `--no-sync` skips re-syncing dependencies since prewarm handled that.

On success, the script prints `provider: <name>` and `<model-id>: present` for both models. If it exits non-zero, surface the error and stop — likely a network/HuggingFace issue, or no LLM provider configured in `~/.config/lodestone/config.toml`.

## Verdict

After printing PASS/FAIL per check, the fix line per FAIL, and (if applicable) any remediation results, print one final line:

- All PASS → `lodestone looks healthy — if mcp__lodestone__* tools still don't appear, open /mcp -> find lodestone -> enable or reconnect; if that doesn't surface them, restart Claude Code once more (full quit, not /reload-plugins)`.
- Venv was FAILed and remediation succeeded (models PASS or also remediated) → `prewarm completed — open /mcp -> find lodestone -> enable or reconnect to retry against the now-populated venv. If tools still don't register, **fully quit and relaunch Claude Code** (not /reload-plugins)`.
- Only models were FAILed and remediation succeeded → `models downloaded — lodestone is ready; no restart needed`.
- Any FAIL whose remediation didn't succeed → `fix the FAILs above (top to bottom), then re-run /lodestone:doctor`.
