---
description: Diagnose lodestone install issues — uv presence, venv state, MCP registration, DB writability, prewarm status, HF model cache, LLM provider config. Use when mcp__lodestone__* tools are missing, when /lodestone:doctor is invoked, or after a fresh /plugin install lodestone.
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
  - Read
  - Write
  - WebSearch
  - AskUserQuestion
---

# /lodestone:doctor

Diagnose the lodestone install. Each section below runs a check; interpret the output and print one PASS/FAIL line per check, then a single fix line per FAIL. Be terse — no preamble, no recap. After diagnostics, run any applicable remediations in the order described in **Remediation order** below. Finish with a one-line overall verdict.

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

!`bash "${CLAUDE_SKILL_DIR}/scripts/check-hf-cache.sh"`

Two CPU-only HuggingFace models are required at ingest time: `BAAI/bge-small-en-v1.5` (~133 MB, embeddings) and `fastino/gliner2-large-v1` (~285 MB, entity extraction). The script uses `huggingface_hub.snapshot_download(local_files_only=True)` to verify completeness — same call path the loader libraries use, just with the network door closed — so partial downloads and missing variant files surface as "NOT cached" rather than false-positive PASSing on directory existence alone. If both report "cached", PASS. If either reports "NOT cached" (or "cached (loose check…)" when the venv isn't ready yet — that's a weaker check), FAIL — fix line: `Claude will download the missing model(s) now — see Model download remediation below`.

### LLM provider config

!`bash "${CLAUDE_SKILL_DIR}/scripts/check-config.sh"`

The output is a single JSON line. Parse `config_status`:

- `"ok"` → PASS (config exists, provider+model set, matching env var present).
- `"missing"` with `providers_with_keys: []` → FAIL. Fix line: `set one of $OPENAI_API_KEY, $ANTHROPIC_API_KEY, or $GEMINI_API_KEY in your shell, then re-run /lodestone:doctor`. **Stop — no remediation possible.**
- `"missing"` or `"incomplete"` with at least one key present → FAIL. Fix line: `Claude will walk you through provider/model selection — see Provider + model setup remediation below`.
- `"malformed"` → FAIL. Fix line: `${config_path} is malformed TOML — open it manually and fix the syntax, then re-run /lodestone:doctor`. **Stop — no remediation possible.**
- `"key_missing"` → FAIL. Fix line: `config pins provider=${config_provider} but its env var is not set in this shell — either export the key or edit ${config_path} to pick a different provider, then re-run /lodestone:doctor`. **Stop — no remediation possible.**

## Remediation order

Run remediations in this order. Skip any whose check PASSed.

1. **Plugin venv first** (if FAILed). Everything downstream needs a working venv, so this is sequential.
2. **HF cache + Provider config together** (parallel when both FAILed):
   - **Both FAILed:** invoke the **Bash tool with `run_in_background: true`** to kick off the HF download (using the `--models-only` flag — see *Remediation — Model download*). Immediately run the Provider + model wizard while the bytes stream in. After writing `config.toml`, **wait for the background Bash to complete** — you'll be notified automatically; do not poll.
   - **Only HF FAILed:** run *Remediation — Model download* in the foreground.
   - **Only Provider config FAILed:** run *Remediation — Provider + model setup*.

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

Before invoking the Bash tool, tell the user:

- **Foreground (HF only):** `Downloading lodestone models (~400 MB total: BAAI/bge-small-en-v1.5 + fastino/gliner2-large-v1). This can take a few minutes depending on your connection — progress isn't visible here, but the download is running.`
- **Parallel (HF + Provider config both FAILed):** `Starting the ~400 MB model download in the background (BAAI/bge-small-en-v1.5 + fastino/gliner2-large-v1) — while it runs, I need a couple quick choices from you for the LLM config.`

Then invoke the **Bash tool** (not the `!` injection mechanism) with this command:

```sh
uv run --no-sync --directory "${CLAUDE_SKILL_DIR}/../.." python -m _system.scripts.validate_models --models-only
```

**Parallel mode:** when the Provider config check also FAILed, set `run_in_background: true` on the Bash call so the wizard can run while the download streams. Otherwise run in the foreground.

The `--models-only` flag skips the provider-config check inside `validate_models` (doctor's own *LLM provider config* check covers it). The script is idempotent — any already-cached model is a no-op, so a partial cache resumes safely. On success the script prints `<model-id>: present` for both models. If it exits non-zero, surface the error and stop — likely a network or HuggingFace issue.

## Remediation — Provider + model setup

**Skip this section unless the LLM provider config check returned a `"missing"` or `"incomplete"` status with at least one provider key present.** The `"malformed"`, `"key_missing"`, and zero-keys cases are *not* remediable here — they require user action.

Re-read the JSON output from `check-config.sh` (above) to recover `providers_with_keys`, `config_status`, `config_provider`, and `config_path`. Then:

1. Open and follow `references/provider_select.md` to resolve a single `provider`. That reference handles the 0 / 1 / many providers cases via `AskUserQuestion`.
2. Open and follow `references/model_select.md` to pick a model and write `~/.config/lodestone/config.toml`. That reference handles the WebSearch + AskUserQuestion + Write flow.

After the config is written, if you launched the HF download in background, **wait for it to finish** (you'll be notified). Then continue to the verdict.

## Verdict

After printing PASS/FAIL per check, the fix line per FAIL, and (if applicable) any remediation results, print one final line:

- All PASS → `lodestone looks healthy — if mcp__lodestone__* tools still don't appear, open /mcp -> find lodestone -> enable or reconnect; if that doesn't surface them, restart Claude Code once more (full quit, not /reload-plugins)`.
- Venv was FAILed and remediation succeeded → `prewarm completed — open /mcp -> find lodestone -> enable or reconnect to retry against the now-populated venv. If tools still don't register, **fully quit and relaunch Claude Code** (not /reload-plugins)`.
- Only models and/or provider config were FAILed and their remediations succeeded → `lodestone is ready — no restart needed; the next /deep-sota run will use the freshly-written config and cached models`.
- A FAIL hit a "stop, no remediation possible" case (no keys / malformed / key_missing) → `fix the FAIL(s) above per their instructions, then re-run /lodestone:doctor`.
- Any other FAIL whose remediation didn't succeed → `fix the FAILs above (top to bottom), then re-run /lodestone:doctor`.
