#!/usr/bin/env bash
# SessionStart prewarm hook for the lodestone plugin.
#
# Hash-gates `uv sync` so the 60-90s first-install cost happens once, in
# the hook (which has a 600s timeout), not in the MCP-server hot path
# (which the client kills at ~30s). Subsequent sessions: pyproject/lock
# unchanged → silent exit 0, near-instant. Stdout is injected into
# Claude's session context (per Claude Code hook docs), so a one-line
# status appears in the user's session when sync actually runs.
set -euo pipefail

# Resolve plugin context. Claude Code's hook runner exports both vars
# (and that's the production path), but /lodestone:doctor remediation
# also invokes this script via the Bash tool in a session that doesn't
# have them set. Self-discovery from the script's own location handles
# that case correctly:
#   - PLUGIN_ROOT is `dirname(dirname(<this script>))` — the script
#     lives at <plugin-root>/bin/lodestone-prewarm.sh by construction.
#   - PLUGIN_DATA defaults to the @-separated convention Claude Code
#     uses today (~/.claude/plugins/data/<plugin>@<marketplace>).
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CLAUDE_PLUGIN_ROOT:=$(cd "$_script_dir/.." && pwd)}"
: "${CLAUDE_PLUGIN_DATA:=$HOME/.claude/plugins/data/lodestone@piercelamb-plugins}"
export CLAUDE_PLUGIN_ROOT CLAUDE_PLUGIN_DATA

if ! command -v uv >/dev/null 2>&1; then
  echo "[lodestone prewarm] uv not found on PATH. Install from https://astral.sh/uv and run /lodestone:doctor." >&2
  exit 1
fi

mkdir -p "$CLAUDE_PLUGIN_DATA"

hash_file="$CLAUDE_PLUGIN_DATA/install.hash"
pyproject="$CLAUDE_PLUGIN_ROOT/pyproject.toml"
lockfile="$CLAUDE_PLUGIN_ROOT/uv.lock"

if [[ ! -f "$pyproject" ]] || [[ ! -f "$lockfile" ]]; then
  echo "[lodestone prewarm] missing pyproject.toml or uv.lock under $CLAUDE_PLUGIN_ROOT" >&2
  exit 1
fi

# sha256(pyproject + lock). Concatenation is fine — collisions across
# realistic edits are astronomically unlikely.
current_hash="$(cat "$pyproject" "$lockfile" | shasum -a 256 | awk '{print $1}')"
stored_hash=""
if [[ -f "$hash_file" ]]; then
  stored_hash="$(cat "$hash_file")"
fi

if [[ "$current_hash" == "$stored_hash" ]]; then
  exit 0
fi

echo "[lodestone] First-time dependency install — running 'uv sync' (~30-90s)."

cd "$CLAUDE_PLUGIN_ROOT"
if ! uv sync; then
  echo "[lodestone prewarm] uv sync failed. Run /lodestone:doctor for diagnostics." >&2
  exit 1
fi

echo "$current_hash" > "$hash_file"
echo "[lodestone] Dependency install complete. If lodestone MCP tools don't appear in this session, fully quit and relaunch Claude Code — Claude Code launches MCP servers in parallel with this hook and the lodestone server may have already failed against the empty venv. The canonical install flow runs /lodestone:doctor before the first restart to avoid this entirely."
