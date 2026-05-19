#!/usr/bin/env bash
# Marketplace-installed plugin entrypoint. Always serves the canonical
# DB at $HOME/.lodestone/lodestone.db — no caller-side overrides, no
# walk-up CWD inference. Dev work in a lodestone clone should use the
# repo's own .mcp.dev.json (server name "lodestone-dev") with the plugin
# disabled in that workspace; see .mcp.dev.json.example.
#
# Dependency install runs in the SessionStart prewarm hook
# (bin/lodestone-prewarm.sh), not here — keeping this entrypoint fast
# enough to beat the client's stdio startup timeout.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "[lodestone plugin] uv not found on PATH. Install from https://astral.sh/uv and retry. Run /lodestone:doctor for diagnostics." >&2
  exit 1
fi

if [[ ! -x "$CLAUDE_PLUGIN_ROOT/.venv/bin/lodestone-mcp" ]]; then
  echo "[lodestone plugin] venv missing at $CLAUDE_PLUGIN_ROOT/.venv/bin/lodestone-mcp." >&2
  echo "[lodestone plugin] The SessionStart prewarm hook should have installed it. Run /lodestone:doctor and restart Claude Code." >&2
  exit 1
fi

# Drop any inherited LODESTONE_DB so the plugin can never be redirected
# to a non-canonical path (e.g. a dev repo's lodestone.db) by the parent
# shell or a sibling MCP config. The canonical path is hardcoded.
unset LODESTONE_DB
export LODESTONE_DB="$HOME/.lodestone/lodestone.db"
mkdir -p "$(dirname "$LODESTONE_DB")"

# cd / so _resolve_db_path's walk-up fallback can't accidentally match a
# pyproject.toml in CLAUDE_PLUGIN_ROOT (the plugin install ships its own
# pyproject named "lodestone"). The env var should always win, but
# belt-and-suspenders.
cd /

exec "$CLAUDE_PLUGIN_ROOT/.venv/bin/lodestone-mcp" "$@"
