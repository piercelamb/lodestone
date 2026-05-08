#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "[lodestone plugin] uv not found on PATH. Install from https://astral.sh/uv and retry." >&2
  exit 1
fi

cd "$CLAUDE_PLUGIN_ROOT"
uv sync --quiet

export LODESTONE_DB="${LODESTONE_DB:-$HOME/.lodestone/lodestone.db}"
mkdir -p "$(dirname "$LODESTONE_DB")"

exec "$CLAUDE_PLUGIN_ROOT/.venv/bin/lodestone-mcp" "$@"
