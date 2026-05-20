#!/usr/bin/env bash
# Verify the plugin venv binary exists. Resolves the plugin root via
# self-discovery from this script's location (the skill lives at
# <plugin-root>/skills/doctor/scripts/, so two levels up is the plugin root).
set -uo pipefail
_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
plugin_root="$(cd "$_dir/../../.." && pwd)"
binary="$plugin_root/.venv/bin/lodestone-mcp"
if [[ -e "$binary" ]]; then
  ls -la "$binary"
else
  echo "venv binary missing at $binary"
fi
