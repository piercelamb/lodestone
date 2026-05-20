#!/usr/bin/env bash
# Print the lodestone prewarm install hash if present. Resolves the plugin
# data dir via $CLAUDE_PLUGIN_DATA if set, else tries the known fallback
# naming conventions Claude Code has used. Output is informational: hash
# means a uv sync has succeeded for the current pyproject/lock.
set -uo pipefail
PD="${CLAUDE_PLUGIN_DATA:-}"
if [[ -z "$PD" ]]; then
  for d in \
    "$HOME/.claude/plugins/data/lodestone@piercelamb-plugins" \
    "$HOME/.claude/plugins/data/lodestone-piercelamb-plugins" \
    "$HOME/.claude/plugins/data/piercelamb-plugins/lodestone"
  do
    if [[ -d "$d" ]]; then PD="$d"; break; fi
  done
fi
if [[ -z "$PD" ]]; then
  echo "no plugin data dir located"
elif [[ -f "$PD/install.hash" ]]; then
  cat "$PD/install.hash"
else
  echo "no install hash yet — prewarm has not completed a successful sync"
fi
