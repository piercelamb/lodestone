#!/usr/bin/env bash
# Informational check: report whether the canonical lodestone DB exists.
# Lives in a script (instead of an inline `!` injection in SKILL.md) so the
# Claude Code permission validator sees a static path — it rejects bash
# commands whose arguments contain `$HOME` or other untracked-variable
# expansion, even when `Bash(stat:*)` is in `allowed-tools`. See issue #74.
set -uo pipefail
db="$HOME/.lodestone/lodestone.db"
if [[ -e "$db" ]]; then
  stat -L "$db" 2>&1
else
  echo "no DB yet (created on first ingest)"
fi
