# Changelog

All notable changes to lodestone will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.2] - 2026-05-23

### Fixed
- `/lodestone:doctor` failing immediately with a `Shell command permission
  check failed` error on the Canonical DB check. Claude Code's permission
  validator now statically rejects `!`-injected bash blocks whose path
  arguments contain an untracked variable like `$HOME` — even when
  `Bash(mkdir:*)` is in `allowed-tools`. Moved the check into
  `skills/doctor/scripts/check-db.sh` (called via the tracked
  `${CLAUDE_SKILL_DIR}` variable, matching the other six checks). Added
  a test guard so any future SKILL.md `!`-injection that references an
  untracked variable fails CI instead of users (#74).

## [0.1.1] - 2026-05-21

### Security
- Added `exclude-newer = "7 days"` under `[tool.uv]` in `pyproject.toml` so new
  PyPI version resolutions skip releases published in the last 7 days. Gives
  the security community a window to flag supply-chain compromises before they
  land in our dependency tree. Existing locked dependencies are unaffected.
