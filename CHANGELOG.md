# Changelog

All notable changes to lodestone will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-06-04

### Added
- `ingest --pdf <path>` for local book PDFs. Default mode outline-splits into
  per-chapter `papers` rows; `--no-split` ingests the whole PDF as one row;
  `--book-slug` + `--chapter-index` handle books whose embedded outline
  doesn't split cleanly (#77).
- `ingest --acl <id-or-url>` for ACL Anthology papers (e.g.
  `2021.acl-long.285`, `P19-1001`, or full URL / `.pdf` / `.xml` / `.bib`
  variants). MODS XML for metadata + pymupdf4llm for body — the Anthology
  has no HTML/LaTeX fulltext (#78).
- `ingest --attach-repo URL --to-paper SLUG_OR_ARXIV` to wire a code repo
  onto an already-indexed paper, for the case where the repo was released
  after the paper landed in lodestone (#79).
- `mcp__lodestone__ingest_paper` now accepts ACL Anthology ids and URLs in
  addition to arxiv. Routes via `parse_acl_id` first; falls back to arxiv on
  `ValueError`. `/deep-sota` can ingest ACL papers through the same MCP
  entry point as arxiv.

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
