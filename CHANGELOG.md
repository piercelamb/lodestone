# Changelog

All notable changes to lodestone will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.2] - 2026-06-17

### Fixed
- HTML fetch now detects arxiv/ar5iv pages that return HTTP 200 with a
  LaTeXML "Fatal error" body (a failed render served as `text/html` on the
  `/html/` path) and treats them as no-rendering, so ingest falls through
  to the LaTeX e-print / PDF fallback ladder instead of indexing the error
  banner as the paper. Previously such papers landed in the "Corrupted Or
  Unavailable Paper Content" collection with a single error-stub section.

## [0.3.1] - 2026-06-05

### Fixed
- Figure downloads now retry against the ar5iv mirror
  (`/html/{id}/assets/{path}`) when arxiv.org's native HTML references
  assets it never materialized (page serves 200, figure srcs 404) —
  previously such papers hard-failed ingest at the convert stage with
  `FigureCountMismatch` and could not be recovered with `--force`.

## [0.3.0] - 2026-06-05

### Added
- `ingest --domain <name>` (and `mcp__lodestone__ingest_paper/_repo/_post`
  `domain` arg) locks the classifier to a single domain: only that
  domain's subtree is rendered to the LLM, the schema enum pins
  `domain_index=[0]`, and the collection enum is the override domain's
  full list (no 30-cap truncation). A not-yet-existing override domain
  is created in the success transaction with `description=NULL` and
  `needs_review=1` so it lands in the same review queue as
  LLM-proposed new domains.
- Override accepts display form OR slug — sanitization is enforced at
  every operator entry point (`ingest`, `classify_paper/_repo` CLIs,
  `fetch_paper/_acl/_post` standalone CLIs, and the three MCP
  dispatchers).

### Changed
- Promoted "Conversation Understanding" to a top-level taxonomy domain
  with 9 collections (transcript engineering, dialogue
  segmentation/summarization, coreference, dialogue acts, knowledge
  extraction, dialogue-grounded retrieval, discourse structure,
  conversational AI foundations). Deprecated the two placeholder
  collections under Retrieval And RAG / Document Understanding And OCR.
  `taxonomy_tree.md` regenerated.

### Fixed
- MCP `_sanitized_domain_arg` now rejects non-string `domain` args with
  a clean `ValueError` (previously raised `AttributeError` on
  `.lower()`) and runs before `check_models()` so bad input fails fast
  instead of paying the full HF model verification + preload cost.
- `classify_repo` ORPHAN short-circuits (no-README / empty-README) now
  run AFTER override sanitization, so an invalid override raises
  instead of silently marking ORPHANED. A valid override on an ORPHAN
  repo logs a warning rather than being silently dropped.

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
