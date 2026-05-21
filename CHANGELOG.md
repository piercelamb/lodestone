# Changelog

All notable changes to lodestone will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-05-21

### Security
- Added `exclude-newer = "7 days"` under `[tool.uv]` in `pyproject.toml` so new
  PyPI version resolutions skip releases published in the last 7 days. Gives
  the security community a window to flag supply-chain compromises before they
  land in our dependency tree. Existing locked dependencies are unaffected.
