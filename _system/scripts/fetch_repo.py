"""Clone a repo and persist its source-file tree.

Pulls a repo's URL from the ``repos`` table, ``git clone --depth 1``
clones it into a tempdir, walks the file tree applying the skip filters
defined below, and persists one row per kept file into ``code_files``.
The repo's top-level README also lands in ``readmes_fts`` as a parallel
BM25 surface.

Filtering is calibrated against what GitHub Linguist, repomix's
``defaultIgnore``, and gitingest's ``DEFAULT_IGNORE_PATTERNS`` converge
on, plus ML-research-repo specifics: notebook-checkpoint dirs,
experiment-tracker artifacts (wandb / mlruns / lightning_logs / hydra /
tensorboard), model-weight extensions (.pt/.safetensors/.gguf/...),
training-data binaries (.parquet/.tfrecord/.npy/...). Defense-in-depth:
size cap, null-byte sniff, UTF-8 replacement-density check, minified-JS
heuristic, ``followlinks=False`` (no symlink escape).

Status outcomes:

- ``REPO_FETCHED`` — clone succeeded.
- ``FAILED_REPO`` — clone failed (404 / network / timeout / non-zero
  git exit). Terminal-but-not-fatal: user can re-run with ``--force``.

No new runtime dependencies — ``git`` is required at the system level.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from _system.db.connection import get_conn, transaction
from _system.db.migrations import init_db
from _system.schemas.repo_metadata import RepoStatus, can_run_from
from _system.utils.logging import get_logger

_LOG = get_logger("scripts.fetch_repo")

_GIT_CLONE_TIMEOUT_S = 120

_DEFAULT_MAX_FILE_BYTES = 1_000_000
_NULL_BYTE_SNIFF_BYTES = 8 * 1024
_UTF8_REPLACEMENT_MAX_RATIO = 0.01
_MINIFIED_AVG_LINE_THRESHOLD = 110
_MINIFIED_SAMPLE_LINES = 100

# Per-extension size ceilings tighter than the global cap. Calibrated
# against the failure mode observed on real arxiv repos: machine-emitted
# JSON/JSONL dumps (per-query BEIR score tables, training-pair logs)
# legitimately fall under 1 MB but contribute zero retrieval value to an
# agent grounding a paper in code. The 50 KB ceiling keeps tiny configs
# (`package.json`, `tsconfig.json`, hyperparam files) while clipping the
# dump tail. Plain source / markdown / notebooks still get the full
# global cap.
_PER_EXT_MAX_BYTES: dict[str, int] = {
    ".json": 50_000,
    ".jsonl": 50_000,
    # CSV/TSV are kept (often eval-result tables / small config matrices)
    # but capped at the same 50 KB as JSON: a multi-MB per-query result
    # table is the same noise-tail problem the JSON cap addresses.
    ".csv": 50_000,
    ".tsv": 50_000,
}


def _max_file_bytes() -> int:
    raw = os.environ.get("LODESTONE_MAX_CODE_FILE_BYTES")
    if raw is None:
        return _DEFAULT_MAX_FILE_BYTES
    try:
        return int(raw)
    except ValueError:
        _LOG.warning(
            "ignoring non-integer LODESTONE_MAX_CODE_FILE_BYTES=%r", raw
        )
        return _DEFAULT_MAX_FILE_BYTES


def _max_bytes_for_ext(ext: str) -> int:
    """Return the size cap for ``ext`` — the per-extension override if
    one exists, else the global cap. ``ext`` is matched case-insensitively.
    """
    capped = _PER_EXT_MAX_BYTES.get(ext.lower())
    if capped is None:
        return _max_file_bytes()
    return min(capped, _max_file_bytes())


# ---------------------------------------------------------------------------
# Skip rules
# ---------------------------------------------------------------------------


# Directory names matched against any path component (case-sensitive).
_SKIP_DIRS: frozenset[str] = frozenset({
    # VCS
    ".git", ".hg", ".svn",
    # Language deps / vendoring
    "node_modules", "bower_components", "vendor", "vendors", "Godeps",
    "__pycache__", ".venv", "venv", "env", "virtualenv", ".bundle",
    # Build / output
    "dist", "build", "out", "target", "bin", "obj",
    ".next", ".nuxt", ".angular", ".expo", ".serverless",
    "cmake-build-debug", "cmake-build-release",
    # Caches
    ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".parcel-cache", ".sass-cache", ".eslintcache", ".nyc_output",
    ".gradle", ".mvn",
    # Coverage / test artifacts
    "coverage", "htmlcov",
    # IDE / editor / OS
    ".idea", ".vscode", ".vs", "__MACOSX",
    # IaC tooling
    ".terraform", ".vagrant",
    # ML-specific (high-value for arxiv repos). `outputs/` and `runs/`
    # are deliberately NOT here — they're often legitimate config/source
    # directories. The size cap covers any large blobs underneath.
    ".ipynb_checkpoints", "wandb", "mlruns", "lightning_logs",
    ".hydra", "tb_logs", "tensorboard_logs",
})


_EGG_INFO_SUFFIX = ".egg-info"


# Lowercased extension compare; matched against the basename's suffix.
_SKIP_EXTS: frozenset[str] = frozenset({
    # Compiled native / object code
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll",
    ".o", ".a", ".obj", ".lib", ".exe",
    ".class", ".jar", ".war", ".tsbuildinfo", ".pdb",
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp",
    ".tiff", ".tif", ".heic", ".heif", ".psd", ".ai", ".eps", ".ico",
    # PDFs / docs
    ".pdf",
    # Audio
    ".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus",
    # Video
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv",
    # Fonts
    ".ttf", ".otf", ".eot", ".woff", ".woff2",
    # Archives / installers / disk images
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    ".iso", ".dmg", ".deb", ".rpm", ".msi", ".whl", ".gem", ".pyz",
    # DBs / committed data blobs
    ".db", ".sqlite", ".sqlite3",
    # ML model weights / serialized artifacts
    ".pt", ".pth", ".ckpt", ".safetensors", ".gguf", ".bin",
    ".onnx", ".tflite", ".pb",
    ".h5", ".hdf5",
    ".pkl", ".pickle", ".joblib", ".dill",
    ".npz", ".npy",
    # Tabular / training-data binaries. .csv/.tsv are intentionally NOT
    # in this list — they're kept (often eval-result tables / small
    # config matrices) but capped via _PER_EXT_MAX_BYTES at 50 KB so
    # multi-MB per-query result dumps land in the same bucket as JSON.
    ".parquet", ".arrow", ".feather", ".tfrecord",
    # Generated / minified web assets (source maps).
    ".map",
    # Editor / OS scratch
    ".swp", ".swo", ".bak", ".tmp",
    # Operational artifacts / corpus fixtures: .log is process output,
    # never source; .txt in arxiv repos is overwhelmingly benchmark
    # corpus or scratch (papers themselves use .md/.rst/.tex). Skipping
    # outright cuts the worst tail of noise without sacrificing
    # search-relevant signal.
    ".log", ".txt",
    # Office documents — binary or proprietary text formats not usefully
    # readable by an LLM. Rare in research repos but cheap to defend.
    ".docx", ".dotx", ".doc", ".rtf", ".wpd",
    # Spreadsheets — binary or macro-laden Excel formats.
    ".xlsx", ".xlsm", ".xltm", ".xltx", ".xls",
    # Presentations.
    ".pptx", ".ppt",
    # LaTeX source + byproducts. The paper's prose already lives in
    # `papers.markdown` (rendered from LaTeXML HTML) and bibliography
    # entries populate `paper_references` — so .tex/.bib are duplicate
    # surfaces with heavy `\command{...}` boilerplate that BM25 won't
    # tokenize cleanly. .cls/.sty are publisher style packages with
    # zero retrieval value. .bbl/.aux/.dvi are auto-generated.
    ".tex", ".bib", ".cls", ".sty", ".bbl", ".aux", ".dvi",
})


# Exact basename match.
_SKIP_FILENAMES: frozenset[str] = frozenset({
    # Lockfiles (drawn from Linguist generated.rb)
    "package-lock.json", "npm-shrinkwrap.json",
    "yarn.lock", "pnpm-lock.yaml",
    "bun.lock", "bun.lockb",
    "uv.lock", "Pipfile.lock", "poetry.lock", "pdm.lock", "pixi.lock",
    "composer.lock", "Gemfile.lock",
    "Cargo.lock", "Cargo.toml.orig",
    "Gopkg.lock", "glide.lock", "go.sum",
    "deno.lock", "flake.lock",
    "MODULE.bazel.lock", "Package.resolved", ".terraform.lock.hcl",
    # Build wrappers (Linguist treats as generated)
    "gradlew", "gradlew.bat", "mvnw", "mvnw.cmd",
    # OS junk
    ".DS_Store", "Thumbs.db", "desktop.ini",
})


# Tensorboard event files match a wildcard rather than exact basename.
_TFEVENTS_PREFIX = "events.out.tfevents."


# Extension → language label. NULL (None) for everything else.
_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".md": "markdown",
    ".rst": "rst",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".sh": "shell",
    ".ipynb": "notebook",
    ".dockerfile": "dockerfile",
}


_README_PRIORITY: tuple[str, ...] = (
    "readme.md", "readme.rst", "readme.txt", "readme",
)


# ---------------------------------------------------------------------------
# Walk + decode
# ---------------------------------------------------------------------------


class _KeptFile(NamedTuple):
    path: str          # repo-root-relative, posix '/'
    language: str | None
    content: str
    size_bytes: int    # post-flatten byte length


def _language_for(basename: str, ext: str) -> str | None:
    if basename.lower() == "dockerfile":
        return "dockerfile"
    return _LANGUAGE_BY_EXT.get(ext.lower())


def _is_skipped_dir_component(name: str) -> bool:
    if name in _SKIP_DIRS:
        return True
    if name.endswith(_EGG_INFO_SUFFIX):
        return True
    return False


def _is_skipped_filename(basename: str, ext: str) -> bool:
    if basename in _SKIP_FILENAMES:
        return True
    if basename.startswith(_TFEVENTS_PREFIX):
        return True
    if ext.lower() in _SKIP_EXTS:
        return True
    return False


def _looks_binary(head: bytes) -> bool:
    """Null-byte sniff: a single ``\\x00`` in the first 8 KB ≈ binary."""
    return b"\x00" in head


def _decode_text(raw: bytes) -> str | None:
    """Decode UTF-8 with `errors='replace'`. Reject if > 1% replacement chars."""
    text = raw.decode("utf-8", errors="replace")
    if not text:
        return text
    replacements = text.count("�")
    if replacements / len(text) > _UTF8_REPLACEMENT_MAX_RATIO:
        return None
    return text


def _is_minified(text: str, ext: str) -> bool:
    """Linguist-style minified detection for .js / .css only.

    Average line length over the first 100 lines > 110 chars ⇒ minified.
    Catches webpack/parcel bundles checked in without a `.min` suffix.
    """
    if ext.lower() not in {".js", ".css"}:
        return False
    lines = text.splitlines()[:_MINIFIED_SAMPLE_LINES]
    if not lines:
        return False
    avg = sum(len(l) for l in lines) / len(lines)
    return avg > _MINIFIED_AVG_LINE_THRESHOLD


def _flatten_notebook(raw: bytes) -> str | None:
    """Concatenate code + markdown cells of an .ipynb. Drops outputs.

    Returns None on JSON parse failure (caller logs + skips the file).
    Cell separators mirror jupytext's percent format so the result reads
    like a percent-script.
    """
    try:
        nb = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        _LOG.warning("ipynb json parse failed: %s", exc)
        return None
    cells = nb.get("cells")
    if not isinstance(cells, list):
        return ""
    parts: list[str] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in ("code", "markdown"):
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if not isinstance(source, str):
            continue
        marker = (
            "\n\n# %% [markdown]\n\n"
            if cell_type == "markdown"
            else "\n\n# %% [code]\n\n"
        )
        parts.append(marker)
        parts.append(source)
    out = "".join(parts).lstrip("\n")
    return out


def _walk_and_collect(repo_root: Path) -> list[_KeptFile]:
    """Walk ``repo_root`` honoring skip rules; return the kept set.

    Symlinks are not followed (``followlinks=False``).
    """
    kept: list[_KeptFile] = []

    for dirpath, dirnames, filenames in os.walk(repo_root, followlinks=False):
        # Prune skipped directories in-place so os.walk doesn't recurse.
        dirnames[:] = [d for d in dirnames if not _is_skipped_dir_component(d)]

        rel_dir = Path(dirpath).relative_to(repo_root)
        for name in filenames:
            ext = os.path.splitext(name)[1]
            if _is_skipped_filename(name, ext):
                continue

            full = Path(dirpath) / name
            try:
                size = full.stat().st_size
            except OSError as exc:
                _LOG.warning("stat failed for %s: %s", full, exc)
                continue

            if size > _max_bytes_for_ext(ext):
                continue

            try:
                with open(full, "rb") as fh:
                    head = fh.read(_NULL_BYTE_SNIFF_BYTES)
                    rest = fh.read()
            except OSError as exc:
                _LOG.warning("read failed for %s: %s", full, exc)
                continue

            raw = head + rest

            if ext.lower() == ".ipynb":
                content = _flatten_notebook(raw)
                if content is None:
                    continue
            else:
                if _looks_binary(head):
                    continue
                decoded = _decode_text(raw)
                if decoded is None:
                    continue
                content = decoded
                if _is_minified(content, ext):
                    continue

            rel_path = (rel_dir / name) if str(rel_dir) != "." else Path(name)
            posix_path = rel_path.as_posix()
            kept.append(
                _KeptFile(
                    path=posix_path,
                    language=_language_for(name, ext),
                    content=content,
                    size_bytes=len(content.encode("utf-8")),
                )
            )

    return kept


def _select_top_level_readme(kept: list[_KeptFile]) -> _KeptFile | None:
    """First-match-wins README pick: top-level only, priority-ordered.

    Notebooks are not eligible — even ``README.ipynb`` is skipped because
    the flattened representation isn't intended for prose search.
    """
    by_lower: dict[str, _KeptFile] = {}
    for kf in kept:
        if "/" in kf.path:
            continue
        if kf.path.lower().endswith(".ipynb"):
            continue
        by_lower[kf.path.lower()] = kf
    for candidate in _README_PRIORITY:
        kf = by_lower.get(candidate)
        if kf is not None:
            return kf
    return None


# ---------------------------------------------------------------------------
# Git interaction
# ---------------------------------------------------------------------------


class _CloneResult(NamedTuple):
    success: bool
    commit_sha: str | None


def _run_git(
    args: list[str], *, cwd: str | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def _clone_repo(url: str, dest: str) -> _CloneResult:
    """``git clone --depth 1 --no-tags`` then ``rev-parse HEAD``.

    Returns ``_CloneResult(success=False, ...)`` on non-zero exit, timeout,
    or any unexpected error from git. The caller maps that to FAILED_REPO.
    """
    try:
        proc = _run_git(
            ["git", "clone", "--depth", "1", "--no-tags", url, dest],
            timeout=_GIT_CLONE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        _LOG.warning("git clone %s timed out after %ss", url, _GIT_CLONE_TIMEOUT_S)
        return _CloneResult(success=False, commit_sha=None)
    except FileNotFoundError as exc:
        # `git` not on PATH — surface clearly. This is a config error,
        # not a per-paper failure; raise so the operator sees it.
        raise RuntimeError("git executable not found on PATH") from exc

    if proc.returncode != 0:
        _LOG.warning(
            "git clone %s exited %d: %s",
            url, proc.returncode, (proc.stderr or "").strip(),
        )
        return _CloneResult(success=False, commit_sha=None)

    rev = _run_git(["git", "-C", dest, "rev-parse", "HEAD"], timeout=10)
    if rev.returncode != 0:
        _LOG.warning(
            "rev-parse HEAD failed in %s: %s",
            dest, (rev.stderr or "").strip(),
        )
        return _CloneResult(success=True, commit_sha=None)
    return _CloneResult(success=True, commit_sha=rev.stdout.strip() or None)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


class _RepoRow(NamedTuple):
    id: int
    repo_slug: str
    url: str
    domain: str | None
    status: str


def _lookup_repo(conn, *, repo_slug: str) -> _RepoRow:
    row = conn.execute(
        "SELECT id, repo_slug, url, domain, status "
        "  FROM repos WHERE repo_slug = ?",
        (repo_slug,),
    ).fetchone()
    if row is None:
        raise ValueError(f"repo not found: repo_slug={repo_slug!r}")
    return _RepoRow(*row)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _persist_failed(conn, repo_id: int) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE repos SET status = ?, fetched_at = ? WHERE id = ?",
            (RepoStatus.FAILED_REPO.value, _now_iso(), repo_id),
        )


def _persist_success(
    conn,
    *,
    repo: _RepoRow,
    files: list[_KeptFile],
    readme: _KeptFile | None,
    commit_sha: str | None,
) -> None:
    """Replace the repo's code_files + readmes_fts rows in one transaction."""
    with transaction(conn):
        conn.execute(
            "DELETE FROM code_files  WHERE repo_id = ?", (repo.id,)
        )
        conn.execute(
            "DELETE FROM readmes_fts WHERE repo_id = ?", (repo.id,)
        )
        if files:
            conn.executemany(
                "INSERT INTO code_files "
                "  (repo_id, path, language, size_bytes, content) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (repo.id, f.path, f.language, f.size_bytes, f.content)
                    for f in files
                ],
            )
        if readme is not None:
            conn.execute(
                "INSERT INTO readmes_fts "
                "  (repo_id, repo_slug, domain, path, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    repo.id, repo.repo_slug, repo.domain or "",
                    readme.path, readme.content,
                ),
            )
        conn.execute(
            "UPDATE repos SET status = ?, commit_sha = ?, fetched_at = ?, "
            "  file_count = ?, has_readme = ? WHERE id = ?",
            (
                RepoStatus.REPO_FETCHED.value, commit_sha, _now_iso(),
                len(files), 1 if readme is not None else 0, repo.id,
            ),
        )


# ---------------------------------------------------------------------------
# Public stage entrypoint
# ---------------------------------------------------------------------------


def fetch_repo(*, conn, repo_slug: str, force: bool = False) -> None:
    """Stage entrypoint — clone, walk, persist for one repo.

    Keyword-only to match the contract enforced by
    :func:`_system.tests.test_ingest.test_stage_function_signatures_are_keyword_only`.
    """
    del force  # cascade is owned by ingest; per-stage idempotency is via DELETE+INSERT.

    repo = _lookup_repo(conn, repo_slug=repo_slug)

    try:
        current = RepoStatus(repo.status)
    except ValueError as exc:
        raise ValueError(
            f"repos.status={repo.status!r} for repo_slug={repo_slug!r} "
            "is not a recognized RepoStatus"
        ) from exc

    if not can_run_from(current, RepoStatus.REPO_FETCHED):
        _LOG.info(
            "fetch_repo skipped for %s: status=%s incompatible with target=%s",
            repo_slug, current, RepoStatus.REPO_FETCHED,
        )
        return

    with tempfile.TemporaryDirectory(prefix=f"lodestone_repo_{repo_slug}_") as tmp:
        clone_dest = str(Path(tmp) / "clone")
        result = _clone_repo(repo.url, clone_dest)
        if not result.success:
            _persist_failed(conn, repo.id)
            return

        kept = _walk_and_collect(Path(clone_dest))
        readme = _select_top_level_readme(kept)
        if readme is None:
            _LOG.info("no top-level README found for %s", repo_slug)

    _persist_success(
        conn,
        repo=repo,
        files=kept,
        readme=readme,
        commit_sha=result.commit_sha,
    )
    _LOG.info(
        "fetch_repo %s: %d files indexed, readme=%s, commit=%s",
        repo_slug, len(kept), readme.path if readme else None,
        result.commit_sha[:8] if result.commit_sha else None,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clone a repo and persist code_files + readmes_fts."
    )
    parser.add_argument("--repo", required=True, help="repo_slug")
    parser.add_argument(
        "--db",
        default=os.environ.get("LODESTONE_DB", "lodestone.db"),
        help="path to the sqlite db (default: $LODESTONE_DB or ./lodestone.db)",
    )
    parser.add_argument("--force", action="store_true",
                        help="(reserved for parity with other stage CLIs)")
    args = parser.parse_args(argv)

    conn = get_conn(Path(args.db))
    try:
        init_db(conn)
        fetch_repo(conn=conn, repo_slug=args.repo, force=args.force)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
