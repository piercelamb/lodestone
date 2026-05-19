"""Mirror this dev tree onto the installed lodestone plugin.

    uv run _system/scripts/sync_plugin.py [--yes] [--dry-run]

Claude Code's ``/plugin install lodestone`` unpacks the marketplace
source into ``~/.claude/plugins/cache/piercelamb-plugins/lodestone/<version>/``;
that directory is what every other Claude Code session boots from. To
make the running plugin reflect uncommitted dev changes, this script
``rsync``s the working tree over the install dir.

Excluded from the mirror: ``.venv`` (rebuilt by ``uv sync --quiet`` on
the next plugin launch), ``.git`` / ``.pytest_cache`` / ``__pycache__``
(churn-only), and any local ``lodestone.db*`` (the plugin's launcher
points at ``$HOME/.lodestone/lodestone.db`` regardless, so a bundled
copy in the install dir is just stale weight). ``rsync --delete`` is
used so files removed in dev disappear from the install too — same
contract as a fresh ``/plugin install``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from _system.utils.logging import get_logger

_LOG = get_logger("scripts.sync_plugin")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _REPO_ROOT / ".claude-plugin/plugin.json"
_PLUGIN_CACHE_BASE = (
    Path.home() / ".claude/plugins/cache/piercelamb-plugins/lodestone"
)

# Paths excluded from the mirror. Trailing slash on dirs is rsync
# convention; rsync without --recursive on a leaf file pattern is fine.
_EXCLUDES = (
    ".venv/",
    ".git/",
    ".pytest_cache/",
    "__pycache__/",
    ".DS_Store",
    "lodestone.db",
    "lodestone.db-wal",
    "lodestone.db-shm",
    "node_modules/",
    # Dev-only project MCP config (gitignored, carries absolute repo
    # paths). The install dir uses the tracked .mcp.json at the plugin
    # root, which points at CLAUDE_PLUGIN_ROOT.
    ".mcp.dev.json",
)


def _resolve_dest(repo_root: Path) -> Path:
    """Locate the install dir for the plugin version declared in this tree.

    Reading the version from ``plugin.json`` (rather than globbing the
    cache) makes the script's behavior deterministic — if the dev tree
    bumped the version but the user hasn't reinstalled, we'll point at a
    dir that doesn't exist and surface a clear error instead of silently
    overwriting an older install.
    """
    manifest = json.loads((repo_root / ".claude-plugin/plugin.json").read_text())
    version = manifest["version"]
    return _PLUGIN_CACHE_BASE / version


def sync_plugin(
    *,
    source: Path,
    dest: Path,
    dry_run: bool = False,
) -> dict:
    """Rsync ``source`` → ``dest`` with the lodestone exclude set.

    Returns ``{"source", "dest", "rsync_returncode", "transferred",
    "deleted"}``. Raises ``FileNotFoundError`` if ``dest`` doesn't exist
    (the plugin must already be installed once via ``/plugin install``)
    or ``RuntimeError`` if rsync isn't on PATH.
    """
    rsync = shutil.which("rsync")
    if rsync is None:
        raise RuntimeError(
            "rsync not on PATH. macOS ships with it by default; install "
            "via Homebrew (`brew install rsync`) if missing."
        )
    if not source.is_dir():
        raise FileNotFoundError(f"source not a directory: {source}")
    if not dest.is_dir():
        raise FileNotFoundError(
            f"plugin install dir not found: {dest}. Run `/plugin install "
            "lodestone` first, then retry."
        )

    cmd: list[str] = [
        rsync, "-a", "--delete", "--itemize-changes",
        *(f"--exclude={pat}" for pat in _EXCLUDES),
    ]
    if dry_run:
        cmd.append("--dry-run")
    # Trailing slash on source: copy CONTENTS of source into dest
    # (rsync(1) "trailing slash" rule). Without it, source dir name
    # gets nested under dest.
    cmd.extend([f"{source}/", f"{dest}/"])

    _LOG.info("rsync %s -> %s (dry_run=%s)", source, dest, dry_run)
    proc = subprocess.run(
        cmd, check=False, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"rsync failed (rc={proc.returncode}):\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    # --itemize-changes prefixes each transfer with a code; deletions
    # are lines starting with "*deleting". Counts are advisory.
    transferred = 0
    deleted = 0
    for line in proc.stdout.splitlines():
        if line.startswith("*deleting"):
            deleted += 1
        elif line and not line.startswith("."):
            transferred += 1

    return {
        "source": str(source),
        "dest": str(dest),
        "rsync_returncode": proc.returncode,
        "transferred": transferred,
        "deleted": deleted,
        "dry_run": dry_run,
    }


def _confirm(source: Path, dest: Path) -> bool:
    print(f"sync plugin: {source} -> {dest}")
    print(f"excludes: {', '.join(_EXCLUDES)}")
    print("rsync --delete is enabled: files removed from dev disappear here too.")
    answer = input("proceed? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror this dev tree onto the installed lodestone plugin "
            "so every Claude Code session sees the latest code."
        ),
    )
    parser.add_argument(
        "--source", type=Path, default=_REPO_ROOT,
        help=f"dev repo root (default: {_REPO_ROOT})",
    )
    parser.add_argument(
        "--dest", type=Path, default=None,
        help=(
            "plugin install dir "
            "(default: derived from .claude-plugin/plugin.json version)"
        ),
    )
    parser.add_argument("--yes", action="store_true",
                        help="skip confirmation prompt")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what rsync would do; make no changes")
    args = parser.parse_args(argv)

    source = args.source.resolve()
    dest = args.dest.resolve() if args.dest else _resolve_dest(source)

    if not args.yes and not args.dry_run and not _confirm(source, dest):
        print("aborted.")
        sys.exit(1)

    summary = sync_plugin(source=source, dest=dest, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
