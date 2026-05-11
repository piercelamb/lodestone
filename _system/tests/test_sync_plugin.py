"""Tests for ``_system/scripts/sync_plugin.py``.

The script shells out to ``rsync``; tests use real rsync against tmp
dirs rather than mocking subprocess so we exercise the actual exclude
patterns and ``--delete`` behavior end-to-end.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from _system.scripts.sync_plugin import sync_plugin, _resolve_dest


_RSYNC = shutil.which("rsync")
_NEEDS_RSYNC = pytest.mark.skipif(
    _RSYNC is None,
    reason="rsync not on PATH; sync_plugin can't run.",
)


def _populate_source(src: Path) -> None:
    """Build a miniature lodestone repo: real files + things we exclude."""
    (src / "_system").mkdir()
    (src / "_system/scripts").mkdir()
    (src / "_system/scripts/foo.py").write_text("print('hi')\n")
    (src / "bin").mkdir()
    (src / "bin/lodestone-mcp-plugin.sh").write_text("#!/bin/bash\necho hi\n")
    (src / "pyproject.toml").write_text("[project]\nname='lodestone'\n")

    (src / ".claude-plugin").mkdir()
    (src / ".claude-plugin/plugin.json").write_text(
        json.dumps({"name": "lodestone", "version": "9.9.9"})
    )

    # Things the exclude list must keep out of the mirror.
    (src / ".venv").mkdir()
    (src / ".venv/marker").write_text("should not be copied\n")
    (src / ".git").mkdir()
    (src / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (src / ".pytest_cache").mkdir()
    (src / ".pytest_cache/v").write_text("cache\n")
    (src / "__pycache__").mkdir()
    (src / "__pycache__/x.pyc").write_text("bytecode\n")
    (src / "lodestone.db").write_text("dev db, do not copy")
    (src / "lodestone.db-wal").write_text("wal")
    (src / "lodestone.db-shm").write_text("shm")
    (src / ".DS_Store").write_text("macos junk")


@_NEEDS_RSYNC
def test_sync_plugin_copies_real_files(tmp_path: Path):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    _populate_source(src)

    summary = sync_plugin(source=src, dest=dest, dry_run=False)

    assert summary["rsync_returncode"] == 0
    assert (dest / "_system/scripts/foo.py").read_text() == "print('hi')\n"
    assert (dest / "bin/lodestone-mcp-plugin.sh").exists()
    assert (dest / "pyproject.toml").exists()
    assert (dest / ".claude-plugin/plugin.json").exists()


@_NEEDS_RSYNC
def test_sync_plugin_honors_excludes(tmp_path: Path):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    _populate_source(src)

    sync_plugin(source=src, dest=dest, dry_run=False)

    for excluded in (
        ".venv", ".git", ".pytest_cache", "__pycache__",
        "lodestone.db", "lodestone.db-wal", "lodestone.db-shm",
        ".DS_Store",
    ):
        assert not (dest / excluded).exists(), (
            f"{excluded!r} leaked into the mirror"
        )


@_NEEDS_RSYNC
def test_sync_plugin_deletes_orphaned_dest_files(tmp_path: Path):
    """Files removed from src must disappear from dest (--delete)."""
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    _populate_source(src)
    # Pre-existing stale file in dest that's not in src.
    (dest / "stale.txt").write_text("old content")

    sync_plugin(source=src, dest=dest, dry_run=False)

    assert not (dest / "stale.txt").exists()


@_NEEDS_RSYNC
def test_sync_plugin_preserves_dest_venv(tmp_path: Path):
    """Critical: dest's existing .venv must NOT be wiped by --delete.

    The plugin launcher built that venv via `uv sync`; rebuilding it
    on every reset would be slow and pointless. Because .venv is in the
    --exclude list, --delete should leave it alone.
    """
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    _populate_source(src)
    (dest / ".venv").mkdir()
    (dest / ".venv/python_marker").write_text("plugin-built venv\n")

    sync_plugin(source=src, dest=dest, dry_run=False)

    assert (dest / ".venv/python_marker").read_text() == "plugin-built venv\n"


@_NEEDS_RSYNC
def test_sync_plugin_dry_run_makes_no_changes(tmp_path: Path):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    _populate_source(src)

    summary = sync_plugin(source=src, dest=dest, dry_run=True)

    assert summary["dry_run"] is True
    # Nothing should have been written.
    assert list(dest.iterdir()) == []


@_NEEDS_RSYNC
def test_sync_plugin_raises_when_dest_missing(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    _populate_source(src)
    dest = tmp_path / "no_such_dir"

    with pytest.raises(FileNotFoundError, match="plugin install dir not found"):
        sync_plugin(source=src, dest=dest, dry_run=False)


def test_resolve_dest_uses_manifest_version(tmp_path: Path):
    """The dest dir is derived from .claude-plugin/plugin.json's version
    field, not globbed off the filesystem — so a version bump in dev
    that hasn't been re-installed surfaces a clear missing-dir error
    instead of overwriting an old cache.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / ".claude-plugin").mkdir()
    (src / ".claude-plugin/plugin.json").write_text(
        json.dumps({"name": "lodestone", "version": "1.2.3"})
    )

    dest = _resolve_dest(src)

    assert dest.name == "1.2.3"
    assert dest.parent.name == "lodestone"
