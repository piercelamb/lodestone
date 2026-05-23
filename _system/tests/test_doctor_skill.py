"""Regression tests for the /lodestone:doctor skill.

Issue #74: Kendall's `/lodestone:doctor` invocation failed before any check
ran because line 45 of skills/doctor/SKILL.md inlined
`mkdir -p "$HOME/.lodestone" && stat -L "$HOME/.lodestone/lodestone.db" ...`
as an `!`-injected bash block. Claude Code's permission validator rejects
bash commands whose path arguments expand an untracked variable like
`$HOME` — even when `Bash(mkdir:*)` is listed in `allowed-tools`. The fix
was to move the check into `skills/doctor/scripts/check-db.sh` (where
`$HOME` expands at script runtime, invisible to the validator) and call
the script via the tracked `${CLAUDE_SKILL_DIR}` variable.

These tests cover both halves:
  1. Behavior of `check-db.sh` (DB present / absent / parent-dir-missing).
  2. A static guard that no `!`-injected bash block in any SKILL.md
     contains an untracked-variable expansion in a path argument — a
     re-regression of #74 would trip this.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DOCTOR_DIR = _REPO_ROOT / "skills" / "doctor"
_CHECK_DB_SH = _DOCTOR_DIR / "scripts" / "check-db.sh"


# ---------------------------------------------------------------------------
# check-db.sh behavior
# ---------------------------------------------------------------------------

def _run_check_db(home: Path) -> subprocess.CompletedProcess[str]:
    """Run check-db.sh with HOME overridden to an isolated dir."""
    env = os.environ.copy()
    env["HOME"] = str(home)
    return subprocess.run(
        ["bash", str(_CHECK_DB_SH)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_check_db_sh_exists_and_is_executable():
    assert _CHECK_DB_SH.is_file(), f"missing {_CHECK_DB_SH}"
    assert os.access(_CHECK_DB_SH, os.X_OK), f"{_CHECK_DB_SH} is not executable"


def test_check_db_reports_missing_db(tmp_path: Path):
    """No ~/.lodestone at all — script must not crash, must report 'no DB yet'."""
    result = _run_check_db(tmp_path)
    assert result.returncode == 0, (
        f"check-db.sh exited {result.returncode}; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "no DB yet" in result.stdout, (
        f"expected 'no DB yet' message, got: {result.stdout!r}"
    )


def test_check_db_reports_missing_db_even_when_dir_exists(tmp_path: Path):
    """~/.lodestone exists but the DB file doesn't — still 'no DB yet'.

    Regression guard: the original inline command included `mkdir -p` as a
    side-effect. The replacement script must not depend on that.
    """
    (tmp_path / ".lodestone").mkdir()
    result = _run_check_db(tmp_path)
    assert result.returncode == 0
    assert "no DB yet" in result.stdout


def test_check_db_stats_existing_db(tmp_path: Path):
    """DB file present — script should emit a stat line, not the fallback."""
    lodestone_dir = tmp_path / ".lodestone"
    lodestone_dir.mkdir()
    db = lodestone_dir / "lodestone.db"
    db.write_bytes(b"")  # empty file is fine; doctor docs say empty DB is OK
    result = _run_check_db(tmp_path)
    assert result.returncode == 0
    assert "no DB yet" not in result.stdout, (
        f"unexpected fallback when DB exists: {result.stdout!r}"
    )
    assert str(db) in result.stdout, (
        f"expected stat output to mention {db}, got: {result.stdout!r}"
    )


# ---------------------------------------------------------------------------
# SKILL.md static guard — the core regression test for #74
# ---------------------------------------------------------------------------

# Variables Claude Code's permission validator is willing to resolve. Anything
# else expanded inside a path argument trips the "untracked-variable output"
# rejection that broke /lodestone:doctor in #74. CLAUDE_PROJECT_DIR and
# CLAUDE_PLUGIN_ROOT round out the docs-blessed set even though the doctor
# skill only uses CLAUDE_SKILL_DIR today.
_TRACKED_VARS = {"CLAUDE_SKILL_DIR", "CLAUDE_PROJECT_DIR", "CLAUDE_PLUGIN_ROOT"}

# Match `$VAR` or `${VAR}` references inside an `!`-injected bash block.
_VAR_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")

# Match a full `!`-injected line: `!`backtick<command>backtick. Captures the command.
_BANG_INJECTION = re.compile(r"^!`([^`]+)`\s*$")


def _all_skill_md_files() -> list[Path]:
    return sorted((_REPO_ROOT / "skills").rglob("SKILL.md"))


def test_skills_directory_has_skill_files():
    """Sanity check — make sure the guard below is actually scanning something."""
    files = _all_skill_md_files()
    assert files, "no SKILL.md files found under skills/"


@pytest.mark.parametrize("skill_md", _all_skill_md_files(), ids=lambda p: p.relative_to(_REPO_ROOT).as_posix())
def test_skill_md_bang_injection_uses_only_tracked_vars(skill_md: Path):
    """Issue #74 regression guard.

    For every `!`-injected bash block in a SKILL.md, every `$VAR` / `${VAR}`
    reference must be in `_TRACKED_VARS`. Untracked-variable expansion in a
    bash command argument is what Claude Code's permission validator
    rejects, and that rejection short-circuits the entire skill before any
    check runs (see issue #74).

    If you need an inline command that depends on `$HOME` (or any other
    untracked variable), move the work into a script under `scripts/` and
    invoke it via `${CLAUDE_SKILL_DIR}/scripts/<name>.sh` — the variable
    then expands at script runtime, invisible to the validator.
    """
    offenders: list[tuple[int, str, str]] = []
    for line_no, line in enumerate(skill_md.read_text(encoding="utf-8").splitlines(), start=1):
        m = _BANG_INJECTION.match(line)
        if not m:
            continue
        command = m.group(1)
        for var_match in _VAR_REF.finditer(command):
            var_name = var_match.group(1)
            if var_name not in _TRACKED_VARS:
                offenders.append((line_no, var_name, command))

    assert not offenders, (
        f"{skill_md.relative_to(_REPO_ROOT)} has `!`-injected bash blocks "
        f"referencing untracked variables. Claude Code's permission validator "
        f"rejects these (see issue #74). Move the command into a script under "
        f"scripts/ and invoke it via ${{CLAUDE_SKILL_DIR}}/scripts/<name>.sh.\n"
        f"Offenders:\n"
        + "\n".join(f"  line {ln}: ${var} in `{cmd}`" for ln, var, cmd in offenders)
    )
