"""Unit tests for ``_system/scripts/fetch_repo.py``.

The clone is monkeypatched everywhere — we never invoke real git or
hit the network. The walker is exercised against synthetic file trees
under ``tmp_path``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from _system.schemas.paper_metadata import PaperStatus
from _system.scripts import fetch_repo as fr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_paper(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str = "2401.99999",
    paper_name: str = "demo_2026",
    code_repo: str | None = "https://github.com/owner/repo",
    domain: str = "rag",
    collection: str = "demo_collection",
    status: PaperStatus = PaperStatus.INDEXED,
) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO domains (name) VALUES (?)", (domain,)
    )
    conn.execute(
        "INSERT OR IGNORE INTO collections (domain, name, description) "
        "VALUES (?, ?, NULL)",
        (domain, collection),
    )
    cur = conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, paper_name, title, authors, date, abstract,
            pdf_url, ingested_at, status, domain, collection, code_repo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            arxiv_id, paper_name, "Title", '["A"]', "2024-01-01", "Abs",
            f"https://arxiv.org/pdf/{arxiv_id}",
            "2024-01-02T00:00:00+00:00",
            status.value, domain, collection, code_repo,
        ),
    )
    return cur.lastrowid


class _FakeClone:
    """Monkeypatchable replacement for ``_clone_repo``.

    Populates ``dest`` with a tree built by ``layout`` (``{path: bytes|str}``).
    """

    def __init__(
        self,
        layout: dict[str, bytes | str] | None = None,
        *,
        success: bool = True,
        commit_sha: str | None = "abcdef0123",
        raise_exc: BaseException | None = None,
    ) -> None:
        self.layout = layout or {}
        self.success = success
        self.commit_sha = commit_sha
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url: str, dest: str) -> fr._CloneResult:
        self.calls.append((url, dest))
        if self.raise_exc is not None:
            raise self.raise_exc
        if not self.success:
            return fr._CloneResult(success=False, commit_sha=None)
        Path(dest).mkdir(parents=True, exist_ok=True)
        for rel, data in self.layout.items():
            path = Path(dest) / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
            with open(path, mode) as fh:
                fh.write(data)
        return fr._CloneResult(success=True, commit_sha=self.commit_sha)


# ---------------------------------------------------------------------------
# Status routing
# ---------------------------------------------------------------------------


def test_no_repo_url_marks_repo_fetched_with_no_files(conn, monkeypatch):
    pid = _seed_paper(conn, code_repo=None)
    fake = _FakeClone()
    monkeypatch.setattr(fr, "_clone_repo", fake)

    fr.fetch_repo(conn=conn, paper_name="demo_2026")

    status = conn.execute(
        "SELECT status FROM papers WHERE id = ?", (pid,)
    ).fetchone()[0]
    assert status == PaperStatus.REPO_FETCHED.value
    assert conn.execute(
        "SELECT COUNT(*) FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchone()[0] == 0
    assert fake.calls == [], "no clone should be attempted"


def test_clone_failure_sets_failed_repo(conn, monkeypatch):
    pid = _seed_paper(conn)
    monkeypatch.setattr(
        fr, "_clone_repo", _FakeClone(success=False)
    )
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    status = conn.execute(
        "SELECT status FROM papers WHERE id = ?", (pid,)
    ).fetchone()[0]
    assert status == PaperStatus.FAILED_REPO.value


def test_timeout_sets_failed_repo(conn, monkeypatch):
    pid = _seed_paper(conn)

    def _raise_timeout(args, **kw):
        raise subprocess.TimeoutExpired(cmd=args, timeout=1)

    # Test the real ``_clone_repo`` translates a TimeoutExpired into
    # ``success=False`` rather than propagating.
    monkeypatch.setattr(fr, "_run_git", _raise_timeout)
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    status = conn.execute(
        "SELECT status FROM papers WHERE id = ?", (pid,)
    ).fetchone()[0]
    assert status == PaperStatus.FAILED_REPO.value


# ---------------------------------------------------------------------------
# Walker / filtering
# ---------------------------------------------------------------------------


def test_walks_and_inserts_text_files(conn, monkeypatch):
    pid = _seed_paper(conn)
    layout = {
        "main.py": "print('hi')\n",
        "README.md": "# Demo Repo\n\nUseful info.\n",
        "src/utils.py": "def f(): pass\n",
    }
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone(layout))

    fr.fetch_repo(conn=conn, paper_name="demo_2026")

    rows = conn.execute(
        "SELECT path, language, size_bytes FROM code_files "
        " WHERE paper_id = ? ORDER BY path", (pid,)
    ).fetchall()
    paths = {r[0] for r in rows}
    assert paths == {"README.md", "main.py", "src/utils.py"}
    langs = {r[0]: r[1] for r in rows}
    assert langs["main.py"] == "python"
    assert langs["README.md"] == "markdown"
    assert all(r[2] > 0 for r in rows)


@pytest.mark.parametrize("name", [
    "model.pt", "weights.safetensors", "data.parquet", "blob.pkl",
    "lib.so", "archive.zip", "img.png",
])
def test_skips_binary_extensions(conn, monkeypatch, name):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "keep.py": "x=1\n",
        name: b"\x00\x01\x02\x03",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"keep.py"}, paths


@pytest.mark.parametrize("dirname", [
    "node_modules", "__pycache__", ".venv", "dist", "build", ".git",
])
def test_skips_vendored_directories(conn, monkeypatch, dirname):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "keep.py": "x=1\n",
        f"{dirname}/junk.py": "x=2\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"keep.py"}


@pytest.mark.parametrize("dirname", [
    ".ipynb_checkpoints", "wandb", "mlruns", "lightning_logs", ".hydra",
])
def test_skips_ml_experiment_dirs(conn, monkeypatch, dirname):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "keep.py": "x=1\n",
        f"{dirname}/run.log": "x=2\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"keep.py"}


@pytest.mark.parametrize("name", [
    "package-lock.json", "uv.lock", "Cargo.lock", "go.sum", "flake.lock",
])
def test_skips_lockfiles_by_name(conn, monkeypatch, name):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "keep.py": "x=1\n",
        name: '{"x": 1}\n',
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"keep.py"}


def test_skips_tfevents_files(conn, monkeypatch):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "keep.py": "x=1\n",
        "logs/events.out.tfevents.1234.host": b"raw",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"keep.py"}


def test_keeps_csv_tsv_json_yaml_configs(conn, monkeypatch):
    pid = _seed_paper(conn)
    layout = {
        "metrics.csv": "a,b,c\n1,2,3\n",
        "table.tsv": "a\tb\n1\t2\n",
        "config.yaml": "lr: 1e-4\n",
        "params.toml": "[a]\nx=1\n",
        "data.json": '{"k":1}\n',
        "settings.ini": "[s]\na=1\n",
    }
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone(layout))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == set(layout.keys())


def test_keeps_models_and_data_dirs_as_source(conn, monkeypatch):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "models/transformer.py": "class M: pass\n",
        "data/loader.py": "def load(): pass\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"models/transformer.py", "data/loader.py"}


def test_skips_files_with_null_bytes_in_first_8kb(conn, monkeypatch):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "good.py": "x=1\n",
        # `.dat` extension isn't in skip list — content sniff catches it.
        "binary.dat": b"hello\x00world\nplus more bytes",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"good.py"}


def test_skips_files_with_high_replacement_char_density(conn, monkeypatch):
    pid = _seed_paper(conn)
    # Latin-1 garbage that decodes-with-replace into mostly U+FFFD.
    payload = bytes(range(0x80, 0xff)) * 4  # high bytes; no nulls
    assert b"\x00" not in payload
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "good.py": "x=1\n",
        "garbage.dat": payload,
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"good.py"}


def test_skips_minified_js_via_avg_line_length_heuristic(conn, monkeypatch):
    pid = _seed_paper(conn)
    long_line = "a" * 500
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "normal.js": "function f(){\n  return 1;\n}\n",
        "bundle.js": long_line + "\n" + long_line + "\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"normal.js"}


def test_does_not_follow_symlinks(conn, monkeypatch, tmp_path):
    pid = _seed_paper(conn)
    # Build the target tree on disk and have the fake clone create a
    # symlink that points outside the clone dir. ``_walk_and_collect``
    # uses os.walk(followlinks=False), so the symlinked dir is not
    # descended into.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("danger=1\n")

    captured: dict[str, str] = {}

    def _fake_clone(url, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "real.py").write_text("ok=1\n")
        os.symlink(outside, Path(dest) / "linked")
        captured["dest"] = dest
        return fr._CloneResult(success=True, commit_sha="aaa")

    monkeypatch.setattr(fr, "_clone_repo", _fake_clone)
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"real.py"}, paths


def test_skips_oversized_files(conn, monkeypatch):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "small.py": "x=1\n",
        # `.md` is not in `_SKIP_EXTS` and falls under the global cap;
        # 600k lines blows past 1 MB.
        "huge.md": "y\n" * 600_000,
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"small.py"}


def test_skips_log_and_txt_extensions(conn, monkeypatch):
    """`.log` is operational output; `.txt` is overwhelmingly benchmark
    corpus or scratch in arxiv repos. Both skip outright regardless of size."""
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "good.py": "x=1\n",
        "run.log": "INFO step=1 loss=0.5\n",
        "notes.txt": "scratchpad\n",
        "experiments/results/eval.log": "0.42\n",
        "benchmark/corpus/alice.txt": "Once upon a time...\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"good.py"}


def test_per_extension_cap_skips_large_json_keeps_small(conn, monkeypatch):
    """JSON / JSONL get a 50 KB cap so per-query score dumps don't
    swallow the DB; small configs remain searchable."""
    pid = _seed_paper(conn)
    big = '{"x":' + ("1," * 30_000) + "0}\n"  # > 50 KB, < 1 MB
    assert 50_000 < len(big) < 1_000_000
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "package.json": '{"name": "demo"}\n',
        "experiments/results/perquery.json": big,
        "experiments/train_pairs.jsonl": big,
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"package.json"}


@pytest.mark.parametrize("name", [
    # Office docs
    "report.docx", "template.dotx", "old_report.doc",
    "memo.rtf", "wp.wpd",
    # Spreadsheets
    "results.xlsx", "macros.xlsm", "tmpl.xltm", "tmpl.xltx", "old.xls",
    # Presentations
    "deck.pptx", "old_deck.ppt",
    # LaTeX source / byproducts
    "paper.tex", "refs.bib", "ieee.cls", "mystyle.sty",
    "paper.bbl", "paper.aux", "paper.dvi",
])
def test_skips_office_and_latex_extensions(conn, monkeypatch, name):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "good.py": "x=1\n",
        name: "anything\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"good.py"}, (name, paths)


def test_per_extension_cap_skips_large_csv_keeps_small(conn, monkeypatch):
    """CSV / TSV stay searchable for small eval-result tables but the
    50 KB cap clips multi-MB per-query dumps."""
    pid = _seed_paper(conn)
    big = "a,b\n" + ("1,2\n" * 30_000)  # > 50 KB, < 1 MB
    assert 50_000 < len(big) < 1_000_000
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "metrics.csv": "method,score\nbm25,0.42\n",
        "experiments/results/perquery.csv": big,
        "experiments/results/perquery.tsv": big.replace(",", "\t"),
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"metrics.csv"}


def test_per_extension_cap_does_not_clip_python(conn, monkeypatch):
    """Real source files (e.g. >50 KB Python) must survive — only
    json/jsonl get the tighter cap."""
    pid = _seed_paper(conn)
    big_py = "# header\n" + ("def f(): pass\n" * 6_000)  # > 50 KB
    assert 50_000 < len(big_py) < 1_000_000
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "src/store.py": big_py,
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"src/store.py"}


def test_flattens_ipynb_keeps_code_and_markdown_cells_drops_outputs(
    conn, monkeypatch,
):
    pid = _seed_paper(conn)
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Title\n", "\n", "Body."]},
            {"cell_type": "code", "source": "x = 1\n", "outputs": [
                {"output_type": "stream", "text": "noise"}
            ]},
            {"cell_type": "raw", "source": "ignore"},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "notebook.ipynb": json.dumps(nb).encode(),
        "main.py": "y=2\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    row = conn.execute(
        "SELECT content, language FROM code_files "
        " WHERE paper_id = ? AND path = ?", (pid, "notebook.ipynb")
    ).fetchone()
    assert row is not None, "notebook should be stored"
    content, language = row
    assert language == "notebook"
    assert "# Title" in content
    assert "x = 1" in content
    assert "noise" not in content
    assert "ignore" not in content


def test_force_replaces_existing_rows(conn, monkeypatch):
    pid = _seed_paper(conn)
    # Pre-seed code_files rows that should be wiped on the next run.
    conn.execute(
        "INSERT INTO code_files (paper_id, path, language, size_bytes, content) "
        "VALUES (?, ?, ?, ?, ?)",
        (pid, "stale.py", "python", 5, "stale\n"),
    )
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "fresh.py": "ok=1\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    paths = {r[0] for r in conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()}
    assert paths == {"fresh.py"}


def test_commit_sha_persisted_to_papers(conn, monkeypatch):
    pid = _seed_paper(conn)
    monkeypatch.setattr(
        fr, "_clone_repo",
        _FakeClone({"x.py": "x=1\n"}, commit_sha="deadbeef00")
    )
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    sha, fetched = conn.execute(
        "SELECT code_repo_commit, code_repo_fetched_at FROM papers "
        " WHERE id = ?", (pid,)
    ).fetchone()
    assert sha == "deadbeef00"
    assert fetched is not None and "T" in fetched


# ---------------------------------------------------------------------------
# README / readmes_fts
# ---------------------------------------------------------------------------


def test_top_level_readme_md_indexed_into_readmes_fts(conn, monkeypatch):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "README.md": "# Demo\n\nMixture-of-experts training pipeline.\n",
        "src/main.py": "x=1\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    rows = conn.execute(
        "SELECT path, paper_name, content FROM readmes_fts WHERE paper_id = ?",
        (pid,),
    ).fetchall()
    assert len(rows) == 1
    path, name, content = rows[0]
    assert path == "README.md"
    assert name == "demo_2026"
    assert "Mixture-of-experts" in content


def test_readme_priority_md_over_rst_over_txt_over_extensionless(
    conn, monkeypatch,
):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "README": "ext-less\n",
        "README.txt": "txt content\n",
        "README.rst": "rst content\n",
        "README.md": "md content\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    row = conn.execute(
        "SELECT path FROM readmes_fts WHERE paper_id = ?", (pid,)
    ).fetchone()
    assert row[0] == "README.md"


def test_subdir_readme_not_indexed_into_readmes_fts(conn, monkeypatch):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "docs/README.md": "subdir readme\n",
        "src/x.py": "y=1\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    rdm = conn.execute(
        "SELECT COUNT(*) FROM readmes_fts WHERE paper_id = ?", (pid,)
    ).fetchone()[0]
    assert rdm == 0
    # But the subdir README still lands in code_files.
    cf = conn.execute(
        "SELECT path FROM code_files WHERE paper_id = ?", (pid,)
    ).fetchall()
    assert ("docs/README.md",) in cf


def test_no_readme_in_repo_no_row_in_readmes_fts(conn, monkeypatch):
    pid = _seed_paper(conn)
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "main.py": "x=1\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    assert conn.execute(
        "SELECT COUNT(*) FROM readmes_fts WHERE paper_id = ?", (pid,)
    ).fetchone()[0] == 0


def test_force_replaces_readme_row(conn, monkeypatch):
    pid = _seed_paper(conn)
    # Pre-seed a stale readme.
    conn.execute(
        "INSERT INTO readmes_fts (paper_id, domain, paper_name, path, content) "
        "VALUES (?, ?, ?, ?, ?)",
        (pid, "rag", "demo_2026", "README.md", "STALE README"),
    )
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "README.md": "FRESH README\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    rows = conn.execute(
        "SELECT content FROM readmes_fts WHERE paper_id = ?", (pid,)
    ).fetchall()
    assert len(rows) == 1
    assert "FRESH" in rows[0][0]
    assert "STALE" not in rows[0][0]


def test_readme_ipynb_skipped_for_readmes_fts(conn, monkeypatch):
    pid = _seed_paper(conn)
    nb = {"cells": [{"cell_type": "markdown", "source": "# notebook readme"}]}
    monkeypatch.setattr(fr, "_clone_repo", _FakeClone({
        "README.ipynb": json.dumps(nb).encode(),
        "main.py": "x=1\n",
    }))
    fr.fetch_repo(conn=conn, paper_name="demo_2026")
    assert conn.execute(
        "SELECT COUNT(*) FROM readmes_fts WHERE paper_id = ?", (pid,)
    ).fetchone()[0] == 0
