"""Unit tests for `_system/latex/eprint.py`."""
from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path

import httpx
import pytest

from _system.latex.eprint import (
    EprintFormat,
    extract_to_tempdir,
    fetch_eprint,
)


FIXTURE = Path(__file__).parent / "fixtures" / "latex" / "simple.tar.gz"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def test_fetch_eprint_returns_tarball_on_tar_content_type():
    blob = FIXTURE.read_bytes()

    def handler(req):
        return httpx.Response(
            200, content=blob,
            headers={"content-type": "application/x-eprint-tar"},
        )

    with _client(handler) as c:
        result = fetch_eprint(c, "2510.07233")
    assert result is not None
    payload, fmt = result
    assert fmt is EprintFormat.TARBALL
    assert payload == blob


def test_fetch_eprint_returns_none_on_404():
    with _client(lambda r: httpx.Response(404)) as c:
        assert fetch_eprint(c, "9999.99999") is None


def test_fetch_eprint_returns_none_on_pdf_only():
    def handler(req):
        return httpx.Response(
            200, content=b"%PDF-1.7",
            headers={"content-type": "application/pdf"},
        )
    with _client(handler) as c:
        assert fetch_eprint(c, "2510.07233") is None


def test_fetch_eprint_caps_oversized_payload():
    big = b"\x00" * (60 * 1024 * 1024)  # 60 MB > 50 MB cap

    def handler(req):
        return httpx.Response(
            200, content=big,
            headers={"content-type": "application/x-eprint-tar"},
        )
    with _client(handler) as c:
        assert fetch_eprint(c, "2510.07233") is None


def test_extract_to_tempdir_handles_real_tarball():
    blob = FIXTURE.read_bytes()
    with extract_to_tempdir(blob, EprintFormat.TARBALL) as root:
        assert (root / "main.tex").is_file()
        assert (root / "f.png").is_file()
        assert b"\\section{Introduction}" in (root / "main.tex").read_bytes()


def test_extract_to_tempdir_handles_single_tex():
    raw = b"\\documentclass{article}\\begin{document}Hi.\\end{document}"
    blob = gzip.compress(raw)
    with extract_to_tempdir(blob, EprintFormat.SINGLE_TEX) as root:
        # When it's not a tarball, _looks_like_tar returns False and the
        # single-tex branch fires.
        files = list(root.iterdir())
        assert len(files) == 1
        assert files[0].name == "main.tex"
        assert files[0].read_bytes() == raw


def test_extract_to_tempdir_blocks_path_traversal():
    """tarfile.data_filter rejects entries that escape the extraction root."""
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w") as tf:
        evil = tarfile.TarInfo(name="../escape.tex")
        data = b"oops"
        evil.size = len(data)
        tf.addfile(evil, io.BytesIO(data))
    blob = gzip.compress(inner.getvalue())
    with pytest.raises(Exception):
        with extract_to_tempdir(blob, EprintFormat.TARBALL) as root:
            pass


def test_extract_to_tempdir_rejects_zip_bomb():
    """A gzipped payload that decompresses past 100 MB is rejected."""
    big = b"A" * (120 * 1024 * 1024)
    blob = gzip.compress(big)
    with pytest.raises(ValueError):
        with extract_to_tempdir(blob, EprintFormat.SINGLE_TEX) as root:
            pass
