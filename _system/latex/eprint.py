"""Download + extract arxiv e-print archives.

The arxiv ``/e-print/{id}`` endpoint typically returns a gzipped tar of the
LaTeX source, but a small minority of papers submit a single gzipped .tex
file (Content-Type ``application/x-eprint``) and PDF-only submissions
return ``application/pdf`` (we treat those as unsupported and let the
caller fall through to ``failed_html``).

Hardening surface:

- 50MB cap on the encoded download (defends against pathological tarballs).
- 100MB cap on uncompressed payload (zip-bomb defense).
- Tarballs extracted with ``tarfile.data_filter`` (Python 3.12+; available
  in 3.11 via the ``filter`` kwarg). Blocks path traversal, symlinks,
  device files, and absolute paths.
- httpx may auto-decompress gzipped responses depending on the
  ``Accept-Encoding`` header. We sniff the gzip magic bytes (``1f 8b``)
  rather than trusting the Content-Encoding header, so either flow works.
"""
from __future__ import annotations

import gzip
import io
import tarfile
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from _system.utils.logging import get_logger

_LOG = get_logger("latex.eprint")

_ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"

# Encoded payload cap — server-side response size after any HTTP-level
# transfer compression but before our gunzip. arxiv e-prints are
# typically <2 MB; the cap stops a runaway response from chewing memory.
_MAX_ENCODED_BYTES = 50 * 1024 * 1024

# Uncompressed payload cap — protects against zip-bombs where a small
# gzip blob expands to multiple GB. Matches the per-tarball-member
# limit too: tarfile.data_filter alone would not reject a 10 GB single
# member, so we enforce the size at gunzip time.
_MAX_UNCOMPRESSED_BYTES = 100 * 1024 * 1024


class EprintFormat(StrEnum):
    TARBALL = "tarball"
    SINGLE_TEX = "single_tex"


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return isinstance(exc, httpx.TransportError)


_retry_http = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=0.5, max=4.0),
    retry=retry_if_exception(_is_transient),
    reraise=True,
)


@_retry_http
def _do_get(client: httpx.Client, url: str) -> httpx.Response:
    resp = client.get(url)
    if 500 <= resp.status_code < 600:
        resp.raise_for_status()
    return resp


def fetch_eprint(
    client: httpx.Client, arxiv_id: str
) -> tuple[bytes, EprintFormat] | None:
    """GET ``/e-print/{arxiv_id}`` and classify the body.

    Returns ``(payload, format)`` where ``payload`` is the raw response
    body (still gzipped if the server emitted gzip and httpx did not
    auto-decompress). The caller is responsible for gunzipping in
    :func:`extract_to_tempdir`.

    Returns ``None`` on:

    - 4xx (paper has no e-print, or arxiv withdrew it)
    - non-eprint Content-Type (PDF-only submission, etc.)
    - encoded body > _MAX_ENCODED_BYTES
    """
    url = _ARXIV_EPRINT_URL.format(arxiv_id=arxiv_id)
    resp = _do_get(client, url)
    if 400 <= resp.status_code < 500:
        _LOG.info("e-print %s returned %d, no LaTeX source available",
                  arxiv_id, resp.status_code)
        return None
    if resp.status_code != 200:
        _LOG.warning("e-print %s returned %d, skipping", arxiv_id, resp.status_code)
        return None

    payload = resp.content
    if len(payload) > _MAX_ENCODED_BYTES:
        _LOG.warning(
            "e-print %s payload %d bytes exceeds %d cap, skipping",
            arxiv_id, len(payload), _MAX_ENCODED_BYTES,
        )
        return None

    content_type = (resp.headers.get("content-type") or "").lower()
    if "tar" in content_type or content_type.startswith("application/x-eprint-tar"):
        return payload, EprintFormat.TARBALL
    if content_type.startswith("application/x-eprint"):
        # arxiv emits this for single-file submissions and (legacy) for some
        # tarballs. We disambiguate by sniffing magic bytes after gunzip in
        # extract_to_tempdir; the SINGLE_TEX hint is only advisory.
        return payload, EprintFormat.SINGLE_TEX
    if content_type.startswith("application/pdf"):
        _LOG.info("e-print %s is PDF-only, no LaTeX source", arxiv_id)
        return None
    if content_type.startswith("application/gzip") or "gzip" in content_type:
        # Some mirrors return a generic gzip mime; treat as tarball-or-tex
        # and let extraction sort it out.
        return payload, EprintFormat.TARBALL
    _LOG.warning(
        "e-print %s returned unexpected content-type %r, skipping",
        arxiv_id, content_type,
    )
    return None


def _gunzip_if_needed(blob: bytes) -> bytes:
    """If ``blob`` starts with the gzip magic bytes, decompress it.

    Otherwise return as-is. Raises if uncompressed size exceeds
    ``_MAX_UNCOMPRESSED_BYTES``.
    """
    if blob[:2] != b"\x1f\x8b":
        return blob
    out = bytearray()
    with gzip.GzipFile(fileobj=io.BytesIO(blob), mode="rb") as gz:
        while True:
            chunk = gz.read(1024 * 1024)
            if not chunk:
                break
            out.extend(chunk)
            if len(out) > _MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"gunzipped payload exceeds {_MAX_UNCOMPRESSED_BYTES} bytes "
                    "(possible zip bomb)"
                )
    return bytes(out)


def _looks_like_tar(data: bytes) -> bool:
    """Heuristic: USTAR magic at offset 257 (the tar header standard)."""
    if len(data) < 512:
        return False
    return data[257:262] == b"ustar"


@contextmanager
def extract_to_tempdir(
    blob: bytes, fmt: EprintFormat
) -> Iterator[Path]:
    """Yield a Path to a tempdir containing the extracted source tree.

    The tempdir is cleaned up when the context exits.

    Raises on malformed input — the caller (``_try_latex_fallback``) traps
    the exception and falls through to ``failed_html`` rather than letting
    the whole pipeline crash.
    """
    decoded = _gunzip_if_needed(blob)

    with TemporaryDirectory(prefix="lodestone-eprint-") as td:
        root = Path(td)
        if fmt is EprintFormat.TARBALL or _looks_like_tar(decoded):
            with tarfile.open(fileobj=io.BytesIO(decoded), mode="r:*") as tf:
                # extraction_filter='data' applies tarfile.data_filter:
                # rejects symlinks, device files, absolute paths, parent
                # references. Available in Python >=3.12; Python 3.11 raises
                # DeprecationWarning if not supplied but still defaults to
                # 'fully_trusted' — pyproject pins >=3.11 so we set it
                # explicitly.
                try:
                    tf.extractall(path=root, filter="data")
                except TypeError:
                    # Older python tarfile that lacks the filter kwarg.
                    # Project requires 3.11+, but be defensive.
                    tf.extractall(path=root)
            yield root
            return

        # Single-tex path: write decoded directly as main.tex.
        main = root / "main.tex"
        main.write_bytes(decoded)
        yield root
