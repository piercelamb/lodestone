"""PDF -> markdown extraction via pymupdf4llm."""
from __future__ import annotations

import pymupdf
import pymupdf4llm

from _system.utils.logging import get_logger

_LOG = get_logger("pdf.extract")


def extract_markdown(
    pdf_bytes: bytes, *, pages: list[int] | None = None,
) -> str | None:
    """Convert PDF bytes to markdown via pymupdf4llm.

    ``pages`` (0-based, matches pymupdf4llm) restricts rendering to a
    page subset — used by the local-PDF chapter splitter. ``None``
    renders the whole document. Returns the stripped markdown string,
    or None if extraction produced nothing usable (empty body) or
    pymupdf raised on the input.
    """
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # pymupdf raises a variety of errors on bad input
        _LOG.warning("pymupdf failed to open PDF stream: %s", exc)
        return None

    kwargs: dict = {
        "write_images": False,
        "embed_images": False,
        "show_progress": False,
    }
    if pages is not None:
        kwargs["pages"] = pages
    try:
        md = pymupdf4llm.to_markdown(doc, **kwargs)
    except Exception as exc:
        _LOG.warning("pymupdf4llm.to_markdown failed: %s", exc)
        return None
    finally:
        doc.close()

    md = (md or "").strip()
    if not md:
        return None
    return md
