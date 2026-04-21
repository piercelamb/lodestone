"""Figure data contract produced by the LaTeXML parser."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class PaperFigure(BaseModel):
    figure_number: int
    figure_id: str
    caption: str
    section_context: str
    image_data: bytes
    mime_type: str
    display_number: Optional[str] = None
