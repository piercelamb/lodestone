"""Taxonomy/classification JSON envelope validation."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ClassificationOutput(BaseModel):
    """Payload from `claude -p --bare --output-format json` in classify_paper."""

    model_config = ConfigDict(extra="forbid")

    domain: str
    domain_is_new: bool
    collection: str
    topics: list[str]
