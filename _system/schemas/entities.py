"""Entity data contracts produced by the GLiNER2 extraction stage."""
from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel


class EntityType(StrEnum):
    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    MODEL = "model"
    TECHNIQUE = "technique"
    BENCHMARK = "benchmark"


class Entity(BaseModel):
    name: str
    type: EntityType
    aliases: list[str] = []
    source_section: str
    description: Optional[str] = None


class PaperEntities(BaseModel):
    paper_name: str
    domain: str
    entities: list[Entity]
