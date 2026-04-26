"""Entity-type label vocabulary shared by the GLiNER2 inference stage and the
``canonical_terms.entity_type`` column."""
from __future__ import annotations

from enum import StrEnum


class EntityType(StrEnum):
    METHOD = "method"
    DATASET = "dataset"
    METRIC = "metric"
    MODEL = "model"
    BENCHMARK = "benchmark"
    SOFTWARE = "software"
    SYSTEM = "system"
    ORGANIZATION = "organization"
    VENUE = "venue"
