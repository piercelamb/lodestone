"""Config loaders. Currently hosts the GLiNER2 YAML config loader."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ChunkConfig(BaseModel):
    max_tokens: int = Field(..., gt=0, le=384)
    overlap_tokens: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _overlap_strictly_less_than_max(self) -> "ChunkConfig":
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                "chunk.overlap_tokens must be strictly less than chunk.max_tokens"
            )
        return self


class GlinerConfig(BaseModel):
    global_threshold: float = Field(..., ge=0.0, le=1.0)
    per_label: dict[str, float]
    chunk: ChunkConfig
    label_descriptions: dict[str, str]

    @field_validator("per_label")
    @classmethod
    def _per_label_range(cls, v: dict[str, float]) -> dict[str, float]:
        for label, val in v.items():
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"per_label[{label!r}]={val} must be in [0.0, 1.0]"
                )
        return v

    @field_validator("label_descriptions")
    @classmethod
    def _descriptions_non_empty(cls, v: dict[str, str]) -> dict[str, str]:
        for label, desc in v.items():
            if not desc or not desc.strip():
                raise ValueError(
                    f"label_descriptions[{label!r}] must be non-empty"
                )
        return v

    @model_validator(mode="after")
    def _label_keys_agree(self) -> "GlinerConfig":
        per_label = set(self.per_label)
        descriptions = set(self.label_descriptions)
        if per_label != descriptions:
            missing_in_desc = per_label - descriptions
            missing_in_thresh = descriptions - per_label
            raise ValueError(
                "per_label and label_descriptions must share the same keys "
                f"(only in per_label: {sorted(missing_in_desc)}, "
                f"only in label_descriptions: {sorted(missing_in_thresh)})"
            )
        return self


_DEFAULT_GLINER_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "gliner.yaml"
)


@lru_cache(maxsize=8)
def _cached_load(path_str: str) -> GlinerConfig:
    with open(path_str, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return GlinerConfig.model_validate(data)


def load_gliner_config(path: Path | str = _DEFAULT_GLINER_PATH) -> GlinerConfig:
    """Load and validate the GLiNER2 YAML config.

    Raises `pydantic.ValidationError` on malformed input (per project convention,
    errors propagate rather than being swallowed).
    """
    return _cached_load(str(Path(path)))
