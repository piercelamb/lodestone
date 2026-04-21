"""Tests for the GLiNER2 config YAML loader."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from _system.utils.config import GlinerConfig, load_gliner_config


def _valid_yaml() -> dict:
    return {
        "global_threshold": 0.45,
        "per_label": {
            "Method": 0.55,
            "Dataset": 0.40,
            "Metric": 0.50,
            "Model": 0.55,
            "Technique": 0.50,
            "Benchmark": 0.45,
        },
        "chunk": {"max_tokens": 350, "overlap_tokens": 20},
        "label_descriptions": {
            "Method": "A named technique.",
            "Dataset": "A data collection.",
            "Metric": "A quantitative measure.",
            "Model": "A trained model.",
            "Technique": "A building-block technique.",
            "Benchmark": "A benchmark suite.",
        },
    }


class TestGlinerConfigValidation:
    def test_accepts_valid_config(self):
        cfg = GlinerConfig.model_validate(_valid_yaml())
        assert cfg.global_threshold == 0.45
        assert cfg.chunk.max_tokens == 350
        assert cfg.chunk.overlap_tokens == 20
        assert set(cfg.per_label) >= {"Method", "Dataset"}

    def test_rejects_missing_global_threshold(self):
        data = _valid_yaml()
        del data["global_threshold"]
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)

    def test_rejects_missing_chunk(self):
        data = _valid_yaml()
        del data["chunk"]
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)

    def test_rejects_chunk_max_tokens_above_384(self):
        data = _valid_yaml()
        data["chunk"]["max_tokens"] = 385
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)

    def test_accepts_chunk_max_tokens_at_limit(self):
        data = _valid_yaml()
        data["chunk"]["max_tokens"] = 384
        cfg = GlinerConfig.model_validate(data)
        assert cfg.chunk.max_tokens == 384

    def test_rejects_overlap_geq_max(self):
        data = _valid_yaml()
        data["chunk"]["overlap_tokens"] = data["chunk"]["max_tokens"]
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)

    def test_rejects_global_threshold_out_of_range(self):
        data = _valid_yaml()
        data["global_threshold"] = 1.5
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)

    def test_rejects_per_label_threshold_out_of_range(self):
        data = _valid_yaml()
        data["per_label"]["Method"] = 1.5
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)

    def test_rejects_empty_label_description(self):
        data = _valid_yaml()
        data["label_descriptions"]["Method"] = "   "
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)

    def test_rejects_key_set_mismatch(self):
        data = _valid_yaml()
        del data["label_descriptions"]["Benchmark"]
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)


class TestLoadGlinerConfig:
    def test_loads_default_file(self):
        cfg = load_gliner_config()
        assert cfg.global_threshold == 0.45
        assert cfg.chunk.max_tokens == 350
        # Title-case labels required by GLiNER2.
        assert {"Method", "Dataset", "Metric", "Model", "Technique", "Benchmark"} <= set(
            cfg.per_label
        )
        assert {"Method", "Dataset", "Metric", "Model", "Technique", "Benchmark"} <= set(
            cfg.label_descriptions
        )

    def test_loads_from_custom_path(self, tmp_path: Path):
        p = tmp_path / "g.yaml"
        p.write_text(yaml.safe_dump(_valid_yaml()))
        cfg = load_gliner_config(p)
        assert cfg.global_threshold == 0.45

    def test_raises_on_bad_file(self, tmp_path: Path):
        data = _valid_yaml()
        data["chunk"]["max_tokens"] = 500
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(data))
        with pytest.raises(ValidationError):
            load_gliner_config(p)
