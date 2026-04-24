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
            "method": 0.60,
            "dataset": 0.40,
            "metric": 0.55,
            "model": 0.55,
            "benchmark": 0.45,
            "software": 0.50,
            "system": 0.50,
            "organization": 0.50,
            "venue": 0.50,
        },
        "chunk": {"max_tokens": 350, "overlap_tokens": 20},
        "label_descriptions": {
            "method": "A named technique, algorithm, procedure, or task.",
            "dataset": "A data collection.",
            "metric": "A quantitative measure.",
            "model": "A trained model.",
            "benchmark": "A benchmark suite.",
            "software": "A library or tool.",
            "system": "A database or service.",
            "organization": "A company or research lab.",
            "venue": "A conference or journal.",
        },
    }


class TestGlinerConfigValidation:
    def test_accepts_valid_config(self):
        cfg = GlinerConfig.model_validate(_valid_yaml())
        assert cfg.global_threshold == 0.45
        assert cfg.chunk.max_tokens == 350
        assert cfg.chunk.overlap_tokens == 20
        assert set(cfg.per_label) >= {"method", "dataset"}

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
        data["per_label"]["method"] = 1.5
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)

    def test_rejects_empty_label_description(self):
        data = _valid_yaml()
        data["label_descriptions"]["method"] = "   "
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)

    def test_rejects_key_set_mismatch(self):
        data = _valid_yaml()
        del data["label_descriptions"]["benchmark"]
        with pytest.raises(ValidationError):
            GlinerConfig.model_validate(data)


class TestLoadGlinerConfig:
    def test_loads_default_file(self):
        cfg = load_gliner_config()
        assert cfg.global_threshold == 0.45
        assert cfg.chunk.max_tokens == 350
        expected = {
            "method", "dataset", "metric", "model", "benchmark",
            "software", "system", "organization", "venue",
        }
        # Lowercase labels — matches the Fastino GLiNER2 training distribution.
        assert expected <= set(cfg.per_label)
        assert expected <= set(cfg.label_descriptions)

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
