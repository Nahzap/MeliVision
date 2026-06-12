"""Tests for slice_utils and config_validator."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses.slice_utils import (
    resolve_slice_geometry,
    split_and_normalize_slices,
)
from src.utils.config_validator import build_pipeline_summary, validate_config


class TestSliceUtils:
    def test_resolve_slice_geometry(self):
        assert resolve_slice_geometry(128, 4) == (4, 32)

    def test_resolve_slice_geometry_invalid(self):
        with pytest.raises(ValueError):
            resolve_slice_geometry(128, 5)

    def test_split_and_normalize_slices(self):
        emb = torch.randn(8, 128)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        slices = split_and_normalize_slices(emb, 4)
        assert len(slices) == 4
        for sl in slices:
            assert sl.shape == (8, 32)
            norms = sl.norm(dim=1)
            assert torch.allclose(norms, torch.ones(8), atol=1e-5)


class TestConfigValidator:
    def _base_config(self):
        return {
            "model": {"embedding_dim": 128},
            "training": {
                "loss": {
                    "type": "sliced_ms",
                    "params": {
                        "num_slices": 4,
                        "alpha": 2.0,
                        "beta": 50.0,
                        "base_margin": 0.5,
                    },
                }
            },
            "data": {"classes_per_batch": 8},
            "hardware": {"use_gpu": False},
            "sota": {"self_distillation": {"enabled": False}},
        }

    def test_valid_sliced_ms(self):
        result = validate_config(self._base_config(), check_data_dirs=False)
        assert result["valid"] is True
        assert "sliced_ms" in result["pipeline_summary"]

    def test_invalid_divisibility(self):
        cfg = self._base_config()
        cfg["model"]["embedding_dim"] = 130
        result = validate_config(cfg, check_data_dirs=False)
        assert result["valid"] is False

    def test_orphan_params_warning(self):
        cfg = self._base_config()
        cfg["training"]["loss"]["params"]["proxy_scale"] = 30.0
        result = validate_config(cfg, check_data_dirs=False)
        assert any("proxy_scale" in w for w in result["warnings"])

    def test_build_pipeline_summary(self):
        summary = build_pipeline_summary(self._base_config())
        assert "4×32D" in summary
        assert "self-distill off" in summary

    def test_resolution_mismatch_warning(self):
        cfg = self._base_config()
        cfg["data"] = {"image_size": 252, "resize_size": 504, "classes_per_batch": 8}
        cfg["grain_detection"] = {"crop_size": 504}
        result = validate_config(cfg, check_data_dirs=False)
        assert result["valid"] is True
        assert any("crop_size" in w for w in result["warnings"])
        assert "252px" in build_pipeline_summary(cfg)
