"""Phase E — microscopy domain tests."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.stain_normalization import MacenkoStainNormalization, ReinhardStainNormalization
from src.utils.config_validator import validate_config
from src.utils.dataset_norm import (
    compute_channel_mean_std,
    imagenet_neutral_bgr_uint8,
    imagenet_neutral_rgb_uint8,
    resolve_normalization_config,
    save_norm_stats,
)
from src.utils.sensor_noise import estimate_gaussian_noise_std, resolve_gauss_noise_var_limit


def _base_config():
    return {
        "model": {"embedding_dim": 128, "pooling": {"strategy": "mean", "layers": [11]}},
        "training": {
            "loss": {
                "type": "sliced_ms",
                "params": {
                    "num_slices": 4,
                    "alpha": 1.0,
                    "beta": 50.0,
                    "base_margin": 0.5,
                },
            }
        },
        "data": {
            "classes_per_batch": 4,
            "test_dir": "data/processed/test",
            "image_size": 252,
            "normalize": {
                "mode": "imagenet",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "grain_detection": {
            "masked_pooling": True,
            "input_size": 320,
            "train_mask_bg_mode": "imagenet_neutral",
            "val_mask_bg_mode": "imagenet_neutral",
        },
        "augmentation": {
            "microscopy_specific": {
                "gaussian_noise": {"enabled": True, "var_limit": [10.0, 50.0], "p": 0.5}
            },
            "sensor_noise_model": {"enabled": False},
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "sota": {"augmentation_mode": "deit3_simple"},
    }


class TestImagenetNeutralBg:
    def test_neutral_maps_near_zero(self):
        r, g, b = imagenet_neutral_rgb_uint8()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for px, m, s in zip((r, g, b), mean, std):
            normed = (px / 255.0 - m) / s
            assert abs(normed) < 0.02

    def test_bgr_order(self):
        b, g, r = imagenet_neutral_bgr_uint8()
        assert (r, g, b) == imagenet_neutral_rgb_uint8()


class TestDatasetNorm:
    def test_compute_mean_std(self):
        imgs = [np.full((32, 32, 3), fill_value=v, dtype=np.uint8) for v in (64, 128, 192)]
        mean, std = compute_channel_mean_std(imgs)
        assert len(mean) == 3
        assert all(s > 0 for s in std)

    def test_resolve_imagenet_mode(self):
        cfg = _base_config()
        resolved = resolve_normalization_config(cfg)
        assert resolved["mode"] == "imagenet"
        assert len(resolved["mean"]) == 3

    def test_save_load_stats(self, tmp_path):
        path = tmp_path / "stats.json"
        save_norm_stats(path, [0.4, 0.4, 0.4], [0.2, 0.2, 0.2], meta={"n": 10})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["mean"] == [0.4, 0.4, 0.4]


class TestStainNormalization:
    def test_reinhard_rgb_uint8(self):
        img = np.random.randint(0, 256, (48, 48, 3), dtype=np.uint8)
        out = ReinhardStainNormalization()(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_macenko_rgb_uint8(self):
        img = np.random.randint(50, 200, (48, 48, 3), dtype=np.uint8)
        out = MacenkoStainNormalization()(img)
        assert out.shape == img.shape


class TestSensorNoise:
    def test_estimate_noise_range(self):
        rng = np.random.default_rng(0)
        base = np.full((64, 64, 3), 128, dtype=np.uint8)
        noisy = np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)
        lo, hi = estimate_gaussian_noise_std([noisy, noisy, noisy])
        assert lo > 0
        assert hi >= lo

    def test_resolve_var_limit_default(self):
        cfg = _base_config()
        lo, hi = resolve_gauss_noise_var_limit(cfg)
        assert lo == 10.0
        assert hi == 50.0


class TestConfigValidatorPhaseE:
    def test_imagenet_neutral_mask_mode_valid(self):
        result = validate_config(_base_config(), check_data_dirs=False)
        assert result["valid"] is True

    def test_u2net_input_size_info_warning(self):
        result = validate_config(_base_config(), check_data_dirs=False)
        assert any("E-05" in w or "U²-Net" in w or "input_size" in w for w in result["warnings"])

    def test_invalid_normalize_mode(self):
        cfg = _base_config()
        cfg["data"]["normalize"]["mode"] = "invalid"
        result = validate_config(cfg, check_data_dirs=False)
        assert result["valid"] is False
