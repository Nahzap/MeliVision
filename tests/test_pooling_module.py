import torch

from src.models.pooling import PoolingModule
from src.models.pooling_config import (
    compute_pooling_output_dim,
    resolve_pooling_config,
    validate_pooling_config,
)
from src.utils.config_validator import validate_config


def test_multi_spectral_output_dim():
    dim = compute_pooling_output_dim(384, "multi_spectral", ["mean", "max", "std"], 2, "per_layer")
    assert dim == 1152


def test_pooling_mean_masked():
    cfg = resolve_pooling_config(
        {"pooling": {"strategy": "mean", "layers": [11], "drop_token": {"enabled": False}}},
        {"masked_pooling": True},
    )
    pool = PoolingModule(cfg, token_dim=32, patch_size=14)
    tokens = {11: torch.randn(2, 16, 32)}
    mask = torch.zeros(2, 56, 56)
    mask[:, 14:42, 14:42] = 1.0
    out = pool(tokens, spatial_mask=mask, image_hw=(56, 56))
    assert out.shape == (2, 32)


def test_pooling_per_layer_tuple():
    cfg = resolve_pooling_config(
        {
            "pooling": {
                "strategy": "multi_spectral",
                "layers": [3, 11],
                "statistics": ["mean", "max", "std"],
                "layer_fusion": "per_layer",
                "drop_token": {"enabled": False},
            }
        }
    )
    pool = PoolingModule(cfg, token_dim=16, patch_size=14)
    tokens = {3: torch.randn(1, 4, 16), 11: torch.randn(1, 4, 16)}
    out = pool(tokens, image_hw=(28, 28))
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == (1, 48)


def test_validate_pooling_config_ok():
    cfg = resolve_pooling_config(
        {
            "pooling": {
                "strategy": "multi_spectral",
                "layers": [3, 11],
                "statistics": ["mean", "max", "std"],
            }
        },
        {"masked_pooling": True},
    )
    errors, warnings = validate_pooling_config(cfg, "lf_hf_dual")
    assert errors == []


def test_fsw_pooling_output_shape():
    cfg = resolve_pooling_config(
        {
            "pooling": {
                "strategy": "fsw",
                "layers": [11],
                "fsw": {"num_iterations": 2, "temperature": 1.0},
                "drop_token": {"enabled": False},
            }
        }
    )
    pool = PoolingModule(cfg, token_dim=32, patch_size=14)
    tokens = {11: torch.randn(2, 16, 32)}
    out = pool(tokens, image_hw=(56, 56))
    assert out.shape == (2, 32)


def test_config_validator_embedding_divisible():
    config = {
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
        "data": {"classes_per_batch": 4, "test_dir": "data/processed/test"},
        "grain_detection": {"masked_pooling": True},
        "sota": {"augmentation_mode": "deit3_simple"},
    }
    result = validate_config(config, check_data_dirs=False)
    assert result["valid"] is True
