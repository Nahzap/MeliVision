"""Tests for multi_tower projection layout (S independent slice towers)."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.analogy_net import AnalogyNet
from src.models.pooling_config import (
    resolve_pooling_config,
    resolve_projection_config,
    resolve_slice_layer_map,
    validate_projection_config,
)
from src.utils.config_validator import build_pipeline_summary, validate_config


POOLING_CFG = {
    "strategy": "multi_spectral",
    "layers": [3, 11],
    "statistics": ["mean", "max", "std"],
    "layer_fusion": "per_layer",
    "drop_token": {"enabled": False},
    "masked": {"enabled": False},
}

PROJECTION_CFG = {"layout": "multi_tower", "slice_layer_map": [11, 11, 3, 3]}


def _make_multi_tower(num_slices=4, embedding_dim=128, pretrained=False):
    return AnalogyNet(
        backbone="resnet18",
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        projection_layout="multi_tower",
        num_slices=num_slices,
        projection_config=PROJECTION_CFG,
        pooling_config=POOLING_CFG,
    )


def test_resolve_slice_layer_map_default():
    pooling = resolve_pooling_config({"pooling": POOLING_CFG})
    proj = resolve_projection_config({"projection": {"layout": "multi_tower"}})
    assert resolve_slice_layer_map(proj, pooling, 4) == [11, 11, 3, 3]


def test_resolve_slice_layer_map_custom():
    pooling = resolve_pooling_config({"pooling": POOLING_CFG})
    proj = resolve_projection_config({"projection": PROJECTION_CFG})
    assert resolve_slice_layer_map(proj, pooling, 4) == [11, 11, 3, 3]


def test_validate_projection_config_ok():
    pooling = resolve_pooling_config({"pooling": POOLING_CFG})
    proj = resolve_projection_config({"projection": PROJECTION_CFG})
    errors, _ = validate_projection_config(proj, pooling, 128, 4)
    assert errors == []


def test_multi_tower_init_has_independent_towers():
    model = _make_multi_tower()
    assert model.projection_towers is not None
    assert len(model.projection_towers) == 4
    assert model.slice_dim == 32
    assert model.slice_layer_map == [11, 11, 3, 3]

    w0 = model.projection_towers[0][0].weight
    w1 = model.projection_towers[1][0].weight
    assert not torch.allclose(w0, w1), "Torres deben tener pesos independientes"


def test_multi_tower_project_features_shape_and_norm():
    model = _make_multi_tower()
    model.eval()
    per_layer_dim = 512  # resnet18
    feat_shallow = torch.randn(2, per_layer_dim)
    feat_deep = torch.randn(2, per_layer_dim)
    features = (feat_shallow, feat_deep)  # sorted layers [3, 11]

    with torch.no_grad():
        embeddings, prenorm = model._project_features(features)

    assert embeddings.shape == (2, 128)
    assert prenorm.shape == (2, 128)
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    slice_blocks = prenorm.view(2, 4, 32)
    slice_norms = torch.norm(slice_blocks, p=2, dim=2)
    assert torch.allclose(slice_norms, torch.ones(2, 4), atol=1e-5)
    expected = prenorm / math.sqrt(4)
    assert torch.allclose(embeddings, expected, atol=1e-5)


def test_multi_tower_slice_layer_routing():
    model = _make_multi_tower()
    model.eval()
    per_layer_dim = 512
    shallow = torch.full((1, per_layer_dim), 1.0)
    deep = torch.full((1, per_layer_dim), 9.0)
    features = (shallow, deep)

    with torch.no_grad():
        _, prenorm = model._project_features(features)

    # slice 0,1 → layer 11 (deep=9); slice 2,3 → layer 3 (shallow=1)
    for sl_idx, expected_val in enumerate([9.0, 9.0, 1.0, 1.0]):
        tower = model.projection_towers[sl_idx]
        layer_idx = model.slice_layer_map[sl_idx]
        feat = deep if layer_idx == 11 else shallow
        ref = torch.nn.functional.normalize(tower(feat), p=2, dim=1)
        assert torch.allclose(prenorm[0, sl_idx * 32 : (sl_idx + 1) * 32], ref[0], atol=1e-4)


def test_config_validator_multi_tower_summary():
    config = {
        "model": {
            "embedding_dim": 128,
            "pooling": POOLING_CFG,
            "projection": PROJECTION_CFG,
        },
        "training": {"loss": {"type": "sliced_ms", "params": {"num_slices": 4}}},
        "data": {"classes_per_batch": 4},
        "grain_detection": {"masked_pooling": False},
    }
    result = validate_config(config, check_data_dirs=False)
    assert result["valid"] is True
    summary = build_pipeline_summary(config)
    assert "multi_tower×4" in summary


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DINOv2 smoke opcional sin GPU")
def test_multi_tower_dinov2_forward_smoke():
    """Forward end-to-end con DINOv2 (opcional, requiere GPU)."""
    model = AnalogyNet(
        backbone="dinov2_vits14",
        embedding_dim=128,
        pretrained=False,
        projection_layout="multi_tower",
        num_slices=4,
        projection_config=PROJECTION_CFG,
        pooling_config={**POOLING_CFG, "masked": {"enabled": True}},
    )
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    mask = torch.ones(1, 224, 224)
    with torch.no_grad():
        out = model(x, mask=mask)
    assert out.shape == (1, 128)
