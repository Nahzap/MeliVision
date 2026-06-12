"""Phase D — SOTA architecture tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.transforms_enhanced import DeiT3Augmentation, get_train_transforms
from src.losses.anti_collapse_loss import AntiCollapseLoss
from src.models.analogy_net import AnalogyNet
from src.models.pooling import PoolingModule
from src.models.pooling_config import resolve_pooling_config
from src.training.trainer_enhanced import MetricLearningTrainer
from src.utils.config_validator import validate_config


def _base_config():
    return {
        "model": {
            "embedding_dim": 128,
            "pooling": {"strategy": "mean", "layers": [11]},
            "projection_head": {
                "use_batch_norm": False,
                "use_layer_norm": True,
            },
        },
        "training": {
            "epochs": 10,
            "warmup_epochs": 0,
            "learning_rate": 1e-4,
            "validation_metric": "mAP@R",
            "finetune_phases": [
                {"epochs": 5, "freeze_backbone": True, "lr_factor": 1.0},
                {"epochs": 5, "freeze_backbone": False, "lr_factor": 0.5},
            ],
            "loss": {
                "type": "sliced_ms",
                "distance": "euclidean",
                "params": {
                    "num_slices": 4,
                    "alpha": 1.0,
                    "beta": 50.0,
                    "base_margin": 0.5,
                },
            },
        },
        "data": {"classes_per_batch": 8, "test_dir": "data/processed/test"},
        "hardware": {"use_gpu": False},
        "evaluation": {"report_final_on": "test"},
        "sota": {
            "augmentation_mode": "deit3_simple",
            "anti_collapse": {"enabled": False, "lambda": 0.1, "t": 2.0},
            "nir": {"enabled": False},
            "dada": {"enabled": False, "dual_view_implemented": False},
            "hier": {"enabled": False},
        },
    }


class TestAntiCollapseLoss:
    def test_uniformity_nonzero(self):
        loss_fn = AntiCollapseLoss(lambda_ac=0.1, t=2.0)
        emb = torch.randn(8, 32, requires_grad=True)
        loss = loss_fn(emb)
        assert loss.requires_grad
        assert float(loss.item()) != 0.0


class TestFSWPooling:
    def test_fsw_iterative_pool(self):
        cfg = resolve_pooling_config(
            {
                "pooling": {
                    "strategy": "fsw",
                    "layers": [11],
                    "fsw": {"num_iterations": 3},
                    "drop_token": {"enabled": False},
                }
            }
        )
        pool = PoolingModule(cfg, token_dim=16, patch_size=14)
        tokens = {11: torch.randn(1, 4, 16)}
        out = pool(tokens, image_hw=(28, 28))
        assert out.shape == (1, 16)


class TestDeiT3Augment:
    def test_deit3_simple_alias(self):
        aug = get_train_transforms(config={"image_size": 224}, mode="deit3_simple")
        assert isinstance(aug, DeiT3Augmentation)


class TestFinetunePhases:
    def test_phase_boundaries(self):
        trainer = MetricLearningTrainer.__new__(MetricLearningTrainer)
        trainer.finetune_phases = [
            {"epochs": 3, "freeze_backbone": True},
            {"epochs": 7, "freeze_backbone": False},
        ]
        boundaries = MetricLearningTrainer._build_finetune_phase_boundaries(trainer)
        assert boundaries == [3, 10]


class TestLayerNormProjection:
    def test_projection_uses_layer_norm_not_bn(self):
        model = AnalogyNet(
            backbone="resnet18",
            embedding_dim=128,
            pretrained=False,
            projection_head_config={
                "hidden_dims": [64],
                "use_batch_norm": False,
                "use_layer_norm": True,
            },
            projection_layout="slice_native",
            pooling_config={"strategy": "mean", "layers": [11]},
        )
        has_ln = any(isinstance(m, nn.LayerNorm) for m in model.projection.modules())
        has_bn = any(isinstance(m, nn.BatchNorm1d) for m in model.projection.modules())
        assert has_ln
        assert not has_bn


class TestSetBackboneFrozen:
    def test_runtime_freeze_toggle(self):
        model = AnalogyNet(
            backbone="resnet18",
            embedding_dim=128,
            pretrained=False,
            freeze_backbone=False,
            projection_layout="slice_native",
            pooling_config={"strategy": "mean", "layers": [11]},
        )
        model.set_backbone_frozen(True, num_unfrozen_blocks=0)
        assert model.freeze_backbone is True
        frozen = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        assert frozen > 0
        model.set_backbone_frozen(False)
        assert all(p.requires_grad for p in model.backbone.parameters())


class TestConfigValidatorPhaseD:
    def test_augmentation_mode_deit3_simple(self):
        cfg = _base_config()
        result = validate_config(cfg, check_data_dirs=False)
        assert result["valid"] is True

    def test_embedding_dim_warning(self):
        cfg = _base_config()
        cfg["model"]["embedding_dim"] = 512
        result = validate_config(cfg, check_data_dirs=False)
        assert any("D-02" in w for w in result["warnings"])

    def test_anti_collapse_enabled_invalid_lambda(self):
        cfg = _base_config()
        cfg["sota"]["anti_collapse"]["enabled"] = True
        cfg["sota"]["anti_collapse"]["lambda"] = 0
        result = validate_config(cfg, check_data_dirs=False)
        assert result["valid"] is False


class TestTrainerAntiCollapse:
    def test_setup_anti_collapse_when_enabled(self):
        trainer = MetricLearningTrainer.__new__(MetricLearningTrainer)
        trainer.config = {"anti_collapse": {"enabled": True, "lambda": 0.2, "t": 2.0}}
        trainer.logger = MagicMock()
        reg = MetricLearningTrainer._setup_anti_collapse(trainer)
        assert reg is not None
        assert isinstance(reg, AntiCollapseLoss)
