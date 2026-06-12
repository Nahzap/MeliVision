"""Phase A — SOTA integrity tests."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.losses.hier_regularizer import HIERRegularizer
from src.losses.nir_regularizer import NIRRegularizer
from src.utils.config_validator import validate_config
from src.utils.sota_integrity import load_hier_taxonomy, validate_sota_integrity


def _base_config():
    return {
        "model": {"embedding_dim": 128},
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
            "classes_per_batch": 8,
            "samples_per_class": 2,
            "batch_size": 16,
            "test_dir": "data/processed/test",
        },
        "hardware": {"use_gpu": False},
        "evaluation": {"report_final_on": "test"},
        "sota": {
            "nir": {"enabled": False},
            "dada": {"enabled": False, "dual_view_implemented": False},
            "hier": {"enabled": False},
            "self_distillation": {"enabled": False},
        },
    }


class TestNIRRegularizer:
    def test_nir_nonzero_with_enough_samples(self):
        reg = NIRRegularizer(lambda_nir=0.5, min_samples_per_class=2)
        emb = torch.randn(8, 32, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss = reg(emb, labels)
        assert loss.requires_grad
        assert float(loss.item()) != 0.0

    def test_nir_zero_when_insufficient_samples(self):
        reg = NIRRegularizer(min_samples_per_class=4)
        emb = torch.randn(4, 32)
        labels = torch.tensor([0, 1, 2, 3])
        loss = reg(emb, labels)
        assert float(loss.item()) == 0.0


class TestSotaIntegrityValidator:
    def test_dada_enabled_fails(self):
        cfg = _base_config()
        cfg["sota"]["dada"]["enabled"] = True
        errors, _ = validate_sota_integrity(cfg, check_files=False)
        assert any("dada" in e.lower() for e in errors)

    def test_hier_without_taxonomy_fails(self):
        cfg = _base_config()
        cfg["sota"]["hier"]["enabled"] = True
        errors, _ = validate_sota_integrity(cfg, check_files=False)
        assert any("taxonomy_path" in e for e in errors)

    def test_config_validator_blocks_dada(self):
        cfg = _base_config()
        cfg["sota"]["dada"]["enabled"] = True
        result = validate_config(cfg, check_data_dirs=False)
        assert result["valid"] is False


class TestHIERTaxonomy:
    def test_load_taxonomy(self, tmp_path):
        tax = {
            "leaf_to_parents": {
                "0": [0, 0],
                "1": [0, 1],
                "2": [1, 0],
            }
        }
        path = tmp_path / "taxonomy.json"
        path.write_text(json.dumps(tax), encoding="utf-8")
        loaded = load_hier_taxonomy(path)
        assert loaded["leaf_to_parents"][1] == [0, 1]

    def test_hier_uses_taxonomy_not_label_div(self, tmp_path):
        tax = {"leaf_to_parents": {"0": [5], "1": [7]}}
        path = tmp_path / "taxonomy.json"
        path.write_text(json.dumps(tax), encoding="utf-8")

        reg = HIERRegularizer(
            embedding_dim=16,
            num_classes=2,
            num_levels=1,
            taxonomy_path=str(path),
        )
        labels = torch.tensor([0, 1])
        emb = torch.randn(2, 16)
        loss = reg(emb, labels)
        assert torch.isfinite(loss)
