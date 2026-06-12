"""Phase F — ablation matrix tests."""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.ablation_matrix import (
    MATRIX_REGISTRY,
    apply_patch,
    build_variant_config,
    collect_run_metrics,
    get_matrix,
    iter_variants,
    list_matrices,
    write_results_csv,
)


def _minimal_config():
    return {
        "experiment": {"name": "baseline"},
        "model": {
            "embedding_dim": 128,
            "freeze_backbone": False,
            "num_unfrozen_blocks": 2,
            "pooling": {
                "strategy": "multi_spectral",
                "masked": {"enabled": True},
                "layer_fusion": "per_layer",
            },
            "projection": {"layout": "multi_tower"},
        },
        "training": {
            "validation_metric": "mAP@R",
            "loss": {"params": {"num_slices": 4}},
        },
        "grain_detection": {"masked_pooling": True, "use_mask": True},
        "sota": {"augmentation_mode": "deit3_simple"},
    }


class TestMatrixRegistry:
    def test_all_seven_matrices(self):
        assert set(list_matrices()) == {
            "pooling", "slices", "layout", "masked", "augment", "freeze", "validation_metric"
        }

    def test_f01_pooling_variants(self):
        m = get_matrix("pooling")
        labels = [v["label"] for v in m["variants"]]
        assert labels == ["multi_spectral", "gem", "mean", "fsw"]


class TestApplyPatch:
    def test_nested_pooling_strategy(self):
        cfg = apply_patch(_minimal_config(), {"model.pooling.strategy": "gem"})
        assert cfg["model"]["pooling"]["strategy"] == "gem"

    def test_slices_patch(self):
        cfg = apply_patch(_minimal_config(), {"training.loss.params.num_slices": 8})
        assert cfg["training"]["loss"]["params"]["num_slices"] == 8

    def test_masked_off(self):
        cfg = apply_patch(
            _minimal_config(),
            {"grain_detection.masked_pooling": False, "model.pooling.masked.enabled": False},
        )
        assert cfg["grain_detection"]["masked_pooling"] is False
        assert cfg["model"]["pooling"]["masked"]["enabled"] is False


class TestBuildVariant:
    def test_experiment_name_set(self):
        variant = get_matrix("augment")["variants"][0]
        cfg = build_variant_config(_minimal_config(), variant, matrix_name="augment")
        assert cfg["experiment"]["name"] == variant["id"]
        assert "ablation:augment" in cfg["experiment"]["description"]


class TestCollectMetrics:
    def test_collect_from_mock_run(self, tmp_path):
        run = tmp_path / "run_test"
        run.mkdir()
        summary = {
            "metric_learning": {
                "R@1": 0.97,
                "mAP@R": 0.96,
                "NMI": 0.94,
                "F1_macro": 0.95,
            }
        }
        with open(run / "evaluation_metrics_summary_final.json", "w", encoding="utf-8") as f:
            json.dump(summary, f)

        spec_dir = run / "images/post_training/val/spectral_analysis"
        spec_dir.mkdir(parents=True)
        with open(spec_dir / "spectral_summary.json", "w", encoding="utf-8") as f:
            json.dump({"global_effective_rank": 41, "embedding_dim": 128}, f)

        metrics = collect_run_metrics(run)
        assert metrics["R@1"] == 0.97
        assert metrics["effective_rank"] == 41


class TestCsvExport:
    def test_write_csv(self, tmp_path):
        rows = [
            {"matrix": "pooling", "variant_id": "pool_gem", "R@1": 0.9, "mAP@R": 0.88},
            {"matrix": "pooling", "variant_id": "pool_mean", "R@1": 0.85},
        ]
        out = write_results_csv(rows, tmp_path / "results.csv")
        text = out.read_text(encoding="utf-8")
        assert "pool_gem" in text
        assert "mAP@R" in text


class TestIterVariants:
    def test_slice_variants_count(self):
        variants = list(iter_variants("slices", _minimal_config()))
        assert len(variants) == 3
        nums = [v[1]["training"]["loss"]["params"]["num_slices"] for v in variants]
        assert sorted(nums) == [2, 4, 8]
