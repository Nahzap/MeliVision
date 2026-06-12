"""Integrated ablation pipeline (GUI / train.py) tests."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.ablation_matrix import (
    MATRIX_REGISTRY,
    build_variant_config,
    count_planned_variants,
    ensure_ablation_defaults,
    execute_integrated_ablation_study,
    format_ablation_startup_summary,
    get_matrix,
    resolve_enabled_matrices,
    variant_matches_baseline,
)


def _config_with_ablation(enabled=True, matrices=None):
    cfg = {
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
            "epochs": 10,
            "validation_metric": "mAP@R",
            "loss": {"params": {"num_slices": 4}},
        },
        "grain_detection": {"masked_pooling": True, "use_mask": True},
        "sota": {"augmentation_mode": "deit3_simple"},
        "evaluation": {
            "ablation": {
                "enabled": enabled,
                "when": "after_training",
                "skip_baseline_variant": True,
                "on_failure": "continue",
                "epochs": None,
                "matrices": matrices or {"pooling": True, "slices": False},
            }
        },
    }
    return cfg


class TestResolveMatrices:
    def test_dict_selection(self):
        abl = {"matrices": {"pooling": True, "slices": False, "unknown": True}}
        assert resolve_enabled_matrices(abl) == ["pooling"]

    def test_list_selection(self):
        abl = {"matrices": ["augment", "freeze"]}
        assert set(resolve_enabled_matrices(abl)) == {"augment", "freeze"}


class TestVariantBaseline:
    def test_pooling_baseline_match(self):
        cfg = _config_with_ablation()
        variant = get_matrix("pooling")["variants"][0]  # multi_spectral
        assert variant_matches_baseline(cfg, variant) is True

    def test_pooling_gem_not_baseline(self):
        cfg = _config_with_ablation()
        variant = get_matrix("pooling")["variants"][1]  # gem
        assert variant_matches_baseline(cfg, variant) is False


class TestCountPlanned:
    def test_disabled_returns_zero(self):
        cfg = _config_with_ablation(enabled=False)
        assert count_planned_variants(cfg) == 0

    def test_pooling_skips_baseline(self):
        cfg = _config_with_ablation(matrices={"pooling": True})
        # 4 variants, 1 matches baseline
        assert count_planned_variants(cfg) == 3


class TestNestedAblationDisabled:
    def test_child_config_disables_ablation(self):
        cfg = _config_with_ablation()
        variant = get_matrix("pooling")["variants"][1]
        child = build_variant_config(
            cfg, variant, matrix_name="pooling", disable_nested_ablation=True
        )
        assert child["evaluation"]["ablation"]["enabled"] is False


class TestDefaultsAndSummary:
    def test_ensure_ablation_defaults_on_empty(self):
        cfg = {"evaluation": {}}
        assert ensure_ablation_defaults(cfg) is True
        assert "ablation" in cfg["evaluation"]
        assert cfg["evaluation"]["ablation"]["when"] == "after_training"

    def test_format_summary_disabled(self):
        cfg = _config_with_ablation(enabled=False)
        text = format_ablation_startup_summary(cfg)
        assert "informe post-entrenamiento" in text

    def test_format_summary_enabled(self):
        cfg = _config_with_ablation(matrices={"pooling": True})
        text = format_ablation_startup_summary(cfg)
        assert "re-entrenamiento" in text
        assert "pooling" in text


class TestAblationEpochsValidation:
    def test_epochs_zero_allowed_in_validator(self):
        from src.utils.config_validator import validate_config

        cfg = _config_with_ablation(enabled=True, matrices={"pooling": True})
        cfg["evaluation"]["ablation"]["epochs"] = 0
        result = validate_config(cfg)
        assert result["valid"], result["errors"]


class TestExecuteIntegrated:
    def test_disabled_skips(self, tmp_path):
        cfg = _config_with_ablation(enabled=False)
        out = execute_integrated_ablation_study(cfg, "config.yaml", tmp_path)
        assert out["status"] == "skipped"
        assert out["enabled"] is False

    def test_no_matrices_skips(self, tmp_path):
        cfg = _config_with_ablation(matrices={"pooling": False, "slices": False})
        out = execute_integrated_ablation_study(cfg, "config.yaml", tmp_path)
        assert out["status"] == "skipped"
        assert "no matrices" in out.get("reason", "")
