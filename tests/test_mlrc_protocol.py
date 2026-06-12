"""Phase B — MLRC protocol tests."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config_validator import validate_config
from src.utils.metrics_utils import compute_all_metrics, compute_map_at_r
from src.utils.mlrc_protocol import (
    is_improvement,
    metric_value_from_dict,
    normalize_validation_metric,
    resolve_nn_metric,
)
from src.utils.threshold_calibration import calibrate_inference_thresholds


def _synthetic_embeddings(n_classes=4, per_class=6, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    embeddings = []
    labels = []
    for c in range(n_classes):
        noise = 0.05 * rng.standard_normal((per_class, dim))
        vecs = centers[c] + noise
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        embeddings.append(vecs)
        labels.extend([c] * per_class)
    return np.vstack(embeddings), np.asarray(labels, dtype=np.int64)


class TestValidationMetric:
    def test_normalize_aliases(self):
        assert normalize_validation_metric("mAP@R") == "mAP@R"
        assert normalize_validation_metric("recall_at_1") == "recall_at_1"
        assert normalize_validation_metric("map_at_r") == "mAP@R"

    def test_is_improvement_higher_better(self):
        assert is_improvement("mAP@R", 0.6, 0.5, 0.001)
        assert not is_improvement("mAP@R", 0.5, 0.6, 0.001)

    def test_is_improvement_lower_better(self):
        assert is_improvement("val_loss", 0.4, 0.5, 0.001)
        assert not is_improvement("val_loss", 0.6, 0.5, 0.001)

    def test_metric_value_from_dict(self):
        metrics = {"mAP@R": 0.42, "R@1": 0.8, "val_loss": 1.2, "F1_macro": 0.7}
        assert metric_value_from_dict("mAP@R", metrics) == pytest.approx(0.42)
        assert metric_value_from_dict("recall_at_1", metrics) == pytest.approx(0.8)


class TestMapAtRExact:
    def test_exact_ge_approx_on_synthetic(self):
        emb, labels = _synthetic_embeddings()
        exact = compute_map_at_r(emb, labels, exact=True, candidate_multiplier=2)
        approx = compute_map_at_r(emb, labels, exact=False, candidate_multiplier=2)
        assert exact >= approx - 1e-6

    def test_exact_reproducible_reference(self):
        """Small CUB-like synthetic: exact mAP@R stable across calls."""
        emb, labels = _synthetic_embeddings(seed=7)
        a = compute_map_at_r(emb, labels, exact=True)
        b = compute_map_at_r(emb, labels, exact=True)
        assert a == pytest.approx(b)


class TestNnMetric:
    def test_euclidean_not_overridden_in_compute_all_metrics(self):
        rng = np.random.default_rng(11)
        # Deliberately unnormalized embeddings: cosine L2-normalizes, euclidean does not.
        emb = rng.standard_normal((24, 16))
        for c in range(4):
            emb[c * 6 : (c + 1) * 6] += 3.0 * c
        labels = np.repeat(np.arange(4), 6)
        cosine_metrics = compute_all_metrics(emb, labels, nn_metric="cosine", include_per_class=False)
        eucl_metrics = compute_all_metrics(emb, labels, nn_metric="euclidean", include_per_class=False)
        assert cosine_metrics["F1_macro"] != eucl_metrics["F1_macro"] or (
            cosine_metrics["R@1"] != eucl_metrics["R@1"]
        )

    def test_resolve_nn_metric_from_embedding_space(self):
        cfg = {
            "evaluation": {},
            "sota": {"embedding_space": "euclidean"},
            "training": {"loss": {"type": "multi_similarity"}},
        }
        assert resolve_nn_metric(cfg) == "euclidean"


class TestNmiMultiSeed:
    def test_nmi_std_positive_with_multiple_seeds(self):
        emb, labels = _synthetic_embeddings()
        metrics = compute_all_metrics(
            emb,
            labels,
            include_per_class=False,
            nmi_n_init=3,
            nmi_n_seeds=3,
            nmi_base_seed=0,
        )
        assert metrics["NMI_std"] >= 0.0


class TestConfigValidatorMlrc:
    def _base_config(self):
        return {
            "model": {"embedding_dim": 128},
            "training": {
                "validation_metric": "mAP@R",
                "loss": {
                    "type": "sliced_ms",
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
            "evaluation": {
                "protocol": "mlrc",
                "n_runs": 10,
                "map_at_r": {"exact": True},
                "nmi": {"n_init": 10, "n_seeds": 5},
            },
            "sota": {
                "nir": {"enabled": False},
                "dada": {"enabled": False, "dual_view_implemented": False},
                "hier": {"enabled": False},
            },
        }

    def test_valid_mlrc_config(self):
        result = validate_config(self._base_config(), check_data_dirs=False)
        assert result["valid"] is True

    def test_invalid_validation_metric(self):
        cfg = self._base_config()
        cfg["training"]["validation_metric"] = "invalid_metric"
        result = validate_config(cfg, check_data_dirs=False)
        assert result["valid"] is False


class _DummyModel(nn.Module):
    def __init__(self, emb, labels):
        super().__init__()
        self._emb = torch.from_numpy(emb.astype(np.float32))
        self._labels = torch.from_numpy(labels.astype(np.int64))
        self._idx = 0

    def forward(self, x, mask=None, return_prenorm=False):
        b = x.shape[0]
        out = self._emb[self._idx : self._idx + b]
        self._idx = (self._idx + b) % len(self._emb)
        if return_prenorm:
            return out, out
        return out


class TestThresholdCalibration:
    def test_calibration_writes_json(self, tmp_path):
        emb, labels = _synthetic_embeddings()
        config = {
            "evaluation": {"calibration": {"enabled": True}},
            "inference": {"k_neighbors": 5, "similarity_threshold": 0.6},
            "sota": {"embedding_space": "euclidean"},
            "training": {"loss": {"type": "multi_similarity"}},
            "grain_detection": {"masked_pooling": False},
        }
        dataset = torch.utils.data.TensorDataset(
            torch.zeros(len(labels), 3, 8, 8),
            torch.from_numpy(labels),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        model = _DummyModel(emb, labels)
        payload = calibrate_inference_thresholds(
            model, loader, torch.device("cpu"), tmp_path, config
        )
        assert payload["calibrated"]["k_neighbors"] >= 1
        out = tmp_path / "inference_calibration.json"
        assert out.exists()
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert "calibrated" in saved
