"""Tests for run artifact resolution (slice-aware inference prerequisites)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.run_artifact_utils import (
    check_slice_inference_artifacts,
    resolve_run_dir_from_checkpoint,
)


class TestResolveRunDir:
    def test_standard_run_layout(self, tmp_path):
        run_dir = tmp_path / "2026-06-09_12-00-00"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "best_model.pth"
        ckpt.touch()
        assert resolve_run_dir_from_checkpoint(ckpt) == run_dir.resolve()

    def test_flat_checkpoint_path(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        ckpt.touch()
        assert resolve_run_dir_from_checkpoint(ckpt) == tmp_path.resolve()


class TestCheckSliceArtifacts:
    def test_slice_representatives_present(self, tmp_path):
        run_dir = tmp_path / "run"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "best_model.pth"
        ckpt.touch()
        (run_dir / "slice_representatives.pt").touch()

        info = check_slice_inference_artifacts(ckpt)
        assert info["has_slice_representatives"] is True
        assert info["classification_mode"] == "slice-aware"
        assert info["primary_artifact"] is not None

    def test_proxy_only_legacy(self, tmp_path):
        run_dir = tmp_path / "run"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "best_model.pth"
        ckpt.touch()
        (run_dir / "class_proxies.pt").touch()

        info = check_slice_inference_artifacts(ckpt)
        assert info["has_class_proxies"] is True
        assert info["classification_mode"] == "slice-aware"

    def test_missing_slice_artifacts_knn_fallback(self, tmp_path):
        run_dir = tmp_path / "run"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt = ckpt_dir / "best_model.pth"
        ckpt.touch()
        (run_dir / "reference_embeddings.pt").touch()
        (run_dir / "class_centroids.pt").touch()

        info = check_slice_inference_artifacts(ckpt)
        assert info["has_slice_representatives"] is False
        assert info["classification_mode"] == "knn_fallback"

    def test_no_artifacts(self, tmp_path):
        ckpt = tmp_path / "best_model.pth"
        ckpt.touch()
        info = check_slice_inference_artifacts(ckpt)
        assert info["classification_mode"] == "knn_fallback"
        assert info["primary_artifact"] is None
