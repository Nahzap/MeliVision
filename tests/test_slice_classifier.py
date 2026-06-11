"""Tests for slice-aware classification."""

import tempfile
from pathlib import Path

import torch

from src.inference.slice_classifier import (
    SliceAwareClassifier,
    SliceRepresentatives,
    compute_slice_centroids_from_embeddings,
    load_slice_representatives,
    save_slice_representatives,
)


def _make_reps(num_slices=4, num_classes=3, slice_dim=8):
    reps = torch.randn(num_slices, num_classes, slice_dim)
    return SliceRepresentatives(
        representatives=reps,
        class_names=["a", "b", "c"],
        num_slices=num_slices,
        slice_dim=slice_dim,
        embedding_dim=num_slices * slice_dim,
        mode="slice_centroids",
        loss_type="sliced_ms",
    )


def test_predict_shape_and_labels():
    reps = _make_reps()
    clf = SliceAwareClassifier(reps)
    emb = torch.randn(5, 32)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    preds, scores, class_scores = clf.predict(emb)
    assert preds.shape == (5,)
    assert scores.shape == (5,)
    assert class_scores.shape == (5, 3)


def test_reconstruct_target_vector_shape():
    reps = _make_reps(num_slices=4, num_classes=3, slice_dim=8)
    clf = SliceAwareClassifier(reps)
    target = clf.reconstruct_target_vector(1)
    assert target.shape == (32,)


def test_slice_centroids_from_embeddings():
    torch.manual_seed(0)
    labels = torch.tensor([0, 0, 1, 1, 2])
    embs = torch.randn(5, 32)
    reps = compute_slice_centroids_from_embeddings(embs, labels, ["a", "b", "c"], num_slices=4)
    assert reps.representatives.shape == (4, 3, 8)
    assert reps.mode == "slice_centroids"


def test_save_load_roundtrip():
    reps = _make_reps()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "slice_representatives.pt"
        save_slice_representatives(path, reps)
        loaded = load_slice_representatives(path, torch.device("cpu"))
        assert loaded.representatives.shape == reps.representatives.shape
        assert loaded.class_names == reps.class_names
        assert loaded.num_slices == reps.num_slices
