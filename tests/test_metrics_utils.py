"""
Tests unitarios para metrics_utils — Métricas de evaluación SOTA (M2).
Cubre: compute_recall_at_k, compute_nmi, compute_embedding_quality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics_utils import (
    compute_recall_at_k,
    compute_nmi,
    compute_embedding_quality,
    compute_accuracy,
    compute_retrieval_metrics,
)


# ============================================================
# Tests M2: compute_recall_at_k
# ============================================================

def test_m2_recall_at_k_perfect_embeddings():
    """M2: Con embeddings perfectamente separados, R@1 debe ser 1.0."""
    np.random.seed(42)
    dim = 64
    n_per_class = 10

    embeddings = []
    labels = []

    for cls_id in range(3):
        center = np.zeros(dim)
        center[cls_id * 20:(cls_id + 1) * 20] = 1.0
        center = center / np.linalg.norm(center)

        for _ in range(n_per_class):
            noise = np.random.randn(dim) * 0.01
            emb = center + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            labels.append(cls_id)

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels)

    metrics = compute_recall_at_k(embeddings, labels, k_list=[1, 5])

    assert metrics['R@1'] == 1.0, f"R@1 debe ser 1.0 para embeddings perfectos, obtenido {metrics['R@1']}"
    assert metrics['R@5'] == 1.0, f"R@5 debe ser 1.0 para embeddings perfectos, obtenido {metrics['R@5']}"


def test_m2_recall_at_k_returns_dict():
    """M2: compute_recall_at_k debe retornar dict con claves R@K."""
    np.random.seed(42)
    embeddings = np.random.randn(20, 64).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0]*10 + [1]*10)

    metrics = compute_recall_at_k(embeddings, labels, k_list=[1, 5, 10])

    assert 'R@1' in metrics, "Debe contener R@1"
    assert 'R@5' in metrics, "Debe contener R@5"
    assert 'R@10' in metrics, "Debe contener R@10"


def test_m2_recall_at_k_values_range():
    """M2: Valores de R@K deben estar en [0, 1]."""
    np.random.seed(42)
    embeddings = np.random.randn(30, 64).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0]*10 + [1]*10 + [2]*10)

    metrics = compute_recall_at_k(embeddings, labels, k_list=[1, 5])

    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k} = {v} fuera de rango [0, 1]"


def test_m2_recall_at_k_monotonic():
    """M2: R@K debe ser monótono: R@1 <= R@5 <= R@10."""
    np.random.seed(42)
    embeddings = np.random.randn(30, 64).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0]*10 + [1]*10 + [2]*10)

    metrics = compute_recall_at_k(embeddings, labels, k_list=[1, 5, 10])

    assert metrics['R@1'] <= metrics['R@5'], "R@1 debe ser <= R@5"
    assert metrics['R@5'] <= metrics['R@10'], "R@5 debe ser <= R@10"


def test_m2_recall_with_sample_embeddings(sample_embeddings):
    """M2: Con embeddings de fixture (3 clases, well-separated), R@1 debe ser alto."""
    embeddings, labels = sample_embeddings
    metrics = compute_recall_at_k(embeddings, labels, k_list=[1, 5])

    assert metrics['R@1'] > 0.5, f"R@1 con embeddings separados debe ser > 0.5, obtenido {metrics['R@1']}"


# ============================================================
# Tests M2: compute_nmi
# ============================================================

def test_m2_nmi_perfect_clusters():
    """M2: Con clusters perfectos, NMI debe ser cercano a 1.0."""
    np.random.seed(42)
    dim = 64

    embeddings = []
    labels = []

    for cls_id in range(3):
        center = np.zeros(dim)
        center[cls_id * 20:(cls_id + 1) * 20] = 5.0

        for _ in range(15):
            emb = center + np.random.randn(dim) * 0.1
            embeddings.append(emb)
            labels.append(cls_id)

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels)

    nmi = compute_nmi(embeddings, labels)

    assert nmi > 0.8, f"NMI con clusters perfectos debe ser > 0.8, obtenido {nmi}"


def test_m2_nmi_range():
    """M2: NMI debe estar en [0, 1]."""
    np.random.seed(42)
    embeddings = np.random.randn(30, 64).astype(np.float32)
    labels = np.array([0]*10 + [1]*10 + [2]*10)

    nmi = compute_nmi(embeddings, labels)

    assert 0.0 <= nmi <= 1.0, f"NMI = {nmi} fuera de rango [0, 1]"


def test_m2_nmi_returns_float():
    """M2: compute_nmi debe retornar float."""
    np.random.seed(42)
    embeddings = np.random.randn(20, 32).astype(np.float32)
    labels = np.array([0]*10 + [1]*10)

    nmi = compute_nmi(embeddings, labels)
    assert isinstance(nmi, float), f"NMI debe ser float, obtenido {type(nmi)}"


# ============================================================
# Tests M2: compute_embedding_quality
# ============================================================

def test_m2_embedding_quality_combined():
    """M2: compute_embedding_quality debe retornar R@K + NMI."""
    np.random.seed(42)
    embeddings = np.random.randn(30, 64).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0]*10 + [1]*10 + [2]*10)

    metrics = compute_embedding_quality(embeddings, labels, k_list=[1, 5])

    assert 'R@1' in metrics, "Debe contener R@1"
    assert 'R@5' in metrics, "Debe contener R@5"
    assert 'NMI' in metrics, "Debe contener NMI"

    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k} = {v} fuera de rango"


# ============================================================
# Tests existentes (regresión)
# ============================================================

def test_compute_accuracy():
    """Regresión: compute_accuracy debe funcionar correctamente."""
    predictions = np.array([0, 1, 2, 0, 1])
    ground_truth = np.array([0, 1, 2, 0, 0])

    acc = compute_accuracy(predictions, ground_truth)
    assert acc == 0.8, f"Accuracy esperado 0.8, obtenido {acc}"


def test_compute_retrieval_metrics():
    """Regresión: compute_retrieval_metrics debe funcionar."""
    query_labels = np.array([0, 1, 2])
    retrieved_labels = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [2, 2, 0],
    ])

    metrics = compute_retrieval_metrics(query_labels, retrieved_labels, k_values=[1, 3])

    assert 'Precision@1' in metrics
    assert 'Precision@3' in metrics


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
