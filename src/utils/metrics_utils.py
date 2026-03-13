"""
Utilidades para cálculo de métricas.
Centraliza operaciones comunes de evaluación.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    normalized_mutual_info_score,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Calcula accuracy simple.
    
    Args:
        predictions: Predicciones (N,)
        ground_truth: Ground truth (N,)
        
    Returns:
        Accuracy (0-1)
    """
    return accuracy_score(ground_truth, predictions)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    class_names: List[str] = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Calcula matriz de confusión.
    
    Args:
        y_true: Ground truth (N,)
        y_pred: Predicciones (N,)
        num_classes: Número de clases (opcional)
        class_names: Nombres de clases (opcional)
        normalize: Si True, normaliza por filas
        
    Returns:
        Matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


def compute_class_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: List[str],
    confidences: np.ndarray = None
) -> pd.DataFrame:
    """
    Calcula métricas por clase.
    
    Args:
        predictions: Predicciones (N,)
        ground_truth: Ground truth (N,)
        class_names: Nombres de clases
        confidences: Confianzas (N,) opcional
        
    Returns:
        DataFrame con métricas por clase
    """
    results = []
    
    for i, class_name in enumerate(class_names):
        mask = ground_truth == i
        
        if mask.sum() == 0:
            continue
        
        class_predictions = predictions[mask]
        class_ground_truth = ground_truth[mask]
        
        correct = (class_predictions == class_ground_truth).sum()
        total = len(class_ground_truth)
        accuracy = correct / total
        
        avg_confidence = None
        if confidences is not None:
            avg_confidence = confidences[mask].mean()
        
        result = {
            'class': class_name,
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }
        
        if avg_confidence is not None:
            result['avg_confidence'] = avg_confidence
        
        results.append(result)
    
    return pd.DataFrame(results)


def compute_retrieval_metrics(
    query_labels: np.ndarray,
    retrieved_labels: np.ndarray,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calcula métricas de retrieval (Precision@K).
    
    Args:
        query_labels: Labels de queries (N,)
        retrieved_labels: Labels de retrieved (N, K)
        k_values: Valores de K para calcular
        
    Returns:
        Dict con Precision@K para cada K
    """
    metrics = {}
    
    for k in k_values:
        top_k = retrieved_labels[:, :k]
        
        correct = 0
        for i, query_label in enumerate(query_labels):
            if query_label in top_k[i]:
                correct += 1
        
        precision_at_k = correct / len(query_labels)
        metrics[f'Precision@{k}'] = precision_at_k
    
    return metrics


def generate_classification_report(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: List[str],
    confidences: np.ndarray = None
) -> str:
    """
    Genera reporte de clasificación en texto.
    
    Args:
        predictions: Predicciones (N,)
        ground_truth: Ground truth (N,)
        class_names: Nombres de clases
        confidences: Confianzas (N,) opcional
        
    Returns:
        String con reporte formateado
    """
    overall_acc = compute_accuracy(predictions, ground_truth)
    
    class_metrics = compute_class_metrics(predictions, ground_truth, class_names, confidences)
    
    report = []
    report.append("=" * 70)
    report.append("CLASSIFICATION REPORT")
    report.append("=" * 70)
    report.append("")
    report.append(f"Overall Accuracy: {overall_acc:.2%}")
    report.append(f"Total Samples: {len(predictions)}")
    report.append("")
    report.append("Per-Class Metrics:")
    report.append("-" * 70)
    
    for _, row in class_metrics.iterrows():
        report.append(f"\n{row['class']}:")
        report.append(f"  Total: {row['total']}")
        report.append(f"  Correct: {row['correct']}")
        report.append(f"  Accuracy: {row['accuracy']:.2%}")
        if 'avg_confidence' in row:
            report.append(f"  Avg Confidence: {row['avg_confidence']:.2%}")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


def compute_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calcula Recall@K para un conjunto de embeddings.
    
    Para cada query, verifica si al menos uno de los K vecinos mas cercanos
    comparte la misma clase. Metrica estandar en Deep Metric Learning
    (Wang et al. CVPR 2019, Kim et al. CVPR 2020).
    
    Args:
        embeddings: Embeddings (N, D), asumidos L2-normalizados
        labels: Labels (N,)
        k_list: Lista de valores K para calcular R@K
        
    Returns:
        Dict con R@K para cada K. Ej: {'R@1': 0.85, 'R@5': 0.95}
    """
    n = len(embeddings)
    max_k = max(k_list)
    
    similarity = np.dot(embeddings, embeddings.T)
    
    np.fill_diagonal(similarity, -1e9)
    
    indices = np.argsort(-similarity, axis=1)[:, :max_k]
    
    metrics = {}
    for k in k_list:
        correct = 0
        for i in range(n):
            top_k_labels = labels[indices[i, :k]]
            if labels[i] in top_k_labels:
                correct += 1
        metrics[f'R@{k}'] = correct / n
    
    return metrics


def compute_nmi(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Calcula Normalized Mutual Information (NMI) entre clusters
    de embeddings y labels reales.
    
    Usa K-Means con K = numero de clases unicas.
    Metrica estandar en Deep Metric Learning (Wang et al. CVPR 2019).
    
    Args:
        embeddings: Embeddings (N, D)
        labels: Labels (N,)
        
    Returns:
        NMI score (0-1)
    """
    n_clusters = len(np.unique(labels))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_assignments = kmeans.fit_predict(embeddings)
    
    nmi = normalized_mutual_info_score(labels, cluster_assignments)
    
    logger.info(f"NMI: {nmi:.4f} (clusters={n_clusters})")
    
    return float(nmi)


def compute_embedding_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: List[int] = [1, 5]
) -> Dict[str, float]:
    """
    Calcula metricas de calidad de embeddings completas.
    Combina Recall@K y NMI.
    
    Args:
        embeddings: Embeddings (N, D)
        labels: Labels (N,)
        k_list: Valores de K para Recall@K
        
    Returns:
        Dict con todas las metricas
    """
    metrics = compute_recall_at_k(embeddings, labels, k_list)
    metrics['NMI'] = compute_nmi(embeddings, labels)
    
    logger.info(f"Embedding quality: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    
    return metrics


def compute_map_at_r(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Calcula Mean Average Precision at R (mAP@R).
    
    Metrica estandar en Deep Metric Learning (Song et al. CVPR 2016,
    Musgrave et al. ECCV 2020). R es el numero de ejemplos positivos
    por query (mismo label).
    
    Para cada query, calcula Average Precision hasta que se hayan
    recuperado todos los R ejemplos positivos, luego promedia
    sobre todas las queries.
    
    Args:
        embeddings: Embeddings L2-normalizados (N, D)
        labels: Labels (N,)
        
    Returns:
        mAP@R score (0-1)
    """
    n = len(embeddings)
    
    # Matriz de similitud coseno
    similarity = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarity, -1e9)  # Ignorar self-similarity
    
    # Ordenar por similitud descendente
    indices = np.argsort(-similarity, axis=1)
    
    average_precisions = []
    
    for i in range(n):
        query_label = labels[i]
        
        # Encontrar todos los positivos (mismo label, excluyendo query)
        positives_mask = (labels == query_label)
        positives_mask[i] = False
        num_positives = positives_mask.sum()
        
        if num_positives == 0:
            continue
        
        # Recuperar indices ordenados por similitud
        retrieved_labels = labels[indices[i]]
        
        # Calcular precision acumulativa hasta recuperar todos los R positivos
        precisions = []
        num_correct = 0
        
        for rank, retrieved_label in enumerate(retrieved_labels, start=1):
            if retrieved_label == query_label:
                num_correct += 1
                precision_at_k = num_correct / rank
                precisions.append(precision_at_k)
                
                # Detenerse cuando se han encontrado todos los positivos
                if num_correct == num_positives:
                    break
        
        # Average Precision para esta query
        if precisions:
            ap = np.mean(precisions)
            average_precisions.append(ap)
    
    # Mean Average Precision
    map_at_r = np.mean(average_precisions) if average_precisions else 0.0
    
    logger.info(f"mAP@R: {map_at_r:.4f}")
    
    return float(map_at_r)


def compute_f1_macro(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Calcula F1-Score Macro usando clasificacion 1-NN.
    
    Critico para datasets con clases desbalanceadas (Barz & Denzler, WACV 2020).
    Calcula F1 para cada clase y luego promedia sin ponderar,
    dando igual peso a clases minoritarias.
    
    Args:
        embeddings: Embeddings (N, D)
        labels: Labels (N,)
        
    Returns:
        F1-Score Macro (0-1)
    """
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Distancias euclidianas
    distances = euclidean_distances(embeddings, embeddings)
    
    # Clasificacion 1-NN (excluir self)
    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.argmin(distances, axis=1)
    predicted_labels = labels[nearest_neighbors]
    
    # F1-Score Macro (promedio no ponderado)
    f1_macro = f1_score(labels, predicted_labels, average='macro', zero_division=0)
    
    logger.info(f"F1-Score Macro: {f1_macro:.4f}")
    
    return float(f1_macro)


def compute_per_class_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calcula metricas por clase para analisis detallado.
    
    Args:
        embeddings: Embeddings (N, D)
        labels: Labels (N,)
        class_names: Nombres de clases (opcional)
        
    Returns:
        Dict con metricas por clase: {class_name: {metric: value}}
    """
    from sklearn.metrics.pairwise import euclidean_distances
    
    distances = euclidean_distances(embeddings, embeddings)
    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.argmin(distances, axis=1)
    predicted_labels = labels[nearest_neighbors]
    
    # Calcular precision, recall, f1 por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted_labels, average=None, zero_division=0
    )
    
    unique_labels = np.unique(labels)
    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_labels]
    
    results = {}
    for idx, label in enumerate(unique_labels):
        class_name = class_names[idx] if idx < len(class_names) else f"Class_{label}"
        results[class_name] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1_score': float(f1[idx]),
            'support': int(support[idx])
        }
    
    return results


def compute_all_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: List[int] = [1, 5, 10],
    class_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Calcula TODAS las metricas SOTA para Deep Metric Learning.
    
    Protocolo completo siguiendo:
    - Song et al. (CVPR 2016): Recall@K, NMI
    - Musgrave et al. (ECCV 2020): mAP@R
    - Barz & Denzler (WACV 2020): F1-Score Macro
    
    Args:
        embeddings: Embeddings L2-normalizados (N, D)
        labels: Labels (N,)
        k_list: Valores de K para Recall@K
        class_names: Nombres de clases (opcional)
        
    Returns:
        Dict con todas las metricas: {'R@1', 'R@5', 'NMI', 'mAP@R', 'F1_macro', 'per_class'}
    """
    logger.info("Computing SOTA metrics for Deep Metric Learning...")
    
    metrics = {}
    
    # Recall@K (Song et al. CVPR 2016)
    recall_metrics = compute_recall_at_k(embeddings, labels, k_list)
    metrics.update(recall_metrics)
    
    # NMI (Song et al. CVPR 2016)
    metrics['NMI'] = compute_nmi(embeddings, labels)
    
    # mAP@R (Musgrave et al. ECCV 2020)
    metrics['mAP@R'] = compute_map_at_r(embeddings, labels)
    
    # F1-Score Macro (Barz & Denzler WACV 2020)
    metrics['F1_macro'] = compute_f1_macro(embeddings, labels)
    
    # Metricas por clase
    metrics['per_class'] = compute_per_class_metrics(embeddings, labels, class_names)
    
    logger.info("="*70)
    logger.info("SOTA Metrics Summary:")
    logger.info(f"  R@1:      {metrics.get('R@1', 0):.4f}")
    logger.info(f"  R@5:      {metrics.get('R@5', 0):.4f}")
    logger.info(f"  mAP@R:    {metrics['mAP@R']:.4f}")
    logger.info(f"  NMI:      {metrics['NMI']:.4f}")
    logger.info(f"  F1-Macro: {metrics['F1_macro']:.4f}")
    logger.info("="*70)
    
    return metrics
