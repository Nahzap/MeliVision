"""
Post-Training Analysis Module
Generates all result visualizations and reports after training completes.

Functions:
    - plot_training_curves: Loss and Recall curves from CSV
    - plot_tsne_3d: 3D t-SNE embedding visualization
    - plot_confusion_matrix: kNN-based confusion matrix
    - plot_distance_distribution: Intra vs inter-class distances
    - generate_summary_report: Markdown summary of training results
    - run_post_training_analysis: Orchestrator that calls all above

Author: MetricLearning Project
Date: 2026-02-06
"""

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from datetime import datetime
import logging
import json
from PIL import Image

# Análisis espectral (SOTA 2026)
from src.visualization.embedding_analysis import EmbeddingSpaceAnalyzer
from src.utils.unicode_utils import sanitize_for_ascii

logger = logging.getLogger(__name__)


def extract_embeddings(model, dataloader, device):
    """Extract all embeddings and labels from a dataloader."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting embeddings"):
            images, labels = batch_data[0], batch_data[1]
            images = images.to(device, non_blocking=True)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(
        f"[extract_embeddings] {embeddings.shape[0]} samples, dim={embeddings.shape[1]}, "
        f"norm: mean={norms.mean():.4f} std={norms.std():.4f}, "
        f"classes={len(np.unique(labels))}"
    )
    return embeddings, labels


def plot_training_curves(csv_path: str, output_dir: str):
    """
    Generate training curves from the metrics CSV.
    Produces: loss_curves.png, recall_curves.png, lr_curve.png
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)

    # --- Loss Curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['epoch'], df['train_loss'], 'b-o', markersize=4, label='Train Loss', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], 'r-o', markersize=4, label='Val Loss', linewidth=2)

    # Mark best epoch
    best_rows = df[df['is_best'] == True]
    if not best_rows.empty:
        last_best = best_rows.iloc[-1]
        ax.axvline(x=last_best['epoch'], color='green', linestyle='--', alpha=0.5, label=f"Best (epoch {int(last_best['epoch'])})")

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss curves saved: {output_dir / 'loss_curves.png'}")

    # --- Recall Curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['epoch'], df['recall_at_1'], 'g-o', markersize=4, label='Recall@1', linewidth=2)
    ax.plot(df['epoch'], df['recall_at_5'], 'm-o', markersize=4, label='Recall@5', linewidth=2)
    ax.plot(df['epoch'], df['best_recall_at_1'], 'k--', alpha=0.5, label='Best R@1', linewidth=1)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title('Recall@K over Training', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'recall_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Recall curves saved: {output_dir / 'recall_curves.png'}")

    # --- Learning Rate Curve ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['epoch'], df['learning_rate'], 'c-o', markersize=3, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'lr_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"LR curve saved: {output_dir / 'lr_curve.png'}")

    # --- Epoch Time ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df['epoch'], df['epoch_time_sec'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Epoch Duration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'epoch_times.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Epoch times saved: {output_dir / 'epoch_times.png'}")

    return df


def plot_tsne_3d(model, dataloader, device, output_path, class_names, perplexity=30):
    """Generate 3D t-SNE embedding visualization with embedding space metadata."""
    logger.info("Generating 3D t-SNE visualization...")

    embeddings, labels = extract_embeddings(model, dataloader, device)
    embed_dim = embeddings.shape[1]
    n_samples = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(f"Embeddings shape: {embeddings.shape}, norms: mean={norms.mean():.4f} std={norms.std():.4f}")

    # Run t-SNE with 3 components
    tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=1000, random_state=42, n_jobs=-1)
    emb_3d = tsne.fit_transform(embeddings)

    # Color palette
    n_classes = len(class_names)
    cmap = cm.get_cmap('tab10' if n_classes <= 10 else 'tab20')
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    # --- 3D Interactive-style plot (multiple angles) ---
    fig = plt.figure(figsize=(20, 16))

    angles = [(30, 45), (30, 135), (60, 0), (10, 90)]
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        for class_id, class_name in enumerate(class_names):
            mask = labels == class_id
            ax.scatter(
                emb_3d[mask, 0], emb_3d[mask, 1], emb_3d[mask, 2],
                c=[colors[class_id]], label=class_name, alpha=0.6, s=15
            )
        ax.set_title(f'View: elev={elev}, azim={azim}', fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, markerscale=2)

    fig.suptitle(
        f'Espacio de Embeddings $\mathbb{{S}}^{{{embed_dim}}}$ — t-SNE 3D\n'
        f'{n_samples} muestras, {len(class_names)} clases, '
        f'‖emb‖ = {norms.mean():.3f} ± {norms.std():.4f}',
        fontsize=16, fontweight='bold', y=1.0
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"3D t-SNE saved: {output_path}")

    # Also save 2D t-SNE
    tsne_2d = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42, n_jobs=-1)
    emb_2d = tsne_2d.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[colors[class_id]],
                        label=class_name, alpha=0.6, s=20)
    axes[0].set_title(
        f'S^{embed_dim} -> t-SNE 2D (por Clase)',
        fontsize=14, fontweight='bold'
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Density plot
    from scipy.stats import gaussian_kde
    xy = np.vstack([emb_2d[:, 0], emb_2d[:, 1]])
    z = gaussian_kde(xy)(xy)
    scatter = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=z, s=20, cmap='viridis', alpha=0.6)
    axes[1].set_title(
        f'S^{embed_dim} -> t-SNE 2D Densidad ({n_samples} muestras)',
        fontsize=14, fontweight='bold'
    )
    plt.colorbar(scatter, ax=axes[1], label='Density')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    tsne_2d_path = output_path.parent / 'tsne_2d.png'
    plt.savefig(tsne_2d_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"2D t-SNE saved: {tsne_2d_path}")

    return embeddings, labels


def plot_embedding_space_analysis(embeddings, labels, class_names, output_path):
    """
    Generate a comprehensive embedding space analysis visualization.
    Shows: norm distribution, angular separation, centroid structure, cosine similarity.
    This plot demonstrates the properties of the learned S^D hypersphere.
    """
    embed_dim = embeddings.shape[1]
    n_samples = embeddings.shape[0]
    n_classes = len(class_names)
    norms = np.linalg.norm(embeddings, axis=1)

    # Normalize for angular/cosine analysis
    emb_normed = embeddings / (norms[:, None] + 1e-8)

    # Per-class centroids (on normalized embeddings)
    centroids = np.zeros((n_classes, embed_dim))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = emb_normed[mask].mean(axis=0)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normed = centroids / (centroid_norms + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f'Análisis del Espacio de Embeddings $\\mathbb{{S}}^{{{embed_dim}}}$\n'
        f'{n_samples} muestras, {n_classes} clases',
        fontsize=18, fontweight='bold', y=0.98
    )

    # --- Panel 1: Norm distribution (or cosine distance distribution if L2-normalized) ---
    ax = axes[0, 0]
    norm_range = norms.max() - norms.min()
    if norm_range < 1e-6:
        # All embeddings are L2-normalized (norm ≈ 1.0) — show pairwise cosine
        # distance distribution instead (more informative)
        rng = np.random.RandomState(42)
        idx = rng.choice(n_samples, size=min(n_samples, 500), replace=False)
        sample_embs = emb_normed[idx]
        cos_dists = 1.0 - (sample_embs @ sample_embs.T)
        triu_idx = np.triu_indices(len(idx), k=1)
        cos_dist_vals = cos_dists[triu_idx]
        n_bins = max(5, min(50, len(cos_dist_vals) // 20))
        ax.hist(cos_dist_vals, bins=n_bins, color='steelblue', edgecolor='black', alpha=0.8, density=True)
        ax.axvline(cos_dist_vals.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Media = {cos_dist_vals.mean():.4f}')
        ax.set_xlabel('Distancia Coseno (1 - cos θ)', fontsize=12)
        ax.set_ylabel('Densidad', fontsize=12)
        ax.set_title(
            f'Distribución de Distancias Coseno\n'
            f'‖emb‖ = {norms.mean():.4f} (L2-normalizado, σ ≈ 0)',
            fontsize=13, fontweight='bold'
        )
    else:
        n_bins = max(5, min(50, int(n_samples / 10)))
        ax.hist(norms, bins=n_bins, color='steelblue', edgecolor='black', alpha=0.8, density=True)
        ax.axvline(norms.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Media = {norms.mean():.4f}')
        ax.axvline(1.0, color='green', linestyle=':', linewidth=2, label='Norma ideal = 1.0')
        ax.set_xlabel('‖embedding‖₂', fontsize=12)
        ax.set_ylabel('Densidad', fontsize=12)
        ax.set_title(
            f'Distribución de Normas (σ = {norms.std():.4f})',
            fontsize=13, fontweight='bold'
        )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Cosine similarity heatmap (centroids) ---
    ax = axes[0, 1]
    cos_sim = centroids_normed @ centroids_normed.T
    im = ax.imshow(cos_sim, cmap='RdYlBu_r', vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if abs(cos_sim[i, j]) > 0.6 else 'black'
            ax.text(j, i, f'{cos_sim[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color=color)
    ax.set_title('Similitud Coseno entre Centroides', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)

    # --- Panel 3: Per-class compactness (intra-class cosine sim) ---
    ax = axes[1, 0]
    compactness = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 1:
            cls_embs = emb_normed[mask]
            cls_cos = cls_embs @ centroids_normed[c]
            compactness.append(cls_cos.mean())
        else:
            compactness.append(0)

    colors_bar = ['#2ecc71' if v >= 0.8 else ('#f39c12' if v >= 0.6 else '#e74c3c')
                  for v in compactness]
    bars = ax.barh(class_names, compactness, color=colors_bar, edgecolor='#333', linewidth=0.5)
    for bar, v in zip(bars, compactness):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Similitud Coseno media al Centroide', fontsize=11)
    ax.set_title('Compacidad Intra-clase', fontsize=13, fontweight='bold')
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
    ax.grid(True, axis='x', alpha=0.3)

    # --- Panel 4: Angular separation between class centroids ---
    ax = axes[1, 1]
    # Compute angular distances (arccos of cosine similarity)
    cos_sim_clipped = np.clip(cos_sim, -1.0, 1.0)
    angular_dist = np.degrees(np.arccos(cos_sim_clipped))
    np.fill_diagonal(angular_dist, 0)

    # Off-diagonal angular distances
    mask_off = ~np.eye(n_classes, dtype=bool)
    off_diag = angular_dist[mask_off]

    n_bins_ang = max(5, min(30, len(off_diag) // 3))
    ax.hist(off_diag, bins=n_bins_ang, color='darkorange', edgecolor='black', alpha=0.8)
    ax.axvline(off_diag.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Media = {off_diag.mean():.1f}deg')
    ax.axvline(90, color='green', linestyle=':', linewidth=2, label='Separacion ideal = 90deg')
    ax.set_xlabel('Separación Angular (grados)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title(
        f'Separacion Angular entre Centroides\n'
        f'Rango: [{off_diag.min():.1f}deg, {off_diag.max():.1f}deg]',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Embedding space analysis saved: {output_path}")

    return {
        'embed_dim': embed_dim,
        'n_samples': n_samples,
        'norm_mean': float(norms.mean()),
        'norm_std': float(norms.std()),
        'mean_angular_sep': float(off_diag.mean()),
        'min_angular_sep': float(off_diag.min()),
        'max_angular_sep': float(off_diag.max()),
        'compactness': {class_names[i]: float(compactness[i]) for i in range(n_classes)},
    }


def plot_confusion_matrix_knn(embeddings, labels, class_names, output_path, k=1):
    """Generate confusion matrix using leave-one-out k-NN classification."""
    logger.info(f"Generating confusion matrix (k-NN, k={k})...")

    # Leave-one-out: for each sample, find its k nearest neighbors (excluding itself)
    n = len(labels)
    dists = euclidean_distances(embeddings)
    np.fill_diagonal(dists, np.inf)

    # k-NN predictions
    predictions = []
    for i in range(n):
        nearest_idx = np.argsort(dists[i])[:k]
        nearest_labels = labels[nearest_idx]
        # Majority vote
        pred = np.bincount(nearest_labels, minlength=len(class_names)).argmax()
        predictions.append(pred)
    predictions = np.array(predictions)

    # Compute confusion matrix
    cm = sk_confusion_matrix(labels, predictions, labels=list(range(len(class_names))))

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # Accuracy
    accuracy = np.trace(cm) / cm.sum()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix (Counts) — Acc: {accuracy:.1%}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved: {output_path}")

    # Per-class accuracy
    per_class_acc = cm_norm.diagonal()
    for i, name in enumerate(class_names):
        logger.info(f"  {name}: {per_class_acc[i]:.1%}")

    return accuracy, per_class_acc


def plot_distance_distribution(embeddings, labels, class_names, output_path, max_pairs=50000):
    """Plot intra-class vs inter-class distance distributions."""
    logger.info("Generating distance distribution...")

    n = len(labels)
    intra_dists = []
    inter_dists = []

    # Sample pairs to avoid O(n^2) explosion
    rng = np.random.RandomState(42)
    indices = rng.choice(n, size=min(n, 2000), replace=False)

    for i in indices:
        for j in indices:
            if i >= j:
                continue
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if labels[i] == labels[j]:
                intra_dists.append(dist)
            else:
                inter_dists.append(dist)
            if len(intra_dists) + len(inter_dists) >= max_pairs:
                break
        if len(intra_dists) + len(inter_dists) >= max_pairs:
            break

    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Histogram
    axes[0].hist(intra_dists, bins=60, alpha=0.6, label='Intra-class', color='green', density=True)
    axes[0].hist(inter_dists, bins=60, alpha=0.6, label='Inter-class', color='red', density=True)
    axes[0].set_xlabel('Euclidean Distance', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Distance Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Box plot per class
    class_intra_dists = {name: [] for name in class_names}
    for i in indices:
        for j in indices:
            if i >= j or labels[i] != labels[j]:
                continue
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            class_intra_dists[class_names[labels[i]]].append(dist)

    box_data = [class_intra_dists[name] for name in class_names if class_intra_dists[name]]
    box_labels = [name for name in class_names if class_intra_dists[name]]
    axes[1].boxplot(box_data, labels=box_labels)
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Intra-class Distance', fontsize=12)
    axes[1].set_title('Intra-class Distance by Class', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    mean_intra = np.mean(intra_dists) if len(intra_dists) > 0 else 0
    mean_inter = np.mean(inter_dists) if len(inter_dists) > 0 else 0
    ratio = mean_inter / mean_intra if mean_intra > 0 else 0

    logger.info(f"Distance stats — Intra: {mean_intra:.4f}, Inter: {mean_inter:.4f}, Ratio: {ratio:.2f}")
    logger.info(f"Distance distribution saved: {output_path}")

    return mean_intra, mean_inter, ratio


def plot_inference_preview(model, centroids, val_loader, device, output_path, class_names,
                          n_samples=10):
    """
    Generate an inference preview grid: random val images with predictions,
    true labels, distances, and confidence color coding.

    Args:
        model: Trained model on device
        centroids: Pre-computed class centroids tensor (n_classes, embed_dim) on device
        val_loader: Validation DataLoader
        device: torch.device
        output_path: Path to save the image
        class_names: List of class names
        n_samples: Number of samples to show
    """
    logger.info(f"Generating inference preview ({n_samples} random samples)...")

    model.eval()

    # 1. Collect val images (we need raw tensors for display)
    all_images = []
    all_labels = []
    for batch_data in val_loader:
        images, labels = batch_data[0], batch_data[1]
        all_images.append(images)
        all_labels.append(labels)
        if sum(len(x) for x in all_images) >= max(n_samples * 5, 200):
            break
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 3. Sample n_samples random indices (balanced across classes)
    rng = np.random.RandomState(42)
    unique_labels = torch.unique(all_labels).numpy()
    per_class = max(1, -(-n_samples // len(unique_labels)))  # ceil division
    selected_idx = []
    for lbl in unique_labels:
        mask = (all_labels.numpy() == lbl).nonzero()[0]
        chosen = rng.choice(mask, size=min(per_class, len(mask)), replace=False)
        selected_idx.extend(chosen.tolist())
    rng.shuffle(selected_idx)
    selected_idx = selected_idx[:n_samples]
    n_samples = len(selected_idx)  # adjust if fewer samples available

    sel_images = all_images[selected_idx]
    sel_labels = all_labels[selected_idx]

    # 4. Run inference
    with torch.no_grad():
        sel_embs = model(sel_images.to(device))
        dists = torch.cdist(sel_embs, centroids)
        pred_classes = torch.argmin(dists, dim=1)
        min_dists = torch.min(dists, dim=1)[0]

    # 5. Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    sel_denorm = (sel_images * std + mean).clamp(0, 1)

    # 6. Create the visualization grid
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else [axes]
    if n_rows == 1:
        axes = np.array([axes])

    for idx in range(n_samples):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col] if n_rows > 1 else axes[0][col]

        img_np = sel_denorm[idx].permute(1, 2, 0).numpy()
        true_lbl = sel_labels[idx].item()
        pred_lbl = pred_classes[idx].item()
        dist_val = min_dists[idx].item()
        correct = true_lbl == pred_lbl

        # Add colored border
        border = 6
        color = [0.2, 0.8, 0.2] if correct else [0.9, 0.2, 0.2]
        bordered = np.ones((img_np.shape[0] + 2 * border, img_np.shape[1] + 2 * border, 3))
        bordered[:, :] = color
        bordered[border:-border, border:-border] = img_np

        ax.imshow(bordered)
        ax.set_xticks([])
        ax.set_yticks([])

        status = "OK" if correct else "WRONG"
        title = (f"Pred: {class_names[pred_lbl]}\n"
                 f"True: {class_names[true_lbl]}\n"
                 f"Dist: {dist_val:.3f} [{status}]")
        title_color = 'green' if correct else 'red'
        ax.set_title(title, fontsize=10, fontweight='bold', color=title_color)

    # Hide unused axes
    for idx in range(n_samples, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col] if n_rows > 1 else axes[0][col]
        ax.axis('off')

    n_correct = sum(1 for i in range(n_samples) if sel_labels[i].item() == pred_classes[i].item())
    fig.suptitle(
        f'Inference Preview — {n_correct}/{n_samples} correct '
        f'(Best Model, Centroid-based Classification)',
        fontsize=16, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Inference preview saved: {output_path} ({n_correct}/{n_samples} correct)")
    return n_correct, n_samples


def compute_and_save_centroids(model, train_loader, device, save_dir, class_names):
    """
    Compute class centroids and reference embeddings from the training set.
    Saves two artifacts:
      - class_centroids.pt  (lightweight — just centroids + metadata)
      - reference_embeddings.pt (full embeddings + labels for kNN)

    This should be called ONCE at the end of training, not during evaluation.

    Returns:
        centroids: tensor (n_classes, embed_dim) on device
    """
    logger.info("Computing class centroids from training set (one-time)...")
    model.eval()

    all_embs = []
    all_labels = []

    with torch.no_grad():
        for batch_data in tqdm(train_loader, desc="Extracting train embeddings", leave=False):
            images, labels = batch_data[0], batch_data[1]
            images = images.to(device, non_blocking=True)
            embs = model(images)
            all_embs.append(embs.cpu())
            all_labels.append(labels)

    all_embs = torch.cat(all_embs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute centroids (mean embedding per class)
    n_classes = len(class_names)
    embed_dim = all_embs.shape[1]
    centroids = torch.zeros(n_classes, embed_dim)
    n_samples_per_class = torch.zeros(n_classes, dtype=torch.long)

    for c in range(n_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            centroids[c] = all_embs[mask].mean(dim=0)
            n_samples_per_class[c] = mask.sum().item()

    # Save artifacts
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Log per-class stats and inter-centroid distances
    for c in range(n_classes):
        cnorm = torch.norm(centroids[c]).item()
        logger.info(f"  Centroid [{class_names[c]}]: n={n_samples_per_class[c]}, norm={cnorm:.4f}")
    inter_dists = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    mask_triu = torch.triu(torch.ones(n_classes, n_classes, dtype=torch.bool), diagonal=1)
    mean_inter = inter_dists[mask_triu].mean().item()
    min_inter = inter_dists[mask_triu].min().item()
    logger.info(f"  Inter-centroid dist: mean={mean_inter:.4f}, min={min_inter:.4f}")

    centroid_data = {
        'centroids': centroids,
        'class_names': list(class_names),
        'n_samples_per_class': n_samples_per_class,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    torch.save(centroid_data, save_dir / 'class_centroids.pt')
    logger.info(f"Class centroids saved: {save_dir / 'class_centroids.pt'} "
                f"({n_classes} classes, {embed_dim}D)")

    ref_data = {
        'embeddings': all_embs,
        'labels': all_labels,
        'class_names': list(class_names),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    torch.save(ref_data, save_dir / 'reference_embeddings.pt')
    logger.info(f"Reference embeddings saved: {save_dir / 'reference_embeddings.pt'} "
                f"({all_embs.shape[0]} samples)")

    return centroids.to(device)


def load_centroids(centroids_path, device):
    """
    Load pre-computed centroids from file.

    Returns:
        centroids: tensor (n_classes, embed_dim) on device
        class_names: list of class name strings
    """
    data = torch.load(centroids_path, map_location=device, weights_only=True)
    centroids = data['centroids'].to(device)
    class_names = data['class_names']
    logger.info(f"Centroids loaded: {centroids_path} "
                f"({centroids.shape[0]} classes, {centroids.shape[1]}D)")
    return centroids, class_names


def generate_inference_gallery(model, centroids, val_loader, device, output_dir, class_names,
                               n_per_class=10):
    """
    Generate a per-class inference gallery with n_per_class random images per class.
    Each class gets its own image grid saved in output_dir/inference_gallery/.
    Also generates a combined summary page.

    Args:
        model: Trained model on device
        centroids: Pre-computed class centroids tensor (n_classes, embed_dim) on device
        val_loader: Validation DataLoader
        device: torch.device
        output_dir: Directory to save gallery images
        class_names: List of class names
        n_per_class: Number of samples per class to show
    """
    gallery_dir = Path(output_dir) / 'inference_gallery'
    gallery_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # 1. Collect ALL val images and run inference
    logger.info("  Running inference on full validation set...")
    all_images = []
    all_labels = []
    all_preds = []
    all_dists = []
    all_all_dists = []

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Gallery inference", leave=False):
            images, labels = batch_data[0], batch_data[1]
            embs = model(images.to(device))
            dists = torch.cdist(embs, centroids)
            preds = torch.argmin(dists, dim=1)
            min_d = torch.min(dists, dim=1)[0]

            denorm = (images * std + mean).clamp(0, 1)
            all_images.append(denorm.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_dists.append(min_d.cpu())
            all_all_dists.append(dists.cpu())

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_dists = torch.cat(all_dists)
    all_all_dists = torch.cat(all_all_dists)

    rng = np.random.RandomState(42)
    class_stats = {}

    # 3. Generate per-class gallery
    for cls_idx, cls_name in enumerate(class_names):
        mask = (all_labels == cls_idx).numpy()
        cls_indices = np.where(mask)[0]

        if len(cls_indices) == 0:
            continue

        cls_preds = all_preds[cls_indices]
        cls_correct = (cls_preds == cls_idx).numpy()
        n_correct = int(cls_correct.sum())
        n_total = len(cls_indices)
        cls_acc = n_correct / n_total if n_total > 0 else 0
        class_stats[cls_name] = {'correct': n_correct, 'total': n_total, 'accuracy': cls_acc}

        # Select samples: mix of correct and incorrect
        correct_idx = cls_indices[cls_correct]
        wrong_idx = cls_indices[~cls_correct]

        # Try to show both correct and incorrect — prioritize showing errors
        n_wrong_show = min(len(wrong_idx), max(3, n_per_class // 3))
        n_correct_show = min(len(correct_idx), n_per_class - n_wrong_show)
        # If not enough wrong, fill with correct
        if n_wrong_show + n_correct_show < n_per_class:
            n_correct_show = min(len(correct_idx), n_per_class - n_wrong_show)

        selected = []
        if len(wrong_idx) > 0:
            chosen_wrong = rng.choice(wrong_idx, size=min(n_wrong_show, len(wrong_idx)), replace=False)
            selected.extend(chosen_wrong.tolist())
        if len(correct_idx) > 0:
            chosen_correct = rng.choice(correct_idx, size=min(n_correct_show, len(correct_idx)), replace=False)
            selected.extend(chosen_correct.tolist())

        rng.shuffle(selected)
        n_show = len(selected)

        if n_show == 0:
            continue

        # Create grid
        n_cols = min(5, n_show)
        n_rows = (n_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5.5 * n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, idx in enumerate(selected):
            row, col = divmod(i, n_cols)
            ax = axes[row][col]

            img_np = all_images[idx].permute(1, 2, 0).numpy()
            pred_lbl = all_preds[idx].item()
            dist_val = all_dists[idx].item()
            correct = pred_lbl == cls_idx

            # Colored border
            border = 5
            color = [0.15, 0.75, 0.15] if correct else [0.9, 0.15, 0.15]
            bordered = np.ones((img_np.shape[0] + 2 * border, img_np.shape[1] + 2 * border, 3))
            bordered[:, :] = color
            bordered[border:-border, border:-border] = img_np

            ax.imshow(bordered)
            ax.set_xticks([])
            ax.set_yticks([])

            if correct:
                title = f"Pred: {class_names[pred_lbl]}\nDist: {dist_val:.3f} [OK]"
                ax.set_title(title, fontsize=9, fontweight='bold', color='green')
            else:
                # Show top-3 distances for wrong predictions
                sample_dists = all_all_dists[idx]
                top3 = torch.topk(sample_dists, 3, largest=False)
                top3_str = ", ".join([f"{class_names[t.item()]}: {d.item():.3f}"
                                     for t, d in zip(top3.indices, top3.values)])
                title = (f"Pred: {class_names[pred_lbl]}\n"
                         f"Dist: {dist_val:.3f} [WRONG]\n"
                         f"Top3: {top3_str}")
                ax.set_title(title, fontsize=8, fontweight='bold', color='red')

        # Hide unused axes
        for i in range(n_show, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row][col].axis('off')

        status_emoji = "OK" if cls_acc >= 0.9 else ("~" if cls_acc >= 0.7 else "!!")
        fig.suptitle(
            f'{cls_name} — {n_correct}/{n_total} correct ({cls_acc:.1%}) [{status_emoji}]',
            fontsize=15, fontweight='bold',
            color='green' if cls_acc >= 0.9 else ('orange' if cls_acc >= 0.7 else 'red'),
            y=1.01
        )

        plt.tight_layout()
        save_path = gallery_dir / f'{cls_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  {cls_name}: {n_correct}/{n_total} ({cls_acc:.1%}) -> {save_path.name}")

    # 4. Generate summary overview
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(class_stats.keys())
    accs = [class_stats[n]['accuracy'] for n in names]
    totals = [class_stats[n]['total'] for n in names]
    corrects = [class_stats[n]['correct'] for n in names]

    colors_bar = ['#2ecc71' if a >= 0.9 else ('#f39c12' if a >= 0.7 else '#e74c3c') for a in accs]
    bars = ax.barh(names, accs, color=colors_bar, edgecolor='#333', linewidth=0.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy (Centroid Classification)', fontsize=14, fontweight='bold')
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    ax.legend(loc='lower right', fontsize=9)

    for bar, acc, c, t in zip(bars, accs, corrects, totals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1%} ({c}/{t})', va='center', fontsize=10, fontweight='bold')

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(gallery_dir / 'summary.png', dpi=150, bbox_inches='tight')
    plt.close()

    total_correct = sum(class_stats[n]['correct'] for n in names)
    total_samples = sum(class_stats[n]['total'] for n in names)
    overall_acc = total_correct / total_samples if total_samples > 0 else 0

    logger.info(f"Inference gallery saved: {gallery_dir} ({total_correct}/{total_samples} = {overall_acc:.1%})")
    return class_stats


def generate_summary_report(csv_path, config, output_path, accuracy=0, per_class_acc=None,
                            dist_ratio=0, class_names=None, emb_stats=None):
    """Generate a markdown summary report of training results."""
    df = pd.read_csv(csv_path)
    output_path = Path(output_path)

    best_row = df.loc[df['recall_at_1'].astype(float).idxmax()]
    final_row = df.iloc[-1]
    total_time = df['epoch_time_sec'].sum()

    def fmt_time(s):
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {sec}s"
        return f"{m}m {sec}s"

    lines = [
        f"# Training Results Report",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total training time:** {fmt_time(total_time)}  ",
        f"**Epochs:** {len(df)}  ",
        "",
        "---",
        "",
        "## Best Model Performance",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best Epoch | {int(best_row['epoch'])} |",
        f"| Recall@1 | {float(best_row['recall_at_1']):.4f} ({float(best_row['recall_at_1']):.1%}) |",
        f"| Recall@5 | {float(best_row['recall_at_5']):.4f} ({float(best_row['recall_at_5']):.1%}) |",
        f"| Val Loss | {float(best_row['val_loss']):.4f} |",
        f"| Train Loss | {float(best_row['train_loss']):.4f} |",
        f"| kNN Accuracy | {accuracy:.4f} ({accuracy:.1%}) |",
        f"| Distance Ratio (inter/intra) | {dist_ratio:.2f} |",
        "",
        "## Final Epoch Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Epoch | {int(final_row['epoch'])} |",
        f"| Train Loss | {float(final_row['train_loss']):.4f} |",
        f"| Val Loss | {float(final_row['val_loss']):.4f} |",
        f"| Recall@1 | {float(final_row['recall_at_1']):.4f} |",
        f"| Recall@5 | {float(final_row['recall_at_5']):.4f} |",
        f"| Learning Rate | {final_row['learning_rate']} |",
        "",
    ]

    # Per-class accuracy table
    if per_class_acc is not None and class_names is not None:
        lines.append("## Per-Class Accuracy (kNN)")
        lines.append("")
        lines.append("| Class | Accuracy |")
        lines.append("|-------|----------|")
        for i, name in enumerate(class_names):
            lines.append(f"| {name} | {per_class_acc[i]:.1%} |")
        lines.append("")

    # Training configuration
    if config:
        lines.extend([
            "## Training Configuration",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Backbone | {config.get('model', {}).get('backbone_name', config.get('model', {}).get('backbone', 'N/A'))} |",
            f"| Embedding Dim | {config.get('model', {}).get('embedding_dim', 'N/A')} |",
            f"| Loss | {config.get('training', {}).get('loss', {}).get('type', 'N/A')} |",
            f"| Optimizer | {config.get('training', {}).get('optimizer', 'N/A')} |",
            f"| LR | {config.get('training', {}).get('learning_rate', 'N/A')} |",
            f"| Batch Size | {config.get('data', {}).get('batch_size', 'N/A')} |",
            f"| Mixed Precision | {config.get('hardware', {}).get('use_amp', 'N/A')} |",
            "",
        ])

    # Embedding space analysis
    if emb_stats:
        lines.extend([
            f"## Embedding Space Analysis ($\\mathbb{{S}}^{{{emb_stats['embed_dim']}}}$)",
            "",
            f"| Propiedad | Valor |",
            f"|-----------|-------|",
            f"| Dimensión | {emb_stats['embed_dim']} |",
            f"| Muestras | {emb_stats['n_samples']} |",
            f"| Norma media | {emb_stats['norm_mean']:.4f} ± {emb_stats['norm_std']:.4f} |",
            f"| Separacion angular media | {emb_stats['mean_angular_sep']:.1f}deg |",
            f"| Separacion angular rango | [{emb_stats['min_angular_sep']:.1f}deg, {emb_stats['max_angular_sep']:.1f}deg] |",
            "",
        ])
        if emb_stats.get('compactness'):
            lines.append("### Compacidad por Clase")
            lines.append("")
            lines.append("| Clase | Cosine Sim al Centroide |")
            lines.append("|-------|------------------------|")
            for cls_name, comp in emb_stats['compactness'].items():
                emoji = '🟢' if comp >= 0.8 else ('🟡' if comp >= 0.6 else '🔴')
                lines.append(f"| {cls_name} | {comp:.4f} {emoji} |")
            lines.append("")

    # Quality assessment
    r1 = float(best_row['recall_at_1'])
    if r1 >= 0.95:
        quality = "EXCELLENT"
    elif r1 >= 0.90:
        quality = "VERY GOOD"
    elif r1 >= 0.80:
        quality = "GOOD"
    elif r1 >= 0.70:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS IMPROVEMENT"

    lines.extend([
        "## Quality Assessment",
        "",
        f"**Overall: {quality}** (R@1 = {r1:.1%})",
        "",
        "---",
        "",
        "## Generated Files",
        "",
        "- `training_metrics.csv` — Epoch-by-epoch metrics",
        "- `images/post_training/loss_curves.png` — Train/Val loss over epochs",
        "- `images/post_training/recall_curves.png` — Recall@1 and Recall@5",
        "- `images/post_training/lr_curve.png` — Learning rate schedule",
        "- `images/post_training/epoch_times.png` — Per-epoch duration",
        "- `images/post_training/tsne_3d.png` — 3D t-SNE embedding clusters (S^D)",
        "- `images/post_training/tsne_2d.png` — 2D t-SNE + density (S^D)",
        "- `images/post_training/embedding_space_analysis.png` — Análisis S^D: normas, coseno, compacidad, separación angular",
        "- `images/post_training/confusion_matrix.png` — kNN confusion matrix",
        "- `images/post_training/distance_distribution.png` — Intra/Inter distances",
    ])

    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    logger.info(f"Summary report saved: {output_path}")
    return report_text


def generate_gradcam_gallery(model, centroids, val_loader, device, output_dir, class_names,
                              n_per_class=4):
    """
    Generate per-class Grad-CAM heatmap gallery.
    For each class, shows n_per_class images with 3 panels:
      [Original | Heatmap (jet colormap) | Overlay]

    Uses EmbeddingSimilarityTarget to produce meaningful heatmaps for Metric Learning
    by maximizing cosine similarity with the predicted class centroid.

    Args:
        model: Trained model on device (eval mode)
        centroids: Pre-computed class centroids tensor (n_classes, embed_dim) on device
        val_loader: Validation DataLoader
        device: torch.device
        output_dir: Base directory to save gallery (creates gradcam_gallery/ subfolder)
        class_names: List of class names
        n_per_class: Number of samples per class to visualize
    """
    from src.utils.inference_utils import overlay_gradcam_on_image
    from src.utils.visual_explainer import create_explainer

    gallery_dir = Path(output_dir) / 'gradcam_gallery'
    gallery_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    explainer = create_explainer(model, device)  # created once, reused for all images

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # 1. Collect all val images, embeddings, predictions
    logger.info("  Collecting validation data for Grad-CAM...")
    all_images = []
    all_labels = []
    all_preds = []
    all_dists = []

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Grad-CAM inference", leave=False):
            images, labels = batch_data[0], batch_data[1]
            embs = model(images.to(device))
            dists = torch.cdist(embs, centroids)
            preds = torch.argmin(dists, dim=1)
            min_d = torch.min(dists, dim=1)[0]

            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_dists.append(min_d.cpu())

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_dists = torch.cat(all_dists)

    rng = np.random.RandomState(42)
    n_classes = len(class_names)

    # 2. For each class, select n_per_class samples and generate Grad-CAM
    for c_idx in range(n_classes):
        class_name = class_names[c_idx]
        mask = (all_labels == c_idx)
        indices = torch.where(mask)[0].numpy()

        if len(indices) == 0:
            logger.warning(f"  No samples for class {class_name}, skipping.")
            continue

        sel_count = min(n_per_class, len(indices))
        sel_indices = rng.choice(indices, size=sel_count, replace=False)

        fig, axes = plt.subplots(sel_count, 3, figsize=(15, 5 * sel_count))
        if sel_count == 1:
            axes = axes[np.newaxis, :]

        for row, idx in enumerate(sel_indices):
            img_tensor = all_images[idx]
            true_lbl = all_labels[idx].item()
            pred_lbl = all_preds[idx].item()
            dist_val = all_dists[idx].item()
            correct = true_lbl == pred_lbl

            # Denormalize for display
            img_denorm = (img_tensor.unsqueeze(0) * std + mean).clamp(0, 1)[0]
            img_np = img_denorm.permute(1, 2, 0).numpy()

            # Generate Grad-CAM with predicted class centroid
            pred_centroid = centroids[pred_lbl]
            heatmap = explainer.generate_heatmap(
                img_tensor.unsqueeze(0), centroid=pred_centroid
            )

            # Overlay
            overlay = overlay_gradcam_on_image(
                (img_np * 255).astype(np.uint8), heatmap, alpha=0.5
            )

            # Panel 1: Original
            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title('Original', fontsize=11, fontweight='bold')
            axes[row, 0].axis('off')

            # Panel 2: Heatmap (jet colormap)
            axes[row, 1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
            axes[row, 1].set_title('Grad-CAM Heatmap', fontsize=11, fontweight='bold')
            axes[row, 1].axis('off')

            # Panel 3: Overlay
            axes[row, 2].imshow(overlay)
            status = "OK" if correct else "WRONG"
            title_color = 'green' if correct else 'red'
            axes[row, 2].set_title(
                f'Pred: {class_names[pred_lbl]} | Dist: {dist_val:.3f} [{status}]',
                fontsize=10, fontweight='bold', color=title_color
            )
            axes[row, 2].axis('off')

        fig.suptitle(
            f'Grad-CAM — {class_name} ({sel_count} samples)',
            fontsize=16, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        safe_name = class_name.replace(' ', '_').replace('/', '_')
        save_path = gallery_dir / f'{safe_name}_gradcam.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Grad-CAM gallery saved: {save_path}")

    # 3. Summary: one sample per class in a single figure
    logger.info("  Generating Grad-CAM summary...")
    fig, axes = plt.subplots(n_classes, 3, figsize=(15, 4 * n_classes))
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for c_idx in range(n_classes):
        class_name = class_names[c_idx]
        mask = (all_labels == c_idx)
        indices = torch.where(mask)[0].numpy()

        if len(indices) == 0:
            for col in range(3):
                axes[c_idx, col].axis('off')
            continue

        idx = rng.choice(indices)
        img_tensor = all_images[idx]
        pred_lbl = all_preds[idx].item()
        dist_val = all_dists[idx].item()
        correct = all_labels[idx].item() == pred_lbl

        img_denorm = (img_tensor.unsqueeze(0) * std + mean).clamp(0, 1)[0]
        img_np = img_denorm.permute(1, 2, 0).numpy()

        pred_centroid = centroids[pred_lbl]
        heatmap = explainer.generate_heatmap(
            img_tensor.unsqueeze(0), centroid=pred_centroid
        )
        overlay = overlay_gradcam_on_image(
            (img_np * 255).astype(np.uint8), heatmap, alpha=0.5
        )

        axes[c_idx, 0].imshow(img_np)
        axes[c_idx, 0].set_ylabel(class_name, fontsize=11, fontweight='bold', rotation=0,
                                   labelpad=100, va='center')
        axes[c_idx, 0].set_xticks([])
        axes[c_idx, 0].set_yticks([])

        axes[c_idx, 1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[c_idx, 1].axis('off')

        axes[c_idx, 2].imshow(overlay)
        status = "OK" if correct else "WRONG"
        title_color = 'green' if correct else 'red'
        axes[c_idx, 2].set_title(
            f'{class_names[pred_lbl]} | d={dist_val:.3f} [{status}]',
            fontsize=9, fontweight='bold', color=title_color
        )
        axes[c_idx, 2].set_xticks([])
        axes[c_idx, 2].set_yticks([])

    # Column headers
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')

    fig.suptitle('Grad-CAM Summary — All Classes', fontsize=18, fontweight='bold', y=1.005)
    plt.tight_layout()
    summary_path = gallery_dir / 'gradcam_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Grad-CAM summary saved: {summary_path}")
    logger.info(f"  Grad-CAM gallery complete: {gallery_dir}")


def generate_fullimage_inference(model, centroids, device, config, output_dir, class_names,
                                  n_images_per_class=3, with_gradcam=True):
    """
    Run inference on FULL-RESOLUTION original images using FullImageClassifier.
    
    Pipeline per image:
      1. Detect grains with U2-Net (FullImageClassifier)
      2. Crop each grain from the original image
      3. Transform + embed each crop with the trained model
      4. Classify each crop by nearest centroid
      5. Draw annotated bounding boxes + labels on the full image
      6. Optionally overlay Grad-CAM heatmaps per grain
    
    NUEVO (2026-03-07): Usa FullImageClassifier en lugar de requerir .seg files.
    Lee imágenes desde annotation_root usando headers de .seg para encontrar paths originales.
    
    Args:
        model: Trained AnalogyNet on device (eval mode)
        centroids: Pre-computed class centroids (n_classes, embed_dim) on device
        device: torch.device
        config: Full configuration dict
        output_dir: Directory to save results
        class_names: List of class names
        n_images_per_class: Number of full images per class to process
        with_gradcam: Whether to generate Grad-CAM overlays per grain
    """
    import cv2
    from src.grain_detection.seg_format import SegFileReader
    from src.grain_detection.full_image_classifier import FullImageClassifier
    from src.utils.inference_utils import overlay_gradcam_on_image
    from src.utils.visual_explainer import create_explainer
    from src.data.transforms_enhanced import get_val_transforms

    gallery_dir = Path(output_dir) / 'fullimage_inference'
    gallery_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    explainer = create_explainer(model, device) if with_gradcam else None

    # Get image size and build validation transform
    image_size = config['data'].get('image_size', 252)
    val_config = {**config.get('augmentation', {}), **config['data']}
    val_transform = get_val_transforms(val_config)
    
    # Cargar reference embeddings para FullImageClassifier
    run_dir = Path(output_dir).parent.parent
    ref_path = run_dir / 'reference_embeddings.pt'
    if not ref_path.exists():
        logger.warning(f"  Reference embeddings not found at {ref_path}, skipping fullimage inference")
        return {}
    
    ref_data = torch.load(ref_path, map_location='cpu', weights_only=False)
    reference_embeddings = ref_data['embeddings'].cpu().numpy() if torch.is_tensor(ref_data['embeddings']) else ref_data['embeddings']
    reference_labels = ref_data['labels'].cpu().numpy() if torch.is_tensor(ref_data['labels']) else ref_data['labels']
    
    # Inicializar FullImageClassifier
    grain_config = config.get('grain_detection', {})
    full_classifier = FullImageClassifier(
        model=model,
        transforms=val_transform,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        class_names=class_names,
        device=device,
        grain_config=grain_config,
        k=5
    )

    # Normalization constants for denormalization
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Color palette for classes (distinct colors)
    palette = [
        (46, 204, 113), (52, 152, 219), (231, 76, 60),
        (241, 196, 15), (155, 89, 182), (26, 188, 156),
        (230, 126, 34), (149, 165, 166), (192, 57, 43), (41, 128, 185),
    ]

    # Buscar imágenes desde annotation_root (donde están los .seg)
    annotation_root = Path(config['data'].get('annotation_root', 'data/annotations'))
    if not annotation_root.exists():
        logger.warning(f"  Annotation root not found: {annotation_root}")
        return {}
    
    logger.info(f"  Full-image inference from annotations: {annotation_root}")

    rng = np.random.RandomState(42)
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    all_results = {}
    total_grains = 0
    total_correct = 0

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = annotation_root / cls_name
        if not cls_dir.is_dir():
            logger.warning(f"  Class annotation dir not found: {cls_dir}")
            continue

        # Find .seg files
        seg_files = sorted(list(cls_dir.glob('*.seg')))
        if len(seg_files) == 0:
            logger.warning(f"  No .seg files for {cls_name}")
            continue
        
        # Select random subset
        sel_count = min(n_images_per_class, len(seg_files))
        sel_seg_paths = [seg_files[i] for i in rng.choice(len(seg_files), sel_count, replace=False)]

        color = palette[cls_idx % len(palette)]

        for seg_path in sel_seg_paths:
            # Extraer path de imagen original desde header del .seg
            try:
                with open(seg_path, 'r') as f:
                    lines = f.readlines()
                img_name = None
                for line in lines[:10]:  # Buscar en primeras 10 líneas
                    if line.startswith('# Imagen:'):
                        img_name = line.split(':', 1)[1].strip()
                        break
                
                if not img_name:
                    logger.warning(f"  No image name found in {seg_path.name}")
                    continue
                
                # Buscar imagen en múltiples ubicaciones
                img_path = None
                search_paths = [
                    seg_path.parent / img_name,  # Mismo dir que .seg
                    Path('data/processed/train') / cls_name / img_name,
                    Path('data/processed/val') / cls_name / img_name,
                    Path('data/processed/test') / cls_name / img_name,
                ]
                for candidate in search_paths:
                    if candidate.exists():
                        img_path = candidate
                        break
                
                if img_path is None or not img_path.exists():
                    logger.warning(f"  Image not found for {seg_path.name}: {img_name}")
                    continue
                
                # Usar FullImageClassifier para detección + clasificación automática
                results = full_classifier.classify_full_image(str(img_path))
                grains = results.get('grains', [])
                
                if len(grains) == 0:
                    logger.warning(f"  No grains detected in {img_path.name}")
                    continue
                
                # Load image for visualization
                full_img = cv2.imread(str(img_path))
                if full_img is None:
                    continue
                h_img, w_img = full_img.shape[:2]
                display_img = full_img.copy()
                
            except Exception as e:
                logger.warning(f"  Error processing {seg_path.name}: {e}")
                continue

            img_grains_correct = 0
            img_grains_total = len(grains)
            
            # FullImageClassifier ya hizo la detección y clasificación
            # Solo necesitamos visualizar los resultados
            for g_idx, grain in enumerate(grains):
                bx, by, bw, bh = grain['bbox']
                pred_cls_name = grain['predicted_class']
                pred_cls = class_names.index(pred_cls_name) if pred_cls_name in class_names else -1
                confidence = grain['confidence']
                
                correct = (pred_cls == cls_idx)
                if correct:
                    img_grains_correct += 1

                # Draw bounding box on full image
                box_color = (0, 200, 0) if correct else (0, 0, 220)
                cv2.rectangle(display_img, (bx, by), (bx + bw, by + bh), box_color, 2)

                # Label
                label = f"{pred_cls_name} ({confidence:.2f})"
                font_scale = max(0.4, min(h_img, w_img) / 2000)
                thickness = max(1, int(min(h_img, w_img) / 800))
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(display_img, (bx, by - th - 6), (bx + tw + 4, by), box_color, -1)
                cv2.putText(display_img, label, (bx + 2, by - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            total_grains += img_grains_total
            total_correct += img_grains_correct
            acc = img_grains_correct / img_grains_total if img_grains_total > 0 else 0

            # Add title bar at top
            title = f"{cls_name} | {img_path.name} | {img_grains_correct}/{img_grains_total} grains correct ({acc:.0%})"
            bar_h = max(30, int(h_img * 0.04))
            title_bar = np.zeros((bar_h, w_img, 3), dtype=np.uint8)
            title_bar[:] = (40, 40, 40)
            cv2.putText(title_bar, title, (10, bar_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.2, (255, 255, 255), thickness)
            display_img = np.vstack([title_bar, display_img])

            # Save
            safe_name = f"{cls_name}_{img_path.stem}".replace(' ', '_')
            save_path = gallery_dir / f"{safe_name}.png"
            cv2.imwrite(str(save_path), display_img)

            all_results.setdefault(cls_name, []).append({
                'image': img_path.name,
                'grains': img_grains_total,
                'correct': img_grains_correct,
                'accuracy': acc,
            })

    # Generate summary figure
    if all_results:
        n_total_images = sum(len(v) for v in all_results.values())
        overall_acc = total_correct / total_grains if total_grains > 0 else 0

        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(all_results.keys())
        accs = []
        for name in names:
            c = sum(r['correct'] for r in all_results[name])
            t = sum(r['grains'] for r in all_results[name])
            accs.append(c / t if t > 0 else 0)

        colors_bar = ['#2ecc71' if a >= 0.9 else ('#f39c12' if a >= 0.7 else '#e74c3c') for a in accs]
        bars = ax.barh(names, accs, color=colors_bar, edgecolor='#333', linewidth=0.5)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Grain Classification Accuracy', fontsize=12)
        ax.set_title(f'Full-Image Inference — {total_correct}/{total_grains} grains ({overall_acc:.1%})',
                      fontsize=14, fontweight='bold')
        ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5)

        for bar, a in zip(bars, accs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{a:.1%}', va='center', fontsize=10, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(gallery_dir / 'fullimage_summary.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"  Full-image inference: {total_correct}/{total_grains} grains "
                     f"({overall_acc:.1%}) across {n_total_images} images -> {gallery_dir}")

    return all_results


def run_post_training_analysis(model, val_loader, config, run_dir, device, centroids=None):
    """
    Full post-training analysis pipeline.
    Generates all visualizations and saves them to run_dir/images/post_training/.
    Uses the BEST model (must be loaded before calling this).

    Centroids must be pre-computed and saved before calling this function.
    If centroids tensor is not passed, loads from class_centroids.pt in run_dir.

    Args:
        model: Trained model (best checkpoint, already on device)
        val_loader: Validation DataLoader (for evaluation)
        config: Full configuration dict
        run_dir: Run directory path (e.g., runs/2026-02-06_20-17-07)
        device: torch.device
        centroids: Pre-computed centroids tensor (n_classes, embed_dim). If None, loaded from file.
    """
    run_dir = Path(run_dir)
    post_dir = run_dir / 'images' / 'post_training'
    post_dir.mkdir(parents=True, exist_ok=True)

    class_names = val_loader.dataset.classes
    csv_path = run_dir / 'training_metrics.csv'

    # Load centroids from file if not provided
    if centroids is None:
        centroids_path = run_dir / 'class_centroids.pt'
        if centroids_path.exists():
            centroids, _ = load_centroids(str(centroids_path), device)
        else:
            logger.error(f"Centroids file not found: {centroids_path}")
            logger.error("Cannot run inference-based analyses without centroids.")
            return

    logger.info("=" * 70)
    logger.info("POST-TRAINING ANALYSIS (Best Model)")
    logger.info("=" * 70)

    # 1. Training curves from CSV
    if csv_path.exists():
        logger.info("[1/10] Generating training curves...")
        plot_training_curves(str(csv_path), str(post_dir))
    else:
        logger.warning(f"Metrics CSV not found: {csv_path}")

    # 2. t-SNE 3D visualization
    logger.info("[2/10] Generating t-SNE visualizations (2D + 3D)...")
    embeddings, labels = plot_tsne_3d(
        model, val_loader, device,
        str(post_dir / 'tsne_3d.png'),
        class_names
    )

    # 3. Embedding space analysis (S^D hypersphere)
    logger.info("[3/10] Generating embedding space analysis...")
    emb_stats = None
    try:
        emb_stats = plot_embedding_space_analysis(
            embeddings, labels, class_names,
            str(post_dir / 'embedding_space_analysis.png')
        )
    except Exception as e:
        logger.error(f"Embedding space analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # 4. Confusion matrix (kNN)
    logger.info("[4/10] Generating confusion matrix...")
    accuracy, per_class_acc = plot_confusion_matrix_knn(
        embeddings, labels, class_names,
        str(post_dir / 'confusion_matrix.png')
    )

    # 5. Distance distribution
    logger.info("[5/10] Generating distance distributions...")
    mean_intra, mean_inter, dist_ratio = plot_distance_distribution(
        embeddings, labels, class_names,
        str(post_dir / 'distance_distribution.png')
    )

    # 6. Inference preview (uses pre-computed centroids — no training data needed)
    logger.info("[6/10] Generating inference preview...")
    plot_inference_preview(
        model, centroids, val_loader, device,
        str(post_dir / 'inference_preview.png'),
        class_names, n_samples=10
    )

    # 7. Attention/Grad-CAM heatmap gallery (uses pre-computed centroids)
    logger.info("[7/10] Generating attention heatmap gallery...")
    try:
        generate_gradcam_gallery(
            model, centroids, val_loader, device,
            str(post_dir), class_names, n_per_class=4
        )
    except Exception as e:
        logger.error(f"Attention/Grad-CAM generation FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # 8. Per-class inference gallery (uses pre-computed centroids)
    logger.info("[8/10] Generating per-class inference gallery...")
    generate_inference_gallery(
        model, centroids, val_loader, device,
        str(post_dir), class_names, n_per_class=10
    )

    # 9. Full-image inference on original test images
    logger.info("[9/10] Generating full-image inference on test images...")
    try:
        generate_fullimage_inference(
            model, centroids, device, config,
            str(post_dir), class_names,
            n_images_per_class=3, with_gradcam=True
        )
    except Exception as e:
        logger.warning(f"Full-image inference failed (non-critical): {e}")
        import traceback
        logger.debug(traceback.format_exc())

    # 10. Summary report
    logger.info("[10/11] Generating summary report...")
    generate_summary_report(
        csv_path=str(csv_path),
        config=config,
        output_path=str(run_dir / 'training_report.md'),
        accuracy=accuracy,
        per_class_acc=per_class_acc,
        dist_ratio=dist_ratio,
        class_names=class_names,
        emb_stats=emb_stats
    )

    # 11. Spectral Analysis (SOTA 2026) - Validación algebraica del colapso dimensional
    logger.info("=" * 70)
    logger.info("[11/11] SPECTRAL ANALYSIS - Dimensional Collapse Detection")
    logger.info("=" * 70)
    try:
        spectral_dir = post_dir / 'spectral_analysis'
        spectral_dir.mkdir(parents=True, exist_ok=True)
        
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names
        )
        
        # Análisis global del espacio completo
        logger.info("  -> Global spectrum analysis...")
        global_results = analyzer.analyze_global_spectrum(str(spectral_dir))
        
        msg1 = f"     Global λ₁: {global_results['collapse_severity']:.3f}"
        logger.info(sanitize_for_ascii(msg1))
        logger.info(f"     Global Effective Rank: {global_results['effective_rank']}/{analyzer.n_dim}")
        
        # Análisis de todos los pares de clases
        logger.info("  -> Analyzing all class pairs...")
        pair_results = analyzer.analyze_all_pairs(str(spectral_dir))
        
        # Identificar clases problemáticas
        collapsed_pairs = [
            (pair, res) for pair, res in pair_results.items()
            if res.get('is_collapsed', False)
        ]
        
        if collapsed_pairs:
            logger.warning(f"  ! COLLAPSED PAIRS DETECTED: {len(collapsed_pairs)}/{len(pair_results)}")
            for (ca, cb), res in collapsed_pairs[:3]:  # Top 3
                msg_warn = f"     - {ca} vs {cb}: λ₁={res['collapse_severity']:.3f}, Rank={res['effective_rank']}"
                logger.warning(sanitize_for_ascii(msg_warn))
        else:
            logger.info(f"  OK No collapsed pairs detected (all healthy)")
        
        logger.info(f"  Spectral analysis complete: {spectral_dir}")
        
    except Exception as e:
        logger.error(f"Spectral analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("=" * 70)
    logger.info(f"Post-training analysis complete!")
    logger.info(f"Results in: {post_dir}")
    logger.info("=" * 70)


def run_evaluation_from_checkpoint(checkpoint_path, config_path='config.yaml', output_dir='results'):
    """
    Standalone evaluation entry point for the GUI Evaluation tab.
    Loads model from checkpoint, loads pre-computed centroids, and runs the
    full analysis pipeline WITHOUT touching the training data.

    Falls back to computing centroids only if the file is missing (with warning).

    Args:
        checkpoint_path: Path to model checkpoint (.pth)
        config_path: Path to config YAML
        output_dir: Directory to save all results

    Returns:
        dict with output paths and metrics
    """
    import yaml
    from torch.utils.data import DataLoader
    from src.models.analogy_net import AnalogyNet
    from src.data.dataset import MielDataset
    from src.data.transforms_enhanced import get_val_transforms

    logger.info("=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)

    # Load current config (for data paths) and checkpoint
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Load checkpoint FIRST to get the training config for model architecture
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Use the config stored in checkpoint for model architecture (it matches the weights)
    train_config = ckpt.get('config', config)
    model_cfg = train_config.get('model', config['model'])
    if train_config is not config:
        logger.info("Using model architecture from checkpoint config (ensures weight compatibility)")

    model = AnalogyNet(
        backbone=model_cfg['backbone'],
        embedding_dim=model_cfg['embedding_dim'],
        pretrained=False,
        projection_head_config=model_cfg.get('projection_head'),
        use_attention=model_cfg.get('use_attention', False),
        freeze_backbone=False
    )

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    epoch_info = ckpt.get('epoch', '?')
    val_loss_info = ckpt.get('val_loss', '?')
    logger.info(f"Model loaded (epoch {epoch_info}, val_loss={val_loss_info})")

    # Use training config for transforms (image_size etc must match what model expects)
    # but current config for data paths (user may have moved data)
    train_aug = train_config.get('augmentation', config.get('augmentation', {}))
    train_data = train_config.get('data', config['data'])
    transform_config = {**train_aug, **train_data}
    # Override data paths from current config (in case they changed)
    transform_config['train_dir'] = config['data']['train_dir']
    transform_config['val_dir'] = config['data']['val_dir']
    val_transforms = get_val_transforms(transform_config)

    # --- Load pre-computed centroids (preferred) or fallback ---
    ckpt_parent = Path(checkpoint_path).parent.parent
    centroids_path = ckpt_parent / 'class_centroids.pt'
    centroids = None

    if centroids_path.exists():
        centroids, _ = load_centroids(str(centroids_path), device)
        logger.info("Using pre-computed centroids from training phase (no train data needed)")
    else:
        logger.warning("class_centroids.pt not found — computing centroids from training data (fallback)")
        logger.warning("This is slow. Re-run training to generate centroids automatically.")
        train_dataset = MielDataset(root=config['data']['train_dir'], transform=val_transforms)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'],
                                  shuffle=False, num_workers=0, pin_memory=True)
        class_names_train = train_dataset.classes
        centroids = compute_and_save_centroids(model, train_loader, device, str(ckpt_parent), class_names_train)

    # Create val dataloader only (evaluation never needs train data when centroids exist)
    logger.info(f"Loading validation data from: {config['data']['val_dir']}")
    val_dataset = MielDataset(root=config['data']['val_dir'], transform=val_transforms)
    logger.info(f"Val: {len(val_dataset)} | Classes: {len(val_dataset.classes)}")

    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=True)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    post_dir = output_dir / 'images'
    post_dir.mkdir(parents=True, exist_ok=True)

    class_names = val_dataset.classes

    # Check if there's a training_metrics.csv alongside the checkpoint
    csv_path = ckpt_parent / 'training_metrics.csv'

    logger.info("=" * 70)
    logger.info("RUNNING EVALUATION ANALYSIS")
    logger.info("=" * 70)

    # 1. Training curves (if CSV exists)
    if csv_path.exists():
        logger.info("[1/9] Generating training curves from CSV...")
        plot_training_curves(str(csv_path), str(post_dir))
    else:
        logger.info("[1/9] No training_metrics.csv found — skipping training curves")

    # 2. t-SNE
    logger.info("[2/9] Generating t-SNE visualizations...")
    embeddings, labels = plot_tsne_3d(
        model, val_loader, device, str(post_dir / 'tsne_3d.png'), class_names
    )

    # 3. Embedding space analysis (S^D hypersphere)
    logger.info("[3/9] Generating embedding space analysis...")
    emb_stats = None
    try:
        emb_stats = plot_embedding_space_analysis(
            embeddings, labels, class_names,
            str(post_dir / 'embedding_space_analysis.png')
        )
    except Exception as e:
        logger.error(f"Embedding space analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # 4. Confusion matrix
    logger.info("[4/9] Generating confusion matrix...")
    accuracy, per_class_acc = plot_confusion_matrix_knn(
        embeddings, labels, class_names, str(post_dir / 'confusion_matrix.png')
    )

    # 5. Distance distribution
    logger.info("[5/9] Generating distance distributions...")
    mean_intra, mean_inter, dist_ratio = plot_distance_distribution(
        embeddings, labels, class_names, str(post_dir / 'distance_distribution.png')
    )

    # 6. Inference preview (uses pre-computed centroids)
    logger.info("[6/9] Generating inference preview...")
    plot_inference_preview(
        model, centroids, val_loader, device,
        str(post_dir / 'inference_preview.png'), class_names, n_samples=10
    )

    # 7. Grad-CAM heatmap gallery (uses pre-computed centroids)
    logger.info("[7/9] Generating Grad-CAM heatmap gallery...")
    try:
        generate_gradcam_gallery(
            model, centroids, val_loader, device,
            str(post_dir), class_names, n_per_class=4
        )
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed (non-critical): {e}")

    # 8. Per-class inference gallery (uses pre-computed centroids)
    logger.info("[8/9] Generating per-class inference gallery...")
    generate_inference_gallery(
        model, centroids, val_loader, device,
        str(post_dir), class_names, n_per_class=10
    )

    # 9. Summary report
    logger.info("[9/9] Generating summary report...")
    if csv_path.exists():
        generate_summary_report(
            csv_path=str(csv_path), config=config,
            output_path=str(output_dir / 'evaluation_report.md'),
            accuracy=accuracy, per_class_acc=per_class_acc,
            dist_ratio=dist_ratio, class_names=class_names,
            emb_stats=emb_stats
        )

    logger.info("=" * 70)
    logger.info(f"Evaluation complete! Results in: {output_dir}")
    logger.info("=" * 70)

    return {
        'output_dir': str(output_dir),
        'accuracy': accuracy,
        'per_class_acc': per_class_acc.tolist() if per_class_acc is not None else [],
        'dist_ratio': dist_ratio,
        'class_names': class_names,
    }
