"""
Run Setup & Pre-Training Analysis Module

Creates structured run directories and generates pre-training dataset
analysis including distribution charts, sample image grids, and statistics.

Folder structure created:
    runs/YYYY-MM-DD_HH-MM-SS/
        config/                  # Copy of config.yaml used
        checkpoints/             # Model checkpoints (best, periodic, final)
        logs/                    # Training logs
        images/
            pre_training/        # Dataset analysis BEFORE training
                class_distribution.png
                sample_grid_per_class.png
                augmentation_preview.png
                dataset_stats.json
            post_training/       # Results AFTER training
                loss_curves.png
                recall_curves.png
                confusion_matrix.png
                embedding_tsne.png

Author: MetricLearning Project
Date: 2026-02-06
"""

import json
import shutil
import random
import logging
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from PIL import Image

from src.utils.config_utils import get_backbone_name

logger = logging.getLogger(__name__)


def create_run_directory(base_dir: str = "runs") -> dict:
    """
    Creates a timestamped run directory with all required subfolders.
    
    Args:
        base_dir: Base directory for all runs
        
    Returns:
        dict with paths to all created directories
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(base_dir) / timestamp
    
    paths = {
        'run_dir': run_dir,
        'config_dir': run_dir / 'config',
        'checkpoints_dir': run_dir / 'checkpoints',
        'logs_dir': run_dir / 'logs',
        'images_dir': run_dir / 'images',
        'pre_training_dir': run_dir / 'images' / 'pre_training',
        'post_training_dir': run_dir / 'images' / 'post_training',
    }
    
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {path}")
    
    # Convert to strings for serialization
    return {k: str(v) for k, v in paths.items()}


def save_config_copy(config_path: str, config_dir: str):
    """Copy the config file used for this run"""
    src = Path(config_path)
    dst = Path(config_dir) / src.name
    shutil.copy2(src, dst)
    logger.info(f"Config saved to: {dst}")


def generate_class_distribution_chart(dataset, save_path: str):
    """
    Generates a high-quality class distribution bar chart.
    
    Args:
        dataset: MielDataset with .classes, .samples, .targets
        save_path: Path to save the chart PNG
    """
    labels = [s[1] for s in dataset.samples]
    class_counts = Counter(labels)
    
    class_names = dataset.classes
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = ax.bar(range(len(class_names)), counts, color=colors, edgecolor='#333333', linewidth=0.8)
    
    # Values on top of bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + max(counts) * 0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Clase', fontsize=13, fontweight='bold')
    ax.set_ylabel('Número de Imágenes', fontsize=13, fontweight='bold')
    ax.set_title('Distribución de Clases en Dataset de Entrenamiento', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add total annotation
    total = sum(counts)
    ax.text(0.98, 0.95, f'Total: {total} imágenes\nClases: {len(class_names)}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Class distribution chart saved: {save_path}")


def _is_grain_dataset(dataset) -> bool:
    """Check if dataset is a SegmentedGrainDataset (grain crops, not full images)."""
    return hasattr(dataset, '_grain_samples') and hasattr(dataset, '_extract_crop')


def _get_grain_crop_pil(dataset, idx: int) -> Image.Image:
    """Get a grain crop as PIL image from SegmentedGrainDataset without transforms."""
    import cv2
    image_path, class_idx, bbox, saliency, contour = dataset._grain_samples[idx]
    image = cv2.imread(image_path)
    if image is None:
        raise IOError(f"No se pudo leer: {image_path}")
    grain = {"bbox": bbox, "saliency": saliency, "contour": contour}
    crop = dataset._extract_crop(image, grain)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def generate_sample_grid(dataset, save_path: str, samples_per_class: int = 10):
    """
    Generates a grid showing top N sample images per class (raw, no transforms).
    For SegmentedGrainDataset, shows actual grain crops instead of full images.
    
    Args:
        dataset: MielDataset or SegmentedGrainDataset (uses .samples and .classes)
        save_path: Path to save the grid PNG
        samples_per_class: Number of sample images per class
    """
    class_names = dataset.classes
    n_classes = len(class_names)
    is_grain = _is_grain_dataset(dataset)
    
    # Group sample INDICES by class
    class_indices = {i: [] for i in range(n_classes)}
    for idx, (path, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)
    
    # Select random sample indices per class
    selected = {}
    for cls_idx, indices in class_indices.items():
        n = min(samples_per_class, len(indices))
        selected[cls_idx] = random.sample(indices, n)
    
    # Create grid
    fig, axes = plt.subplots(n_classes, samples_per_class, 
                              figsize=(samples_per_class * 2.2, n_classes * 2.5))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    title = 'Crops de Granos por Clase' if is_grain else 'Muestras por Clase (Top 10 aleatorias)'
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    for cls_idx in range(n_classes):
        indices = selected.get(cls_idx, [])
        for col in range(samples_per_class):
            ax = axes[cls_idx, col]
            if col < len(indices):
                try:
                    if is_grain:
                        img = _get_grain_crop_pil(dataset, indices[col])
                    else:
                        img = Image.open(dataset.samples[indices[col]][0]).convert('RGB')
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                       transform=ax.transAxes, color='gray')
            
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col == 0:
                ax.set_ylabel(class_names[cls_idx], fontsize=10, fontweight='bold', rotation=0,
                            labelpad=80, ha='right', va='center')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Sample grid saved: {save_path}")


def generate_augmentation_preview(dataset, transform_config: dict, save_path: str, n_samples: int = 4):
    """
    Shows original vs augmented images side by side.
    For SegmentedGrainDataset, shows grain crops instead of full images.
    
    Args:
        dataset: MielDataset or SegmentedGrainDataset
        transform_config: Augmentation config dict
        save_path: Path to save the preview PNG
        n_samples: Number of sample images to show
    """
    from src.data.transforms_enhanced import get_train_transforms
    
    train_tf = get_train_transforms(transform_config)
    is_grain = _is_grain_dataset(dataset)
    
    # Pick one random sample index per class
    class_sample_idx = {}
    for idx, (path, label) in enumerate(dataset.samples):
        if label not in class_sample_idx:
            class_sample_idx[label] = idx
        if len(class_sample_idx) >= n_samples:
            break
    
    sample_indices = list(class_sample_idx.values())[:n_samples]
    sample_labels = list(class_sample_idx.keys())[:n_samples]
    
    fig, axes = plt.subplots(n_samples, 5, figsize=(14, n_samples * 3))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    title = 'Aumentación de Crops de Granos' if is_grain else 'Vista Previa de Aumentación de Datos'
    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    
    # Column headers
    col_titles = ['Original', 'Aug #1', 'Aug #2', 'Aug #3', 'Aug #4']
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')
    
    for row, (sample_idx, label) in enumerate(zip(sample_indices, sample_labels)):
        try:
            if is_grain:
                img = _get_grain_crop_pil(dataset, sample_idx)
            else:
                img = Image.open(dataset.samples[sample_idx][0]).convert('RGB')
            img_np = np.array(img)
        except Exception:
            continue
        
        # Original
        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(dataset.classes[label], fontsize=9, fontweight='bold',
                                rotation=0, labelpad=60, ha='right', va='center')
        
        # Augmented versions
        for col in range(1, 5):
            try:
                # MicroscopyAugmentation.__call__ returns tensor directly
                # Albumentations Compose returns dict with 'image' key
                result = train_tf(img_np)
                if isinstance(result, dict):
                    augmented = result['image']
                else:
                    augmented = result
                if hasattr(augmented, 'permute'):
                    aug_np = augmented.permute(1, 2, 0).numpy()
                    # Denormalize (ImageNet normalization)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    aug_np = np.clip(aug_np * std + mean, 0, 1)
                else:
                    aug_np = augmented
                axes[row, col].imshow(aug_np)
            except Exception:
                axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center',
                                   transform=axes[row, col].transAxes)
        
        for col in range(5):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Augmentation preview saved: {save_path}")


def _sample_image_sizes(dataset, max_samples: int = 200) -> dict:
    """Sample image sizes from dataset to compute statistics."""
    # For grain datasets, all crops have the same size
    if _is_grain_dataset(dataset):
        cs = dataset.crop_size
        return {
            'sampled': len(dataset),
            'width_min': cs, 'width_max': cs, 'width_mean': float(cs),
            'height_min': cs, 'height_max': cs, 'height_mean': float(cs),
            'aspect_ratio_mean': 1.0,
            'note': f'Grain crops ({cs}x{cs}px)',
        }
    widths, heights = [], []
    indices = random.sample(range(len(dataset.samples)), min(max_samples, len(dataset.samples)))
    for idx in indices:
        path = dataset.samples[idx][0]
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue
    if not widths:
        return {}
    return {
        'sampled': len(widths),
        'width_min': min(widths),
        'width_max': max(widths),
        'width_mean': round(np.mean(widths), 1),
        'height_min': min(heights),
        'height_max': max(heights),
        'height_mean': round(np.mean(heights), 1),
        'aspect_ratio_mean': round(np.mean([w/h for w, h in zip(widths, heights)]), 3),
    }


def generate_dataset_stats(train_dataset, val_dataset, config: dict, save_path: str):
    """
    Generates a JSON file with comprehensive dataset statistics.
    
    Args:
        train_dataset: Training MielDataset
        val_dataset: Validation MielDataset
        config: Full config dict
        save_path: Path to save the JSON
    """
    train_labels = [s[1] for s in train_dataset.samples]
    val_labels = [s[1] for s in val_dataset.samples]
    
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    
    # Sample image sizes
    logger.info("Sampling image sizes for statistics...")
    img_size_stats = _sample_image_sizes(train_dataset)
    
    stats = {
        'timestamp': datetime.datetime.now().isoformat(),
        'train': {
            'total_samples': len(train_dataset),
            'num_classes': len(train_dataset.classes),
            'classes': train_dataset.classes,
            'class_distribution': {
                train_dataset.classes[k]: v for k, v in sorted(train_counts.items())
            },
            'min_samples_per_class': min(train_counts.values()),
            'max_samples_per_class': max(train_counts.values()),
            'mean_samples_per_class': round(np.mean(list(train_counts.values())), 1),
            'std_samples_per_class': round(np.std(list(train_counts.values())), 1),
            'image_sizes': img_size_stats,
        },
        'val': {
            'total_samples': len(val_dataset),
            'num_classes': len(val_dataset.classes),
            'class_distribution': {
                val_dataset.classes[k]: v for k, v in sorted(val_counts.items())
            },
        },
        'config_summary': {
            'batch_size': config['data']['batch_size'],
            'classes_per_batch': config['data']['classes_per_batch'],
            'samples_per_class': config['data']['samples_per_class'],
            'image_size': config['data']['image_size'],
            'epochs': config['training']['epochs'],
            'learning_rate': config['training']['learning_rate'],
            'loss_type': config['training']['loss']['type'],
            'optimizer': config['training']['optimizer'],
            'model_backbone': get_backbone_name(config),
            'embedding_dim': config['model']['embedding_dim'],
            'use_amp': config['hardware']['use_amp'],
        }
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset stats saved: {save_path}")
    return stats


def generate_class_balance_chart(train_dataset, val_dataset, save_path: str):
    """
    Generates a grouped bar chart comparing train vs val distribution.
    """
    class_names = train_dataset.classes
    n_classes = len(class_names)
    
    train_labels = [s[1] for s in train_dataset.samples]
    val_labels = [s[1] for s in val_dataset.samples]
    
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    
    train_vals = [train_counts.get(i, 0) for i in range(n_classes)]
    val_vals = [val_counts.get(i, 0) for i in range(n_classes)]
    
    x = np.arange(n_classes)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_vals, width, label='Train', color='#4ec9b0', edgecolor='#333', linewidth=0.5)
    bars2 = ax.bar(x + width/2, val_vals, width, label='Val', color='#dcdcaa', edgecolor='#333', linewidth=0.5)
    
    ax.set_xlabel('Clase', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número de Imágenes', fontsize=12, fontweight='bold')
    ax.set_title('Distribución Train vs Validación por Clase', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Values on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{int(h)}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Class balance chart saved: {save_path}")


def run_pre_training_analysis(config: dict, config_path: str, train_dataset, val_dataset) -> dict:
    """
    Full pre-training analysis pipeline.
    Creates run directory and generates all pre-training visualizations.
    
    Args:
        config: Full config dict
        config_path: Path to the config YAML file
        train_dataset: Training MielDataset (raw, with transforms)
        val_dataset: Validation MielDataset
        
    Returns:
        dict with run_paths and generated file paths
    """
    # Create run directory
    run_paths = create_run_directory()
    pre_dir = run_paths['pre_training_dir']
    
    # Save config copy
    save_config_copy(config_path, run_paths['config_dir'])
    
    # Generate all pre-training analysis
    charts = {}
    
    # 1. Class distribution chart
    dist_path = str(Path(pre_dir) / 'class_distribution.png')
    generate_class_distribution_chart(train_dataset, dist_path)
    charts['class_distribution'] = dist_path
    
    # 2. Train vs Val balance chart
    balance_path = str(Path(pre_dir) / 'train_val_balance.png')
    generate_class_balance_chart(train_dataset, val_dataset, balance_path)
    charts['train_val_balance'] = balance_path
    
    # 3. Sample grid per class (top 10)
    grid_path = str(Path(pre_dir) / 'sample_grid_per_class.png')
    # Use raw dataset (no transforms) for sample grid — grain or image mode
    dataset_mode = config.get('grain_detection', {}).get('dataset_mode', 'image')
    if dataset_mode == 'grain':
        from src.data.grain_dataset import SegmentedGrainDataset
        grain_cfg = config.get('grain_detection', {})
        raw_train = SegmentedGrainDataset(
            root=config['data']['train_dir'],
            transform=None,
            crop_padding=grain_cfg.get('crop_padding', 0.2),
            crop_size=grain_cfg.get('crop_size', 256),
            annotation_root='data/annotations',
        )
    else:
        from src.data.dataset import MielDataset
        raw_train = MielDataset(root=config['data']['train_dir'], transform=None)
    generate_sample_grid(raw_train, grid_path, samples_per_class=10)
    charts['sample_grid'] = grid_path
    
    # 4. Augmentation preview
    aug_path = str(Path(pre_dir) / 'augmentation_preview.png')
    transform_config = {**config['augmentation'], **config['data']}
    generate_augmentation_preview(raw_train, transform_config, aug_path, n_samples=4)
    charts['augmentation_preview'] = aug_path
    
    # 5. Dataset stats JSON
    stats_path = str(Path(pre_dir) / 'dataset_stats.json')
    stats = generate_dataset_stats(train_dataset, val_dataset, config, stats_path)
    charts['dataset_stats'] = stats_path
    
    result = {
        'run_paths': run_paths,
        'charts': charts,
        'stats': stats,
    }
    
    logger.info(f"Pre-training analysis complete. Run directory: {run_paths['run_dir']}")
    return result
