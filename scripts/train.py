"""
Training Script for Metric Learning System
Implements SOTA techniques (2024) for honey type classification

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume models/checkpoint_epoch_20.pth

Author: MetricLearning Project
Date: 2026-01-05
"""

import argparse
import yaml
import torch
import logging
import time
import statistics
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler

from src.models.analogy_net import AnalogyNet
from src.data.dataset import MielDataset
from src.data.transforms_enhanced import get_train_transforms, get_val_transforms
from src.training.trainer_enhanced import MetricLearningTrainer


def setup_logging(log_dir: str = 'logs', log_level: str = 'INFO'):
    """Setup logging configuration"""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """
    Create train and validation dataloaders with M-Per-Class sampling.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger = logging.getLogger(__name__)
    
    # Get data configuration
    data_config = config['data']
    aug_config = config['augmentation']
    
    # Merge data and augmentation config for transforms
    transform_config = {**aug_config, **data_config}
    
    # Create transforms (support DeiT III 3-Augment mode)
    aug_mode = config.get('sota', {}).get('augmentation_mode', 'full')
    train_transforms = get_train_transforms(transform_config, mode=aug_mode)
    val_transforms = get_val_transforms(transform_config)
    
    # num_workers > 0 enables parallel data loading in background processes,
    # keeping the GPU fed while the next batch is prepared.
    # Default to 4 workers on Windows (spawn-based multiprocessing).
    num_workers = min(data_config.get('num_workers', 4), 4)
    if num_workers == 0:
        logger.warning("⚠️  num_workers=0: data loading is BLOCKING. "
                       "Consider setting num_workers: 2-4 in config.yaml for faster training.")
    
    # Create datasets — choose between full images or segmented grain crops
    dataset_mode = config.get('grain_detection', {}).get('dataset_mode', 'image')
    
    if dataset_mode == 'grain':
        # Modo grain: cada sample es un crop de grano individual (de .seg files)
        # Los .seg DEBEN existir previamente (generados en el tab de Contornos, paso 2)
        from src.data.grain_dataset import SegmentedGrainDataset
        grain_cfg = config.get('grain_detection', {})
        
        use_mask = grain_cfg.get('use_mask', True)
        use_masked_pooling = grain_cfg.get('masked_pooling', False)
        # Gray background: after ImageNet normalization ≈ zero → minimal contribution to mean pool
        # This naturally suppresses background without masked pooling (avoids mask-augmentation desync)
        mask_bg = 'gray'
        
        logger.info(f"[GRAIN MODE] Loading segmented grain crops from: {data_config['train_dir']}")
        logger.info(f"[GRAIN MODE] use_mask={use_mask}, mask_bg={mask_bg}, masked_pooling={use_masked_pooling}")
        train_dataset = SegmentedGrainDataset(
            root=data_config['train_dir'],
            transform=train_transforms,
            crop_padding=grain_cfg.get('crop_padding', 0.2),
            crop_size=grain_cfg.get('crop_size', 252),
            use_mask=use_mask,
            mask_bg_mode=mask_bg,
            min_saliency=0.0,
            return_mask=use_masked_pooling,
            annotation_root='data/annotations',
        )
        
        if len(train_dataset) == 0:
            raise RuntimeError(
                f"No se encontraron granos en {data_config['train_dir']}. "
                f"Ejecuta primero 'Análisis de Contornos' para generar los archivos .seg."
            )
        
        logger.info(f"[GRAIN MODE] Loading segmented grain crops from: {data_config['val_dir']}")
        val_dataset = SegmentedGrainDataset(
            root=data_config['val_dir'],
            transform=val_transforms,
            crop_padding=grain_cfg.get('crop_padding', 0.2),
            crop_size=grain_cfg.get('crop_size', 252),
            use_mask=use_mask,
            mask_bg_mode=mask_bg,
            min_saliency=0.0,
            return_mask=use_masked_pooling,
            annotation_root='data/annotations',
        )
        
        if len(val_dataset) == 0:
            logger.warning(
                f"[GRAIN MODE] Val set vacío ({data_config['val_dir']}). "
                f"¿Faltan archivos .seg? Ejecuta 'Análisis de Contornos' sobre todos los splits."
            )
        
        logger.info(f"[GRAIN MODE] Train grains: {len(train_dataset)} | Val grains: {len(val_dataset)}")
    else:
        # Modo image: cada sample es una imagen completa (ImageFolder estándar)
        logger.info(f"Loading training data from: {data_config['train_dir']}")
        train_dataset = MielDataset(
            root=data_config['train_dir'],
            transform=train_transforms
        )
        
        logger.info(f"Loading validation data from: {data_config['val_dir']}")
        val_dataset = MielDataset(
            root=data_config['val_dir'],
            transform=val_transforms
        )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.classes)}")
    
    # Update config with number of classes (needed for ArcFace)
    config['num_classes'] = len(train_dataset.classes)
    
    # Create M-Per-Class sampler for training (CRITICAL for metric learning)
    # .targets works for both MielDataset and SegmentedGrainDataset
    train_labels = train_dataset.targets
    sampler = MPerClassSampler(
        labels=train_labels,
        m=data_config['samples_per_class'],
        length_before_new_iter=len(train_dataset)
    )
    
    # DataLoader configuration for Windows:
    #   - persistent_workers=True: workers stay alive between epochs (avoids
    #     costly respawn of 4 Python processes every epoch)
    #   - prefetch_factor=4: each worker buffers 4 batches → 16 total in queue,
    #     enough to absorb I/O variance and keep the GPU fed
    #   - num_workers=4: good balance for Windows (spawn-based multiprocessing)
    pin_memory = data_config.get('pin_memory', True)
    use_persistent = num_workers > 0
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=data_config.get('drop_last', True),
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=use_persistent,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    logger.info(f"Batch configuration: {data_config['classes_per_batch']} classes × {data_config['samples_per_class']} samples = {data_config['batch_size']} batch size")
    
    return train_loader, val_loader


def create_model(config: dict) -> AnalogyNet:
    """
    Create AnalogyNet model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AnalogyNet model
    """
    logger = logging.getLogger(__name__)
    
    model_config = config['model']
    
    sota_config = config.get('sota', {})
    model = AnalogyNet(
        backbone=model_config['backbone'],
        embedding_dim=model_config['embedding_dim'],
        pretrained=model_config['pretrained'],
        projection_head_config=model_config.get('projection_head'),
        use_attention=model_config.get('use_attention', False),
        freeze_backbone=model_config.get('freeze_backbone', False),
        embedding_space=sota_config.get('embedding_space', 'euclidean'),
        hyperbolic_curvature=sota_config.get('hyperbolic_curvature', 1.0)
    )
    
    logger.info(f"Model created: {model_config['backbone']}")
    logger.info(f"Embedding dimension: {model_config['embedding_dim']}")
    logger.info(f"Pretrained: {model_config['pretrained']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def estimate_training_time(config: dict, train_loader, val_loader, model, logger, benchmark_batches: int = 8) -> dict:
    """
    Benchmarks real training steps (forward + loss + backward + optimizer)
    and validation batches to estimate total training time accurately.
    Uses num_workers=0 internally to avoid Windows multiprocessing deadlocks,
    then adds measured data-loading overhead factor.
    
    Args:
        config: Full configuration dictionary
        train_loader: Training DataLoader (used only for batch count / dataset size)
        val_loader: Validation DataLoader (used only for batch count / dataset size)
        model: The model (already on device)
        logger: Logger instance
        benchmark_batches: Number of batches to benchmark
        
    Returns:
        Dict with detailed time estimates per phase and total
    """
    from torch.amp import autocast, GradScaler
    from pytorch_metric_learning.losses import MultiSimilarityLoss, TripletMarginLoss
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['hardware'].get('use_gpu', True) else 'cpu')
    model.to(device)
    model.train()
    
    use_amp = config['hardware'].get('use_amp', True)
    epochs = config['training']['epochs']
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    batch_size = config['data']['batch_size']
    
    # Setup loss function for realistic benchmark
    # Filter params to only those accepted by each loss type
    loss_type = config['training']['loss']['type']
    all_loss_params = config['training']['loss'].get('params', {})
    if loss_type == 'multi_similarity':
        ms_keys = {'alpha', 'beta', 'base'}
        filtered = {k: v for k, v in all_loss_params.items() if k in ms_keys}
        bench_loss_func = MultiSimilarityLoss(**filtered).to(device)
    elif loss_type == 'triplet':
        tri_keys = {'margin', 'triplet_margin'}
        filtered = {k: v for k, v in all_loss_params.items() if k in tri_keys}
        if 'triplet_margin' in filtered:
            filtered['margin'] = filtered.pop('triplet_margin')
        bench_loss_func = TripletMarginLoss(**filtered).to(device)
    else:
        bench_loss_func = MultiSimilarityLoss().to(device)
    
    # Setup optimizer and scaler for realistic benchmark
    bench_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bench_scaler = GradScaler('cuda') if use_amp else None
    gradient_clip_norm = config['training'].get('gradient_clip_norm', 5.0)
    
    # Create lightweight single-process loaders for benchmarking
    bench_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    bench_val_loader = DataLoader(
        val_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
    
    # --- CUDA warmup (2 batches, NOT timed) ---
    # First CUDA ops trigger kernel compilation + memory allocation → 5-10x slower.
    # Discard warmup batches so they don't pollute the estimate.
    warmup_batches = 2
    logger.info(f"CUDA warmup ({warmup_batches} batches, not timed)...")
    bench_train_iter = iter(bench_train_loader)
    for i in range(warmup_batches):
        try:
            batch_data = next(bench_train_iter)
        except StopIteration:
            bench_train_iter = iter(bench_train_loader)
            batch_data = next(bench_train_iter)
        images, labels = batch_data[0], batch_data[1]
        images = images.to(device)
        labels = labels.to(device)
        bench_optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast('cuda'):
                embeddings = model(images)
                loss = bench_loss_func(embeddings, labels)
            bench_scaler.scale(loss).backward()
            bench_scaler.unscale_(bench_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            bench_scaler.step(bench_optimizer)
            bench_scaler.update()
        else:
            embeddings = model(images)
            loss = bench_loss_func(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            bench_optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
    logger.info("Warmup complete.")
    
    # --- Benchmark FULL training steps (forward + loss + backward + optimizer) ---
    logger.info(f"Benchmarking {benchmark_batches} training steps (forward + loss + backward + optimizer)...")
    train_times = []
    data_load_times = []
    
    t_data_start = time.perf_counter()
    for i in range(min(benchmark_batches, num_train_batches)):
        try:
            batch_data = next(bench_train_iter)
        except StopIteration:
            bench_train_iter = iter(bench_train_loader)
            batch_data = next(bench_train_iter)
        images, labels = batch_data[0], batch_data[1]
        images = images.to(device)
        labels = labels.to(device)
        data_load_times.append(time.perf_counter() - t_data_start)
        
        t0 = time.perf_counter()
        bench_optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with autocast('cuda'):
                embeddings = model(images)
                loss = bench_loss_func(embeddings, labels)
            bench_scaler.scale(loss).backward()
            bench_scaler.unscale_(bench_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            bench_scaler.step(bench_optimizer)
            bench_scaler.update()
        else:
            embeddings = model(images)
            loss = bench_loss_func(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            bench_optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_times.append(time.perf_counter() - t0)
        t_data_start = time.perf_counter()
    
    # --- Benchmark validation batches ---
    logger.info("Benchmarking validation batches...")
    val_times = []
    model.eval()
    bench_val_iter = iter(bench_val_loader)
    with torch.no_grad():
        for i in range(min(benchmark_batches, num_val_batches)):
            try:
                batch_data = next(bench_val_iter)
            except StopIteration:
                break
            images, labels = batch_data[0], batch_data[1]
            images = images.to(device)
            labels = labels.to(device)
            
            t0 = time.perf_counter()
            if use_amp:
                with autocast('cuda'):
                    embeddings = model(images)
                    loss = bench_loss_func(embeddings, labels)
            else:
                embeddings = model(images)
                loss = bench_loss_func(embeddings, labels)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            val_times.append(time.perf_counter() - t0)
    
    # --- Calculate estimates (use MEDIAN for robustness against outliers) ---
    avg_train_step = statistics.median(train_times) if train_times else 1.0
    avg_val_batch = statistics.median(val_times) if val_times else 0.5
    avg_data_load = statistics.median(data_load_times) if data_load_times else 0.1
    
    # Data loading overhead:
    #   num_workers > 0: loading is overlapped with GPU work; add ~15% for stalls
    #   num_workers = 0: loading is BLOCKING (sequential), so total = step + data_load
    num_workers = config['data'].get('num_workers', 4)
    if num_workers > 0:
        data_overhead_factor = 1.15
    else:
        data_overhead_factor = 1.0 + (avg_data_load / avg_train_step) if avg_train_step > 0 else 1.5
    
    avg_train_batch_full = avg_train_step * data_overhead_factor
    
    # Validation also includes Recall@K computation overhead (~5s per epoch)
    recall_overhead_sec = 5.0
    
    # Reference embedding extraction: happens ONCE at the end (final_model.pth),
    # NOT during training epochs.  Centroids are also computed once post-training.
    # So per-epoch overhead = 0; one-time cost added separately.
    ref_embed_one_time_sec = num_train_batches * avg_val_batch  # one full pass over train set
    
    train_epoch_sec = avg_train_batch_full * num_train_batches
    val_epoch_sec = (avg_val_batch * num_val_batches) + recall_overhead_sec
    epoch_total_sec = train_epoch_sec + val_epoch_sec
    total_sec = (epoch_total_sec * epochs) + ref_embed_one_time_sec
    
    def fmt_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s}s"
        else:
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return f"{h}h {m}m {s}s"
    
    estimate = {
        'avg_train_batch_sec': avg_train_batch_full,
        'avg_val_batch_sec': avg_val_batch,
        'avg_train_step_sec': avg_train_step,
        'avg_data_load_sec': avg_data_load,
        'data_overhead_factor': data_overhead_factor,
        'train_batches_per_epoch': num_train_batches,
        'val_batches_per_epoch': num_val_batches,
        'train_epoch_sec': train_epoch_sec,
        'val_epoch_sec': val_epoch_sec,
        'ref_embed_overhead_sec': ref_embed_one_time_sec,
        'epoch_total_sec': epoch_total_sec,
        'epochs': epochs,
        'total_sec': total_sec,
        'train_epoch_fmt': fmt_time(train_epoch_sec),
        'val_epoch_fmt': fmt_time(val_epoch_sec),
        'epoch_total_fmt': fmt_time(epoch_total_sec),
        'total_fmt': fmt_time(total_sec),
        'batch_size': config['data']['batch_size'],
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'num_classes': len(train_loader.dataset.classes),
        'class_names': train_loader.dataset.classes,
        'model_backbone': config['model']['backbone'],
        'embedding_dim': config['model']['embedding_dim'],
        'loss_type': loss_type,
        'optimizer': config['training']['optimizer'],
        'learning_rate': config['training']['learning_rate'],
        'scheduler': config['training']['scheduler'],
        'use_amp': use_amp,
        'num_workers': num_workers,
    }
    
    logger.info("=" * 70)
    logger.info("TRAINING TIME ESTIMATE")
    logger.info("=" * 70)
    logger.info(f"Batch size: {estimate['batch_size']}  |  num_workers: {num_workers}")
    logger.info(f"Median train step (fwd+bwd+opt): {avg_train_step:.4f}s  (range: {min(train_times):.4f}-{max(train_times):.4f}s)")
    logger.info(f"Median data load: {avg_data_load:.4f}s  (range: {min(data_load_times):.4f}-{max(data_load_times):.4f}s)")
    logger.info(f"Data overhead factor: {data_overhead_factor:.2f}x  ({'blocking I/O' if num_workers == 0 else 'overlapped'})")
    if num_workers == 0:
        logger.info(f"⚠️  num_workers=0: data loading is BLOCKING. GPU idles during disk I/O.")
        logger.info(f"    Consider enabling cache_in_memory in config or setting num_workers > 0.")
    logger.info(f"Train: {num_train_batches} batches × {avg_train_batch_full:.3f}s = {estimate['train_epoch_fmt']}/epoch")
    logger.info(f"Val:   {num_val_batches} batches × {avg_val_batch:.3f}s + recall = {estimate['val_epoch_fmt']}/epoch")
    logger.info(f"Ref embed (one-time, post-training): {fmt_time(ref_embed_one_time_sec)}")
    logger.info(f"Epoch total: {estimate['epoch_total_fmt']}")
    logger.info(f"Total estimated: {estimate['epochs']} epochs × {estimate['epoch_total_fmt']} + ref_embed = {estimate['total_fmt']}")
    logger.info("=" * 70)
    
    # Clean up benchmark objects
    del bench_optimizer, bench_loss_func
    if bench_scaler:
        del bench_scaler
    
    return estimate


def run_training(args, estimate_only: bool = False, run_dir: str = None):
    """
    Core training function that accepts an args Namespace.
    Can be called from CLI (main()) or from GUI.
    
    Args:
        args: Namespace with config, resume, gpu (optional), checkpoint (optional)
        estimate_only: If True, only benchmark and return time estimate + pre-training analysis
        run_dir: If provided, use this as the run directory for checkpoints/models/logs
        
    Returns:
        If estimate_only: dict with time estimates + pre-training analysis paths
        Otherwise: None
    """
    
    # Load configuration
    config = load_config(args.config)
    
    # Override GPU if specified
    if args.gpu is not None:
        config['hardware']['gpu_ids'] = [args.gpu]
    
    # Setup logging
    logger = setup_logging(
        log_dir=config['paths']['log_dir'],
        log_level=config['logging']['log_level']
    )
    
    logger.info("="*70)
    logger.info("METRIC LEARNING TRAINING - SOTA 2024")
    logger.info("="*70)
    logger.info(f"Configuration file: {args.config}")
    
    # Full config diagnostic dump
    grain_cfg = config.get('grain_detection', {})
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    logger.info("--- CONFIG DUMP ---")
    logger.info(f"  Model: {model_cfg['backbone']}, emb_dim={model_cfg['embedding_dim']}, pretrained={model_cfg['pretrained']}")
    logger.info(f"  Projection: {model_cfg.get('projection_head', {}).get('hidden_dims', [])}, dropout={model_cfg.get('projection_head', {}).get('dropout', 0)}")
    logger.info(f"  Data: image_size={data_cfg.get('image_size')}, resize={data_cfg.get('resize_size')}, batch={data_cfg['batch_size']}, workers={data_cfg.get('num_workers', 4)}")
    logger.info(f"  Grain: mode={grain_cfg.get('dataset_mode')}, crop_size={grain_cfg.get('crop_size')}, use_mask={grain_cfg.get('use_mask')}, masked_pooling={grain_cfg.get('masked_pooling')}")
    logger.info(f"  Training: loss={train_cfg['loss']['type']}, lr={train_cfg['learning_rate']}, epochs={train_cfg['epochs']}, warmup={train_cfg.get('warmup_epochs', 0)}")
    logger.info(f"  Scheduler: {train_cfg.get('scheduler')}, T_max={train_cfg.get('scheduler_params', {}).get('T_max')}")
    logger.info(f"  AMP: {config.get('hardware', {}).get('use_amp', True)}, grad_clip={train_cfg.get('gradient_clip_norm', 5.0)}")
    logger.info(f"  Early stop: patience={train_cfg.get('early_stopping', {}).get('patience')}, min_delta={train_cfg.get('early_stopping', {}).get('min_delta')}")
    logger.info("--- END CONFIG ---")
    
    # Set random seed for reproducibility
    if config['reproducibility']['seed'] is not None:
        torch.manual_seed(config['reproducibility']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['reproducibility']['seed'])
        logger.info(f"Random seed set to: {config['reproducibility']['seed']}")
    
    # Set deterministic mode or enable cudnn benchmark for speed
    if config['reproducibility']['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic mode enabled")
    elif config['reproducibility'].get('benchmark', True):
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark mode enabled (faster convolutions)")
    
    # Create dataloaders
    logger.info("\n" + "="*70)
    logger.info("CREATING DATALOADERS")
    logger.info("="*70)
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model
    logger.info("\n" + "="*70)
    logger.info("CREATING MODEL")
    logger.info("="*70)
    model = create_model(config)
    
    # --- Time estimation (skip if run_dir already exists = already estimated) ---
    if run_dir and not estimate_only:
        logger.info("\n" + "="*70)
        logger.info("SKIPPING BENCHMARK (already estimated in pre-analysis)")
        logger.info("="*70)
    else:
        logger.info("\n" + "="*70)
        logger.info("BENCHMARKING (estimating training time...)")
        logger.info("="*70)
        time_estimate = estimate_training_time(config, train_loader, val_loader, model, logger)
    
    if estimate_only:
        # Run pre-training analysis and create run directory
        from scripts.run_setup import run_pre_training_analysis
        analysis = run_pre_training_analysis(
            config=config,
            config_path=args.config,
            train_dataset=train_loader.dataset,
            val_dataset=val_loader.dataset
        )
        time_estimate['analysis'] = analysis
        time_estimate['run_dir'] = analysis['run_paths']['run_dir']
        
        # Clean up GPU memory before returning
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return time_estimate
    
    # Merge training and hardware configs for trainer
    sota_config = config.get('sota', {})
    trainer_config = {
        **config['training'],
        **config['hardware'],
        'num_classes': config.get('num_classes'),
        'embedding_dim': config['model']['embedding_dim'],
        'use_gpu': config['hardware']['use_gpu'],
        'use_amp': config['hardware']['use_amp'],
        'epochs': config['training']['epochs'],
        'save_every_n_epochs': config['training']['save_every_n_epochs'],
        # SOTA regularizers (2022-2024)
        'nir': sota_config.get('nir', {}),
        'hier': sota_config.get('hier', {}),
        'dada': sota_config.get('dada', {}),
        'self_distillation': sota_config.get('self_distillation', {}),
    }
    
    # Create trainer
    logger.info("\n" + "="*70)
    logger.info("CREATING TRAINER")
    logger.info("="*70)
    trainer = MetricLearningTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        logger=logger,
        checkpoint_dir=run_dir
    )
    
    # Print training configuration
    logger.info("\n" + "="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Loss function: {config['training']['loss']['type']}")
    logger.info(f"Optimizer: {config['training']['optimizer']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Weight decay: {config['training']['weight_decay']}")
    logger.info(f"Scheduler: {config['training']['scheduler']}")
    logger.info(f"Gradient clipping: {config['training']['gradient_clip_norm']}")
    logger.info(f"Mixed precision: {config['hardware']['use_amp']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Warmup epochs: {config['training']['warmup_epochs']}")
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    
    try:
        train_losses, val_losses = trainer.fit(
            epochs=config['training']['epochs'],
            resume_from=args.resume
        )
        
        # Save final model to run directory
        final_dir = Path(run_dir) / 'checkpoints' if run_dir else Path('models')
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            val_loss=trainer.best_val_loss,
            filepath=str(final_dir / 'final_model.pth')
        )
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING SUMMARY")
        logger.info("="*70)
        logger.info(f"Total epochs: {len(train_losses)}")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"Final val loss: {val_losses[-1]:.4f}")
        logger.info("="*70)
        
        # --- Post-training analysis (curves, t-SNE, confusion matrix, inference, report) ---
        if run_dir:
            from scripts.post_training import run_post_training_analysis, compute_and_save_centroids
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Load BEST model for analysis (not the final epoch model)
            best_ckpt = Path(run_dir) / 'checkpoints' / 'best_model.pth'
            if best_ckpt.exists():
                logger.info("Loading BEST model checkpoint for post-training analysis...")
                ckpt = torch.load(str(best_ckpt), map_location=device, weights_only=True)
                model.load_state_dict(ckpt['model_state_dict'])
                logger.info(f"Best model loaded (epoch {ckpt.get('epoch', '?')+1}, val_loss={ckpt.get('val_loss', '?')})")
            model.to(device)

            # Compute and save centroids + reference embeddings (ONE-TIME, training artifact)
            class_names = train_loader.dataset.classes
            centroids = compute_and_save_centroids(
                model, train_loader, device, run_dir, class_names
            )

            # Run analysis using pre-computed centroids (no train_loader needed)
            run_post_training_analysis(model, val_loader, config, run_dir, device, centroids=centroids)

            # Re-save best_model.pth as SELF-CONTAINED inference checkpoint
            # (includes reference embeddings + centroids + class_names)
            logger.info("Re-saving best_model.pth as self-contained inference checkpoint...")
            ref_path = Path(run_dir) / 'reference_embeddings.pt'
            cent_path = Path(run_dir) / 'class_centroids.pt'
            if best_ckpt.exists() and ref_path.exists() and cent_path.exists():
                best_data = torch.load(str(best_ckpt), map_location='cpu', weights_only=False)
                ref_data = torch.load(str(ref_path), map_location='cpu', weights_only=False)
                cent_data = torch.load(str(cent_path), map_location='cpu', weights_only=False)
                best_data['reference_embeddings'] = ref_data['embeddings'].numpy() if torch.is_tensor(ref_data['embeddings']) else ref_data['embeddings']
                best_data['reference_labels'] = ref_data['labels'].numpy() if torch.is_tensor(ref_data['labels']) else ref_data['labels']
                best_data['class_names'] = cent_data['class_names']
                best_data['centroids'] = cent_data['centroids']
                best_data['config'] = config
                torch.save(best_data, str(best_ckpt))
                logger.info(f"Self-contained checkpoint saved: {best_ckpt} "
                            f"({len(best_data['class_names'])} classes, "
                            f"{len(best_data['reference_embeddings'])} ref embeddings)")
        
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            val_loss=trainer.val_losses[-1] if trainer.val_losses else float('inf'),
            filepath=str(Path(run_dir) / 'checkpoints' / 'interrupted_checkpoint.pth') if run_dir else 'models/interrupted_checkpoint.pth',
            save_reference=False
        )
        logger.info("Checkpoint saved to run directory")
    
    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {str(e)}", exc_info=True)
        raise


def main():
    """CLI entry point — parses arguments and calls run_training()"""
    parser = argparse.ArgumentParser(description='Train Metric Learning Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (overrides config)')
    args = parser.parse_args()
    run_training(args)


if __name__ == '__main__':
    main()
