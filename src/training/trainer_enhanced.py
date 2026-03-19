"""
Enhanced Metric Learning Trainer with SOTA Techniques (2024)
Supports Multi-Similarity Loss, ArcFace, Triplet Loss, and advanced training strategies

Author: MetricLearning Project
Date: 2026-01-05
Based on: SOTA research investigation (docs/09_SOTA_research_investigation.md)
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm
from pytorch_metric_learning import losses, miners, distances
from pathlib import Path
import logging
import csv
import time
from typing import Dict, Optional, Tuple
import numpy as np

# SOTA losses and regularizers (2022-2024)
from src.losses.nir_regularizer import NIRRegularizer
from src.losses.hier_regularizer import HIERRegularizer
from src.losses.sg_softmax import SGSoftmaxLoss
from src.losses.dada_wrapper import DADAWrapper
from src.training.ema_teacher import EMATeacher, SelfDistillationLoss
from src.training.hard_mining import HardNegativeMiner


class MetricLearningTrainer:
    """
    Enhanced Trainer for Metric Learning with SOTA techniques.
    
    Features:
    - Multi-Similarity Loss (SOTA 2024)
    - ArcFace Loss
    - Triplet Loss with flexible mining
    - Mixed Precision Training (AMP)
    - Gradient Clipping
    - Learning Rate Warmup
    - Cosine Annealing Scheduler
    - Early Stopping
    - Comprehensive Logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict,
        logger: Optional[logging.Logger] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize trainer with model and configuration.
        
        Args:
            model: Neural network model (AnalogyNet)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            logger: Optional logger instance
            checkpoint_dir: Optional run directory for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_mem = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            self.logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"Model VRAM after load: {torch.cuda.memory_allocated(self.device)/1e6:.0f} MB")
        
        # Loss function setup (SOTA 2024)
        self.loss_func = self._setup_loss_function()
        
        # Miner setup (if needed)
        self.miner = self._setup_miner()
        
        # SOTA Regularizers (2022-2024)
        self.nir_reg = self._setup_nir()
        self.hier_reg = self._setup_hier()
        self.dada_wrapper = self._setup_dada()
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        
        # Scheduler setup
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Self-distillation (Zeng et al., 2024)
        self.ema_teacher = None
        self.sd_loss = None
        self._setup_self_distillation()
        
        # Hard Negative Mining (OHEM)
        self.hard_miner = None
        self._setup_hard_mining()
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_recall_at_1 = 0.0
        self.val_metrics_history = []
        self.epochs_without_improvement = 0
        
        # Validation metric for best model selection
        self.validation_metric = config.get('validation_metric', 'recall_at_1')
        
        # Gradient clipping
        self.gradient_clip_norm = config.get('gradient_clip_norm', 5.0)
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping', {}).get('patience', 10)
        self.early_stopping_min_delta = config.get('early_stopping', {}).get('min_delta', 0.001)
        
        # Checkpoint directory (run_dir/checkpoints or default models/)
        if checkpoint_dir:
            self.checkpoint_dir = str(Path(checkpoint_dir) / 'checkpoints')
        else:
            self.checkpoint_dir = 'models'
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self._logged_first_batch = False
        
        self.logger.info(f"Trainer initialized with loss: {self.config['loss']['type']}")
        self.logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger('MetricLearningTrainer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def _setup_loss_function(self) -> nn.Module:
        """
        Setup loss function based on configuration.
        Supports: Multi-Similarity, Triplet, ArcFace
        """
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'multi_similarity')
        params = loss_config.get('params', {})
        
        # Use cosine similarity for all losses (SOTA recommendation)
        distance = distances.CosineSimilarity()
        
        if loss_type == 'multi_similarity':
            # Multi-Similarity Loss (SOTA 2024)
            self.logger.info("Using Multi-Similarity Loss (SOTA 2024)")
            return losses.MultiSimilarityLoss(
                alpha=params.get('alpha', 2.0),
                beta=params.get('beta', 50.0),
                base=params.get('base', 0.5),
                distance=distance
            )
        
        elif loss_type == 'triplet':
            # Triplet Margin Loss
            self.logger.info("Using Triplet Margin Loss")
            return losses.TripletMarginLoss(
                margin=params.get('triplet_margin', 0.2),
                distance=distance
            )
        
        elif loss_type == 'proxy_anchor':
            # Proxy Anchor Loss (Kim et al. CVPR 2020) - SOTA for small datasets
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for Proxy Anchor loss")
            
            self.logger.info(f"Using Proxy Anchor Loss with {num_classes} classes")
            self._proxy_anchor_loss = losses.ProxyAnchorLoss(
                num_classes=num_classes,
                embedding_size=self.config.get('embedding_dim', 128),
                margin=params.get('margin', 0.1),
                alpha=params.get('alpha', 32)
            )
            return self._proxy_anchor_loss
        
        elif loss_type == 'arcface':
            # ArcFace Loss (requires number of classes)
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for ArcFace loss")
            
            self.logger.info(f"Using ArcFace Loss with {num_classes} classes")
            return losses.ArcFaceLoss(
                num_classes=num_classes,
                embedding_size=self.config.get('embedding_dim', 128),
                margin=params.get('margin', 0.5),
                scale=params.get('scale', 64)
            )
        
        elif loss_type == 'sg_softmax':
            # SG-Softmax Loss (Yang et al., AAAI 2023)
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for SG-Softmax loss")
            
            self.logger.info(f"Using SG-Softmax Loss with {num_classes} classes")
            self._sg_softmax_loss = SGSoftmaxLoss(
                embedding_dim=self.config.get('embedding_dim', 128),
                num_classes=num_classes,
                scale=params.get('scale', 30.0),
                margin=params.get('margin', 0.4)
            )
            return self._sg_softmax_loss
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _setup_miner(self) -> Optional[nn.Module]:
        """
        Setup mining strategy if needed.
        Multi-Similarity Loss doesn't require mining.
        """
        loss_type = self.config.get('loss', {}).get('type', 'multi_similarity')
        
        # Multi-Similarity handles mining internally
        if loss_type == 'multi_similarity':
            return None
        
        # Setup miner for Triplet Loss
        mining_config = self.config.get('mining', {})
        if not mining_config.get('enabled', True):
            return None
        
        mining_type = mining_config.get('type', 'semihard')
        margin = mining_config.get('margin', 0.2)
        
        self.logger.info(f"Using {mining_type} mining strategy")
        
        if mining_type == 'semihard':
            return miners.TripletMarginMiner(
                margin=margin,
                type_of_triplets='semihard'
            )
        elif mining_type == 'hard':
            return miners.TripletMarginMiner(
                margin=margin,
                type_of_triplets='hard'
            )
        elif mining_type == 'distance_weighted':
            return miners.DistanceWeightedMiner(
                cutoff=0.5,
                nonzero_loss_cutoff=1.4
            )
        else:
            return None
    
    def _setup_nir(self) -> Optional[NIRRegularizer]:
        """Setup Non-Isotropy Regularization (Roth et al., CVPR 2022)."""
        nir_config = self.config.get('nir', {})
        if not nir_config.get('enabled', False):
            return None
        
        reg = NIRRegularizer(
            lambda_nir=nir_config.get('lambda', 0.5),
            eps=nir_config.get('eps', 1e-4)
        )
        self.logger.info(f"NIR regularization enabled (lambda={nir_config.get('lambda', 0.5)})")
        return reg
    
    def _setup_hier(self) -> Optional[HIERRegularizer]:
        """Setup Hierarchical Regularization (Kim et al., CVPR 2023)."""
        hier_config = self.config.get('hier', {})
        if not hier_config.get('enabled', False):
            return None
        
        num_classes = self.config.get('num_classes')
        if num_classes is None:
            self.logger.warning("HIER requires num_classes, skipping")
            return None
        
        reg = HIERRegularizer(
            embedding_dim=self.config.get('embedding_dim', 128),
            num_classes=num_classes,
            num_levels=hier_config.get('num_levels', 3),
            lambda_hier=hier_config.get('lambda', 0.1),
            temperature=hier_config.get('temperature', 0.1)
        ).to(self.device)
        self.logger.info(f"HIER regularization enabled (lambda={hier_config.get('lambda', 0.1)}, levels={hier_config.get('num_levels', 3)})")
        return reg
    
    def _setup_dada(self) -> Optional[DADAWrapper]:
        """Setup DADA wrapper (Ren et al., AAAI 2024)."""
        dada_config = self.config.get('dada', {})
        if not dada_config.get('enabled', False):
            return None
        
        wrapper = DADAWrapper(
            base_loss=self.loss_func,
            lambda_dada=dada_config.get('lambda', 0.1),
            kernel_bandwidth=dada_config.get('kernel_bandwidth', 1.0)
        )
        self.logger.info(f"DADA wrapper enabled (lambda={dada_config.get('lambda', 0.1)})")
        return wrapper
    
    def _setup_self_distillation(self):
        """Setup Self-Distillation with EMA Teacher (Zeng et al., 2024)."""
        sd_config = self.config.get('self_distillation', {})
        if not sd_config.get('enabled', False):
            return
        
        self.ema_teacher = EMATeacher(
            self.model,
            alpha=sd_config.get('ema_alpha', 0.999)
        )
        self.ema_teacher.to(self.device)
        
        self.sd_loss = SelfDistillationLoss(
            temperature=sd_config.get('temperature', 0.1),
            lambda_sd=sd_config.get('lambda', 0.5)
        )
        self.logger.info(
            f"Self-distillation enabled (alpha={sd_config.get('ema_alpha', 0.999)}, "
            f"temp={sd_config.get('temperature', 0.1)}, lambda={sd_config.get('lambda', 0.5)})"
        )
    
    def _setup_hard_mining(self):
        """Setup Hard Negative Mining (OHEM) if enabled in config."""
        hard_mining_config = self.config.get('training', {}).get('hard_mining', {})
        
        if not hard_mining_config.get('enabled', False):
            self.logger.info("Hard Negative Mining: Disabled")
            return
        
        hard_ratio = hard_mining_config.get('ratio', 0.3)
        min_samples = hard_mining_config.get('min_samples', 32)
        warmup_epochs = hard_mining_config.get('warmup_epochs', 0)
        adaptive = hard_mining_config.get('adaptive', False)
        
        self.hard_miner = HardNegativeMiner(
            hard_ratio=hard_ratio,
            min_samples=min_samples,
            warmup_epochs=warmup_epochs,
            adaptive_ratio=adaptive
        )
        
        self.logger.info(
            f"Hard Negative Mining: Enabled (ratio={hard_ratio}, "
            f"min_samples={min_samples}, warmup={warmup_epochs})"
        )
    
    def _compute_loss_per_sample(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computa pérdida por sample individual (sin reducción) para Hard Negative Mining.
        
        Args:
            embeddings: Tensor (B, D) de embeddings normalizados
            labels: Tensor (B,) de etiquetas
        
        Returns:
            loss_per_sample: Tensor (B,) con pérdida de cada sample
        """
        loss_type = self.config.get('training', {}).get('loss', {}).get('type', 'multi_similarity')
        
        if loss_type == 'arcface':
            # ArcFace: Cross-entropy per sample
            W = self.loss_func.W.weight
            embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            cosine = embeddings_norm @ W.t()
            
            margin = self.loss_func.margin
            scale = self.loss_func.scale
            
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1)
            
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2).clamp(0, 1))
            phi = cosine * torch.cos(torch.tensor(margin, device=cosine.device)) - sine * torch.sin(torch.tensor(margin, device=cosine.device))
            
            output = torch.where(one_hot.bool(), phi, cosine) * scale
            loss_per_sample = torch.nn.functional.cross_entropy(output, labels, reduction='none')
            
        else:
            # Fallback: distancia-based loss per sample
            dist_mat = torch.cdist(embeddings, embeddings, p=2)
            labels_expanded = labels.unsqueeze(1)
            pos_mask = (labels_expanded == labels_expanded.t()) & (torch.eye(len(labels), device=embeddings.device) == 0)
            neg_mask = labels_expanded != labels_expanded.t()
            
            loss_per_sample = torch.zeros(len(embeddings), device=embeddings.device)
            
            for i in range(len(embeddings)):
                if pos_mask[i].sum() > 0 and neg_mask[i].sum() > 0:
                    pos_dist = dist_mat[i][pos_mask[i]].max()
                    neg_dist = dist_mat[i][neg_mask[i]].min()
                    loss_per_sample[i] = torch.clamp(pos_dist - neg_dist + 0.2, min=0.0)
        
        return loss_per_sample
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer (AdamW recommended by SOTA).
        For proxy-based losses, creates separate param groups for model and proxies.
        Supports differential LR: backbone gets lr * backbone_lr_factor."""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Differential learning rate: backbone (pretrained) vs projection head (from scratch)
        backbone_lr_factor = self.config.get('backbone_lr_factor', 1.0)
        if backbone_lr_factor < 1.0 and hasattr(self.model, 'backbone'):
            backbone_params = list(self.model.backbone.parameters())
            backbone_param_ids = {id(p) for p in backbone_params}
            head_params = [p for p in self.model.parameters() if id(p) not in backbone_param_ids]
            param_groups = [
                {'params': backbone_params, 'lr': lr * backbone_lr_factor, 'name': 'backbone'},
                {'params': head_params, 'lr': lr, 'name': 'projection'},
            ]
            self.logger.info(
                f"Differential LR: backbone={lr * backbone_lr_factor:.2e} "
                f"({len(backbone_params)} tensors), "
                f"projection={lr:.2e} ({len(head_params)} tensors)"
            )
        else:
            param_groups = [{'params': self.model.parameters(), 'lr': lr}]
        
        if hasattr(self, '_proxy_anchor_loss'):
            proxy_lr = lr * 100
            param_groups.append({'params': self._proxy_anchor_loss.proxies, 'lr': proxy_lr})
            self.logger.info(f"Proxy Anchor: model lr={lr:.2e}, proxy lr={proxy_lr:.2e}")
        
        if hasattr(self, '_sg_softmax_loss'):
            proxy_lr = lr * 100
            param_groups.append({'params': self._sg_softmax_loss.proxies, 'lr': proxy_lr})
            self.logger.info(f"SG-Softmax: model lr={lr:.2e}, proxy lr={proxy_lr:.2e}")
        
        # HIER proxies
        if self.hier_reg is not None:
            param_groups.append({'params': self.hier_reg.parameters(), 'lr': lr * 10})
            self.logger.info(f"HIER proxies added to optimizer (lr={lr*10:.2e})")
        
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Store initial LR for warmup to reference
        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        params = self.config.get('scheduler_params', {})
        
        if scheduler_type == 'cosine':
            # Cosine Annealing (SOTA recommendation)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=params.get('T_max', 50),
                eta_min=params.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=params.get('mode', 'min'),
                factor=params.get('factor', 0.5),
                patience=params.get('patience', 5),
                min_lr=params.get('min_lr', 1e-7)
            )
        elif scheduler_type == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('learning_rate', 1e-4),
                epochs=self.config.get('epochs', 50),
                steps_per_epoch=len(self.train_loader)
            )
        else:
            return None
    
    def _warmup_lr(self, epoch: int):
        """
        Apply learning rate warmup for first few epochs.
        SOTA best practice to stabilize training.
        Respects per-group base LR (differential LR for backbone vs projection).
        """
        warmup_epochs = self.config.get('warmup_epochs', 5)
        if epoch < warmup_epochs:
            progress = epoch / warmup_epochs  # 0 → 1
            lr_strs = []
            for param_group in self.optimizer.param_groups:
                base_lr = param_group.get('initial_lr', param_group['lr'])
                warmup_lr = 1e-6 + (base_lr - 1e-6) * progress
                param_group['lr'] = warmup_lr
                name = param_group.get('name', 'default')
                lr_strs.append(f"{name}={warmup_lr:.2e}")
            self.logger.info(f"Warmup LR: {', '.join(lr_strs)}")
    
    def _compute_total_loss(self, embeddings, labels, images):
        """
        Compute total loss = base loss + SOTA regularizers.
        
        Integrates:
        - Base metric loss (MS, Triplet, ProxyAnchor, ArcFace, SG-Softmax)
        - NIR regularization (Roth et al., CVPR 2022)
        - HIER regularization (Kim et al., CVPR 2023)
        - DADA alignment (Ren et al., AAAI 2024)
        - Self-distillation (Zeng et al., 2024)
        """
        # Base loss
        if self.dada_wrapper is not None:
            loss = self.dada_wrapper(embeddings, labels)
        elif self.miner is not None:
            hard_pairs = self.miner(embeddings, labels)
            loss = self.loss_func(embeddings, labels, hard_pairs)
        else:
            loss = self.loss_func(embeddings, labels)
        
        # NIR: Non-Isotropy Regularization
        if self.nir_reg is not None:
            nir_loss = self.nir_reg(embeddings, labels)
            loss = loss + nir_loss
        
        # HIER: Hierarchical Regularization
        if self.hier_reg is not None:
            hier_loss = self.hier_reg(embeddings, labels)
            loss = loss + hier_loss
        
        # Self-distillation with EMA Teacher
        if self.ema_teacher is not None and self.sd_loss is not None:
            teacher_embeddings = self.ema_teacher.get_teacher_embeddings(images)
            sd_loss_val = self.sd_loss(embeddings, teacher_embeddings)
            loss = loss + sd_loss_val
        
        return loss
    
    def train_epoch(self) -> float:
        """Train one epoch with mixed precision and SOTA regularizers."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # GPU utilization tracking
        data_time_total = 0.0
        gpu_time_total = 0.0
        t_data_start = time.time()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        
        for batch_idx, batch_data in enumerate(pbar):
            # Support both (images, labels) and (images, labels, masks)
            if len(batch_data) == 3:
                images, labels, masks = batch_data
                masks = masks.to(self.device, non_blocking=True)
            else:
                images, labels = batch_data
                masks = None
            data_time_total += time.time() - t_data_start
            
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # First-batch diagnostic (once per training run)
            if not self._logged_first_batch:
                self._logged_first_batch = True
                n_classes_batch = len(torch.unique(labels))
                self.logger.info(f"  [DIAG] First batch shapes: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
                self.logger.info(f"  [DIAG] Batch: {images.shape[0]} samples, {n_classes_batch} unique classes")
                self.logger.info(f"  [DIAG] Image range: [{images.min():.3f}, {images.max():.3f}] (post-normalize)")
                self.logger.info(f"  [DIAG] Masks: {'provided '+str(tuple(masks.shape)) if masks is not None else 'None (standard pooling)'}")
                self.logger.info(f"  [DIAG] AMP: {self.use_amp}, device: {self.device}")
            
            t_gpu_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with autocast('cuda'):
                    embeddings = self.model(images, mask=masks)
                    
                    # Hard Negative Mining: computar pérdida sin reducción si está habilitado
                    if self.hard_miner is not None:
                        # Computar pérdida por sample (sin reducción)
                        loss_per_sample = self._compute_loss_per_sample(embeddings, labels)
                        
                        # Seleccionar hard examples
                        hard_mask = self.hard_miner.mine_hard_examples(loss_per_sample)
                        
                        # Loss final: promedio de hard examples
                        loss = loss_per_sample[hard_mask].mean()
                    else:
                        # Modo estándar: todas las muestras
                        loss = self._compute_total_loss(embeddings, labels, images)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                embeddings = self.model(images, mask=masks)
                
                # Hard Negative Mining (sin AMP)
                if self.hard_miner is not None:
                    loss_per_sample = self._compute_loss_per_sample(embeddings, labels)
                    hard_mask = self.hard_miner.mine_hard_examples(loss_per_sample)
                    loss = loss_per_sample[hard_mask].mean()
                else:
                    loss = self._compute_total_loss(embeddings, labels, images)
                
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.gradient_clip_norm
                )
                self.optimizer.step()
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            gpu_time_total += time.time() - t_gpu_start
            
            # Update EMA teacher after each step
            if self.ema_teacher is not None:
                self.ema_teacher.update(self.model)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss/num_batches:.4f}',
                'gnorm': f'{grad_norm:.2f}'
            })
            t_data_start = time.time()
        
        avg_loss = total_loss / num_batches
        
        # Log GPU utilization stats
        total_time = data_time_total + gpu_time_total
        if total_time > 0:
            gpu_pct = gpu_time_total / total_time * 100
            self.logger.info(
                f"  GPU: {gpu_pct:.0f}% util ({gpu_time_total:.1f}s compute, {data_time_total:.1f}s data)"
            )
            if self.device.type == 'cuda':
                self.logger.info(
                    f"  VRAM: {torch.cuda.memory_allocated(self.device)/1e6:.0f} MB / "
                    f"{torch.cuda.max_memory_allocated(self.device)/1e6:.0f} MB peak"
                )
        self.logger.info(f"  Train loss: {avg_loss:.6f} (last grad_norm: {grad_norm:.3f})")
        
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict]:
        """
        Validate the model. Computes loss and SOTA metrics.
        
        Returns:
            Tuple of (avg_loss, metrics_dict) with R@1, R@5, mAP@R, NMI, F1-macro
        """
        from src.utils.metrics_utils import compute_all_metrics
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            
            for batch_data in pbar:
                if len(batch_data) == 3:
                    images, labels, masks = batch_data
                    masks = masks.to(self.device, non_blocking=True)
                else:
                    images, labels = batch_data
                    masks = None
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                embeddings = self.model(images, mask=masks)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                if self.miner is not None:
                    hard_pairs = self.miner(embeddings, labels)
                    loss = self.loss_func(embeddings, labels, hard_pairs)
                else:
                    loss = self.loss_func(embeddings, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if num_batches == 0:
            self.logger.warning("Validation set is empty (0 batches). Skipping validation metrics.")
            return 0.0, {'val_loss': 0.0, 'recall_at_1': 0.0, 'recall_at_5': 0.0}
        
        avg_loss = total_loss / num_batches
        
        embeddings_np = np.vstack(all_embeddings)
        labels_np = np.concatenate(all_labels)
        
        # Compute ALL SOTA metrics (CVPR 2016-2024)
        class_names = getattr(self.val_loader.dataset, 'classes', None)
        sota_metrics = compute_all_metrics(
            embeddings_np, 
            labels_np, 
            k_list=[1, 5, 10],
            class_names=class_names
        )
        
        # Embedding diagnostics
        norms = np.linalg.norm(embeddings_np, axis=1)
        self.logger.info(
            f"  Val embeddings: n={embeddings_np.shape[0]}, dim={embeddings_np.shape[1]}, "
            f"norm: {norms.mean():.4f}±{norms.std():.4f}"
        )
        
        metrics = {
            'val_loss': avg_loss,
            'R@1': sota_metrics.get('R@1', 0.0),
            'R@5': sota_metrics.get('R@5', 0.0),
            'R@10': sota_metrics.get('R@10', 0.0),
            'mAP@R': sota_metrics.get('mAP@R', 0.0),
            'NMI': sota_metrics.get('NMI', 0.0),
            'F1_macro': sota_metrics.get('F1_macro', 0.0)
        }
        
        return avg_loss, metrics
    
    def _extract_reference_embeddings(self):
        """Extrae embeddings de referencia del conjunto de entrenamiento"""
        self.logger.info("Extracting reference embeddings from training set...")
        self.model.eval()
        
        embeddings_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.train_loader, desc="Building reference"):
                images, labels = batch_data[0], batch_data[1]
                images = images.to(self.device)
                embeddings = self.model(images)
                embeddings_list.append(embeddings.cpu().numpy())
                labels_list.append(labels.numpy())
        
        reference_embeddings = np.vstack(embeddings_list)
        reference_labels = np.concatenate(labels_list)
        
        # Obtener nombres de clases del dataset
        class_names = self.train_loader.dataset.classes
        
        self.logger.info(f"Reference embeddings: {reference_embeddings.shape}")
        return reference_embeddings, reference_labels, class_names
    
    def save_checkpoint(self, epoch: int, val_loss: float, filepath: str, save_reference: bool = True):
        """
        Save model checkpoint with optional reference embeddings.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            filepath: Path to save checkpoint
            save_reference: If True, includes reference embeddings for inference
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        # Guardar embeddings de referencia para inferencia
        if save_reference:
            ref_embeddings, ref_labels, class_names = self._extract_reference_embeddings()
            checkpoint['reference_embeddings'] = ref_embeddings
            checkpoint['reference_labels'] = ref_labels
            checkpoint['class_names'] = class_names
            self.logger.info("Reference embeddings included in checkpoint")
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        self.logger.info(f"Checkpoint loaded: {filepath} (epoch {self.current_epoch})")
    
    def fit(self, epochs: int, resume_from: Optional[str] = None) -> Tuple[list, list]:
        """
        Train the model for N epochs.
        
        Args:
            epochs: Number of epochs to train
            resume_from: Optional checkpoint path to resume from
            
        Returns:
            Tuple of (train_losses, val_losses)
        """
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        # Initialize metrics CSV in run directory for progressive tracking
        run_dir = str(Path(self.checkpoint_dir).parent)
        self.metrics_csv_path = Path(run_dir) / 'training_metrics.csv'
        with open(self.metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss', 
                'recall_at_1', 'recall_at_5', 'recall_at_10',
                'mAP@R', 'NMI', 'F1_macro',
                'learning_rate', 'is_best', 'best_recall_at_1', 'epoch_time_sec'
            ])
        self.logger.info(f"Metrics CSV: {self.metrics_csv_path}")
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            self.logger.info(f"{'='*70}")
            
            # Learning rate warmup
            self._warmup_lr(epoch)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_metrics_history.append(val_metrics)
            
            # Update scheduler (skip during warmup to avoid conflict)
            warmup_epochs = self.config.get('warmup_epochs', 5)
            if self.scheduler and epoch >= warmup_epochs:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            recall_at_1 = val_metrics.get('R@1', 0.0)
            recall_at_5 = val_metrics.get('R@5', 0.0)
            recall_at_10 = val_metrics.get('R@10', 0.0)
            map_at_r = val_metrics.get('mAP@R', 0.0)
            nmi = val_metrics.get('NMI', 0.0)
            f1_macro = val_metrics.get('F1_macro', 0.0)
            
            # Log SOTA metrics
            self.logger.info(f"\n--- SOTA Metrics (CVPR 2016-2024) ---")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Val Loss:   {val_loss:.4f}")
            self.logger.info(f"  R@1:        {recall_at_1:.4f}")
            self.logger.info(f"  R@5:        {recall_at_5:.4f}")
            self.logger.info(f"  R@10:       {recall_at_10:.4f}")
            self.logger.info(f"  mAP@R:      {map_at_r:.4f}")
            self.logger.info(f"  NMI:        {nmi:.4f}")
            self.logger.info(f"  F1-Macro:   {f1_macro:.4f}")
            self.logger.info(f"  LR:         {current_lr:.2e}")
            
            # Determine if this is the best model
            is_best = False
            if self.validation_metric == 'recall_at_1':
                if recall_at_1 > self.best_recall_at_1 + self.early_stopping_min_delta:
                    self.best_recall_at_1 = recall_at_1
                    is_best = True
            else:
                if val_loss < self.best_val_loss - self.early_stopping_min_delta:
                    is_best = True
            
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss, str(Path(self.checkpoint_dir) / 'best_model.pth'), save_reference=False)
                self.logger.info(f"  >> New best model saved! (R@1={recall_at_1:.4f})")
            else:
                self.epochs_without_improvement += 1
                self.logger.info(f"  -- No improvement for {self.epochs_without_improvement} epochs")
            
            # Save epoch metrics to CSV (progressive — survives interruptions)
            epoch_time = time.time() - epoch_start_time
            with open(self.metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                    f'{recall_at_1:.6f}', f'{recall_at_5:.6f}', f'{recall_at_10:.6f}',
                    f'{map_at_r:.6f}', f'{nmi:.6f}', f'{f1_macro:.6f}',
                    f'{current_lr:.2e}', is_best, f'{self.best_recall_at_1:.6f}',
                    f'{epoch_time:.1f}'
                ])
            
            # Periodic checkpoint (no reference embeddings — only on final save)
            if (epoch + 1) % self.config.get('save_every_n_epochs', 10) == 0:
                self.save_checkpoint(epoch, val_loss, str(Path(self.checkpoint_dir) / f'checkpoint_epoch_{epoch+1}.pth'), save_reference=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                self.logger.info(f"\n[STOP] Early stopping: no improvement for {self.early_stopping_patience} epochs")
                break
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"{'='*70}\n")
        
        return self.train_losses, self.val_losses
