"""
Tests unitarios para trainer_enhanced — Funcionalidades SOTA (M2, M5).
Cubre: Proxy Anchor Loss setup, Recall@K en validación, validation_metric config.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.analogy_net import AnalogyNet


def _make_trainer_config(loss_type='multi_similarity', num_classes=3, validation_metric='recall_at_1'):
    """Crea config mínima para instanciar MetricLearningTrainer."""
    return {
        'model': {
            'backbone': 'resnet18',
            'embedding_dim': 128,
            'pretrained': False,
        },
        'use_gpu': False,
        'use_amp': False,
        'loss': {
            'type': loss_type,
            'params': {
                'alpha': 2.0,
                'beta': 50.0,
                'base': 0.5,
                'margin': 0.1,
                'scale': 32,
            }
        },
        'mining': {'enabled': False},
        'optimizer': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'scheduler_params': {'T_max': 10, 'eta_min': 1e-6},
        'gradient_clip_norm': 5.0,
        'early_stopping': {'patience': 5, 'min_delta': 0.001},
        'warmup_epochs': 0,
        'num_classes': num_classes,
        'embedding_dim': 128,
        'validation_metric': validation_metric,
    }


def _make_dummy_loader(n_samples=24, n_classes=3, batch_size=8):
    """Crea un DataLoader dummy para tests."""
    images = torch.randn(n_samples, 3, 224, 224)
    labels = torch.tensor([i % n_classes for i in range(n_samples)])
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataset.classes = [f'class_{i}' for i in range(n_classes)]
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ============================================================
# Tests M5: Proxy Anchor Loss
# ============================================================

def test_m5_proxy_anchor_loss_setup():
    """M5: Trainer con proxy_anchor loss debe crear ProxyAnchorLoss."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config(loss_type='proxy_anchor', num_classes=3)
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer = MetricLearningTrainer(model, loader, loader, config)

    assert hasattr(trainer, '_proxy_anchor_loss'), "Debe tener atributo _proxy_anchor_loss"
    assert trainer.loss_func is not None, "Loss func no debe ser None"


def test_m5_proxy_anchor_optimizer_param_groups():
    """M5: Con proxy_anchor, optimizer debe tener 2 param groups (model + proxies)."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config(loss_type='proxy_anchor', num_classes=3)
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer = MetricLearningTrainer(model, loader, loader, config)

    param_groups = trainer.optimizer.param_groups
    assert len(param_groups) == 2, f"Esperado 2 param groups, obtenido {len(param_groups)}"

    proxy_lr = param_groups[1]['lr']
    model_lr = param_groups[0]['lr']
    assert proxy_lr > model_lr, f"Proxy lr ({proxy_lr}) debe ser > model lr ({model_lr})"


def test_m5_proxy_anchor_forward_pass():
    """M5: Forward pass con proxy_anchor loss debe producir loss escalar."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config(loss_type='proxy_anchor', num_classes=3)
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer = MetricLearningTrainer(model, loader, loader, config)

    images = torch.randn(6, 3, 224, 224)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    model.train()
    embeddings = model(images)
    loss = trainer.loss_func(embeddings, labels)

    assert loss.dim() == 0, f"Loss debe ser escalar, shape={loss.shape}"
    assert not torch.isnan(loss), "Loss no debe ser NaN"
    assert loss.item() >= 0, "Loss debe ser >= 0"


def test_m5_multi_similarity_loss_no_proxy():
    """M5: Con multi_similarity, no debe haber _proxy_anchor_loss."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config(loss_type='multi_similarity')
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer = MetricLearningTrainer(model, loader, loader, config)

    assert not hasattr(trainer, '_proxy_anchor_loss'), "multi_similarity no debe tener _proxy_anchor_loss"
    assert len(trainer.optimizer.param_groups) == 1, "multi_similarity debe tener 1 param group"


# ============================================================
# Tests M2: Validate con Recall@K
# ============================================================

def test_m2_validate_returns_tuple():
    """M2: validate() debe retornar (loss, metrics_dict)."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config()
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer = MetricLearningTrainer(model, loader, loader, config)
    result = trainer.validate()

    assert isinstance(result, tuple), f"validate() debe retornar tuple, obtenido {type(result)}"
    assert len(result) == 2, f"Debe retornar 2 elementos, obtenido {len(result)}"

    val_loss, metrics = result
    assert isinstance(val_loss, float), f"val_loss debe ser float, obtenido {type(val_loss)}"
    assert isinstance(metrics, dict), f"metrics debe ser dict, obtenido {type(metrics)}"


def test_m2_validate_contains_recall():
    """M2: metrics dict debe contener R@1 y R@5."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config()
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer = MetricLearningTrainer(model, loader, loader, config)
    _, metrics = trainer.validate()

    assert 'R@1' in metrics, "metrics debe contener R@1"
    assert 'R@5' in metrics, "metrics debe contener R@5"
    assert 'val_loss' in metrics, "metrics debe contener val_loss"


def test_m2_validate_recall_range():
    """M2: R@1 y R@5 deben estar en [0, 1]."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config()
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer = MetricLearningTrainer(model, loader, loader, config)
    _, metrics = trainer.validate()

    assert 0.0 <= metrics['R@1'] <= 1.0, f"R@1 = {metrics['R@1']} fuera de rango"
    assert 0.0 <= metrics['R@5'] <= 1.0, f"R@5 = {metrics['R@5']} fuera de rango"


def test_m2_validation_metric_config():
    """M2: validation_metric debe leerse del config."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config_r1 = _make_trainer_config(validation_metric='recall_at_1')
    config_vl = _make_trainer_config(validation_metric='val_loss')

    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer_r1 = MetricLearningTrainer(model, loader, loader, config_r1)
    assert trainer_r1.validation_metric == 'recall_at_1'

    trainer_vl = MetricLearningTrainer(model, loader, loader, config_vl)
    assert trainer_vl.validation_metric == 'val_loss'


def test_m2_best_recall_initialized():
    """M2: best_recall_at_1 debe inicializarse en 0.0."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config()
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer = MetricLearningTrainer(model, loader, loader, config)

    assert trainer.best_recall_at_1 == 0.0, "best_recall_at_1 debe iniciar en 0.0"
    assert trainer.val_metrics_history == [], "val_metrics_history debe iniciar vacío"


# ============================================================
# Tests: Loss types existentes (regresión)
# ============================================================

def test_triplet_loss_setup():
    """Regresión: Triplet loss debe funcionar."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config(loss_type='triplet')
    config['mining'] = {'enabled': True, 'type': 'semihard', 'margin': 0.2}
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    trainer = MetricLearningTrainer(model, loader, loader, config)
    assert trainer.loss_func is not None
    assert trainer.miner is not None


def test_invalid_loss_type():
    """Regresión: Loss type inválido debe lanzar ValueError."""
    from src.training.trainer_enhanced import MetricLearningTrainer

    config = _make_trainer_config(loss_type='invalid_loss')
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    loader = _make_dummy_loader()

    with pytest.raises(ValueError, match="Unknown loss type"):
        MetricLearningTrainer(model, loader, loader, config)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
