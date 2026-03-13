"""
Tests unitarios para AnalogyNet — Modelo de Metric Learning.
Cubre: output shape, L2 norm, backbones, projection head configurable (M1).
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.analogy_net import AnalogyNet


# ============================================================
# Tests originales (backward compatibility)
# ============================================================

def test_output_shape():
    """Verifica que el output tenga la forma correcta (sin config = backward compat)."""
    model = AnalogyNet(embedding_dim=128)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"


def test_l2_normalization():
    """Verifica que todos los embeddings tengan norma 1."""
    model = AnalogyNet(embedding_dim=128)
    x = torch.randn(4, 3, 224, 224)
    embeddings = model(x)
    
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
        f"Embeddings no están normalizados: {norms}"


def test_deterministic_output():
    """Verifica que el modelo sea determinista en eval mode."""
    model = AnalogyNet(embedding_dim=128)
    model.eval()
    
    x = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    
    assert torch.allclose(out1, out2), "El modelo no es determinista"


def test_gradient_flow():
    """Verifica que los gradientes fluyan correctamente."""
    model = AnalogyNet(embedding_dim=128)
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    
    embeddings = model(x)
    loss = embeddings.sum()
    loss.backward()
    
    assert x.grad is not None, "Los gradientes no fluyen al input"
    assert not torch.isnan(x.grad).any(), "Gradientes contienen NaN"


def test_different_backbones():
    """Verifica que diferentes backbones funcionen."""
    for backbone in ['resnet18', 'resnet50']:
        model = AnalogyNet(backbone=backbone, embedding_dim=128)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 128)


# ============================================================
# Tests M1: Projection Head Configurable
# ============================================================

def test_m1_default_projection_no_config():
    """M1: Sin config, debe usar projection [256] sin BN (backward compat)."""
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    
    layers = list(model.projection.children())
    assert isinstance(layers[0], nn.Linear), "Primera capa debe ser Linear"
    assert layers[0].out_features == 256, f"Esperado 256, obtenido {layers[0].out_features}"
    assert isinstance(layers[1], nn.ReLU), "Segunda capa debe ser ReLU"
    assert isinstance(layers[2], nn.Linear), "Tercera capa debe ser Linear"
    assert layers[2].out_features == 128, f"Output debe ser 128"
    
    has_bn = any(isinstance(l, nn.BatchNorm1d) for l in layers)
    assert not has_bn, "Sin config no debe haber BatchNorm"


def test_m1_projection_with_config(projection_head_config):
    """M1: Con config [512, 256] + BN, debe construir projection correctamente."""
    model = AnalogyNet(
        backbone='resnet18',
        embedding_dim=128,
        pretrained=False,
        projection_head_config=projection_head_config
    )
    
    layers = list(model.projection.children())
    
    assert isinstance(layers[0], nn.Linear), "Capa 0 debe ser Linear"
    assert layers[0].in_features == 512, "Input de resnet18 = 512"
    assert layers[0].out_features == 512, f"Primera hidden dim = 512"
    
    assert isinstance(layers[1], nn.BatchNorm1d), "Capa 1 debe ser BatchNorm1d"
    assert layers[1].num_features == 512
    
    assert isinstance(layers[2], nn.ReLU), "Capa 2 debe ser ReLU"
    
    assert isinstance(layers[3], nn.Linear), "Capa 3 debe ser Linear"
    assert layers[3].out_features == 256
    
    assert isinstance(layers[4], nn.BatchNorm1d), "Capa 4 debe ser BatchNorm1d"
    assert isinstance(layers[5], nn.ReLU), "Capa 5 debe ser ReLU"
    
    assert isinstance(layers[6], nn.Linear), "Última capa debe ser Linear"
    assert layers[6].out_features == 128, "Output final = embedding_dim"


def test_m1_projection_with_dropout():
    """M1: Con dropout > 0, debe incluir Dropout layers."""
    config = {
        'hidden_dims': [256],
        'use_batch_norm': False,
        'dropout': 0.3
    }
    model = AnalogyNet(
        backbone='resnet18',
        embedding_dim=128,
        pretrained=False,
        projection_head_config=config
    )
    
    has_dropout = any(isinstance(l, nn.Dropout) for l in model.projection.children())
    assert has_dropout, "Con dropout=0.3 debe haber Dropout layer"


def test_m1_projection_output_shape_with_config(projection_head_config, small_batch):
    """M1: Output shape debe ser correcto con projection head configurable."""
    model = AnalogyNet(
        backbone='resnet18',
        embedding_dim=128,
        pretrained=False,
        projection_head_config=projection_head_config
    )
    model.eval()
    
    with torch.no_grad():
        out = model(small_batch)
    
    assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"


def test_m1_l2_norm_with_config(projection_head_config, small_batch):
    """M1: Embeddings deben estar L2-normalizados con projection head configurable."""
    model = AnalogyNet(
        backbone='resnet18',
        embedding_dim=128,
        pretrained=False,
        projection_head_config=projection_head_config
    )
    model.eval()
    
    with torch.no_grad():
        embeddings = model(small_batch)
    
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), \
        f"Embeddings con config no están normalizados: {norms}"


def test_m1_resnet50_with_config(projection_head_config):
    """M1: ResNet50 + config debe funcionar (input features = 2048)."""
    model = AnalogyNet(
        backbone='resnet50',
        embedding_dim=128,
        pretrained=False,
        projection_head_config=projection_head_config
    )
    
    first_linear = list(model.projection.children())[0]
    assert first_linear.in_features == 2048, "ResNet50 output = 2048"
    
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 128)


def test_m1_gradient_flow_with_config(projection_head_config):
    """M1: Gradientes deben fluir con BN en projection head."""
    model = AnalogyNet(
        backbone='resnet18',
        embedding_dim=128,
        pretrained=False,
        projection_head_config=projection_head_config
    )
    
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    embeddings = model(x)
    loss = embeddings.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradientes no fluyen con BN config"
    assert not torch.isnan(x.grad).any(), "Gradientes con NaN"
    
    for name, param in model.projection.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Sin gradiente en {name}"


def test_m1_invalid_backbone():
    """M1: Backbone inválido debe lanzar ValueError."""
    with pytest.raises(ValueError, match="no soportado"):
        AnalogyNet(backbone='vgg16', embedding_dim=128)


# ============================================================
# Tests M4: Multi-backbone (ConvNeXt V2, DINOv2)
# ============================================================

def test_m4_convnext_v2_tiny():
    """M4: ConvNeXt V2 Tiny debe producir embeddings correctos."""
    try:
        model = AnalogyNet(backbone='convnext_v2_tiny', embedding_dim=128, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"
        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5), "L2 norm debe ser 1.0"
    except ValueError:
        pytest.skip("ConvNeXt V2 Tiny not available in this torchvision version")


def test_m4_backbone_features_dict():
    """M4: BACKBONE_FEATURES debe contener todos los backbones soportados."""
    expected = ['resnet18', 'resnet50', 'convnext_v2_tiny', 'dinov2_vits14', 'dinov2_vitb14']
    for name in expected:
        assert name in AnalogyNet.BACKBONE_FEATURES, f"'{name}' missing in BACKBONE_FEATURES"


def test_m4_freeze_backbone():
    """M4: Con freeze_backbone=True, params del backbone no deben tener requires_grad."""
    model = AnalogyNet(
        backbone='resnet18', embedding_dim=128, pretrained=False,
        freeze_backbone=True
    )
    
    for param in model.backbone.parameters():
        assert not param.requires_grad, "Backbone params deben estar frozen"
    
    for param in model.projection.parameters():
        assert param.requires_grad, "Projection params deben ser trainable"


def test_m4_freeze_backbone_forward():
    """M4: Forward debe funcionar con backbone frozen."""
    model = AnalogyNet(
        backbone='resnet18', embedding_dim=128, pretrained=False,
        freeze_backbone=True
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 128)


# ============================================================
# Tests M10: CBAM Attention
# ============================================================

def test_m10_cbam_creation():
    """M10: Modelo con use_attention=True debe tener CBAM module."""
    model = AnalogyNet(
        backbone='resnet18', embedding_dim=128, pretrained=False,
        use_attention=True
    )
    assert model.attention is not None, "attention module debe existir"


def test_m10_cbam_disabled():
    """M10: Modelo con use_attention=False no debe tener CBAM."""
    model = AnalogyNet(
        backbone='resnet18', embedding_dim=128, pretrained=False,
        use_attention=False
    )
    assert model.attention is None, "attention module debe ser None"


def test_m10_cbam_output_shape():
    """M10: Output shape debe ser correcto con CBAM."""
    model = AnalogyNet(
        backbone='resnet18', embedding_dim=128, pretrained=False,
        use_attention=True
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 128), f"Expected (2, 128), got {out.shape}"


def test_m10_cbam_l2_norm():
    """M10: Embeddings con CBAM deben estar L2-normalizados."""
    model = AnalogyNet(
        backbone='resnet18', embedding_dim=128, pretrained=False,
        use_attention=True
    )
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), f"Norms: {norms}"


def test_m10_cbam_gradient_flow():
    """M10: Gradientes deben fluir a través de CBAM."""
    model = AnalogyNet(
        backbone='resnet18', embedding_dim=128, pretrained=False,
        use_attention=True
    )
    x = torch.randn(2, 3, 224, 224, requires_grad=True)
    embeddings = model(x)
    loss = embeddings.sum()
    loss.backward()
    
    assert x.grad is not None, "Gradientes no fluyen con CBAM"
    assert not torch.isnan(x.grad).any(), "Gradientes con NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
