"""
Tests unitarios para inference_utils — Funcionalidades SOTA (M3, M6).
Cubre: generate_gradcam, overlay_gradcam_on_image, get_embedding_with_tta.
"""

import pytest
import torch
import numpy as np
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.analogy_net import AnalogyNet


def _create_temp_image(size=(224, 224)):
    """Crea una imagen temporal para tests."""
    img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    img.save(path)
    return path


# ============================================================
# Tests M3: Grad-CAM
# ============================================================

def test_m3_generate_gradcam_output_shape():
    """M3: Grad-CAM debe generar heatmap (H, W)."""
    from src.utils.inference_utils import generate_gradcam
    
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    model.eval()
    
    image_tensor = torch.randn(1, 3, 224, 224)
    device = torch.device('cpu')
    
    heatmap = generate_gradcam(model, image_tensor, device)
    
    assert heatmap.ndim == 2, f"Heatmap debe ser 2D, got {heatmap.ndim}D"
    assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0, "Heatmap no debe estar vacío"


def test_m3_generate_gradcam_value_range():
    """M3: Valores del heatmap deben estar en [0, 1]."""
    from src.utils.inference_utils import generate_gradcam
    
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    model.eval()
    
    image_tensor = torch.randn(1, 3, 224, 224)
    device = torch.device('cpu')
    
    heatmap = generate_gradcam(model, image_tensor, device)
    
    assert heatmap.min() >= 0.0, f"Min del heatmap debe ser >= 0, got {heatmap.min()}"
    assert heatmap.max() <= 1.0, f"Max del heatmap debe ser <= 1, got {heatmap.max()}"


def test_m3_overlay_gradcam():
    """M3: overlay_gradcam_on_image debe producir imagen RGB."""
    from src.utils.inference_utils import overlay_gradcam_on_image
    
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    heatmap = np.random.rand(224, 224).astype(np.float32)
    
    result = overlay_gradcam_on_image(image, heatmap, alpha=0.5)
    
    assert result.shape == (224, 224, 3), f"Expected (224, 224, 3), got {result.shape}"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"


def test_m3_gradcam_auto_target_layer():
    """M3: Grad-CAM debe auto-detectar target_layer de ResNet."""
    from src.utils.inference_utils import generate_gradcam
    
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    
    image_tensor = torch.randn(1, 3, 224, 224)
    
    heatmap = generate_gradcam(model, image_tensor, torch.device('cpu'))
    assert heatmap is not None, "Debe auto-detectar target_layer"


# ============================================================
# Tests M6: TTA
# ============================================================

def test_m6_get_embedding_with_tta_shape():
    """M6: TTA embedding debe tener shape (1, embedding_dim)."""
    from src.utils.inference_utils import get_embedding_with_tta
    from torchvision import transforms as T
    
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    model.eval()
    
    img_path = _create_temp_image()
    
    try:
        tta_transforms = [
            T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((224, 224)), T.RandomVerticalFlip(p=1.0), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        ]
        
        embedding = get_embedding_with_tta(model, img_path, tta_transforms, torch.device('cpu'))
        
        assert embedding.shape == (1, 128), f"Expected (1, 128), got {embedding.shape}"
    finally:
        os.unlink(img_path)


def test_m6_tta_embedding_l2_normalized():
    """M6: TTA embedding debe estar L2-normalizado."""
    from src.utils.inference_utils import get_embedding_with_tta
    from torchvision import transforms as T
    
    model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
    model.eval()
    
    img_path = _create_temp_image()
    
    try:
        tta_transforms = [
            T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        ]
        
        embedding = get_embedding_with_tta(model, img_path, tta_transforms, torch.device('cpu'))
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-4, f"TTA embedding norm debe ser ~1.0, got {norm}"
    finally:
        os.unlink(img_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
