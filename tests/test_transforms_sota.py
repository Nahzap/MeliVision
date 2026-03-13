"""
Tests unitarios para transforms_enhanced — Funcionalidades SOTA (M7).
Cubre: ReinhardStainNormalization.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms_enhanced import ReinhardStainNormalization


# ============================================================
# Tests M7: Stain Normalization
# ============================================================

def test_m7_reinhard_output_shape():
    """M7: Output debe tener mismo shape que input."""
    norm = ReinhardStainNormalization()
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    result = norm(image)
    
    assert result.shape == image.shape, f"Expected {image.shape}, got {result.shape}"


def test_m7_reinhard_output_dtype():
    """M7: Output debe ser uint8."""
    norm = ReinhardStainNormalization()
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    result = norm(image)
    
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"


def test_m7_reinhard_output_range():
    """M7: Output valores deben estar en [0, 255]."""
    norm = ReinhardStainNormalization()
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    
    result = norm(image)
    
    assert result.min() >= 0, f"Min debe ser >= 0, got {result.min()}"
    assert result.max() <= 255, f"Max debe ser <= 255, got {result.max()}"


def test_m7_reinhard_custom_stats():
    """M7: Debe aceptar estadísticas personalizadas."""
    custom_mean = [80.0, 5.0, 5.0]
    custom_std = [20.0, 8.0, 8.0]
    
    norm = ReinhardStainNormalization(target_mean=custom_mean, target_std=custom_std)
    
    assert norm.target_mean == custom_mean
    assert norm.target_std == custom_std
    
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    result = norm(image)
    assert result.shape == image.shape


def test_m7_reinhard_pil_input():
    """M7: Debe aceptar PIL Image (convertir internamente)."""
    from PIL import Image
    
    norm = ReinhardStainNormalization()
    pil_image = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    
    result = norm(pil_image)
    
    assert isinstance(result, np.ndarray), "Output debe ser numpy array"
    assert result.shape == (50, 50, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
