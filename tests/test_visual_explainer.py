"""
Tests for the unified visual_explainer module.
Covers: BaseVisualExplainer, CNNExplainer, ConvNeXtExplainer, TransformerExplainer,
        create_explainer factory, and backward-compatible generate_gradcam proxy.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.analogy_net import AnalogyNet


# ============================================================
# Helpers
# ============================================================

def _make_image(batch=1, size=224):
    """Random normalised image tensor."""
    return torch.randn(batch, 3, size, size)


def _make_centroid(dim=128):
    """Random L2-normalised centroid."""
    c = torch.randn(dim)
    return c / c.norm()


# ============================================================
# 1. Factory routing
# ============================================================

class TestFactoryRouting:
    """create_explainer should return the correct subclass."""

    def test_resnet_routes_to_cnn_explainer(self):
        from src.utils.visual_explainer import create_explainer, CNNExplainer
        model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
        explainer = create_explainer(model, torch.device('cpu'))
        assert isinstance(explainer, CNNExplainer)

    def test_resnet50_routes_to_cnn_explainer(self):
        from src.utils.visual_explainer import create_explainer, CNNExplainer
        model = AnalogyNet(backbone='resnet50', embedding_dim=128, pretrained=False)
        explainer = create_explainer(model, torch.device('cpu'))
        assert isinstance(explainer, CNNExplainer)


# ============================================================
# 2. CNN Explainer (ResNet)
# ============================================================

class TestCNNExplainer:
    """Grad-CAM on ResNet-18 / ResNet-50."""

    def test_resnet18_heatmap_shape(self):
        from src.utils.visual_explainer import CNNExplainer
        model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
        model.eval()
        explainer = CNNExplainer(model, torch.device('cpu'))
        hm = explainer.generate_heatmap(_make_image())
        assert hm.ndim == 2, f"Expected 2-D, got {hm.ndim}-D"
        assert hm.shape == (224, 224), f"Expected (224,224), got {hm.shape}"

    def test_resnet18_heatmap_range(self):
        from src.utils.visual_explainer import CNNExplainer
        model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
        model.eval()
        explainer = CNNExplainer(model, torch.device('cpu'))
        hm = explainer.generate_heatmap(_make_image())
        assert hm.min() >= 0.0 and hm.max() <= 1.0, f"Range [{hm.min()}, {hm.max()}]"

    def test_resnet18_with_centroid(self):
        from src.utils.visual_explainer import CNNExplainer
        model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
        model.eval()
        explainer = CNNExplainer(model, torch.device('cpu'))
        hm = explainer.generate_heatmap(_make_image(), centroid=_make_centroid(128))
        assert hm.ndim == 2

    def test_resnet50_heatmap(self):
        from src.utils.visual_explainer import CNNExplainer
        model = AnalogyNet(backbone='resnet50', embedding_dim=128, pretrained=False)
        model.eval()
        explainer = CNNExplainer(model, torch.device('cpu'))
        hm = explainer.generate_heatmap(_make_image())
        assert hm.ndim == 2 and hm.shape[0] > 0


# ============================================================
# 3. Backward compatibility — generate_gradcam proxy
# ============================================================

class TestBackwardCompatibility:
    """generate_gradcam must keep the same API contract."""

    def test_generate_gradcam_resnet18(self):
        from src.utils.inference_utils import generate_gradcam
        model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
        model.eval()
        hm = generate_gradcam(model, _make_image(), torch.device('cpu'))
        assert hm.ndim == 2
        assert hm.min() >= 0.0 and hm.max() <= 1.0

    def test_generate_gradcam_with_centroid(self):
        from src.utils.inference_utils import generate_gradcam
        model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
        model.eval()
        hm = generate_gradcam(model, _make_image(), torch.device('cpu'),
                              centroid=_make_centroid(128))
        assert hm.ndim == 2

    def test_overlay_still_works(self):
        from src.utils.inference_utils import overlay_gradcam_on_image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        hm = np.random.rand(224, 224).astype(np.float32)
        result = overlay_gradcam_on_image(img, hm, alpha=0.5)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.uint8


# ============================================================
# 4. TransformerExplainer — topology parser (mocked)
# ============================================================

class TestTransformerTopology:
    """Token topology parsing must compute correct patch_start."""

    def test_plain_vit_patch_start_is_1(self):
        """ViT without distillation or register tokens → patch_start = 1."""
        from src.utils.visual_explainer import TransformerExplainer

        # Build a minimal mock that looks like a plain ViT
        mock_vit = MagicMock()
        mock_vit.dist_token = None
        mock_vit.blocks = MagicMock()

        mock_backbone = MagicMock()
        mock_backbone.model = mock_vit
        mock_backbone.num_register_tokens = 0
        mock_backbone.blocks = mock_vit.blocks
        mock_backbone.patch_embed = MagicMock()
        # forward_for_attention is present
        mock_backbone.forward_for_attention = MagicMock()

        mock_model = MagicMock()
        mock_model.backbone = mock_backbone

        explainer = TransformerExplainer(mock_model, torch.device('cpu'))
        assert explainer.patch_start == 1
        assert explainer.has_dist_token is False

    def test_deit_patch_start_is_2(self):
        """DeiT with distillation token → patch_start = 2."""
        from src.utils.visual_explainer import TransformerExplainer

        mock_vit = MagicMock()
        mock_vit.dist_token = torch.zeros(1)  # truthy = has DIST
        mock_vit.blocks = MagicMock()

        mock_backbone = MagicMock()
        # DeiT: backbone IS the timm model, no .model wrapper
        mock_backbone.blocks = mock_vit.blocks
        mock_backbone.patch_embed = MagicMock()
        mock_backbone.dist_token = mock_vit.dist_token
        del mock_backbone.model  # no wrapper
        del mock_backbone.num_register_tokens

        mock_model = MagicMock()
        mock_model.backbone = mock_backbone

        explainer = TransformerExplainer(mock_model, torch.device('cpu'))
        assert explainer.patch_start == 2
        assert explainer.has_dist_token is True

    def test_dinov2_reg_patch_start_is_5(self):
        """DINOv2 with 4 register tokens → patch_start = 1 + 4 = 5."""
        from src.utils.visual_explainer import TransformerExplainer

        mock_vit = MagicMock()
        mock_vit.dist_token = None
        mock_vit.blocks = MagicMock()
        mock_vit.num_register_tokens = 4

        mock_backbone = MagicMock()
        mock_backbone.model = mock_vit
        mock_backbone.num_register_tokens = 4
        mock_backbone.blocks = mock_vit.blocks
        mock_backbone.patch_embed = MagicMock()
        mock_backbone.forward_for_attention = MagicMock()

        mock_model = MagicMock()
        mock_model.backbone = mock_backbone

        explainer = TransformerExplainer(mock_model, torch.device('cpu'))
        assert explainer.patch_start == 5
        assert explainer.has_dist_token is False


# ============================================================
# 5. Heatmap discriminability (variance sanity check)
# ============================================================

class TestHeatmapDiscriminability:
    """Two different images should (statistically) produce different heatmaps."""

    def test_different_images_produce_different_heatmaps(self):
        from src.utils.visual_explainer import CNNExplainer
        model = AnalogyNet(backbone='resnet18', embedding_dim=128, pretrained=False)
        model.eval()
        explainer = CNNExplainer(model, torch.device('cpu'))

        hm1 = explainer.generate_heatmap(torch.randn(1, 3, 224, 224))
        hm2 = explainer.generate_heatmap(torch.randn(1, 3, 224, 224))

        # They won't be identical (different random input); correlation < 1.0
        corr = np.corrcoef(hm1.flatten(), hm2.flatten())[0, 1]
        assert corr < 0.999, f"Heatmaps too similar ({corr:.4f}), potential uniform-map bug"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
