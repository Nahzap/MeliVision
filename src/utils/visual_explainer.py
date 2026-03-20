"""
visual_explainer.py -- Unified Visual Explainability (XAI) Module
================================================================
Provides architecture-aware heatmap generation for all supported backbones:

  CNN-based:   ResNet, ConvNeXt V2           -> Standard Grad-CAM (Selvaraju et al., ICCV 2017)
  ViT-based:   DeiT, plain ViT               -> Activation Grad-CAM with token->grid reshape
  ViT-based:   DINOv2 (has register tokens)  -> Gradient-weighted Self-Attention (Chefer et al.)

Design:
  - Strategy + Factory pattern: each backbone family has its own Explainer subclass.
  - Token topology is analyzed dynamically (CLS, DIST, register tokens).
  - DeiT / vanilla ViTs are routed to CNNExplainer + reshape_transform, which
    uses activation-gradient GradCAM on blocks[-1].norm1 (pre-attention LayerNorm,
    the only sub-layer that yields non-zero GradCAM signal for DeiT due to
    residual-connection gradient dilution in other sub-layers). This suppresses
    the "Attention Sink" artefact (Darcet et al., ICLR 2024) where DeiT dumps
    uninformative attention mass onto corner patches.
  - DINOv2 (with register tokens that absorb attention sinks) keeps using
    the pure gradient-weighted Self-Attention via TransformerExplainer.
  - Bilinear interpolation replaces cv2.INTER_CUBIC to suppress ringing artefacts
    on low-resolution patch grids (14x14 -> 224x224).

References:
  [1] Selvaraju et al., "Grad-CAM", ICCV 2017
  [2] Chefer et al., "Transformer Interpretability Beyond Attention Visualization", CVPR 2021
  [3] Liu et al., "A ConvNet for the 2020s (ConvNeXt)", CVPR 2022
  [4] Touvron et al., "DeiT -- Training data-efficient image transformers", ICML 2021
  [5] Darcet et al., "Vision Transformers Need Registers", ICLR 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Abstract Base
# ═══════════════════════════════════════════════════════════════════════

class BaseVisualExplainer(ABC):
    """
    Abstract base for all visual-explainability strategies.

    Every concrete subclass must implement ``generate_heatmap`` which
    returns a numpy float32 array of shape (H, W) in [0, 1].
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model
        self.device = device
        self._hooks: list = []
        self._cache: dict = {}

    # ── hook helpers ──────────────────────────────────────────────────
    def _attach_hook(self, module: nn.Module, hook_fn):
        handle = module.register_forward_hook(hook_fn)
        self._hooks.append(handle)

    def _detach_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        # DO NOT clear cache here, otherwise finally blocks will destroy
        # the captured tensors before they can be read.

    # ── public interface ──────────────────────────────────────────────
    @abstractmethod
    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        centroid: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Return a heatmap of shape (H_img, W_img) with values in [0, 1].
        """
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
# CNN Explainers
# ═══════════════════════════════════════════════════════════════════════

class CNNExplainer(BaseVisualExplainer):
    """
    Standard Grad-CAM for pure CNNs (ResNet, EfficientNet, etc.).

    Also supports ViT-like backbones when a ``reshape_transform`` is supplied
    to convert (B, N_tokens, D) -> (B, D, H, W) before GradCAM processing.
    This is the preferred XAI method for DeiT and plain ViTs that suffer
    from "Attention Sink" artefacts in raw self-attention maps.

    Uses ``pytorch_grad_cam.GradCAM`` with automatic target-layer detection.
    """

    def __init__(self, model: nn.Module, device: torch.device,
                 target_layer: nn.Module = None,
                 reshape_transform=None):
        super().__init__(model, device)
        self.target_layer = target_layer or self._detect_target_layer()
        self._reshape_transform = reshape_transform

    # ── layer detection ───────────────────────────────────────────────
    def _detect_target_layer(self) -> nn.Module:
        backbone = getattr(self._model, "backbone", self._model)

        # ResNet family
        if hasattr(backbone, "layer4"):
            logger.info("[XAI-CNN] target_layer = backbone.layer4[-1]")
            return backbone.layer4[-1]

        # torchvision ConvNeXt V1 (features sequential)
        if hasattr(backbone, "features"):
            logger.info("[XAI-CNN] target_layer = backbone.features[-1]")
            return backbone.features[-1]

        # Generic last child with parameters
        for child in reversed(list(backbone.children())):
            if list(child.parameters()):
                logger.info(f"[XAI-CNN] target_layer = {type(child).__name__} (generic)")
                return child

        raise RuntimeError("Could not auto-detect a valid 4-D convolutional target layer.")

    # ── heatmap ───────────────────────────────────────────────────────
    def generate_heatmap(self, image_tensor, centroid=None):
        from pytorch_grad_cam import GradCAM
        from src.utils.inference_utils import EmbeddingSimilarityTarget

        targets = [EmbeddingSimilarityTarget(centroid)] if centroid is not None else None
        cam = GradCAM(
            model=self._model,
            target_layers=[self.target_layer],
            reshape_transform=self._reshape_transform,
        )

        try:
            grayscale = cam(input_tensor=image_tensor.to(self.device), targets=targets)
            return grayscale[0]
        finally:
            cam = None  # release references


class ConvNeXtExplainer(BaseVisualExplainer):
    """
    Grad-CAM for hierarchical ConvNeXt V1/V2 from timm.

    ConvNeXt stages sometimes output (B, L, C) tensors internally;
    a ``reshape_transform`` is supplied to coerce them back to (B, C, H, W).
    """

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.target_layer = self._detect_stage_layer()

    def _detect_stage_layer(self) -> nn.Module:
        backbone = getattr(self._model, "backbone", self._model)

        # timm ConvNeXt V2: stages[-1][-1]
        if hasattr(backbone, "stages"):
            last_stage = backbone.stages[-1]
            layer = last_stage[-1] if hasattr(last_stage, "__getitem__") else last_stage
            logger.info(f"[XAI-ConvNeXt] target_layer = stages[-1][-1] ({type(layer).__name__})")
            return layer

        # Fallback to generic CNN detection
        return CNNExplainer(self._model, self.device)._detect_target_layer()

    @staticmethod
    def _reshape_transform(tensor):
        """(B, L, C) → (B, C, H, W) when the layer returns a flat sequence."""
        if tensor.dim() == 3:
            B, L, C = tensor.shape
            h = w = int(L ** 0.5)
            if h * w == L:
                return tensor.transpose(1, 2).reshape(B, C, h, w)
        return tensor

    def generate_heatmap(self, image_tensor, centroid=None):
        from pytorch_grad_cam import GradCAM
        from src.utils.inference_utils import EmbeddingSimilarityTarget

        targets = [EmbeddingSimilarityTarget(centroid)] if centroid is not None else None
        cam = GradCAM(
            model=self._model,
            target_layers=[self.target_layer],
            reshape_transform=self._reshape_transform,
        )

        try:
            grayscale = cam(input_tensor=image_tensor.to(self.device), targets=targets)
            return grayscale[0]
        except ValueError as e:
            # If reshape_transform wasn't needed, retry without it
            if "Invalid grads shape" in str(e):
                logger.warning("[XAI-ConvNeXt] Retrying without reshape_transform")
                cam2 = GradCAM(model=self._model, target_layers=[self.target_layer])
                grayscale = cam2(input_tensor=image_tensor.to(self.device), targets=targets)
                return grayscale[0]
            raise
        finally:
            cam = None


# ═══════════════════════════════════════════════════════════════════════
# Unified Transformer Explainer
# ═══════════════════════════════════════════════════════════════════════

class TransformerExplainer(BaseVisualExplainer):
    """
    Gradient-weighted CLS-to-patch self-attention for **all** ViT variants.

    Handles:
      • Standard ViT  [CLS, P₁…Pₙ]         → patch_start = 1
      • DeiT          [CLS, DIST, P₁…Pₙ]    → patch_start = 2
      • DINOv2        [CLS, P₁…Pₙ]          → patch_start = 1
      • DINOv2-reg    [CLS, R₁…Rₖ, P₁…Pₙ]  → patch_start = 1 + k

    Key architectural fixes over the old code:
      1. Uses ``forward_for_attention()`` / ``forward_features()`` to bypass
         DINOv2Backbone.forward(), which discards the CLS token.
      2. Uses ``F.interpolate(mode='bilinear')`` instead of ``cv2.INTER_CUBIC``
         to eliminate Gibbs-like ringing artefacts on low-res patch grids.
      3. Dynamically parses the token topology so patch_start is always correct.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self._raw_vit = self._unwrap_backbone()
        self._parse_token_topology()

    # ── unwrap ────────────────────────────────────────────────────────
    def _unwrap_backbone(self) -> nn.Module:
        """Get the raw timm/facebookresearch ViT, stripping DINOv2Backbone."""
        if hasattr(self._model, "backbone"):
            bk = self._model.backbone
            if hasattr(bk, "model"):       # DINOv2Backbone wrapper
                return bk.model
            return bk                       # timm ViT / DeiT direct
        return self._model

    # ── topology ──────────────────────────────────────────────────────
    def _parse_token_topology(self):
        self.has_dist_token = (
            hasattr(self._raw_vit, "dist_token")
            and self._raw_vit.dist_token is not None
        )

        self.num_register_tokens = 0
        if hasattr(self._model, "backbone") and hasattr(self._model.backbone, "num_register_tokens"):
            self.num_register_tokens = self._model.backbone.num_register_tokens
        elif hasattr(self._raw_vit, "num_register_tokens"):
            self.num_register_tokens = self._raw_vit.num_register_tokens

        # CLS(1) + [DIST(1)?] + [REG(k)?] = prefix length
        self.patch_start = (
            1
            + (1 if self.has_dist_token else 0)
            + self.num_register_tokens
        )

        logger.info(
            f"[XAI-ViT] Token topology: patch_start={self.patch_start} "
            f"(CLS=1, DIST={int(self.has_dist_token)}, REG={self.num_register_tokens})"
        )

    # ── forward bypass ────────────────────────────────────────────────
    def _forward_bypass(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a standard ViT forward pass that preserves ALL tokens.

        Priority:
          1. DINOv2Backbone.forward_for_attention()  (already handles FB DINOv2)
          2. raw_vit.forward_features()              (timm models)
          3. Manual patch_embed → pos_embed → blocks → norm
        """
        backbone = getattr(self._model, "backbone", None)

        # Path 1: custom method on DINOv2Backbone
        if backbone is not None and hasattr(backbone, "forward_for_attention"):
            return backbone.forward_for_attention(x)

        # Path 2: timm forward_features (DeiT, plain ViT)
        if hasattr(self._raw_vit, "forward_features"):
            out = self._raw_vit.forward_features(x)
            if isinstance(out, dict):
                # Some timm versions return a dict — rare but possible
                raise TypeError(f"forward_features returned dict: {list(out.keys())}")
            return out

        # Path 3: manual reconstruction (bare facebookresearch model)
        v = self._raw_vit
        tokens = v.patch_embed(x)
        if hasattr(v, "pos_embed"):
            tokens = tokens + v.pos_embed
        elif hasattr(v, "_pos_embed"):
            tokens = v._pos_embed(tokens)
        if hasattr(v, "prepare_tokens_with_masks"):
            tokens = v.prepare_tokens_with_masks(x)
        for blk in v.blocks:
            tokens = blk(tokens)
        tokens = v.norm(tokens)
        return tokens

    # ── hook callback ─────────────────────────────────────────────────
    def _qkv_hook(self, module, input_args, output):
        self._cache["qkv"] = output
        if output.requires_grad:
            output.retain_grad()
            self._cache["qkv_ref"] = output
        else:
            self._cache["qkv_ref"] = None

    # ── main entry point ──────────────────────────────────────────────
    def generate_heatmap(self, image_tensor, centroid=None):
        image_tensor = image_tensor.to(self.device)
        self._model.eval()
        self._detach_all()
        self._cache.clear()

        last_block = self._raw_vit.blocks[-1]
        num_heads = last_block.attn.num_heads
        self._attach_hook(last_block.attn.qkv, self._qkv_hook)

        # ── forward + optional backward ──────────────────────────────
        image_tensor = image_tensor.detach().requires_grad_(False)
        try:
            with torch.enable_grad():
                features = self._forward_bypass(image_tensor)

                # Rebuild the embedding to compute metric-learning gradient
                if features.dim() == 3:
                    patch_features = features[:, self.patch_start:]
                    pooled = patch_features.mean(dim=1)
                else:
                    pooled = features

                if hasattr(self._model, "attention") and self._model.attention is not None:
                    pooled = self._model.attention(pooled)

                if hasattr(self._model, "projection"):
                    emb = self._model.projection(pooled)
                    emb = F.normalize(emb, p=2, dim=1)
                else:
                    emb = pooled

                if centroid is not None:
                    c = centroid.to(self.device).detach()
                    sim = F.cosine_similarity(emb, c.unsqueeze(0), dim=1).sum()
                    sim.backward()
        finally:
            self._detach_all()

        # ── compute attention map ────────────────────────────────────
        qkv = self._cache["qkv"].detach()
        qkv_grad = (
            self._cache["qkv_ref"].grad
            if self._cache.get("qkv_ref") is not None
            else None
        )

        B, N, C = qkv.shape
        head_dim = C // (3 * num_heads)

        qkv_r = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv_r.unbind(0)

        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)          # (B, heads, N, N)

        # CLS (idx 0) → patch tokens only
        cls_attn = attn[0, :, 0, self.patch_start:]  # (heads, P)

        logger.info(
            f"[XAI-ViT] N_tokens={N}, patch_start={self.patch_start}, "
            f"n_patches={cls_attn.shape[-1]}, grad_available={qkv_grad is not None}, "
            f"backbone={type(self._raw_vit).__name__}"
        )

        # ── gradient weighting ───────────────────────────────────────
        if qkv_grad is not None:
            g = qkv_grad.abs().mean(dim=(0, 1))                       # (3*dim,)
            hi = g.reshape(3, num_heads, head_dim).mean(dim=(0, 2))    # (heads,)
            hi = hi / (hi.sum() + 1e-8)
            weighted = (cls_attn * hi.unsqueeze(-1)).sum(dim=0)        # (P,)
        else:
            weighted = cls_attn.mean(dim=0)

        # ── spatial reshape + bilinear upsample ──────────────────────
        P = weighted.shape[0]
        h_p = w_p = int(P ** 0.5)

        spatial = weighted.reshape(1, 1, h_p, w_p).float()
        H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]

        upsampled = F.interpolate(
            spatial, size=(H_img, W_img), mode="bilinear", align_corners=False
        ).squeeze()

        arr = upsampled.detach().cpu().numpy()
        vmin, vmax = arr.min(), arr.max()
        return np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Factory / Proxy
# ═══════════════════════════════════════════════════════════════════════

def _is_vit_backbone(model) -> bool:
    """Check if the model uses a ViT-based backbone."""
    if hasattr(model, "backbone"):
        bk = model.backbone
        if hasattr(bk, "blocks") and hasattr(bk, "patch_embed"):
            return True
        if hasattr(bk, "model") and hasattr(bk.model, "blocks"):
            return True
    return False


def _has_register_tokens(model) -> bool:
    """True if the ViT has register tokens (DINOv2-reg)."""
    bk = getattr(model, "backbone", model)
    raw = getattr(bk, "model", bk)
    n_reg = getattr(bk, "num_register_tokens", 0) or getattr(raw, "num_register_tokens", 0)
    return n_reg > 0


def _is_dinov2(model) -> bool:
    """True if the backbone is a DINOv2 model (with or without registers)."""
    bk = getattr(model, "backbone", model)
    cls_name = type(bk).__name__.lower()
    raw = getattr(bk, "model", bk)
    raw_name = type(raw).__name__.lower()
    return "dinov2" in cls_name or "dinov2" in raw_name


def _make_vit_reshape_transform(raw_vit):
    """
    Build a closure that converts (B, N_tokens, D) -> (B, D, H, W) by
    stripping CLS/DIST/register prefix tokens and reshaping patches to a grid.

    This is used for activation-GradCAM on ViTs that lack register tokens
    (DeiT, plain ViT), avoiding the "Attention Sink" artefact.
    """
    has_dist = hasattr(raw_vit, "dist_token") and raw_vit.dist_token is not None
    n_reg = getattr(raw_vit, "num_register_tokens", 0) or 0
    prefix_len = 1 + (1 if has_dist else 0) + n_reg

    def reshape(tensor):
        if tensor.dim() == 3:
            B, N, D = tensor.shape
            n_patches = N - prefix_len
            h = w = int(n_patches ** 0.5)
            if h * w == n_patches:
                patches = tensor[:, prefix_len:, :]  # strip CLS/DIST
                return patches.transpose(1, 2).reshape(B, D, h, w)
        return tensor

    logger.info(
        f"[XAI-Factory] Built vit_reshape_transform: prefix_len={prefix_len} "
        f"(CLS=1, DIST={int(has_dist)}, REG={n_reg})"
    )
    return reshape


def create_explainer(model: nn.Module, device: torch.device) -> BaseVisualExplainer:
    """
    Factory: instantiate the correct explainer for ``model``'s backbone.

    Routing logic:
      1. ConvNeXt         → ConvNeXtExplainer
      2. DINOv2           → TransformerExplainer (attention sinks absorbed by registers)
      3. DeiT / plain ViT → CNNExplainer + reshape_transform (activation GradCAM)
      4. CNN (ResNet, etc) → CNNExplainer

    Returns:
        BaseVisualExplainer subclass ready to call .generate_heatmap()
    """
    bk_name = type(getattr(model, "backbone", model)).__name__.lower()

    # ── ConvNeXt ──────────────────────────────────────────────────────
    if "convnext" in bk_name:
        return ConvNeXtExplainer(model, device)

    # ── ViT family ────────────────────────────────────────────────────
    if _is_vit_backbone(model):
        # DINOv2 uses register tokens → pure Self-Attention is clean
        if _is_dinov2(model):
            logger.info("[XAI-Factory] DINOv2 detected -> TransformerExplainer (QKV attention)")
            return TransformerExplainer(model, device)

        # DeiT / plain ViT -> Activation GradCAM to suppress attention sinks
        raw_vit = getattr(model, "backbone", model)
        if hasattr(raw_vit, "model"):
            raw_vit = raw_vit.model

        target_layer = raw_vit.blocks[-1].norm1
        reshape_fn = _make_vit_reshape_transform(raw_vit)

        logger.info(
            "[XAI-Factory] ViT/DeiT detected -> CNNExplainer + reshape_transform "
            "(target=blocks[-1].norm1, suppressing attention sinks)"
        )
        return CNNExplainer(model, device, target_layer=target_layer,
                            reshape_transform=reshape_fn)

    # ── Default: classic CNN (ResNet, EfficientNet, etc.) ─────────────
    return CNNExplainer(model, device)
