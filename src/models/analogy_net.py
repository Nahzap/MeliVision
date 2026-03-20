import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    """Channel Attention Module (parte de CBAM). Ref: [R5] Zolfaghari & Sajedi 2024."""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.sigmoid(self.fc(x))
        return x * out


class CBAM1D(nn.Module):
    """
    Channel-Based Attention Module for 1D feature vectors.
    Simplified CBAM for embeddings (no spatial dimension).
    Ref: [R5] Zolfaghari & Sajedi 2024.
    """
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
    
    def forward(self, x):
        return self.channel_attention(x)


class DINOv2Backbone(nn.Module):
    """
    Wrapper for DINOv2 that uses mean pooling of patch tokens
    instead of the default CLS token. This ensures gradients flow
    through spatial tokens for meaningful Grad-CAM heatmaps.
    
    Supports masked mean pooling: when a spatial mask is provided,
    only patch tokens inside the grain (mask > 0) contribute to the
    embedding. Background patches are excluded, forcing the model to
    learn grain-internal features (exine texture, apertures, colpi).
    
    Preserves 'blocks' and 'patch_embed' attributes for automatic
    Grad-CAM target layer detection.
    """
    
    def __init__(self, dino_model):
        super().__init__()
        self.model = dino_model
        # Expose attributes for GradCAM detection
        self.blocks = dino_model.blocks
        self.patch_embed = dino_model.patch_embed
        self.norm = dino_model.norm
        self.patch_size = dino_model.patch_embed.patch_size[0]  # typically 14
        self._logged_first_forward = False
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, 3, H, W) input images
            mask: (B, H_mask, W_mask) optional spatial mask [0/1 float].
                  If provided, only patches overlapping mask>0 are pooled.
                  If None, all patches are pooled (backward-compatible).
        """
        # Run through patch embedding + positional encoding + blocks + norm
        # Using get_intermediate_layers with reshape=False returns (B, N_patches, D)
        out = self.model.get_intermediate_layers(x, n=1, reshape=False)
        patch_tokens = out[0]  # (B, N_patches, D) — CLS already excluded
        
        if not self._logged_first_forward:
            self._logged_first_forward = True
            B, N, D = patch_tokens.shape
            h_p = int(N ** 0.5)
            logger.info(
                f"[DINOv2] First forward: input={tuple(x.shape)}, "
                f"patches={N} ({h_p}x{h_p}), dim={D}, "
                f"pooling={'masked' if mask is not None else 'standard mean'}"
            )
        
        if mask is not None:
            # Convert spatial mask to patch-level mask
            # patch grid size: H_patches = H // patch_size, W_patches = W // patch_size
            B, N, D = patch_tokens.shape
            h_patches = w_patches = int(N ** 0.5)
            
            # Resize mask to patch grid using avg pooling
            # mask: (B, H, W) → (B, 1, H, W) → avg_pool → (B, 1, h_p, w_p) → (B, N)
            m = mask.unsqueeze(1).float()  # (B, 1, H, W)
            m = nn.functional.adaptive_avg_pool2d(m, (h_patches, w_patches))  # (B, 1, h_p, w_p)
            patch_mask = m.reshape(B, -1)  # (B, N)
            
            # Threshold: patch is "inside grain" if >25% of its area is masked
            patch_mask = (patch_mask > 0.25).float()  # (B, N)
            
            # Ensure at least 1 patch is active (avoid division by zero)
            patch_mask = torch.clamp(patch_mask, min=0)
            active_count = patch_mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1)
            
            # Masked mean pooling: only grain patches contribute
            features = (patch_tokens * patch_mask.unsqueeze(-1)).sum(dim=1) / active_count  # (B, D)
        else:
            # Standard mean pooling (backward-compatible)
            features = patch_tokens.mean(dim=1)  # (B, D)
        
        return features
    
    def forward_for_attention(self, x):
        """
        Standard forward pass preserving ALL tokens (CLS + register + patches).
        Used exclusively for attention map extraction — NOT for training.
        
        Unlike self.forward() which uses get_intermediate_layers() + mean pooling
        (optimized for metric learning embeddings), this method preserves the
        CLS token needed for CLS-to-patch attention maps (Caron et al., ICCV 2021).
        
        NOTE: Facebook's DINOv2 forward_features() returns a dict, not a tensor.
        We use prepare_tokens_with_masks() + blocks + norm to get a clean tensor.
        
        Args:
            x: (B, 3, H, W) input images
            
        Returns:
            (B, 1 + num_register_tokens + N_patches, D) tensor with CLS at position 0
        """
        raw = self.model
        # Facebook's DINOv2: forward_features returns dict — bypass it
        if hasattr(raw, 'prepare_tokens_with_masks'):
            tokens = raw.prepare_tokens_with_masks(x)
            for blk in raw.blocks:
                tokens = blk(tokens)
            tokens = raw.norm(tokens)
            return tokens  # (B, 1 + num_reg + N_patches, D)
        # timm ViT (fallback): forward_features returns tensor
        elif hasattr(raw, 'forward_features'):
            out = raw.forward_features(x)
            if isinstance(out, dict):
                # Shouldn't happen, but handle gracefully
                raise TypeError(f"forward_features returned dict with keys: {list(out.keys())}")
            return out
        else:
            return raw(x)
    
    @property
    def num_register_tokens(self):
        """Number of register tokens in the DINOv2 model (0 for standard, 4 for _reg variants)."""
        return getattr(self.model, 'num_register_tokens', 0)


class AnalogyNet(nn.Module):
    """
    Encoder para Metric Learning con soporte multi-backbone.
    
    Transforma imágenes en vectores de características normalizados (embeddings)
    que residen en una hiperesfera unitaria.
    
    Args:
        backbone (str): Tipo de backbone. Opciones:
            'resnet18', 'resnet50', 'convnext_v2_tiny', 'dinov2_vits14', 'dinov2_vitb14'
        embedding_dim (int): Dimensión del vector de salida
        pretrained (bool): Usar pesos pre-entrenados
        projection_head_config (dict, optional): Configuración del projection head.
            Keys: 'hidden_dims' (list[int]), 'use_batch_norm' (bool), 'dropout' (float).
            Si es None, usa default [256] sin BatchNorm (backward compatible).
        use_attention (bool): Si True, inserta CBAM entre backbone y projection head.
        freeze_backbone (bool): Si True, congela el backbone (útil para DINOv2).
    
    Example:
        >>> model = AnalogyNet(backbone='resnet18', embedding_dim=128)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> embeddings = model(x)
        >>> print(embeddings.shape)  # torch.Size([4, 128])
        >>> print(torch.norm(embeddings[0]))  # tensor(1.0000)
    """
    
    BACKBONE_FEATURES = {
        'resnet18': 512,
        'resnet50': 2048,
        'convnext_v2_tiny': 768,
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vits14_reg': 384,
        'dinov2_vitb14_reg': 768,
        'deit_small_patch16_224': 384,
        'efficientnet_b4': 1792,
        'swin_tiny_patch4_window7_224': 768,
    }
    
    def __init__(self, backbone='resnet18', embedding_dim=128, pretrained=True,
                 projection_head_config=None, use_attention=False, freeze_backbone=False,
                 embedding_space='euclidean', hyperbolic_curvature=1.0):
        super(AnalogyNet, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.embedding_space = embedding_space
        
        self.backbone, num_features = self._build_backbone(backbone, pretrained)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info(f"Backbone '{backbone}' frozen (only projection head trainable)")
        
        self.attention = CBAM1D(num_features) if use_attention else None
        
        self.projection = self._build_projection_head(
            num_features, embedding_dim, projection_head_config
        )
        
        self._init_projection_weights()
        
        # Hyperbolic projection (Ermolov et al., CVPR 2022)
        self.hyp_projection = None
        if embedding_space == 'hyperbolic':
            try:
                from src.models.hyperbolic import PoincareBallProjection
                self.hyp_projection = PoincareBallProjection(curvature=hyperbolic_curvature)
                logger.info(f"Hyperbolic embedding space enabled (curvature={hyperbolic_curvature})")
            except ImportError:
                logger.warning("geoopt not installed, falling back to euclidean. pip install geoopt")
                self.embedding_space = 'euclidean'
    
    def _build_backbone(self, backbone, pretrained):
        """
        Construye el backbone según el tipo especificado.
        
        Returns:
            (nn.Module, int): backbone module y número de features de salida
        """
        if backbone not in self.BACKBONE_FEATURES:
            supported = ', '.join(self.BACKBONE_FEATURES.keys())
            raise ValueError(f"Backbone '{backbone}' no soportado. Opciones: {supported}")
        
        num_features = self.BACKBONE_FEATURES[backbone]
        
        if backbone == 'resnet18':
            model = models.resnet18(weights='DEFAULT' if pretrained else None)
            model.fc = nn.Identity()
        elif backbone == 'resnet50':
            model = models.resnet50(weights='DEFAULT' if pretrained else None)
            model.fc = nn.Identity()
        elif backbone == 'convnext_v2_tiny':
            try:
                import timm
                model = timm.create_model(
                    'convnextv2_tiny.fcmae_ft_in22k_in1k',
                    pretrained=pretrained
                )
                model.reset_classifier(0)
            except ImportError:
                logger.warning("timm not available, falling back to torchvision ConvNeXt V1")
                model = models.convnext_tiny(weights='DEFAULT' if pretrained else None)
                model.classifier = nn.Sequential(
                    model.classifier[0],
                    nn.Identity()
                )
            except Exception:
                raise ValueError("ConvNeXt V2 Tiny requires timm >= 0.9.0: pip install timm>=0.9.0")
        elif backbone == 'deit_small_patch16_224':
            # DeiT-S: Vision Transformer supervisado (Touvron et al. ICML 2021)
            try:
                import timm
                model = timm.create_model(
                    'deit_small_patch16_224.fb_in1k',
                    pretrained=pretrained,
                    num_classes=0  # Remove classification head
                )
            except ImportError:
                raise ValueError("DeiT requires timm >= 0.9.0: pip install timm>=0.9.0")
        elif backbone == 'efficientnet_b4':
            # EfficientNet-B4: CNN eficiente (Tan & Le, ICML 2019)
            try:
                import timm
                model = timm.create_model(
                    'efficientnet_b4.ra2_in1k',
                    pretrained=pretrained,
                    num_classes=0
                )
            except ImportError:
                raise ValueError("EfficientNet requires timm >= 0.9.0: pip install timm>=0.9.0")
        elif backbone == 'swin_tiny_patch4_window7_224':
            # Swin Transformer Tiny: ViT jerárquico (Liu et al. ICCV 2021)
            try:
                import timm
                model = timm.create_model(
                    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
                    pretrained=pretrained,
                    num_classes=0
                )
            except ImportError:
                raise ValueError("Swin Transformer requires timm >= 0.9.0: pip install timm>=0.9.0")
        elif backbone.startswith('dinov2_'):
            # Map our backbone names to torch.hub model names
            hub_name_map = {
                'dinov2_vits14': 'dinov2_vits14',
                'dinov2_vitb14': 'dinov2_vitb14',
                'dinov2_vits14_reg': 'dinov2_vits14_reg4',
                'dinov2_vitb14_reg': 'dinov2_vitb14_reg4',
            }
            hub_name = hub_name_map.get(backbone, backbone)
            try:
                raw_model = torch.hub.load('facebookresearch/dinov2', hub_name, pretrained=pretrained)
                # Wrap with DINOv2Backbone: uses mean pooling of patch tokens
                # instead of CLS token, enabling proper Grad-CAM gradients
                model = DINOv2Backbone(raw_model)
            except Exception:
                raise ValueError(
                    f"DINOv2 '{backbone}' (hub: '{hub_name}') requires internet and torch.hub. "
                    "Install with: pip install timm"
                )
        else:
            raise ValueError(f"Backbone '{backbone}' no soportado.")
        
        # Obtener num_features dinámicamente si está disponible
        num_features = self._get_num_features(backbone, model)
        logger.info(f"Backbone '{backbone}' loaded: {num_features} features")
        return model, num_features
    
    def _get_num_features(self, backbone_name, backbone_module):
        """
        Obtiene el número de features del backbone dinámicamente.
        
        Args:
            backbone_name: Nombre del backbone
            backbone_module: Módulo del backbone ya instanciado
            
        Returns:
            int: Número de features de salida
        """
        # Para modelos timm, usar atributo num_features si existe
        if hasattr(backbone_module, 'num_features'):
            return backbone_module.num_features
        
        # Para DINOv2 y otros sin num_features, usar dict estático
        if backbone_name in self.BACKBONE_FEATURES:
            return self.BACKBONE_FEATURES[backbone_name]
        
        # Fallback: intentar inferir del último módulo
        logger.warning(
            f"Could not detect num_features for '{backbone_name}'. "
            f"Using static value from BACKBONE_FEATURES."
        )
        return self.BACKBONE_FEATURES.get(backbone_name, 512)
    
    def _build_projection_head(self, num_features, embedding_dim, config):
        """
        Construye el Projection Head desde configuración.
        
        Args:
            num_features: Dimensión de salida del backbone
            embedding_dim: Dimensión final del embedding
            config: Dict con 'hidden_dims', 'use_batch_norm', 'dropout'.
                    Si None, usa default [256] sin BN (backward compatible).
        
        Returns:
            nn.Sequential: Projection head
        """
        if config is None:
            return nn.Sequential(
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, embedding_dim)
            )
        
        hidden_dims = config.get('hidden_dims', [256])
        use_bn = config.get('use_batch_norm', False)
        dropout = config.get('dropout', 0.0)
        
        layers = []
        in_dim = num_features
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, embedding_dim))
        
        return nn.Sequential(*layers)
    
    def _init_projection_weights(self):
        """Inicializa los pesos del Projection Head con Kaiming Normal."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Batch de imágenes (B, 3, 224, 224)
            mask (torch.Tensor, optional): Spatial mask (B, H, W) for masked pooling.
                Only used with DINOv2 backbone. None = standard pooling.
        
        Returns:
            torch.Tensor: Embeddings normalizados (B, embedding_dim)
        """
        if mask is not None and hasattr(self.backbone, 'patch_size'):
            features = self.backbone(x, mask=mask)
        else:
            features = self.backbone(x)
        
        if features.dim() > 2:
            features = features.mean(dim=list(range(2, features.dim())))
        
        if self.attention is not None:
            features = self.attention(features)
        
        embeddings = self.projection(features)
        
        if self.hyp_projection is not None:
            embeddings = self.hyp_projection(embeddings)
        else:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding_dim(self):
        """Retorna la dimensión del embedding."""
        return self.embedding_dim
