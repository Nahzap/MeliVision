import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


def batched_1d_kmeans(x, k=3, max_iter=10):
    """
    1D K-means for a batch of tensors to approximate Multi-level Otsu thresholds.
    x shape: (B, N)
    """
    B, N = x.shape
    # Initialize centroids using quantiles
    quantiles = torch.linspace(0.1, 0.9, k, device=x.device)
    centroids = torch.quantile(x, quantiles, dim=1).t()  # (B, k)
    
    for _ in range(max_iter):
        dists = torch.abs(x.unsqueeze(-1) - centroids.unsqueeze(1))  # (B, N, k)
        assignments = torch.argmin(dists, dim=-1)
        
        new_centroids = torch.zeros_like(centroids)
        for c in range(k):
            mask = (assignments == c).float()
            sums = (x * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-8)
            new_centroids[:, c] = sums / counts
            
        centroids = new_centroids
        
    centroids, _ = torch.sort(centroids, dim=-1)
    thresholds = (centroids[:, :-1] + centroids[:, 1:]) / 2.0
    return thresholds  # (B, k-1)


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


class SE1D(nn.Module):
    """
    Channel-Based Attention Module for 1D feature vectors.
    Simplified Squeeze-and-Excitation (SE) for embeddings (no spatial dimension).
    Ref: [R5] Zolfaghari & Sajedi 2024.
    """
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
    
    def forward(self, x):
        return self.channel_attention(x)


class DINOv2Backbone(nn.Module):
    """
    Wrapper for DINOv2 that uses multi-spectral pooling of patch tokens
    (mean, max, and std) instead of the default CLS token. This ensures 
    gradients flow through spatial tokens for meaningful Grad-CAM heatmaps
    and preserves spectral richness.
    
    Supports masked multi-spectral pooling: when a spatial mask is provided,
    only patch tokens inside the grain (mask > 0) contribute to the
    embedding. Background patches are excluded.
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
        
        # Override num_features since we concatenate mean, max, and std
        self.num_features = getattr(dino_model, 'embed_dim', 384) * 3
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, 3, H, W) input images
            mask: (B, H_mask, W_mask) optional spatial mask [0/1 float].
        """
        outputs = self.model.get_intermediate_layers(x, n=[3, 11], reshape=False, norm=False)
        tokens_hf = outputs[0]  # Layer 3
        tokens_lf = outputs[1]  # Layer 11
        
        if hasattr(self.model, 'norm'):
            tokens_hf = self.model.norm(tokens_hf)
            tokens_lf = self.model.norm(tokens_lf)
            
        B, N, D = tokens_hf.shape
        
        if not self._logged_first_forward:
            self._logged_first_forward = True
            H_p = W_p = int(N ** 0.5)
            logger.info(
                f"[DINOv2] First forward: input={tuple(x.shape)}, "
                f"patches={N} ({H_p}x{H_p}), dim={D}, "
                f"pooling={'masked multi-spectral (3,11)' if mask is not None else 'multi-spectral (3,11)'}"
            )
            
        # Base patch mask
        if mask is not None:
            h_patches = x.shape[2] // self.patch_size
            w_patches = x.shape[3] // self.patch_size
            m = mask.unsqueeze(1).float()
            m = nn.functional.adaptive_avg_pool2d(m, (h_patches, w_patches))
            patch_mask = m.reshape(B, -1)
            patch_mask = torch.where(patch_mask > 0.01, patch_mask, torch.zeros_like(patch_mask))
        else:
            patch_mask = torch.ones((B, N), device=tokens_hf.device)
            
        # Spatial Token Dropout (DropToken) for Whole-Grain Learning
        if self.training:
            drop_prob = 0.3
            random_mask = torch.empty((B, N), device=tokens_hf.device).bernoulli_(1 - drop_prob)
            patch_mask = patch_mask * random_mask
            
        active_count = patch_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pm = patch_mask.unsqueeze(-1)
        
        # Mean
        mean_lf = (tokens_lf * pm).sum(dim=1) / active_count
        mean_hf = (tokens_hf * pm).sum(dim=1) / active_count
        
        # Max
        # Ignore masked tokens by setting them to -inf
        tokens_lf_masked = torch.where(pm > 0, tokens_lf, torch.tensor(-float('inf'), device=tokens_lf.device))
        tokens_hf_masked = torch.where(pm > 0, tokens_hf, torch.tensor(-float('inf'), device=tokens_hf.device))
        max_lf, _ = tokens_lf_masked.max(dim=1)
        max_hf, _ = tokens_hf_masked.max(dim=1)
        
        # Standard Deviation
        var_lf = (((tokens_lf - mean_lf.unsqueeze(1))**2) * pm).sum(dim=1) / active_count
        var_hf = (((tokens_hf - mean_hf.unsqueeze(1))**2) * pm).sum(dim=1) / active_count
        std_lf = torch.sqrt(var_lf + 1e-6)
        std_hf = torch.sqrt(var_hf + 1e-6)
        
        feat_lf = torch.cat([mean_lf, max_lf, std_lf], dim=1)
        feat_hf = torch.cat([mean_hf, max_hf, std_hf], dim=1)
        
        # Returns a tuple of (LF, HF) embeddings
        return feat_lf, feat_hf
    
    def forward_for_attention(self, x):
        """
        Standard forward pass for attention map extraction.
        """
        raw = self.model
        if hasattr(raw, 'forward_features'):
            out = raw.forward_features(x)
            if isinstance(out, dict):
                cls_token = out.get('x_norm_clstoken', out.get('cls_token', None))
                patch_tokens = out.get('x_norm_patchtokens', out.get('x', None))
                if cls_token is not None and patch_tokens is not None:
                    if cls_token.dim() == 2:
                        cls_token = cls_token.unsqueeze(1)
                    return torch.cat([cls_token, patch_tokens], dim=1)
                else:
                    raise TypeError(f"Unexpected dict keys from forward_features: {list(out.keys())}")
            return out
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
        use_attention (bool): Si True, inserta SE entre backbone y projection head.
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
            # SOTA S-tier: Freeze everything except the last 2 blocks, norm, and projection head
            for name, param in self.backbone.named_parameters():
                if not any(unfreeze_key in name for unfreeze_key in ['blocks.10', 'blocks.11']):
                    if name not in ['norm.weight', 'norm.bias']:
                        param.requires_grad = False
            logger.info(f"Backbone '{self.backbone_name}' partially frozen (last 2 blocks, norm, and projection head trainable)")
        
        self.attention = SE1D(num_features) if use_attention else None
        
        # Creamos dos cabezales paralelos, cada uno proyectando a la mitad de embedding_dim
        emb_dim_half = embedding_dim // 2
        self.projection_lf = self._build_projection_head(
            num_features, emb_dim_half, projection_head_config, embedding_space
        )
        self.projection_hf = self._build_projection_head(
            num_features, emb_dim_half, projection_head_config, embedding_space
        )
        
        self._init_projection_weights()
        
        # Hyperbolic projection (Ermolov et al., CVPR 2022)
        self.hyp_projection = None
        if embedding_space == 'hyperbolic':
            from src.models.hyperbolic import PoincareBallProjection
            self.hyp_projection = PoincareBallProjection(curvature=hyperbolic_curvature)
            logger.info(f"Hyperbolic embedding space enabled (curvature={hyperbolic_curvature})")
    
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
    
    def _build_projection_head(self, num_features, embedding_dim, config, embedding_space='euclidean'):
        """
        Construye el cabezal de proyección lineal.
        
        Args:
            num_features: Dimensión de salida del backbone
            embedding_dim: Dimensión final del embedding
            config: Dict con 'hidden_dims', 'use_batch_norm', 'dropout'.
                    Si None, usa default [256] sin BN (backward compatible).
        
        Returns:
            nn.Sequential: Projection head
        """
        use_bias = (embedding_space != 'hyperbolic')
        
        if config is None:
            return nn.Sequential(
                nn.Linear(num_features, 256, bias=use_bias),
                nn.ReLU(inplace=True),
                nn.Linear(256, embedding_dim, bias=use_bias)
            )
        
        hidden_dims = config.get('hidden_dims', [256])
        use_bn = config.get('use_batch_norm', False)
        dropout = config.get('dropout', 0.0)
        
        layers = []
        in_dim = num_features
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim, bias=use_bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, embedding_dim, bias=use_bias))
        
        if config is not None:
            if config.get('normalize_embeddings_ln', False):
                layers.append(nn.LayerNorm(embedding_dim))
            if config.get('use_hardtanh', False):
                min_val = config.get('hardtanh_min', -2.0)
                max_val = config.get('hardtanh_max', 2.0)
                layers.append(nn.Hardtanh(min_val=min_val, max_val=max_val))
        
        return nn.Sequential(*layers)
    
    def _init_projection_weights(self):
        """Inicializa los pesos del Projection Head con Kaiming Normal."""
        emb_dim_half = self.embedding_dim // 2
        for proj in [self.projection_lf, self.projection_hf]:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    if m.out_features == emb_dim_half:
                        nn.init.orthogonal_(m.weight)
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # Inicia pesos de atencion si existen
        if self.attention is not None:
            for m in self.attention.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def train(self, mode=True):
        """Override train to maintain frozen backbone in eval mode."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
    
    def forward(self, x, mask=None, return_prenorm=False):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Batch de imágenes (B, 3, 224, 224)
            mask (torch.Tensor, optional): Spatial mask (B, H, W) for masked pooling.
                Only used with DINOv2 backbone. None = standard pooling.
            return_prenorm (bool): Si True, devuelve (embeddings, prenorm_embeddings)
        
        Returns:
            torch.Tensor o tuple: Embeddings normalizados (B, embedding_dim) o tupla si return_prenorm=True
        """
        if mask is not None:
            # "Zero activation" removido: dejar que DINOv2 vea la imagen original completa.
            # Los píxeles en cero interactúan mal con los Positional Embeddings y la Auto-Atención.
            # El filtrado ocurrirá únicamente a la salida (Masked Mean Pooling).
            # x = x * mask.unsqueeze(1).float()
            
            if hasattr(self.backbone, 'patch_size'):
                features = self.backbone(x, mask=mask)
            else:
                logger.warning("Mask provided but backbone does not support spatial masking. Ignoring mask.")
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        
        # Manejar tupla de salida (LF, HF) desde DINOv2
        if isinstance(features, tuple) and len(features) == 2:
            feat_lf, feat_hf = features
        else:
            if features.dim() > 2:
                features = features.mean(dim=list(range(2, features.dim())))
            feat_lf, feat_hf = features, features
        
        if self.attention is not None:
            feat_lf = self.attention(feat_lf)
            feat_hf = self.attention(feat_hf)
        
        prenorm_lf = self.projection_lf(feat_lf)
        prenorm_hf = self.projection_hf(feat_hf)
        
        # L2-Cat-L2 Topology:
        # 1. Normalizar las representaciones a la hiperesfera unitaria de forma independiente
        norm_lf = F.normalize(prenorm_lf, p=2, dim=1)
        norm_hf = F.normalize(prenorm_hf, p=2, dim=1)
        
        # 2. Concatenar. El vector resultante tendrá una norma euclidiana de sqrt(1^2 + 1^2) = sqrt(2)
        prenorm_embeddings = torch.cat([norm_lf, norm_hf], dim=1)
        
        if self.hyp_projection is not None:
            if self.embedding_space == 'hyperbolic':
                embeddings = self.hyp_projection(prenorm_embeddings)
            else:
                import math
                embeddings = prenorm_embeddings / math.sqrt(2.0)
        else:
            import math
            embeddings = prenorm_embeddings / math.sqrt(2.0)
        
        if return_prenorm:
            return embeddings, prenorm_embeddings
        return embeddings
    
    def get_embedding_dim(self):
        """Retorna la dimensión del embedding."""
        return self.embedding_dim
