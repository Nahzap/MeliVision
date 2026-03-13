"""
Utilidades para inferencia y extracción de embeddings.
Centraliza operaciones comunes de inferencia.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List, Union, Optional
import logging

logger = logging.getLogger(__name__)


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    return_paths: bool = False,
    show_progress: bool = True
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Extrae embeddings de todas las imágenes en un dataloader.
    
    Args:
        model: Modelo para extraer embeddings
        dataloader: DataLoader con imágenes
        device: Dispositivo (cuda/cpu)
        return_paths: Si True, retorna también rutas de imágenes
        show_progress: Si True, muestra barra de progreso
        
    Returns:
        Si return_paths=False: (embeddings, labels)
        Si return_paths=True: (embeddings, labels, paths)
        
    Example:
        >>> embeddings, labels = extract_embeddings(model, test_loader, device)
        >>> print(embeddings.shape)  # (N, embedding_dim)
    """
    model.eval()
    
    embeddings_list = []
    labels_list = []
    paths_list = [] if return_paths else None
    
    desc = "Extracting embeddings"
    iterator = tqdm(dataloader, desc=desc) if show_progress else dataloader
    
    with torch.no_grad():
        for batch in iterator:
            if return_paths:
                if len(batch) == 3:
                    images, labels, paths = batch
                    paths_list.extend(paths)
                else:
                    raise ValueError("Dataset debe retornar paths cuando return_paths=True")
            else:
                images, labels = batch
            
            images = images.to(device)
            embeddings = model(images)
            
            embeddings_list.append(embeddings.cpu().numpy())
            labels_list.append(labels.numpy())
    
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    
    logger.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    if return_paths:
        return embeddings, labels, paths_list
    else:
        return embeddings, labels


def build_embedding_database(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Construye base de datos de embeddings desde un dataset.
    
    Args:
        model: Modelo para extraer embeddings
        dataset: Dataset con imágenes
        device: Dispositivo (cuda/cpu)
        batch_size: Tamaño de batch
        num_workers: Workers para DataLoader
        
    Returns:
        (embeddings, labels, paths, class_names)
    """
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    embeddings, labels, paths = extract_embeddings(
        model, dataloader, device, return_paths=True, show_progress=True
    )
    
    class_names = dataset.classes if hasattr(dataset, 'classes') else None
    
    return embeddings, labels, paths, class_names


def get_single_embedding(
    model: nn.Module,
    image_path: str,
    transform,
    device: torch.device
) -> np.ndarray:
    """
    Obtiene embedding de una sola imagen.
    
    Args:
        model: Modelo para extraer embedding
        image_path: Ruta a la imagen
        transform: Transformación a aplicar
        device: Dispositivo (cuda/cpu)
        
    Returns:
        Embedding (1, embedding_dim)
    """
    from PIL import Image
    
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(image_tensor)
    
    return embedding.cpu().numpy()


class EmbeddingSimilarityTarget:
    """
    Target para Grad-CAM en Metric Learning.
    Maximiza la similitud coseno entre el embedding de salida y un centroide
    de clase dado, generando gradientes significativos para el heatmap.
    """
    def __init__(self, centroid_vector: torch.Tensor):
        self.centroid = centroid_vector
    
    def __call__(self, model_output):
        if self.centroid.device != model_output.device:
            self.centroid = self.centroid.to(model_output.device)
        similarity = torch.nn.functional.cosine_similarity(
            model_output, self.centroid.unsqueeze(0), dim=1
        )
        return similarity


def _is_vit_backbone(model):
    """Check if the model uses a ViT-based backbone (DINOv2, ViT, etc.)."""
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        return hasattr(backbone, 'blocks') and hasattr(backbone, 'patch_embed')
    return False


def _get_dino_model(model):
    """Get the inner DINOv2 model from AnalogyNet > DINOv2Backbone > model."""
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'model'):
        return model.backbone.model
    return None


def extract_dino_attention(model, image_tensor, device):
    """
    Extract CLS-to-patch attention maps from DINOv2's last transformer block.
    Ref: Caron et al. (DINO, ICCV 2021) — self-attention heads naturally
    segment foreground objects. This is the correct approach for ViTs
    (GradCAM was designed for CNNs and often fails on ViT architectures).

    Args:
        model: AnalogyNet with DINOv2Backbone
        image_tensor: (1, 3, H, W) normalized tensor
        device: torch.device

    Returns:
        attn_map: numpy array (H_img, W_img) in [0, 1], attention heatmap
        head_maps: numpy array (num_heads, h_patches, w_patches) per-head attention
    """
    import cv2

    model.eval()
    image_tensor = image_tensor.to(device)
    dino_model = _get_dino_model(model)
    if dino_model is None:
        raise ValueError("Model does not have DINOv2 backbone")

    last_block = dino_model.blocks[-1]
    attn_module = last_block.attn
    num_heads = attn_module.num_heads

    # Hook to capture QKV from last block's attention
    captured = {}
    def qkv_hook(module, input_args, output):
        captured['qkv'] = output

    hook = attn_module.qkv.register_forward_hook(qkv_hook)

    with torch.no_grad():
        _ = model(image_tensor)

    hook.remove()

    qkv = captured['qkv']  # (B, N_tokens, 3 * dim)
    B, N, _ = qkv.shape
    head_dim = qkv.shape[-1] // (3 * num_heads)

    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)  # each: (B, heads, N, head_dim)

    # Attention weights: softmax(Q @ K^T / sqrt(d_k))
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)  # (B, heads, N, N)

    # CLS token (index 0) attention to patch tokens (index 1:)
    cls_attn = attn[0, :, 0, 1:]  # (heads, N_patches)
    n_patches = cls_attn.shape[-1]
    h_p = w_p = int(n_patches ** 0.5)

    # Per-head attention maps
    head_maps = cls_attn.reshape(num_heads, h_p, w_p).cpu().numpy()

    # Average over heads for the combined map
    avg_attn = cls_attn.mean(dim=0)  # (N_patches,)
    attn_spatial = avg_attn.reshape(h_p, w_p).cpu().numpy()

    # Normalize to [0, 1]
    attn_spatial = (attn_spatial - attn_spatial.min()) / (attn_spatial.max() - attn_spatial.min() + 1e-8)

    # Resize to image resolution
    H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
    attn_map = cv2.resize(attn_spatial, (W_img, H_img), interpolation=cv2.INTER_CUBIC)
    attn_map = np.clip(attn_map, 0, 1)

    return attn_map, head_maps


def extract_gradient_weighted_attention(model, image_tensor, device, centroid):
    """
    Gradient-weighted attention for metric learning: combines DINOv2
    self-attention with gradients of cosine similarity to class centroid.
    This produces task-specific attention maps that show which patches
    contribute most to the predicted embedding's similarity with the centroid.

    Ref: Chefer et al. (Transformer Interpretability, CVPR 2021)

    Args:
        model: AnalogyNet with DINOv2Backbone
        image_tensor: (1, 3, H, W) normalized tensor
        device: torch.device
        centroid: (embed_dim,) class centroid tensor

    Returns:
        attn_map: numpy array (H_img, W_img) in [0, 1]
    """
    import cv2

    model.eval()
    image_tensor = image_tensor.to(device).requires_grad_(False)
    centroid = centroid.to(device)
    dino_model = _get_dino_model(model)
    if dino_model is None:
        raise ValueError("Model does not have DINOv2 backbone")

    last_block = dino_model.blocks[-1]
    attn_module = last_block.attn
    num_heads = attn_module.num_heads

    captured = {}
    def qkv_hook(module, input_args, output):
        captured['qkv'] = output
        output.retain_grad()
        captured['qkv_ref'] = output

    hook = attn_module.qkv.register_forward_hook(qkv_hook)

    # Forward WITH gradients
    embedding = model(image_tensor)
    similarity = torch.nn.functional.cosine_similarity(
        embedding, centroid.unsqueeze(0), dim=1
    )
    similarity.backward()

    hook.remove()

    qkv = captured['qkv'].detach()
    qkv_grad = captured['qkv_ref'].grad

    B, N, _ = qkv.shape
    head_dim = qkv.shape[-1] // (3 * num_heads)

    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)

    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)  # (B, heads, N, N)

    # CLS attention to patches
    cls_attn = attn[0, :, 0, 1:]  # (heads, N_patches)

    # Weight by gradient magnitude (which heads matter for this centroid)
    if qkv_grad is not None:
        grad_importance = qkv_grad.abs().mean(dim=(0, 1))  # (3*dim,)
        # Reshape to per-head importance
        head_importance = grad_importance.reshape(3, num_heads, head_dim).mean(dim=(0, 2))  # (heads,)
        head_importance = head_importance / (head_importance.sum() + 1e-8)
        weighted_attn = (cls_attn * head_importance.unsqueeze(-1)).sum(dim=0)  # (N_patches,)
    else:
        weighted_attn = cls_attn.mean(dim=0)

    n_patches = weighted_attn.shape[-1]
    h_p = w_p = int(n_patches ** 0.5)
    attn_spatial = weighted_attn.reshape(h_p, w_p).cpu().numpy()

    # Normalize to [0, 1]
    attn_spatial = (attn_spatial - attn_spatial.min()) / (attn_spatial.max() - attn_spatial.min() + 1e-8)

    H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
    attn_map = cv2.resize(attn_spatial, (W_img, H_img), interpolation=cv2.INTER_CUBIC)
    attn_map = np.clip(attn_map, 0, 1)

    return attn_map


def _vit_reshape_transform(tensor):
    """
    Reshape transform for ViT-based Grad-CAM.
    ViT blocks output (B, num_tokens, embed_dim) where num_tokens = 1 (CLS) + H*W patches.
    This removes the CLS token and reshapes to (B, embed_dim, H, W) spatial format.
    """
    if tensor.dim() == 3:
        result = tensor[:, 1:, :]  # skip CLS token
        num_patches = result.shape[1]
        h = w = int(num_patches ** 0.5)
        result = result.reshape(result.shape[0], h, w, result.shape[2])
        result = result.permute(0, 3, 1, 2)
        return result
    return tensor


def _detect_target_layer(model):
    """
    Auto-detecta la última capa del backbone para Grad-CAM.
    Soporta ResNet (layer4), ConvNeXt (features[-1]), DINOv2, y genéricos.
    """
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'layer4'):
            return backbone.layer4[-1]
        elif hasattr(backbone, 'features'):
            return backbone.features[-1]
        elif hasattr(backbone, 'blocks'):
            return backbone.blocks[-1]
        else:
            children = list(backbone.children())
            if children:
                return children[-1]
    raise ValueError("No se pudo detectar target_layer. Proporcione uno manualmente.")


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    target_layer=None,
    centroid: torch.Tensor = None
) -> np.ndarray:
    """
    Genera mapa de calor de atención para explicabilidad visual.

    Para ViT/DINOv2: usa mapas de atención nativos del transformer
    (Caron et al., ICCV 2021) con pesos de gradiente para task-specificity
    (Chefer et al., CVPR 2021). Mucho más fiable que GradCAM en ViTs.

    Para CNNs: usa Grad-CAM clásico (Selvaraju et al., ICCV 2017).

    Args:
        model: AnalogyNet
        image_tensor: (1, 3, H, W) normalizado
        device: torch.device
        target_layer: Capa objetivo (solo para CNN fallback)
        centroid: Centroide de clase (embed_dim,) para gradient weighting

    Returns:
        Heatmap numpy array (H, W) en [0, 1]
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    # ViT/DINOv2: use native attention maps (SOTA for transformers)
    if _is_vit_backbone(model):
        try:
            if centroid is not None:
                return extract_gradient_weighted_attention(model, image_tensor, device, centroid)
            else:
                attn_map, _ = extract_dino_attention(model, image_tensor, device)
                return attn_map
        except Exception as e:
            logger.warning(f"DINOv2 attention extraction failed: {e}, falling back to GradCAM")

    # CNN fallback: classic Grad-CAM
    from pytorch_grad_cam import GradCAM

    if target_layer is None:
        target_layer = _detect_target_layer(model)

    reshape_fn = _vit_reshape_transform if _is_vit_backbone(model) else None
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_fn)

    targets = None
    if centroid is not None:
        targets = [EmbeddingSimilarityTarget(centroid)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    return grayscale_cam


def overlay_gradcam_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Superpone heatmap sobre imagen original.
    
    Args:
        image: Imagen RGB (H, W, 3) en [0, 255] o [0, 1]
        heatmap: Heatmap (H, W) en [0, 1]
        alpha: Transparencia del heatmap
        
    Returns:
        Imagen con heatmap superpuesto (H, W, 3) en [0, 255] uint8
    """
    import cv2

    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    blended = (1 - alpha) * image + alpha * heatmap_rgb
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return blended


def get_embedding_with_tta(
    model: nn.Module,
    image_path: str,
    tta_transforms: list,
    device: torch.device
) -> np.ndarray:
    """
    Obtiene embedding con Test-Time Augmentation (TTA).
    Aplica N transforms, obtiene N embeddings, promedia y re-normaliza L2.
    Ref: Zolfaghari & Sajedi 2024 [R5], Goncalves et al. 2022 [R7].
    
    Args:
        model: Modelo para extraer embeddings
        image_path: Ruta a la imagen
        tta_transforms: Lista de transforms a aplicar
        device: Dispositivo
        
    Returns:
        Embedding promediado y L2-normalizado (1, embedding_dim)
    """
    from PIL import Image
    
    model.eval()
    image = Image.open(image_path).convert('RGB')
    
    all_embeddings = []
    with torch.no_grad():
        for transform in tta_transforms:
            image_tensor = transform(image).unsqueeze(0).to(device)
            embedding = model(image_tensor).cpu().numpy()
            all_embeddings.append(embedding)
    
    avg_embedding = np.mean(all_embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding, axis=1, keepdims=True)
    
    logger.info(f"TTA embedding: {len(tta_transforms)} augmentations averaged")
    return avg_embedding


def compute_pairwise_distances(
    embeddings1: np.ndarray,
    embeddings2: Optional[np.ndarray] = None,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Calcula distancias por pares entre embeddings.
    
    Args:
        embeddings1: Primer conjunto de embeddings (N, D)
        embeddings2: Segundo conjunto (M, D). Si None, usa embeddings1
        metric: 'euclidean' o 'cosine'
        
    Returns:
        Matriz de distancias (N, M)
    """
    from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
    
    if embeddings2 is None:
        embeddings2 = embeddings1
    
    if metric == 'euclidean':
        distances = euclidean_distances(embeddings1, embeddings2)
    elif metric == 'cosine':
        distances = cosine_distances(embeddings1, embeddings2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances
