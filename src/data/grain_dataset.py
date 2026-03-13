"""
SegmentedGrainDataset — Dataset que lee anotaciones .seg y genera crops on-the-fly.

A diferencia de MielDataset (1 sample = 1 imagen completa),
aquí 1 sample = 1 grano individual recortado de la imagen original.

No genera imágenes adicionales en disco. Los crops se computan
en runtime a partir de la imagen original + las coordenadas del .seg.

Uso:
    dataset = SegmentedGrainDataset("data/processed/train", transform=val_transform)
    crop_tensor, class_idx = dataset[0]
"""

import collections
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.grain_detection.seg_format import SegFileReader, has_seg_file

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class SegmentedGrainDataset(Dataset):
    """
    Dataset que genera crops de granos individuales on-the-fly.

    Escanea un directorio ImageFolder buscando pares imagen/.seg.
    Cada grano anotado en un .seg se convierte en un sample independiente.

    Args:
        root: Directorio raíz (estructura ImageFolder: root/class/img.ext)
        transform: Transforms de torchvision a aplicar sobre el crop PIL
        crop_padding: Padding proporcional alrededor del bbox del grano
        crop_size: Tamaño final del crop cuadrado (para AnalogyNet)
        use_mask: Si True, aplica la máscara del grano (fondo fuera del contorno)
        mask_bg_mode: Modo de fondo fuera del contorno cuando use_mask=True.
            'random' = color aleatorio/ruido (training, evita shortcut learning)
            'gray'   = gris neutro 128 (inference/val)
            'black'  = negro puro (legacy, NO recomendado — causa shortcut learning)
        min_saliency: Umbral mínimo de saliency para incluir un grano
        return_metadata: Si True, retorna (tensor, class_idx, metadata_dict)
        return_mask: Si True, retorna máscara binaria del grano para masked pooling
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        crop_padding: float = 0.2,
        crop_size: int = 256,
        use_mask: bool = False,
        mask_bg_mode: str = 'random',
        min_saliency: float = 0.0,
        return_metadata: bool = False,
        return_mask: bool = False,
        annotation_root: Optional[str] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.crop_padding = crop_padding
        self.crop_size = crop_size
        self.use_mask = use_mask
        self.mask_bg_mode = mask_bg_mode
        self.min_saliency = min_saliency
        self.return_metadata = return_metadata
        self.return_mask = return_mask
        
        # Annotation root: directorio centralizado para .seg files
        # Si es None, busca .seg junto a las imágenes (legacy behavior)
        if annotation_root is None:
            self.annotation_root = None
            logger.info("[SegmentedGrainDataset] Usando anotaciones junto a imágenes (legacy mode)")
        else:
            self.annotation_root = Path(annotation_root)
            if not self.annotation_root.exists():
                raise FileNotFoundError(f"Annotation root no existe: {self.annotation_root}")
            logger.info(f"[SegmentedGrainDataset] Usando anotaciones centralizadas en: {self.annotation_root}")

        # _grain_samples: cached grain data — avoids re-reading .seg in __getitem__
        # Each entry: (image_path, class_idx, bbox, saliency, contour_or_None)
        self._grain_samples: List[Tuple[str, int, Tuple[int,int,int,int], float, Optional[np.ndarray]]] = []
        # samples: 2-tuples (image_path, class_idx) — compatible con MielDataset / run_setup
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.targets: List[int] = []

        # LRU cache for images: avoids re-reading the same image for multiple grains
        self._image_cache: collections.OrderedDict = collections.OrderedDict()
        self._image_cache_size = 64

        self._logged_first_getitem = False

        self._build_index()

        logger.info(
            f"[SegmentedGrainDataset] {len(self.samples)} granos de "
            f"{len(self.classes)} clases en {self.root}"
        )
        logger.info(
            f"  config: crop_size={self.crop_size}, padding={self.crop_padding}, "
            f"use_mask={self.use_mask}, mask_bg={self.mask_bg_mode}, return_mask={self.return_mask}"
        )
        stats = self.get_stats()
        for cls_name, count in stats.get('class_distribution', {}).items():
            logger.info(f"  {cls_name}: {count} granos")

    def _build_index(self):
        """
        Escanea el directorio buscando pares imagen/.seg.
        Cada grano en un .seg se convierte en un sample independiente.
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self.root}")

        class_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()]
        )
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        skipped_no_seg = 0
        skipped_saliency = 0

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Buscar .seg en annotation_root si está configurado, sino junto a la imagen
                if self.annotation_root is not None:
                    seg_path = self.annotation_root / class_dir.name / f"{img_path.stem}.seg"
                else:
                    seg_path = img_path.with_suffix(".seg")
                
                if not seg_path.exists():
                    skipped_no_seg += 1
                    continue

                try:
                    grains = SegFileReader.read_grains(str(seg_path))
                except Exception as e:
                    logger.warning(
                        f"[SegGrainDataset] Error leyendo {seg_path.name}: {e}"
                    )
                    continue

                img_str = str(img_path)
                for grain in grains:
                    if grain["saliency"] < self.min_saliency:
                        skipped_saliency += 1
                        continue

                    # Cache grain data directly — no .seg re-reads at runtime
                    contour = grain.get("contour")  # numpy array or None
                    self._grain_samples.append((
                        img_str,
                        class_idx,
                        tuple(grain["bbox"]),
                        grain["saliency"],
                        contour,
                    ))
                    self.samples.append((img_str, class_idx))
                    self.targets.append(class_idx)

        if skipped_no_seg > 0:
            logger.info(
                f"[SegGrainDataset] {skipped_no_seg} imágenes sin .seg (omitidas)"
            )
        if skipped_saliency > 0:
            logger.debug(
                f"[SegGrainDataset] {skipped_saliency} granos bajo min_saliency"
            )

    def __len__(self) -> int:
        return len(self._grain_samples)

    def _load_image(self, image_path: str) -> np.ndarray:
        """Loads image with simple LRU cache (pickle-safe for DataLoader workers)."""
        if image_path in self._image_cache:
            self._image_cache.move_to_end(image_path)
            return self._image_cache[image_path]
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"No se pudo leer: {image_path}")
        self._image_cache[image_path] = image
        if len(self._image_cache) > self._image_cache_size:
            self._image_cache.popitem(last=False)
        return image

    def __getitem__(self, idx: int):
        image_path, class_idx, bbox, saliency, contour = self._grain_samples[idx]

        # Cargar imagen (LRU cached — misma imagen para múltiples granos)
        image = self._load_image(image_path)

        # Construir grain dict desde datos cacheados (sin leer .seg)
        grain = {"bbox": bbox, "saliency": saliency, "contour": contour}

        # Extraer crop
        crop = self._extract_crop(image, grain)

        # Computar máscara binaria del grano (para masking y/o masked pooling)
        mask_binary = None
        if self.use_mask and contour is not None:
            crop, mask_binary = self._apply_mask(crop, grain, image.shape[:2])

        # Convertir BGR → RGB → PIL
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        if self.transform:
            crop_tensor = self.transform(crop_pil)
        else:
            crop_tensor = T.ToTensor()(crop_pil)

        if self.return_metadata:
            meta = {
                "image_path": image_path,
                "grain_index": idx,
                "bbox": bbox,
                "saliency": saliency,
                "class_name": self.classes[class_idx],
            }
            return crop_tensor, class_idx, meta

        # First-call diagnostic (once, in worker 0)
        if not self._logged_first_getitem:
            self._logged_first_getitem = True
            logger.info(
                f"  [DIAG] First crop: tensor={tuple(crop_tensor.shape)}, "
                f"range=[{crop_tensor.min():.3f}, {crop_tensor.max():.3f}], "
                f"class={self.classes[class_idx]}, mask_applied={mask_binary is not None}"
            )

        if self.return_mask and mask_binary is not None:
            mask_tensor = torch.from_numpy(
                (mask_binary > 0).astype(np.float32)
            )
            return crop_tensor, class_idx, mask_tensor

        return crop_tensor, class_idx

    def _extract_crop(self, image: np.ndarray, grain: Dict) -> np.ndarray:
        """Extrae crop cuadrado centrado en el grano con padding."""
        h_img, w_img = image.shape[:2]
        x, y, w, h = grain["bbox"]

        pad_x = int(w * self.crop_padding)
        pad_y = int(h * self.crop_padding)

        side = max(w + 2 * pad_x, h + 2 * pad_y)

        cx = x + w // 2
        cy = y + h // 2

        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w_img, x1 + side)
        y2 = min(h_img, y1 + side)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(
                crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR
            )

        return crop

    def _apply_mask(
        self, crop: np.ndarray, grain: Dict, original_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplica máscara del contorno con fondo configurable.

        Retorna:
            (crop_masked, mask_binary) donde mask_binary es (crop_size, crop_size) uint8 [0/255]

        mask_bg_mode:
            'random' — fondo aleatorio (evita shortcut learning por borde constante)
            'gray'   — gris neutro 128 (inference/val)
            'black'  — negro puro (legacy, causa shortcut learning)
        """
        contour = grain.get("contour")
        if contour is None or len(contour) < 3:
            # Sin contorno válido: retornar crop sin máscara
            full_mask = np.full((self.crop_size, self.crop_size), 255, dtype=np.uint8)
            return crop, full_mask

        h_img, w_img = original_shape
        x, y, w, h = grain["bbox"]
        pad_x = int(w * self.crop_padding)
        pad_y = int(h * self.crop_padding)
        side = max(w + 2 * pad_x, h + 2 * pad_y)

        cx = x + w // 2
        cy = y + h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)

        # Crear máscara en coordenadas del crop original
        mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        pts = contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(mask_full, [pts], -1, 255, -1)

        x2 = min(w_img, x1 + side)
        y2 = min(h_img, y1 + side)
        mask_crop = mask_full[y1:y2, x1:x2]
        mask_crop = cv2.resize(
            mask_crop, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST
        )

        # Aplicar fondo según modo
        bg_pixels = mask_crop == 0
        if self.mask_bg_mode == 'random':
            # 50% color sólido aleatorio, 50% ruido gaussiano
            if np.random.random() < 0.5:
                bg_color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
                crop[bg_pixels] = bg_color[0, 0]
            else:
                noise = np.random.randint(0, 256, crop.shape, dtype=np.uint8)
                crop[bg_pixels] = noise[bg_pixels]
        elif self.mask_bg_mode == 'gray':
            crop[bg_pixels] = 128
        else:  # 'black' (legacy)
            crop[bg_pixels] = 0

        return crop, mask_crop

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del dataset."""
        class_counts = {}
        for _, class_idx, _, _, _ in self._grain_samples:
            name = self.classes[class_idx]
            class_counts[name] = class_counts.get(name, 0) + 1

        return {
            "total_grains": len(self.samples),
            "num_classes": len(self.classes),
            "classes": self.classes,
            "grains_per_class": class_counts,
        }
