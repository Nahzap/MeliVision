"""
FullImageClassifier — Pipeline completo para clasificar granos en imagen de cámara.

Flujo:
    1. Detecta granos con PollenGrainDetector (U²-Net saliency)
    2. Recorta cada grano detectado
    3. Extrae embedding con AnalogyNet (modelo entrenado con metric learning)
    4. Clasifica via k-NN contra base de referencia (centroides o embeddings)
    5. Genera visualización con bboxes, labels, y heatmap de confianza

Uso:
    classifier = FullImageClassifier.from_checkpoint(
        checkpoint_path="models/final_model.pth",
        config_path="config.yaml",
    )
    results = classifier.classify_full_image("foto_camara.jpg")
    annotated = classifier.draw_results(image, results)
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)


class FullImageClassifier:
    """
    Pipeline end-to-end: imagen completa de cámara → clasificación por grano.

    Combina:
    - PollenGrainDetector (U²-Net) para localizar granos
    - AnalogyNet (metric learning) para extraer embeddings
    - k-NN para clasificar cada grano contra la base de referencia
    """

    def __init__(
        self,
        model: nn.Module,
        transforms,
        reference_embeddings: np.ndarray,
        reference_labels: np.ndarray,
        class_names: List[str],
        device: torch.device,
        grain_config: Optional[Dict] = None,
        k: int = 5,
    ):
        """
        Args:
            model: AnalogyNet entrenado (en eval mode)
            transforms: Transforms de validación (torchvision)
            reference_embeddings: Base de referencia (N, D)
            reference_labels: Labels de referencia (N,)
            class_names: Nombres de clases
            device: torch device
            grain_config: Config para PollenGrainDetector (opcional, usa defaults)
            k: Vecinos para k-NN
        """
        self.model = model
        self.model.eval()
        self.transforms = transforms
        self.device = device
        self.k = k

        # Clasificador k-NN con threshold de rechazo OOD
        from src.inference.knn_classifier import KNNClassifier
        dist_threshold = (grain_config or {}).get("distance_threshold", 0.0)
        self.knn = KNNClassifier(
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            class_names=class_names,
            k=k,
            distance_threshold=dist_threshold,
        )
        self.class_names = class_names

        # Detector de granos (lazy-loaded)
        self._grain_config = grain_config or {}
        self._detector = None

        # Parámetros de crop (deben coincidir con el entrenamiento)
        self.crop_padding = self._grain_config.get("crop_padding", 0.2)
        self.crop_size = self._grain_config.get("crop_size", 256)
        self.use_mask = self._grain_config.get("use_mask", True)

        logger.info(
            f"[FullImageClassifier] Inicializado — "
            f"{len(class_names)} clases, k={k}, "
            f"{len(reference_embeddings)} embeddings de referencia"
        )

    @property
    def detector(self):
        """Lazy-load del detector para no cargar U²-Net hasta que se necesite."""
        if self._detector is None:
            from src.grain_detection.grain_detector import PollenGrainDetector
            self._detector = PollenGrainDetector(config=self._grain_config)
        return self._detector

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "config.yaml",
        reference_path: Optional[str] = None,
        k: int = 5,
    ) -> "FullImageClassifier":
        """
        Factory method: carga modelo + referencia desde archivos.

        Args:
            checkpoint_path: Ruta al checkpoint del modelo entrenado
            config_path: Ruta al config.yaml
            reference_path: Ruta a reference_embeddings.pt (opcional,
                           busca en checkpoint o en el mismo directorio)
            k: Vecinos para k-NN

        Returns:
            FullImageClassifier listo para usar
        """
        from src.utils.config_utils import load_config, get_transforms_from_config
        from src.utils.model_utils import load_model_from_checkpoint

        config = load_config(config_path)
        device = torch.device(
            "cuda" if torch.cuda.is_available() and config.get("use_gpu", True)
            else "cpu"
        )

        # Cargar modelo
        model = load_model_from_checkpoint(
            checkpoint_path, config, device, eval_mode=True
        )

        # Transforms de validación
        transforms = get_transforms_from_config(config, mode="val")

        # Cargar embeddings de referencia
        ref_data = cls._load_reference(checkpoint_path, reference_path, device)
        reference_embeddings = ref_data["embeddings"]
        reference_labels = ref_data["labels"]
        class_names = ref_data["class_names"]

        # Config de detección de granos
        grain_config = config.get("grain_detection", {})

        return cls(
            model=model,
            transforms=transforms,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            class_names=class_names,
            device=device,
            grain_config=grain_config,
            k=k,
        )

    @staticmethod
    def _load_reference(
        checkpoint_path: str,
        reference_path: Optional[str],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Carga embeddings de referencia desde checkpoint o archivo separado."""
        # 1. Intentar archivo separado
        if reference_path and Path(reference_path).exists():
            data = torch.load(reference_path, map_location="cpu", weights_only=False)
            embs = data["embeddings"].numpy() if torch.is_tensor(data["embeddings"]) else data["embeddings"]
            labels = data["labels"].numpy() if torch.is_tensor(data["labels"]) else data["labels"]
            return {
                "embeddings": embs,
                "labels": labels,
                "class_names": data["class_names"],
            }

        # 2. Buscar en el mismo directorio del checkpoint y en el directorio padre
        #    (estructura: runs/XXX/checkpoints/best_model.pth, refs en runs/XXX/)
        ckpt_dir = Path(checkpoint_path).parent
        search_dirs = [ckpt_dir, ckpt_dir.parent]
        for search_dir in search_dirs:
            ref_file = search_dir / "reference_embeddings.pt"
            cent_file = search_dir / "class_centroids.pt"
            if ref_file.exists() and cent_file.exists():
                ref_data = torch.load(ref_file, map_location="cpu", weights_only=False)
                cent_data = torch.load(cent_file, map_location="cpu", weights_only=False)
                embs = ref_data["embeddings"].numpy() if torch.is_tensor(ref_data["embeddings"]) else ref_data["embeddings"]
                labels = ref_data["labels"].numpy() if torch.is_tensor(ref_data["labels"]) else ref_data["labels"]
                return {
                    "embeddings": embs,
                    "labels": labels,
                    "class_names": cent_data["class_names"],
                }

        # 3. Intentar desde dentro del checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "reference_embeddings" in ckpt:
            embs = ckpt["reference_embeddings"]
            if torch.is_tensor(embs):
                embs = embs.numpy()
            labels = ckpt["reference_labels"]
            if torch.is_tensor(labels):
                labels = labels.numpy()
            return {
                "embeddings": embs,
                "labels": labels,
                "class_names": ckpt["class_names"],
            }

        raise FileNotFoundError(
            f"No se encontraron embeddings de referencia.\n"
            f"Buscado en: {reference_path}, {ckpt_dir}, y dentro del checkpoint.\n"
            f"Ejecuta post_training.py primero para generar reference_embeddings.pt"
        )

    # =========================================================================
    # Pipeline principal
    # =========================================================================

    def classify_full_image(
        self,
        image_or_path,
        return_crops: bool = False,
    ) -> Dict[str, Any]:
        """
        Pipeline completo: detectar granos → clasificar cada uno.

        Args:
            image_or_path: np.ndarray BGR o ruta a imagen
            return_crops: Si True, incluye los crops en los resultados

        Returns:
            Dict con:
                - grains: lista de dicts por grano detectado
                    {bbox, contour, saliency, predicted_class, confidence,
                     class_distribution, embedding, crop?}
                - summary: {total_grains, per_class_counts, elapsed_ms}
                - image_shape: (H, W)
        """
        t0 = time.perf_counter()

        # Cargar imagen
        if isinstance(image_or_path, (str, Path)):
            image = cv2.imread(str(image_or_path))
            if image is None:
                raise IOError(f"No se pudo leer: {image_or_path}")
        else:
            image = image_or_path

        h, w = image.shape[:2]

        # 1. Detectar granos con U²-Net
        t1 = time.perf_counter()
        saliency_map, detected_grains = self.detector.detect(image)
        t_detect = (time.perf_counter() - t1) * 1000

        logger.info(
            f"[FullImage] Detección: {len(detected_grains)} granos en {t_detect:.0f}ms"
        )

        if not detected_grains:
            return {
                "grains": [],
                "summary": {
                    "total_grains": 0,
                    "per_class_counts": {},
                    "elapsed_ms": (time.perf_counter() - t0) * 1000,
                },
                "image_shape": (h, w),
                "saliency_map": saliency_map,
            }

        # 2. Recortar y clasificar cada grano
        t2 = time.perf_counter()
        grain_results = []

        for grain in detected_grains:
            # Extraer crop + aplicar máscara de contorno (fondo gris neutro)
            crop_bgr = self._extract_crop(image, grain)
            if self.use_mask and grain.contour is not None:
                crop_bgr, _ = self._apply_contour_mask(crop_bgr, image, grain)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            # Embedding
            if self.transforms:
                tensor = self.transforms(crop_pil)
            else:
                from torchvision import transforms as T
                tensor = T.ToTensor()(crop_pil)

            tensor = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(tensor).cpu().numpy()

            # Clasificar via k-NN
            prediction = self.knn.predict(
                embedding, return_distances=True, return_neighbors=False
            )
            distribution = self.knn.get_class_distribution(embedding)

            result = {
                "index": grain.index,
                "bbox": grain.bbox,
                "contour": grain.contour,
                "saliency": grain.saliency_prob,
                "predicted_class": prediction["predicted_class"],
                "predicted_label": prediction["predicted_label"],
                "confidence": prediction["confidence"],
                "mean_distance": prediction["mean_distance"],
                "class_distribution": distribution,
                "embedding": embedding.squeeze(),
            }
            if return_crops:
                result["crop"] = crop_bgr

            grain_results.append(result)

        t_classify = (time.perf_counter() - t2) * 1000

        # 3. Resumen
        per_class_counts = {}
        for gr in grain_results:
            cls = gr["predicted_class"]
            per_class_counts[cls] = per_class_counts.get(cls, 0) + 1

        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(
            f"[FullImage] Clasificación: {len(grain_results)} granos en {t_classify:.0f}ms "
            f"(total {elapsed:.0f}ms)"
        )

        return {
            "grains": grain_results,
            "summary": {
                "total_grains": len(grain_results),
                "per_class_counts": per_class_counts,
                "elapsed_ms": elapsed,
                "detect_ms": t_detect,
                "classify_ms": t_classify,
            },
            "image_shape": (h, w),
            "saliency_map": saliency_map,
        }

    def _extract_crop(self, image: np.ndarray, grain) -> np.ndarray:
        """Extrae crop cuadrado centrado en el grano con padding."""
        h_img, w_img = image.shape[:2]
        bx, by, bw, bh = grain.bbox

        pad_x = int(bw * self.crop_padding)
        pad_y = int(bh * self.crop_padding)
        side = max(bw + 2 * pad_x, bh + 2 * pad_y)

        cx = bx + bw // 2
        cy = by + bh // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w_img, x1 + side)
        y2 = min(h_img, y1 + side)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        else:
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))
        return crop

    def _apply_contour_mask(
        self, crop: np.ndarray, image: np.ndarray, grain
    ) -> tuple:
        """
        Aplica máscara del contorno: fondo gris neutro fuera del grano.
        Usa gris (128) en inferencia para ser consistente con training (val usa 'gray').

        Returns:
            (crop_masked, mask_binary) — mask is (crop_size, crop_size) uint8 [0/255]
        """
        contour = grain.contour
        if contour is None or len(contour) < 3:
            full_mask = np.full((self.crop_size, self.crop_size), 255, dtype=np.uint8)
            return crop, full_mask

        h_img, w_img = image.shape[:2]
        bx, by, bw, bh = grain.bbox
        pad_x = int(bw * self.crop_padding)
        pad_y = int(bh * self.crop_padding)
        side = max(bw + 2 * pad_x, bh + 2 * pad_y)

        cx = bx + bw // 2
        cy = by + bh // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w_img, x1 + side)
        y2 = min(h_img, y1 + side)

        # Crear máscara en coordenadas de imagen completa
        mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        pts = contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(mask_full, [pts], -1, 255, -1)

        # Recortar y redimensionar la máscara igual que el crop
        mask_crop = mask_full[y1:y2, x1:x2]
        mask_crop = cv2.resize(
            mask_crop, (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_NEAREST,
        )

        # Fondo gris neutro (consistente con training val: mask_bg_mode='gray')
        crop = crop.copy()
        crop[mask_crop == 0] = 128
        return crop, mask_crop

    # =========================================================================
    # Visualización
    # =========================================================================

    # Paleta de colores por clase (cíclica)
    CLASS_COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128),
    ]

    def draw_results(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        draw_contours: bool = True,
        draw_labels: bool = True,
        draw_confidence: bool = True,
        font_scale: float = 0.6,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Dibuja resultados de clasificación sobre la imagen.

        Args:
            image: Imagen BGR original
            results: Output de classify_full_image
            draw_contours: Dibujar contornos de granos
            draw_labels: Dibujar etiquetas de clase
            draw_confidence: Incluir confianza en la etiqueta

        Returns:
            Imagen anotada (copia)
        """
        annotated = image.copy()
        grains = results.get("grains", [])

        # Mapear clases a colores
        class_color_map = {}
        for i, name in enumerate(self.class_names):
            class_color_map[name] = self.CLASS_COLORS[i % len(self.CLASS_COLORS)]

        for grain in grains:
            cls = grain["predicted_class"]
            conf = grain["confidence"]
            color = class_color_map.get(cls, (255, 255, 255))
            bx, by, bw, bh = grain["bbox"]

            # Contorno
            if draw_contours and grain.get("contour") is not None:
                contour = grain["contour"]
                if len(contour) >= 3:
                    pts = contour.reshape(-1, 1, 2).astype(np.int32)
                    cv2.drawContours(annotated, [pts], -1, color, thickness)

            # BBox
            cv2.rectangle(
                annotated, (bx, by), (bx + bw, by + bh), color, thickness
            )

            # Label
            if draw_labels:
                label = cls
                if draw_confidence:
                    label = f"{cls} ({conf:.0%})"

                label_size, baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                lw, lh = label_size

                # Fondo del label
                cv2.rectangle(
                    annotated,
                    (bx, by - lh - baseline - 4),
                    (bx + lw + 4, by),
                    color, -1,
                )
                cv2.putText(
                    annotated, label,
                    (bx + 2, by - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 1, cv2.LINE_AA,
                )

        return annotated

    def generate_heatmap(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Genera heatmap de confianza sobre la imagen.

        Cada grano se colorea según la confianza de clasificación:
        - Verde: alta confianza (>80%)
        - Amarillo: media (50-80%)
        - Rojo: baja (<50%)

        Args:
            image: Imagen BGR original
            results: Output de classify_full_image
            alpha: Transparencia del heatmap (0-1)

        Returns:
            Imagen con heatmap overlay
        """
        overlay = image.copy()
        grains = results.get("grains", [])

        for grain in grains:
            conf = grain["confidence"]
            contour = grain.get("contour")
            bx, by, bw, bh = grain["bbox"]

            # Color según confianza: verde(alto) → amarillo(medio) → rojo(bajo)
            if conf >= 0.8:
                heat_color = (0, 200, 0)     # Verde
            elif conf >= 0.5:
                heat_color = (0, 200, 200)   # Amarillo
            else:
                heat_color = (0, 0, 200)     # Rojo

            if contour is not None and len(contour) >= 3:
                pts = contour.reshape(-1, 1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [pts], heat_color)
            else:
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), heat_color, -1)

        # Blend
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return result

    def generate_class_heatmap(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Genera heatmap coloreado por clase predicha.

        Cada grano se rellena con el color de su clase.

        Args:
            image: Imagen BGR original
            results: Output de classify_full_image
            alpha: Transparencia

        Returns:
            Imagen con class-heatmap overlay
        """
        overlay = image.copy()
        grains = results.get("grains", [])

        class_color_map = {}
        for i, name in enumerate(self.class_names):
            class_color_map[name] = self.CLASS_COLORS[i % len(self.CLASS_COLORS)]

        for grain in grains:
            cls = grain["predicted_class"]
            color = class_color_map.get(cls, (128, 128, 128))
            contour = grain.get("contour")
            bx, by, bw, bh = grain["bbox"]

            if contour is not None and len(contour) >= 3:
                pts = contour.reshape(-1, 1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
            else:
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, -1)

        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return result
