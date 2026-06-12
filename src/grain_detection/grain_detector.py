"""
PollenGrainDetector — Detector multi-grano de polen usando U²-Net.

Uso previsto: **solo preprocesamiento** (generar .seg / recortes antes del entrenamiento).
No requiere fine-tune en el pipeline de metric learning.

Adaptado del U2NetDetector validado en XYZ_Ctrl_L206_GUI.
Pipeline: Saliency map → Umbral adaptativo → Morfología → Contornos → Filtrado

Ref:
- Qin et al. (2020) U²-Net: Going Deeper with Nested U-Structure for SOD
- Expert Systems with Applications (2024): Saliency-guided cell counting
- XYZ_Ctrl_L206_GUI/src/core/detection/u2net_detector.py (implementación validada)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detected_grain import DetectedGrain
from .salient_detector import SalientObjectDetector
from .saliency_grain_extraction import (
    bbox_iou,
    extract_grains_from_saliency,
    nms_merge_grains,
)

logger = logging.getLogger(__name__)


class PollenGrainDetector:
    """
    Detector de granos de polen individuales usando U²-Net como SOD.

    Genera lista de DetectedGrain con bbox, contorno, centroide y saliency.
    Los parámetros están preconfigurados para microscopía de polen (modo SENSITIVE
    de XYZ) pero son completamente configurables.

    Uso:
        detector = PollenGrainDetector()
        saliency, grains = detector.detect(image)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el detector.

        Args:
            config: Dict con parámetros. Keys opcionales:
                model_type, input_size, saliency_threshold, adaptive_k,
                min_area, max_area, morph_kernel_size, min_circularity,
                crop_padding, crop_size, device
        """
        config = config or {}

        # Parámetros de detección (defaults = modo SENSITIVE para polen)
        self.saliency_threshold = config.get("saliency_threshold", 0.30)
        self.adaptive_k = config.get("adaptive_k", 0.30)
        self.min_area = config.get("min_area", 200)
        self.max_area = config.get("max_area", 200000)
        self.morph_kernel_size = config.get("morph_kernel_size", 3)
        self.min_circularity = config.get("min_circularity", 0.3)

        # Parámetros de crop
        self.crop_padding = config.get("crop_padding", 0.20)
        self.crop_size = config.get("crop_size", 256)

        # Modelo U²-Net
        model_type = config.get("model_type", "u2netp")
        input_size = config.get("input_size", 320)
        device = config.get("device", None)

        self._sod = SalientObjectDetector(
            model_type=model_type,
            device=device,
            input_size=input_size,
            auto_download=True,
        )

        logger.info(
            f"[PollenGrainDetector] Inicializado — "
            f"threshold={self.saliency_threshold}, k={self.adaptive_k}, "
            f"min_area={self.min_area}, kernel={self.morph_kernel_size}"
        )

    def detect(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[DetectedGrain]]:
        """
        Detecta granos de polen individuales en la imagen.

        Args:
            image: Imagen BGR (numpy array), cualquier resolución.

        Returns:
            saliency_map: Mapa de probabilidad [0-1] del tamaño original.
            grains: Lista de DetectedGrain ordenados por área descendente.
        """
        if image is None or image.size == 0:
            return np.zeros((100, 100), dtype=np.float32), []

        t0 = time.perf_counter()

        # Paso 1: Saliency map
        saliency = self._sod.get_saliency_map(image)

        grains = self._extract_grains(saliency, image)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[PollenGrainDetector] {len(grains)} granos detectados en {elapsed:.0f}ms"
        )

        return saliency, grains

    def detect_exhaustive(
        self, image: np.ndarray, *, iou_threshold: float = 0.35
    ) -> Tuple[np.ndarray, List[DetectedGrain]]:
        """
        Detección agresiva para full-image inference: múltiples umbrales y fusión NMS.

        Encuentra granos no anotados en .seg (descubrimiento más allá de etiquetas previas).
        """
        if image is None or image.size == 0:
            return np.zeros((100, 100), dtype=np.float32), []

        t0 = time.perf_counter()
        saliency = self._sod.get_saliency_map(image)

        merged: List[DetectedGrain] = []
        k_scales = (1.0, 0.65, 0.4)
        circ_scales = (1.0, 0.75, 0.5)
        for k_scale in k_scales:
            for circ_scale in circ_scales:
                merged.extend(
                    self._extract_grains(
                        saliency,
                        image,
                        adaptive_k_override=max(0.08, self.adaptive_k * k_scale),
                        min_circularity_override=max(0.12, self.min_circularity * circ_scale),
                    )
                )

        grains = nms_merge_grains(merged, iou_threshold=iou_threshold)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[PollenGrainDetector] exhaustive: {len(grains)} granos "
            f"(candidatos={len(merged)}) en {elapsed:.0f}ms"
        )
        return saliency, grains

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        return bbox_iou(a, b)

    def _nms_merge_grains(
        self, grains: List[DetectedGrain], *, iou_threshold: float = 0.35
    ) -> List[DetectedGrain]:
        return nms_merge_grains(grains, iou_threshold=iou_threshold)

    def _extract_grains(
        self,
        saliency: np.ndarray,
        image: np.ndarray,
        *,
        adaptive_k_override: Optional[float] = None,
        min_circularity_override: Optional[float] = None,
    ) -> List[DetectedGrain]:
        """Delega extracción morfológica a saliency_grain_extraction."""
        return extract_grains_from_saliency(
            saliency,
            image,
            adaptive_k=self.adaptive_k,
            min_area=self.min_area,
            max_area=self.max_area,
            morph_kernel_size=self.morph_kernel_size,
            min_circularity=self.min_circularity,
            adaptive_k_override=adaptive_k_override,
            min_circularity_override=min_circularity_override,
        )


    def set_parameters(self, **kwargs):
        """Actualiza parámetros de detección."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"[PollenGrainDetector] {key} = {value}")

    def get_parameters(self) -> Dict:
        """Retorna parámetros actuales."""
        return {
            "saliency_threshold": self.saliency_threshold,
            "adaptive_k": self.adaptive_k,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "morph_kernel_size": self.morph_kernel_size,
            "min_circularity": self.min_circularity,
            "crop_padding": self.crop_padding,
            "crop_size": self.crop_size,
        }

    def is_ready(self) -> bool:
        return self._sod.is_ready()
