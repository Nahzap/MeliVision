"""
PollenGrainDetector — Detector multi-grano de polen usando U²-Net.

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

        # Paso 2: Extraer granos del mapa de saliencia
        grains = self._extract_grains(saliency, image)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[PollenGrainDetector] {len(grains)} granos detectados en {elapsed:.0f}ms"
        )

        return saliency, grains

    def _extract_grains(
        self, saliency: np.ndarray, image: np.ndarray
    ) -> List[DetectedGrain]:
        """
        Extrae granos individuales del mapa de saliencia.

        Pipeline:
        1. Umbral adaptativo (media + k*std)
        2. Operaciones morfológicas (close + open)
        3. Extracción de contornos externos
        4. Filtrado por área y circularidad
        """
        # Estadísticas del mapa
        sal_mean = float(np.mean(saliency))
        sal_std = float(np.std(saliency))
        sal_max = float(np.max(saliency))

        # Umbral adaptativo
        adaptive_threshold = sal_mean + self.adaptive_k * sal_std
        adaptive_threshold = max(0.15, min(0.5, adaptive_threshold))

        if sal_max < 0.3:
            adaptive_threshold = min(adaptive_threshold, sal_max * 0.7)

        logger.debug(
            f"[Extract] mean={sal_mean:.3f}, std={sal_std:.3f}, "
            f"max={sal_max:.3f} → threshold={adaptive_threshold:.3f}"
        )

        # Binarizar
        binary = (saliency > adaptive_threshold).astype(np.uint8) * 255

        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Contornos externos
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        grains = []
        rejected_small = 0
        rejected_large = 0
        rejected_circ = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.min_area:
                rejected_small += 1
                continue
            if area > self.max_area:
                rejected_large += 1
                continue

            # Circularidad
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
            else:
                circularity = 0.0

            if circularity < self.min_circularity:
                rejected_circ += 1
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Centroide
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2

            # Probabilidad promedio en la región
            mask = np.zeros(saliency.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            probability = float(np.mean(saliency[mask > 0]))

            # Contorno como array Nx2
            contour_points = contour.reshape(-1, 2)

            grain = DetectedGrain(
                index=len(grains),
                bbox=(x, y, w, h),
                area=float(area),
                saliency_prob=probability,
                centroid=(cx, cy),
                contour=contour_points,
            )
            grains.append(grain)

        if rejected_small or rejected_large or rejected_circ:
            logger.debug(
                f"[Extract] Rechazados: {rejected_small} (área<min), "
                f"{rejected_large} (área>max), {rejected_circ} (circularidad)"
            )

        # Ordenar por área descendente y reindexar
        grains.sort(key=lambda g: g.area, reverse=True)
        for i, g in enumerate(grains):
            g.index = i

        return grains


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
