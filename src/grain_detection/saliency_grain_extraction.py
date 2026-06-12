"""
Utilidades compartidas para extraer granos desde mapas de saliencia.

Usadas por PollenGrainDetector (U²-Net, localización + preprocesamiento .seg) y
ModelGrainDetector (backend legacy: atención ViT).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .detected_grain import DetectedGrain

logger = logging.getLogger(__name__)


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-6)


def nms_merge_grains(
    grains: List[DetectedGrain], *, iou_threshold: float = 0.35
) -> List[DetectedGrain]:
    if not grains:
        return []
    ordered = sorted(grains, key=lambda g: g.saliency_prob, reverse=True)
    kept: List[DetectedGrain] = []
    for grain in ordered:
        if all(bbox_iou(grain.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(grain)
    kept.sort(key=lambda g: g.area, reverse=True)
    for i, g in enumerate(kept):
        g.index = i
    return kept


def extract_grains_from_saliency(
    saliency: np.ndarray,
    image: np.ndarray,
    *,
    adaptive_k: float = 0.30,
    min_area: int = 200,
    max_area: int = 200000,
    morph_kernel_size: int = 3,
    min_circularity: float = 0.3,
    adaptive_k_override: Optional[float] = None,
    min_circularity_override: Optional[float] = None,
) -> List[DetectedGrain]:
    """Extrae granos individuales de un mapa de saliencia [0-1]."""
    if saliency is None or saliency.size == 0:
        return []

    sal_mean = float(np.mean(saliency))
    sal_std = float(np.std(saliency))
    sal_max = float(np.max(saliency))

    k_val = adaptive_k if adaptive_k_override is None else adaptive_k_override
    circ_min = (
        min_circularity
        if min_circularity_override is None
        else min_circularity_override
    )

    adaptive_threshold = sal_mean + k_val * sal_std
    adaptive_threshold = max(0.08, min(0.5, adaptive_threshold))

    if sal_max < 0.3:
        adaptive_threshold = min(adaptive_threshold, sal_max * 0.75)

    binary = (saliency > adaptive_threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (morph_kernel_size, morph_kernel_size),
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    grains: List[DetectedGrain] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
        if circularity < circ_min:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2

        mask = np.zeros(saliency.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        probability = float(np.mean(saliency[mask > 0]))

        grains.append(
            DetectedGrain(
                index=len(grains),
                bbox=(x, y, w, h),
                area=float(area),
                saliency_prob=probability,
                centroid=(cx, cy),
                contour=contour.reshape(-1, 2),
            )
        )

    grains.sort(key=lambda g: g.area, reverse=True)
    for i, g in enumerate(grains):
        g.index = i
    return grains
