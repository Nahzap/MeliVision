"""Presets for morphological grain extraction (sensitive / balanced / strict)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

PROFILES: Dict[str, Dict[str, Any]] = {
    "sensitive": {
        "adaptive_k": 0.18,
        "min_area": 250,
        "max_area": 120000,
        "max_area_frac": 0.15,
        "min_circularity": 0.25,
        "max_bbox_side_frac": 0.45,
        "max_aspect_ratio": 3.0,
        "morph_kernel_size": 2,
    },
    "balanced": {
        "adaptive_k": 0.22,
        "min_area": 400,
        "max_area": 100000,
        "min_circularity": 0.32,
        "max_bbox_side_frac": 0.40,
        "max_aspect_ratio": 2.8,
        "morph_kernel_size": 3,
    },
    "strict": {
        "adaptive_k": 0.30,
        "min_area": 1000,
        "max_area": 80000,
        "min_circularity": 0.45,
        "max_bbox_side_frac": 0.35,
        "max_aspect_ratio": 2.5,
        "morph_kernel_size": 3,
    },
}


def resolve_grain_detection_config(grain_cfg: dict | None) -> dict:
    """Merge detection_profile preset into grain_detection config."""
    cfg = deepcopy(grain_cfg or {})
    profile = str(cfg.pop("detection_profile", "balanced")).strip().lower()
    preset = PROFILES.get(profile, PROFILES["balanced"])
    for key, value in preset.items():
        cfg.setdefault(key, value)
    cfg["detection_profile"] = profile if profile in PROFILES else "balanced"
    return cfg
