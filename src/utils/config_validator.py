"""
Validación de precondiciones del pipeline MetricLearning.

Usado al arranque de la GUI y en scripts CLI para detectar configuraciones
inconsistentes antes de entrenar o inferir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _get_nested(config: dict, *keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_pipeline_summary(config: dict) -> str:
    """Resumen legible del pipeline configurado."""
    loss_type = _get_nested(config, "training", "loss", "type", default="?")
    num_slices = _get_nested(config, "training", "loss", "params", "num_slices", default=4)
    embed_dim = _get_nested(config, "model", "embedding_dim", default=128)
    slice_dim = embed_dim // num_slices if num_slices else "?"
    sd = _get_nested(config, "sota", "self_distillation", "enabled", default=False)
    sd_label = "on" if sd else "off"
    return f"{loss_type} {num_slices}×{slice_dim}D | self-distill {sd_label}"


def validate_config(
    config: dict,
    config_path: Optional[str] = None,
    check_data_dirs: bool = True,
) -> Dict[str, Any]:
    """
    Valida coherencia de configuración del pipeline multi-slice.

    Returns:
        dict con keys: valid, errors, warnings, pipeline_summary
    """
    errors: List[str] = []
    warnings: List[str] = []

    embed_dim = int(_get_nested(config, "model", "embedding_dim", default=128))
    loss_type = _get_nested(config, "training", "loss", "type", default="")
    loss_params = _get_nested(config, "training", "loss", "params", default={}) or {}
    num_slices = int(loss_params.get("num_slices", 4))
    classes_per_batch = int(_get_nested(config, "data", "classes_per_batch", default=8))

    if embed_dim % num_slices != 0:
        errors.append(
            f"model.embedding_dim ({embed_dim}) no es divisible por "
            f"num_slices ({num_slices})"
        )

    if loss_type in ("sliced_ms", "sliced_proxy"):
        if classes_per_batch < 2:
            errors.append(
                f"training.loss.type={loss_type} requiere data.classes_per_batch >= 2 "
                f"(actual: {classes_per_batch})"
            )
        if loss_type == "sliced_ms" and "num_slices" not in loss_params:
            warnings.append("sliced_ms sin num_slices explícito; se usará default 4")

    use_gpu = _get_nested(config, "hardware", "use_gpu", default=True)
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.append("hardware.use_gpu=true pero CUDA no está disponible")
        except ImportError:
            warnings.append("PyTorch no importable; no se pudo verificar GPU")

    if check_data_dirs:
        for split_key in ("train_dir", "val_dir", "test_dir"):
            rel = _get_nested(config, "data", split_key)
            if not rel:
                warnings.append(f"data.{split_key} no definido")
                continue
            path = Path(rel)
            if not path.exists():
                warnings.append(f"Directorio no encontrado: {rel} (data.{split_key})")
            elif not any(path.iterdir()) if path.is_dir() else False:
                warnings.append(f"Directorio vacío: {rel} (data.{split_key})")

    if config_path and not Path(config_path).exists():
        errors.append(f"Archivo de configuración no encontrado: {config_path}")

    # Parámetros huérfanos bajo sliced_ms (no bloquean, solo avisan)
    if loss_type == "sliced_ms":
        orphan_keys = {"proxy_scale", "proxy_margin", "margin", "scale", "base", "use_ldm"}
        present = orphan_keys.intersection(loss_params.keys())
        if present:
            warnings.append(
                f"Parámetros no usados por sliced_ms en loss.params: {sorted(present)}"
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "pipeline_summary": build_pipeline_summary(config),
    }


def validate_config_file(
    config_path: str,
    check_data_dirs: bool = True,
) -> Dict[str, Any]:
    """Carga YAML y valida."""
    path = Path(config_path)
    if not path.exists():
        return {
            "valid": False,
            "errors": [f"Archivo no encontrado: {config_path}"],
            "warnings": [],
            "pipeline_summary": "config no cargada",
        }
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    result = validate_config(config, config_path=str(path), check_data_dirs=check_data_dirs)
    result["config_path"] = str(path.resolve())
    return result
