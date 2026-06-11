"""Helpers to release DataLoader worker RAM and auto-tune CPU/RAM for throughput."""

from __future__ import annotations

import gc
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def dispose_dataloader_workers(loader) -> None:
    """Shut down a DataLoader's worker pool and drop any active iterator."""
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        if hasattr(iterator, "_shutdown_workers"):
            try:
                iterator._shutdown_workers()
            except Exception as exc:
                logger.debug("DataLoader worker shutdown: %s", exc)
        loader._iterator = None


def dispose_data_engine(data_engine) -> None:
    """Release train and val worker pools held by GrainDataEngine."""
    if data_engine is None:
        return
    dispose_dataloader_workers(getattr(data_engine, "train_loader", None))
    dispose_dataloader_workers(getattr(data_engine, "val_loader", None))
    if hasattr(data_engine, "release_train_workers"):
        data_engine.release_train_workers()
    if hasattr(data_engine, "release_val_workers"):
        data_engine.release_val_workers()


def free_training_memory(stage: str = "") -> None:
    """Best-effort release of cached RAM / VRAM between estimate and train phases."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("CUDA memory cleanup: %s", exc)
    if stage:
        logger.info("Memory cleanup after %s", stage)


def estimate_worker_ram_mb(config: dict) -> dict:
    """Rough per-process RAM budget for DataLoader workers (Windows spawn)."""
    data = config.get("data", {})
    grain = config.get("grain_detection", {})
    train_w = max(0, int(data.get("num_workers", 0)))
    val_raw = data.get("val_num_workers", -1)
    val_w = train_w if val_raw is None or int(val_raw) < 0 else max(0, min(int(val_raw), train_w))
    cache_mb = int(grain.get("image_cache_max_mb", 48))
    prefetch = max(1, int(data.get("prefetch_factor", 2)))
    overhead_mb = 120 + prefetch * 30
    per_worker = cache_mb + overhead_mb
    policy = data.get("loader_policy", {}) or {}
    exclusive = bool(policy.get("exclusive_train_workers_during_val", True))
    active_workers = max(train_w, val_w) if exclusive else (train_w + val_w)
    return {
        "train_workers": train_w,
        "val_workers": val_w,
        "per_worker_mb": per_worker,
        "total_workers_mb": active_workers * per_worker,
        "exclusive_workers": exclusive,
        "prefetch_factor": prefetch,
        "image_cache_max_mb": cache_mb,
    }


def auto_tune_resources(config: dict, log: Optional[logging.Logger] = None) -> dict:
    """
    Fill in missing DataLoader / cache settings based on available system resources.

    IMPORTANT: Values already defined in config.yaml are NEVER overridden.
    Only missing keys get auto-tuned defaults.
    """
    log = log or logger
    data = config.setdefault("data", {})
    grain = config.setdefault("grain_detection", {})

    if not data.get("auto_tune_resources", True):
        return estimate_worker_ram_mb(config)

    try:
        import psutil

        vm = psutil.virtual_memory()
        avail_gb = vm.available / (1024**3)
        total_gb = vm.total / (1024**3)
    except ImportError:
        avail_gb = 8.0
        total_gb = 16.0
        log.debug("psutil not installed - using default auto_tune assumptions")

    cpu = os.cpu_count() or 4
    max_workers = max(2, min(6, cpu - 1))
    if avail_gb >= 20:
        default_workers = min(max_workers, 4)
        default_prefetch = 4
        default_cache_mb = 128
        default_cache_slots = 8
        default_budget_mb = int(min(avail_gb * 1024 * 0.45, 3200))
    elif avail_gb >= 12:
        default_workers = min(max_workers, 4)
        default_prefetch = 4
        default_cache_mb = 96
        default_cache_slots = 6
        default_budget_mb = int(min(avail_gb * 1024 * 0.40, 2400))
    elif avail_gb >= 8:
        default_workers = min(max_workers, 3)
        default_prefetch = 3
        default_cache_mb = 64
        default_cache_slots = 5
        default_budget_mb = int(min(avail_gb * 1024 * 0.35, 1600))
    else:
        default_workers = min(max_workers, 2)
        default_prefetch = 2
        default_cache_mb = 48
        default_cache_slots = 4
        default_budget_mb = 900

    # Respetar config.yaml: solo llenar valores faltantes
    data.setdefault("num_workers", default_workers)
    data.setdefault("prefetch_factor", default_prefetch)
    data.setdefault("worker_ram_budget_mb", default_budget_mb)

    val_raw = data.get("val_num_workers", -1)
    if val_raw is not None and int(val_raw) >= 0:
        data["val_num_workers"] = min(int(val_raw), int(data["num_workers"]))

    grain.setdefault("image_cache_max_mb", default_cache_mb)
    grain.setdefault("image_cache_size", default_cache_slots)
    grain.setdefault("max_cached_image_mb", 32)

    # Grain val: main-process loader avoids spawning a 2nd worker pool each epoch
    grain_enabled = bool(grain.get("enabled", True))
    val_raw = data.get("val_num_workers", -1)
    if grain_enabled and (val_raw is None or int(val_raw) < 0):
        data["val_num_workers"] = 0
        log.info(
            "Auto-tune: val_num_workers=0 (grain mode — single worker pool for train only)"
        )

    aug = config.setdefault("augmentation", {})
    if aug.get("performance_mode", True):
        micro = aug.setdefault("microscopy_specific", {})
        micro.setdefault("optical_distortion", {})["enabled"] = False
        domain = aug.setdefault("domain_augmentation", {})
        if domain.get("enabled", True):
            domain["enabled"] = False

    log.info(
        "Auto-tune: CPU=%d cores, RAM %.1f/%.1f GB avail -> workers=%d, prefetch=%d, "
        "cache=%dMB x %d slots, worker_budget=%dMB",
        cpu,
        avail_gb,
        total_gb,
        int(data["num_workers"]),
        int(data["prefetch_factor"]),
        int(grain["image_cache_max_mb"]),
        int(grain["image_cache_size"]),
        int(data["worker_ram_budget_mb"]),
    )
    return estimate_worker_ram_mb(config)


def apply_ram_safe_dataloader_limits(config: dict, logger: Optional[logging.Logger] = None) -> dict:
    """
    Clamp worker/cache settings only when estimated worker RAM exceeds budget.
  """
    log = logger or logging.getLogger(__name__)
    data = config.setdefault("data", {})
    grain = config.setdefault("grain_detection", {})

    budget_mb = int(data.get("worker_ram_budget_mb", 900))
    est = estimate_worker_ram_mb(config)

    if est["total_workers_mb"] <= budget_mb:
        return est

    log.warning(
        "Worker RAM estimate %.0f MB exceeds budget %d MB - reducing parallelism.",
        est["total_workers_mb"],
        budget_mb,
    )

    train_w = est["train_workers"]
    while est["total_workers_mb"] > budget_mb and train_w > 0:
        train_w -= 1
        data["num_workers"] = train_w
        val_raw = data.get("val_num_workers", -1)
        if val_raw is not None and int(val_raw) >= 0:
            data["val_num_workers"] = min(int(val_raw), train_w)
        est = estimate_worker_ram_mb(config)

    if est["total_workers_mb"] > budget_mb:
        grain["image_cache_max_mb"] = max(32, int(grain.get("image_cache_max_mb", 48)) - 32)
        grain["image_cache_size"] = max(2, int(grain.get("image_cache_size", 4)) - 1)
        data["prefetch_factor"] = max(1, int(data.get("prefetch_factor", 2)) - 1)
        est = estimate_worker_ram_mb(config)

    log.info(
        "RAM-safe DataLoader: train_workers=%d, val_workers=%d, cache=%dMB, prefetch=%d "
        "(est. workers RAM ~%.0f MB)",
        est["train_workers"],
        est["val_workers"],
        est["image_cache_max_mb"],
        est["prefetch_factor"],
        est["total_workers_mb"],
    )
    return est
