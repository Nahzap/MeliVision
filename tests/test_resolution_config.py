"""Tests for canonical pipeline resolution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.resolution_config import resolve_effective_image_size, sync_resolution_config


def test_resolve_prefers_data_image_size():
    cfg = {"data": {"image_size": 252}, "grain_detection": {"crop_size": 504}}
    assert resolve_effective_image_size(cfg) == 252


def test_sync_aligns_dependent_keys():
    cfg = {"data": {"image_size": 252, "resize_size": 504}, "grain_detection": {"crop_size": 504}}
    out, changed = sync_resolution_config(cfg)
    assert changed is True
    assert out["data"]["resize_size"] == 252
    assert out["grain_detection"]["crop_size"] == 252
