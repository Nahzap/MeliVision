"""Full-image localization backend selection."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grain_detection.full_image_classifier import (
    LOCALIZATION_DETECTORS,
    resolve_localization_detector,
)


class TestResolveLocalizationDetector:
    def test_default_is_u2net(self):
        assert resolve_localization_detector({}) == "u2net"
        assert resolve_localization_detector(None) == "u2net"

    @pytest.mark.parametrize("alias", ["pollen", "u2-net", "U2NET"])
    def test_u2net_aliases(self, alias):
        assert resolve_localization_detector({"localization_detector": alias}) == "u2net"

    @pytest.mark.parametrize("alias", ["vit", "model", "analogynet", "vit_attention"])
    def test_vit_aliases(self, alias):
        assert resolve_localization_detector({"localization_detector": alias}) == "vit"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="localization_detector"):
            resolve_localization_detector({"localization_detector": "yolo"})

    def test_known_backends(self):
        assert LOCALIZATION_DETECTORS == ("u2net", "vit")
