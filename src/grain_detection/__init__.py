"""
grain_detection — Módulo de detección y segmentación de granos de polen.

Pipeline: U²-Net (SOD) → Contornos → Anotaciones .seg → Crops on-the-fly

Componentes:
- salient_detector: Wrapper U²-Net para mapas de saliencia
- grain_detector: PollenGrainDetector (multi-grano con filtrado)
- seg_format: Parser/Writer del formato de anotación .seg
- detected_grain: Dataclasses DetectedGrain y GrainCountResult
- annotator: SegmentationAnnotator (generación batch de .seg)
"""

from .detected_grain import DetectedGrain, GrainCountResult
from .seg_format import SegFileWriter, SegFileReader
from .full_image_classifier import FullImageClassifier
