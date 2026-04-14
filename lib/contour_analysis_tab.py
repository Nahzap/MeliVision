"""
ContourAnalysisTab — Tab GUI para análisis de contornos y segmentación de granos.

Posición: Tab index 2 (después de Gestor de Datos, antes de Entrenamiento).

Secciones:
1. Configuración de segmentación (parámetros U²-Net + botones de acción)
2. Explorador de contornos (imagen con contornos + crop del grano seleccionado)
3. Estadísticas (conteos, distribución por clase)
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSlider, QSpinBox, QProgressBar, QFormLayout, QSplitter,
    QSizePolicy, QMessageBox, QScrollArea, QFrame, QListWidget, QListWidgetItem,
    QCheckBox, QDoubleSpinBox, QShortcut
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QKeySequence

from lib.styles import Styles, Colors

# Configurar logger con handler de consola
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class ClickableImageLabel(QLabel):
    """QLabel that emits click coordinates mapped to original image space.

    Left-click  → ``left_clicked(x_orig, y_orig)``
    Right-click → ``right_clicked(x_orig, y_orig)``

    The label stores the original image dimensions and the displayed pixmap
    so it can map widget-space coordinates back to the original image.
    """
    left_clicked = pyqtSignal(int, int)
    right_clicked = pyqtSignal(int, int)
    left_double_clicked = pyqtSignal(int, int)
    right_double_clicked = pyqtSignal(int, int)
    left_dragged = pyqtSignal(int, int)
    right_dragged = pyqtSignal(int, int)
    left_released = pyqtSignal(int, int)
    right_released = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orig_w = 0
        self._orig_h = 0
        self.setMouseTracking(True)

    def set_original_size(self, w: int, h: int):
        self._orig_w = w
        self._orig_h = h

    # Signals for mouse wheel navigation
    wheel_up = pyqtSignal()
    wheel_down = pyqtSignal()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.wheel_up.emit()
        elif delta < 0:
            self.wheel_down.emit()

    def mousePressEvent(self, event):
        mapped = self._map_event_to_orig(event.pos().x(), event.pos().y())
        if mapped is None:
            return super().mousePressEvent(event)
        ox, oy = mapped

        if event.button() == Qt.LeftButton:
            self.left_clicked.emit(ox, oy)
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(ox, oy)

    def mouseDoubleClickEvent(self, event):
        mapped = self._map_event_to_orig(event.pos().x(), event.pos().y())
        if mapped is None:
            return super().mouseDoubleClickEvent(event)
        ox, oy = mapped
        if event.button() == Qt.LeftButton:
            self.left_double_clicked.emit(ox, oy)
        elif event.button() == Qt.RightButton:
            self.right_double_clicked.emit(ox, oy)

    def mouseMoveEvent(self, event):
        mapped = self._map_event_to_orig(event.pos().x(), event.pos().y())
        if mapped is not None:
            ox, oy = mapped
            btns = event.buttons()
            if btns & Qt.LeftButton:
                self.left_dragged.emit(ox, oy)
            if btns & Qt.RightButton:
                self.right_dragged.emit(ox, oy)
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        mapped = self._map_event_to_orig(event.pos().x(), event.pos().y())
        if mapped is not None:
            ox, oy = mapped
            if event.button() == Qt.LeftButton:
                self.left_released.emit(ox, oy)
            elif event.button() == Qt.RightButton:
                self.right_released.emit(ox, oy)
        return super().mouseReleaseEvent(event)

    def _map_event_to_orig(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        pm = self.pixmap()
        if pm is None or pm.isNull() or self._orig_w == 0:
            return None

        # Widget may be larger than pixmap — compute offset
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        off_x = (lw - pw) // 2 if lw > pw else 0
        off_y = (lh - ph) // 2 if lh > ph else 0

        rx = x - off_x
        ry = y - off_y
        if rx < 0 or ry < 0 or rx >= pw or ry >= ph:
            return None

        # Map to original image coords
        ox = int(rx / pw * self._orig_w)
        oy = int(ry / ph * self._orig_h)
        return (ox, oy)


class ManualContourWorker(QThread):
    """Run U2-Net on a local crop around a click point (non-blocking)."""
    finished = pyqtSignal(list)   # list of grain dicts (may be empty)
    error = pyqtSignal(str)

    def __init__(self, image_bgr, click_x, click_y, config, crop_radius=250):
        super().__init__()
        self.image_bgr = image_bgr
        self.click_x = click_x
        self.click_y = click_y
        self.config = config
        self.crop_radius = crop_radius

    def run(self):
        try:
            from src.grain_detection.grain_detector import PollenGrainDetector
            h_img, w_img = self.image_bgr.shape[:2]
            cx, cy = self.click_x, self.click_y
            r = self.crop_radius

            # Extract local crop around the click point
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w_img, cx + r)
            y2 = min(h_img, cy + r)
            crop = self.image_bgr[y1:y2, x1:x2]

            if crop.size == 0:
                self.finished.emit([])
                return

            click_in_crop_x = cx - x1
            click_in_crop_y = cy - y1

            # Run detector on the local crop
            detector = PollenGrainDetector(config=self.config)
            saliency = detector._sod.get_saliency_map(crop)
            seeded = self._u2net_seeded_contour(
                saliency, crop, click_in_crop_x, click_in_crop_y
            )
            if seeded is not None and len(seeded) >= 3:
                bx, by, bw, bh = cv2.boundingRect(seeded.reshape(-1, 1, 2))
                local_mask = np.zeros(saliency.shape, dtype=np.uint8)
                cv2.drawContours(local_mask, [seeded.reshape(-1, 1, 2)], -1, 255, -1)
                sal_prob = float(np.mean(saliency[local_mask > 0])) if np.any(local_mask > 0) else float(
                    saliency[min(max(click_in_crop_y, 0), saliency.shape[0] - 1),
                             min(max(click_in_crop_x, 0), saliency.shape[1] - 1)]
                )
                contour_global = seeded.copy()
                contour_global[:, 0] += x1
                contour_global[:, 1] += y1
                grain_dict = {
                    "index": 0,
                    "class_index": 0,
                    "bbox": (bx + x1, by + y1, bw, bh),
                    "saliency": sal_prob,
                    "contour": contour_global,
                }
                self.finished.emit([grain_dict])
                return

            _, grains = detector.detect(crop)

            if not grains:
                # Fallback SIEMPRE basado en U2NET (sin geometría inventada):
                # construir región desde saliency conectada al punto semilla.
                contour_seed = self._u2net_seeded_contour(
                    saliency, crop, click_in_crop_x, click_in_crop_y
                )
                if contour_seed is None or len(contour_seed) < 3:
                    self.finished.emit([])
                    return

                bx, by, bw, bh = cv2.boundingRect(contour_seed.reshape(-1, 1, 2))
                local_mask = np.zeros(saliency.shape, dtype=np.uint8)
                cv2.drawContours(local_mask, [contour_seed.reshape(-1, 1, 2)], -1, 255, -1)
                sal_prob = float(np.mean(saliency[local_mask > 0])) if np.any(local_mask > 0) else float(
                    saliency[min(max(click_in_crop_y, 0), saliency.shape[0] - 1),
                             min(max(click_in_crop_x, 0), saliency.shape[1] - 1)]
                )

                contour_global = contour_seed.copy()
                contour_global[:, 0] += x1
                contour_global[:, 1] += y1

                grain_dict = {
                    "index": 0,
                    "class_index": 0,
                    "bbox": (bx + x1, by + y1, bw, bh),
                    "saliency": sal_prob,
                    "contour": contour_global,
                }
                self.finished.emit([grain_dict])
                return

            # Preferir contorno que contenga la semilla; fallback: centroide más cercano.
            containing = []
            for g in grains:
                if g.contour is None or len(g.contour) < 3:
                    continue
                inside = cv2.pointPolygonTest(
                    g.contour.reshape(-1, 1, 2).astype(np.int32),
                    (float(click_in_crop_x), float(click_in_crop_y)),
                    False,
                )
                if inside >= 0:
                    containing.append(g)

            if containing:
                best = max(containing, key=lambda g: float(g.saliency_prob))
            else:
                best = min(grains, key=lambda g: (
                    (g.centroid[0] - click_in_crop_x) ** 2 +
                    (g.centroid[1] - click_in_crop_y) ** 2
                ))

            # Map contour and bbox back to original image coords
            contour = best.contour.copy() if best.contour is not None else None
            if contour is not None:
                contour[:, 0] += x1
                contour[:, 1] += y1

            bx, by, bw, bh = best.bbox
            grain_dict = {
                "index": 0,
                "class_index": 0,
                "bbox": (bx + x1, by + y1, bw, bh),
                "saliency": best.saliency_prob,
                "contour": contour,
            }
            self.finished.emit([grain_dict])
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    @staticmethod
    def _u2net_seeded_contour(
        saliency: np.ndarray, crop_bgr: np.ndarray, seed_x: int, seed_y: int
    ) -> Optional[np.ndarray]:
        """
        Deriva UN único contorno externo desde saliency U2NET usando semilla.
        - Sin Canny.
        - Selección por componente conectada.
        - Frontera externa elegida por área (Green/shoelace).
        """
        if saliency is None or saliency.size == 0:
            return None

        h, w = saliency.shape[:2]
        sx = int(np.clip(seed_x, 0, w - 1))
        sy = int(np.clip(seed_y, 0, h - 1))
        seed_val = float(saliency[sy, sx])
        mean_val = float(np.mean(saliency))
        std_val = float(np.std(saliency))
        min_area = max(30, int(0.002 * h * w))
        max_area = max(min_area + 1, int(0.92 * h * w))

        thresholds = [
            max(0.30, seed_val * 0.98, mean_val + 0.8 * std_val),
            max(0.24, seed_val * 0.90, mean_val + 0.5 * std_val),
            max(0.19, seed_val * 0.80, mean_val + 0.2 * std_val),
            max(0.15, seed_val * 0.70, mean_val),
            max(0.12, seed_val * 0.60),
            max(0.08, mean_val * 0.8),
        ]

        for thr in thresholds:
            binary = (saliency >= thr).astype(np.uint8) * 255
            kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # Suaviza ruido interno pero mantiene borde principal.
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
            if num_labels <= 1:
                continue

            seed_label = int(labels[sy, sx])
            if seed_label > 0 and min_area <= stats[seed_label, cv2.CC_STAT_AREA] <= max_area:
                comp = (labels == seed_label).astype(np.uint8) * 255
                cnt = ManualContourWorker._single_external_contour(comp)
                if cnt is not None and len(cnt) >= 3:
                    return cnt

            # Si la semilla cae fuera, elegir componente más cercana al click
            best_label = -1
            best_d2 = float("inf")
            for lbl in range(1, num_labels):
                area = int(stats[lbl, cv2.CC_STAT_AREA])
                if area < min_area or area > max_area:
                    continue
                cx, cy = centroids[lbl]
                d2 = (cx - sx) ** 2 + (cy - sy) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_label = lbl

            if best_label > 0:
                comp = (labels == best_label).astype(np.uint8) * 255
                cnt = ManualContourWorker._single_external_contour(comp)
                if cnt is not None and len(cnt) >= 3:
                    return cnt

        return None

    @staticmethod
    def _single_external_contour(mask: np.ndarray) -> Optional[np.ndarray]:
        """Devuelve un único contorno externo y uniforme de una máscara."""
        if mask is None or mask.size == 0:
            return None

        filled = ManualContourWorker._fill_holes(mask)
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        best = max(contours, key=lambda c: abs(ManualContourWorker._green_area(c.reshape(-1, 2))))
        best = best.reshape(-1, 2).astype(np.int32)
        # Suavizado leve para evitar contorno "dienteado" interno
        eps = max(1.0, 0.0025 * cv2.arcLength(best.reshape(-1, 1, 2), True))
        approx = cv2.approxPolyDP(best.reshape(-1, 1, 2), eps, True)
        return approx.reshape(-1, 2).astype(np.int32)

    @staticmethod
    def _fill_holes(mask: np.ndarray) -> np.ndarray:
        """Rellena huecos internos para obtener máscara uniforme."""
        m = mask.copy().astype(np.uint8)
        h, w = m.shape[:2]
        ff = m.copy()
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(ff, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(ff)
        return cv2.bitwise_or(m, holes)

    @staticmethod
    def _green_area(contour_xy: np.ndarray) -> float:
        """
        Área orientada por Teorema de Green (shoelace):
        A = 1/2 ∮ (x dy - y dx)
        """
        if contour_xy is None or len(contour_xy) < 3:
            return 0.0
        pts = contour_xy.astype(np.float64)
        x = pts[:, 0]
        y = pts[:, 1]
        x2 = np.roll(x, -1)
        y2 = np.roll(y, -1)
        return 0.5 * float(np.sum(x * y2 - x2 * y))


# Colores para contornos de granos (cíclicos)
GRAIN_COLORS = [
    (66, 133, 244),    # Azul
    (52, 168, 83),     # Verde
    (234, 67, 53),     # Rojo
    (251, 188, 4),     # Amarillo
    (171, 71, 188),    # Púrpura
    (0, 172, 193),     # Cyan
    (255, 112, 67),    # Naranja
    (63, 81, 181),     # Índigo
]


def numpy_to_qpixmap(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> QPixmap:
    """Convierte imagen numpy (BGR o RGB) a QPixmap escalado."""
    if image is None or image.size == 0:
        return QPixmap()

    if len(image.shape) == 2:
        h, w = image.shape
        qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = image.shape
        if ch == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            qimg = QImage(image.data, w, h, ch * w, QImage.Format_RGBA8888)

    pixmap = QPixmap.fromImage(qimg)

    if pixmap.width() > max_width or pixmap.height() > max_height:
        pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return pixmap


class AnnotationWorker(QThread):
    """Worker thread para generar segmentaciones sin bloquear la GUI."""
    progress = pyqtSignal(int, int, str)   # current, total, message
    finished = pyqtSignal(dict)             # stats
    error = pyqtSignal(str)

    def __init__(self, root_dir, overwrite, config, class_filter=None):
        super().__init__()
        self.root_dir = root_dir
        self.overwrite = overwrite
        self.config = config
        self.class_filter = class_filter

    def run(self):
        try:
            from src.grain_detection.annotator import SegmentationAnnotator
            annotator = SegmentationAnnotator(config=self.config)

            def on_progress(current, total, msg):
                self.progress.emit(current, total, msg)

            stats = annotator.annotate_directory(
                self.root_dir,
                overwrite=self.overwrite,
                progress_callback=on_progress,
                class_filter=self.class_filter,
                annotation_root="data/annotations"  # Sistema centralizado
            )
            # Añadir info de ubicación para feedback en GUI
            stats["annotation_root"] = "data/annotations"
            self.finished.emit(stats)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class AllSplitsAnnotationWorker(QThread):
    """Worker que segmenta train + val + test secuencialmente con un solo U²-Net."""
    progress = pyqtSignal(int, int, str)   # current, total, message
    finished = pyqtSignal(dict)             # combined stats
    error = pyqtSignal(str)

    SPLITS = ["data/processed/train", "data/processed/val", "data/processed/test"]

    def __init__(self, overwrite, config):
        super().__init__()
        self.overwrite = overwrite
        self.config = config

    def run(self):
        try:
            from src.grain_detection.annotator import SegmentationAnnotator
            annotator = SegmentationAnnotator(config=self.config)

            combined = {
                "total_images": 0, "annotated": 0, "skipped": 0,
                "errors": 0, "total_grains": 0, "grains_per_class": {},
                "per_split": {},
            }

            for split_dir in self.SPLITS:
                if not Path(split_dir).exists():
                    continue

                split_name = Path(split_dir).name  # train, val, test

                def on_progress(current, total, msg, _sn=split_name):
                    self.progress.emit(current, total, f"[{_sn}] {msg}")

                stats = annotator.annotate_directory(
                    split_dir,
                    overwrite=self.overwrite,
                    progress_callback=on_progress,
                    annotation_root="data/annotations"  # Sistema centralizado
                )
                stats["annotation_root"] = "data/annotations"
                combined["per_split"][split_name] = stats
                combined["total_images"] += stats.get("total_images", 0)
                combined["annotated"] += stats.get("annotated", 0)
                combined["skipped"] += stats.get("skipped", 0)
                combined["errors"] += stats.get("errors", 0)
                combined["total_grains"] += stats.get("total_grains", 0)
                for cls, cnt in stats.get("grains_per_class", {}).items():
                    combined["grains_per_class"][cls] = combined["grains_per_class"].get(cls, 0) + cnt

            self.finished.emit(combined)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class ContourAnalysisTab(QWidget):
    """
    Tab de análisis de contornos y segmentación de granos.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._worker = None

        # Estado del explorador
        self._current_class = None
        self._current_images = []
        self._current_image_idx = 0
        self._current_grains = []
        self._selected_grain_idx = -1
        self._current_image_bgr = None
        self._seg_dirty = False  # True si se editó el .seg actual sin guardar
        self._current_seg_path = None  # Ruta del .seg cargado
        
        # Historial de cambios para UNDO (stack de estados de grains)
        self._grains_history = []
        self._active_grain_idx = -1  # Grano activo para fusión incremental

        # Estado de modo de edición
        self._edit_mode = "auto"  # auto | manual_roi

        # ROI manual (puntos ilimitados include/exclude)
        self._roi_include_polygons = []
        self._roi_exclude_polygons = []
        self._roi_current_include = []
        self._roi_current_exclude = []
        self._manual_roi_seed_kind = None  # include | exclude
        self._manual_roi_seed_point = None  # (x, y)
        self._roi_include_mask = None  # acumulación unificada en tiempo real
        self._roi_exclude_mask = None  # acumulación unificada en tiempo real
        self._roi_block_mask = None  # anclas de exclusión (no expansión)
        self._roi_block_points = []  # puntos rojos ancla
        self._roi_last_include_seed = None  # última semilla de inclusión (x, y)
        self._roi_suggest_points = []  # sugerencias de expansión (puntos azules)
        self._seed_retry_depth = 0
        self._pending_ai_seed = None  # (x, y, kind)
        self._last_include_drag_pt = None
        self._last_exclude_drag_pt = None
        self._drag_step_px = 9

        self._init_ui()
        self.setFocusPolicy(Qt.StrongFocus)
        self._setup_shortcuts()

        # Auto-cargar .seg existentes al iniciar (deferred para no bloquear)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(500, self._refresh_explorer)
        logger.info("[ContourAnalysisTab] Inicializado")

    def keyPressEvent(self, event):
        key = event.key()
        if self._is_manual_roi_mode():
            if key in (Qt.Key_Return, Qt.Key_Enter):
                # Enter cierra polígono activo; si no hay, aplica ROI.
                if self._roi_current_include or self._roi_current_exclude:
                    self._close_active_manual_polygon()
                else:
                    self._apply_manual_roi_to_grain()
                return
            if key == Qt.Key_C:
                self._close_active_manual_polygon()
                return
            if key == Qt.Key_A:
                self._apply_manual_roi_to_grain()
                return
            if key == Qt.Key_Escape:
                self._cancel_manual_roi_session()
                return

        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self._seg_dirty:
                self._on_save_seg()
        elif key == Qt.Key_Left:
            self._on_prev_image()
        elif key == Qt.Key_Right:
            self._on_next_image()
        else:
            super().keyPressEvent(event)

    def _setup_shortcuts(self):
        """Atajos rápidos de ROI manual (fase 2)."""
        self._shortcut_close_roi = QShortcut(QKeySequence("C"), self)
        self._shortcut_close_roi.activated.connect(self._close_active_manual_polygon)

        self._shortcut_apply_roi = QShortcut(QKeySequence("A"), self)
        self._shortcut_apply_roi.activated.connect(self._apply_manual_roi_to_grain)

        self._shortcut_cancel_roi = QShortcut(QKeySequence("Esc"), self)
        self._shortcut_cancel_roi.activated.connect(self._cancel_manual_roi_session)

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        header = QLabel("Análisis de Contornos")
        header.setStyleSheet(Styles.HEADER_SECTION)
        layout.addWidget(header)

        # Splitter vertical: Config arriba, Explorador abajo
        splitter = QSplitter(Qt.Vertical)

        # === Sección 1: Configuración ===
        config_widget = self._build_config_section()
        splitter.addWidget(config_widget)

        # === Sección 2: Explorador de Contornos ===
        explorer_widget = self._build_explorer_section()
        splitter.addWidget(explorer_widget)

        splitter.setSizes([250, 550])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)

        # === Sección 3: Estadísticas (barra inferior) ===
        stats_widget = self._build_stats_section()
        layout.addWidget(stats_widget)

        self.setLayout(layout)
        self._on_edit_mode_changed(self.edit_mode_combo.currentIndex())

    # =========================================================================
    # Sección 1: Configuración de Segmentación
    # =========================================================================

    def _build_config_section(self) -> QWidget:
        group = QGroupBox("Configuración de Segmentación")
        main_layout = QVBoxLayout()

        # Fila 1: Dataset selector
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["data/processed/train", "data/processed/val", "data/processed/test"])
        self.dataset_combo.setMinimumWidth(250)
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        row1.addWidget(self.dataset_combo)

        row1.addWidget(QLabel("Modelo:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["u2netp (rápido, 4.7MB)", "u2net (preciso, 176MB)"])
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setToolTip("u2netp recomendado. u2net requiere descargar 176MB adicionales.")
        row1.addWidget(self.model_combo)
        row1.addStretch()
        main_layout.addLayout(row1)

        # Fila 2: Parámetros — 3 columnas
        params_layout = QHBoxLayout()

        # Columna 1: Saliency
        col1 = QFormLayout()
        self.threshold_slider = self._make_slider(5, 80, 30, "Threshold")
        col1.addRow("Threshold:", self.threshold_slider)

        self.adaptive_k_slider = self._make_slider(5, 80, 30, "Adaptive k")
        col1.addRow("Adaptive k:", self.adaptive_k_slider)

        params_layout.addLayout(col1)

        # Columna 2: Área y forma
        col2 = QFormLayout()
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(10, 50000)
        self.min_area_spin.setValue(200)
        self.min_area_spin.setSuffix(" px²")
        self.min_area_spin.setToolTip("Área mínima de un grano (en pixeles²)")
        col2.addRow("Min área:", self.min_area_spin)

        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1000, 2000000)
        self.max_area_spin.setValue(200000)
        self.max_area_spin.setSuffix(" px²")
        self.max_area_spin.setSingleStep(10000)
        self.max_area_spin.setToolTip("Área máxima de un grano. Subir para granos grandes.")
        col2.addRow("Max área:", self.max_area_spin)

        self.circularity_slider = self._make_slider(0, 80, 30, "Circularidad")
        self.circularity_slider.setToolTip("Mín circularidad (0=cualquier forma, 0.80=casi circular)")
        col2.addRow("Min circ.:", self.circularity_slider)

        params_layout.addLayout(col2)

        # Columna 3: Morfología y crop
        col3 = QFormLayout()
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 15)
        self.kernel_spin.setValue(3)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setToolTip("Tamaño kernel morfológico (impar)")
        col3.addRow("Kernel morf.:", self.kernel_spin)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 80)
        self.padding_spin.setValue(20)
        self.padding_spin.setSuffix(" %")
        self.padding_spin.setToolTip("Padding alrededor del bbox para el crop")
        col3.addRow("Crop padding:", self.padding_spin)

        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setRange(64, 512)
        self.crop_size_spin.setValue(256)
        self.crop_size_spin.setSingleStep(64)
        self.crop_size_spin.setSuffix(" px")
        self.crop_size_spin.setToolTip("Tamaño del crop cuadrado de salida")
        col3.addRow("Crop size:", self.crop_size_spin)

        params_layout.addLayout(col3)
        main_layout.addLayout(params_layout)

        # Fila 3: Botones de acción
        buttons_layout = QHBoxLayout()

        self.btn_load = QPushButton("Cargar Existentes")
        self.btn_load.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_load.setToolTip("Carga los .seg existentes sin ejecutar el modelo")
        self.btn_load.clicked.connect(self._refresh_explorer)
        buttons_layout.addWidget(self.btn_load)

        self.btn_generate = QPushButton("Generar Faltantes")
        self.btn_generate.setStyleSheet(Styles.BUTTON_SECONDARY)
        self.btn_generate.setToolTip("Genera .seg solo para imágenes sin anotación")
        self.btn_generate.clicked.connect(self._on_generate)
        buttons_layout.addWidget(self.btn_generate)

        self.btn_regenerate = QPushButton("Regenerar Todo")
        self.btn_regenerate.setStyleSheet(Styles.BUTTON_SECONDARY)
        self.btn_regenerate.clicked.connect(self._on_regenerate)
        buttons_layout.addWidget(self.btn_regenerate)

        self.btn_validate = QPushButton("Validar")
        self.btn_validate.setStyleSheet(Styles.BUTTON_SECONDARY)
        self.btn_validate.clicked.connect(self._on_validate)
        buttons_layout.addWidget(self.btn_validate)

        self.btn_save_registry = QPushButton("Guardar Registros")
        self.btn_save_registry.setStyleSheet(
            "QPushButton { background: #8e44ad; color: white; padding: 6px 14px; "
            "border-radius: 4px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background: #9b59b6; }"
        )
        self.btn_save_registry.setToolTip(
            "Guarda un registro completo de todas las segmentaciones actuales.\n"
            "Permite restaurar el estado sin volver a escanear ni ejecutar U\u00b2-Net."
        )
        self.btn_save_registry.clicked.connect(self._on_save_registry)
        buttons_layout.addWidget(self.btn_save_registry)

        self.btn_segment_all = QPushButton("Segmentar TODOS los Splits")
        self.btn_segment_all.setStyleSheet(
            "QPushButton { background: #e67e22; color: white; padding: 6px 14px; "
            "border-radius: 4px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background: #f39c12; }"
        )
        self.btn_segment_all.setToolTip(
            "Genera .seg para train, val y test de una sola vez.\n"
            "Requisito obligatorio antes de entrenar en modo grain."
        )
        self.btn_segment_all.clicked.connect(self._on_segment_all_splits)
        buttons_layout.addWidget(self.btn_segment_all)

        main_layout.addLayout(buttons_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(Styles.LABEL_INFO)
        main_layout.addWidget(self.status_label)

        group.setLayout(main_layout)
        return group

    def _make_slider(self, min_val, max_val, default, name) -> QWidget:
        """Crea un widget slider + label de valor."""
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setMinimumWidth(120)

        value_label = QLabel(f"{default / 100:.2f}")
        value_label.setMinimumWidth(40)
        slider.valueChanged.connect(lambda v: value_label.setText(f"{v / 100:.2f}"))

        layout.addWidget(slider)
        layout.addWidget(value_label)
        container.setLayout(layout)

        container._slider = slider
        container._label = value_label
        return container

    # =========================================================================
    # Sección 2: Explorador de Contornos
    # =========================================================================

    def _build_explorer_section(self) -> QWidget:
        group = QGroupBox("Explorador de Contornos")
        main_layout = QVBoxLayout()

        # ===== Layout 3 paneles: Clases (izq) | Imagen (centro) | Grano (der) =====
        view_splitter = QSplitter(Qt.Horizontal)

        # ─── Panel izquierdo: Selector de clases (checkable) ───
        class_panel = QWidget()
        class_layout = QVBoxLayout()
        class_layout.setContentsMargins(2, 2, 2, 2)
        class_layout.setSpacing(4)

        class_header = QLabel("Clases")
        class_header.setStyleSheet("font-weight: bold; font-size: 12px;")
        class_layout.addWidget(class_header)

        # Botones seleccionar todo / ninguno
        class_btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("Todas")
        self.btn_select_all.setMaximumHeight(24)
        self.btn_select_all.clicked.connect(self._on_select_all_classes)
        class_btn_layout.addWidget(self.btn_select_all)

        self.btn_select_none = QPushButton("Ninguna")
        self.btn_select_none.setMaximumHeight(24)
        self.btn_select_none.clicked.connect(self._on_select_no_classes)
        class_btn_layout.addWidget(self.btn_select_none)

        self.btn_apply_classes = QPushButton("Aplicar")
        self.btn_apply_classes.setMaximumHeight(24)
        self.btn_apply_classes.setStyleSheet(
            "QPushButton { background: #2980b9; color: white; font-weight: bold; border-radius: 3px; }"
            "QPushButton:hover { background: #3498db; }"
        )
        self.btn_apply_classes.setToolTip("Filtra el explorador por las clases seleccionadas")
        self.btn_apply_classes.clicked.connect(self._on_apply_class_filter)
        class_btn_layout.addWidget(self.btn_apply_classes)
        class_layout.addLayout(class_btn_layout)

        # Lista checkable de clases
        self.class_list = QListWidget()
        self.class_list.setMinimumWidth(160)
        self.class_list.setMaximumWidth(220)
        self.class_list.setStyleSheet(
            f"QListWidget {{ background: {Colors.BG_WIDGET}; border: 1px solid {Colors.BORDER_LIGHT}; }}"
            f"QListWidget::item {{ padding: 3px; }}"
        )
        class_layout.addWidget(self.class_list)

        # Contador de selección
        self.class_count_label = QLabel("0 / 0 clases")
        self.class_count_label.setStyleSheet("color: #888; font-size: 11px;")
        class_layout.addWidget(self.class_count_label)

        class_panel.setLayout(class_layout)
        view_splitter.addWidget(class_panel)

        # ─── Panel central: Imagen + navegación ───
        center_panel = QWidget()
        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(2, 2, 2, 2)

        # Barra de navegación de imágenes
        nav_layout = QHBoxLayout()

        nav_layout.addWidget(QLabel("Imagen:"))
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(220)
        self.image_combo.currentTextChanged.connect(self._on_image_changed)
        nav_layout.addWidget(self.image_combo)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setMaximumWidth(32)
        self.btn_prev.clicked.connect(self._on_prev_image)
        nav_layout.addWidget(self.btn_prev)

        self.nav_label = QLabel("0 / 0")
        self.nav_label.setMinimumWidth(70)
        self.nav_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.nav_label)

        self.btn_next = QPushButton("▶")
        self.btn_next.setMaximumWidth(32)
        self.btn_next.clicked.connect(self._on_next_image)
        nav_layout.addWidget(self.btn_next)

        nav_layout.addStretch()
        center_layout.addLayout(nav_layout)

        # Imagen con contornos (clickable for manual contour interaction)
        self.image_label = ClickableImageLabel("Selecciona clases y pulsa Aplicar")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet(f"border: 1px solid {Colors.BORDER_LIGHT}; background: {Colors.BG_WIDGET};")
        self.image_label.left_clicked.connect(self._on_image_left_click)
        self.image_label.right_clicked.connect(self._on_image_right_click)
        self.image_label.left_double_clicked.connect(self._on_image_left_double_click)
        self.image_label.right_double_clicked.connect(self._on_image_right_double_click)
        self.image_label.left_dragged.connect(self._on_image_left_drag)
        self.image_label.right_dragged.connect(self._on_image_right_drag)
        self.image_label.left_released.connect(self._on_image_left_release)
        self.image_label.right_released.connect(self._on_image_right_release)
        self.image_label.wheel_up.connect(self._on_prev_image)
        self.image_label.wheel_down.connect(self._on_next_image)
        center_layout.addWidget(self.image_label)

        # Info de granos
        self.grain_info_label = QLabel("")
        self.grain_info_label.setStyleSheet(Styles.LABEL_INFO)
        self.grain_info_label.setWordWrap(True)
        center_layout.addWidget(self.grain_info_label)

        center_panel.setLayout(center_layout)
        view_splitter.addWidget(center_panel)

        # Vista derecha: Crop del grano seleccionado + metadata
        right_container = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(2, 2, 2, 2)

        right_layout.addWidget(QLabel("Grano seleccionado:"))

        self.crop_label = QLabel("Click en un grano")
        self.crop_label.setAlignment(Qt.AlignCenter)
        self.crop_label.setMinimumSize(256, 256)
        self.crop_label.setMaximumSize(300, 300)
        self.crop_label.setStyleSheet(f"border: 1px solid {Colors.BORDER_LIGHT}; background: {Colors.BG_WIDGET};")
        right_layout.addWidget(self.crop_label)

        # Metadata del grano
        self.grain_meta_label = QLabel("")
        self.grain_meta_label.setStyleSheet(Styles.LABEL_INFO)
        self.grain_meta_label.setWordWrap(True)
        right_layout.addWidget(self.grain_meta_label)

        # Selector de grano
        grain_nav = QHBoxLayout()
        self.btn_prev_grain = QPushButton("< Grano")
        self.btn_prev_grain.clicked.connect(self._on_prev_grain)
        grain_nav.addWidget(self.btn_prev_grain)

        self.grain_nav_label = QLabel("—")
        self.grain_nav_label.setAlignment(Qt.AlignCenter)
        grain_nav.addWidget(self.grain_nav_label)

        self.btn_next_grain = QPushButton("Grano >")
        self.btn_next_grain.clicked.connect(self._on_next_grain)
        grain_nav.addWidget(self.btn_next_grain)
        right_layout.addLayout(grain_nav)

        # --- Botones de edición ---
        edit_layout = QHBoxLayout()

        self.btn_delete_grain = QPushButton("Eliminar grano")
        self.btn_delete_grain.setStyleSheet(
            f"QPushButton {{ background: #c0392b; color: white; padding: 4px 10px; "
            f"border-radius: 3px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #e74c3c; }}"
        )
        self.btn_delete_grain.setToolTip("Elimina el grano seleccionado de la anotación")
        self.btn_delete_grain.clicked.connect(self._on_delete_grain)
        edit_layout.addWidget(self.btn_delete_grain)

        self.btn_save_seg = QPushButton("Guardar .seg")
        self.btn_save_seg.setStyleSheet(
            f"QPushButton {{ background: #27ae60; color: white; padding: 4px 10px; "
            f"border-radius: 3px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #2ecc71; }}"
            f"QPushButton:disabled {{ background: #7f8c8d; }}"
        )
        self.btn_save_seg.setToolTip("Guarda los cambios en el archivo .seg")
        self.btn_save_seg.setEnabled(False)
        self.btn_save_seg.clicked.connect(self._on_save_seg)
        edit_layout.addWidget(self.btn_save_seg)

        self.edit_status_label = QLabel("")
        self.edit_status_label.setStyleSheet("color: #888; font-size: 11px;")
        edit_layout.addWidget(self.edit_status_label)

        edit_layout.addStretch()
        right_layout.addLayout(edit_layout)

        # --- Edición ROI Manual (fase 1 + fase 2) ---
        manual_group = QGroupBox("ROI Manual")
        manual_layout = QVBoxLayout()
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(4)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Modo:"))
        self.edit_mode_combo = QComboBox()
        self.edit_mode_combo.addItems([
            "Auto asistido (U2NET)",
            "ROI manual (incluir/excluir)",
        ])
        self.edit_mode_combo.currentIndexChanged.connect(self._on_edit_mode_changed)
        mode_layout.addWidget(self.edit_mode_combo)
        manual_layout.addLayout(mode_layout)

        self.roi_snap_edges_cb = QCheckBox("Snap a borde (opcional)")
        self.roi_snap_edges_cb.setChecked(True)
        self.roi_snap_edges_cb.setToolTip("Ajusta cada click al borde más cercano detectado por Canny.")
        manual_layout.addWidget(self.roi_snap_edges_cb)

        self.roi_ai_assist_cb = QCheckBox("Asistencia AI por click (U2NET)")
        self.roi_ai_assist_cb.setChecked(True)
        self.roi_ai_assist_cb.setToolTip(
            "Cada click se usa como semilla para proponer automáticamente una región local con U2NET."
        )
        manual_layout.addWidget(self.roi_ai_assist_cb)

        self.roi_multi_component_cb = QCheckBox("Crear múltiples instancias desde ROI")
        self.roi_multi_component_cb.setChecked(False)
        self.roi_multi_component_cb.setToolTip(
            "Si ROI final tiene varias islas, crea múltiples granos. "
            "Desactivar para usar solo el contorno mayor."
        )
        manual_layout.addWidget(self.roi_multi_component_cb)

        close_row = QHBoxLayout()
        self.btn_close_include = QPushButton("Cerrar inclusión")
        self.btn_close_include.clicked.connect(
            lambda: self._close_manual_polygon("include")
        )
        close_row.addWidget(self.btn_close_include)

        self.btn_close_exclude = QPushButton("Cerrar exclusión")
        self.btn_close_exclude.clicked.connect(
            lambda: self._close_manual_polygon("exclude")
        )
        close_row.addWidget(self.btn_close_exclude)
        manual_layout.addLayout(close_row)

        action_row = QHBoxLayout()
        self.btn_apply_roi = QPushButton("Aplicar ROI")
        self.btn_apply_roi.clicked.connect(self._apply_manual_roi_to_grain)
        action_row.addWidget(self.btn_apply_roi)

        self.btn_clear_roi = QPushButton("Limpiar ROI")
        self.btn_clear_roi.clicked.connect(self._clear_manual_roi_points)
        action_row.addWidget(self.btn_clear_roi)

        self.btn_cancel_roi = QPushButton("Cancelar")
        self.btn_cancel_roi.clicked.connect(self._cancel_manual_roi_session)
        action_row.addWidget(self.btn_cancel_roi)
        manual_layout.addLayout(action_row)

        self.roi_hint_label = QLabel(
            "Izq: incluir (semilla AI U2NET)  |  Der: ancla roja (bloquea expansión)\n"
            "Si desactivas asistencia AI: clicks agregan puntos manuales include/exclude.\n"
            "Atajos: C cerrar, A aplicar, Esc cancelar"
        )
        self.roi_hint_label.setStyleSheet("color: #888; font-size: 11px;")
        self.roi_hint_label.setWordWrap(True)
        manual_layout.addWidget(self.roi_hint_label)

        manual_group.setLayout(manual_layout)
        right_layout.addWidget(manual_group)

        right_layout.addStretch()
        right_container.setLayout(right_layout)
        view_splitter.addWidget(right_container)

        view_splitter.setSizes([180, 500, 280])
        main_layout.addWidget(view_splitter)

        group.setLayout(main_layout)
        return group

    # =========================================================================
    # Sección 3: Estadísticas
    # =========================================================================

    def _build_stats_section(self) -> QWidget:
        group = QGroupBox("Estadísticas")
        layout = QHBoxLayout()

        self.stats_label = QLabel("Sin datos — genera segmentaciones primero.")
        self.stats_label.setStyleSheet(Styles.LABEL_INFO)
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        group.setLayout(layout)
        group.setMaximumHeight(80)
        return group

    # =========================================================================
    # Acciones
    # =========================================================================

    def _get_model_type(self) -> str:
        """Retorna el model_type seleccionado."""
        return "u2netp" if "u2netp" in self.model_combo.currentText() else "u2net"

    def _check_weights_exist(self) -> bool:
        """Verifica que los pesos del modelo seleccionado existen."""
        from pathlib import Path
        model_type = self._get_model_type()
        weights_dir = Path(__file__).resolve().parent.parent / "models" / "weights"
        weights_path = weights_dir / f"{model_type}.pth"
        if not weights_path.exists():
            QMessageBox.critical(
                self, "Pesos no encontrados",
                f"No se encontraron pesos para {model_type} en:\n{weights_path}\n\n"
                f"Opciones:\n"
                f"1. Selecciona 'u2netp' (recomendado, 4.7MB)\n"
                f"2. Ejecuta: python setup_u2net.py --model {model_type}\n"
                f"3. Copia {model_type}.pth desde XYZ_Ctrl_L206_GUI"
            )
            return False
        return True

    def _get_config(self) -> Dict:
        """Recopila parámetros de la UI como dict de configuración."""
        return {
            "model_type": self._get_model_type(),
            "saliency_threshold": self.threshold_slider._slider.value() / 100.0,
            "adaptive_k": self.adaptive_k_slider._slider.value() / 100.0,
            "min_area": self.min_area_spin.value(),
            "max_area": self.max_area_spin.value(),
            "min_circularity": self.circularity_slider._slider.value() / 100.0,
            "morph_kernel_size": self.kernel_spin.value(),
            "crop_padding": self.padding_spin.value() / 100.0,
            "crop_size": self.crop_size_spin.value(),
        }

    def _on_dataset_changed(self, text):
        """Cuando cambia el dataset seleccionado, auto-cargar .seg existentes."""
        self._refresh_explorer()

    def _on_segment_all_splits(self):
        """Segmenta train + val + test secuencialmente con un solo U²-Net."""
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Ocupado", "Ya hay una segmentación en progreso.")
            return

        if not self._check_weights_exist():
            return

        # Verificar que exista al menos train
        if not Path("data/processed/train").exists():
            QMessageBox.warning(self, "Error",
                "No se encontró data/processed/train.\n"
                "Primero prepara el dataset en 'Gestor de Datos'.")
            return

        config = self._get_config()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_generate.setEnabled(False)
        self.btn_regenerate.setEnabled(False)
        self.btn_segment_all.setEnabled(False)
        self.status_label.setText("Segmentando train + val + test...")

        self._worker = AllSplitsAnnotationWorker(overwrite=False, config=config)
        self._worker.progress.connect(self._on_annotation_progress)
        self._worker.finished.connect(self._on_all_splits_finished)
        self._worker.error.connect(self._on_annotation_error)
        self._worker.start()

    def _on_all_splits_finished(self, combined_stats):
        """Callback cuando termina la segmentación de todos los splits."""
        self.progress_bar.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.btn_regenerate.setEnabled(True)
        self.btn_segment_all.setEnabled(True)

        per_split = combined_stats.get("per_split", {})
        parts = []
        for split_name in ["train", "val", "test"]:
            s = per_split.get(split_name, {})
            grains = s.get("total_grains", 0)
            annotated = s.get("annotated", 0)
            skipped = s.get("skipped", 0)
            parts.append(f"{split_name}: {grains} granos ({annotated} nuevos, {skipped} existentes)")

        self.status_label.setText(" | ".join(parts))

        if self.parent_window and hasattr(self.parent_window, 'log_widget'):
            total_grains = combined_stats.get("total_grains", 0)
            self.parent_window.log_widget.append_log(
                f"Segmentación completada (todos los splits): {total_grains} granos totales",
                "SUCCESS"
            )

        # Auto-guardar registro del split actual ANTES de refresh
        self._save_registry_silent()
        self._refresh_explorer(expand_checked=True)

    def _on_generate(self):
        """Genera .seg para el dataset seleccionado (skip existentes)."""
        if not self._check_weights_exist():
            return
        self._run_annotation(overwrite=False)

    def _on_regenerate(self):
        """Regenera TODOS los .seg del dataset seleccionado."""
        if not self._check_weights_exist():
            return
        reply = QMessageBox.question(
            self, "Confirmar",
            "¿Regenerar TODAS las segmentaciones?\nLos .seg existentes serán sobrescritos.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._run_annotation(overwrite=True)

    def _run_annotation(self, overwrite: bool):
        """Ejecuta la anotación en un thread separado (solo clases seleccionadas)."""
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Ocupado", "Ya hay una segmentación en progreso.")
            return

        root_dir = self.dataset_combo.currentText()
        if not Path(root_dir).exists():
            QMessageBox.warning(self, "Error", f"Directorio no encontrado: {root_dir}")
            return

        # Filtrar por clases seleccionadas
        selected = self._get_checked_classes()
        if not selected:
            QMessageBox.warning(self, "Sin clases", "Selecciona al menos una clase para anotar.")
            return

        config = self._get_config()

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.btn_generate.setEnabled(False)
        self.btn_regenerate.setEnabled(False)
        n_cls = len(selected)
        self.status_label.setText(f"Iniciando segmentación ({n_cls} clase{'s' if n_cls != 1 else ''})...")

        self._worker = AnnotationWorker(root_dir, overwrite, config, class_filter=selected)
        self._worker.progress.connect(self._on_annotation_progress)
        self._worker.finished.connect(self._on_annotation_finished)
        self._worker.error.connect(self._on_annotation_error)
        self._worker.start()

    def _on_annotation_progress(self, current, total, msg):
        pct = int(current / max(1, total) * 100)
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"[{current}/{total}] {msg}")

    def _on_annotation_finished(self, stats):
        try:
            self.progress_bar.setVisible(False)
            self.btn_generate.setEnabled(True)
            self.btn_regenerate.setEnabled(True)

            self.status_label.setText(
                f"Completado: {stats.get('annotated', 0)} anotados, "
                f"{stats.get('skipped', 0)} omitidos, "
                f"{stats.get('total_grains', 0)} granos totales "
                f"({stats.get('elapsed_ms', 0):.0f}ms)"
            )

            self._update_stats(stats)

            # Auto-guardar registro ANTES de refresh (para que fast path lo encuentre)
            self._save_registry_silent(annotation_stats=stats)

            # Refresh con expand_checked=True para mostrar clases recién anotadas
            self._refresh_explorer(expand_checked=True)
            
            # Recargar visor para mostrar las imágenes con .seg recién creados
            self._on_apply_class_filter()
            logger.info("[Explorer] Visor actualizado con .seg recién generados")

            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                annot_location = stats.get('annotation_root', 'junto a imágenes')
                self.parent_window.log_widget.append_log(
                    f"✅ Segmentación completada\n\n"
                    f"📁 Archivos .seg guardados en: {annot_location}/\n\n"
                    f"Imágenes procesadas: {stats['total_images']}\n"
                    f"Anotadas: {stats['annotated']}\n"
                    f"Omitidas: {stats['skipped']}\n"
                    f"Errores: {stats['errors']}\n\n"
                    f"Granos detectados: {stats['total_grains']}\n"
                    f"Promedio: {stats.get('grains_per_image', 0):.1f} granos/imagen\n\n"
                    f"Desglose por clase:\n",
                    "SUCCESS"
                )
        except Exception as e:
            import traceback
            logger.error(f"[Annotation] Error en callback finished: {e}\n{traceback.format_exc()}")
            self.status_label.setText(f"Error post-anotación: {e}")

    def _on_annotation_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.btn_generate.setEnabled(True)
        self.btn_regenerate.setEnabled(True)
        self.status_label.setText(f"Error: {error_msg[:100]}")
        QMessageBox.critical(self, "Error", f"Error en segmentación:\n{error_msg[:500]}")

    def _on_save_registry(self):
        """Guarda registro completo de segmentaciones actuales (botón manual)."""
        root_dir = self.dataset_combo.currentText()
        if not Path(root_dir).exists():
            QMessageBox.warning(self, "Error", f"Directorio no encontrado: {root_dir}")
            return

        try:
            from src.grain_detection.contour_registry import ContourRegistry
            config = self._get_config()
            reg_path = ContourRegistry.save(
                root_dir, 
                detection_params=config,
                annotation_root="data/annotations"  # Sistema centralizado
            )
            registry = ContourRegistry.load(root_dir)

            summary = registry["summary"]
            self.status_label.setText(
                f"Registros guardados: {summary['images_with_seg']} imgs con .seg, "
                f"{summary['total_grains']} granos — {reg_path.name}"
            )

            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                annot_location = registry.get('annotation_root', 'junto a imágenes')
                self.parent_window.log_widget.append_log(
                    f"Registro guardado: {reg_path} "
                    f"({summary['images_with_seg']} imgs, {summary['total_grains']} granos, "
                    f"{summary['num_classes']} clases)",
                    "SUCCESS"
                )
                self.parent_window.log_widget.append_log(
                    f"Archivos .seg guardados en: {annot_location}/", "INFO"
                )

            QMessageBox.information(
                self, "Registros Guardados",
                f"Registro guardado exitosamente en:\n{reg_path}\n\n"
                f"Imágenes: {summary['total_images']}\n"
                f"Con .seg: {summary['images_with_seg']}\n"
                f"Granos totales: {summary['total_grains']}\n"
                f"Clases: {summary['num_classes']}\n\n"
                f"Parámetros guardados. Al presionar 'Cargar Existentes'\n"
                f"se restaurará desde este registro instantáneamente."
            )
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Error guardando registro:\n{e}")
            logger.error(f"[Registry] Error guardando: {e}\n{traceback.format_exc()}")

    def _save_registry_silent(self, annotation_stats=None):
        """Guarda registro silenciosamente (sin diálogos). Usado en auto-save."""
        root_dir = self.dataset_combo.currentText()
        if not Path(root_dir).exists():
            return
        try:
            from src.grain_detection.contour_registry import ContourRegistry
            config = self._get_config()
            reg_path = ContourRegistry.save(
                root_dir, 
                detection_params=config, 
                annotation_stats=annotation_stats,
                annotation_root="data/annotations"  # Sistema centralizado
            )
            logger.info(f"[Registry] Auto-guardado: {reg_path}")
            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                self.parent_window.log_widget.append_log(
                    f"Registro auto-guardado: {reg_path.name}", "DEBUG"
                )
        except Exception as e:
            logger.warning(f"[Registry] Error en auto-save: {e}")

    def _on_validate(self):
        """Valida integridad de pares imagen/.seg (no requiere modelo U²-Net)."""
        root_dir = self.dataset_combo.currentText()
        if not Path(root_dir).exists():
            QMessageBox.warning(self, "Error", f"Directorio no encontrado: {root_dir}")
            return

        try:
            from src.grain_detection.annotator import SegmentationAnnotator
            annotator = SegmentationAnnotator()  # Lazy: no carga modelo
            
            # Usar sistema centralizado
            result = annotator.validate_annotations(
                root_dir,
                annotation_root="data/annotations"
            )
            
            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                self.parent_window.log_widget.append_log(
                    f"Validación completada: {result['valid']}/{result['total_images']} con .seg válido",
                    "INFO"
                )

            msg = (
                f"Total imágenes: {result['total_images']}\n"
                f"Con .seg válido: {result['valid']}\n"
                f"Sin .seg: {len(result['missing_seg'])}\n"
                f"Corruptos: {len(result['corrupt_seg'])}\n"
                f"Huérfanos: {len(result['orphan_seg'])}\n\n"
                f"📁 Buscando en: data/annotations/"
            )
            QMessageBox.information(self, "Validación", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error validando:\n{e}")

    # =========================================================================
    # Explorador
    # =========================================================================

    def _refresh_explorer(self, expand_checked=False):
        """Refresca la lista de clases y el explorador con el dataset actual.

        Args:
            expand_checked: If True, auto-check all classes that have .seg files
                (in addition to previously checked classes). Used after annotation
                to ensure newly annotated classes become visible.

        Fast path: Si existe contour_registry.json, carga conteos desde ahí
        (instantáneo, sin escanear archivos). Fallback: escaneo de directorio.
        """
        root_dir = self.dataset_combo.currentText()
        root = Path(root_dir)
        if not root.exists():
            self.status_label.setText(f"Directorio no encontrado: {root_dir}")
            return

        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])

        # Intentar fast path desde registro
        seg_counts = {}
        img_counts = {}
        registry_loaded = False

        try:
            from src.grain_detection.contour_registry import ContourRegistry
            registry = ContourRegistry.load(root_dir)
            if registry and not ContourRegistry.is_stale(root_dir):
                # Fast path: conteos desde el registro
                for entry in registry["images"]:
                    cls = entry["class"]
                    img_counts[cls] = img_counts.get(cls, 0) + 1
                    if entry.get("seg_path"):
                        seg_counts[cls] = seg_counts.get(cls, 0) + 1

                # Restaurar parámetros de detección del registro
                self._restore_params_from_registry(registry.get("detection_params", {}))

                summary = registry["summary"]
                self.status_label.setText(
                    f"Registro cargado: {summary['images_with_seg']} imgs con .seg, "
                    f"{summary['total_grains']} granos "
                    f"({registry['timestamp'][:19]})"
                )
                registry_loaded = True
                logger.info("[Explorer] Cargado desde contour_registry.json (fast path)")
        except Exception as e:
            logger.debug(f"[Explorer] Registry fast-path falló: {e}")

        if not registry_loaded:
            # Fallback: escaneo directo de directorio
            # Buscar .seg en data/annotations/ (sistema centralizado)
            annot_root = Path("data/annotations")
            logger.info(f"[Explorer] Escaneando directorio (fallback) - buscando .seg en {annot_root}")
            
            for cls in classes:
                cls_dir = root / cls
                imgs = [f for f in cls_dir.iterdir()
                        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}]
                
                # Buscar .seg en data/annotations/clase/
                cls_annot_dir = annot_root / cls
                segs = []
                if cls_annot_dir.exists():
                    for img_file in imgs:
                        seg_file = cls_annot_dir / f"{img_file.stem}.seg"
                        if seg_file.exists():
                            segs.append(img_file)
                
                img_counts[cls] = len(imgs)
                seg_counts[cls] = len(segs)
                logger.debug(f"[Explorer] {cls}: {len(imgs)} imgs, {len(segs)} con .seg")

        # Guardar clases previamente checkeadas
        prev_checked = self._get_checked_classes()

        # Poblar lista checkable
        self.class_list.clear()
        for cls in classes:
            n_seg = seg_counts.get(cls, 0)
            n_img = img_counts.get(cls, 0)
            item = QListWidgetItem(f"{cls}  ({n_seg}/{n_img})")
            item.setData(Qt.UserRole, cls)  # nombre limpio
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Mantener check previo o activar por default si tiene .seg
            if expand_checked:
                # After annotation: check if previously checked OR has .seg data
                item.setCheckState(Qt.Checked if (cls in prev_checked or n_seg > 0) else Qt.Unchecked)
            elif prev_checked:
                item.setCheckState(Qt.Checked if cls in prev_checked else Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked if n_seg > 0 else Qt.Unchecked)
            self.class_list.addItem(item)

        self._update_class_count_label()
        self._on_apply_class_filter()
        self._update_stats_from_directory(root_dir)

    def _restore_params_from_registry(self, params: Dict):
        """Restaura los parámetros de la GUI desde los guardados en el registro."""
        if not params:
            return
        try:
            if "saliency_threshold" in params:
                self.threshold_slider._slider.setValue(int(params["saliency_threshold"] * 100))
            if "adaptive_k" in params:
                self.adaptive_k_slider._slider.setValue(int(params["adaptive_k"] * 100))
            if "min_area" in params:
                self.min_area_spin.setValue(params["min_area"])
            if "max_area" in params:
                self.max_area_spin.setValue(params["max_area"])
            if "min_circularity" in params:
                self.circularity_slider._slider.setValue(int(params["min_circularity"] * 100))
            if "morph_kernel_size" in params:
                self.kernel_spin.setValue(params["morph_kernel_size"])
            if "crop_padding" in params:
                self.padding_spin.setValue(int(params["crop_padding"] * 100))
            if "crop_size" in params:
                self.crop_size_spin.setValue(params["crop_size"])
            if "model_type" in params:
                idx = 0 if params["model_type"] == "u2netp" else 1
                self.model_combo.setCurrentIndex(idx)
            logger.info("[Explorer] Parámetros restaurados desde registro")
        except Exception as e:
            logger.warning(f"[Explorer] Error restaurando parámetros: {e}")

    def _get_checked_classes(self) -> List[str]:
        """Retorna lista de nombres de clases checkeadas."""
        checked = []
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.data(Qt.UserRole))
        return checked

    def _update_class_count_label(self):
        """Actualiza el contador de clases seleccionadas."""
        total = self.class_list.count()
        checked = len(self._get_checked_classes())
        self.class_count_label.setText(f"{checked} / {total} clases")

    def _on_select_all_classes(self):
        """Checkea todas las clases."""
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(Qt.Checked)
        self._update_class_count_label()

    def _on_select_no_classes(self):
        """Descheckea todas las clases."""
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(Qt.Unchecked)
        self._update_class_count_label()

    def _on_apply_class_filter(self):
        """Aplica el filtro de clases: recarga imágenes de las clases seleccionadas.

        Shows ALL images from selected classes, prioritizing those with .seg files first.
        Images without .seg are marked as "[sin .seg]" for easy identification.
        """
        selected_classes = self._get_checked_classes()
        self._update_class_count_label()

        root_dir = self.dataset_combo.currentText()
        root = Path(root_dir)

        # Recopilar TODAS las imágenes de las clases seleccionadas
        # Buscar .seg en data/annotations/ (sistema centralizado)
        annot_root = Path("data/annotations")
        images_with_seg = []
        images_without_seg = []
        
        for cls in selected_classes:
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            
            cls_annot_dir = annot_root / cls
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}:
                    # Buscar .seg en data/annotations/clase/
                    seg_file = cls_annot_dir / f"{f.stem}.seg"
                    if seg_file.exists():
                        images_with_seg.append(f)
                    else:
                        images_without_seg.append(f)
        
        # Combinar: primero las que tienen .seg, luego las que no
        images = images_with_seg + images_without_seg
        
        logger.info(f"[Explorer] Encontradas {len(images)} imágenes totales "
                   f"({len(images_with_seg)} con .seg, {len(images_without_seg)} sin .seg) "
                   f"en {len(selected_classes)} clases")

        self._current_images = images
        self._current_image_idx = 0

        # Poblar combo de imágenes con clase como prefijo y estado de .seg
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        
        for img in images_with_seg:
            cls_name = img.parent.name
            self.image_combo.addItem(f"[{cls_name}] {img.stem}", str(img))
        
        for img in images_without_seg:
            cls_name = img.parent.name
            self.image_combo.addItem(f"[{cls_name}] {img.stem} [sin .seg]", str(img))
        
        self.image_combo.blockSignals(False)

        if images:
            self.image_combo.setCurrentIndex(0)
            self._load_current_image()
        else:
            n_cls = len(selected_classes)
            self.image_label.setText(
                f"Sin imágenes en {n_cls} clase{'s' if n_cls != 1 else ''} seleccionada{'s' if n_cls != 1 else ''}"
                if n_cls > 0 else "Selecciona al menos una clase"
            )
            self.grain_info_label.setText("")
            self.nav_label.setText("0 / 0")

    def _on_image_changed(self, image_name):
        """Cuando cambia la imagen seleccionada en el combo."""
        idx = self.image_combo.currentIndex()
        if idx >= 0 and idx < len(self._current_images):
            self._current_image_idx = idx
            self._load_current_image()

    def _on_prev_image(self):
        if self._current_image_idx > 0:
            self._current_image_idx -= 1
            self.image_combo.setCurrentIndex(self._current_image_idx)

    def _on_next_image(self):
        if self._current_image_idx < len(self._current_images) - 1:
            self._current_image_idx += 1
            self.image_combo.setCurrentIndex(self._current_image_idx)

    def _load_current_image(self):
        """Carga la imagen actual y dibuja contornos."""
        if not self._current_images or self._current_image_idx >= len(self._current_images):
            return

        # Si hay cambios sin guardar, preguntar
        if self._seg_dirty:
            reply = QMessageBox.question(
                self, "Cambios sin guardar",
                "Hay cambios sin guardar en el .seg actual.\n¿Guardar antes de cambiar?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save
            )
            if reply == QMessageBox.Save:
                self._on_save_seg()
            elif reply == QMessageBox.Cancel:
                return
            self._seg_dirty = False
            self.btn_save_seg.setEnabled(False)
            self.edit_status_label.setText("")

        img_path = self._current_images[self._current_image_idx]
        
        # Limpiar historial al cambiar de imagen
        self._clear_history()
        self._reset_manual_roi_session()
        
        # Buscar .seg en data/annotations/ (sistema centralizado)
        class_name = img_path.parent.name
        annot_root = Path("data/annotations")
        seg_path = annot_root / class_name / f"{img_path.stem}.seg"
        self._current_seg_path = seg_path
        
        logger.debug(f"[Explorer] Cargando {img_path.name}, buscando .seg en {seg_path}")

        # Cargar imagen
        image = cv2.imread(str(img_path))
        if image is None:
            self.image_label.setText(f"Error leyendo {img_path.name}")
            return

        self._current_image_bgr = image.copy()
        h_img, w_img = image.shape[:2]
        self.image_label.set_original_size(w_img, h_img)

        # Leer .seg (solo si existe — no es error que falte)
        if seg_path.exists():
            try:
                from src.grain_detection.seg_format import SegFileReader
                result = SegFileReader.read(str(seg_path))
                self._current_grains = result.get("grains", [])
            except Exception as e:
                self._current_grains = []
                logger.warning(f"Error leyendo {seg_path.name}: {e}")
        else:
            self._current_grains = []

        # Dibujar contornos sobre la imagen
        annotated = image.copy()
        for i, grain in enumerate(self._current_grains):
            color = GRAIN_COLORS[i % len(GRAIN_COLORS)]
            contour = grain.get("contour")
            if contour is not None and len(contour) >= 3:
                pts = contour.reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(annotated, [pts], -1, color, 2)

            # Dibujar bbox
            bx, by, bw, bh = grain["bbox"]
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 1)

            # Label
            cv2.putText(annotated, f"#{i}", (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Mostrar
        pixmap = numpy_to_qpixmap(annotated, max_width=700, max_height=500)
        self.image_label.setPixmap(pixmap)

        # Info de granos
        n = len(self._current_grains)
        active_info = f" [Grano #{self._active_grain_idx} en edición]" if self._active_grain_idx >= 0 else ""
        if self._is_manual_roi_mode():
            if self.roi_ai_assist_cb.isChecked():
                hint = (
                    "Izq: incluir (semilla AI U2NET) | "
                    "Der: ancla roja (bloquea expansión verde) | "
                    "Doble click: cerrar polígono"
                )
            else:
                hint = "Izq: incluir puntos | Der: excluir puntos | Doble click: cerrar polígono"
        else:
            hint = "Click izq: añadir/fusionar región | Click der: eliminar contorno o UNDO"
        self.grain_info_label.setText(
            f"{n} grano{'s' if n != 1 else ''}{active_info} — "
            f"{hint}"
        )
        if self._is_manual_roi_mode() and self.roi_ai_assist_cb.isChecked():
            self.grain_info_label.setText(
                self.grain_info_label.text() + " | Arrastre izq: sembrado AI continuo | Arrastre der: barrera continua"
            )

        total = len(self._current_images)
        self.nav_label.setText(f"{self._current_image_idx + 1} / {total}")

        # Seleccionar primer grano
        if self._current_grains:
            self._selected_grain_idx = 0
            self._show_selected_grain()
        else:
            self._selected_grain_idx = -1
            self.crop_label.setText("Sin granos")
            self.grain_meta_label.setText("")
            self.grain_nav_label.setText("—")

    def _on_prev_grain(self):
        if self._current_grains and self._selected_grain_idx > 0:
            self._selected_grain_idx -= 1
            self._show_selected_grain()

    def _on_next_grain(self):
        if self._current_grains and self._selected_grain_idx < len(self._current_grains) - 1:
            self._selected_grain_idx += 1
            self._show_selected_grain()

    def _show_selected_grain(self):
        """Muestra el crop del grano seleccionado."""
        if (self._current_image_bgr is None or
                self._selected_grain_idx < 0 or
                self._selected_grain_idx >= len(self._current_grains)):
            return

        grain = self._current_grains[self._selected_grain_idx]
        image = self._current_image_bgr

        # Extraer crop
        h_img, w_img = image.shape[:2]
        bx, by, bw, bh = grain["bbox"]
        padding = self.padding_spin.value() / 100.0

        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        side = max(bw + 2 * pad_x, bh + 2 * pad_y)

        cx = bx + bw // 2
        cy = by + bh // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w_img, x1 + side)
        y2 = min(h_img, y1 + side)

        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crop = cv2.resize(crop, (256, 256))
        else:
            crop = np.zeros((256, 256, 3), dtype=np.uint8)

        pixmap = numpy_to_qpixmap(crop, max_width=280, max_height=280)
        self.crop_label.setPixmap(pixmap)

        # Metadata
        sal = grain.get("saliency", 0)
        contour = grain.get("contour")
        n_pts = len(contour) if contour is not None else 0
        area_approx = bw * bh

        self.grain_meta_label.setText(
            f"Grano #{self._selected_grain_idx}\n"
            f"BBox: ({bx}, {by}, {bw}, {bh})\n"
            f"Saliency: {sal:.3f}\n"
            f"Área bbox: {area_approx} px²\n"
            f"Contorno: {n_pts} puntos"
        )

        total = len(self._current_grains)
        self.grain_nav_label.setText(f"{self._selected_grain_idx + 1} / {total}")

    # =========================================================================
    # Edición de segmentaciones
    # =========================================================================

    def _mark_dirty(self):
        """Marca el .seg actual como modificado."""
        self._seg_dirty = True
        self.btn_save_seg.setEnabled(True)
        self.edit_status_label.setText("● Cambios sin guardar")
        self.edit_status_label.setStyleSheet("color: #e67e22; font-size: 11px; font-weight: bold;")

    def _mark_clean(self):
        """Marca el .seg actual como guardado."""
        self._seg_dirty = False
        self.btn_save_seg.setEnabled(False)
        self.edit_status_label.setText("✓ Guardado")
        self.edit_status_label.setStyleSheet("color: #27ae60; font-size: 11px;")

    def _is_manual_roi_mode(self) -> bool:
        return self._edit_mode == "manual_roi"

    def _is_ai_roi_assist_enabled(self) -> bool:
        return self._is_manual_roi_mode() and hasattr(self, "roi_ai_assist_cb") and self.roi_ai_assist_cb.isChecked()

    def _on_edit_mode_changed(self, index: int):
        self._edit_mode = "manual_roi" if index == 1 else "auto"
        manual_enabled = self._is_manual_roi_mode()
        for w in (
            self.roi_snap_edges_cb,
            self.roi_ai_assist_cb,
            self.roi_multi_component_cb,
            self.btn_close_include,
            self.btn_close_exclude,
            self.btn_apply_roi,
            self.btn_clear_roi,
            self.btn_cancel_roi,
            self.roi_hint_label,
        ):
            w.setEnabled(manual_enabled)

        if not manual_enabled:
            self._reset_manual_roi_session()
        self._redraw_contours()

    def _clear_manual_roi_points(self):
        self._roi_include_polygons = []
        self._roi_exclude_polygons = []
        self._roi_current_include = []
        self._roi_current_exclude = []
        self._roi_include_mask = None
        self._roi_exclude_mask = None
        self._roi_block_mask = None
        self._roi_block_points = []
        self._redraw_contours()
        if self._is_manual_roi_mode():
            self.grain_info_label.setText("ROI manual limpio.")

    def _cancel_manual_roi_session(self):
        if not self._is_manual_roi_mode():
            return
        self._reset_manual_roi_session()
        self._redraw_contours()
        self.grain_info_label.setText("Edición ROI cancelada.")

    def _reset_manual_roi_session(self):
        self._roi_include_polygons = []
        self._roi_exclude_polygons = []
        self._roi_current_include = []
        self._roi_current_exclude = []
        self._manual_roi_seed_kind = None
        self._manual_roi_seed_point = None
        self._roi_include_mask = None
        self._roi_exclude_mask = None
        self._roi_block_mask = None
        self._roi_block_points = []
        self._roi_last_include_seed = None
        self._roi_suggest_points = []
        self._seed_retry_depth = 0
        self._pending_ai_seed = None
        self._last_include_drag_pt = None
        self._last_exclude_drag_pt = None

    def _close_manual_polygon(self, kind: str):
        if not self._is_manual_roi_mode():
            return
        if kind == "include":
            if len(self._roi_current_include) >= 3:
                self._roi_include_polygons.append(self._roi_current_include.copy())
            self._roi_current_include = []
        elif kind == "exclude":
            if len(self._roi_current_exclude) >= 3:
                self._roi_exclude_polygons.append(self._roi_current_exclude.copy())
            self._roi_current_exclude = []
        self._redraw_contours()

    def _close_active_manual_polygon(self):
        if not self._is_manual_roi_mode():
            return
        if len(self._roi_current_include) >= 3:
            self._close_manual_polygon("include")
            self.grain_info_label.setText("Polígono de inclusión cerrado.")
            return
        if len(self._roi_current_exclude) >= 3:
            self._close_manual_polygon("exclude")
            self.grain_info_label.setText("Polígono de exclusión cerrado.")
            return
        self.grain_info_label.setText("No hay polígono activo con al menos 3 puntos.")

    def _maybe_snap_to_edge(self, x: int, y: int, radius: int = 12) -> Tuple[int, int]:
        """Ajusta el punto al borde local más cercano (opcional)."""
        if (
            not self._is_manual_roi_mode()
            or self._current_image_bgr is None
            or not self.roi_snap_edges_cb.isChecked()
        ):
            return (x, y)

        img = self._current_image_bgr
        h, w = img.shape[:2]
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius + 1)
        y2 = min(h, y + radius + 1)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return (x, y)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            return (x, y)

        abs_xs = xs + x1
        abs_ys = ys + y1
        d2 = (abs_xs - x) ** 2 + (abs_ys - y) ** 2
        k = int(np.argmin(d2))
        return (int(abs_xs[k]), int(abs_ys[k]))

    def _ensure_roi_masks(self):
        """Inicializa máscaras acumuladas de ROI con tamaño de imagen actual."""
        if self._current_image_bgr is None:
            return
        h, w = self._current_image_bgr.shape[:2]
        if self._roi_include_mask is None or self._roi_include_mask.shape != (h, w):
            self._roi_include_mask = np.zeros((h, w), dtype=np.uint8)
        if self._roi_exclude_mask is None or self._roi_exclude_mask.shape != (h, w):
            self._roi_exclude_mask = np.zeros((h, w), dtype=np.uint8)
        if self._roi_block_mask is None or self._roi_block_mask.shape != (h, w):
            self._roi_block_mask = np.zeros((h, w), dtype=np.uint8)

    def _add_exclusion_anchor(self, x: int, y: int, radius: int = 4):
        """
        Agrega ancla roja de exclusión:
        - bloquea expansión futura de inclusión en ese punto;
        - elimina localmente ese punto del área verde ya acumulada.
        """
        self._ensure_roi_masks()
        if self._roi_block_mask is None:
            return
        cx, cy = int(x), int(y)
        # Frente de puntos: cada ancla agrega disco + posible conexión al punto previo.
        cv2.circle(self._roi_block_mask, (cx, cy), int(radius), 255, -1)
        if self._roi_block_points:
            px, py = self._roi_block_points[-1]
            dist = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
            if dist <= 120.0:
                thickness = max(2, int(radius * 2))
                cv2.line(
                    self._roi_block_mask,
                    (int(px), int(py)),
                    (cx, cy),
                    255,
                    thickness=thickness,
                )
        self._roi_block_points.append((cx, cy))
        if self._roi_include_mask is not None:
            inhibit = self._get_anchor_inhibit_mask()
            self._roi_include_mask[inhibit > 0] = 0
            self._prune_disconnected_include_regions()

    def _get_anchor_inhibit_mask(self, dilation_px: int = 6) -> np.ndarray:
        """Construye máscara de inhibición a partir de anclas rojas."""
        self._ensure_roi_masks()
        if self._roi_block_mask is None:
            return None
        if dilation_px <= 0:
            return self._roi_block_mask.copy()
        k = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(self._roi_block_mask, kernel, iterations=1)

    def _prune_disconnected_include_regions(self, seed: Optional[Tuple[int, int]] = None):
        """
        Elimina regiones verdes inconexas, conservando solo el componente principal.
        Prioridad:
        1) componente que contiene la semilla indicada;
        2) componente que contiene la última semilla de inclusión;
        3) componente de mayor área.
        """
        if self._roi_include_mask is None:
            return
        mask = self._roi_include_mask
        if np.count_nonzero(mask) == 0:
            return

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 2:
            return

        h, w = mask.shape[:2]

        def _label_at(pt):
            if pt is None:
                return 0
            x, y = int(pt[0]), int(pt[1])
            if x < 0 or y < 0 or x >= w or y >= h:
                return 0
            return int(labels[y, x])

        keep_label = _label_at(seed)
        if keep_label <= 0:
            keep_label = _label_at(self._roi_last_include_seed)
        if keep_label <= 0:
            comps = [(lbl, int(stats[lbl, cv2.CC_STAT_AREA])) for lbl in range(1, num_labels)]
            comps = [c for c in comps if c[1] > 0]
            if not comps:
                return
            keep_label = max(comps, key=lambda c: c[1])[0]

        pruned = np.zeros_like(mask, dtype=np.uint8)
        pruned[labels == keep_label] = 255
        self._roi_include_mask = pruned

    def _apply_ai_exclusion_contour(self, contour: np.ndarray, seed: Optional[Tuple[int, int]] = None):
        """
        Aplica exclusión asistida por U2NET de forma controlada:
        - toma la región propuesta por U2NET;
        - selecciona el componente conectado más relevante para la semilla;
        - restringe la sustracción al entorno de la máscara verde (evita recortes lejanos).
        """
        self._ensure_roi_masks()
        if contour is None or len(contour) < 3 or self._roi_include_mask is None:
            return

        h, w = self._roi_include_mask.shape[:2]
        cand = np.zeros((h, w), dtype=np.uint8)
        pts = contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(cand, [pts], -1, 255, -1)

        # Elegir componente conectado guiado por semilla (o más cercano).
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cand, 8)
        chosen = None
        if num_labels > 1:
            if seed is not None:
                sx = int(np.clip(seed[0], 0, w - 1))
                sy = int(np.clip(seed[1], 0, h - 1))
                lbl = int(labels[sy, sx])
                if lbl > 0 and stats[lbl, cv2.CC_STAT_AREA] >= 3:
                    chosen = (labels == lbl).astype(np.uint8) * 255
            if chosen is None:
                # fallback: componente más cercana a semilla o mayor área
                if seed is not None:
                    sx, sy = seed
                    best_lbl = -1
                    best_d2 = float("inf")
                    for lbl in range(1, num_labels):
                        area = int(stats[lbl, cv2.CC_STAT_AREA])
                        if area < 3:
                            continue
                        cx, cy = centroids[lbl]
                        d2 = (cx - sx) ** 2 + (cy - sy) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_lbl = lbl
                    if best_lbl > 0:
                        chosen = (labels == best_lbl).astype(np.uint8) * 255
                if chosen is None:
                    areas = [(lbl, int(stats[lbl, cv2.CC_STAT_AREA])) for lbl in range(1, num_labels)]
                    areas = [x for x in areas if x[1] >= 3]
                    if areas:
                        best_lbl = max(areas, key=lambda x: x[1])[0]
                        chosen = (labels == best_lbl).astype(np.uint8) * 255
        else:
            chosen = cand

        if chosen is None:
            return

        # Exclusión LOCAL alrededor de la semilla para evitar "comerse" toda la máscara.
        include_local = cv2.bitwise_and(chosen, self._roi_include_mask)
        if np.count_nonzero(include_local) == 0:
            # Si no intersecta con verde, al menos bloquear su región para siguientes expansiones.
            self._roi_block_mask = cv2.bitwise_or(self._roi_block_mask, chosen)
            return

        x, y, bw, bh = cv2.boundingRect(chosen)
        local_radius = int(np.clip(0.22 * max(bw, bh), 10, 42))
        if seed is not None:
            sx = int(np.clip(seed[0], 0, w - 1))
            sy = int(np.clip(seed[1], 0, h - 1))
        else:
            sx = int(np.clip(x + bw // 2, 0, w - 1))
            sy = int(np.clip(y + bh // 2, 0, h - 1))

        local_circle = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(local_circle, (sx, sy), local_radius, 255, -1)
        remove_mask = cv2.bitwise_and(include_local, local_circle)

        if np.count_nonzero(remove_mask) == 0:
            # fallback local: punto más cercano dentro de include_local
            ys, xs = np.where(include_local > 0)
            if len(xs) == 0:
                return
            d2 = (xs - sx) ** 2 + (ys - sy) ** 2
            k = int(np.argmin(d2))
            px, py = int(xs[k]), int(ys[k])
            local_circle = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(local_circle, (px, py), max(8, local_radius // 2), 255, -1)
            remove_mask = cv2.bitwise_and(include_local, local_circle)

        if np.count_nonzero(remove_mask) == 0:
            return

        # Límite de seguridad: no retirar más de una fracción por click.
        include_area = int(np.count_nonzero(self._roi_include_mask))
        max_remove = max(40, int(0.18 * include_area))
        rem_area = int(np.count_nonzero(remove_mask))
        if rem_area > max_remove and seed is not None:
            ys, xs = np.where(remove_mask > 0)
            d2 = (xs - sx) ** 2 + (ys - sy) ** 2
            order = np.argsort(d2)
            keep_idx = order[:max_remove]
            limited = np.zeros((h, w), dtype=np.uint8)
            limited[ys[keep_idx], xs[keep_idx]] = 255
            remove_mask = limited

        # Quitar realmente de la máscara verde y bloquear re-expansión local.
        self._roi_include_mask[remove_mask > 0] = 0
        self._roi_block_mask = cv2.bitwise_or(self._roi_block_mask, remove_mask)
        self._prune_disconnected_include_regions(seed=self._roi_last_include_seed)

    def _accumulate_roi_seed_contour(self, kind: str, contour: np.ndarray, seed: Optional[Tuple[int, int]] = None):
        """
        Acumula la región de una semilla en máscaras unificadas (unión progresiva).
        Esto evita que queden submáscaras separadas sin integración.
        """
        self._ensure_roi_masks()
        if contour is None or len(contour) < 3:
            return

        poly = contour.reshape(-1, 2).astype(np.int32)
        fill = np.zeros_like(self._roi_include_mask, dtype=np.uint8)
        cv2.drawContours(fill, [poly.reshape(-1, 1, 2)], -1, 255, -1)

        if kind == "include":
            # Las anclas rojas bloquean expansión de la zona verde
            inhibit = self._get_anchor_inhibit_mask()
            if inhibit is not None:
                fill = cv2.bitwise_and(fill, cv2.bitwise_not(inhibit))
            self._roi_include_mask = cv2.bitwise_or(self._roi_include_mask, fill)
            self._roi_last_include_seed = seed
            self._prune_disconnected_include_regions(seed=seed)
            # En modo AI, evitar duplicar geometría en listas de polígonos
            # (la máscara acumulada es la fuente de verdad).
            if not self._is_ai_roi_assist_enabled():
                self._roi_include_polygons.append([(int(p[0]), int(p[1])) for p in poly])
        else:
            self._roi_exclude_mask = cv2.bitwise_or(self._roi_exclude_mask, fill)
            if not self._is_ai_roi_assist_enabled():
                self._roi_exclude_polygons.append([(int(p[0]), int(p[1])) for p in poly])

    def _request_ai_roi_seed(self, x_orig: int, y_orig: int, kind: str):
        """Lanza propuesta local U2NET para click-semilla include/exclude."""
        if self._current_image_bgr is None:
            return
        if hasattr(self, "_manual_worker") and self._manual_worker is not None and self._manual_worker.isRunning():
            self._pending_ai_seed = (int(x_orig), int(y_orig), kind)
            return
        if not self._check_weights_exist():
            return

        sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
        self._manual_roi_seed_kind = kind
        self._manual_roi_seed_point = (sx, sy)

        config = self._get_config()
        self.grain_info_label.setText(
            f"U2-Net proponiendo región {'inclusión' if kind == 'include' else 'exclusión'} en ({sx}, {sy})..."
        )
        self._manual_worker = ManualContourWorker(
            self._current_image_bgr, sx, sy, config, crop_radius=300
        )
        self._manual_worker.finished.connect(self._on_manual_roi_seed_result)
        self._manual_worker.error.connect(self._on_manual_roi_seed_error)
        self._manual_worker.start()

    def _drain_pending_ai_seed(self):
        if self._pending_ai_seed is None:
            return
        x, y, kind = self._pending_ai_seed
        self._pending_ai_seed = None
        self._request_ai_roi_seed(x, y, kind)

    def _is_far_enough_from_last(self, last_pt: Optional[Tuple[int, int]], x: int, y: int) -> bool:
        if last_pt is None:
            return True
        dx = int(x) - int(last_pt[0])
        dy = int(y) - int(last_pt[1])
        return (dx * dx + dy * dy) >= (self._drag_step_px * self._drag_step_px)

    def _suggest_expansion_points(
        self, seed: Optional[Tuple[int, int]], max_points: int = 6
    ) -> List[Tuple[int, int]]:
        """
        Sugiere puntos de expansión alrededor del borde del área verde actual.
        Prioriza vecindad de la semilla fallida y cobertura del frente de crecimiento.
        """
        if self._roi_include_mask is None or np.count_nonzero(self._roi_include_mask) == 0:
            return []

        mask = self._roi_include_mask
        h, w = mask.shape[:2]
        ring = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
        ys, xs = np.where(ring > 0)
        if len(xs) == 0:
            return []

        if seed is not None:
            sx = int(np.clip(seed[0], 0, w - 1))
            sy = int(np.clip(seed[1], 0, h - 1))
        else:
            sx, sy = w // 2, h // 2

        d2 = (xs - sx) ** 2 + (ys - sy) ** 2
        order = np.argsort(d2)
        candidates = [(int(xs[i]), int(ys[i])) for i in order[: min(len(order), 2000)]]

        picked = []
        min_sep2 = 18 * 18
        for p in candidates:
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep2 for q in picked):
                picked.append(p)
            if len(picked) >= max_points:
                break
        return picked

    def _on_manual_roi_seed_result(self, grain_dicts):
        """Callback de semilla AI: agrega contorno propuesto a include o exclude."""
        kind = self._manual_roi_seed_kind
        seed = self._manual_roi_seed_point
        self._manual_roi_seed_kind = None
        self._manual_roi_seed_point = None

        if kind not in ("include", "exclude"):
            self.grain_info_label.setText("Semilla AI sin contexto de edición.")
            return

        contour = None
        if grain_dicts:
            contour = grain_dicts[0].get("contour")

        if contour is None or len(contour) < 3:
            if kind == "include" and self._is_ai_roi_assist_enabled() and self._seed_retry_depth < 2:
                self._seed_retry_depth += 1
                suggestions = self._suggest_expansion_points(seed, max_points=6)
                self._roi_suggest_points = suggestions
                self._redraw_contours()
                if suggestions:
                    nx, ny = suggestions[0]
                    self.grain_info_label.setText(
                        f"No se detectó en semilla {seed}. Sugerencia de expansión en ({nx}, {ny}) "
                        f"[reintento {self._seed_retry_depth}/2]."
                    )
                    self._request_ai_roi_seed(nx, ny, "include")
                    return
            self.grain_info_label.setText(
                "U2-Net no pudo derivar región válida para esta semilla."
            )
            return

        if kind == "exclude" and self._is_ai_roi_assist_enabled():
            self._apply_ai_exclusion_contour(contour, seed=seed)
        else:
            self._accumulate_roi_seed_contour(kind, contour, seed=seed)
            if kind == "include":
                self._seed_retry_depth = 0
                self._roi_suggest_points = []

        self._redraw_contours()
        incl_stat = (
            np.count_nonzero(self._roi_include_mask) if self._roi_include_mask is not None else len(self._roi_include_polygons)
        )
        excl_stat = (
            len(self._roi_block_points) if self._is_ai_roi_assist_enabled()
            else (np.count_nonzero(self._roi_exclude_mask) if self._roi_exclude_mask is not None else len(self._roi_exclude_polygons))
        )
        self.grain_info_label.setText(
            f"Región AI {'incluida' if kind == 'include' else 'excluida'} desde semilla {seed}. "
            f"Incl: {incl_stat} | Excl/anclas: {excl_stat}"
        )
        self._drain_pending_ai_seed()

    def _on_manual_roi_seed_error(self, error_msg: str):
        self._manual_roi_seed_kind = None
        self._manual_roi_seed_point = None
        self.grain_info_label.setText(f"Error U2-Net semilla ROI: {error_msg[:80]}")
        logger.error(f"[ManualROI] Error semilla AI: {error_msg}")
        self._drain_pending_ai_seed()

    def _build_manual_roi_mask(self) -> Optional[np.ndarray]:
        from src.grain_detection.manual_roi import rasterize_polygons

        if self._current_image_bgr is None:
            return None
        h, w = self._current_image_bgr.shape[:2]

        # Polígonos manuales (solo cuando NO se usa asistencia AI por semilla)
        if self._is_ai_roi_assist_enabled():
            include_mask_poly = np.zeros((h, w), dtype=np.uint8)
            exclude_mask_poly = np.zeros((h, w), dtype=np.uint8)
        else:
            include_polys = self._roi_include_polygons.copy()
            exclude_polys = self._roi_exclude_polygons.copy()
            # Cerrar implícitamente polígonos en curso al aplicar si tienen >=3 puntos
            if len(self._roi_current_include) >= 3:
                include_polys.append(self._roi_current_include.copy())
            if len(self._roi_current_exclude) >= 3:
                exclude_polys.append(self._roi_current_exclude.copy())
            include_mask_poly = rasterize_polygons((h, w), include_polys)
            exclude_mask_poly = rasterize_polygons((h, w), exclude_polys)

        include_mask = include_mask_poly
        exclude_mask = exclude_mask_poly

        # Integrar también acumulación unificada de semillas AI
        if self._roi_include_mask is not None and self._roi_include_mask.shape == (h, w):
            include_mask = cv2.bitwise_or(include_mask, self._roi_include_mask)
        if self._roi_exclude_mask is not None and self._roi_exclude_mask.shape == (h, w):
            exclude_mask = cv2.bitwise_or(exclude_mask, self._roi_exclude_mask)
        if self._roi_block_mask is not None and self._roi_block_mask.shape == (h, w):
            exclude_mask = cv2.bitwise_or(exclude_mask, self._roi_block_mask)

        if np.count_nonzero(include_mask) == 0:
            return None
        keep_mask = cv2.bitwise_and(include_mask, cv2.bitwise_not(exclude_mask))
        return keep_mask

    def _draw_manual_roi_overlay(self, base_image: np.ndarray) -> np.ndarray:
        """Dibuja overlay semitransparente de ROI manual sobre la imagen."""
        out = base_image.copy()
        overlay = out.copy()
        ai_mode = self._is_ai_roi_assist_enabled()

        def _draw_poly(polys, color_fill, color_line):
            for poly in polys:
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(overlay, [pts], color_fill)
                    cv2.polylines(out, [pts], isClosed=True, color=color_line, thickness=2)

        # En modo manual puro, mostrar polígonos explícitos.
        # En modo AI, la fuente de verdad son las máscaras acumuladas.
        if not ai_mode:
            _draw_poly(self._roi_include_polygons, (0, 140, 0), (0, 255, 0))
            for poly in self._roi_exclude_polygons:
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(out, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # Máscara verde acumulada unificada
        if self._roi_include_mask is not None:
            overlay[self._roi_include_mask > 0] = (0, 140, 0)
        # Zona bloqueada/excluida en rojo suave para depuración visual
        if self._roi_block_mask is not None:
            overlay[self._roi_block_mask > 0] = (0, 0, 120)

        cv2.addWeighted(overlay, 0.25, out, 0.75, 0.0, out)

        # Dibujar anclas rojas explícitas
        for px, py in self._roi_block_points:
            cv2.circle(out, (int(px), int(py)), 4, (0, 0, 255), 2)

        # Sugerencias de expansión (puntos azules)
        for sx, sy in self._roi_suggest_points:
            cv2.circle(out, (int(sx), int(sy)), 4, (255, 120, 0), 2)

        def _draw_open(points, color):
            if not points:
                return
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=2)
            for p in points:
                cv2.circle(out, (int(p[0]), int(p[1])), 3, color, -1)

        _draw_open(self._roi_current_include, (0, 255, 0))
        _draw_open(self._roi_current_exclude, (0, 0, 255))
        return out

    def _apply_manual_roi_to_grain(self):
        """Aplica ROI manual include/exclude al grano activo o crea instancias nuevas."""
        if not self._is_manual_roi_mode() or self._current_image_bgr is None:
            return

        mask = self._build_manual_roi_mask()
        if mask is None:
            self.grain_info_label.setText("ROI manual vacío: falta polígono de inclusión.")
            return

        from src.grain_detection.manual_roi import extract_contours_from_mask, contour_bbox

        contours = extract_contours_from_mask(mask, min_area=3.0)
        if not contours:
            self.grain_info_label.setText("ROI resultante vacío tras aplicar exclusión.")
            return

        self._push_history()
        multi = self.roi_multi_component_cb.isChecked()
        contour_list = contours if multi else contours[:1]

        # Selección inteligente de destino:
        # - si la ROI nueva solapa fuertemente un grano existente, se interpreta como edición;
        # - si no solapa, se agrega como nuevo objeto (evita reemplazos erróneos).
        target_idx = self._find_best_overlapping_grain(mask)

        if target_idx >= 0:
            base_sal = float(self._current_grains[target_idx].get("saliency", 0.5))
        else:
            base_sal = 0.5

        created = []
        for c in contour_list:
            bx, by, bw, bh = contour_bbox(c)
            created.append(
                {
                    "index": 0,
                    "class_index": 0,
                    "bbox": (bx, by, bw, bh),
                    "saliency": base_sal,
                    "contour": c.astype(np.int32),
                }
            )

        if target_idx >= 0:
            # Reemplaza grano target y opcionalmente agrega componentes extra.
            self._current_grains[target_idx] = created[0]
            for extra in created[1:]:
                self._current_grains.append(extra)
            self._selected_grain_idx = target_idx
            self._active_grain_idx = target_idx
        else:
            self._current_grains.extend(created)
            self._selected_grain_idx = len(self._current_grains) - len(created)
            self._active_grain_idx = self._selected_grain_idx

        for i, g in enumerate(self._current_grains):
            g["index"] = i

        self._mark_dirty()
        self._reset_manual_roi_session()
        self._redraw_contours()
        self._show_selected_grain()
        self.grain_info_label.setText(
            f"ROI aplicado. {len(created)} instancia{'s' if len(created) != 1 else ''} "
            f"actualizada{'s' if len(created) != 1 else ''}."
        )

    def _find_best_overlapping_grain(self, roi_mask: np.ndarray, min_overlap: float = 0.25) -> int:
        """
        Retorna el índice del grano con mayor solapamiento con roi_mask.
        Si no hay solapamiento suficiente, retorna -1 para forzar creación de nuevo objeto.
        """
        if roi_mask is None or not self._current_grains:
            return -1

        best_idx = -1
        best_score = 0.0
        h, w = roi_mask.shape[:2]

        for idx, grain in enumerate(self._current_grains):
            contour = grain.get("contour")
            if contour is None or len(contour) < 3:
                continue

            gmask = np.zeros((h, w), dtype=np.uint8)
            pts = contour.reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(gmask, [pts], -1, 255, -1)

            inter = cv2.bitwise_and(roi_mask, gmask)
            inter_area = float(np.count_nonzero(inter))
            grain_area = float(np.count_nonzero(gmask))
            if grain_area <= 0:
                continue

            # fracción del grano cubierta por ROI nueva
            overlap = inter_area / grain_area
            if overlap > best_score:
                best_score = overlap
                best_idx = idx

        return best_idx if best_score >= min_overlap else -1

    def _delete_grain_near_point(self, x_orig: int, y_orig: int) -> bool:
        """
        Elimina el grano cercano al punto (si corresponde). Retorna True si eliminó.
        """
        if self._current_image_bgr is None or not self._current_grains:
            return False

        best_idx = -1
        best_dist = float("inf")
        for i, grain in enumerate(self._current_grains):
            bx, by, bw, bh = grain["bbox"]
            cx = bx + bw // 2
            cy = by + bh // 2
            dist = ((x_orig - cx) ** 2 + (y_orig - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx < 0:
            return False

        grain = self._current_grains[best_idx]
        bx, by, bw, bh = grain["bbox"]
        max_dist = max(bw, bh) * 1.5
        if best_dist > max_dist:
            return False

        self._push_history()
        removed = self._current_grains.pop(best_idx)
        logger.info(f"[ManualContour] Eliminado grano #{best_idx} por click derecho, bbox={removed['bbox']}")

        for i, g in enumerate(self._current_grains):
            g["index"] = i

        self._mark_dirty()
        self._redraw_contours()

        if len(self._current_grains) == 0:
            self._selected_grain_idx = -1
            self._active_grain_idx = -1
            self.crop_label.setText("Sin granos")
            self.grain_meta_label.setText("")
            self.grain_nav_label.setText("—")
        else:
            self._selected_grain_idx = min(best_idx, len(self._current_grains) - 1)
            if self._active_grain_idx == best_idx:
                self._active_grain_idx = -1
            self._show_selected_grain()

        self.grain_info_label.setText(
            f"Grano #{best_idx} eliminado. {len(self._current_grains)} restantes."
        )
        return True

    def _on_delete_grain(self):
        """Elimina el grano seleccionado de la anotación actual."""
        if not self._current_grains or self._selected_grain_idx < 0:
            QMessageBox.information(self, "Info", "No hay grano seleccionado para eliminar.")
            return

        idx = self._selected_grain_idx
        n = len(self._current_grains)

        reply = QMessageBox.question(
            self, "Confirmar eliminación",
            f"¿Eliminar grano #{idx} de {n}?\n"
            f"BBox: {self._current_grains[idx]['bbox']}\n"
            f"Saliency: {self._current_grains[idx].get('saliency', 0):.3f}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Eliminar el grano
        removed = self._current_grains.pop(idx)
        logger.info(f"[Edit] Eliminado grano #{idx}, bbox={removed['bbox']}")

        # Re-indexar granos restantes
        for i, g in enumerate(self._current_grains):
            g["index"] = i

        self._mark_dirty()

        # Ajustar selección
        if len(self._current_grains) == 0:
            self._selected_grain_idx = -1
            self.crop_label.setText("Sin granos")
            self.grain_meta_label.setText("")
            self.grain_nav_label.setText("—")
        else:
            self._selected_grain_idx = min(idx, len(self._current_grains) - 1)

        # Re-dibujar imagen con contornos actualizados
        self._redraw_contours()

        if self._selected_grain_idx >= 0:
            self._show_selected_grain()

    def _on_save_seg(self):
        """Guarda el .seg actual con los granos editados."""
        if self._current_seg_path is None:
            QMessageBox.warning(self, "Error", "No hay .seg cargado para guardar.")
            return

        # Safety: check if we'd lose data by saving fewer grains
        existing_count = 0
        if self._current_seg_path and Path(self._current_seg_path).exists():
            try:
                from src.grain_detection.seg_format import SegFileReader
                existing_count = SegFileReader.count_grains(str(self._current_seg_path))
            except Exception:
                pass

        new_count = len(self._current_grains) if self._current_grains else 0

        if new_count == 0 and existing_count > 0:
            reply = QMessageBox.warning(
                self, "Advertencia: Pérdida de datos",
                f"El .seg actual tiene {existing_count} grano(s).\n"
                f"Vas a guardar 0 granos — esto BORRARÁ las anotaciones existentes.\n\n"
                f"Se creará un backup automático (.seg.bak).\n"
                f"¿Continuar?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        elif new_count < existing_count:
            reply = QMessageBox.question(
                self, "Confirmar reducción",
                f"El .seg actual tiene {existing_count} grano(s), "
                f"pero guardarás {new_count}.\n"
                f"Se creará un backup automático (.seg.bak).\n"
                f"¿Continuar?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return

        try:
            from src.grain_detection.seg_format import SegFileWriter

            # Recuperar metadata de la imagen actual
            img_path = self._current_images[self._current_image_idx]
            h, w = self._current_image_bgr.shape[:2] if self._current_image_bgr is not None else (0, 0)

            SegFileWriter.write(
                str(self._current_seg_path),
                img_path.name,
                (h, w),
                self._current_grains,
                {"editado_manualmente": "true"},
            )

            self._mark_clean()
            logger.info(f"[Edit] Guardado {self._current_seg_path.name}: {len(self._current_grains)} granos")

            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                self.parent_window.log_widget.append_log(
                    f"Seg guardado: {self._current_seg_path.name} ({len(self._current_grains)} granos)",
                    "SUCCESS"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardando .seg:\n{e}")
            logger.error(f"[Edit] Error guardando {self._current_seg_path}: {e}")

    # =========================================================================
    # Historial de cambios (UNDO)
    # =========================================================================
    
    def _push_history(self):
        """Guarda el estado actual de granos en el historial para UNDO."""
        import copy
        # Deep copy para evitar referencias compartidas
        state_snapshot = copy.deepcopy(self._current_grains)
        self._grains_history.append(state_snapshot)
        
        # Limitar historial a últimos 20 estados para no consumir memoria
        if len(self._grains_history) > 20:
            self._grains_history.pop(0)
        
        logger.debug(f"[History] Estado guardado. Historial: {len(self._grains_history)} estados")
    
    def _pop_history(self):
        """Restaura el último estado de granos del historial."""
        if not self._grains_history:
            return self._current_grains
        
        import copy
        restored_state = self._grains_history.pop()
        logger.debug(f"[History] Estado restaurado. Historial: {len(self._grains_history)} estados")
        return copy.deepcopy(restored_state)
    
    def _clear_history(self):
        """Limpia el historial de cambios (usado al cambiar de imagen)."""
        self._grains_history = []
        self._active_grain_idx = -1
        logger.debug("[History] Historial limpiado")

    # =========================================================================
    # Manual contour interaction (click on image)
    # =========================================================================

    def _on_image_left_click(self, x_orig, y_orig):
        """Click izquierdo: auto U2NET o agregar punto de inclusión (ROI manual)."""
        if self._current_image_bgr is None:
            return

        if self._is_manual_roi_mode():
            if self.roi_ai_assist_cb.isChecked():
                self._seed_retry_depth = 0
                self._roi_suggest_points = []
                self._last_include_drag_pt = (int(x_orig), int(y_orig))
                self._request_ai_roi_seed(x_orig, y_orig, "include")
            else:
                sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
                self._last_include_drag_pt = (int(sx), int(sy))
                self._roi_current_include.append((sx, sy))
                self._redraw_contours()
                self.grain_info_label.setText(
                    f"Inclusión manual: {len(self._roi_current_include)} punto(s) en polígono activo."
                )
            return

        # Prevent multiple concurrent workers
        if hasattr(self, '_manual_worker') and self._manual_worker is not None and self._manual_worker.isRunning():
            self.grain_info_label.setText("U2-Net procesando... espera.")
            return

        if not self._check_weights_exist():
            return

        config = self._get_config()
        self.grain_info_label.setText(f"Proponiendo contorno en ({x_orig}, {y_orig})...")

        self._manual_worker = ManualContourWorker(
            self._current_image_bgr, x_orig, y_orig, config, crop_radius=300
        )
        self._manual_worker.finished.connect(self._on_manual_contour_result)
        self._manual_worker.error.connect(self._on_manual_contour_error)
        self._manual_worker.start()

    def _on_image_left_double_click(self, x_orig, y_orig):
        """Doble click izquierdo: cierra polígono de inclusión en ROI manual."""
        if not self._is_manual_roi_mode():
            return
        self._close_manual_polygon("include")
        self.grain_info_label.setText("Polígono de inclusión cerrado.")

    def _on_image_right_double_click(self, x_orig, y_orig):
        """Doble click derecho: cierra polígono de exclusión en ROI manual."""
        if not self._is_manual_roi_mode():
            return
        self._close_manual_polygon("exclude")
        self.grain_info_label.setText("Polígono de exclusión cerrado.")

    def _on_manual_contour_result(self, grain_dicts):
        """Callback when U2-Net manual contour suggestion finishes.
        
        Nuevo comportamiento: fusiona el contorno detectado con el grano activo.
        Si no hay grano activo, crea uno nuevo.
        """
        if not grain_dicts:
            self.grain_info_label.setText(
                "No se detectó grano en esa posición. Intenta otro punto."
            )
            return

        new_grain = grain_dicts[0]
        new_contour = new_grain.get("contour")
        
        if new_contour is None or len(new_contour) < 3:
            self.grain_info_label.setText("Contorno inválido detectado.")
            return

        # Guardar estado actual para UNDO
        self._push_history()
        
        # Si hay un grano activo, FUSIONAR contornos
        if self._active_grain_idx >= 0 and self._active_grain_idx < len(self._current_grains):
            active_grain = self._current_grains[self._active_grain_idx]
            existing_contour = active_grain.get("contour")
            
            if existing_contour is not None and len(existing_contour) >= 3:
                # Fusionar contornos usando convex hull de ambos
                import cv2
                all_points = np.vstack([existing_contour, new_contour])
                merged_hull = cv2.convexHull(all_points)
                
                # Actualizar grano activo con contorno fusionado
                active_grain["contour"] = merged_hull
                
                # Recalcular bbox
                x, y, w, h = cv2.boundingRect(merged_hull)
                active_grain["bbox"] = (x, y, w, h)
                
                # Actualizar saliency (promedio)
                old_sal = active_grain.get("saliency", 0)
                new_sal = new_grain.get("saliency", 0)
                active_grain["saliency"] = (old_sal + new_sal) / 2
                
                self._mark_dirty()
                self._redraw_contours()
                self._show_selected_grain()
                
                self.grain_info_label.setText(
                    f"Región fusionada con grano #{self._active_grain_idx}. "
                    f"Click izq para añadir más regiones, click der para deshacer."
                )
                logger.info(f"[ManualContour] Región fusionada con grano #{self._active_grain_idx}")
                return
        
        # Si no hay grano activo O no existe, crear NUEVO grano
        # Check for overlap with existing grains
        new_bx, new_by, new_bw, new_bh = new_grain["bbox"]
        new_cx = new_bx + new_bw // 2
        new_cy = new_by + new_bh // 2

        for idx, existing in enumerate(self._current_grains):
            ex, ey, ew, eh = existing["bbox"]
            ecx = ex + ew // 2
            ecy = ey + eh // 2
            dist = ((new_cx - ecx) ** 2 + (new_cy - ecy) ** 2) ** 0.5
            min_size = min(new_bw, new_bh, ew, eh) * 0.5
            if dist < min_size:
                # Overlap detectado - activar este grano para edición
                self._active_grain_idx = idx
                self._selected_grain_idx = idx
                self.grain_info_label.setText(
                    f"Grano #{idx} activado para edición. "
                    f"Click izq para añadir regiones, click der para deshacer."
                )
                self._show_selected_grain()
                return

        # Crear nuevo grano
        new_grain["index"] = len(self._current_grains)
        self._current_grains.append(new_grain)
        self._mark_dirty()

        # Activar el nuevo grano para edición
        self._active_grain_idx = len(self._current_grains) - 1
        self._selected_grain_idx = self._active_grain_idx
        self._redraw_contours()
        self._show_selected_grain()

        self.grain_info_label.setText(
            f"Grano #{new_grain['index']} creado y activado. "
            f"Click izq para añadir más regiones."
        )
        logger.info(
            f"[ManualContour] Grano nuevo creado: bbox={new_grain['bbox']}, "
            f"saliency={new_grain['saliency']:.3f}"
        )

    def _on_manual_contour_error(self, error_msg):
        """Callback when U2-Net manual contour suggestion fails."""
        self.grain_info_label.setText(f"Error U2-Net: {error_msg[:80]}")
        logger.error(f"[ManualContour] Error: {error_msg}")

    def _on_image_right_click(self, x_orig, y_orig):
        """Click derecho: excluir punto (ROI manual) o eliminar/undo (modo auto).
        
        Comportamiento inteligente:
        1. Si hay un grano cercano al click → ELIMINAR ese grano
        2. Si NO hay grano cercano → UNDO última operación del historial
        """
        if self._current_image_bgr is None:
            return

        if self._is_manual_roi_mode():
            has_mask_work = self._roi_include_mask is not None and np.count_nonzero(self._roi_include_mask) > 0
            has_include_work = bool(self._roi_include_polygons or self._roi_current_include or has_mask_work)

            # Si no se está construyendo inclusión, mantener comportamiento de borrado
            # para evitar confusión entre objetos ya creados.
            if not has_include_work and self._delete_grain_near_point(x_orig, y_orig):
                return

            if self.roi_ai_assist_cb.isChecked():
                sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
                self._last_exclude_drag_pt = (int(sx), int(sy))
                self._add_exclusion_anchor(sx, sy, radius=4)
                self._redraw_contours()
                self.grain_info_label.setText(
                    f"Barrera roja extendida en ({sx}, {sy}). Bloquea expansión del ROI verde."
                )
            else:
                sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
                self._last_exclude_drag_pt = (int(sx), int(sy))
                self._roi_current_exclude.append((sx, sy))
                self._redraw_contours()
                self.grain_info_label.setText(
                    f"Exclusión manual: {len(self._roi_current_exclude)} punto(s) en polígono activo."
                )
            return

        # Modo auto: primero intentar eliminar grano cercano
        if self._delete_grain_near_point(x_orig, y_orig):
            self.grain_info_label.setText(
                f"{self.grain_info_label.text()} Click der en vacío para UNDO."
            )
            return
        
        # Si NO hay grano cercano, hacer UNDO
        if not self._grains_history:
            self.grain_info_label.setText(
                "No hay contorno cerca del click ni operaciones para deshacer."
            )
            return
        
        # Restaurar estado anterior del historial
        self._current_grains = self._pop_history()
        
        # Re-indexar granos
        for i, g in enumerate(self._current_grains):
            g["index"] = i
        
        self._mark_dirty()
        self._redraw_contours()
        
        # Mantener selección si es válida
        if self._selected_grain_idx >= len(self._current_grains):
            self._selected_grain_idx = len(self._current_grains) - 1
        
        if self._active_grain_idx >= len(self._current_grains):
            self._active_grain_idx = -1
        
        if self._selected_grain_idx >= 0:
            self._show_selected_grain()
        else:
            self.crop_label.setText("Sin granos")
            self.grain_meta_label.setText("")
            self.grain_nav_label.setText("—")
        
        n_history = len(self._grains_history)
        self.grain_info_label.setText(
            f"Última operación deshecha. {n_history} operacion{'es' if n_history != 1 else ''} en historial."
        )
        logger.info(f"[ManualContour] UNDO ejecutado. Historial: {n_history} estados")

    def _on_image_left_drag(self, x_orig, y_orig):
        """Arrastre izquierdo: sembrado continuo para inclusión."""
        if self._current_image_bgr is None or not self._is_manual_roi_mode():
            return
        if not self._is_far_enough_from_last(self._last_include_drag_pt, x_orig, y_orig):
            return

        if self.roi_ai_assist_cb.isChecked():
            self._last_include_drag_pt = (int(x_orig), int(y_orig))
            self._request_ai_roi_seed(x_orig, y_orig, "include")
        else:
            sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
            self._last_include_drag_pt = (int(sx), int(sy))
            self._roi_current_include.append((sx, sy))
            self._redraw_contours()

    def _on_image_right_drag(self, x_orig, y_orig):
        """Arrastre derecho: barrera roja continua o exclusión manual continua."""
        if self._current_image_bgr is None or not self._is_manual_roi_mode():
            return
        if not self._is_far_enough_from_last(self._last_exclude_drag_pt, x_orig, y_orig):
            return

        if self.roi_ai_assist_cb.isChecked():
            sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
            self._last_exclude_drag_pt = (int(sx), int(sy))
            self._add_exclusion_anchor(sx, sy, radius=4)
            self._redraw_contours()
        else:
            sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
            self._last_exclude_drag_pt = (int(sx), int(sy))
            self._roi_current_exclude.append((sx, sy))
            self._redraw_contours()

    def _on_image_left_release(self, x_orig, y_orig):
        self._last_include_drag_pt = None

    def _on_image_right_release(self, x_orig, y_orig):
        self._last_exclude_drag_pt = None

    def _redraw_contours(self):
        """Re-dibuja los contornos sobre la imagen actual sin recargar."""
        if self._current_image_bgr is None:
            return

        annotated = self._current_image_bgr.copy()
        for i, grain in enumerate(self._current_grains):
            color = GRAIN_COLORS[i % len(GRAIN_COLORS)]
            contour = grain.get("contour")
            if contour is not None and len(contour) >= 3:
                pts = contour.reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(annotated, [pts], -1, color, 2)

            bx, by, bw, bh = grain["bbox"]
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 1)
            cv2.putText(annotated, f"#{i}", (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if self._is_manual_roi_mode():
            annotated = self._draw_manual_roi_overlay(annotated)

        pixmap = numpy_to_qpixmap(annotated, max_width=700, max_height=500)
        self.image_label.setPixmap(pixmap)

        n = len(self._current_grains)
        edit_tag = " ● Editado" if self._seg_dirty else ""
        if self._is_manual_roi_mode():
            if self.roi_ai_assist_cb.isChecked():
                click_mode = "semilla U2NET + ancla roja"
            else:
                click_mode = "puntos manuales"
            hint = (
                f"Modo ROI manual ({click_mode}): Izq incluir | Der ancla/excluir | "
                f"C cerrar | A aplicar | Esc cancelar"
            )
        else:
            hint = "Click izq: proponer contorno | Click der: eliminar contorno"
        self.grain_info_label.setText(
            f"{n} grano{'s' if n != 1 else ''}{edit_tag} — "
            f"{hint}"
        )

    # =========================================================================
    # Estadísticas
    # =========================================================================

    def _update_stats(self, stats: Dict):
        """Actualiza la sección de estadísticas con el resultado de la anotación."""
        total_imgs = stats.get("total_images", 0)
        total_grains = stats.get("total_grains", 0)
        avg = stats.get("grains_per_image", 0)
        per_class = stats.get("grains_per_class", {})

        class_str = "  |  ".join(f"{k}: {v}" for k, v in sorted(per_class.items()))
        self.stats_label.setText(
            f"Imágenes: {total_imgs}  |  Granos: {total_grains}  |  "
            f"Promedio/img: {avg:.1f}\n"
            f"Por clase: {class_str}"
        )

    def _update_stats_from_directory(self, root_dir: str):
        """Calcula estadísticas rápidas escaneando .seg existentes."""
        root = Path(root_dir)
        if not root.exists():
            return

        total_images = 0
        total_seg = 0
        total_grains = 0
        grains_per_class = {}

        from src.grain_detection.seg_format import SegFileReader

        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            class_grains = 0

            for f in class_dir.iterdir():
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}:
                    total_images += 1
                    seg = f.with_suffix(".seg")
                    if seg.exists():
                        total_seg += 1
                        try:
                            n = SegFileReader.count_grains(str(seg))
                            class_grains += n
                        except Exception:
                            pass

            total_grains += class_grains
            if class_grains > 0:
                grains_per_class[class_name] = class_grains

        if total_images > 0:
            avg = total_grains / max(1, total_seg) if total_seg > 0 else 0
            class_str = "  |  ".join(f"{k}: {v}" for k, v in sorted(grains_per_class.items()))
            self.stats_label.setText(
                f"Imágenes: {total_images}  |  Con .seg: {total_seg}  |  "
                f"Granos: {total_grains}  |  Promedio/img: {avg:.1f}\n"
                f"Por clase: {class_str if class_str else '—'}"
            )
        else:
            self.stats_label.setText("Sin datos — genera segmentaciones primero.")
