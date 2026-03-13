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
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSlider, QSpinBox, QProgressBar, QFormLayout, QSplitter,
    QSizePolicy, QMessageBox, QScrollArea, QFrame, QListWidget, QListWidgetItem,
    QCheckBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orig_w = 0
        self._orig_h = 0

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
        pm = self.pixmap()
        if pm is None or pm.isNull() or self._orig_w == 0:
            return super().mousePressEvent(event)

        # Widget may be larger than pixmap — compute offset
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        off_x = (lw - pw) // 2 if lw > pw else 0
        off_y = (lh - ph) // 2 if lh > ph else 0

        rx = event.pos().x() - off_x
        ry = event.pos().y() - off_y
        if rx < 0 or ry < 0 or rx >= pw or ry >= ph:
            return super().mousePressEvent(event)

        # Map to original image coords
        ox = int(rx / pw * self._orig_w)
        oy = int(ry / ph * self._orig_h)

        if event.button() == Qt.LeftButton:
            self.left_clicked.emit(ox, oy)
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(ox, oy)


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

            # Run detector on the local crop
            detector = PollenGrainDetector(config=self.config)
            _, grains = detector.detect(crop)

            if not grains:
                self.finished.emit([])
                return

            # Pick the grain closest to the click point (in crop coords)
            click_in_crop_x = cx - x1
            click_in_crop_y = cy - y1
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

        self._init_ui()
        self.setFocusPolicy(Qt.StrongFocus)

        # Auto-cargar .seg existentes al iniciar (deferred para no bloquear)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(500, self._refresh_explorer)
        logger.info("[ContourAnalysisTab] Inicializado")

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            if self._seg_dirty:
                self._on_save_seg()
        elif key == Qt.Key_Left:
            self._on_prev_image()
        elif key == Qt.Key_Right:
            self._on_next_image()
        else:
            super().keyPressEvent(event)

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
        self.grain_info_label.setText(
            f"{n} grano{'s' if n != 1 else ''}{active_info} — "
            f"Click izq: añadir/fusionar región | Click der: eliminar contorno o UNDO"
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
        """Left-click: propose a new contour at (x_orig, y_orig) using U2-Net."""
        if self._current_image_bgr is None:
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
        """Right-click: ELIMINAR contorno cercano al click, o UNDO si no hay contorno cerca.
        
        Comportamiento inteligente:
        1. Si hay un grano cercano al click → ELIMINAR ese grano
        2. Si NO hay grano cercano → UNDO última operación del historial
        """
        if self._current_image_bgr is None:
            return
        
        # Buscar grano más cercano al click
        best_idx = -1
        best_dist = float('inf')
        
        for i, grain in enumerate(self._current_grains):
            bx, by, bw, bh = grain["bbox"]
            cx = bx + bw // 2
            cy = by + bh // 2
            dist = ((x_orig - cx) ** 2 + (y_orig - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        # Si hay un grano y el click está razonablemente cerca, ELIMINAR
        if best_idx >= 0:
            grain = self._current_grains[best_idx]
            bx, by, bw, bh = grain["bbox"]
            max_dist = max(bw, bh) * 1.5  # 1.5x el lado más grande del bbox
            
            if best_dist <= max_dist:
                # Guardar estado antes de eliminar
                self._push_history()
                
                # Eliminar el grano
                removed = self._current_grains.pop(best_idx)
                logger.info(f"[ManualContour] Eliminado grano #{best_idx} por click derecho, bbox={removed['bbox']}")
                
                # Re-indexar
                for i, g in enumerate(self._current_grains):
                    g["index"] = i
                
                self._mark_dirty()
                self._redraw_contours()
                
                # Ajustar selección
                if len(self._current_grains) == 0:
                    self._selected_grain_idx = -1
                    self._active_grain_idx = -1
                    self.crop_label.setText("Sin granos")
                    self.grain_meta_label.setText("")
                    self.grain_nav_label.setText("—")
                else:
                    self._selected_grain_idx = min(best_idx, len(self._current_grains) - 1)
                    if self._active_grain_idx == best_idx:
                        self._active_grain_idx = -1  # Desactivar si se eliminó el grano activo
                    self._show_selected_grain()
                
                self.grain_info_label.setText(
                    f"Grano #{best_idx} eliminado. {len(self._current_grains)} restantes. "
                    f"Click der en vacío para UNDO."
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

        pixmap = numpy_to_qpixmap(annotated, max_width=700, max_height=500)
        self.image_label.setPixmap(pixmap)

        n = len(self._current_grains)
        edit_tag = " ● Editado" if self._seg_dirty else ""
        self.grain_info_label.setText(
            f"{n} grano{'s' if n != 1 else ''}{edit_tag} — "
            f"Click izq: proponer contorno | Click der: eliminar contorno"
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
