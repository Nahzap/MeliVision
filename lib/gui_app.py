"""
MetricLearning - Aplicación GUI con PyQt5
Interfaz gráfica profesional siguiendo estándares IEEE para usabilidad.

Arquitectura: Model-View-Controller (MVC)
- Model: Lógica de negocio (src/)
- View: Interfaz gráfica (PyQt5)
- Controller: Coordinación entre Model y View
"""

import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QComboBox, QSpinBox, QProgressBar, QMessageBox, QGroupBox,
    QFormLayout, QCheckBox, QListWidget, QSplitter, QStatusBar,
    QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont, QTextCursor, QPalette, QColor, QPixmap, QImage
import datetime
import traceback

from src.utils.logging_utils import setup_logger, cleanup_old_logs
from lib.config_editor import ConfigEditorTab
from lib.dataset_manager_gui import DatasetManagerTab
from lib.inference_tab import InferenceTab
from lib.contour_analysis_tab import ContourAnalysisTab
from lib.styles import Styles, Colors, apply_global_styles


class WorkerThread(QThread):
    """
    Thread worker para operaciones largas sin bloquear la GUI.
    Sigue patrón de diseño Observer para comunicación asíncrona.
    """
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def run(self):
        """Ejecuta función en thread separado"""
        try:
            self.status.emit("Iniciando operación...")
            self.result = self.func(*self.args, **self.kwargs)
            self.finished.emit(self.result)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{str(e)}\n\nTraceback:\n{tb}")


class LogWidget(QTextEdit):
    """
    Widget personalizado para mostrar logs en tiempo real.
    Implementa patrón Singleton para logging centralizado.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(120)
        self.setMaximumHeight(250)
        self.setStyleSheet(Styles.LOG_WIDGET)
    
    def append_log(self, message, level="INFO"):
        """Agrega mensaje de log con color según nivel y timestamp"""
        colors = {
            "DEBUG": "#808080",
            "INFO": Colors.SUCCESS,
            "WARNING": Colors.WARNING,
            "ERROR": Colors.ERROR,
            "SUCCESS": Colors.SUCCESS
        }
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = colors.get(level, Colors.TEXT_LIGHT)
        formatted = f'<span style="color: {color};">[{timestamp}] [{level}] {message}</span>'
        self.append(formatted)
        
        # Auto-scroll al final
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)


class TrainingTab(QWidget):
    """
    Tab para entrenamiento del modelo.
    Implementa formulario con validación y feedback visual.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self._run_dir = None
        self._estimate_done = False
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # === Configuración ===
        config_group = QGroupBox("Configuración de Entrenamiento")
        config_layout = QFormLayout()
        
        # Config file
        config_file_layout = QHBoxLayout()
        self.config_path = QLineEdit("config.yaml")
        self.config_path.setPlaceholderText("Ruta al archivo de configuración")
        config_browse = QPushButton("📁 Buscar")
        config_browse.clicked.connect(self.browse_config)
        config_file_layout.addWidget(self.config_path)
        config_file_layout.addWidget(config_browse)
        config_layout.addRow("Archivo Config:", config_file_layout)
        
        # Checkpoint (opcional)
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText("Opcional: checkpoint para reanudar")
        checkpoint_browse = QPushButton("📁 Buscar")
        checkpoint_browse.clicked.connect(self.browse_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_path)
        checkpoint_layout.addWidget(checkpoint_browse)
        config_layout.addRow("Checkpoint:", checkpoint_layout)
        
        # Resume checkbox
        self.resume_check = QCheckBox("Reanudar entrenamiento")
        config_layout.addRow("", self.resume_check)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # === Controles ===
        controls_layout = QHBoxLayout()
        
        self.estimate_btn = QPushButton("⏱️ Estimar Tiempo")
        self.estimate_btn.clicked.connect(self.estimate_time)
        
        self.train_btn = QPushButton("🚀 Iniciar Entrenamiento")
        self.train_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        self.train_btn.setToolTip("Primero debes ejecutar 'Estimar Tiempo'")
        
        self.stop_btn = QPushButton("⏹ Detener")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        
        controls_layout.addWidget(self.estimate_btn)
        controls_layout.addWidget(self.train_btn)
        controls_layout.addWidget(self.stop_btn)
        layout.addLayout(controls_layout)
        
        # === Progress ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # === Status ===
        self.status_label = QLabel("Listo para entrenar")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # === Terminal de Entrenamiento ===
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                padding: 10px;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
        """)
        self.output_area.setPlaceholderText("Presiona '⏱️ Estimar Tiempo' para generar el análisis pre-entrenamiento...")
        layout.addWidget(self.output_area, 1)
        
        self.setLayout(layout)
    
    def browse_config(self):
        """Buscar archivo de configuración"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Config", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path.setText(file_path)
    
    def browse_checkpoint(self):
        """Buscar checkpoint"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Checkpoint", "", "PyTorch Files (*.pth *.pt)"
        )
        if file_path:
            self.checkpoint_path.setText(file_path)
    
    def _build_args(self):
        """Construye argparse.Namespace desde los campos de la UI"""
        import argparse
        config_path = self.config_path.text()
        checkpoint_path = self.checkpoint_path.text()
        return argparse.Namespace(
            config=config_path,
            resume=checkpoint_path if (self.resume_check.isChecked() and checkpoint_path) else None,
            gpu=None
        )
    
    def _validate_inputs(self):
        """Valida inputs y retorna True si son válidos"""
        config_path = self.config_path.text()
        if not config_path or not Path(config_path).exists():
            QMessageBox.warning(self, "Error", "Archivo de configuración no válido")
            return False
        checkpoint_path = self.checkpoint_path.text()
        if checkpoint_path and not Path(checkpoint_path).exists():
            QMessageBox.warning(self, "Error", "Checkpoint no encontrado")
            return False
        return True
    
    def estimate_time(self):
        """Estima tiempo de entrenamiento en background y escribe resultados al log"""
        if not self._validate_inputs():
            return
        
        self.estimate_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("⏱️ Estimando tiempo de entrenamiento...")
        self.parent_window.log_widget.append_log("Estimando tiempo de entrenamiento...", "INFO")
        
        from scripts.train import run_training
        args = self._build_args()
        self.estimate_worker = WorkerThread(run_training, args, estimate_only=True)
        self.estimate_worker.finished.connect(self._on_estimate_ready)
        self.estimate_worker.error.connect(self._on_estimate_error)
        self.estimate_worker.start()
    
    def _on_estimate_error(self, error_msg):
        """Callback cuando falla la estimación"""
        self.estimate_btn.setEnabled(True)
        self.train_btn.setEnabled(self._estimate_done)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Error en estimación")
        self.parent_window.log_widget.append_log(f"Error estimando: {error_msg.split(chr(10))[0]}", "ERROR")
    
    def _on_estimate_ready(self, estimate):
        """Callback cuando la estimación termina — muestra análisis pre-entrenamiento en output_area"""
        self.estimate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if not isinstance(estimate, dict):
            self.status_label.setText("❌ Error en estimación")
            self.parent_window.log_widget.append_log("Estimación no retornó datos válidos", "ERROR")
            return
        
        # Store run_dir and enable training
        self._run_dir = estimate.get('run_dir')
        self._estimate_done = True
        self.train_btn.setEnabled(True)
        self.train_btn.setToolTip("")
        
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        analysis = estimate.get('analysis', {})
        charts = analysis.get('charts', {})
        run_paths = analysis.get('run_paths', {})
        
        stats = analysis.get('stats', {})
        train_stats = stats.get('train', {})
        
        sep = '<span style="color: #569cd6;">' + '═' * 60 + '</span><br>'
        hdr = lambda t: f'<span style="color: #4ec9b0; font-size: 14px; font-weight: bold;">  {t}</span><br>'
        lbl = lambda t: f'<span style="color: #dcdcaa;">{t}</span>'
        val = lambda t: f'<span style="color: #d4d4d4;">{t}</span>'
        dim = lambda t: f'<span style="color: #808080;">{t}</span>'
        
        html_parts = [
            '<div style="font-family: Consolas, monospace; color: #d4d4d4; background: #1e1e1e; padding: 8px;">',
        ]
        
        # === SECTION 1: Run info ===
        html_parts += [
            sep,
            f'<span style="color: #4ec9b0; font-size: 16px; font-weight: bold;">  📋 ANÁLISIS PRE-ENTRENAMIENTO</span><br>',
            sep,
            dim(f'  [{ts}]') + '<br>',
        ]
        if run_paths:
            html_parts.append(dim(f'  📁 {run_paths.get("run_dir", "N/A")}') + '<br>')
        html_parts.append('<br>')
        
        # === SECTION 2: Model & Hyperparameters ===
        html_parts += [
            hdr('🧠 Modelo & Hiperparámetros'),
            val(f'     Backbone:      <b>{estimate.get("model_backbone", "N/A")}</b>') + '<br>',
            val(f'     Embedding dim: <b>{estimate.get("embedding_dim", "N/A")}</b>') + '<br>',
            val(f'     Loss:          <b>{estimate.get("loss_type", "N/A")}</b>') + '<br>',
            val(f'     Optimizer:     <b>{estimate.get("optimizer", "N/A")}</b>') + '<br>',
            val(f'     Learning rate: <b>{estimate.get("learning_rate", "N/A")}</b>') + '<br>',
            val(f'     Scheduler:     <b>{estimate.get("scheduler", "N/A")}</b>') + '<br>',
            val(f'     Mixed prec:    <b>{"Sí" if estimate.get("use_amp") else "No"}</b>') + '<br>',
            val(f'     Workers:       <b>{estimate.get("num_workers", "N/A")}</b>') + '<br>',
            '<br>',
        ]
        
        # === SECTION 3: Dataset summary ===
        html_parts += [
            hdr('📊 Dataset'),
            val(f'     Train:      <b>{estimate["train_samples"]}</b> imágenes') + '<br>',
            val(f'     Val:        <b>{estimate["val_samples"]}</b> imágenes') + '<br>',
            val(f'     Clases:     <b>{estimate.get("num_classes", "N/A")}</b>') + '<br>',
            val(f'     Batch size: <b>{estimate["batch_size"]}</b>') + '<br>',
        ]
        
        # Image size stats
        img_sizes = train_stats.get('image_sizes', {})
        if img_sizes:
            html_parts += [
                '<br>' + lbl('     Tamaño de imágenes (muestreo):') + '<br>',
                val(f'       Ancho:  min={img_sizes["width_min"]}  max={img_sizes["width_max"]}  media={img_sizes["width_mean"]}') + '<br>',
                val(f'       Alto:   min={img_sizes["height_min"]}  max={img_sizes["height_max"]}  media={img_sizes["height_mean"]}') + '<br>',
                val(f'       Aspect ratio medio: <b>{img_sizes["aspect_ratio_mean"]}</b>') + '<br>',
                dim(f'       (muestreadas {img_sizes["sampled"]} imágenes)') + '<br>',
            ]
        
        # Per-class distribution table
        class_dist = train_stats.get('class_distribution', {})
        if class_dist:
            html_parts.append('<br>' + lbl('     Distribución por clase:') + '<br>')
            max_name_len = max(len(n) for n in class_dist.keys()) if class_dist else 10
            for cls_name, count in class_dist.items():
                pct = count / estimate["train_samples"] * 100 if estimate["train_samples"] > 0 else 0
                bar_len = int(pct / 2)
                bar = '█' * bar_len + '░' * (50 - bar_len)
                html_parts.append(
                    val(f'       {cls_name:<{max_name_len}}  {count:>5}  ({pct:5.1f}%)  ') +
                    f'<span style="color: #4ec9b0;">{bar}</span><br>'
                )
            if train_stats.get('min_samples_per_class') is not None:
                html_parts.append(dim(
                    f'       Min: {train_stats["min_samples_per_class"]} | '
                    f'Max: {train_stats["max_samples_per_class"]} | '
                    f'Media: {train_stats["mean_samples_per_class"]} | '
                    f'Std: {train_stats["std_samples_per_class"]}'
                ) + '<br>')
        html_parts.append('<br>')
        
        # === SECTION 4: Benchmark results ===
        html_parts += [
            hdr('⚡ Benchmark de Rendimiento'),
            val(f'     Forward+Loss+Backward+Opt: <b>{estimate.get("avg_train_step_sec", 0):.3f}s</b>/batch') + '<br>',
            val(f'     Data loading:              <b>{estimate.get("avg_data_load_sec", 0):.3f}s</b>/batch') + '<br>',
            val(f'     Overhead factor:           <b>{estimate.get("data_overhead_factor", 1):.2f}x</b>') + '<br>',
            val(f'     Validación:                <b>{estimate["avg_val_batch_sec"]:.3f}s</b>/batch') + '<br>',
            '<br>',
        ]
        
        # === SECTION 5: Time estimate ===
        html_parts += [
            hdr('⏱️ Estimación de Tiempo'),
            val(f'     Entrenamiento: {estimate["train_batches_per_epoch"]} batches × '
                f'{estimate["avg_train_batch_sec"]:.2f}s = <b>{estimate["train_epoch_fmt"]}</b>') + '<br>',
            val(f'     Validación:    {estimate["val_batches_per_epoch"]} batches × '
                f'{estimate["avg_val_batch_sec"]:.2f}s + recall = <b>{estimate["val_epoch_fmt"]}</b>') + '<br>',
            val(f'     Ref. embeddings (una vez, post-training):  <b>{estimate.get("ref_embed_overhead_sec", 0):.0f}s</b>') + '<br>',
            val(f'     Total época:   <b>{estimate["epoch_total_fmt"]}</b>') + '<br>',
            '<br>',
            f'<span style="color: #4ec9b0; font-size: 15px; font-weight: bold;">'
            f'  🕐 TOTAL ESTIMADO: {estimate["epochs"]} épocas × '
            f'{estimate["epoch_total_fmt"]} + ref_embed = {estimate["total_fmt"]}</span><br>',
            '<br>', sep,
        ]
        
        # === SECTION 6: Pre-training chart images ===
        chart_titles = {
            'class_distribution': '📊 Distribución de Clases',
            'train_val_balance': '⚖️ Balance Train vs Validación',
            'sample_grid': '🖼️ Muestras por Clase (Top 10)',
            'augmentation_preview': '🔄 Vista Previa de Aumentación',
        }
        
        for key, title in chart_titles.items():
            img_path = charts.get(key)
            if img_path and Path(img_path).exists():
                abs_path = str(Path(img_path).resolve()).replace('\\', '/')
                html_parts.append(f'<br>{hdr(title)}')
                html_parts.append(f'<img src="file:///{abs_path}" width="750"><br>')
        
        html_parts.append('</div>')
        
        self.output_area.clear()
        self.output_area.setHtml("".join(html_parts))
        
        # Scroll to top
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.output_area.setTextCursor(cursor)
        
        self.status_label.setText(f"✅ Análisis listo — Estimado: {estimate['total_fmt']} ({estimate['epochs']} épocas) — Listo para entrenar")
        self.parent_window.log_widget.append_log(
            f"Análisis pre-entrenamiento completado: {estimate['total_fmt']} total ({estimate['epochs']} épocas)", "SUCCESS"
        )
        self.parent_window.log_widget.append_log(
            f"Carpeta de resultados: {self._run_dir}", "INFO"
        )
    
    def start_training(self):
        """Inicia entrenamiento directamente en thread separado"""
        if not self._validate_inputs():
            return
        
        if not self._estimate_done or not self._run_dir:
            QMessageBox.warning(self, "Estimación requerida",
                "Debes ejecutar 'Estimar Tiempo' antes de iniciar el entrenamiento.\n"
                "Esto crea la carpeta de resultados y el análisis pre-entrenamiento.")
            return
        
        args = self._build_args()
        
        self.train_btn.setEnabled(False)
        self.estimate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("🔄 Entrenando modelo...")
        
        self.parent_window.log_widget.append_log("Iniciando entrenamiento...", "INFO")
        self.parent_window.log_widget.append_log(f"Config: {args.config}", "DEBUG")
        self.parent_window.log_widget.append_log(f"Resume: {args.resume or 'No'}", "DEBUG")
        self.parent_window.log_widget.append_log(f"Run dir: {self._run_dir}", "DEBUG")
        
        from scripts.train import run_training
        self.worker = WorkerThread(run_training, args, run_dir=self._run_dir)
        self.worker.finished.connect(self.training_finished)
        self.worker.error.connect(self.training_error)
        self.worker.start()
    
    def stop_training(self):
        """Detiene entrenamiento"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        self.training_finished(None)
        self.parent_window.log_widget.append_log("Entrenamiento detenido por usuario", "WARNING")
    
    def training_finished(self, result):
        """Callback cuando termina entrenamiento"""
        self._estimate_done = False
        self._run_dir = None
        self.train_btn.setEnabled(False)
        self.train_btn.setToolTip("Primero debes ejecutar 'Estimar Tiempo'")
        self.estimate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("✅ Entrenamiento completado")
        self.parent_window.log_widget.append_log("Entrenamiento finalizado", "SUCCESS")
        
        QMessageBox.information(self, "Éxito", "Entrenamiento completado exitosamente")
    
    def training_error(self, error_msg):
        """Callback cuando hay error"""
        self._estimate_done = False
        self._run_dir = None
        self.train_btn.setEnabled(False)
        self.train_btn.setToolTip("Primero debes ejecutar 'Estimar Tiempo'")
        self.estimate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Error en entrenamiento")
        
        # Log each line of the error for readability
        for line in error_msg.split('\n'):
            line = line.strip()
            if line:
                self.parent_window.log_widget.append_log(line, "ERROR")
        
        # Show only the first line in the dialog (short message)
        short_msg = error_msg.split('\n')[0]
        QMessageBox.critical(self, "Error", f"Error durante entrenamiento:\n{short_msg}\n\nRevisa el Registro de Actividad para más detalles.")


class EvaluationTab(QWidget):
    """
    Tab para evaluación del modelo.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # === Configuración ===
        config_group = QGroupBox("Configuración de Evaluación")
        config_layout = QFormLayout()
        
        # Checkpoint
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText("Ruta al checkpoint del modelo")
        checkpoint_browse = QPushButton("📁 Buscar")
        checkpoint_browse.clicked.connect(self.browse_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_path)
        checkpoint_layout.addWidget(checkpoint_browse)
        config_layout.addRow("Checkpoint:", checkpoint_layout)
        
        # Output dir
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit("results/")
        self.output_path.setPlaceholderText("Directorio de salida")
        output_browse = QPushButton("📁 Buscar")
        output_browse.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_path)
        output_layout.addWidget(output_browse)
        config_layout.addRow("Salida:", output_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # === Controles ===
        self.evaluate_btn = QPushButton("📊 EVALUAR MODELO")
        self.evaluate_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.evaluate_btn.clicked.connect(self.run_evaluation)
        layout.addWidget(self.evaluate_btn)
        
        # === Progress ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(Styles.PROGRESS_BAR)
        layout.addWidget(self.progress_bar)
        
        # === Status ===
        self.status_label = QLabel("✅ Listo para evaluar")
        self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def browse_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Checkpoint", "", "PyTorch Files (*.pth *.pt)"
        )
        if file_path:
            self.checkpoint_path.setText(file_path)
    
    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar Directorio de Salida"
        )
        if dir_path:
            self.output_path.setText(dir_path)
    
    def run_evaluation(self):
        """Ejecuta evaluacion completa con post-training analysis"""
        checkpoint = self.checkpoint_path.text()
        output_dir = self.output_path.text()
        
        if not checkpoint or not Path(checkpoint).exists():
            QMessageBox.warning(self, "Error", "Checkpoint no valido o no existe")
            return
        
        self.evaluate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Evaluando modelo...")
        self.status_label.setStyleSheet(Styles.LABEL_INFO if hasattr(Styles, 'LABEL_INFO') else "")
        
        self.parent_window.log_widget.append_log(f"Evaluacion: {checkpoint}", "INFO")
        self.parent_window.log_widget.append_log(f"Salida: {output_dir}", "INFO")
        
        from scripts.post_training import run_evaluation_from_checkpoint
        
        self.worker = WorkerThread(
            run_evaluation_from_checkpoint,
            checkpoint, 'config.yaml', output_dir
        )
        self.worker.finished.connect(self.evaluation_finished)
        self.worker.error.connect(self.evaluation_error)
        self.worker.start()
    
    def evaluation_finished(self, result):
        self.evaluate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Evaluacion completada")
        self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
        
        output_dir = self.output_path.text()
        acc = result.get('accuracy', 0) if isinstance(result, dict) else 0
        self.parent_window.log_widget.append_log(
            f"Evaluacion finalizada — Accuracy: {acc:.1%}", "SUCCESS"
        )
        
        QMessageBox.information(
            self, "Evaluacion Completa",
            f"Resultados guardados en: {output_dir}\n\n"
            f"Accuracy (kNN): {acc:.1%}\n\n"
            f"Archivos generados:\n"
            f"  - t-SNE 3D + 2D\n"
            f"  - Confusion Matrix\n"
            f"  - Distance Distribution\n"
            f"  - Inference Preview (10 imgs)\n"
            f"  - Evaluation Report (.md)"
        )
    
    def evaluation_error(self, error_msg):
        self.evaluate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Error en evaluación")
        
        for line in error_msg.split('\n'):
            line = line.strip()
            if line:
                self.parent_window.log_widget.append_log(line, "ERROR")
        
        short_msg = error_msg.split('\n')[0]
        QMessageBox.critical(self, "Error", f"Error durante evaluación:\n{short_msg}\n\nRevisa el Registro de Actividad para más detalles.")


class MainWindow(QMainWindow):
    """
    Ventana principal de la aplicación.
    Implementa patrón MVC y mejores prácticas de usabilidad IEEE.
    """
    def __init__(self):
        super().__init__()
        self.logger = setup_logger("gui_app")
        self.init_ui()
        
        # Limpieza de logs al iniciar
        cleanup_old_logs(days_to_keep=7)
    
    def init_ui(self):
        """Inicializa interfaz de usuario"""
        self.setWindowTitle("MetricLearning - Sistema de Clasificación de Imágenes")
        self.setGeometry(50, 50, 1400, 900)
        self.setMinimumSize(1000, 700)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)
        
        # === Header ===
        self.header = QLabel("MetricLearning")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet(Styles.HEADER_MAIN)
        self.header.setFixedHeight(50)
        main_layout.addWidget(self.header)
        
        # === Splitter: Tabs arriba, Log abajo ===
        splitter = QSplitter(Qt.Vertical)
        
        # === Tabs ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(Styles.TAB_WIDGET)
        
        # Agregar tabs — dataset tab wrapped in scroll area
        self.config_tab = ConfigEditorTab("config.yaml", self)
        self.dataset_tab = DatasetManagerTab(self)
        self.contour_tab = ContourAnalysisTab(self)
        self.training_tab = TrainingTab(self)
        self.inference_tab = InferenceTab(self)
        self.evaluation_tab = EvaluationTab(self)
        
        # Wrap dataset tab in scroll area (it has too much content)
        dataset_scroll = QScrollArea()
        dataset_scroll.setWidgetResizable(True)
        dataset_scroll.setWidget(self.dataset_tab)
        dataset_scroll.setFrameShape(QScrollArea.NoFrame)
        
        # Wrap contour tab in scroll area
        contour_scroll = QScrollArea()
        contour_scroll.setWidgetResizable(True)
        contour_scroll.setWidget(self.contour_tab)
        contour_scroll.setFrameShape(QScrollArea.NoFrame)
        
        self.tabs.addTab(self.config_tab, "⚙️ Configuración")
        self.tabs.addTab(dataset_scroll, "📊 Gestor de Datos")
        self.tabs.addTab(contour_scroll, "🔬 Análisis de Contornos")
        self.tabs.addTab(self.training_tab, "🚀 Entrenamiento")
        self.tabs.addTab(self.inference_tab, "🔍 Inferencia")
        self.tabs.addTab(self.evaluation_tab, "� Evaluación")
        
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        splitter.addWidget(self.tabs)
        
        # === Log Widget ===
        log_container = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(4)
        
        log_header_layout = QHBoxLayout()
        log_label = QLabel("Registro de Actividad")
        log_label.setStyleSheet("font-weight: bold; color: #0e639c; font-size: 10pt; padding: 2px;")
        log_header_layout.addWidget(log_label)
        log_header_layout.addStretch()
        
        clear_btn = QPushButton("🗑️ Limpiar Logs")
        clear_btn.setFixedWidth(120)
        clear_btn.clicked.connect(self.log_widget_clear)
        log_header_layout.addWidget(clear_btn)
        log_layout.addLayout(log_header_layout)
        
        self.log_widget = LogWidget()
        log_layout.addWidget(self.log_widget)
        
        log_container.setLayout(log_layout)
        splitter.addWidget(log_container)
        
        # Proportions: 75% tabs, 25% log
        splitter.setSizes([650, 250])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        central_widget.setLayout(main_layout)
        
        # === Status Bar ===
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo")
        
        # === Menu Bar ===
        self.create_menu_bar()
        
        # Log inicial
        self.log_widget.append_log("Aplicación iniciada correctamente", "SUCCESS")
        self.logger.info("GUI iniciada")
    
    def log_widget_clear(self):
        """Limpia el widget de logs"""
        self.log_widget.clear()
        self.log_widget.append_log("Logs limpiados", "INFO")
    
    def _on_tab_changed(self, index):
        """Callback cuando se cambia de tab — actualiza header y registra"""
        tab_titles = {
            0: "Configuracion del Sistema",
            1: "Gestion de Datasets",
            2: "Analisis de Contornos",
            3: "Entrenamiento de Modelos",
            4: "Inferencia y Prediccion",
            5: "Evaluacion y Resultados",
        }
        tab_names = ["Configuracion", "Gestor de Datos", "Contornos", "Entrenamiento", "Inferencia", "Evaluacion"]
        if 0 <= index < len(tab_names):
            self.header.setText(f"MetricLearning  |  {tab_titles.get(index, '')}")
            self.status_bar.showMessage(f"Tab: {tab_names[index]}")
            self.log_widget.append_log(f"Navegacion -> {tab_names[index]}", "DEBUG")
    
    def create_menu_bar(self):
        """Crea barra de menú"""
        menubar = self.menuBar()
        
        # Archivo
        file_menu = menubar.addMenu("&Archivo")
        
        exit_action = file_menu.addAction("&Salir")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Ayuda
        help_menu = menubar.addMenu("&Ayuda")
        
        about_action = help_menu.addAction("&Acerca de")
        about_action.triggered.connect(self.show_about)
        
        docs_action = help_menu.addAction("&Documentación")
        docs_action.triggered.connect(self.show_docs)
    
    def show_about(self):
        """Muestra diálogo Acerca de"""
        QMessageBox.about(
            self,
            "Acerca de MetricLearning",
            """
            <h2>MetricLearning v2.0</h2>
            <p>Sistema de clasificación de imágenes usando Metric Learning.</p>
            <p><b>Características:</b></p>
            <ul>
                <li>Arquitectura ResNet50 + Projection Head</li>
                <li>Multi-Similarity Loss</li>
                <li>Clasificación k-NN</li>
                <li>Interfaz gráfica profesional</li>
            </ul>
            <p><b>Desarrollado con:</b> PyTorch, PyQt5</p>
            """
        )
    
    def show_docs(self):
        """Muestra documentación"""
        QMessageBox.information(
            self,
            "Documentación",
            """
            <h3>Documentación disponible en:</h3>
            <p>📁 <b>docs/README_ANALISIS.md</b></p>
            <p>📁 <b>docs/NUEVA_ESTRUCTURA_PROYECTO.md</b></p>
            <p>📁 <b>README.md</b></p>
            <br>
            <p>Para más información, consulta los archivos de documentación en la carpeta docs/</p>
            """
        )
    
    def closeEvent(self, event):
        """Maneja cierre de aplicación"""
        reply = QMessageBox.question(
            self,
            "Confirmar Salida",
            "¿Está seguro que desea salir?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.logger.info("Aplicación cerrada por usuario")
            event.accept()
        else:
            event.ignore()


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    
    # Aplicar estilos globales centralizados
    apply_global_styles(app)
    
    # Crear y mostrar ventana principal
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
