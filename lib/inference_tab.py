"""
InferenceTab - Tab de inferencia con soporte para carpetas y reportes de ingeniería.
Permite inferencia en imágenes individuales o carpetas completas con generación de reportes.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QTextEdit, QFileDialog, QMessageBox,
    QComboBox, QSpinBox, QFormLayout, QProgressBar
)
from PyQt5.QtCore import QThread, pyqtSignal
import logging

from lib.styles import Styles, Colors

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """Worker thread para inferencia en carpeta usando detección de granos"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    detailed_progress = pyqtSignal(object)
    
    def __init__(self, checkpoint, folder, database, k):
        super().__init__()
        self.checkpoint = checkpoint
        self.folder = folder
        self.database = database
        self.k = k
    
    def run(self):
        """Ejecuta inferencia con detección de granos por imagen"""
        try:
            from src.grain_detection.full_image_classifier import FullImageClassifier
            from src.utils.report_generator import InferenceReportGenerator
            import numpy as np
            import cv2

            # Inicializar pipeline completo: U²-Net detector + modelo + kNN
            logger.info("Inicializando FullImageClassifier...")
            classifier = FullImageClassifier.from_checkpoint(
                checkpoint_path=self.checkpoint,
                config_path="config.yaml",
                k=self.k,
            )
            
            # Buscar todas las imágenes en la carpeta
            folder_path = Path(self.folder)
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            
            for ext in valid_extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
            # Deduplicar (glob mayúsculas puede repetir en Windows)
            image_files = sorted(set(image_files))
            
            if not image_files:
                raise ValueError("No se encontraron imágenes en la carpeta")
            
            # Crear carpeta de salida UNICA con timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = folder_path / f"inference_{timestamp}"
            output_dir.mkdir(exist_ok=True)
            
            # Procesar cada imagen con detección de granos
            predictions = []
            all_grain_details = []
            total = len(image_files)
            running_class_counts = {}  # conteo acumulado por clase
            running_total_grains = 0
            
            # Milestones cada 5% para guardar imágenes de muestra
            sample_pcts = set(range(0, 101, 5))  # {0, 5, 10, ..., 100}
            saved_samples = set()
            
            for idx, img_path in enumerate(image_files, 1):
                try:
                    # Pipeline: detectar granos → recortar → clasificar cada grano
                    results = classifier.classify_full_image(str(img_path))
                    grains = results.get("grains", [])
                    
                    if not grains:
                        logger.warning(f"No se detectaron granos en {img_path.name}")
                        predictions.append({
                            'image_path': str(img_path),
                            'predicted_class': 'NO_GRAINS_DETECTED',
                            'confidence': 0.0,
                            'mean_distance': 0.0,
                            'n_grains': 0,
                            'class_distribution': {},
                        })
                        self.progress.emit(idx, total)
                        self.detailed_progress.emit({
                            'idx': idx, 'total': total,
                            'image_name': img_path.name,
                            'grains_this': 0,
                            'total_grains': running_total_grains,
                            'class_counts': dict(running_class_counts),
                        })
                        continue
                    
                    # Agregar detalles por grano
                    for grain in grains:
                        all_grain_details.append({
                            'image_path': str(img_path),
                            'image_name': img_path.name,
                            'grain_index': grain['index'],
                            'predicted_class': grain['predicted_class'],
                            'confidence': grain['confidence'],
                            'mean_distance': grain['mean_distance'],
                            'bbox': grain['bbox'],
                        })
                    
                    # Predicción por imagen: voto mayoritario de granos detectados
                    grain_classes = [g['predicted_class'] for g in grains]
                    unique, counts = np.unique(grain_classes, return_counts=True)
                    dominant_class = unique[np.argmax(counts)]
                    dominant_conf = np.max(counts) / len(grains)
                    
                    # Distancia media de todos los granos
                    mean_dist = np.mean([g['mean_distance'] for g in grains])
                    
                    # Distribución de clases en la imagen
                    class_dist = {}
                    for cls, cnt in zip(unique, counts):
                        class_dist[cls] = float(cnt / len(grains))
                    
                    predictions.append({
                        'image_path': str(img_path),
                        'predicted_class': dominant_class,
                        'confidence': float(dominant_conf),
                        'mean_distance': float(mean_dist),
                        'n_grains': len(grains),
                        'class_distribution': class_dist,
                    })
                    
                    # Actualizar conteos acumulados
                    running_total_grains += len(grains)
                    for cls in grain_classes:
                        running_class_counts[cls] = running_class_counts.get(cls, 0) + 1
                    
                    # Guardar imagen anotada solo en milestones cada 5%
                    pct = int((idx / total) * 100)
                    save_this = False
                    for p in sample_pcts:
                        if p not in saved_samples and pct >= p:
                            saved_samples.add(p)
                            save_this = True
                    
                    if save_this:
                        try:
                            image_bgr = cv2.imread(str(img_path))
                            if image_bgr is not None:
                                annotated = classifier.draw_results(image_bgr, results)
                                out_path = output_dir / f"sample_{pct:03d}pct_{img_path.stem}.png"
                                cv2.imwrite(str(out_path), annotated)
                        except Exception as e_draw:
                            logger.warning(f"Error dibujando resultados en {img_path.name}: {e_draw}")
                    
                    self.progress.emit(idx, total)
                    self.detailed_progress.emit({
                        'idx': idx, 'total': total,
                        'image_name': img_path.name,
                        'grains_this': len(grains),
                        'total_grains': running_total_grains,
                        'class_counts': dict(running_class_counts),
                    })
                    
                except Exception as e:
                    logger.warning(f"Error procesando {img_path}: {e}")
                    predictions.append({
                        'image_path': str(img_path),
                        'predicted_class': f'ERROR: {e}',
                        'confidence': 0.0,
                        'mean_distance': 0.0,
                        'n_grains': 0,
                        'class_distribution': {},
                    })
                    self.progress.emit(idx, total)
                    self.detailed_progress.emit({
                        'idx': idx, 'total': total,
                        'image_name': img_path.name,
                        'grains_this': 0,
                        'total_grains': running_total_grains,
                        'class_counts': dict(running_class_counts),
                    })
            
            # Generar reporte
            report_gen = InferenceReportGenerator(output_dir="reports")
            
            model_info = {
                'name': 'AnalogyNet + FullImageClassifier',
                'checkpoint': self.checkpoint
            }
            
            report_path = report_gen.generate_report(
                predictions=predictions,
                folder_path=str(folder_path),
                model_info=model_info,
                class_names=classifier.class_names
            )
            
            # Guardar detalle por grano como CSV
            if all_grain_details:
                import csv
                grain_csv = output_dir / "grain_details.csv"
                with open(grain_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'image_name', 'grain_index', 'predicted_class',
                        'confidence', 'mean_distance'
                    ])
                    writer.writeheader()
                    for g in all_grain_details:
                        writer.writerow({
                            'image_name': g['image_name'],
                            'grain_index': g['grain_index'],
                            'predicted_class': g['predicted_class'],
                            'confidence': g['confidence'],
                            'mean_distance': f"{g['mean_distance']:.4f}",
                        })
                logger.info(f"Detalle por grano guardado en {grain_csv}")
            
            self.finished.emit({
                'report_path': report_path,
                'total_images': len(predictions),
                'total_grains': len(all_grain_details),
                'predictions': predictions,
                'output_dir': str(output_dir),
            })
            
        except Exception as e:
            import traceback
            logger.error(f"Error en inferencia: {traceback.format_exc()}")
            self.error.emit(str(e))


class InferenceTab(QWidget):
    """Tab de inferencia con soporte para carpetas y reportes de ingeniería"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Modelo
        model_group = QGroupBox("Modelo")
        model_layout = QFormLayout()
        
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_input = QLineEdit()
        self.checkpoint_input.setPlaceholderText("Selecciona el modelo entrenado (.pth)")
        checkpoint_btn = QPushButton("📁 Buscar")
        checkpoint_btn.clicked.connect(self.browse_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_input)
        checkpoint_layout.addWidget(checkpoint_btn)
        model_layout.addRow("Checkpoint:", checkpoint_layout)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Modo de inferencia
        mode_group = QGroupBox("Modo de Inferencia")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Carpeta Completa (con Reporte)", "Imagen Individual"])
        self.mode_combo.currentIndexChanged.connect(self.toggle_inference_mode)
        mode_layout.addWidget(self.mode_combo)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Carpeta de inferencia
        self.folder_group = QGroupBox("Carpeta a Analizar")
        folder_layout = QFormLayout()
        
        folder_dir_layout = QHBoxLayout()
        self.folder_path = QLineEdit()
        self.folder_path.setPlaceholderText("Directorio con imágenes a clasificar")
        folder_browse = QPushButton("📁 Buscar")
        folder_browse.clicked.connect(self.browse_folder)
        folder_dir_layout.addWidget(self.folder_path)
        folder_dir_layout.addWidget(folder_browse)
        folder_layout.addRow("Carpeta:", folder_dir_layout)
        
        self.folder_group.setLayout(folder_layout)
        layout.addWidget(self.folder_group)
        
        # Imagen Individual
        self.image_group = QGroupBox("Imagen a Clasificar")
        image_layout = QFormLayout()
        
        query_file_layout = QHBoxLayout()
        self.query_path = QLineEdit()
        self.query_path.setPlaceholderText("Ruta a la imagen")
        query_browse = QPushButton("📁 Buscar")
        query_browse.clicked.connect(self.browse_query)
        query_file_layout.addWidget(self.query_path)
        query_file_layout.addWidget(query_browse)
        image_layout.addRow("Imagen:", query_file_layout)
        
        self.image_group.setLayout(image_layout)
        self.image_group.setVisible(False)
        layout.addWidget(self.image_group)
        
        # Base de Datos (Opcional)
        db_group = QGroupBox("Base de Datos de Referencia (Opcional)")
        db_layout = QFormLayout()
        
        db_dir_layout = QHBoxLayout()
        self.database_path = QLineEdit("")
        self.database_path.setPlaceholderText("Opcional: usa embeddings del checkpoint si vacío")
        db_browse = QPushButton("📁 Buscar")
        db_browse.clicked.connect(self.browse_database)
        db_dir_layout.addWidget(self.database_path)
        db_dir_layout.addWidget(db_browse)
        db_layout.addRow("Directorio:", db_dir_layout)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)
        
        # Opciones
        options_group = QGroupBox("Opciones")
        options_layout = QFormLayout()
        
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 50)
        self.k_spin.setValue(5)
        options_layout.addRow("k vecinos:", self.k_spin)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Botón Ejecutar
        self.infer_btn = QPushButton("🔍 Ejecutar Inferencia")
        self.infer_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.infer_btn.clicked.connect(self.run_inference)
        layout.addWidget(self.infer_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Resultados
        results_group = QGroupBox("Resultados")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(150)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def toggle_inference_mode(self):
        """Alterna entre modo carpeta e imagen individual"""
        is_folder_mode = self.mode_combo.currentIndex() == 0
        self.folder_group.setVisible(is_folder_mode)
        self.image_group.setVisible(not is_folder_mode)
    
    def browse_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Checkpoint", "", "PyTorch Files (*.pth *.pt)"
        )
        if file_path:
            self.checkpoint_input.setText(file_path)
    
    def browse_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar Carpeta para Inferencia"
        )
        if dir_path:
            self.folder_path.setText(dir_path)
    
    def browse_query(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.query_path.setText(file_path)
    
    def browse_database(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar Directorio de Base de Datos"
        )
        if dir_path:
            self.database_path.setText(dir_path)
    
    def run_inference(self):
        """Ejecuta inferencia según el modo seleccionado"""
        is_folder_mode = self.mode_combo.currentIndex() == 0
        
        if is_folder_mode:
            self.run_folder_inference()
        else:
            self.run_single_image_inference()
    
    def run_single_image_inference(self):
        """Ejecuta inferencia en imagen individual"""
        checkpoint = self.checkpoint_input.text()
        query = self.query_path.text()
        database = self.database_path.text()
        
        # Validación
        if not checkpoint or not Path(checkpoint).exists():
            QMessageBox.warning(self, "Error", "Checkpoint no válido")
            return
        
        if not query or not Path(query).exists():
            QMessageBox.warning(self, "Error", "Imagen query no válida")
            return
        
        # Base de datos es opcional (usa embeddings del checkpoint si no se proporciona)
        if database and not Path(database).exists():
            QMessageBox.warning(self, "Error", "Directorio de base de datos no válido")
            return
        
        self.infer_btn.setEnabled(False)
        self.results_text.clear()
        self.results_text.append("🔄 Procesando...")
        
        if self.parent_window:
            self.parent_window.log_widget.append_log("Iniciando inferencia...", "INFO")
        
        # Ejecutar inferencia
        try:
            from scripts.inference import MicroscopyImageRetrieval
            
            # Base de datos opcional
            database = database if database and Path(database).exists() else None
            retrieval = MicroscopyImageRetrieval(
                config_path="config.yaml",
                checkpoint_path=checkpoint,
                database_path=database
            )
            
            k = self.k_spin.value()
            result = retrieval.classify_image(query, k=k)
            
            self.results_text.clear()
            self.results_text.append("=" * 50)
            self.results_text.append("📊 RESULTADO DE CLASIFICACIÓN")
            self.results_text.append("=" * 50)
            self.results_text.append(f"\n✅ Clase Predicha: {result['predicted_class']}")
            self.results_text.append(f"📈 Confianza: {result['confidence']:.2%}")
            self.results_text.append(f"📏 Distancia: {result['distance']:.4f}")
            
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Clasificación: {result['predicted_class']} ({result['confidence']:.2%})",
                    "SUCCESS"
                )
        
        except Exception as e:
            self.results_text.clear()
            self.results_text.append(f"❌ Error: {str(e)}")
            if self.parent_window:
                self.parent_window.log_widget.append_log(f"Error en inferencia: {e}", "ERROR")
            QMessageBox.critical(self, "Error", f"Error durante inferencia:\n{str(e)}")
        
        finally:
            self.infer_btn.setEnabled(True)
    
    def run_folder_inference(self):
        """Ejecuta inferencia en carpeta completa y genera reporte"""
        checkpoint = self.checkpoint_input.text()
        folder = self.folder_path.text()
        database = self.database_path.text()
        
        # Validación
        if not checkpoint or not Path(checkpoint).exists():
            QMessageBox.warning(self, "Error", "Checkpoint no válido")
            return
        
        if not folder or not Path(folder).exists():
            QMessageBox.warning(self, "Error", "Carpeta no válida")
            return
        
        # Base de datos es opcional
        if database and not Path(database).exists():
            QMessageBox.warning(self, "Error", "Directorio de base de datos no válido")
            return
        
        self.infer_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        self.results_text.append("🔄 Procesando carpeta...")
        
        if self.parent_window:
            self.parent_window.log_widget.append_log("Iniciando inferencia en carpeta...", "INFO")
        
        # Ejecutar en thread
        k = self.k_spin.value()
        self.worker = InferenceWorker(checkpoint, folder, database, k)
        self.worker.finished.connect(self.folder_inference_finished)
        self.worker.error.connect(self.folder_inference_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.detailed_progress.connect(self.update_detailed_progress)
        self.worker.start()
    
    def update_progress(self, current, total):
        """Actualiza barra de progreso"""
        percentage = int((current / total) * 100)
        self.progress_bar.setValue(percentage)
    
    def update_detailed_progress(self, info):
        """Muestra conteo en vivo de granos y clases identificadas"""
        idx = info['idx']
        total = info['total']
        img_name = info['image_name']
        grains_this = info['grains_this']
        total_grains = info['total_grains']
        class_counts = info['class_counts']
        pct = int((idx / total) * 100)
        
        self.results_text.clear()
        self.results_text.append(f"Procesando {idx}/{total} ({pct}%)")
        self.results_text.append(f"Ultima: {img_name} -> {grains_this} granos")
        self.results_text.append(f"")
        self.results_text.append(f"=== ACUMULADO ===")
        self.results_text.append(f"Total granos detectados: {total_grains}")
        
        if class_counts:
            self.results_text.append(f"")
            self.results_text.append(f"Clases identificadas:")
            for cls in sorted(class_counts, key=class_counts.get, reverse=True):
                cnt = class_counts[cls]
                bar = '#' * min(cnt, 40)
                self.results_text.append(f"  {cls}: {cnt} granos  {bar}")
    
    def folder_inference_finished(self, result):
        """Callback cuando termina inferencia de carpeta"""
        self.infer_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        report_path = result['report_path']
        total = result['total_images']
        total_grains = result.get('total_grains', 0)
        output_dir = result.get('output_dir', '')
        
        self.results_text.clear()
        self.results_text.append("=" * 50)
        self.results_text.append("INFERENCIA COMPLETADA")
        self.results_text.append("=" * 50)
        self.results_text.append(f"\nTotal imagenes procesadas: {total}")
        self.results_text.append(f"Total granos detectados: {total_grains}")
        
        # Resumen por clase
        predictions = result.get('predictions', [])
        class_counts = {}
        for pred in predictions:
            cls = pred.get('predicted_class', 'unknown')
            n = pred.get('n_grains', 0)
            if cls not in class_counts:
                class_counts[cls] = {'images': 0, 'grains': 0}
            class_counts[cls]['images'] += 1
            class_counts[cls]['grains'] += n
        
        if class_counts:
            self.results_text.append(f"\nDistribucion por clase:")
            for cls, info in sorted(class_counts.items()):
                self.results_text.append(
                    f"  {cls}: {info['images']} imgs, {info['grains']} granos"
                )
        
        self.results_text.append(f"\nReporte: {report_path}")
        if output_dir:
            self.results_text.append(f"Imagenes anotadas: {output_dir}")
        
        if self.parent_window:
            self.parent_window.log_widget.append_log(
                f"Inferencia completada: {total} imagenes, {total_grains} granos",
                "SUCCESS"
            )
        
        # Abrir reporte automáticamente
        reply = QMessageBox.question(
            self,
            "Reporte Generado",
            f"Inferencia completada exitosamente.\n\n"
            f"Imagenes procesadas: {total}\n"
            f"Granos detectados: {total_grains}\n"
            f"Reporte: {report_path}\n\n"
            f"Desea abrir el reporte ahora?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            import webbrowser
            webbrowser.open(str(report_path))
    
    def folder_inference_error(self, error_msg):
        """Callback cuando hay error en inferencia de carpeta"""
        self.infer_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.results_text.clear()
        self.results_text.append(f"❌ Error: {error_msg}")
        
        if self.parent_window:
            self.parent_window.log_widget.append_log(f"Error en inferencia: {error_msg}", "ERROR")
        
        QMessageBox.critical(self, "Error", f"Error durante inferencia:\n{error_msg}")
