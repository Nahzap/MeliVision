"""
Editor de Configuración Visual para config.yaml
Implementa mejores prácticas IEEE 9241-143 (Formularios)

Características:
- Validación en tiempo real
- Feedback visual inmediato
- Agrupación lógica de campos
- Tooltips informativos
- Guardado automático con backup
"""

import yaml
import shutil
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QPushButton, QScrollArea, QLabel, QMessageBox, QFileDialog,
    QTabWidget, QTextEdit, QDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette

# Import para auto-configuración de backbones
from src.data.transforms_enhanced import get_input_size_for_backbone


class ConfigChangesDialog(QDialog):
    """
    Diálogo para mostrar cambios automáticos aplicados al cambiar backbone.
    Permite al usuario ver y copiar los cambios realizados.
    """
    def __init__(self, changes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Información - Cambios Aplicados")
        self.setModal(False)  # No modal para poder seguir trabajando
        self.setMinimumWidth(500)
        self.setMinimumHeight(300)
        
        layout = QVBoxLayout()
        
        # Header
        header = QLabel("✅ Configuración Actualizada Automáticamente")
        header.setStyleSheet("""
            QLabel {
                font-size: 12pt;
                font-weight: bold;
                color: #0e639c;
                padding: 10px;
                background-color: #e8f4f8;
                border-radius: 5px;
            }
        """)
        layout.addWidget(header)
        
        # Descripción
        desc = QLabel("Los siguientes parámetros se actualizaron según las características del backbone seleccionado:")
        desc.setWordWrap(True)
        desc.setStyleSheet("padding: 10px; color: #666;")
        layout.addWidget(desc)
        
        # Lista de cambios
        self.changes_text = QTextEdit()
        self.changes_text.setReadOnly(True)
        self.changes_text.setStyleSheet("""
            QTextEdit {
                background-color: #f9f9f9;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 8px;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }
        """)
        
        # Formatear cambios
        changes_html = "<table cellpadding='5' style='width: 100%;'>"
        changes_html += "<tr style='background-color: #e0e0e0; font-weight: bold;'>"
        changes_html += "<td>Parámetro</td><td>Valor Anterior</td><td>→</td><td>Nuevo Valor</td></tr>"
        
        for param, old_val, new_val in changes:
            changes_html += f"<tr>"
            changes_html += f"<td style='color: #0e639c;'><b>{param}</b></td>"
            changes_html += f"<td style='color: #888;'>{old_val}</td>"
            changes_html += f"<td>→</td>"
            changes_html += f"<td style='color: #4ec9b0; font-weight: bold;'>{new_val}</td>"
            changes_html += "</tr>"
        
        changes_html += "</table>"
        self.changes_text.setHtml(changes_html)
        layout.addWidget(self.changes_text)
        
        # Nota informativa
        note = QLabel("💡 Estos cambios se aplicaron automáticamente pero AÚN NO se han guardado en config.yaml")
        note.setWordWrap(True)
        note.setStyleSheet("""
            QLabel {
                background-color: #fff4e6;
                border: 1px solid #ffa94d;
                border-radius: 3px;
                padding: 8px;
                color: #d9480f;
            }
        """)
        layout.addWidget(note)
        
        # Botones
        button_box = QDialogButtonBox()
        copy_btn = button_box.addButton("📋 Copiar", QDialogButtonBox.ActionRole)
        copy_btn.clicked.connect(self.copy_to_clipboard)
        close_btn = button_box.addButton("Cerrar", QDialogButtonBox.AcceptRole)
        close_btn.clicked.connect(self.accept)
        
        layout.addWidget(button_box)
        self.setLayout(layout)
    
    def copy_to_clipboard(self):
        """Copia los cambios al portapapeles"""
        from PyQt5.QtWidgets import QApplication
        text = self.changes_text.toPlainText()
        QApplication.clipboard().setText(text)
        QMessageBox.information(self, "Copiado", "Cambios copiados al portapapeles")


class ConfigField(QWidget):
    """
    Widget base para campos de configuración con validación.
    Implementa IEEE 9241-143: Diseño de formularios.
    """
    valueChanged = pyqtSignal()
    
    def __init__(self, label, tooltip="", parent=None):
        super().__init__(parent)
        self.label_text = label
        self.tooltip_text = tooltip
        self.is_valid = True
        
    def get_value(self):
        """Obtiene valor del campo"""
        raise NotImplementedError
    
    def set_value(self, value):
        """Establece valor del campo"""
        raise NotImplementedError
    
    def validate(self):
        """Valida el campo"""
        return True
    
    def mark_invalid(self, message=""):
        """Marca campo como inválido"""
        self.is_valid = False
        self.setStyleSheet("border: 2px solid #f48771; border-radius: 3px;")
        if message:
            self.setToolTip(f"❌ {message}")
    
    def mark_valid(self):
        """Marca campo como válido"""
        self.is_valid = True
        self.setStyleSheet("")
        self.setToolTip(self.tooltip_text)


class StringField(ConfigField):
    """Campo de texto simple"""
    def __init__(self, label, default="", tooltip="", placeholder="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QLineEdit()
        self.widget.setText(default)
        self.widget.setPlaceholderText(placeholder)
        self.widget.setToolTip(tooltip)
        self.widget.textChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.text()
    
    def set_value(self, value):
        self.widget.setText(str(value))


class IntField(ConfigField):
    """Campo numérico entero"""
    def __init__(self, label, default=0, min_val=0, max_val=1000, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QSpinBox()
        self.widget.setRange(min_val, max_val)
        self.widget.setValue(default)
        self.widget.setToolTip(tooltip)
        self.widget.valueChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.value()
    
    def set_value(self, value):
        self.widget.setValue(int(value))


class FloatField(ConfigField):
    """Campo numérico decimal"""
    def __init__(self, label, default=0.0, min_val=0.0, max_val=1.0, 
                 decimals=4, step=0.01, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QDoubleSpinBox()
        self.widget.setRange(min_val, max_val)
        self.widget.setDecimals(decimals)
        self.widget.setSingleStep(step)
        self.widget.setValue(default)
        self.widget.setToolTip(tooltip)
        self.widget.valueChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.value()
    
    def set_value(self, value):
        self.widget.setValue(float(value))


class BoolField(ConfigField):
    """Campo booleano (checkbox)"""
    def __init__(self, label, default=False, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QCheckBox()
        self.widget.setChecked(default)
        self.widget.setToolTip(tooltip)
        self.widget.stateChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.isChecked()
    
    def set_value(self, value):
        self.widget.setChecked(bool(value))


class ChoiceField(ConfigField):
    """Campo de selección (combobox)"""
    def __init__(self, label, choices, default=None, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QComboBox()
        self.widget.addItems(choices)
        if default:
            index = self.widget.findText(default)
            if index >= 0:
                self.widget.setCurrentIndex(index)
        self.widget.setToolTip(tooltip)
        self.widget.currentTextChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        return self.widget.currentText()
    
    def set_value(self, value):
        index = self.widget.findText(str(value))
        if index >= 0:
            self.widget.setCurrentIndex(index)


class PathField(ConfigField):
    """Campo de ruta con botón de búsqueda"""
    def __init__(self, label, default="", tooltip="", is_dir=False, parent=None):
        super().__init__(label, tooltip, parent)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.widget = QLineEdit()
        self.widget.setText(default)
        self.widget.setToolTip(tooltip)
        self.widget.textChanged.connect(self.valueChanged.emit)
        
        self.browse_btn = QPushButton("📁")
        self.browse_btn.setMaximumWidth(40)
        self.browse_btn.clicked.connect(lambda: self.browse(is_dir))
        
        layout.addWidget(self.widget)
        layout.addWidget(self.browse_btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.widget_container = container
    
    def browse(self, is_dir):
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo")
        
        if path:
            self.widget.setText(path)
    
    def get_value(self):
        return self.widget.text()
    
    def set_value(self, value):
        self.widget.setText(str(value))


class ListField(ConfigField):
    """Campo para listas (valores separados por comas)"""
    def __init__(self, label, default=None, tooltip="", parent=None):
        super().__init__(label, tooltip, parent)
        self.widget = QLineEdit()
        if default:
            self.widget.setText(", ".join(map(str, default)))
        self.widget.setPlaceholderText("Valores separados por comas")
        self.widget.setToolTip(tooltip)
        self.widget.textChanged.connect(self.valueChanged.emit)
    
    def get_value(self):
        text = self.widget.text().strip()
        if not text:
            return []
        # Intenta convertir a números si es posible
        values = [v.strip() for v in text.split(',')]
        try:
            return [float(v) if '.' in v else int(v) for v in values]
        except:
            return values
    
    def set_value(self, value):
        if isinstance(value, list):
            self.widget.setText(", ".join(map(str, value)))
        else:
            self.widget.setText(str(value))


class ConfigEditorTab(QWidget):
    """
    Tab principal del editor de configuración.
    Implementa IEEE 9241-143 para diseño de formularios.
    """
    configChanged = pyqtSignal()
    
    def __init__(self, config_path="config.yaml", parent=None):
        super().__init__(parent)
        self.config_path = Path(config_path)
        self.fields = {}
        self.config_data = {}
        self.parent_window = parent
        
        self.init_ui()
        self.load_config()
    
    def init_ui(self):
        """Inicializa interfaz de usuario"""
        main_layout = QVBoxLayout()
        
        # Header con info
        header = QLabel("⚙️ Editor de Configuración")
        header.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #0e639c;
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(header)
        
        # Info del archivo
        self.file_info = QLabel(f"📄 Archivo: {self.config_path.absolute()}")
        self.file_info.setStyleSheet("padding: 5px; color: #666;")
        main_layout.addWidget(self.file_info)
        
        # Scroll area para formulario
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Widget contenedor del formulario
        form_widget = QWidget()
        form_layout = QVBoxLayout()
        
        # === TABS para secciones ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 5px;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0e639c;
                color: white;
            }
        """)
        
        # Crear tabs por sección
        self.create_model_tab()
        self.create_data_tab()
        self.create_augmentation_tab()
        self.create_training_tab()
        self.create_inference_tab()
        self.create_paths_tab()
        self.create_hardware_tab()
        
        form_layout.addWidget(self.tabs)
        form_widget.setLayout(form_layout)
        scroll.setWidget(form_widget)
        main_layout.addWidget(scroll)
        
        # Botones de acción
        buttons_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("📂 Recargar")
        self.load_btn.clicked.connect(self.load_config)
        self.load_btn.setToolTip("Recargar configuración desde archivo")
        
        self.validate_btn = QPushButton("✓ Validar")
        self.validate_btn.clicked.connect(self.validate_all)
        self.validate_btn.setToolTip("Validar todos los campos")
        
        self.save_btn = QPushButton("💾 Guardar")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        self.save_btn.clicked.connect(self.save_config)
        self.save_btn.setToolTip("Guardar cambios en config.yaml")
        
        self.reset_btn = QPushButton("↺ Restaurar")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.reset_btn.setToolTip("Restaurar valores originales")
        
        buttons_layout.addWidget(self.load_btn)
        buttons_layout.addWidget(self.validate_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.reset_btn)
        buttons_layout.addWidget(self.save_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # Status
        self.status_label = QLabel("Listo")
        self.status_label.setStyleSheet("padding: 5px; color: #666;")
        main_layout.addWidget(self.status_label)
        
        self.setLayout(main_layout)
    
    def create_model_tab(self):
        """Tab de configuración del modelo"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Experiment group (nuevo: tracking y metadata)
        exp_group = QGroupBox("Experimento")
        exp_layout = QFormLayout()
        
        self.fields['experiment.name'] = StringField(
            "Name",
            default="dinov2_vits14_baseline",
            tooltip="Nombre del experimento para tracking",
            placeholder="nombre_modelo_baseline"
        )
        exp_layout.addRow("Name:", self.fields['experiment.name'].widget)
        
        self.fields['experiment.description'] = StringField(
            "Description",
            default="Baseline experiment",
            tooltip="Descripción breve del experimento",
            placeholder="Descripción del modelo y configuración"
        )
        exp_layout.addRow("Description:", self.fields['experiment.description'].widget)
        
        exp_group.setLayout(exp_layout)
        layout.addWidget(exp_group)
        
        # Model group
        model_group = QGroupBox("Arquitectura del Modelo")
        model_layout = QFormLayout()
        
        self.fields['model.backbone_name'] = ChoiceField(
            "Backbone",
            ['dinov2_vits14', 'dinov2_vitb14', 'resnet50', 'convnext_v2_tiny',
             'deit_small_patch16_224', 'efficientnet_b4', 'swin_tiny_patch4_window7_224',
             'resnet18', 'dinov2_vits14_reg', 'dinov2_vitb14_reg'],
            tooltip="Arquitectura base del modelo (DINOv2 recomendado para fine-grained)"
        )
        # Conectar cambio de backbone a auto-configuración
        self.fields['model.backbone_name'].widget.currentTextChanged.connect(self.on_backbone_changed)
        model_layout.addRow("Backbone:", self.fields['model.backbone_name'].widget)
        
        self.fields['model.embedding_dim'] = IntField(
            "Embedding Dim",
            default=128, min_val=64, max_val=512,
            tooltip="Dimensión del vector de embeddings (recomendado: 128)"
        )
        model_layout.addRow("Embedding Dim:", self.fields['model.embedding_dim'].widget)
        
        self.fields['model.pretrained'] = BoolField(
            "Pretrained",
            tooltip="Usar pesos pre-entrenados de ImageNet"
        )
        model_layout.addRow("Pretrained:", self.fields['model.pretrained'].widget)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Projection Head group
        proj_group = QGroupBox("Projection Head")
        proj_layout = QFormLayout()
        
        self.fields['model.projection_head.hidden_dims'] = ListField(
            "Hidden Dims",
            tooltip="Dimensiones de capas ocultas (ej: 512, 256)"
        )
        proj_layout.addRow("Hidden Dims:", self.fields['model.projection_head.hidden_dims'].widget)
        
        self.fields['model.projection_head.use_batch_norm'] = BoolField(
            "Batch Norm",
            tooltip="Usar Batch Normalization"
        )
        proj_layout.addRow("Batch Norm:", self.fields['model.projection_head.use_batch_norm'].widget)
        
        self.fields['model.projection_head.dropout'] = FloatField(
            "Dropout",
            default=0.0, min_val=0.0, max_val=0.9, decimals=2, step=0.1,
            tooltip="Tasa de dropout (0.0 = sin dropout)"
        )
        proj_layout.addRow("Dropout:", self.fields['model.projection_head.dropout'].widget)
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🤖 Modelo")
    
    def create_data_tab(self):
        """Tab de configuración de datos"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Paths group
        paths_group = QGroupBox("Rutas de Datos")
        paths_layout = QFormLayout()
        
        self.fields['data.train_dir'] = PathField(
            "Train Dir",
            tooltip="Directorio de entrenamiento",
            is_dir=True
        )
        paths_layout.addRow("Train:", self.fields['data.train_dir'].widget_container)
        
        self.fields['data.val_dir'] = PathField(
            "Val Dir",
            tooltip="Directorio de validación",
            is_dir=True
        )
        paths_layout.addRow("Validation:", self.fields['data.val_dir'].widget_container)
        
        self.fields['data.test_dir'] = PathField(
            "Test Dir",
            tooltip="Directorio de test",
            is_dir=True
        )
        paths_layout.addRow("Test:", self.fields['data.test_dir'].widget_container)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Sampler group
        sampler_group = QGroupBox("Sampler (P×K Strategy)")
        sampler_layout = QFormLayout()
        
        self.fields['data.classes_per_batch'] = IntField(
            "Classes per Batch",
            default=3, min_val=2, max_val=20,
            tooltip="P: Número de clases por batch"
        )
        sampler_layout.addRow("Classes (P):", self.fields['data.classes_per_batch'].widget)
        
        self.fields['data.samples_per_class'] = IntField(
            "Samples per Class",
            default=8, min_val=2, max_val=32,
            tooltip="K: Muestras por clase"
        )
        sampler_layout.addRow("Samples (K):", self.fields['data.samples_per_class'].widget)
        
        self.fields['data.batch_size'] = IntField(
            "Batch Size",
            default=24, min_val=4, max_val=256,
            tooltip="Tamaño del batch (P × K)"
        )
        sampler_layout.addRow("Batch Size:", self.fields['data.batch_size'].widget)
        
        sampler_group.setLayout(sampler_layout)
        layout.addWidget(sampler_group)
        
        # DataLoader group
        loader_group = QGroupBox("DataLoader")
        loader_layout = QFormLayout()
        
        self.fields['data.num_workers'] = IntField(
            "Num Workers",
            default=0, min_val=0, max_val=16,
            tooltip="Número de workers (0 para Windows)"
        )
        loader_layout.addRow("Num Workers:", self.fields['data.num_workers'].widget)
        
        self.fields['data.pin_memory'] = BoolField(
            "Pin Memory",
            tooltip="Habilitar pin_memory para GPU"
        )
        loader_layout.addRow("Pin Memory:", self.fields['data.pin_memory'].widget)
        
        self.fields['data.drop_last'] = BoolField(
            "Drop Last",
            tooltip="Descartar último batch incompleto"
        )
        loader_layout.addRow("Drop Last:", self.fields['data.drop_last'].widget)
        
        loader_group.setLayout(loader_layout)
        layout.addWidget(loader_group)
        
        # Image preprocessing
        img_group = QGroupBox("Preprocesamiento de Imágenes")
        img_layout = QFormLayout()
        
        self.fields['data.image_size'] = IntField(
            "Image Size",
            default=224, min_val=64, max_val=512,
            tooltip="Tamaño de imagen final"
        )
        img_layout.addRow("Image Size:", self.fields['data.image_size'].widget)
        
        self.fields['data.resize_size'] = IntField(
            "Resize Size",
            default=256, min_val=64, max_val=512,
            tooltip="Tamaño de redimensionamiento inicial"
        )
        img_layout.addRow("Resize Size:", self.fields['data.resize_size'].widget)
        
        img_group.setLayout(img_layout)
        layout.addWidget(img_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "💾 Datos")
    
    def create_augmentation_tab(self):
        """Tab de augmentación"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Geometric
        geom_group = QGroupBox("Augmentaciones Geométricas")
        geom_layout = QFormLayout()
        
        self.fields['augmentation.random_horizontal_flip'] = FloatField(
            "H Flip",
            default=0.5, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad de flip horizontal"
        )
        geom_layout.addRow("Horizontal Flip:", self.fields['augmentation.random_horizontal_flip'].widget)
        
        self.fields['augmentation.random_vertical_flip'] = FloatField(
            "V Flip",
            default=0.5, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Probabilidad de flip vertical"
        )
        geom_layout.addRow("Vertical Flip:", self.fields['augmentation.random_vertical_flip'].widget)
        
        self.fields['augmentation.random_rotation'] = IntField(
            "Rotation",
            default=180, min_val=0, max_val=360,
            tooltip="Grados de rotación máxima"
        )
        geom_layout.addRow("Rotation (deg):", self.fields['augmentation.random_rotation'].widget)
        
        self.fields['augmentation.random_crop'] = BoolField(
            "Random Crop",
            tooltip="Aplicar random crop"
        )
        geom_layout.addRow("Random Crop:", self.fields['augmentation.random_crop'].widget)
        
        geom_group.setLayout(geom_layout)
        layout.addWidget(geom_group)
        
        # Color
        color_group = QGroupBox("Augmentaciones de Color")
        color_layout = QFormLayout()
        
        self.fields['augmentation.color_jitter.brightness'] = FloatField(
            "Brightness",
            default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Factor de brillo"
        )
        color_layout.addRow("Brightness:", self.fields['augmentation.color_jitter.brightness'].widget)
        
        self.fields['augmentation.color_jitter.contrast'] = FloatField(
            "Contrast",
            default=0.3, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Factor de contraste"
        )
        color_layout.addRow("Contrast:", self.fields['augmentation.color_jitter.contrast'].widget)
        
        self.fields['augmentation.color_jitter.saturation'] = FloatField(
            "Saturation",
            default=0.2, min_val=0.0, max_val=1.0, decimals=2, step=0.1,
            tooltip="Factor de saturación"
        )
        color_layout.addRow("Saturation:", self.fields['augmentation.color_jitter.saturation'].widget)
        
        self.fields['augmentation.color_jitter.hue'] = FloatField(
            "Hue",
            default=0.1, min_val=0.0, max_val=0.5, decimals=2, step=0.05,
            tooltip="Factor de matiz"
        )
        color_layout.addRow("Hue:", self.fields['augmentation.color_jitter.hue'].widget)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # Normalization
        norm_group = QGroupBox("Normalización (ImageNet)")
        norm_layout = QFormLayout()
        
        self.fields['augmentation.normalize.mean'] = ListField(
            "Mean",
            tooltip="Media para normalización (R, G, B)"
        )
        norm_layout.addRow("Mean:", self.fields['augmentation.normalize.mean'].widget)
        
        self.fields['augmentation.normalize.std'] = ListField(
            "Std",
            tooltip="Desviación estándar (R, G, B)"
        )
        norm_layout.addRow("Std:", self.fields['augmentation.normalize.std'].widget)
        
        norm_group.setLayout(norm_layout)
        layout.addWidget(norm_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🎨 Augmentación")
    
    def create_training_tab(self):
        """Tab de entrenamiento"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Epochs
        epochs_group = QGroupBox("Épocas")
        epochs_layout = QFormLayout()
        
        self.fields['training.epochs'] = IntField(
            "Epochs",
            default=30, min_val=1, max_val=1000,
            tooltip="Número total de épocas"
        )
        epochs_layout.addRow("Epochs:", self.fields['training.epochs'].widget)
        
        self.fields['training.warmup_epochs'] = IntField(
            "Warmup",
            default=3, min_val=0, max_val=20,
            tooltip="Épocas de warmup"
        )
        epochs_layout.addRow("Warmup:", self.fields['training.warmup_epochs'].widget)
        
        epochs_group.setLayout(epochs_layout)
        layout.addWidget(epochs_group)
        
        # Loss
        loss_group = QGroupBox("Función de Pérdida")
        loss_layout = QFormLayout()
        
        self.fields['training.loss.type'] = ChoiceField(
            "Loss Type",
            ['multi_similarity', 'triplet', 'arcface'],
            tooltip="Tipo de función de pérdida"
        )
        loss_layout.addRow("Type:", self.fields['training.loss.type'].widget)
        
        self.fields['training.loss.params.alpha'] = FloatField(
            "Alpha",
            default=2.0, min_val=0.1, max_val=10.0, decimals=2, step=0.1,
            tooltip="Peso de pares positivos (Multi-Similarity)"
        )
        loss_layout.addRow("Alpha:", self.fields['training.loss.params.alpha'].widget)
        
        self.fields['training.loss.params.beta'] = FloatField(
            "Beta",
            default=50.0, min_val=1.0, max_val=100.0, decimals=1, step=1.0,
            tooltip="Peso de pares negativos (Multi-Similarity)"
        )
        loss_layout.addRow("Beta:", self.fields['training.loss.params.beta'].widget)
        
        loss_group.setLayout(loss_layout)
        layout.addWidget(loss_group)
        
        # Optimizer
        opt_group = QGroupBox("Optimizador")
        opt_layout = QFormLayout()
        
        self.fields['training.optimizer'] = ChoiceField(
            "Optimizer",
            ['adam', 'adamw', 'sgd'],
            tooltip="Tipo de optimizador"
        )
        opt_layout.addRow("Type:", self.fields['training.optimizer'].widget)
        
        self.fields['training.learning_rate'] = FloatField(
            "Learning Rate",
            default=0.0001, min_val=0.00001, max_val=0.1, decimals=6, step=0.00001,
            tooltip="Tasa de aprendizaje"
        )
        opt_layout.addRow("Learning Rate:", self.fields['training.learning_rate'].widget)
        
        self.fields['training.weight_decay'] = FloatField(
            "Weight Decay",
            default=0.0001, min_val=0.0, max_val=0.01, decimals=6, step=0.00001,
            tooltip="Regularización L2"
        )
        opt_layout.addRow("Weight Decay:", self.fields['training.weight_decay'].widget)
        
        self.fields['training.backbone_lr_factor'] = FloatField(
            "Backbone LR Factor",
            default=0.1, min_val=0.001, max_val=1.0, decimals=3, step=0.01,
            tooltip="Factor multiplicativo para LR del backbone (0.1 = 10x menor que projection head)"
        )
        opt_layout.addRow("Backbone LR Factor:", self.fields['training.backbone_lr_factor'].widget)

        self.fields['training.gradient_clip_norm'] = FloatField(
            "Gradient Clip Norm",
            default=5.0, min_val=0.0, max_val=50.0, decimals=1, step=0.5,
            tooltip="Norma máxima para gradient clipping (0 = desactivado)"
        )
        opt_layout.addRow("Gradient Clip:", self.fields['training.gradient_clip_norm'].widget)

        opt_group.setLayout(opt_layout)
        layout.addWidget(opt_group)
        
        # Scheduler
        sched_group = QGroupBox("Scheduler")
        sched_layout = QFormLayout()
        
        self.fields['training.scheduler'] = ChoiceField(
            "Scheduler",
            ['reduce_on_plateau', 'cosine', 'onecycle'],
            tooltip="Tipo de scheduler"
        )
        sched_layout.addRow("Type:", self.fields['training.scheduler'].widget)

        self.fields['training.scheduler_params.T_max'] = IntField(
            "T_max",
            default=50, min_val=1, max_val=1000,
            tooltip="Período del cosine scheduler (debe coincidir con epochs para evitar restart)"
        )
        sched_layout.addRow("T_max:", self.fields['training.scheduler_params.T_max'].widget)

        self.fields['training.scheduler_params.eta_min'] = FloatField(
            "Eta Min",
            default=0.000001, min_val=0.0, max_val=0.01, decimals=7, step=0.0000001,
            tooltip="LR mínimo del cosine scheduler"
        )
        sched_layout.addRow("Eta Min:", self.fields['training.scheduler_params.eta_min'].widget)

        sched_group.setLayout(sched_layout)
        layout.addWidget(sched_group)

        # Early Stopping
        es_group = QGroupBox("Early Stopping")
        es_layout = QFormLayout()

        self.fields['training.early_stopping.enabled'] = BoolField(
            "Enabled",
            default=True,
            tooltip="Activar early stopping"
        )
        es_layout.addRow("Activado:", self.fields['training.early_stopping.enabled'].widget)

        self.fields['training.early_stopping.patience'] = IntField(
            "Patience",
            default=10, min_val=1, max_val=100,
            tooltip="Épocas sin mejora antes de detener el entrenamiento"
        )
        es_layout.addRow("Patience:", self.fields['training.early_stopping.patience'].widget)

        self.fields['training.early_stopping.min_delta'] = FloatField(
            "Min Delta",
            default=0.001, min_val=0.0, max_val=0.1, decimals=4, step=0.0001,
            tooltip="Mejora mínima para considerar progreso"
        )
        es_layout.addRow("Min Delta:", self.fields['training.early_stopping.min_delta'].widget)

        self.fields['training.early_stopping.restore_best_weights'] = BoolField(
            "Restore Best",
            default=True,
            tooltip="Restaurar pesos del mejor epoch al detenerse"
        )
        es_layout.addRow("Restore Best:", self.fields['training.early_stopping.restore_best_weights'].widget)

        es_group.setLayout(es_layout)
        layout.addWidget(es_group)

        # Checkpoints
        ckpt_group = QGroupBox("Checkpoints")
        ckpt_layout = QFormLayout()

        self.fields['training.save_every_n_epochs'] = IntField(
            "Save Every N",
            default=10, min_val=1, max_val=100,
            tooltip="Guardar checkpoint cada N épocas"
        )
        ckpt_layout.addRow("Guardar cada:", self.fields['training.save_every_n_epochs'].widget)

        self.fields['training.save_best_only'] = BoolField(
            "Save Best Only",
            default=True,
            tooltip="Solo guardar el mejor modelo (además de checkpoints periódicos)"
        )
        ckpt_layout.addRow("Solo mejor:", self.fields['training.save_best_only'].widget)

        ckpt_group.setLayout(ckpt_layout)
        layout.addWidget(ckpt_group)

        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🚀 Entrenamiento")
    
    def create_inference_tab(self):
        """Tab de inferencia"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # FAISS
        faiss_group = QGroupBox("Configuración FAISS")
        faiss_layout = QFormLayout()
        
        self.fields['inference.index_type'] = ChoiceField(
            "Index Type",
            ['flat', 'ivf', 'ivfpq'],
            tooltip="Tipo de índice FAISS"
        )
        faiss_layout.addRow("Index Type:", self.fields['inference.index_type'].widget)
        
        self.fields['inference.metric_type'] = ChoiceField(
            "Metric",
            ['inner_product', 'l2'],
            tooltip="Métrica de distancia"
        )
        faiss_layout.addRow("Metric:", self.fields['inference.metric_type'].widget)
        
        self.fields['inference.k_neighbors'] = IntField(
            "K Neighbors",
            default=5, min_val=1, max_val=100,
            tooltip="Número de vecinos a buscar"
        )
        faiss_layout.addRow("K Neighbors:", self.fields['inference.k_neighbors'].widget)
        
        self.fields['inference.use_gpu'] = BoolField(
            "Use GPU",
            tooltip="Usar GPU para FAISS"
        )
        faiss_layout.addRow("Use GPU:", self.fields['inference.use_gpu'].widget)
        
        faiss_group.setLayout(faiss_layout)
        layout.addWidget(faiss_group)
        
        # Thresholds
        thresh_group = QGroupBox("Umbrales")
        thresh_layout = QFormLayout()
        
        self.fields['inference.similarity_threshold'] = FloatField(
            "Similarity",
            default=0.6, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Umbral de similitud"
        )
        thresh_layout.addRow("Similarity:", self.fields['inference.similarity_threshold'].widget)
        
        self.fields['inference.confidence_threshold'] = FloatField(
            "Confidence",
            default=0.8, min_val=0.0, max_val=1.0, decimals=2, step=0.05,
            tooltip="Umbral de confianza"
        )
        thresh_layout.addRow("Confidence:", self.fields['inference.confidence_threshold'].widget)
        
        thresh_group.setLayout(thresh_layout)
        layout.addWidget(thresh_group)
        
        # Device
        device_group = QGroupBox("Dispositivo")
        device_layout = QFormLayout()
        
        self.fields['inference.device'] = ChoiceField(
            "Device",
            ['cuda', 'cpu'],
            tooltip="Dispositivo para inferencia"
        )
        device_layout.addRow("Device:", self.fields['inference.device'].widget)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "🔍 Inferencia")
    
    def create_paths_tab(self):
        """Tab de rutas"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        paths_group = QGroupBox("Rutas del Sistema")
        paths_layout = QFormLayout()
        
        self.fields['paths.model_save_dir'] = PathField(
            "Models",
            tooltip="Directorio para modelos",
            is_dir=True
        )
        paths_layout.addRow("Models:", self.fields['paths.model_save_dir'].widget_container)
        
        self.fields['paths.checkpoint_dir'] = PathField(
            "Checkpoints",
            tooltip="Directorio para checkpoints",
            is_dir=True
        )
        paths_layout.addRow("Checkpoints:", self.fields['paths.checkpoint_dir'].widget_container)
        
        self.fields['paths.log_dir'] = PathField(
            "Logs",
            tooltip="Directorio para logs",
            is_dir=True
        )
        paths_layout.addRow("Logs:", self.fields['paths.log_dir'].widget_container)
        
        self.fields['paths.experiment_dir'] = PathField(
            "Experiments",
            tooltip="Directorio para experimentos",
            is_dir=True
        )
        paths_layout.addRow("Experiments:", self.fields['paths.experiment_dir'].widget_container)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "📁 Rutas")
    
    def create_hardware_tab(self):
        """Tab de hardware"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # GPU
        gpu_group = QGroupBox("GPU")
        gpu_layout = QFormLayout()
        
        self.fields['hardware.use_gpu'] = BoolField(
            "Use GPU",
            tooltip="Habilitar GPU"
        )
        gpu_layout.addRow("Use GPU:", self.fields['hardware.use_gpu'].widget)
        
        self.fields['hardware.gpu_ids'] = ListField(
            "GPU IDs",
            tooltip="IDs de GPUs a usar (ej: 0, 1)"
        )
        gpu_layout.addRow("GPU IDs:", self.fields['hardware.gpu_ids'].widget)
        
        self.fields['hardware.use_amp'] = BoolField(
            "Mixed Precision",
            tooltip="Usar Automatic Mixed Precision"
        )
        gpu_layout.addRow("Mixed Precision:", self.fields['hardware.use_amp'].widget)
        
        gpu_group.setLayout(gpu_layout)
        layout.addWidget(gpu_group)
        
        # Distributed
        dist_group = QGroupBox("Entrenamiento Distribuido")
        dist_layout = QFormLayout()
        
        self.fields['hardware.distributed'] = BoolField(
            "Distributed",
            tooltip="Habilitar entrenamiento distribuido"
        )
        dist_layout.addRow("Distributed:", self.fields['hardware.distributed'].widget)
        
        self.fields['hardware.world_size'] = IntField(
            "World Size",
            default=1, min_val=1, max_val=16,
            tooltip="Número de procesos"
        )
        dist_layout.addRow("World Size:", self.fields['hardware.world_size'].widget)
        
        dist_group.setLayout(dist_layout)
        layout.addWidget(dist_group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "⚡ Hardware")
    
    def load_config(self):
        """Carga configuración desde archivo"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            
            # Cargar valores en campos
            self.load_values_from_dict(self.config_data)
            
            self.status_label.setText(f"✅ Configuración cargada: {datetime.now().strftime('%H:%M:%S')}")
            self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
            
            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                self.parent_window.log_widget.append_log("Configuración cargada", "SUCCESS")
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error cargando configuración:\n{str(e)}")
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet("color: #f48771; padding: 5px;")
    
    def load_values_from_dict(self, data, prefix=""):
        """Carga valores recursivamente desde diccionario"""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self.load_values_from_dict(value, full_key)
            elif full_key in self.fields:
                self.fields[full_key].set_value(value)
    
    def save_config(self):
        """Guarda configuración en archivo"""
        # Validar primero
        if not self.validate_all():
            QMessageBox.warning(
                self,
                "Validación Fallida",
                "Hay campos con valores inválidos. Por favor corríjalos antes de guardar."
            )
            return
        
        # Confirmar
        reply = QMessageBox.question(
            self,
            "Confirmar Guardado",
            f"¿Guardar cambios en {self.config_path.name}?\n\nSe creará un backup automático.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply != QMessageBox.Yes:
            return
        
        try:
            # Crear backup
            backup_path = self.config_path.with_suffix('.yaml.backup')
            shutil.copy2(self.config_path, backup_path)
            
            # Actualizar diccionario con valores de campos
            self.update_dict_from_fields(self.config_data)
            
            # Guardar
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            
            self.status_label.setText(f"✅ Guardado exitosamente: {datetime.now().strftime('%H:%M:%S')}")
            self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
            
            if self.parent_window:
                self.parent_window.log_widget.append_log(
                    f"Configuración guardada en {self.config_path.name}",
                    "SUCCESS"
                )
            
            QMessageBox.information(
                self,
                "Éxito",
                f"Configuración guardada exitosamente.\n\nBackup creado: {backup_path.name}"
            )
            
            self.configChanged.emit()
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardando configuración:\n{str(e)}")
            self.status_label.setText(f"❌ Error: {str(e)}")
            self.status_label.setStyleSheet("color: #f48771; padding: 5px;")
    
    def update_dict_from_fields(self, data, prefix=""):
        """Actualiza diccionario recursivamente desde campos"""
        for key in list(data.keys()):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(data[key], dict):
                self.update_dict_from_fields(data[key], full_key)
            elif full_key in self.fields:
                data[key] = self.fields[full_key].get_value()
    
    def validate_all(self):
        """Valida todos los campos"""
        all_valid = True
        
        for key, field in self.fields.items():
            if not field.validate():
                all_valid = False
                field.mark_invalid()
            else:
                field.mark_valid()
        
        if all_valid:
            self.status_label.setText("✅ Todos los campos son válidos")
            self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
        else:
            self.status_label.setText("❌ Hay campos inválidos")
            self.status_label.setStyleSheet("color: #f48771; padding: 5px;")
        
        return all_valid
    
    def reset_to_defaults(self):
        """Restaura valores por defecto"""
        reply = QMessageBox.question(
            self,
            "Confirmar Reset",
            "¿Restaurar todos los valores a sus valores por defecto?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.load_config()
    
    def on_backbone_changed(self, backbone_name):
        """
        Auto-configura parámetros al cambiar el backbone.
        Actualiza resolución, batch_size si es necesario, y otros parámetros.
        """
        if not backbone_name:
            return
        
        changes = []  # Lista de (parámetro, valor_anterior, valor_nuevo)
        
        try:
            # 1. Obtener resolución óptima para el backbone
            resolved_size = get_input_size_for_backbone(backbone_name)
            
            # 2. Actualizar data.image_size
            if 'data.image_size' in self.fields:
                old_size = self.fields['data.image_size'].get_value()
                if old_size != resolved_size:
                    self.fields['data.image_size'].set_value(resolved_size)
                    changes.append(('data.image_size', old_size, resolved_size))
            
            # 3. Actualizar augmentation.image_size si existe
            if 'augmentation.image_size' in self.fields:
                old_aug_size = self.fields['augmentation.image_size'].get_value()
                if old_aug_size != resolved_size:
                    self.fields['augmentation.image_size'].set_value(resolved_size)
                    changes.append(('augmentation.image_size', old_aug_size, resolved_size))
            
            # 4. Ajustar samples_per_class según VRAM del modelo
            # Modelos grandes necesitan menos samples por batch
            samples_per_class_map = {
                'efficientnet_b4': 5,          # Modelo grande
                'dinov2_vitb14': 7,            # Modelo mediano-grande
                'dinov2_vitb14_reg': 7,        # Modelo mediano-grande
                'convnext_v2_tiny': 7,         # Modelo mediano
                'swin_tiny_patch4_window7_224': 7,  # Modelo mediano
                'dinov2_vits14': 14,           # Modelo pequeño (default)
                'dinov2_vits14_reg': 14,       # Modelo pequeño
                'resnet50': 14,                # CNN clásico, eficiente
                'resnet18': 14,                # CNN pequeño
                'deit_small_patch16_224': 14,  # ViT pequeño
            }
            
            if 'data.samples_per_class' in self.fields:
                old_samples = self.fields['data.samples_per_class'].get_value()
                recommended_samples = samples_per_class_map.get(backbone_name, 14)
                if old_samples != recommended_samples:
                    self.fields['data.samples_per_class'].set_value(recommended_samples)
                    changes.append(('data.samples_per_class', old_samples, recommended_samples))
            
            # 5. Recalcular batch_size = classes_per_batch × samples_per_class
            if 'data.classes_per_batch' in self.fields and 'data.samples_per_class' in self.fields:
                classes_per_batch = self.fields['data.classes_per_batch'].get_value()
                samples_per_class = self.fields['data.samples_per_class'].get_value()
                calculated_batch = classes_per_batch * samples_per_class
                
                if 'data.batch_size' in self.fields:
                    old_batch = self.fields['data.batch_size'].get_value()
                    if old_batch != calculated_batch:
                        self.fields['data.batch_size'].set_value(calculated_batch)
                        changes.append(('data.batch_size', old_batch, f'{calculated_batch} ({classes_per_batch}×{samples_per_class})'))
            
            # 6. Sincronizar grain_detection.crop_size con image_size
            if 'grain_detection.crop_size' in self.fields:
                old_crop_size = self.fields['grain_detection.crop_size'].get_value()
                if old_crop_size != resolved_size:
                    self.fields['grain_detection.crop_size'].set_value(resolved_size)
                    changes.append(('grain_detection.crop_size', old_crop_size, resolved_size))
            
            # 7. Actualizar experiment.name si existe
            if 'experiment.name' in self.fields:
                old_exp_name = self.fields['experiment.name'].get_value()
                # Generar nombre sugerido: backbone + baseline
                suggested_name = f"{backbone_name}_baseline"
                if old_exp_name != suggested_name:
                    self.fields['experiment.name'].set_value(suggested_name)
                    changes.append(('experiment.name', old_exp_name, suggested_name))
            
            # 8. Actualizar experiment.description si existe
            if 'experiment.description' in self.fields:
                backbone_descriptions = {
                    'dinov2_vits14': 'DINOv2 ViT-S/14 baseline with ArcFace',
                    'dinov2_vitb14': 'DINOv2 ViT-B/14 baseline with ArcFace',
                    'resnet50': 'ResNet-50 baseline (CNN clásico)',
                    'convnext_v2_tiny': 'ConvNeXt V2 Tiny baseline (CNN moderno)',
                    'deit_small_patch16_224': 'DeiT-S baseline (ViT supervisado)',
                    'efficientnet_b4': 'EfficientNet-B4 baseline (CNN eficiente)',
                    'swin_tiny_patch4_window7_224': 'Swin Transformer Tiny baseline',
                }
                if backbone_name in backbone_descriptions:
                    old_desc = self.fields['experiment.description'].get_value()
                    new_desc = backbone_descriptions[backbone_name]
                    if old_desc != new_desc:
                        self.fields['experiment.description'].set_value(new_desc)
                        changes.append(('experiment.description', old_desc, new_desc))
            
            # Mostrar diálogo con cambios si hubo alguno
            if changes:
                dialog = ConfigChangesDialog(changes, self)
                dialog.show()
                
                # Log en la aplicación principal
                if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                    self.parent_window.log_widget.append_log(
                        f"Auto-configuración aplicada para '{backbone_name}': {len(changes)} parámetros actualizados",
                        "SUCCESS"
                    )
                
                self.status_label.setText(f"✅ Auto-configurado para {backbone_name} ({len(changes)} cambios)")
                self.status_label.setStyleSheet("color: #4ec9b0; padding: 5px;")
            
        except Exception as e:
            error_msg = f"Error en auto-configuración: {str(e)}"
            QMessageBox.warning(self, "Error", error_msg)
            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                self.parent_window.log_widget.append_log(error_msg, "ERROR")
