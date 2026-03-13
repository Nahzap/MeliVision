"""
Sistema de logging centralizado para el proyecto MetricLearning.
Configura logging a consola y archivo con rotación diaria.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "MetricLearning",
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5
) -> logging.Logger:
    """
    Configura sistema de logging con salida a consola y archivo.
    
    Args:
        name: Nombre del logger
        log_dir: Directorio para logs
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Si True, muestra logs en consola
        file_output: Si True, guarda logs en archivo
        max_bytes: Tamaño máximo del archivo de log (10MB por defecto)
        backup_count: Número de archivos de respaldo a mantener
        
    Returns:
        Logger configurado
        
    Example:
        >>> logger = setup_logger("training", log_level=logging.DEBUG)
        >>> logger.info("Training started")
        >>> logger.error("Error occurred", exc_info=True)
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        logger.handlers.clear()
    
    logger.setLevel(log_level)
    logger.propagate = False
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        # Ensure utf-8 encoding on Windows (default cp1252 breaks on unicode chars)
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except Exception:
                pass
        logger.addHandler(console_handler)
    
    if file_output:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = log_path / f"{name}_{today}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file: {log_file.absolute()}")
    
    return logger


def get_logger(name: str = "MetricLearning") -> logging.Logger:
    """
    Obtiene logger existente o crea uno nuevo.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


def log_system_info(logger: logging.Logger):
    """
    Registra información del sistema.
    
    Args:
        logger: Logger a usar
    """
    import torch
    import platform
    
    logger.info("=" * 70)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 70)
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info("=" * 70)


def log_config(logger: logging.Logger, config: dict):
    """
    Registra configuración del modelo.
    
    Args:
        logger: Logger a usar
        config: Diccionario de configuración
    """
    logger.info("=" * 70)
    logger.info("CONFIGURATION")
    logger.info("=" * 70)
    
    def log_dict(d: dict, indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")
    
    log_dict(config)
    logger.info("=" * 70)


def log_model_summary(logger: logging.Logger, model, input_size: tuple = (1, 3, 224, 224)):
    """
    Registra resumen del modelo.
    
    Args:
        logger: Logger a usar
        model: Modelo PyTorch
        input_size: Tamaño de entrada para el modelo
    """
    import torch
    
    logger.info("=" * 70)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    try:
        dummy_input = torch.randn(input_size)
        if next(model.parameters()).is_cuda:
            dummy_input = dummy_input.cuda()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"Input shape: {input_size}")
        logger.info(f"Output shape: {output.shape}")
    except Exception as e:
        logger.warning(f"Could not compute model output shape: {e}")
    
    logger.info("=" * 70)


def log_dataset_info(logger: logging.Logger, dataset, name: str = "Dataset"):
    """
    Registra información del dataset.
    
    Args:
        logger: Logger a usar
        dataset: Dataset PyTorch
        name: Nombre del dataset
    """
    logger.info("=" * 70)
    logger.info(f"{name.upper()} INFORMATION")
    logger.info("=" * 70)
    logger.info(f"Total samples: {len(dataset)}")
    
    if hasattr(dataset, 'classes'):
        logger.info(f"Number of classes: {len(dataset.classes)}")
        logger.info(f"Classes: {dataset.classes}")
    
    if hasattr(dataset, 'class_to_idx'):
        from collections import Counter
        labels = [dataset.targets[i] if hasattr(dataset, 'targets') else dataset[i][1] for i in range(len(dataset))]
        class_counts = Counter(labels)
        
        logger.info("Class distribution:")
        for class_idx, count in sorted(class_counts.items()):
            class_name = dataset.classes[class_idx] if hasattr(dataset, 'classes') else f"Class {class_idx}"
            logger.info(f"  {class_name}: {count} samples ({count/len(dataset)*100:.2f}%)")
    
    logger.info("=" * 70)


def log_training_epoch(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    metrics: Optional[dict] = None
):
    """
    Registra información de una época de entrenamiento.
    
    Args:
        logger: Logger a usar
        epoch: Número de época actual
        total_epochs: Total de épocas
        train_loss: Loss de entrenamiento
        val_loss: Loss de validación (opcional)
        metrics: Métricas adicionales (opcional)
    """
    logger.info("=" * 70)
    logger.info(f"EPOCH {epoch}/{total_epochs}")
    logger.info("=" * 70)
    logger.info(f"Train Loss: {train_loss:.6f}")
    
    if val_loss is not None:
        logger.info(f"Val Loss: {val_loss:.6f}")
    
    if metrics:
        logger.info("Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.6f}")
            else:
                logger.info(f"  {key}: {value}")
    
    logger.info("=" * 70)


def cleanup_old_logs(log_dir: str = "logs", days_to_keep: int = 7):
    """
    Limpia logs antiguos.
    
    Args:
        log_dir: Directorio de logs
        days_to_keep: Días de logs a mantener
    """
    from datetime import timedelta
    
    log_path = Path(log_dir)
    if not log_path.exists():
        return
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    deleted_count = 0
    for log_file in log_path.glob("*.log*"):
        try:
            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_time < cutoff_date:
                log_file.unlink()
                deleted_count += 1
        except Exception:
            pass
    
    if deleted_count > 0:
        logger = get_logger()
        logger.info(f"Cleaned up {deleted_count} old log files")
