# Tests del Sistema de Anotaciones Centralizadas

**Archivo:** `test_centralized_annotations.py`  
**Cobertura:** Sistema completo de anotaciones externas persistentes

---

## Estructura de Tests

### 1. `TestMigrationToCentralized`
Tests para el script de migración de anotaciones.

- ✅ `test_migration_creates_annotation_root` - Verifica creación de `data/annotations/`
- ✅ `test_migration_handles_duplicates` - Manejo de .seg duplicados
- ✅ `test_migration_dry_run_no_changes` - Dry-run no modifica archivos

### 2. `TestValidationCentralized`
Tests para el script de validación de integridad.

- ✅ `test_validation_all_images_have_annotations` - Validación exitosa
- ✅ `test_validation_detects_missing_annotations` - Detecta .seg faltantes
- ✅ `test_validation_detects_orphan_annotations` - Detecta .seg huérfanos

### 3. `TestReorganizeSplitsSafe`
Tests para reorganización segura de splits.

- ✅ `test_reorganize_moves_only_images` - Solo mueve imágenes, no .seg
- ✅ `test_reorganize_validates_annotations_exist` - Valida existencia de anotaciones
- ✅ `test_reorganize_dry_run_no_changes` - Dry-run no mueve archivos

### 4. `TestSegmentedGrainDatasetWithAnnotationRoot`
Tests para `SegmentedGrainDataset` con anotaciones centralizadas.

- ✅ `test_dataset_loads_from_centralized_annotations` - Carga desde `data/annotations/`
- ✅ `test_dataset_fails_if_annotation_root_missing` - Falla si directorio no existe
- ✅ `test_dataset_legacy_mode_without_annotation_root` - Modo legacy funciona

### 5. `TestSegmentationAnnotatorWithAnnotationRoot`
Tests para `SegmentationAnnotator` con anotaciones centralizadas.

- ✅ `test_annotator_generates_in_centralized_dir` - Genera en directorio centralizado
- ✅ `test_annotator_legacy_mode_without_annotation_root` - Modo legacy funciona

---

## Ejecutar Tests

### Todos los tests
```bash
python tests/run_centralized_tests.py
```

### Verbose (ver output completo)
```bash
python tests/run_centralized_tests.py --verbose
```

### Solo una clase de tests
```bash
python tests/run_centralized_tests.py --single TestMigrationToCentralized
```

### Con pytest directamente
```bash
.venv\Scripts\python.exe -m pytest tests/test_centralized_annotations.py -v
```

---

## Fixtures

### `temp_dataset`
Crea un dataset temporal con estructura completa para testing:

```
temp_dir/
  processed/
    train/
      ClassA/
        img_001.png + img_001.seg (2 granos)
      ClassB/
        img_002.png + img_002.seg (3 granos)
    val/
      ClassA/
        img_003.png + img_003.seg (1 grano)
      ClassB/
        img_004.png + img_004.seg (2 granos)
    test/
      ClassA/
        img_005.png + img_005.seg (1 grano)
```

**Cleanup:** Automático después de cada test.

---

## Cobertura de Código

### Archivos Testeados

1. **`scripts/migrate_annotations_to_centralized.py`**
   - Migración completa
   - Manejo de duplicados y conflictos
   - Dry-run mode

2. **`scripts/validate_annotations_centralized.py`**
   - Validación de integridad
   - Detección de problemas

3. **`scripts/reorganize_splits_safe.py`**
   - Reorganización sin pérdida
   - Validación pre-reorganización

4. **`src/data/grain_dataset.py`**
   - Carga con `annotation_root`
   - Modo legacy (backward-compatible)

5. **`src/grain_detection/annotator.py`**
   - Generación en directorio centralizado
   - Modo legacy

### Cobertura Estimada
- **Scripts nuevos:** 90%+ (funciones críticas 100%)
- **Código modificado:** 85%+ (paths nuevos + legacy)

---

## Agregar Nuevos Tests

### Template para nuevo test

```python
def test_nueva_funcionalidad(temp_dataset):
    """Descripción del test"""
    import os
    original_cwd = os.getcwd()
    os.chdir(temp_dataset)
    
    try:
        # Setup
        # ...
        
        # Acción
        # ...
        
        # Verificación
        assert condition
        
    finally:
        os.chdir(original_cwd)
```

### Ejecutar solo el nuevo test

```bash
python tests/run_centralized_tests.py --single test_nueva_funcionalidad
```

---

## CI/CD Integration

### GitHub Actions (ejemplo)

```yaml
- name: Run centralized annotations tests
  run: |
    source .venv/bin/activate
    python tests/run_centralized_tests.py --verbose
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

python tests/run_centralized_tests.py
if [ $? -ne 0 ]; then
    echo "❌ Tests failed, aborting commit"
    exit 1
fi
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pytest'"

```bash
.venv\Scripts\python.exe -m pip install pytest
```

### Error: "No module named 'cv2'"

```bash
.venv\Scripts\python.exe -m pip install opencv-python
```

### Tests pasan localmente pero fallan en CI

- Verificar paths absolutos vs relativos
- Verificar permisos de escritura en temp dirs
- Verificar que pytest use el venv correcto

---

## Mantenimiento

### Actualizar después de cambios

Si modificas el sistema de anotaciones:

1. **Actualizar tests existentes** si cambia la API
2. **Añadir tests nuevos** para funcionalidad nueva
3. **Ejecutar suite completa** antes de commit
4. **Actualizar este README** si cambia la estructura

### Frecuencia de Ejecución

- **Antes de cada commit:** Tests relacionados con cambios
- **Antes de cada PR:** Suite completa
- **CI/CD:** Suite completa en cada push
- **Nightly:** Suite completa + tests de integración

---

**Última actualización:** 2026-02-18  
**Mantenedor:** Equipo MetricLearning  
**Estado:** ✅ Todos los tests pasando
