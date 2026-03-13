"""
Test end-to-end: SegmentationAnnotator → .seg files → SegmentedGrainDataset
Verifies the full pipeline from detection to dataset loading.

NOTA: Solo procesa 3 imágenes por clase para mantener el test rápido.
El dataset val tiene 837 imágenes — procesarlas todas tomaría minutos.
"""
import sys
import os
import time
import shutil
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from src.grain_detection.grain_detector import PollenGrainDetector
from src.grain_detection.seg_format import SegFileWriter, SegFileReader
from src.data.grain_dataset import SegmentedGrainDataset


MAX_IMAGES_PER_CLASS = 3


def discover_test_images(root, max_per_class=MAX_IMAGES_PER_CLASS):
    """Descubre hasta max_per_class imágenes por cada clase."""
    images = []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        pngs = sorted(class_dir.glob("*.png"))[:max_per_class]
        images.extend(pngs)
    return images


def test_annotate_and_load():
    """
    End-to-end con LÍMITE de imágenes:
    1. Copiar N imágenes a tmpdir
    2. Anotar con PollenGrainDetector → .seg
    3. Cargar con SegmentedGrainDataset
    4. Verificar DataLoader
    """
    print("=" * 60)
    print("END-TO-END: Detector → .seg → SegmentedGrainDataset")
    print(f"(Limitado a {MAX_IMAGES_PER_CLASS} imágenes/clase)")
    print("=" * 60)

    # Buscar split válido
    source_root = None
    for split in ["val", "train"]:
        root = Path(f"data/processed/{split}")
        if root.exists() and any(d.is_dir() for d in root.iterdir()):
            source_root = root
            break

    if source_root is None:
        print("SKIP: No valid dataset found")
        return False

    images = discover_test_images(source_root)
    print(f"  Source: {source_root} ({len(images)} imágenes seleccionadas)")

    # Crear directorio temporal con estructura ImageFolder
    tmpdir = Path(tempfile.mkdtemp(prefix="ml_e2e_test_"))
    print(f"  Tmpdir: {tmpdir}")

    try:
        # Copiar imágenes a tmpdir
        for img in images:
            class_name = img.parent.name
            dest_dir = tmpdir / class_name
            dest_dir.mkdir(exist_ok=True)
            shutil.copy2(str(img), str(dest_dir / img.name))

        total_copied = sum(1 for _ in tmpdir.rglob("*.png"))
        print(f"  Copiadas: {total_copied} imágenes\n")

        # ============================================================
        # PASO 1: Anotar con PollenGrainDetector
        # ============================================================
        print("[1] Anotando con PollenGrainDetector")
        print("-" * 40)

        import cv2
        detector = PollenGrainDetector()
        total_grains = 0
        grains_per_class = {}

        for img_path in sorted(tmpdir.rglob("*.png")):
            t0 = time.perf_counter()
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  ERROR: No se pudo leer {img_path.name}")
                continue

            h, w = image.shape[:2]
            saliency, grains = detector.detect(image)
            elapsed = (time.perf_counter() - t0) * 1000

            # Escribir .seg
            detections = []
            for g in grains:
                detections.append({
                    "index": g.index,
                    "class_index": 0,
                    "bbox": g.bbox,
                    "saliency": g.saliency_prob,
                    "contour": g.contour,
                })

            seg_path = img_path.with_suffix(".seg")
            SegFileWriter.write(
                str(seg_path), img_path.name, (h, w),
                detections, detector.get_parameters()
            )

            class_name = img_path.parent.name
            grains_per_class[class_name] = grains_per_class.get(class_name, 0) + len(grains)
            total_grains += len(grains)

            print(f"  {img_path.name}: {len(grains)} granos ({elapsed:.0f}ms)")

        print(f"\n  Total granos: {total_grains}")
        print(f"  Por clase: {grains_per_class}")

        # ============================================================
        # PASO 2: Verificar .seg round-trip
        # ============================================================
        print(f"\n[2] Verificando .seg files")
        print("-" * 40)

        seg_files = list(tmpdir.rglob("*.seg"))
        print(f"  .seg creados: {len(seg_files)}")

        for seg in seg_files[:3]:
            result = SegFileReader.read(str(seg))
            n = len(result["grains"])
            meta = result.get("metadata", {})
            print(f"  {seg.name}: {n} granos, meta={list(meta.keys())}")

        assert len(seg_files) == total_copied, f"Seg count mismatch: {len(seg_files)} vs {total_copied}"
        print("  PASS: Todos los .seg creados correctamente")

        # ============================================================
        # PASO 3: Cargar con SegmentedGrainDataset
        # ============================================================
        print(f"\n[3] Cargando SegmentedGrainDataset")
        print("-" * 40)

        dataset = SegmentedGrainDataset(
            root=str(tmpdir),
            crop_padding=0.20,
            crop_size=256,
            return_metadata=True,
        )

        ds_stats = dataset.get_stats()
        print(f"  Samples (granos): {ds_stats['total_grains']}")
        print(f"  Clases: {ds_stats['num_classes']}")
        print(f"  Por clase: {ds_stats['grains_per_class']}")

        assert len(dataset) > 0, "Dataset vacío"
        assert len(dataset) == total_grains, f"Dataset size mismatch: {len(dataset)} vs {total_grains}"
        print("  PASS: Dataset cargado correctamente")

        # ============================================================
        # PASO 4: Cargar samples individuales
        # ============================================================
        print(f"\n[4] Cargando samples")
        print("-" * 40)

        for i in range(min(5, len(dataset))):
            crop_tensor, class_idx, meta = dataset[i]
            print(f"  #{i}: shape={tuple(crop_tensor.shape)}, "
                  f"class={meta['class_name']}, "
                  f"bbox={meta['bbox']}, sal={meta['saliency']:.3f}")

            assert crop_tensor.shape == (3, 256, 256)
            assert 0 <= class_idx < len(dataset.classes)

        print("  PASS")

        # ============================================================
        # PASO 5: DataLoader compatibility
        # ============================================================
        print(f"\n[5] DataLoader batch test")
        print("-" * 40)

        dataset_simple = SegmentedGrainDataset(
            root=str(tmpdir), crop_padding=0.20, crop_size=256, return_metadata=False,
        )
        from torch.utils.data import DataLoader
        batch_size = min(4, len(dataset_simple))
        loader = DataLoader(dataset_simple, batch_size=batch_size, shuffle=True)
        images, labels = next(iter(loader))
        print(f"  Batch: images={tuple(images.shape)}, labels={labels.tolist()}")

        assert images.shape == (batch_size, 3, 256, 256)
        print("  PASS")

        # ============================================================
        print(f"\n{'=' * 60}")
        print("ALL END-TO-END TESTS PASSED")
        print(f"{'=' * 60}")
        return True

    finally:
        # Limpiar tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    ok = test_annotate_and_load()
    sys.exit(0 if ok else 1)
