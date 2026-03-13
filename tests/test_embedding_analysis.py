"""
Tests para Embedding Space Analyzer

Verifica que el análisis espectral del espacio latente funciona correctamente:
- Inicialización con embeddings válidos
- Computación de eigenvalues correcta
- Cálculo de effective rank
- Detección de colapso dimensional
- Generación de visualizaciones

Autor: MetricLearning Test Suite
Fecha: 2026-03-06
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.visualization.embedding_analysis import EmbeddingSpaceAnalyzer


class TestEmbeddingSpaceAnalyzer:
    """Suite de tests para análisis espectral."""
    
    @pytest.fixture
    def setup_synthetic_data(self):
        """Fixture para crear embeddings sintéticos de prueba."""
        np.random.seed(42)
        
        n_samples = 200
        n_dim = 128
        n_classes = 4
        
        # Embeddings con estructura de clases
        embeddings = []
        labels = []
        
        for c in range(n_classes):
            # Centroide de clase + ruido
            centroid = np.random.randn(n_dim)
            centroid = centroid / np.linalg.norm(centroid)  # Normalizar
            
            class_embeddings = centroid + np.random.randn(n_samples // n_classes, n_dim) * 0.1
            # Normalizar embeddings (S^{d-1})
            class_embeddings = class_embeddings / np.linalg.norm(class_embeddings, axis=1, keepdims=True)
            
            embeddings.append(class_embeddings)
            labels.extend([c] * (n_samples // n_classes))
        
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        class_names = [f"Class_{i}" for i in range(n_classes)]
        
        return {
            'embeddings': embeddings,
            'labels': labels,
            'class_names': class_names,
            'n_samples': n_samples,
            'n_dim': n_dim,
            'n_classes': n_classes
        }
    
    @pytest.fixture
    def setup_collapsed_data(self):
        """Fixture para crear embeddings con colapso dimensional."""
        np.random.seed(123)
        
        n_samples = 100
        n_dim = 128
        n_classes = 3
        
        # Todos los embeddings en una dirección dominante (colapso)
        dominant_dir = np.random.randn(n_dim)
        dominant_dir = dominant_dir / np.linalg.norm(dominant_dir)
        
        embeddings = []
        labels = []
        
        for c in range(n_classes):
            # 90% en dirección dominante, 10% ortogonal
            class_embs = dominant_dir + np.random.randn(n_samples // n_classes, n_dim) * 0.05
            class_embs = class_embs / np.linalg.norm(class_embs, axis=1, keepdims=True)
            
            embeddings.append(class_embs)
            labels.extend([c] * (n_samples // n_classes))
        
        embeddings = np.vstack(embeddings)
        labels = np.array(labels)
        class_names = [f"Collapsed_{i}" for i in range(n_classes)]
        
        return {
            'embeddings': embeddings,
            'labels': labels,
            'class_names': class_names
        }
    
    def test_analyzer_initialization(self, setup_synthetic_data):
        """Test 1: Verificar que el analyzer se inicializa correctamente."""
        data = setup_synthetic_data
        
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=data['embeddings'],
            labels=data['labels'],
            class_names=data['class_names']
        )
        
        assert analyzer is not None, "Analyzer no se inicializó"
        assert analyzer.n_samples == data['n_samples'], "n_samples incorrecto"
        assert analyzer.n_dim == data['n_dim'], "n_dim incorrecto"
        assert len(analyzer.class_names) == data['n_classes'], "class_names incorrecto"
        
        print(f"✓ Analyzer inicializado: {analyzer.n_samples} samples, dim={analyzer.n_dim}")
    
    def test_eigenvalue_computation(self, setup_synthetic_data):
        """Test 2: Verificar que los eigenvalues se computan correctamente."""
        data = setup_synthetic_data
        
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=data['embeddings'],
            labels=data['labels'],
            class_names=data['class_names']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Analizar un par de clases
            results = analyzer.analyze_class_pair(
                "Class_0", "Class_1", tmpdir
            )
        
        # Verificaciones
        assert 'eigenvalues' in results, "Falta eigenvalues"
        assert 'explained_variance' in results, "Falta explained_variance"
        assert 'cumulative_variance' in results, "Falta cumulative_variance"
        
        eigenvalues = results['eigenvalues']
        
        # Eigenvalues deben estar ordenados descendentemente
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:]), "Eigenvalues no están ordenados"
        
        # Todos deben ser no-negativos
        assert np.all(eigenvalues >= 0), "Eigenvalues negativos detectados"
        
        # Suma de explained_variance debe ser ~1.0
        total_var = results['explained_variance'].sum()
        assert abs(total_var - 1.0) < 0.01, f"Varianza total no es 1.0: {total_var}"
        
        print(f"✓ Eigenvalues computados: λ₁={eigenvalues[0]:.4f}")
    
    def test_effective_rank_calculation(self, setup_synthetic_data):
        """Test 3: Verificar cálculo del effective rank."""
        data = setup_synthetic_data
        
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=data['embeddings'],
            labels=data['labels'],
            class_names=data['class_names']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = analyzer.analyze_class_pair(
                "Class_0", "Class_1", tmpdir
            )
        
        effective_rank = results['effective_rank']
        
        # Effective rank debe estar en rango válido
        assert 1 <= effective_rank <= data['n_dim'], \
            f"Effective rank fuera de rango: {effective_rank}"
        
        # Para embeddings sintéticos bien separados, debería ser relativamente bajo
        # pero no colapsado (>5)
        assert effective_rank > 5, "Effective rank demasiado bajo para datos sintéticos"
        
        print(f"✓ Effective rank: {effective_rank}/{data['n_dim']}")
    
    def test_collapse_detection(self, setup_collapsed_data):
        """Test 4: Verificar detección de colapso dimensional."""
        data = setup_collapsed_data
        
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=data['embeddings'],
            labels=data['labels'],
            class_names=data['class_names']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = analyzer.analyze_class_pair(
                "Collapsed_0", "Collapsed_1", tmpdir
            )
        
        # Datos colapsados deben ser detectados
        assert results['is_collapsed'] == True, \
            "No detectó colapso en datos colapsados artificialmente"
        
        # collapse_severity debe ser alto (>0.8)
        assert results['collapse_severity'] > 0.80, \
            f"Collapse severity muy bajo: {results['collapse_severity']}"
        
        # Effective rank debe ser muy bajo
        assert results['effective_rank'] < 10, \
            f"Effective rank muy alto para datos colapsados: {results['effective_rank']}"
        
        print(f"✓ Colapso detectado: severity={results['collapse_severity']:.3f}, rank={results['effective_rank']}")
    
    def test_healthy_vs_collapsed(self, setup_synthetic_data, setup_collapsed_data):
        """Test 5: Verificar diferencia entre datos saludables y colapsados."""
        healthy_data = setup_synthetic_data
        collapsed_data = setup_collapsed_data
        
        # Analyzer para datos saludables
        healthy_analyzer = EmbeddingSpaceAnalyzer(
            embeddings=healthy_data['embeddings'],
            labels=healthy_data['labels'],
            class_names=healthy_data['class_names']
        )
        
        # Analyzer para datos colapsados
        collapsed_analyzer = EmbeddingSpaceAnalyzer(
            embeddings=collapsed_data['embeddings'],
            labels=collapsed_data['labels'],
            class_names=collapsed_data['class_names']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            healthy_results = healthy_analyzer.analyze_global_spectrum(tmpdir)
            collapsed_results = collapsed_analyzer.analyze_global_spectrum(str(Path(tmpdir) / 'collapsed'))
        
        # Datos saludables deben tener menor collapse severity
        assert healthy_results['collapse_severity'] < collapsed_results['collapse_severity'], \
            "Datos saludables tienen mayor severidad que colapsados"
        
        # Datos saludables deben tener mayor effective rank
        assert healthy_results['effective_rank'] > collapsed_results['effective_rank'], \
            "Datos saludables tienen menor rank que colapsados"
        
        print(f"✓ Healthy: λ₁={healthy_results['collapse_severity']:.3f}, rank={healthy_results['effective_rank']}")
        print(f"✓ Collapsed: λ₁={collapsed_results['collapse_severity']:.3f}, rank={collapsed_results['effective_rank']}")
    
    def test_analyze_all_pairs(self, setup_synthetic_data):
        """Test 6: Verificar análisis de todos los pares."""
        data = setup_synthetic_data
        
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=data['embeddings'],
            labels=data['labels'],
            class_names=data['class_names']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pair_results = analyzer.analyze_all_pairs(tmpdir, max_pairs=3)
        
        # Debe haber generado resultados
        assert len(pair_results) > 0, "No se generaron resultados de pares"
        assert len(pair_results) <= 3, "Se generaron más pares de los solicitados"
        
        # Cada resultado debe tener las keys esperadas
        for pair, results in pair_results.items():
            assert isinstance(pair, tuple), "Key debe ser tuple"
            assert len(pair) == 2, "Pair debe tener 2 elementos"
            assert 'eigenvalues' in results, "Falta eigenvalues"
            assert 'effective_rank' in results, "Falta effective_rank"
            assert 'is_collapsed' in results, "Falta is_collapsed"
        
        print(f"✓ Analizados {len(pair_results)} pares de clases")
    
    def test_visualization_generation(self, setup_synthetic_data):
        """Test 7: Verificar que se generan las visualizaciones."""
        data = setup_synthetic_data
        
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=data['embeddings'],
            labels=data['labels'],
            class_names=data['class_names']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Análisis de un par
            analyzer.analyze_class_pair("Class_0", "Class_1", str(tmppath))
            
            # Verificar que se generó el gráfico
            expected_file = tmppath / "spectral_analysis_Class_0_Class_1.png"
            assert expected_file.exists(), f"No se generó visualización: {expected_file}"
            
            # Verificar que tiene contenido
            assert expected_file.stat().st_size > 0, "Archivo de visualización vacío"
        
        print("✓ Visualización generada correctamente")
    
    def test_report_generation(self, setup_synthetic_data):
        """Test 8: Verificar generación del reporte textual."""
        data = setup_synthetic_data
        
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=data['embeddings'],
            labels=data['labels'],
            class_names=data['class_names']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Análisis de todos los pares
            analyzer.analyze_all_pairs(str(tmppath), max_pairs=3)
            
            # Verificar que se generó el reporte
            report_file = tmppath / "spectral_analysis_report.txt"
            assert report_file.exists(), "No se generó reporte"
            
            # Verificar contenido
            with open(report_file, 'r') as f:
                content = f.read()
            
            assert len(content) > 0, "Reporte vacío"
            assert "SPECTRAL ANALYSIS REPORT" in content, "Falta header en reporte"
            assert "Effective Rank" in content, "Falta información en reporte"
        
        print("✓ Reporte generado correctamente")
    
    def test_edge_case_single_class(self):
        """Test 9: Verificar comportamiento con una sola clase."""
        np.random.seed(42)
        
        embeddings = np.random.randn(50, 64)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.zeros(50, dtype=int)
        class_names = ["OnlyClass"]
        
        analyzer = EmbeddingSpaceAnalyzer(embeddings, labels, class_names)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # analyze_all_pairs no debe generar pares (solo 1 clase)
            pair_results = analyzer.analyze_all_pairs(tmpdir)
            assert len(pair_results) == 0, "No debe haber pares con 1 sola clase"
            
            # analyze_global_spectrum debe funcionar
            global_results = analyzer.analyze_global_spectrum(tmpdir)
            assert 'eigenvalues' in global_results, "Falta eigenvalues en análisis global"
        
        print("✓ Maneja single class correctamente")
    
    def test_edge_case_insufficient_samples(self, setup_synthetic_data):
        """Test 10: Verificar comportamiento con muy pocas muestras."""
        data = setup_synthetic_data
        
        # Tomar solo 2 muestras por clase
        indices = []
        for c in range(data['n_classes']):
            class_indices = np.where(data['labels'] == c)[0][:2]
            indices.extend(class_indices)
        
        embeddings_small = data['embeddings'][indices]
        labels_small = data['labels'][indices]
        
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=embeddings_small,
            labels=labels_small,
            class_names=data['class_names']
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Debe manejar gracefully pocas muestras
            results = analyzer.analyze_class_pair(
                data['class_names'][0], data['class_names'][1], tmpdir
            )
            
            # Puede fallar con "Insufficient samples" o retornar error key
            if 'error' in results:
                assert results['error'] in ['Insufficient samples', 'Zero variance'], \
                    f"Error inesperado: {results.get('error')}"
                print("✓ Maneja pocas muestras correctamente (error esperado)")
            else:
                # Si no falla, debe tener resultados válidos
                assert 'eigenvalues' in results
                print("✓ Procesó pocas muestras exitosamente")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
