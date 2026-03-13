"""
Tests unitarios para FAISSIndex — Búsqueda vectorial eficiente (M8).
Cubre: FAISSIndex con flat_l2, flat_ip, search, add, reset, save/load.
"""

import pytest
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _skip_if_no_faiss():
    """Skip test si faiss no está instalado."""
    try:
        import faiss
        return False
    except ImportError:
        return True


skipif_no_faiss = pytest.mark.skipif(_skip_if_no_faiss(), reason="faiss not installed")


@skipif_no_faiss
class TestFAISSIndex:
    
    def _make_embeddings(self, n=100, dim=128, n_classes=3):
        np.random.seed(42)
        embeddings = np.random.randn(n, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.array([i % n_classes for i in range(n)])
        return embeddings, labels
    
    def test_m8_flat_l2_creation(self):
        """M8: Crear FAISSIndex con flat_l2."""
        from src.inference.faiss_index import FAISSIndex
        index = FAISSIndex(dimension=128, index_type='flat_l2')
        assert index.ntotal == 0
    
    def test_m8_flat_ip_creation(self):
        """M8: Crear FAISSIndex con flat_ip (inner product)."""
        from src.inference.faiss_index import FAISSIndex
        index = FAISSIndex(dimension=128, index_type='flat_ip')
        assert index.ntotal == 0
    
    def test_m8_add_vectors(self):
        """M8: Agregar vectores al índice."""
        from src.inference.faiss_index import FAISSIndex
        
        embeddings, labels = self._make_embeddings()
        index = FAISSIndex(dimension=128, index_type='flat_ip')
        index.add(embeddings, labels)
        
        assert index.ntotal == 100
    
    def test_m8_search_returns_correct_shape(self):
        """M8: search debe retornar (distances, indices, labels) con shapes correctos."""
        from src.inference.faiss_index import FAISSIndex
        
        embeddings, labels = self._make_embeddings()
        index = FAISSIndex(dimension=128, index_type='flat_ip')
        index.add(embeddings, labels)
        
        query = embeddings[:5]
        distances, indices, result_labels = index.search(query, k=3)
        
        assert distances.shape == (5, 3), f"Expected (5, 3), got {distances.shape}"
        assert indices.shape == (5, 3), f"Expected (5, 3), got {indices.shape}"
        assert result_labels.shape == (5, 3), f"Expected (5, 3), got {result_labels.shape}"
    
    def test_m8_search_self_is_nearest(self):
        """M8: Cada vector debe ser su propio vecino más cercano."""
        from src.inference.faiss_index import FAISSIndex
        
        embeddings, labels = self._make_embeddings()
        index = FAISSIndex(dimension=128, index_type='flat_ip')
        index.add(embeddings, labels)
        
        distances, indices, _ = index.search(embeddings[:10], k=1)
        
        for i in range(10):
            assert indices[i, 0] == i, f"Vector {i} debe ser su propio vecino, got {indices[i, 0]}"
    
    def test_m8_search_labels_match(self):
        """M8: Labels retornados deben corresponder a los índices."""
        from src.inference.faiss_index import FAISSIndex
        
        embeddings, labels = self._make_embeddings()
        index = FAISSIndex(dimension=128, index_type='flat_ip')
        index.add(embeddings, labels)
        
        _, indices, result_labels = index.search(embeddings[:5], k=3)
        
        for i in range(5):
            for j in range(3):
                idx = indices[i, j]
                assert result_labels[i, j] == labels[idx], \
                    f"Label mismatch at ({i},{j}): got {result_labels[i,j]}, expected {labels[idx]}"
    
    def test_m8_reset(self):
        """M8: reset() debe vaciar el índice."""
        from src.inference.faiss_index import FAISSIndex
        
        embeddings, labels = self._make_embeddings()
        index = FAISSIndex(dimension=128, index_type='flat_ip')
        index.add(embeddings, labels)
        
        assert index.ntotal == 100
        index.reset()
        assert index.ntotal == 0
        assert index.labels is None
    
    def test_m8_save_and_load(self):
        """M8: save/load debe preservar el índice."""
        from src.inference.faiss_index import FAISSIndex
        
        embeddings, labels = self._make_embeddings(n=50, dim=64)
        index = FAISSIndex(dimension=64, index_type='flat_ip')
        index.add(embeddings, labels)
        
        with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as f:
            path = f.name
        
        try:
            index.save(path)
            
            index2 = FAISSIndex(dimension=64, index_type='flat_ip')
            index2.load(path)
            
            assert index2.ntotal == 50
            assert len(index2.labels) == 50
            
            d1, _, _ = index.search(embeddings[:3], k=5)
            d2, _, _ = index2.search(embeddings[:3], k=5)
            np.testing.assert_array_almost_equal(d1, d2, decimal=5)
        finally:
            os.unlink(path)
            labels_path = path + '.labels.npy'
            if os.path.exists(labels_path):
                os.unlink(labels_path)
    
    def test_m8_invalid_index_type(self):
        """M8: Tipo de índice inválido debe lanzar ValueError."""
        from src.inference.faiss_index import FAISSIndex
        
        with pytest.raises(ValueError, match="Unknown index_type"):
            FAISSIndex(dimension=128, index_type='invalid')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
