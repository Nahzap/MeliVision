"""
Tests para Hard Negative Mining (OHEM)

Verifica que la implementación de OHEM funciona correctamente:
- Inicialización con parámetros correctos
- Selección de hard examples basada en pérdida
- Enforcement del ratio configurado
- Constraint de min_samples
- Warmup epochs funcionan correctamente

Autor: MetricLearning Test Suite
Fecha: 2026-03-06
"""

import pytest
import torch
from src.training.hard_mining import HardNegativeMiner


class TestHardNegativeMiner:
    """Suite de tests para Hard Negative Mining."""
    
    @pytest.fixture
    def setup_miner(self):
        """Fixture para crear HardNegativeMiner con parámetros estándar."""
        miner = HardNegativeMiner(
            hard_ratio=0.3,
            min_samples=32,
            warmup_epochs=0,
            adaptive_ratio=False
        )
        return miner
    
    def test_hard_miner_initialization(self):
        """Test 1: Verificar que el miner se inicializa correctamente."""
        miner = HardNegativeMiner(hard_ratio=0.3, min_samples=32)
        
        assert miner is not None, "Miner no se inicializó"
        assert miner.hard_ratio == 0.3, "hard_ratio incorrecto"
        assert miner.min_samples == 32, "min_samples incorrecto"
        assert miner.warmup_epochs == 0, "warmup_epochs incorrecto"
        assert miner.current_epoch == 0, "current_epoch debe iniciar en 0"
        
        print(f"✓ HardNegativeMiner inicializado correctamente: ratio={miner.hard_ratio}")
    
    def test_mine_hard_examples_selection(self, setup_miner):
        """Test 2: Verificar que selecciona correctamente los hard examples."""
        miner = setup_miner
        
        # Crear pérdidas de prueba (B=100)
        batch_size = 100
        loss_per_sample = torch.randn(batch_size)
        
        # Minar hard examples
        hard_mask = miner.mine_hard_examples(loss_per_sample)
        
        # Verificaciones
        assert hard_mask.dtype == torch.bool, "Mask debe ser booleano"
        assert hard_mask.size(0) == batch_size, "Tamaño de mask incorrecto"
        
        num_selected = hard_mask.sum().item()
        expected = max(32, int(0.3 * batch_size))  # max(min_samples, ratio * B)
        
        assert num_selected == expected, f"Seleccionó {num_selected}, esperado {expected}"
        
        # Verificar que seleccionó los más difíciles
        selected_losses = loss_per_sample[hard_mask]
        not_selected_losses = loss_per_sample[~hard_mask]
        
        if len(not_selected_losses) > 0:
            assert selected_losses.min() >= not_selected_losses.max(), \
                "No seleccionó correctamente los hard examples"
        
        print(f"✓ Selección correcta: {num_selected}/{batch_size} samples ({num_selected/batch_size*100:.1f}%)")
    
    def test_hard_ratio_enforcement(self):
        """Test 3: Verificar que el ratio se respeta para diferentes batch sizes."""
        ratios = [0.2, 0.3, 0.5]
        batch_sizes = [50, 100, 200]
        
        for ratio in ratios:
            miner = HardNegativeMiner(hard_ratio=ratio, min_samples=10)
            
            for batch_size in batch_sizes:
                loss_per_sample = torch.randn(batch_size)
                hard_mask = miner.mine_hard_examples(loss_per_sample)
                
                num_selected = hard_mask.sum().item()
                expected = max(10, int(ratio * batch_size))
                
                assert num_selected == expected, \
                    f"ratio={ratio}, B={batch_size}: seleccionó {num_selected}, esperado {expected}"
        
        print(f"✓ Ratio enforcement validado para {len(ratios)} ratios y {len(batch_sizes)} batch sizes")
    
    def test_min_samples_constraint(self):
        """Test 4: Verificar que min_samples se respeta incluso con batch pequeño."""
        miner = HardNegativeMiner(hard_ratio=0.5, min_samples=40)
        
        # Batch muy pequeño (menor que min_samples)
        batch_size = 30
        loss_per_sample = torch.randn(batch_size)
        hard_mask = miner.mine_hard_examples(loss_per_sample)
        
        num_selected = hard_mask.sum().item()
        
        # Debe seleccionar TODO el batch (30) porque min_samples=40 > 30
        assert num_selected == batch_size, \
            f"Con B={batch_size} < min_samples=40, debe seleccionar todo: {num_selected}/{batch_size}"
        
        # Batch mayor que min_samples
        batch_size = 100
        loss_per_sample = torch.randn(batch_size)
        hard_mask = miner.mine_hard_examples(loss_per_sample)
        
        num_selected = hard_mask.sum().item()
        expected = max(40, int(0.5 * 100))  # max(40, 50) = 50
        
        assert num_selected == expected, f"Seleccionó {num_selected}, esperado {expected}"
        
        print(f"✓ min_samples constraint funciona correctamente")
    
    def test_warmup_epochs(self):
        """Test 5: Verificar que durante warmup usa todas las muestras."""
        miner = HardNegativeMiner(hard_ratio=0.3, min_samples=32, warmup_epochs=3)
        
        batch_size = 100
        loss_per_sample = torch.randn(batch_size)
        
        # Epoch 0, 1, 2: warmup (debe seleccionar todo)
        for epoch in range(3):
            miner.set_epoch(epoch)
            hard_mask = miner.mine_hard_examples(loss_per_sample)
            num_selected = hard_mask.sum().item()
            
            assert num_selected == batch_size, \
                f"Epoch {epoch} (warmup): debe seleccionar todo, seleccionó {num_selected}"
        
        # Epoch 3: warmup terminado (debe aplicar ratio)
        miner.set_epoch(3)
        hard_mask = miner.mine_hard_examples(loss_per_sample)
        num_selected = hard_mask.sum().item()
        expected = max(32, int(0.3 * batch_size))
        
        assert num_selected == expected, \
            f"Epoch 3 (post-warmup): seleccionó {num_selected}, esperado {expected}"
        
        print(f"✓ Warmup epochs funcionan correctamente")
    
    def test_adaptive_ratio(self):
        """Test 6: Verificar que adaptive ratio ajusta según pérdida promedio."""
        miner_adaptive = HardNegativeMiner(hard_ratio=0.3, min_samples=10, adaptive_ratio=True)
        miner_adaptive.set_epoch(1)  # Salir de warmup
        
        batch_size = 100
        
        # Pérdidas bajas (fácil) → ratio estándar
        easy_losses = torch.randn(batch_size) * 0.1 + 0.5  # mean ≈ 0.5
        hard_mask_easy = miner_adaptive.mine_hard_examples(easy_losses)
        num_easy = hard_mask_easy.sum().item()
        
        # Pérdidas altas (difícil) → ratio aumentado
        hard_losses = torch.randn(batch_size) * 0.1 + 2.0  # mean ≈ 2.0
        hard_mask_hard = miner_adaptive.mine_hard_examples(hard_losses)
        num_hard = hard_mask_hard.sum().item()
        
        # Con adaptive, pérdidas altas deben seleccionar más samples
        assert num_hard >= num_easy, \
            f"Adaptive debe seleccionar más con pérdida alta: {num_hard} vs {num_easy}"
        
        print(f"✓ Adaptive ratio funciona: easy={num_easy}, hard={num_hard}")
    
    def test_statistics_tracking(self, setup_miner):
        """Test 7: Verificar que las estadísticas se registran correctamente."""
        miner = setup_miner
        
        # Procesar varios batches
        for _ in range(5):
            loss_per_sample = torch.randn(100)
            miner.mine_hard_examples(loss_per_sample)
        
        stats = miner.get_stats()
        
        # Verificar que se registraron
        assert stats['total_batches'] == 5, "total_batches incorrecto"
        assert stats['total_samples_seen'] == 500, "total_samples_seen incorrecto"
        assert stats['total_samples_mined'] > 0, "No se minaron samples"
        assert 0 < stats['avg_mining_ratio'] <= 1.0, "avg_mining_ratio fuera de rango"
        assert stats['mining_active'] == True, "Mining debe estar activo"
        
        print(f"✓ Estadísticas: {stats['total_samples_mined']}/{stats['total_samples_seen']} "
              f"({stats['avg_mining_ratio']*100:.1f}%)")
    
    def test_return_stats_option(self, setup_miner):
        """Test 8: Verificar que return_stats=True funciona."""
        miner = setup_miner
        
        loss_per_sample = torch.randn(100)
        hard_mask, stats = miner.mine_hard_examples(loss_per_sample, return_stats=True)
        
        # Verificar que retornó stats
        assert isinstance(stats, dict), "stats debe ser dict"
        assert 'num_selected' in stats, "Falta num_selected"
        assert 'num_total' in stats, "Falta num_total"
        assert 'selection_ratio' in stats, "Falta selection_ratio"
        assert 'avg_loss_hard' in stats, "Falta avg_loss_hard"
        
        print(f"✓ return_stats funciona: {stats['num_selected']}/{stats['num_total']}")
    
    def test_edge_case_empty_batch(self, setup_miner):
        """Test 9: Verificar comportamiento con batch vacío."""
        miner = setup_miner
        
        loss_per_sample = torch.tensor([])
        
        # Debe manejar gracefully
        try:
            hard_mask = miner.mine_hard_examples(loss_per_sample)
            assert hard_mask.size(0) == 0, "Mask debe estar vacío"
            print("✓ Maneja batch vacío correctamente")
        except Exception as e:
            pytest.fail(f"No debería fallar con batch vacío: {e}")
    
    def test_edge_case_single_sample(self, setup_miner):
        """Test 10: Verificar con batch de 1 sample."""
        miner = setup_miner
        
        loss_per_sample = torch.tensor([1.5])
        hard_mask = miner.mine_hard_examples(loss_per_sample)
        
        # Debe seleccionar el único sample
        assert hard_mask.sum().item() == 1, "Debe seleccionar el único sample"
        assert hard_mask[0] == True, "El sample debe estar seleccionado"
        
        print("✓ Maneja single sample correctamente")


class TestHardMiningIntegration:
    """Tests de integración con el trainer."""
    
    def test_trainer_config_loading(self):
        """Test: Verificar que config.yaml carga hard_mining correctamente."""
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            pytest.skip("config.yaml no encontrado")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        hard_mining_config = config.get('training', {}).get('hard_mining', {})
        
        # Verificar que está configurado
        assert hard_mining_config.get('enabled') in [True, False], "enabled debe ser bool"
        
        if hard_mining_config.get('enabled'):
            assert 'ratio' in hard_mining_config, "Falta parámetro 'ratio'"
            assert 'min_samples' in hard_mining_config, "Falta parámetro 'min_samples'"
            
            ratio = hard_mining_config['ratio']
            min_samples = hard_mining_config['min_samples']
            
            # Verificar rangos razonables
            assert 0.1 <= ratio <= 0.8, f"ratio {ratio} fuera de rango [0.1, 0.8]"
            assert 8 <= min_samples <= 128, f"min_samples {min_samples} fuera de rango [8, 128]"
            
            print(f"✓ Config válido: enabled={hard_mining_config['enabled']}, "
                  f"ratio={ratio}, min_samples={min_samples}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
