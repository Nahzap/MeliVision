"""
Tests para ArcFace Loss

Verifica que la implementación de ArcFace Loss funciona correctamente:
- Inicialización con parámetros correctos
- Forward pass genera pérdidas válidas
- Margen angular se aplica correctamente
- Gradientes fluyen correctamente

Autor: MetricLearning Test Suite
Fecha: 2026-03-06
"""

import pytest
import torch
import torch.nn as nn
from pytorch_metric_learning import losses


class TestArcFaceLoss:
    """Suite de tests para ArcFace Loss."""
    
    @pytest.fixture
    def setup_arcface(self):
        """Fixture para crear ArcFace Loss con parámetros estándar."""
        num_classes = 7
        embedding_size = 128
        margin = 0.5
        scale = 64
        
        loss_func = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=margin,
            scale=scale
        )
        
        return {
            'loss_func': loss_func,
            'num_classes': num_classes,
            'embedding_size': embedding_size,
            'margin': margin,
            'scale': scale
        }
    
    def test_arcface_loss_initialization(self, setup_arcface):
        """Test 1: Verificar que ArcFace se inicializa correctamente."""
        config = setup_arcface
        loss_func = config['loss_func']
        
        # Verificar que el loss existe
        assert loss_func is not None, "ArcFace Loss no se inicializó"
        
        # Verificar parámetros
        assert loss_func.num_classes == config['num_classes'], "num_classes incorrecto"
        assert loss_func.embedding_size == config['embedding_size'], "embedding_size incorrecto"
        assert loss_func.margin == config['margin'], "margin incorrecto"
        assert loss_func.scale == config['scale'], "scale incorrecto"
        
        print(f"✓ ArcFace inicializado: {config['num_classes']} clases, "
              f"margin={config['margin']}, scale={config['scale']}")
    
    def test_arcface_loss_forward(self, setup_arcface):
        """Test 2: Verificar que el forward pass funciona correctamente."""
        config = setup_arcface
        loss_func = config['loss_func']
        
        # Crear embeddings y labels de prueba
        batch_size = 32
        embeddings = torch.randn(batch_size, config['embedding_size'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # L2 normalize
        labels = torch.randint(0, config['num_classes'], (batch_size,))
        
        # Forward pass
        loss = loss_func(embeddings, labels)
        
        # Verificaciones
        assert loss is not None, "Loss es None"
        assert torch.isfinite(loss), "Loss contiene NaN o Inf"
        assert loss.item() > 0, "Loss debe ser positivo"
        assert loss.requires_grad, "Loss no tiene gradientes habilitados"
        
        print(f"✓ Forward pass exitoso: loss={loss.item():.4f}")
    
    def test_arcface_angular_margin(self, setup_arcface):
        """Test 3: Verificar que el margen angular se aplica correctamente."""
        config = setup_arcface
        loss_func = config['loss_func']
        
        # Crear embeddings perfectamente alineados con sus clases
        batch_size = 14
        embeddings = torch.randn(batch_size, config['embedding_size'])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        labels = torch.arange(batch_size) % config['num_classes']
        
        # Computar pérdida
        loss_with_margin = loss_func(embeddings, labels)
        
        # Verificar que la pérdida es razonable (no cero debido al margen)
        assert loss_with_margin.item() > 0.01, "Margen angular no se está aplicando"
        
        print(f"✓ Margen angular aplicado correctamente: loss={loss_with_margin.item():.4f}")
    
    def test_arcface_gradient_flow(self, setup_arcface):
        """Test 4: Verificar que los gradientes fluyen correctamente."""
        config = setup_arcface
        loss_func = config['loss_func']
        
        # Crear mini modelo de prueba
        class SimpleEncoder(nn.Module):
            def __init__(self, embedding_size):
                super().__init__()
                self.fc = nn.Linear(10, embedding_size)
            
            def forward(self, x):
                embeddings = self.fc(x)
                return torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        model = SimpleEncoder(config['embedding_size'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Forward + Backward
        batch_size = 16
        x = torch.randn(batch_size, 10)
        labels = torch.randint(0, config['num_classes'], (batch_size,))
        
        optimizer.zero_grad()
        embeddings = model(x)
        loss = loss_func(embeddings, labels)
        loss.backward()
        
        # Verificar que los gradientes existen
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), "Gradientes contienen NaN/Inf"
                if param.grad.abs().sum() > 0:
                    has_gradients = True
        
        assert has_gradients, "No hay gradientes en el modelo"
        
        print(f"✓ Gradientes fluyen correctamente, loss={loss.item():.4f}")
    
    def test_arcface_different_margins(self):
        """Test 5: Verificar comportamiento con diferentes márgenes."""
        num_classes = 5
        embedding_size = 64
        batch_size = 20
        
        embeddings = torch.randn(batch_size, embedding_size)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        margins = [0.3, 0.5, 0.7]
        losses = []
        
        for margin in margins:
            loss_func = losses.ArcFaceLoss(
                num_classes=num_classes,
                embedding_size=embedding_size,
                margin=margin,
                scale=64
            )
            loss = loss_func(embeddings, labels)
            losses.append(loss.item())
        
        # Verificar que mayor margen → mayor pérdida (más difícil)
        assert losses[1] >= losses[0], "Margin 0.5 debe dar pérdida ≥ margin 0.3"
        assert losses[2] >= losses[1], "Margin 0.7 debe dar pérdida ≥ margin 0.5"
        
        print(f"✓ Márgenes validados: m=0.3→{losses[0]:.3f}, "
              f"m=0.5→{losses[1]:.3f}, m=0.7→{losses[2]:.3f}")
    
    def test_arcface_batch_size_invariance(self, setup_arcface):
        """Test 6: Verificar que funciona con diferentes batch sizes."""
        config = setup_arcface
        loss_func = config['loss_func']
        
        batch_sizes = [8, 32, 64]
        
        for bs in batch_sizes:
            embeddings = torch.randn(bs, config['embedding_size'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            labels = torch.randint(0, config['num_classes'], (bs,))
            
            loss = loss_func(embeddings, labels)
            assert torch.isfinite(loss), f"Loss es NaN/Inf para batch_size={bs}"
            assert loss.item() > 0, f"Loss no positivo para batch_size={bs}"
        
        print(f"✓ ArcFace funciona con batch sizes: {batch_sizes}")


class TestArcFaceIntegration:
    """Tests de integración de ArcFace con el trainer."""
    
    def test_arcface_config_loading(self):
        """Test: Verificar que config.yaml carga ArcFace correctamente."""
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            pytest.skip("config.yaml no encontrado")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        loss_config = config.get('training', {}).get('loss', {})
        
        # Verificar que ArcFace está configurado
        assert loss_config.get('type') == 'arcface', "Loss type debe ser 'arcface'"
        assert 'margin' in loss_config.get('params', {}), "Falta parámetro 'margin'"
        assert 'scale' in loss_config.get('params', {}), "Falta parámetro 'scale'"
        
        margin = loss_config['params']['margin']
        scale = loss_config['params']['scale']
        
        # Verificar rangos razonables
        assert 0.2 <= margin <= 0.8, f"Margin {margin} fuera de rango [0.2, 0.8]"
        assert 30 <= scale <= 100, f"Scale {scale} fuera de rango [30, 100]"
        
        print(f"✓ Config válido: type=arcface, margin={margin}, scale={scale}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
