import torch
import sys
import os

# Set PYTHONPATH explicitly for testing
sys.path.insert(0, os.path.abspath('.'))

try:
    from src.models.analogy_net import AnalogyNet
    from src.losses.amplitude_penalty import AmplitudePenaltyLoss
    
    print("="*60)
    print("TEST 1: AnalogyNet Forward Pass & return_prenorm flag")
    print("="*60)
    
    # 1. Configurar modelo (con projection head similar a config.yaml)
    model = AnalogyNet(
        backbone='resnet18',
        embedding_dim=128,
        projection_head_config={
            'hidden_dims': [256, 256],
            'use_batch_norm': True,
            'dropout': 0.1,
            'normalize_embeddings_ln': False,
            'use_hardtanh': False
        }
    )
    
    # Batch dummy de 4 imágenes
    x = torch.randn(4, 3, 224, 224)
    
    # Test flag
    embeddings, prenorm = model(x, return_prenorm=True)
    
    print(f"[+] Embeddings shape: {embeddings.shape}")
    print(f"[+] Prenorm shape: {prenorm.shape}")
    
    # Verificar normas L2
    norm_embeddings = embeddings.norm(p=2, dim=1)
    norm_prenorm = prenorm.norm(p=2, dim=1)
    
    print(f"\n[+] Norma L2 de embeddings (esperado ~1.0): {norm_embeddings.mean().item():.4f}")
    print(f"[+] Norma L2 de prenorm (esperado >1.0): {norm_prenorm.mean().item():.4f}")
    
    assert torch.allclose(norm_embeddings, torch.ones_like(norm_embeddings), atol=1e-4), "Los embeddings finales NO están normalizados a 1."
    assert norm_prenorm.mean().item() > 1.4, "Los embeddings crudos (prenorm) son sospechosamente pequeños."
    
    print("\n" + "="*60)
    print("TEST 2: Verificación de Amplitude Penalty Loss")
    print("="*60)
    
    # 2. Probar la función de pérdida que estaba fallando
    amp_loss_fn = AmplitudePenaltyLoss(lambda_penalty=1.0, penalty_type='clamping', margin=0.5)
    
    # Valores absolutos promedio (para entender por qué margin=0.5 mataba el gradiente antes)
    print(f"[+] Valor absoluto medio de embeddings normalizados: {embeddings.abs().mean().item():.5f}")
    print(f"[+] Valor absoluto máximo de embeddings normalizados: {embeddings.abs().max().item():.5f}")
    print(f"[+] Valor absoluto medio de prenorm: {prenorm.abs().mean().item():.5f}")
    print(f"[+] Valor absoluto máximo de prenorm: {prenorm.abs().max().item():.5f}")
    
    # Probar con embeddings normalizados (el comportamiento BUGGEADO anterior)
    loss_dead = amp_loss_fn(embeddings, None)
    print(f"\n[+] Amplitude Penalty usando vectores normalizados (Antiguo Bug): {loss_dead.item():.6f}")
    assert loss_dead.item() < 1e-6, "¡La pérdida debería ser 0 con margin=0.5 sobre vectores unitarios!"
    
    # Probar con prenorm con un margin menor (e.g. 0.05) para asegurar gradientes
    # Dado que el max(abs(prenorm)) es ~0.35, un margen de 0.05 forzará que Relu(abs(x) - margin) sea > 0
    amp_loss_fn_grad = AmplitudePenaltyLoss(lambda_penalty=1.0, penalty_type='clamping', margin=0.05)
    loss_fixed = amp_loss_fn_grad(prenorm, None)
    print(f"[+] Amplitude Penalty usando vectores pre-normalizados (Arreglado, margin=0.05): {loss_fixed.item():.6f}")
    assert loss_fixed.item() > 0.0, "¡La pérdida arreglada debería ser > 0 para forzar gradientes!"
    
    # 3. Test de gradientes
    print("\n" + "="*60)
    print("TEST 3: Flujo de Gradientes")
    print("="*60)
    
    loss_fixed.backward()
    
    grad_norm = 0.0
    for name, param in list(model.projection_lf.named_parameters()) + list(model.projection_hf.named_parameters()):
        if param.grad is not None:
            grad_norm += param.grad.norm().item()
            
    print(f"[+] Norma total de gradientes fluyendo por el projection head: {grad_norm:.6f}")
    assert grad_norm > 0, "No hay gradientes fluyendo hacia el modelo."
    
    print("\n[SUCCESS] Todos los test de arquitectura y matemáticas pasaron exitosamente.")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print("\n[FAILED] El test falló.")
    sys.exit(1)
