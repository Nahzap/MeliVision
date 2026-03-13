"""
Script para extraer embeddings de reference_embeddings.pt y guardarlos en formato numpy.
Ejecutar desde el entorno de MetricLearning donde torch funciona correctamente.
"""
import torch
import numpy as np
from pathlib import Path

def extract_to_numpy(pt_file, output_file):
    """Extrae datos de .pt y guarda en .npz"""
    print(f"Cargando {pt_file}...")
    data = torch.load(pt_file, map_location='cpu')
    
    print(f"Tipo: {type(data)}")
    print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    
    # Extraer tensores
    embeddings = data['embeddings']
    labels = data['labels']
    class_names = data['class_names']
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Classes: {class_names}")
    
    # Convertir a numpy
    embeddings_np = embeddings.cpu().numpy() if embeddings.is_cuda else embeddings.numpy()
    labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()
    
    # Guardar en .npz (formato numpy nativo)
    np.savez_compressed(
        output_file,
        embeddings=embeddings_np,
        labels=labels_np,
        class_names=np.array(class_names, dtype=object)
    )
    
    print(f"✅ Guardado en {output_file}")
    print(f"   Embeddings: {embeddings_np.shape}")
    print(f"   Labels: {labels_np.shape}")
    print(f"   Classes: {len(class_names)}")

if __name__ == '__main__':
    run_dir = Path('runs/2026-02-12_01-09-35')
    pt_file = run_dir / 'reference_embeddings.pt'
    output_file = run_dir / 'reference_embeddings.npz'
    
    extract_to_numpy(pt_file, output_file)
