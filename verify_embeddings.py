import torch
from src.models.analogy_net import AnalogyNet
import yaml

def verify():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model = AnalogyNet(
        backbone=config['model']['backbone_name'],
        embedding_dim=config['model']['embedding_dim'],
        pretrained=False,
        projection_head_config=config['model']['projection_head']
    )
    model.eval()
    
    # Batch de prueba
    x = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        # Extracción manual de features
        features = model.backbone(x)
        if features.dim() > 2:
            features = features.mean(dim=list(range(2, features.dim())))
        
        # Pasar por projection head capa por capa para ver qué pasa
        out = features
        print(f"Features backbone - min: {out.min():.4f}, max: {out.max():.4f}, mean: {out.mean():.4f}, std: {out.std():.4f}")
        
        # Projection head raw output
        raw_proj = model.projection(out)
        print(f"\nSalida cruda del Projection Head (con LayerNorm y HardTanh):")
        print(f"min: {raw_proj.min():.4f}, max: {raw_proj.max():.4f}, mean: {raw_proj.mean():.4f}, std: {raw_proj.std():.4f}")
        
        # Comprobar saturación (cuántos valores tocan los extremos de Hardtanh -2 y 2)
        saturados_max = (raw_proj >= 1.99).sum().item()
        saturados_min = (raw_proj <= -1.99).sum().item()
        total_vals = raw_proj.numel()
        print(f"Valores saturados en max (+2): {saturados_max}/{total_vals} ({(saturados_max/total_vals)*100:.1f}%)")
        print(f"Valores saturados en min (-2): {saturados_min}/{total_vals} ({(saturados_min/total_vals)*100:.1f}%)")
        
        # Embeddings finales (normalizados L2)
        embeddings = torch.nn.functional.normalize(raw_proj, p=2, dim=1)
        print(f"\nEmbeddings Finales L2-Normalizados:")
        print(f"min: {embeddings.min():.4f}, max: {embeddings.max():.4f}, mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
        print(f"Norma L2 por sample: {torch.norm(embeddings, p=2, dim=1)}")

if __name__ == "__main__":
    verify()
