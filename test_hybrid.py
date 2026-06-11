import torch
from src.losses.hybrid_loss import HybridProxyMSLoss

if __name__ == '__main__':
    print("Testing HybridProxyMSLoss...")
    loss_fn = HybridProxyMSLoss(embedding_dim=128, num_classes=10)
    
    embeddings = torch.nn.functional.normalize(torch.randn(32, 128, requires_grad=True), p=2, dim=1)
    labels = torch.randint(0, 10, (32,))
    
    # Check what returns nan
    proxy_loss = loss_fn.proxy_loss(embeddings, labels)
    ms_loss = loss_fn.ms_loss(embeddings, labels)
    print(f"Proxy loss: {proxy_loss}")
    print(f"MS loss: {ms_loss}")
    
    loss = loss_fn(embeddings, labels)
    print(f"Loss computed successfully: {loss.item()}")
    
    # Check backward pass
    loss.backward()
    print("Backward pass successful!")
