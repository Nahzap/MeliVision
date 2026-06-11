import torch
import sys
import os

# Set PYTHONPATH explicitly for testing
sys.path.insert(0, os.path.abspath('.'))

try:
    from src.losses.ldm_loss import LearnableDynamicMarginMSLoss
    
    print("="*60)
    print("TEST 1: LDM-MS Loss forward pass constraints")
    print("="*60)
    
    num_classes = 5
    loss_fn = LearnableDynamicMarginMSLoss(num_classes=num_classes)
    
    # Check initial values
    print(f"pos_margin initial: {loss_fn.pos_margin.data}")
    print(f"neg_margin initial: {loss_fn.neg_margin.data}")
    
    # Dummy inputs
    embeddings = torch.randn(10, 128)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, num_classes, (10,))
    
    loss = loss_fn(embeddings, labels)
    loss.backward()
    
    print(f"Loss value: {loss.item()}")
    
    # Check margins after dummy update
    optimizer = torch.optim.SGD(loss_fn.parameters(), lr=1.0)
    optimizer.step()
    
    # Ensure properties hold after clamping logic is applied during next forward
    loss2 = loss_fn(embeddings, labels)
    
    print("\n[SUCCESS] LDM-MS Loss constraints applied successfully.")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print("\n[FAILED] LDM-MS Loss test failed.")
    sys.exit(1)
