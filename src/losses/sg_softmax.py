"""
Stop-Gradient Softmax Loss (SG-Softmax) for Deep Metric Learning.
Based on: Yang, Wang & Zhang (2023) "Stop-Gradient Softmax Loss for
Deep Metric Learning" (AAAI 2023)

Applies stop_gradient to the softmax denominator to stabilize gradients
and achieve superior performance over Multi-Similarity Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SGSoftmaxLoss(nn.Module):
    """
    Stop-Gradient Softmax Loss.
    
    A proxy-based loss that normalizes embeddings and proxies on the 
    hypersphere, adds an angular margin to positive pairs, and applies
    stop-gradient to the log-sum-exp denominator for stable optimization.
    
    Args:
        embedding_dim (int): Dimension of embeddings.
        num_classes (int): Number of classes.
        scale (float): Scaling factor for logits. Default 30.0.
        margin (float): Additive angular margin. Default 0.4.
    """
    
    def __init__(self, embedding_dim, num_classes, scale=30.0, margin=0.4):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes
        
        # Learnable class proxies
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies.unsqueeze(0))
        
        logger.info(
            f"SG-Softmax initialized: {num_classes} classes, "
            f"scale={scale}, margin={margin}"
        )
    
    def forward(self, embeddings, labels):
        """
        Compute SG-Softmax loss.
        
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels
            
        Returns:
            Scalar loss
        """
        # Normalize embeddings and proxies
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        # Cosine similarity: (B, C)
        sim = embeddings @ proxies.t()
        
        # Add angular margin to positive (target) classes
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        sim_with_margin = sim - self.margin * one_hot
        
        # Scale logits
        logits = self.scale * sim_with_margin
        
        # Stop-gradient softmax: detach denominator
        log_numerator = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        log_denominator = torch.logsumexp(logits.detach(), dim=1)
        
        loss = -(log_numerator - log_denominator).mean()
        return loss
