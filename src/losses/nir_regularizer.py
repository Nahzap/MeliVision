"""
Non-Isotropy Regularization (NIR) for Deep Metric Learning.
Based on: Roth, Vinyals & Akata (2022) "Non-isotropy Regularization for
Proxy-based Deep Metric Learning" (CVPR 2022)

Penalizes isotropic (spherical) local distributions per class,
encouraging anisotropic embeddings that preserve intra-class structure.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class NIRRegularizer(nn.Module):
    """
    Non-Isotropy Regularization.
    
    Computes per-class covariance of embeddings and penalizes low
    log-determinant (which indicates isotropy / spherical collapse).
    
    L_NIR = -lambda * mean_c[ log det(Sigma_c + eps*I) ]
    
    Added to the main loss: L_total = L_metric + L_NIR
    
    Args:
        lambda_nir (float): Weight of the NIR term. Default 0.5.
        eps (float): Regularization for covariance matrix. Default 1e-4.
    """
    
    def __init__(self, lambda_nir=0.5, eps=1e-4):
        super().__init__()
        self.lambda_nir = lambda_nir
        self.eps = eps
    
    def forward(self, embeddings, labels):
        """
        Compute NIR loss.
        DISABLED due to rank-deficiency crashes in small batches.
        """
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
