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
        
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels
            
        Returns:
            Scalar NIR loss (to be added to main loss)
        """
        unique_labels = labels.unique()
        nir_loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        
        for c in unique_labels:
            mask = labels == c
            class_embeds = embeddings[mask]  # (n_c, D)
            
            if class_embeds.shape[0] < 3:
                continue
            
            # Center embeddings
            centered = class_embeds - class_embeds.mean(dim=0, keepdim=True)
            
            # Covariance matrix: (D, D)
            n = class_embeds.shape[0]
            cov = (centered.t() @ centered) / (n - 1)
            
            # Add small identity for numerical stability
            reg = self.eps * torch.eye(cov.shape[0], device=cov.device)
            cov_reg = cov + reg
            
            # Log-determinant via eigenvalues (more stable than torch.logdet)
            eigvals = torch.linalg.eigvalsh(cov_reg)
            log_det = eigvals.clamp(min=1e-8).log().sum()
            
            # Negative log-det: minimizing this maximizes determinant (anisotropy)
            nir_loss = nir_loss - log_det
            count += 1
        
        if count > 0:
            nir_loss = self.lambda_nir * (nir_loss / count)
        
        return nir_loss
