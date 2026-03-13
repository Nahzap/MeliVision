"""
Hierarchical Regularization (HIER) for Deep Metric Learning.
Based on: Kim, Jeong & Kwak (2023) "HIER: Metric Learning Beyond Class Labels
via Hierarchical Regularization" (CVPR 2023)

Learns a hierarchical proxy tree that captures multi-level structure
beyond flat class labels. Agnostic to the base loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class HIERRegularizer(nn.Module):
    """
    Hierarchical Regularization for metric learning.
    
    Builds a binary proxy tree where leaf proxies correspond to classes
    and internal nodes capture shared structure. The regularization loss
    enforces that parent-child distances reflect the hierarchy.
    
    L_HIER = lambda * sum_levels[ CrossEntropy(sim(x, proxies_level), labels_level) ]
    
    Args:
        embedding_dim (int): Dimension of embeddings.
        num_classes (int): Number of leaf classes.
        num_levels (int): Depth of the proxy tree. Default 3.
        lambda_hier (float): Weight of HIER term. Default 0.1.
        temperature (float): Temperature for softmax. Default 0.1.
    """
    
    def __init__(self, embedding_dim, num_classes, num_levels=3,
                 lambda_hier=0.1, temperature=0.1):
        super().__init__()
        self.num_levels = num_levels
        self.lambda_hier = lambda_hier
        self.temperature = temperature
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Leaf proxies (one per class)
        self.leaf_proxies = nn.Parameter(
            torch.randn(num_classes, embedding_dim) * 0.01
        )
        nn.init.kaiming_normal_(self.leaf_proxies.unsqueeze(0))
        
        # Internal proxies for each level (binary tree above leaves)
        # Level 0 = leaves (num_classes), Level 1 = ceil(num_classes/2), etc.
        self.internal_proxies = nn.ParameterList()
        n = num_classes
        for level in range(num_levels):
            n = max(1, (n + 1) // 2)
            self.internal_proxies.append(
                nn.Parameter(torch.randn(n, embedding_dim) * 0.01)
            )
        
        logger.info(
            f"HIER initialized: {num_classes} leaves, {num_levels} levels, "
            f"lambda={lambda_hier}"
        )
    
    def _get_parent_labels(self, labels, level):
        """Map leaf labels to parent labels at a given level."""
        parent_labels = labels.clone()
        for _ in range(level + 1):
            parent_labels = parent_labels // 2
        return parent_labels
    
    def forward(self, embeddings, labels):
        """
        Compute hierarchical regularization loss.
        
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels
            
        Returns:
            Scalar HIER regularization loss
        """
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        hier_loss = torch.tensor(0.0, device=embeddings.device)
        
        # Level 0: Leaf-level proxy assignment (standard proxy loss)
        leaf_proxies_norm = F.normalize(self.leaf_proxies, p=2, dim=1)
        sim_leaf = embeddings_norm @ leaf_proxies_norm.t() / self.temperature
        hier_loss = hier_loss + F.cross_entropy(sim_leaf, labels)
        
        # Higher levels: hierarchical grouping
        for level_idx, proxies in enumerate(self.internal_proxies):
            parent_labels = self._get_parent_labels(labels, level_idx)
            
            # Clamp to valid range
            num_parent_classes = proxies.shape[0]
            parent_labels = parent_labels.clamp(0, num_parent_classes - 1)
            
            proxies_norm = F.normalize(proxies, p=2, dim=1)
            sim = embeddings_norm @ proxies_norm.t() / self.temperature
            hier_loss = hier_loss + F.cross_entropy(sim, parent_labels)
        
        return self.lambda_hier * hier_loss / (1 + len(self.internal_proxies))
