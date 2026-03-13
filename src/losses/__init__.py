"""
SOTA Loss Functions and Regularizers for Deep Metric Learning.

Implements:
- NIR: Non-Isotropy Regularization (Roth et al., CVPR 2022)
- HIER: Hierarchical Regularization (Kim et al., CVPR 2023)
- SG-Softmax: Stop-Gradient Softmax Loss (Yang et al., AAAI 2023)
- DADA: Data-Augmented Domain Adaptation wrapper (Ren et al., AAAI 2024)
"""

from src.losses.nir_regularizer import NIRRegularizer
from src.losses.hier_regularizer import HIERRegularizer
from src.losses.sg_softmax import SGSoftmaxLoss
from src.losses.dada_wrapper import DADAWrapper

__all__ = [
    'NIRRegularizer',
    'HIERRegularizer',
    'SGSoftmaxLoss',
    'DADAWrapper',
]
