##the contrastive loss takes in embeddings distance and labels
#it encourages the model to use a small distance between embeddings with small distances
#and large distances for emebeddings with large distances

import torch
import torch.nn as nn
from typing import Dict, Any



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss as defined in:
    "Dimensionality Reduction by Learning an Invariant Mapping" (Hadsell et al.)
    Encourages small Euclidean distance for same‑person pairs and large distance
    (at least margin) for different‑person pairs.
    """
    def __init__(self, margin: float = 2.0):
        """
        Args:
            margin: The minimum distance that should separate negative pairs.
        """
        super().__init__()
        self.margin = margin  # Store the margin as an instance attribute

    def forward(self, distance:torch.Tensor, label:torch.Tensor)-> torch.Tensor:
        """
        Args:
            distance: 1D tensor of Euclidean distances between embeddings (batch_size,).
            label: 1D tensor of ground truth labels (1 for same, 0 for different).
            e.g (1. dist: [0.157]: label ==1, 2. dist: [2.57]:label==0)
        Returns:
            Scalar loss value.
        """
        #positive pairs: where label == 1: loss = 0.5 * distance^2
        loss_poss = 0.5 * (distance**2)

        #negative pairs: loss = 0.5 * max(0, margin-distance)^2
        #clamp ensures we dont push distances beyond the margin

        loss_neg = (1-label) * (torch.clamp(self.margin - distance,min=0.0)**2)

        #combine both terms, multiply by 0.5 and take the mean over the batch
        total_loss = (loss_poss + loss_neg).mean() / 2.0
        return total_loss
    
