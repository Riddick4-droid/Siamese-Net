#this module designs and explain the triiplet loss for siamese network which
#includes taking 3 images-the anchor, the positive and the negative
#it is designed with online hard and semi-hard mining. for each anchor
#in the batch, selects the hardest positive (same idenity) and hardest negative (different identity) to form a triplet
import torch
import torch.nn as nn
from typing import Dict, Any

from src.logger import get_logger

logger = get_logger(__name__)

class TripletLoss(nn.Module):
    """
    Triplet loss with online hard or semi‑hard mining.
    For each anchor in the batch, selects the hardest positive (same identity)
    and hardest negative (different identity) to form a triplet.
    """
    def __init__(self,margin:float=1.0, mining: str = "semi-hard"):
        """
        Args:
            margin: Desired separation between positive and negative distances.
            mining: Mining strategy ('hard' or 'semi-hard').
        """
        super().__init__()
        self.margin = margin
        self.mining = mining.lower()

    def forward(self, embeddings:torch.Tensor, labels:torch.Tensor)->torch.Tensor:
        """
        Args:
            embeddings: 2D tensor of shape (batch_size, embedding_dim).
            labels: 1D tensor of identity labels (any integer type) of length batch_size.
        Returns:
            Scalar triplet loss, or zero if no valid triplets can be formed.
        """
        #compute all pairwise distances (squares euclidean would also work here, but we use p=2)
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)

        batch_size = embeddings.size(0)
        loss = 0.0
        count=0

        #iterate over every sample as a potential anchor
        for i in range(batch_size):
            #positive mask: same identity but not the anchor itself
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size) != i)

            #negative mask: different identity
            neg_mask = labels != labels[i]

            #if no positive or  no negative exists for this anchor, skip it
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            #hardest positive:  smallest distance among positives
            pos_dist = distance_matrix[i][pos_mask].min()
            
            #hardest negative: smallest distance among negatives (most confusing)
            neg_dist = distance_matrix[i][neg_mask].min()

            if self.mining == "semi-hard":
                #for semi-hard we only consider triplets where negative is harder than positive
                #but not yet within the margin (i.e neg_dist > pos_dist and (neg_dist - pos_dist)<margin)
                if neg_dist > pos_dist and (neg_dist-pos_dist) < self.margin:
                    loss += torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
                    count+=1
            elif self.mining == "hard":
                #hard mining: always use hardest negative regardless of distance
                loss += torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
                count +=1

        if count == 0:
            #if no triplets could be formed, return a zero loss that still requires grad
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        return loss / count

