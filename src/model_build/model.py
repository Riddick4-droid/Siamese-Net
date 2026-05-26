##this is the final model which combines the feature extractor and the embedding model to the siamese model which
#computes the pairwise or euclidean distance between the two image embeddings

import torch
import torch.nn as nn
from typing import Dict, Any

from src.model_build.embedding import build_embedding_network
from src.logger import get_logger

logger = get_logger(__name__)

class SiameseNet(nn.Module):
    """
    Siamese network that takes two input images, passes them through a shared
    embedding network, and returns both embeddings and their Euclidean distance.
    """
    def __init__(self, embedding_net:nn.Module):
        super(SiameseNet,self).__init__()
        self.embedding_net = embedding_net

    def forward(self, img1:torch.Tensor, img2:torch.Tensor):
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)
        distance = torch.norm(emb1-emb2, p=2, dim=1)
        return emb1, emb2, distance
    
def build_model(config:Dict[str,Any])->SiameseNet:
    """
    Build the full Siamese network from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        SiameseNetwork instance with the chosen embedding backbone.
    """
    embedding_net = build_embedding_network(config=config)
    model= SiameseNet(embedding_net=embedding_net)
    logger.info("SiameseNet built with backbone: %d", config["model"]["backbone"])
    return model

