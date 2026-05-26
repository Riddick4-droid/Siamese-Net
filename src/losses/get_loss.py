import torch 
import torch.nn as nn
from typing import Dict, Any

from src.logger import get_logger
from src.losses.contrasiveloss import ContrastiveLoss
from src.losses.tripletloss import TripletLoss


logger = get_logger(__name__)

def get_loss_function(config: Dict[str,Any])->nn.Module:
    """
    Factory that returns the appropriate loss module based on the config.
    """
    loss_type = config["loss"]["type"]
    margin = config["loss"]["margin"]

    if loss_type == "contrastive":
        logger.info("using contrastive with margin=%.2f", margin)
        return ContrastiveLoss(margin=margin)
    elif loss_type == "triplet":
        mining = config["loss"].get("mining","semi-hard")
        logger.info("using triplet loss with margin=%.2f, mining=%s", margin, mining)
        return TripletLoss(margin=margin, mining=mining)
    else:
        raise ValueError(f"unknown loss type: {loss_type}.choose either 'contrastive' or 'triplet'")