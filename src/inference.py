"""
Inference module: given paths to two face images, predict whether they are
the same person or different, using a trained Siamese network.
"""
from pathlib import Path
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.logger import get_logger
from src.utils import get_device
from src.model_build.model import build_model

logger = get_logger(__name__)

def load_model_for_inference(config:Dict[str, Any])->nn.Module:
    """
    Load the trained Siamese model from the checkpoint specified in the config.

    Args:
        config: Full configuration dictionary.

    Returns:
        Loaded model in eval mode on the appropriate device.
    """
    device = get_device(config["training"]["device"])
    model_path = config["inference"]["saved_model_path"]
    model = build_model(config).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint)

    model.eval()

    logger.info("model loaded from %s on %s", model_path, device)
    return model

def preprocess_input(image_path:str, image_size: int)->torch.Tensor:
    """
    Load and preprocess an image for inference (same transform as training).

    Args:
        image_path: Path to the image file.
        image_size: Target square size.

    Returns:
        Preprocessed image tensor of shape (1, 3, H, W).
    """
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    img = Image.open(image_path).convert("RGB")

    img_tensor = transform(img).unsqueeze(0) #this adds the batch dimension (1, C, H,W) c=3 for RGB

    return img_tensor

def predict(model:nn.Module, img1_path:str, img2_path:str, config:Dict[Any,str])->Tuple[float, bool]:
    """
    Predict whether two face images belong to the same person.

    Args:
        model: Trained SiameseNetwork.
        img1_path: Path to first image.
        img2_path: Path to second image.
        config: Full configuration dictionary.

    Returns:
        Tuple of (euclidean_distance, is_same_person).
    """
    device = get_device(config["training"]["device"])
    image_size = config["data"]["image_size"]
    threshold = config["inference"]["threshold"]

    img1 = preprocess_input(image_path=img1_path, image_size=image_size).to(device)
    img2 = preprocess_input(image_path=img2_path, image_size=image_size).to(device)

    with torch.no_grad():
        _,_, distances = model(img1, img2)
        distance_value = distances.item()

    is_same = distance_value < threshold
    logger.info("compared %s adn %s: distance=%.4f, same=%s",Path(img1_path).name, Path(img2_path).name, distance_value, is_same)
    return distance_value, is_same
