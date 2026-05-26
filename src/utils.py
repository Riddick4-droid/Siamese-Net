#these are utitlity functions designed to help in the entire pipeline

import os
import yaml
import random
import torch
import numpy as np
from typing import Any, Dict


def load_config(config_path:str)->Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dict with configuration.

    Raises:
        FileNotFoundError: If config_path does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed:int)->None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str="auto")->torch.device:
    """
    Return a torch.device based on the string.
    If "auto", use CUDA if available, else CPU.

    Args:
        device_str: "cuda", "cpu", or "auto".

    Returns:
        torch.device object.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_str in ("cuda", "cpu"):
        return torch.device(device_str)
    raise ValueError(f"Invalid device string: {device_str}. Choose 'cuda','cpu',or 'auto'")


def get_model_info(model: torch.nn.Module) -> str:
    """
    Return a string with total parameter count and the model's string representation.

    Args:
        model: PyTorch module.

    Returns:
        Info string.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"Model: {model.__class__.__name__}\n"
        f"Total parameters: {total_params:,}\n"
        f"Trainable parameters: {trainable_params:,}\n"
        f"Structure: \n{model}"
    )

