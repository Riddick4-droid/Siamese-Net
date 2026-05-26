"""
Standalone evaluation on a held‑out test set.
Loads a trained model, computes verification metrics,
generates plots, and saves them to an artefact directory.
Optionally logs to MLflow.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

#src
from src.logger import get_logger
from src.utils import load_config, set_seed, get_device
from src.data_extractor.dataset import create_dataset
from src.model_build.model import build_model
from src.metrics import compute_roc_metrics

from src.train_model.trainer import _plot_distance_distribution, _plot_roc_curve

logger = get_logger(__name__)

def evaluate(
        config:Dict[str,Any],
        model_path: str,
        output_dir: Optional[str]=None,
)->Dict[str,Any]:
    """
    Run evaluation on the test split.

    Args:
        config: Configuration dictionary.
        model_path: Path to saved model weights (.pth).
        output_dir: Directory for saving artefacts; if None, uses config's evaluation.output_dir.

    Returns:
        Dictionary of metrics (AUC, EER, best_threshold, best_accuracy, confusion_matrix).
    """
    set_seed(config["split"]["seed"])
    device = get_device(config["training"]["device"])
    logger.info("using device %s", device)

    #output dir
    if output_dir is None:
        output_dir = config["evaluation"]["output_dir"]
    artefact_dir = Path(output_dir)
    artefact_dir.mkdir(parents=True, exist_ok=True)

    #build test dataset using the unified split
    _,_, test_dataset = create_dataset(config=config)
    if test_dataset is None:
        raise ValueError("test dataset is empty. set test_ratio > 0 in config")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["validation"]["batch_size"],
        shuffle=False,
        num_workers=config["validation"]["num_workers"]
    )

    #load model
    model = build_model(config=config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    logger.info("model loaded from %s", model_path)

    #collect distances  and labels
    all_distances = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Evaluating....")
        for img1, img2, labels in pbar:
            img1, img2 = img1.to(device), img2.to(device)
            _,_, distances = model(img1, img2)
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())

    distances = np.array(all_distances)
    labels = np.array(all_labels)
    logger.info("test set size: %d pairs", len(distances))

    #metrics
    metrics = compute_roc_metrics(distances, labels)

    #plots
    dist_plot_path = artefact_dir / "test_distance_distribution.png"
    _plot_distance_distribution(
        distances=distances, labels=labels,
        best_threshold=metrics["best_threshold"],
        save_path = str(dist_plot_path)
    )

    roc_plot_path = artefact_dir / "test_roc_curve.png"
    _plot_roc_curve(
        np.array(metrics["fpr"]),
        np.array(metrics["tpr"]),
        metrics["auc"],
        metrics["eer"],
        save_path = str(roc_plot_path)
    )

    #save metrics as JSON
    metrics_path = artefact_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k:v for k,v in metrics.items() if k not in ("fpr","tpr")}, f, indent=2)
    logger.info("metrics saved to %s", metrics_path)

    #mlflow (optional)
    if config["logging"].get("mlflow_tracking_uri"):
        import mlflow
        mlflow.set_tracking_uri(config["logging"]["mlflow_tracking_uri"])
        mlflow.set_experiment(config["logging"]["mlflow_experiment_name"])
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics({
                "test_auc": metrics["auc"],
                "test_eer": metrics["eer"],
                "best_threshold": metrics["best_threshold"],
                "best_accuracy": metrics["best_accuracy"],
            })
            mlflow.log_artifact(str(dist_plot_path))
            mlflow.log_artifact(str(roc_plot_path))
            mlflow.log_artifact(str(metrics_path))
            logger.info("Metrics and artefacts logged to MLflow")
    return metrics