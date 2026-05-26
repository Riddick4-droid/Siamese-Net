"""
AWS SageMaker training entry point.
Parses hyperparameters from the command line, overrides a base YAML configuration,
and launches the training loop.
"""
import argparse
import os
import json
from pathlib import Path

from src.logger import get_logger
from src.utils import load_config
from src.train_model.trainer import train

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Sagemaker siamese training")

    # Base config file (optional)
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to base YAML config (default: configs/config.yaml)")

    # Hyperparameters (override config)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--margin", type=float, default=None)

    # SageMaker specific
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./checkpoints"),
                        help="Directory where the model will be saved (default: SM_MODEL_DIR or ./checkpoints)")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", None),
                        help="Optional training data channel (not used by our kagglehub flow, for compatibility)")

    return parser.parse_args()


def merge_config(base_config, args):
    """Override base config fields with command-line arguments if provided."""
    if args.epochs is not None:
        base_config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        base_config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        base_config["training"]["learning_rate"] = args.lr
    if args.embedding_dim is not None:
        base_config["model"]["embedding_dim"] = args.embedding_dim
    if args.backbone is not None:
        base_config["model"]["backbone"] = args.backbone
    if args.loss is not None:
        base_config["loss"]["type"] = args.loss
    if args.margin is not None:
        base_config["loss"]["margin"] = args.margin
    # Override checkpoint_dir to model_dir so SageMaker can package artefacts
    base_config["training"]["checkpoint_dir"] = args.model_dir
    return base_config

def main():
    args = parse_args()
    logger.info("Starting SageMaker training with arguments: %s", args)

    # Load base config
    if not os.path.exists(args.config):
        logger.warning("Config file %s not found. Using default minimal config.", args.config)
        # minimal fallback
        config = {
            "data": {
                "dataset_name": "jessicali9530/lfw-dataset",
                "cache_dir": "~/.cache/siamese_face",
                "image_size": 128, #or 256
                "allowed_extensions": [".jpg", ".jpeg", ".png"]
            },
            "split": {
                "train_ratio": 0.72,
                "val_ratio": 0.18,
                "test_ratio": 0.10,
                "seed": 42
            },
            "model": {"embedding_dim": 128, "backbone": "custom_cnn"},
            "loss": {"type": "contrastive", "margin": 2.0},
            "training": {
                "batch_size": 64, "epochs": 5, "learning_rate": 0.001,
                "weight_decay": 0.0001, "num_workers": 2, "log_interval": 20,
                "device": "auto", "checkpoint_dir": args.model_dir, "resume_from": None
            },
            "validation": {
                "batch_size": 64, "num_workers": 2,
                "t_sne_epochs": [1, 5], "t_sne_samples": 8, "t_sne_images_per_person": 5
            },
            "evaluation": {"threshold": 1.0, "output_dir": "artefacts/"},
            "logging": {"level": "INFO", "mlflow_tracking_uri": None, "mlflow_experiment_name": "siamese_face"}
        }
    else:
        config = load_config(args.config)

    # Merge SageMaker overrides
    config = merge_config(config, args)

    logger.info("Configuration: %s", json.dumps(config, indent=2, default=str))

    # Ensure checkpoint/model directory exists
    Path(config["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # Run training
    model = train(config)

    logger.info("Training finished. Model artefacts saved to %s", config["training"]["checkpoint_dir"])


if __name__ == "__main__":
    main()