"""
Command-line interface for the Siamese Face Similarity project.

Commands:
    ingest    Download and cache the dataset.
    train     Run the training loop.
    evaluate  Evaluate a trained model on the test set.
    infer     Predict whether two face images are the same person.

Usage examples:
    python main.py ingest --config configs/config.yaml
    python main.py train --config configs/config.yaml
    python main.py evaluate --config configs/config.yaml --model-path checkpoints/best_model.pth
    python main.py infer --config configs/config.yaml --img1 path/to/face1.jpg --img2 path/to/face2.jpg
"""

import argparse
import sys

from src.logger import get_logger
from src.utils import load_config

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Siamese network for facial similarity -CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    def add_config_args(p):
        p.add_argument("--config", required=True, type=str,help="path to yaml configs file")

    #ingest
    p_ingest = subparsers.add_parser("ingest", help="download and cache the dataset")
    add_config_args(p_ingest)

    #train
    p_train = subparsers.add_parser("train", help="train the model")
    add_config_args(p_train)

    #evaluate
    p_eval = subparsers.add_parser("evaluate" ,help="evalaute a trained model on the test set.")
    add_config_args(p_eval)
    p_eval.add_argument("--model-path", required=True, help="path to the saved model checkpoint (.pth)")
    p_eval.add_argument("--output-dir", default=None, help="directory to save the evaluation artefacts (defaults from config file)")

    #infer
    p_infer = subparsers.add_parser("infer", help="compare two face images")
    add_config_args(p_infer)
    p_infer.add_argument("--img1", required=True, help="Path to the first image")
    p_infer.add_argument("--img2", required=True, help="Path to second face image")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    #load config once
    config = load_config(args.config)
    logger.info("Loaded configuration from %s", args.config)

    # Dispatch
    if args.command == "ingest":
        from src.data_extractor.data_ingestion import download_and_cache_dataset
        download_and_cache_dataset(config)
        logger.info("Data ingestion completed.")
        
    elif args.command == "train":
        from src.train_model.trainer import train
        train(config)
        logger.info("Training completed.")

    elif args.command == "evaluate":
        from src.evaluate_model.evaluate import evaluate
        evaluate(config, model_path=args.model_path, output_dir=args.output_dir)
        logger.info("Evaluation completed.")

    elif args.command == "infer":
        from src.inference import load_model_for_inference, predict
        model = load_model_for_inference(config)
        distance, is_same = predict(model, args.img1, args.img2, config)
        print(f"Distance: {distance:.4f}")
        print(f"Same person: {is_same}")

#standalone--allows this file to run on its own-especially on the command line
if __name__ == "__main__":
    main()