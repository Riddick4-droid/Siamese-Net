"""
Training loop for Siamese Network, including validation, visualization, and
experiment tracking (local artefacts + optional MLflow).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from PIL import Image
import random

from src.logger import get_logger
from src.utils import get_device, get_model_info, set_seed
from src.data_extractor.dataset import create_dataset ,SiameseDataset
from src.model_build.model import build_model
from src.losses.get_loss import get_loss_function
from src.metrics import compute_accuracy, compute_roc_metrics
from src.exceptions import ProjectException

logger = get_logger(__name__)

#visulization helper
def _visualize_embeddings(
        model:torch.nn.Module,
        dataset: SiameseDataset,
        device: torch.device,
        num_persons: int=8, #we want only 8 people
        images_per_person: int=3,
        title: str = "t-SNE Embeddings",
        save_path: Optional[str]=None
):
    """t‑SNE plot of embeddings from a subset of identities."""
    model.eval()
    eligible_persons = [(p,imgs) for p, imgs in dataset.person_to_images.items() if len(imgs) >= images_per_person]
    if len(eligible_persons) < num_persons:
        selected = eligible_persons
    else:
        selected = random.sample(eligible_persons, num_persons)
    embeddings = []
    labels = []
    with torch.no_grad():
        for person, img_paths in selected:
            chosen_imgs = random.sample(img_paths, images_per_person) #sample 3 images
            for img_name in chosen_imgs:
                #open the image path with PIL
                img =Image.open(dataset.root_dir/person/img_name).convert("RGB")
                #convert to tensor and apply transformations
                img_tensor = dataset.transform(img).unsqueeze(0).to(device)
                #embed
                emb = model.embedding_net(img_tensor).cpu().numpy()[0]
                #add the embeddings for each image
                embeddings.append(emb)
                #add labels
                labels.append(person)
    embeddings_np = np.array(embeddings)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    emb_2d = tsne.fit_transform(embeddings_np)

    fig, ax = plt.subplots(figsize=(15,15))
    unique_labels = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0,1, len(unique_labels)))
    for i, person in enumerate(unique_labels):
        idx = [j for j,p in enumerate(labels) if p == person]
        ax.scatter(emb_2d[idx,0], emb_2d[idx,1], c=[colors[i]], label=person, alpha=0.8)
    ax.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize=8)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("t-SNE plot saved to %s", save_path)

    plt.close(fig)
    return fig

def _plot_distance_distribution(
        distances: np.ndarray,
        labels: np.ndarray,
        best_threshold:Optional[float]=None,
        save_path: Optional[str]=None
):
    """Histogram of distances for positive/negative pairs."""
    pos_dist = distances[labels==1]
    neg_dist = distances[labels==0]
    fig,ax = plt.subplots(figsize=(8,6))
    ax.hist(pos_dist, bins=30, alpha=0.5, label="same person")
    ax.hist(neg_dist, bins=30, alpha=0.5, label="Differen persons")
    if best_threshold is not None:
        ax.axvline(x=best_threshold, color="red", linestyle="--", label=f"best threshold = {best_threshold:.3f}")
    ax.set_xlabel("Euclidean Distance")
    ax.set_ylabel("Frequency")
    ax.set_title("Distance Distribiton")
    ax.legend()
    if save_path:
        fig.savefig(save_path)
        logger.info("Distance distribution plot saved to %s", save_path)
    plt.close(fig)
    return fig

def _plot_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        eer: float,
        save_path: Optional[str]=None
):
    """Plot ROC curve."""
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.4f})")
    ax.plot([0,1],[0,1], "k--")
    eer_idx = np.argmin(np.abs(fpr-(1-tpr)))
    ax.scatter(fpr[eer_idx], tpr[eer_idx], c="red", label=f"EER = {eer:.3f}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("reciever operating characteristics")
    ax.legend()

    if save_path:
        fig.savefig(save_path)
        logger.info("ROC curve saved to %s", save_path)
    plt.close(fig)
    return fig

##--main training function---
def train(config: Dict[str, Any])->torch.nn.Module:
    """
    Run the training loop with validation and tracking.

    Args:
        config: Full configuration dictionary.

    Returns:
        Trained SiameseNetwork model.
    """
    set_seed(config["split"]["seed"])
    device = get_device(config["training"]["device"])
    logger.info("using device %s", device)

    #datasets and loader - this uses a unified create_datasets; ignore test set
    train_dataset, val_dataset,_ = create_dataset(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size = config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"]
        
    )

    #model, loss, optimizer
    model = build_model(config).to(device)
    criterion = get_loss_function(config)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay",0.0)
    )

    #mlflow setup (optional)
    mlflow_active = False
    if config["logging"].get("mlflow_tracking_uri","http://localhost:8000"):
        try:
            import mlflow
            mlflow.set_tracking_uri(config["logging"]["mlflow_tracking_uri"])
            mlflow.set_experiment(config["logging"]["mlflow_experiment_name"])
            mlflow.start_run()
            mlflow_active = True
            logger.info("MLflow tracking enabled at %s", config["logging"]["mlflow_tracking_uri"])
            mlflow.log_params({
                "backbone": config["model"]["backbone"],
                "embedding_dim": config["model"]["embedding_dim"],
                "loss": config["loss"]["type"],
                "margin": config["loss"]["margin"],
                "batch_size": config["training"]["batch_size"],
                "learning_rate": config["training"]["learning_rate"],
                "epochs": config["training"]["epochs"],    
                })
            
            #checkpoint dir
            ckpt_dir = Path(config["training"]["checkpoint_dir"])
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            best_val_acc = 0.0

            epochs = config["training"]["epochs"]
            log_interval = config["training"]["log_interval"]

            for epoch in range(1, epochs+1):
                model.train()
                running_loss = 0.0
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
                for batch_idx, (img1, img2, labels) in enumerate(pbar):
                    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).squeeze()

                    optimizer.zero_grad()
                    _,_,distances = model(img1, img2)
                    loss = criterion(distances, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    if (batch_idx + 1) % log_interval == 0:
                        avg_loss = running_loss / log_interval
                        pbar.set_postfix(loss=avg_loss)
                        if mlflow_active:
                            mlflow.log_metric("train_loss", avg_loss, step=epoch * len(train_loader) + batch_idx)
                        running_loss = 0.0
                #validation accuracy
                val_acc = _validate_accuracy(model, val_loader, device, threshold=config["evaluation"]["threshold"])
                logger.info("Epoch %d - Validation accuracy: %.4f", epoch, val_acc)
                if mlflow_active:
                    mlflow.log_metric("val accuracy", val_acc, step=epoch)
                
                #t-SNE visualisation at specified epochs
                if epoch in config["validation"].get("t_sne_epochs",[]):
                    tsne_save = ckpt_dir / f"tsne_epoch_{epoch}.png"
                    _visualize_embeddings(
                        model,
                        val_dataset,
                        device,
                        num_persons=config["validation"].get("t_sne_samples",8),
                        images_per_person=config["validation"].get("t_sne_images_per_person",5),
                        title=f"t-SNE after epochs {epoch}",
                        save_path=str(tsne_save)
                    )
                    if mlflow_active:
                        mlflow.log_artifact(str(tsne_save))
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), ckpt_dir/"best_model.pth")
                    logger.info(f"Best model saved with accuracy %.4f", best_val_acc)
            logger.info("final evaluation on validation set...")
            eval_metrics = _final_evaluation(model, val_loader, device, config, ckpt_dir, mlflow_active)
            logger.info("Evaluation metrics: %s", {k: v for k, v in eval_metrics.items()
                                          if k not in ("fpr", "tpr", "confusion_matrix")})
            
            if mlflow_active:
                mlflow.end_run()
            return model
        except Exception as e:
            raise ProjectException(f" Exception found at {e}")
    
def _validate_accuracy(model, dataloader, device, threshold=1.0)->float:
    """Compute verification accuracy on a given dataloader."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img1, img2 ,labels in dataloader:
            img1,img2,labels = img1.to(device),img2.to(device), labels.to(device)
            _,_, distances = model(img1,img2)
            preds = (distances < threshold).float()
            correct += (preds == labels.squeeze()).sum().item()
            total += labels.size(0)

    return correct/total


def _final_evaluation(model, dataloader, device, config, artefact_dir, mlflow_active):
    """Full evaluation: ROC, best threshold, confusion matrix, plots."""
    model.eval()
    all_distances = []
    all_labels = []
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            _, _, distances = model(img1, img2)
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(labels.squeeze().cpu().numpy())

    distances = np.array(all_distances)
    labels = np.array(all_labels)

    metrics = compute_roc_metrics(distances, labels)

    artefact_dir = Path(artefact_dir) if isinstance(artefact_dir, str) else artefact_dir
    dist_plot_path = artefact_dir / "val_distance_distribution.png"
    _plot_distance_distribution(
        distances, labels, best_threshold=metrics["best_threshold"],
        save_path=str(dist_plot_path),
    )

    roc_plot_path = artefact_dir / "val_roc_curve.png"
    _plot_roc_curve(
        np.array(metrics["fpr"]), np.array(metrics["tpr"]),
        metrics["auc"], metrics["eer"],
        save_path=str(roc_plot_path),
    )

    if mlflow_active:
        import mlflow
        mlflow.log_metrics({
            "val_auc": metrics["auc"],
            "val_eer": metrics["eer"],
            "best_threshold": metrics["best_threshold"],
            "best_accuracy": metrics["best_accuracy"],
        })
        mlflow.log_artifact(str(dist_plot_path))
        mlflow.log_artifact(str(roc_plot_path))

    return metrics