"""
Metrics for face verification evaluation: accuracy, ROC, AUC, EER,
and best threshold selection.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix

from src.logger import get_logger

logger = get_logger(__name__)

def compute_accuracy(distance:np.ndarray, labels: np.ndarray, threshold:float)->float:
    """
    Compute verification accuracy for a given distance threshold.

    Args:
        distance: 1D array of Euclidean distances.
        labels: 1D array of ground truth (1 for same, 0 for different).
        threshold: Distance below which a pair is predicted as "same".

    Returns:
        Accuracy (float between 0 and 1).
    """

    #convert distances to binary predictions: 1 if distance < threshold (same person)
    preds = (distance < threshold).astype(np.int32)
    acc = accuracy_score(labels, preds)
    return acc

def compute_best_threshold(distance:np.ndarray, labels:np.ndarray)->tuple[float, float]:
    """
    Find the threshold that maximizes verification accuracy.

    Args:
        distance: 1D array of Euclidean distances.
        labels: 1D array of ground truth (1 for same, 0 for different).

    Returns:
        Tuple of (best_threshold, best_accuracy).
    """
    best_acc = 0.0
    best_thresh = 0.0

    #sweep threshs from min ti max distance in 200 steps
    for thresh in np.linspace(distance.min(), distance.max(), 200):
        preds = (distance < thresh).astype(np.int32)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    logger.info("best threshold found: %.4f with accuracy:%.4f", best_thresh, best_acc)
    return best_thresh, best_acc

def compute_eer_from_roc(fpr: np.ndarray, tpr:np.ndarray)->float:
    """
    Compute Equal Error Rate (EER) from ROC curve data.
    EER is the point where FPR = 1 - TPR (false negative rate).

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.

    Returns:
        EER value (float).
    """
    fnr = 1 - tpr

    #find index where absolute difference between FPR and FNR is minimal
    eer_idx = np.nanargmin(np.abs(fpr-fnr))
    eer = fpr[eer_idx]
    return eer

def compute_roc_metrics(distance:np.ndarray, labels:np.ndarray)->float:
    """
    Compute ROC curve, AUC, EER, best threshold and confusion matrix.

    Args:
        distance: 1D array of Euclidean distances.
        labels: 1D array of ground truth (1 for same, 0 for different).

    Returns:
        Dictionary with keys: 'fpr', 'tpr', 'auc', 'eer', 'best_threshold',
        'best_accuracy', 'confusion_matrix'.
    """
    #for roc we treat smaller distances as higher confidence for 'same person'
    #so negate the distance to make a score where larger = more similar
    scores  = -distance
    fpr, tpr,_ = roc_curve(labels, scores)
    roc_auc = auc(fpr,tpr)
    eer = compute_eer_from_roc(fpr,tpr)

    best_thresh, best_acc = compute_best_threshold(distance, labels)

    #confusion matrix
    preds_best = (distance < best_thresh).astype(np.int32)
    cm = confusion_matrix(labels, preds_best)

    metrics={
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": roc_auc,
        "eer": eer,
        "best_threshold":best_thresh,
        "best_accuracy": best_acc,
        "confusion_matrix":cm.tolist()
    }
    logger.info("ROC AUC: %.4f, EER: %.4f", roc_auc, eer)
    return metrics

# compute_accuracy: simple accuracy for a given threshold.

# compute_best_threshold: grid search over 200 thresholds to find the one that maximizes accuracy; returns both best threshold and accuracy.

# compute_eer_from_roc: calculates the Equal Error Rate from the ROC curve by finding where FPR ≈ 1 − TPR.

# compute_roc_metrics: the main evaluation function that takes distances and labels, computes ROC, AUC, EER, best threshold, best accuracy, and confusion matrix. Returns a structured dict for easy logging or serialization.

# All functions operate on NumPy arrays (as collected during evaluation), keeping them independent of PyTorch.