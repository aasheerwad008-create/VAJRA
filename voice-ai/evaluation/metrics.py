"""
VAJRA Voice AI — Evaluation Metrics.

Computes comprehensive metrics for model evaluation:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
    - Equal Error Rate (EER) — for speaker verification
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MetricsReport:
    """Complete metrics report for a model."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    eer: float
    num_samples: int
    class_distribution: Dict[int, int]


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: Optional[np.ndarray] = None,
) -> MetricsReport:
    """
    Compute comprehensive evaluation metrics.

    Parameters
    ----------
    labels : ndarray, shape (N,)
        Ground truth labels.
    predictions : ndarray, shape (N,)
        Predicted class labels.
    scores : ndarray, shape (N,), optional
        Prediction scores/probabilities for the positive class.
        Required for ROC-AUC and EER.

    Returns
    -------
    MetricsReport
    """
    n = len(labels)
    if n == 0:
        return MetricsReport(
            accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
            roc_auc=0.0, eer=0.0, num_samples=0, class_distribution={},
        )

    # Accuracy
    accuracy = float(np.mean(labels == predictions))

    # For binary classification (positive class = 1)
    tp = int(np.sum((predictions == 1) & (labels == 1)))
    fp = int(np.sum((predictions == 1) & (labels == 0)))
    fn = int(np.sum((predictions == 0) & (labels == 1)))

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    # ROC-AUC
    roc_auc = 0.0
    if scores is not None:
        roc_auc = compute_roc_auc(labels, scores)

    # EER
    eer = 0.0
    if scores is not None:
        eer = compute_eer(labels, scores)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}

    return MetricsReport(
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        roc_auc=round(roc_auc, 4),
        eer=round(eer, 4),
        num_samples=n,
        class_distribution=dist,
    )


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute the Equal Error Rate (EER).

    Parameters
    ----------
    labels : ndarray, shape (N,)
        Binary ground truth (0 = negative, 1 = positive).
    scores : ndarray, shape (N,)
        Prediction scores (higher → more likely positive).

    Returns
    -------
    float : EER in [0, 1].
    """
    if len(labels) == 0:
        return 0.0

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    desc = np.argsort(-scores)
    labels_sorted = labels[desc]

    tp = 0
    prev_far, prev_frr = 0.0, 1.0

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        fp = (i + 1) - tp
        fn = n_pos - tp

        far = fp / n_neg
        frr = fn / n_pos

        if far >= frr:
            denom = (far - prev_far) + (prev_frr - frr)
            if denom > 0:
                alpha = (prev_frr - prev_far) / denom
                eer = prev_far + alpha * (far - prev_far)
            else:
                eer = far
            return float(np.clip(eer, 0.0, 1.0))

        prev_far, prev_frr = far, frr

    return float(np.clip(prev_far, 0.0, 1.0))


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Compute ROC-AUC using the trapezoidal rule.

    Parameters
    ----------
    labels : ndarray, shape (N,)
        Binary ground truth.
    scores : ndarray, shape (N,)
        Prediction scores.

    Returns
    -------
    float : AUC in [0, 1].
    """
    if len(labels) == 0:
        return 0.0

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    desc = np.argsort(-scores)
    labels_sorted = labels[desc]

    tpr_points: List[float] = [0.0]
    fpr_points: List[float] = [0.0]
    tp = 0
    fp = 0

    for i in range(len(labels_sorted)):
        if labels_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_points.append(tp / n_pos)
        fpr_points.append(fp / n_neg)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_points)):
        auc += (fpr_points[i] - fpr_points[i - 1]) * (tpr_points[i] + tpr_points[i - 1]) / 2

    return float(np.clip(auc, 0.0, 1.0))
