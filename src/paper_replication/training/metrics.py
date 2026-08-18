"""Macro-averaged precision/recall/F1, matching paper Table I's reported metrics."""

from __future__ import annotations

import torch


def confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """(num_classes, num_classes) confusion matrix; rows=true, cols=predicted."""
    indices = y_true.long() * num_classes + y_pred.long()
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def precision_recall_f1(confusion: torch.Tensor) -> dict[str, float]:
    """Macro-averaged precision/recall/F1/accuracy from a confusion matrix.

    Classes with no predictions (precision) or no true instances (recall)
    get 0 for that metric rather than dividing by zero.
    """
    true_positive = confusion.diagonal().float()
    predicted_positive = confusion.sum(dim=0).float()
    actual_positive = confusion.sum(dim=1).float()

    precision = torch.where(
        predicted_positive > 0,
        true_positive / predicted_positive,
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        actual_positive > 0,
        true_positive / actual_positive,
        torch.zeros_like(true_positive),
    )
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom))

    total = confusion.sum().item()
    accuracy = (true_positive.sum() / total).item() if total > 0 else 0.0

    return {
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "accuracy": accuracy,
    }
