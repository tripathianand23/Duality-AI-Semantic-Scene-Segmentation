"""
Semantic segmentation metrics: pixel accuracy, per-class IoU, mean IoU.
"""

from __future__ import annotations

import torch


def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    preds, targets: (N, H, W) int64
    Returns (C, C) counts: target=i, pred=j
    """
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    if ignore_index is not None:
        keep = targets != ignore_index
        preds = preds[keep]
        targets = targets[keep]
    mask = (targets >= 0) & (targets < num_classes) & (preds >= 0) & (preds < num_classes)
    preds = preds[mask]
    targets = targets[mask]
    idx = num_classes * targets + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).to(torch.float32)

def metrics_from_confusion(cm: torch.Tensor) -> dict[str, torch.Tensor | float]:
    """
    cm: (C, C) with rows = true class, cols = pred class
    """
    num_classes = cm.shape[0]
    diag = torch.diag(cm)
    sum_row = cm.sum(dim=1)
    sum_col = cm.sum(dim=0)
    union = sum_row + sum_col - diag
    iou_per_class = torch.where(union > 0, diag / union, torch.zeros_like(diag))
    miou = iou_per_class.mean().item()
    correct = diag.sum()
    total = cm.sum()
    pix_acc = (correct / total).item() if total > 0 else 0.0
    return {
        "iou_per_class": iou_per_class,
        "miou": miou,
        "pixel_accuracy": pix_acc,
    }


@torch.no_grad()
def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, float | list[float]]:
    """
    logits: (N, C, H, W), targets: (N, H, W)
    """
    preds = logits.argmax(dim=1)
    cm = confusion_matrix(preds, targets, num_classes, ignore_index=ignore_index)
    out = metrics_from_confusion(cm)
    return {
        "pixel_accuracy": float(out["pixel_accuracy"]),
        "miou": float(out["miou"]),
        "iou_per_class": out["iou_per_class"].cpu().tolist(),
    }


def accumulate_confusion(
    cm_acc: torch.Tensor | None,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    preds = logits.argmax(dim=1)
    cm = confusion_matrix(preds, targets, num_classes, ignore_index=ignore_index)
    if cm_acc is None:
        return cm
    return cm_acc + cm
