"""
Visualization helpers: colorize masks and build overlay panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


# BGR-friendly distinct colors for 10 classes (for cv2 drawing)
COLOR_MAP_BGR: np.ndarray = np.array(
    [
        [0, 100, 0],
        [0, 200, 0],
        [0, 200, 150],
        [0, 150, 100],
        [43, 90, 139],
        [255, 0, 255],
        [33, 67, 101],
        [128, 128, 128],
        [140, 180, 210],
        [235, 206, 135],
    ],
    dtype=np.uint8,
)


def mask_to_color(mask: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    mask: HxW int/bool with class ids 0..num_classes-1
    Returns HxWx3 BGR uint8
    """
    h, w = mask.shape[:2]
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(num_classes, len(COLOR_MAP_BGR))):
        color[mask == c] = COLOR_MAP_BGR[c]
    return color


def blend_overlay(image_bgr: np.ndarray, mask_color_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = (image_bgr.astype(np.float32) * (1 - alpha) + mask_color_bgr.astype(np.float32) * alpha).astype(np.uint8)
    return out


def logits_to_label(logits: torch.Tensor) -> np.ndarray:
    """logits (1,C,H,W) or (C,H,W) -> HxW numpy int"""
    if logits.ndim == 4:
        logits = logits[0]
    pred = logits.argmax(dim=0).detach().cpu().numpy().astype(np.int32)
    return pred


def show_panel(
    image_bgr: np.ndarray,
    gt_mask: np.ndarray | None,
    pred_mask: np.ndarray,
    class_names: Sequence[str] | None = None,
    save_path: Path | None = None,
    title: str | None = None,
) -> None:
    """
    image_bgr: HxWx3
    gt_mask: HxW or None
    pred_mask: HxW
    """
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred_color = cv2.cvtColor(mask_to_color(pred_mask), cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(blend_overlay(image_bgr, mask_to_color(pred_mask)), cv2.COLOR_BGR2RGB)

    cols = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if title:
        fig.suptitle(title)

    idx = 0
    axes[idx].imshow(rgb)
    axes[idx].set_title("Image")
    axes[idx].axis("off")
    idx += 1

    if gt_mask is not None:
        gt_color = cv2.cvtColor(mask_to_color(gt_mask), cv2.COLOR_BGR2RGB)
        axes[idx].imshow(gt_color)
        axes[idx].set_title("Ground truth")
        axes[idx].axis("off")
        idx += 1

    axes[idx].imshow(pred_color)
    axes[idx].set_title("Prediction")
    axes[idx].axis("off")
    idx += 1

    axes[idx].imshow(overlay_rgb)
    axes[idx].set_title("Overlay")
    axes[idx].axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
