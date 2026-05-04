"""
Run inference on testImages with the best checkpoint; save predicted masks and overlays.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestImageDataset
from model import build_model
from utils import ensure_dirs, get_device, load_config
from visualize import blend_overlay, mask_to_color


@torch.no_grad()
def run_test(cfg: dict, checkpoint_path: Path, device: torch.device) -> None:
    pred_dir = Path(cfg["outputs"]["prediction_dir"])
    overlay_dir = Path(cfg["outputs"].get("overlay_dir", "outputs/overlays"))
    ensure_dirs(pred_dir, overlay_dir)

    ds = TestImageDataset(cfg)
    if len(ds) == 0:
        raise RuntimeError("No test images found. Check dataset.test_images path.")

    loader = DataLoader(
        ds,
        batch_size=max(1, int(cfg["training"]["batch_size"]) // 2),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ckpt.get("config", cfg)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    num_classes = int(model_cfg["model"]["num_classes"])

    for batch in tqdm(loader, desc="test inference"):
        imgs = batch["image"].to(device, non_blocking=True)
        stems = batch["stem"]
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        for i, stem in enumerate(stems):
            pred = preds[i]
            out_mask_path = pred_dir / f"{stem}_pred.png"
            cv2.imwrite(str(out_mask_path), pred)

            # Overlay on original resolution: reload original image and resize pred
            root = Path(cfg["dataset"]["root_dir"])
            test_folder = root / cfg["dataset"]["test_images"]
            # find extension
            candidates = list(test_folder.glob(f"{stem}.*"))
            img_path = next((p for p in candidates if p.suffix.lower() in {".png", ".jpg", ".jpeg"}), None)
            if img_path is None:
                continue
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            pred_resized = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            color = mask_to_color(pred_resized.astype(np.int32), num_classes=num_classes)
            over = blend_overlay(bgr, color, alpha=0.45)
            cv2.imwrite(str(overlay_dir / f"{stem}_overlay.png"), over)

    print(f"Saved predictions to {pred_dir}")
    print(f"Saved overlays to {overlay_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg["training"].get("device", "auto"))
    run_test(cfg, Path(args.checkpoint), device)


if __name__ == "__main__":
    main()
