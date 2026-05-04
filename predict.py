"""
Predict segmentation for a single image or a folder of images (no labels required).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import build_model
from utils import ensure_dirs, get_device, load_config
from visualize import blend_overlay, logits_to_label, mask_to_color, show_panel


def load_checkpoint(path: Path, device: torch.device, fallback_cfg: dict) -> tuple[torch.nn.Module, dict]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config") or fallback_cfg
    if cfg is None:
        raise ValueError("Checkpoint missing 'config'; pass --config for architecture.")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def predict_image(
    model: torch.nn.Module,
    cfg: dict,
    image_path: Path,
    device: torch.device,
) -> np.ndarray:
    from dataset import build_transforms, image_hwc_to_tensor

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(image_path))
    h, w = image.shape[:2]
    transform = build_transforms(cfg, "val")
    aug = transform(image=image, mask=np.zeros((h, w), dtype=np.uint8))
    tensor = image_hwc_to_tensor(aug["image"]).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred = logits_to_label(logits)
    # Resize prediction back to original size
    pred_up = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return pred_up.astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Used if checkpoint has no embedded config.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--output_dir", type=str, default="outputs/predict_single")
    parser.add_argument("--show", action="store_true", help="Save matplotlib panel PNGs.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    device = get_device(base_cfg["training"].get("device", "auto"))
    ckpt_path = Path(args.checkpoint)
    model, cfg = load_checkpoint(ckpt_path, device, base_cfg)

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    ensure_dirs(out_dir)

    paths: list[Path]
    if inp.is_dir():
        paths = sorted(
            [p for p in inp.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        )
    else:
        paths = [inp]

    num_classes = int(cfg["model"]["num_classes"])
    names = cfg.get("class_names")

    for p in tqdm(paths, desc="predict"):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        pred = predict_image(model, cfg, p, device)
        stem = p.stem
        cv2.imwrite(str(out_dir / f"{stem}_pred.png"), pred.astype(np.uint8))
        color = mask_to_color(pred, num_classes=num_classes)
        over = blend_overlay(bgr, color, alpha=0.45)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), over)
        if args.show:
            show_panel(
                bgr,
                gt_mask=None,
                pred_mask=pred,
                class_names=names,
                save_path=out_dir / f"{stem}_panel.png",
                title=stem,
            )

    print(f"Done. Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
