"""
Utility helpers: config loading, device selection, dataset safety checks, logging.
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str) -> torch.device:
    pref = (preference or "auto").lower().strip()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {preference}")


def list_image_files(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    if not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix in exts])


def stem_matches(image_path: Path, mask_path: Path) -> bool:
    return image_path.stem == mask_path.stem


def pair_image_mask(
    images_dir: Path, masks_dir: Path
) -> tuple[list[tuple[Path, Path]], list[str]]:
    """
    Match image/mask pairs by filename stem. Returns (pairs, warnings).
    """
    warnings: list[str] = []
    imgs = {p.stem: p for p in list_image_files(images_dir)}
    mks = {p.stem: p for p in list_image_files(masks_dir)}
    stems = sorted(set(imgs.keys()) & set(mks.keys()))
    missing_masks = sorted(set(imgs.keys()) - set(mks.keys()))
    missing_imgs = sorted(set(mks.keys()) - set(imgs.keys()))
    if missing_masks:
        warnings.append(f"{len(missing_masks)} images have no mask (by stem): {missing_masks[:5]}...")
    if missing_imgs:
        warnings.append(f"{len(missing_imgs)} masks have no image (by stem): {missing_imgs[:5]}...")
    pairs = [(imgs[s], mks[s]) for s in stems]
    return pairs, warnings


def ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def append_csv_row(csv_path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def dataset_safety_report(
    cfg: dict[str, Any],
    sample_unique_mask_values: int = 5,
) -> None:
    """
    Print dataset existence, counts, and sample unique mask values before training.
    """
    root = Path(cfg["dataset"]["root_dir"])
    train_i = root / cfg["dataset"]["train_images"]
    train_m = root / cfg["dataset"]["train_masks"]
    val_i = root / cfg["dataset"]["val_images"]
    val_m = root / cfg["dataset"]["val_masks"]
    test_i = root / cfg["dataset"]["test_images"]

    print("\n=== Dataset safety check ===")
    for name, p in [
        ("Train images", train_i),
        ("Train masks", train_m),
        ("Val images", val_i),
        ("Val masks", val_m),
        ("Test images (not used for train)", test_i),
    ]:
        exists = p.is_dir()
        print(f"  {name}: {p} -> {'OK' if exists else 'MISSING'}")
        if not exists:
            print(f"    WARNING: path does not exist or is not a directory.")

    train_pairs, w1 = pair_image_mask(train_i, train_m)
    val_pairs, w2 = pair_image_mask(val_i, val_m)
    test_files = list_image_files(test_i)

    print(f"\n  Train pairs (matched by stem): {len(train_pairs)}")
    print(f"  Val pairs (matched by stem):   {len(val_pairs)}")
    print(f"  Test images:                   {len(test_files)}")

    for w in w1 + w2:
        print(f"  WARN: {w}")

    # Sample mask statistics (read first train mask if any)
    try:
        import cv2

        if train_pairs:
            sample_mask_path = train_pairs[0][1]
            m = cv2.imread(str(sample_mask_path), cv2.IMREAD_UNCHANGED)
            if m is None:
                print(f"  WARN: could not read sample mask: {sample_mask_path}")
            else:
                if m.ndim == 3 and m.shape[2] >= 3:
                    flat = m[..., :3].reshape(-1, 3)
                    # unique RGB rows (cap for print)
                    uniq = np.unique(flat, axis=0)
                    print(
                        f"  Sample mask (first train): RGB shape={m.shape}, "
                        f"unique RGB colors (capped display): {len(uniq)}"
                    )
                    for u in uniq[:sample_unique_mask_values]:
                        print(f"    RGB {u.tolist()}")
                else:
                    u = np.unique(m)
                    print(
                        f"  Sample mask (first train): grayscale shape={m.shape}, "
                        f"unique values (first {sample_unique_mask_values}): "
                        f"{u[:sample_unique_mask_values].tolist()}"
                    )
    except Exception as e:
        print(f"  WARN: mask sample read failed: {e}")

    print("=== End dataset check ===\n")
