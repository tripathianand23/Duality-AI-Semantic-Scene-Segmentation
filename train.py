"""
Train semantic segmentation model; logs CSV, saves best checkpoint by val mIoU, plots curves.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pyexpat import model
from random import sample

import matplotlib.pyplot as plt
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OffroadSegmentationDataset
from metrics import accumulate_confusion, metrics_from_confusion
from model import build_model
from utils import (
    append_csv_row,
    dataset_safety_report,
    ensure_dirs,
    get_device,
    load_config,
    set_seed,
)

IGNORE_INDEX = 255


def build_losses(cfg: dict, num_classes: int) -> tuple[nn.Module, nn.Module | None, float]:
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    dice = None
    w = float(cfg["training"].get("dice_weight", 0.5))
    if cfg["training"].get("use_dice_loss", False):
        dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=IGNORE_INDEX)
    return ce, dice, w


class CombinedLoss(nn.Module):
    def __init__(self, ce: nn.Module, dice: nn.Module | None, dice_w: float):
        super().__init__()
        self.ce = ce
        self.dice = dice
        self.dice_w = dice_w

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, target)
        if self.dice is not None:
            loss = loss + self.dice_w * self.dice(logits, target)
        return loss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: GradScaler | None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
        bs = imgs.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    use_amp: bool,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    cm_acc = None
    for batch in tqdm(loader, desc="val", leave=False):
        imgs = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, masks)
        else:
            logits = model(imgs)
            loss = criterion(logits, masks)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs
        cm_acc = accumulate_confusion(cm_acc, logits, masks, num_classes, ignore_index=IGNORE_INDEX)
    assert cm_acc is not None
    m = metrics_from_confusion(cm_acc)
    return total_loss / max(n, 1), float(m["miou"]), float(m["pixel_accuracy"])


def plot_curves(csv_path: Path, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "train_loss" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curves.png", dpi=150)
        plt.close()
    if "val_miou" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["val_miou"], label="val_mIoU")
        plt.xlabel("epoch")
        plt.ylabel("mIoU")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "miou_curve.png", dpi=150)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg["training"].get("seed", 42)))
    device = get_device(cfg["training"].get("device", "auto"))
    num_classes = int(cfg["model"]["num_classes"])

    ckpt_dir = Path(cfg["outputs"]["checkpoint_dir"])
    runs_dir = Path(cfg["outputs"]["runs_dir"])
    ensure_dirs(ckpt_dir, runs_dir, Path(cfg["outputs"]["prediction_dir"]).parent)

    dataset_safety_report(cfg)

    train_ds = OffroadSegmentationDataset(cfg, "train")
    val_ds = OffroadSegmentationDataset(cfg, "val")
    for w in train_ds.pair_warnings + val_ds.pair_warnings:
        print(f"Dataset pairing note: {w}")

    if len(train_ds) == 0:
        raise RuntimeError("No training pairs found. Check dataset paths and filename stems.")
    if len(val_ds) == 0:
        # --- Sanity check: catch NaN/bad labels before training ---
        print("Running data sanity check...")
        sample = train_ds[0]
        img, mask = sample["image"], sample["mask"]
        print(f"  Image dtype={img.dtype}, min={img.min():.3f}, max={img.max():.3f}, has_nan={torch.isnan(img).any()}")
        print(f"  Mask  dtype={mask.dtype}, min={mask.min()}, max={mask.max()}, unique={mask.unique()}")
        assert not torch.isnan(img).any(), "NaN found in image!"
        assert mask.dtype == torch.int64, f"Mask dtype must be int64, got {mask.dtype}"
        raise RuntimeError("No validation pairs found.")
        

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg).to(device)
    ce, dice, dice_w = build_losses(cfg, num_classes)
    criterion = CombinedLoss(ce, dice, dice_w).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )

    epochs = int(cfg["training"]["epochs"])
    sched_name = str(cfg["training"].get("scheduler", "cosine")).lower()
    if sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg["training"].get("plateau_factor", 0.5)),
            patience=int(cfg["training"].get("plateau_patience", 3)),
            min_lr=float(cfg["training"].get("plateau_min_lr", 1e-6)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    use_amp = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    csv_path = Path(cfg["outputs"].get("log_csv", "runs/training_log.csv"))
    fieldnames = ["epoch", "lr", "train_loss", "val_loss", "val_miou", "val_pixel_acc"]

    best_miou = -1.0
    best_path = ckpt_dir / "best_model.pth"

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp, scaler)
        val_loss, val_miou, val_pix = validate(
            model, val_loader, criterion, device, num_classes, use_amp and device.type == "cuda"
        )
        lr = optimizer.param_groups[0]["lr"]
        if sched_name == "plateau":
            scheduler.step(val_miou)
        else:
            scheduler.step()

        print(
            f"Epoch {epoch}/{epochs}  lr={lr:.2e}  train_loss={tr_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  val_pix_acc={val_pix:.4f}"
        )
        append_csv_row(
            csv_path,
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "val_pixel_acc": val_pix,
            },
            fieldnames,
        )

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_miou": val_miou,
                    "config": cfg,
                },
                best_path,
            )
            print(f"  Saved new best model (mIoU={val_miou:.4f}) -> {best_path}")

    plot_curves(csv_path, runs_dir)
    print(f"Training complete. Best mIoU={best_miou:.4f}. Plots saved under {runs_dir}")


if __name__ == "__main__":
    main()
