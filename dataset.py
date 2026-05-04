"""
PyTorch Dataset for offroad segmentation with RGB or grayscale masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import list_image_files, pair_image_mask


def image_hwc_to_tensor(image: np.ndarray) -> torch.Tensor:
    """Float HWC (after Normalize) -> float CHW tensor."""
    return torch.from_numpy(np.transpose(image, (2, 0, 1))).float()


class MaskEncoder:
    """
    Converts raw mask arrays to class index maps (H, W) with values in [0, num_classes-1]
    or ignore_index (255) for unknown labels.
    """

    def __init__(self, cfg: dict[str, Any]):
        me = cfg.get("mask_encoding", {})
        self.num_classes = int(cfg["model"]["num_classes"])
        self.ignore_index = 255
        palette = me.get("rgb_palette") or []
        self.rgb_to_class: dict[tuple[int, int, int], int] = {}
        for cid, rgb in enumerate(palette):
            if rgb is None or len(rgb) < 3:
                continue
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            # Store both RGB and BGR so cv2.imread masks match regardless of channel order
            self.rgb_to_class[(r, g, b)] = cid
            self.rgb_to_class[(b, g, r)] = cid
        # Grayscale explicit map: str/int keys from yaml -> int class
        gmap = me.get("grayscale_value_to_class") or {}
        self.gray_to_class: dict[int, int] = {}
        for k, v in gmap.items():
            self.gray_to_class[int(k)] = int(v)

    def encode(self, mask: np.ndarray) -> np.ndarray:
        """
        mask: HxW or HxWxC uint8/float from cv2
        Returns HxW uint8 with class ids 0..C-1 or ignore_index.
        """
        if mask.ndim == 2 or (mask.ndim == 3 and mask.shape[2] == 1):
            gray = mask if mask.ndim == 2 else mask[..., 0]
            return self._encode_grayscale(gray)

        # BGR from cv2 -> interpret as RGB order for palette matching
        if mask.ndim == 3 and mask.shape[2] >= 3:
            rgb = mask[..., :3]
            # If all channels equal within tolerance, treat as grayscale
            if np.allclose(rgb[..., 0], rgb[..., 1]) and np.allclose(rgb[..., 0], rgb[..., 2]):
                return self._encode_grayscale(rgb[..., 0])
            return self._encode_rgb(rgb)
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    def _encode_grayscale(self, gray: np.ndarray) -> np.ndarray:
        g = gray.astype(np.int64)
        out = np.full(g.shape, self.ignore_index, dtype=np.uint8)
        if self.gray_to_class:
            for raw, cid in self.gray_to_class.items():
                out[g == raw] = cid
            return out.astype(np.uint8)
        # Identity for 0..num_classes-1
        valid = (g >= 0) & (g < self.num_classes)
        out[valid] = g[valid].astype(np.uint8)
        return out.astype(np.uint8)

    def _encode_rgb(self, bgr: np.ndarray) -> np.ndarray:
        bgr_u8 = bgr.astype(np.uint8)
        h, w, _ = bgr_u8.shape
        flat = bgr_u8.reshape(-1, 3)
        out_flat = np.full(flat.shape[0], self.ignore_index, dtype=np.uint8)
        if not self.rgb_to_class:
            return out_flat.reshape(h, w)
        for (t0, t1, t2), cid in self.rgb_to_class.items():
            match = (flat[:, 0] == t0) & (flat[:, 1] == t1) & (flat[:, 2] == t2)
            out_flat[match] = np.uint8(cid)
        return out_flat.reshape(h, w)


def build_transforms(
    cfg: dict[str, Any], split: str
) -> A.Compose:
    """
    split: 'train' | 'val'
    """
    size = int(cfg["training"]["image_size"])
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if split == "train":
        tlist: list[Any] = []
        tlist.append(A.HorizontalFlip(p=0.5))
        if cfg["training"].get("augment_brightness_contrast", True):
            tlist.append(A.RandomBrightnessContrast(p=0.5))
        if cfg["training"].get("augment_random_scale_crop", True):
            smin = float(cfg["training"].get("min_scale", 0.8))
            smax = float(cfg["training"].get("max_scale", 1.2))
            # Always output (size, size); random geometry before normalize
            tlist.append(
                A.Resize(height=size, width=size)
            )
            tlist.append(
                A.Resize(
                    height=size,
                    width=size,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                )
            )
        else:
            tlist.append(
                A.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)
            )
        if cfg["training"].get("augment_gaussian_noise", False):
            tlist.append(A.GaussNoise(p=0.2))
        tlist.append(A.Normalize(mean=mean, std=std))
        return A.Compose(tlist, additional_targets={"mask": "mask"})

    # val / test: deterministic resize
    return A.Compose(
        [
            A.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=mean, std=std),
        ],
        additional_targets={"mask": "mask"},
    )


class OffroadSegmentationDataset(Dataset):
    def __init__(
        self,
        cfg: dict[str, Any],
        split: str,
    ):
        assert split in {"train", "val"}
        self.cfg = cfg
        self.split = split
        root = Path(cfg["dataset"]["root_dir"])
        if split == "train":
            img_dir = root / cfg["dataset"]["train_images"]
            msk_dir = root / cfg["dataset"]["train_masks"]
        else:
            img_dir = root / cfg["dataset"]["val_images"]
            msk_dir = root / cfg["dataset"]["val_masks"]

        self.pairs, self.pair_warnings = pair_image_mask(img_dir, msk_dir)
        self.encoder = MaskEncoder(cfg)
        self.transform = build_transforms(cfg, split)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ip, mp = self.pairs[idx]
        image = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {ip}")
        mask_raw = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if mask_raw is None:
            raise FileNotFoundError(f"Could not read mask: {mp}")
        mask_idx = self.encoder.encode(mask_raw)
        augmented = self.transform(image=image, mask=mask_idx)
        img_t = image_hwc_to_tensor(augmented["image"])
        msk_np = augmented["mask"]
        if msk_np.ndim == 3:
            msk_np = msk_np[..., 0]
        msk = torch.from_numpy(msk_np.astype(np.int64)).long()
        return {"image": img_t, "mask": msk, "stem": ip.stem}


class TestImageDataset(Dataset):
    """Images only for test folder."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        root = Path(cfg["dataset"]["root_dir"])
        self.paths = list_image_files(root / cfg["dataset"]["test_images"])
        self.transform = build_transforms(cfg, "val")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        p = self.paths[idx]
        image = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read: {p}")
        h, w = image.shape[:2]
        augmented = self.transform(image=image, mask=np.zeros((h, w), dtype=np.uint8))
        img_t = image_hwc_to_tensor(augmented["image"])
        return {"image": img_t, "stem": p.stem, "path": str(p)}
