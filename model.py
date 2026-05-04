"""
Build segmentation model from config using segmentation_models_pytorch.
"""

from __future__ import annotations

from typing import Any

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(cfg: dict[str, Any]) -> nn.Module:
    mcfg = cfg["model"]
    arch = mcfg["architecture"].lower().strip()
    encoder = mcfg["encoder"]
    num_classes = int(mcfg["num_classes"])
    encoder_weights = mcfg.get("encoder_weights") or ("imagenet" if mcfg.get("pretrained", True) else None)

    kwargs = dict(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
    )

    if arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(**kwargs)
    elif arch == "unet":
        model = smp.Unet(**kwargs)
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Use deeplabv3plus or unet.")

    return model
