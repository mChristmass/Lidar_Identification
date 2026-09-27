from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cmx_initial_transfer.official.dual_segformer import mit_b0, mit_b1, mit_b2
from experiments.cmx_initial_transfer.official.mlp_decoder import DecoderHead


BACKBONES = {
    "CMX_B0": (mit_b0, [32, 64, 160, 256]),
    "CMX_B1": (mit_b1, [64, 128, 320, 512]),
    "CMX_B2": (mit_b2, [64, 128, 320, 512]),
}


class CMXPaper(nn.Module):
    def __init__(self, variant: str, num_classes: int, image_size: int = 480,
                 decoder_dim: int = 256):
        super().__init__()
        if variant not in BACKBONES:
            raise ValueError(f"Unknown CMX variant {variant}")
        factory, channels = BACKBONES[variant]
        self.variant = variant
        self.backbone = factory(img_size=image_size, in_chans=1, norm_fuse=nn.BatchNorm2d)
        self.decode_head = DecoderHead(
            in_channels=channels, num_classes=num_classes, norm_layer=nn.BatchNorm2d,
            embed_dim=decoder_dim, align_corners=False,
        )
        for module in self.decode_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(inputs[:, :1], inputs[:, 1:2])
        logits = self.decode_head(features)
        logits = F.interpolate(logits, inputs.shape[-2:], mode="bilinear", align_corners=False)
        return {"logits": logits}


def _unwrap(value: Any) -> dict[str, torch.Tensor]:
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(value, dict) and isinstance(value.get(key), dict):
            value = value[key]
            break
    if not isinstance(value, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return {str(key): tensor for key, tensor in value.items() if isinstance(tensor, torch.Tensor)}


def load_imagenet(model: CMXPaper, checkpoint: str | Path) -> dict:
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        raw = torch.load(checkpoint, map_location="cpu")
    source = _unwrap(raw)
    target = model.backbone.state_dict()
    adapted = {}
    for source_key, tensor in source.items():
        key = source_key.removeprefix("module.").removeprefix("backbone.")
        destinations = [key]
        if "patch_embed" in key:
            destinations.append(key.replace("patch_embed", "extra_patch_embed", 1))
        elif "block" in key:
            destinations.append(key.replace("block", "extra_block", 1))
        elif "norm" in key:
            destinations.append(key.replace("norm", "extra_norm", 1))
        for destination in destinations:
            if destination not in target:
                continue
            value = tensor
            expected = target[destination]
            if value.shape != expected.shape:
                if value.ndim == 4 and value.shape[1] == 3 and expected.shape[1] == 1:
                    value = value.mean(1, keepdim=True)
                else:
                    continue
            adapted[destination] = value
    incompatible = model.backbone.load_state_dict(adapted, strict=False)
    encoder = [key for key in target if key.startswith(
        ("patch_embed", "extra_patch_embed", "block", "extra_block", "norm", "extra_norm"))]
    coverage = sum(key in adapted for key in encoder) / max(1, len(encoder))
    if coverage < 0.95:
        raise RuntimeError(f"Incompatible {model.variant} checkpoint: encoder coverage={coverage:.3f}")
    return {"coverage": coverage, "loaded": len(adapted),
            "missing": list(incompatible.missing_keys), "unexpected": list(incompatible.unexpected_keys)}
