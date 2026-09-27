from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

from .corruptions import apply_corruption, load_calibration


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def robust_depth(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        low, high = np.percentile(depth[valid], [1, 99])
        if high > low:
            output[valid] = np.clip((depth[valid] - low) / (high - low), 0, 1)
    return output


class NYUv2PaperDataset(Dataset):
    def __init__(self, root: str | Path, manifest: str | Path, calibration: str | Path,
                 *, corruption: str = "clean", severity: str = "medium", split: str,
                 image_size: tuple[int, int] = (480, 640), seed: int = 42,
                 training: bool = False):
        self.root = Path(root)
        self.rows = read_manifest(Path(manifest))
        self.calibration = load_calibration(calibration)
        self.corruption = corruption
        self.severity = severity
        self.split = split
        self.image_size = tuple(image_size)
        self.seed = int(seed)
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        intensity = np.asarray(Image.open(self.root / row["intensity"]), dtype=np.float32)
        depth = np.asarray(Image.open(self.root / row["depth"]), dtype=np.float32)
        label = np.asarray(Image.open(self.root / row["label"]), dtype=np.int64)
        intensity, depth, valid = apply_corruption(
            intensity, depth, self.corruption, self.severity, self.calibration,
            global_seed=self.seed, split=self.split, sample_id=row["sample_id"],
        )
        depth = robust_depth(depth, valid > 0)
        inputs = torch.from_numpy(np.stack((intensity, depth, valid), axis=0).copy())
        target = torch.from_numpy(label.copy())
        inputs = F.interpolate(inputs[None], self.image_size, mode="bilinear", align_corners=False)[0]
        target = F.interpolate(target[None, None].float(), self.image_size, mode="nearest")[0, 0].long()
        if self.training:
            if torch.rand(()) < 0.5:
                inputs = inputs.flip(-1)
                target = target.flip(-1)
        return {"inputs": inputs, "labels": target, "sample_id": row["sample_id"]}


class PrivatePaperDataset(Dataset):
    """The frozen private split. Indices are 1-based on disk by design."""

    def __init__(self, merged_dir: str | Path, indices: str | Path, *,
                 training: bool = False, seed: int = 42):
        self.root = Path(merged_dir)
        self.ids = np.load(indices).astype(np.int64)
        self.intensity = np.load(self.root / "intensity.npy", mmap_mode="r")
        self.depth = np.load(self.root / "depth.npy", mmap_mode="r")
        self.training = bool(training)
        self.seed = int(seed)
        if self.intensity.shape != self.depth.shape:
            raise ValueError("Private intensity/depth arrays are not aligned")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict:
        frame = int(self.ids[index])
        intensity = np.asarray(self.intensity[frame - 1], dtype=np.float32)
        depth = np.asarray(self.depth[frame - 1], dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        intensity = (intensity - np.nanpercentile(intensity, 1)) / max(
            float(np.nanpercentile(intensity, 99) - np.nanpercentile(intensity, 1)), 1e-6
        )
        intensity = np.clip(np.nan_to_num(intensity), 0, 1)
        depth = robust_depth(depth, valid)
        label = np.asarray(Image.open(self.root / "label" / f"{frame}.png"), dtype=np.int64)
        inputs = torch.from_numpy(np.stack((intensity, depth, valid.astype(np.float32))).copy())
        target = torch.from_numpy(label.copy())
        if self.training:
            if torch.rand(()) < 0.5:
                inputs = inputs.flip(-1)
                target = target.flip(-1)
        return {"inputs": inputs, "labels": target, "sample_id": str(frame)}
