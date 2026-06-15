from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.dataset_stage1_input_ablation import normalize_nonzero


def build_boundary_band(mask, radius=2):
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=radius)
    eroded = cv2.erode(mask, kernel, iterations=radius)
    return (dilated != eroded).astype(np.float32)


class ReliabilityGateDataset(Dataset):
    input_items = ["intensity", "depth", "depth_valid", "local_depth_edge"]
    input_channels = 4

    def __init__(
        self,
        raw_items,
        label_dir,
        indices,
        boundary_radius=2,
        edge_threshold=0.10,
    ):
        self.intensity = np.load(Path(raw_items["intensity"]), mmap_mode="r")
        self.depth = np.load(Path(raw_items["depth"]), mmap_mode="r")
        self.local_edge = np.load(Path(raw_items["local_depth_edge"]), mmap_mode="r")
        self.label_dir = Path(label_dir)
        self.indices = [int(index) for index in indices]
        self.boundary_radius = int(boundary_radius)
        self.edge_threshold = float(edge_threshold)

    def __len__(self):
        return len(self.indices)

    def _label_path(self, label_index):
        direct_path = self.label_dir / f"{label_index}.png"
        if direct_path.exists():
            return direct_path
        padded_path = self.label_dir / f"{label_index:03d}.png"
        if padded_path.exists():
            return padded_path
        raise FileNotFoundError(f"Missing label for index {label_index}")

    def __getitem__(self, idx):
        label_index = self.indices[idx]
        array_index = label_index - 1

        intensity = normalize_nonzero(
            np.asarray(self.intensity[array_index], dtype=np.float32)
        )
        raw_depth = np.asarray(self.depth[array_index], dtype=np.float32)
        depth = normalize_nonzero(raw_depth)
        valid = (raw_depth > 0).astype(np.float32)
        edge = np.asarray(self.local_edge[array_index], dtype=np.float32)

        label = np.array(
            Image.open(self._label_path(label_index)).convert("L"),
            dtype=np.uint8,
        )
        label = (label > 0).astype(np.int64)
        boundary = build_boundary_band(label, radius=self.boundary_radius)

        strong_edge = edge >= self.edge_threshold
        gate_target = (strong_edge & (boundary > 0)).astype(np.float32)
        gate_valid = strong_edge.astype(np.float32)

        x = np.stack([intensity, depth, valid, edge], axis=0).astype(np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(label).long(),
            torch.from_numpy(gate_target),
            torch.from_numpy(gate_valid),
            torch.from_numpy(boundary),
        )
