from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def normalize_nonzero(arr):
    arr = arr.astype(np.float32)
    valid = arr > 0
    if not valid.any():
        return np.zeros_like(arr, dtype=np.float32)
    min_value = float(arr[valid].min())
    max_value = float(arr[valid].max())
    scale = max(max_value - min_value, 1e-6)
    return np.where(valid, (arr - min_value) / scale + 0.1, 0.0).astype(np.float32)


def normalize_robust(arr, valid_mask=None, percentile=99.0):
    arr = arr.astype(np.float32)
    if valid_mask is None:
        valid_mask = arr != 0
    values = arr[valid_mask]
    if values.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    scale = max(float(np.percentile(np.abs(values), percentile)), 1e-6)
    normalized = np.clip(arr / scale, 0.0, 1.0)
    return np.where(valid_mask, normalized, 0.0).astype(np.float32)


def build_local_depth_edge(depth, erode_iterations=2):
    depth = depth.astype(np.float32)
    valid = (depth > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    safe_valid = cv2.erode(valid, kernel, iterations=erode_iterations).astype(bool)

    depth_filled = depth.copy()
    if valid.any():
        depth_filled[~valid.astype(bool)] = float(np.median(depth[valid.astype(bool)]))
    smoothed = cv2.GaussianBlur(depth_filled, (3, 3), 0)
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return normalize_robust(magnitude, valid_mask=safe_valid, percentile=99.0)


class Stage1InputAblationDataset(Dataset):
    EXPERIMENT_INPUTS = {
        "A": ["intensity", "depth", "depth_edge"],
        "B": ["intensity", "local_depth_edge"],
    }

    def __init__(self, raw_items, mask_dir, indices, experiment, edge_erode_iterations=2):
        if experiment not in self.EXPERIMENT_INPUTS:
            raise ValueError(f"Unsupported experiment: {experiment}")

        self.experiment = experiment
        self.input_items = self.EXPERIMENT_INPUTS[experiment]
        self.arrays = {
            name: np.load(Path(path)).astype(np.float32)
            for name, path in raw_items.items()
        }
        self.mask_dir = Path(mask_dir)
        self.indices = list(indices)
        self.edge_erode_iterations = edge_erode_iterations

    @property
    def input_channels(self):
        return len(self.input_items)

    def __len__(self):
        return len(self.indices)

    def _feature(self, name, array_index):
        if name == "intensity":
            return normalize_nonzero(self.arrays["intensity"][array_index])
        if name == "depth":
            return normalize_nonzero(self.arrays["depth"][array_index])
        if name == "depth_edge":
            edge = self.arrays["depth_edge"][array_index]
            return normalize_robust(edge, valid_mask=edge != 0, percentile=99.0)
        if name == "local_depth_edge":
            return build_local_depth_edge(
                self.arrays["depth"][array_index],
                erode_iterations=self.edge_erode_iterations,
            )
        raise KeyError(name)

    def __getitem__(self, idx):
        label_index = int(self.indices[idx])
        array_index = label_index - 1
        features = np.stack(
            [self._feature(name, array_index) for name in self.input_items],
            axis=0,
        ).astype(np.float32)

        mask_path = self.mask_dir / f"{label_index:03d}.png"
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask > 0).astype(np.int64)
        return torch.from_numpy(features), torch.from_numpy(mask).long()
