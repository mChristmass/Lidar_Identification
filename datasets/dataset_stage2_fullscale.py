import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def normalize_nonzero(arr):
    arr = arr.astype(np.float32)
    valid = arr > 0
    if valid.any():
        min_value = arr[valid].min()
        max_value = arr.max()
        denom = max(max_value - min_value, 1e-6)
        return np.where(valid, (arr - min_value) / denom + 0.1, 0.0).astype(np.float32)
    return arr


def normalize_standard(arr):
    arr = arr.astype(np.float32)
    return ((arr - arr.mean()) / (arr.std() + 1e-6)).astype(np.float32)


class Stage2FullScaleDataset(Dataset):
    def __init__(
        self,
        raw_items,
        prior_dir,
        mask_dir,
        indices,
        input_items,
    ):
        self.raw_items = {name: np.load(os.fspath(path)).astype(np.float32) for name, path in raw_items.items()}
        self.stage1_logits = np.load(os.fspath(Path(prior_dir) / "logits.npy")).astype(np.float32)
        self.priors = {
            "prob": np.load(os.fspath(Path(prior_dir) / "prob.npy")).astype(np.float32),
            "coarse_mask": np.load(os.fspath(Path(prior_dir) / "coarse_mask.npy")).astype(np.float32),
            "roi_mask": np.load(os.fspath(Path(prior_dir) / "roi_mask.npy")).astype(np.float32),
        }
        self.mask_dir = Path(mask_dir)
        self.indices = list(indices)
        self.input_items = list(input_items)

    def __len__(self):
        return len(self.indices)

    def _load_mask(self, label_index):
        mask_path = self.mask_dir / f"{label_index:03d}.png"
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask, dtype=np.uint8)
        if mask.max() > 1:
            mask[mask > 1] = 1
        return mask.astype(np.int64)

    def _load_input_item(self, item_name, array_index):
        if item_name in self.raw_items:
            arr = self.raw_items[item_name][array_index]
            if item_name == "intensity":
                return normalize_nonzero(arr)
            return normalize_standard(arr)
        if item_name in self.priors:
            return np.clip(self.priors[item_name][array_index], 0.0, 1.0).astype(np.float32)
        raise KeyError(f"Unsupported input item: {item_name}")

    def __getitem__(self, idx):
        label_index = int(self.indices[idx])
        array_index = label_index - 1

        feature_list = [
            self._load_input_item(item_name, array_index)
            for item_name in self.input_items
        ]
        x = np.stack(feature_list, axis=0).astype(np.float32)
        y = self._load_mask(label_index)
        base_logits = self.stage1_logits[array_index].astype(np.float32)
        roi_mask = self.priors["roi_mask"][array_index].astype(np.float32)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y).long(),
            torch.from_numpy(base_logits),
            torch.from_numpy(roi_mask),
            label_index,
        )
