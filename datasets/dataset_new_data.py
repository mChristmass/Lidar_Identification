from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.dataset_stage1_input_ablation import normalize_nonzero


class NewDataSegmentationDataset(Dataset):
    EXPERIMENT_INPUTS = {
        "I0": ["intensity"],
        "ID": ["intensity", "depth"],
        "IDE": ["intensity", "depth", "local_depth_edge"],
        "C1": ["intensity", "local_depth_edge"],
        "C1L": ["intensity", "local_depth_edge"],
        "D1": ["intensity", "local_depth_edge"],
        "D3": ["intensity", "local_depth_edge"],
    }

    def __init__(self, raw_items, label_dir, indices, experiment):
        if experiment not in self.EXPERIMENT_INPUTS:
            raise ValueError(f"Unsupported experiment: {experiment}")
        self.experiment = experiment
        self.input_items = self.EXPERIMENT_INPUTS[experiment]
        self.label_dir = Path(label_dir)
        self.indices = [int(index) for index in indices]
        self.arrays = {
            name: np.load(Path(raw_items[name]), mmap_mode="r")
            for name in self.input_items
        }

    @property
    def input_channels(self):
        return len(self.input_items)

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
        features = []
        for name in self.input_items:
            array = np.asarray(self.arrays[name][array_index], dtype=np.float32)
            if name in {"intensity", "depth"}:
                array = normalize_nonzero(array)
            features.append(array)

        x = np.stack(features, axis=0).astype(np.float32)
        y = np.array(Image.open(self._label_path(label_index)).convert("L"), dtype=np.uint8)
        y = (y > 0).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y).long()
