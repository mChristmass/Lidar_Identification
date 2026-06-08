from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.dataset_stage1_input_ablation import build_local_depth_edge, normalize_nonzero


class DualBranchDataset(Dataset):
    input_items = ["intensity", "local_depth_edge"]
    input_channels = 2

    def __init__(self, raw_items, mask_dir, indices, edge_erode_iterations=2):
        self.intensity = np.load(Path(raw_items["intensity"])).astype(np.float32)
        self.depth = np.load(Path(raw_items["depth"])).astype(np.float32)
        self.mask_dir = Path(mask_dir)
        self.indices = list(indices)
        self.edge_erode_iterations = edge_erode_iterations

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label_index = int(self.indices[idx])
        array_index = label_index - 1

        intensity = normalize_nonzero(self.intensity[array_index])
        local_edge = build_local_depth_edge(
            self.depth[array_index],
            erode_iterations=self.edge_erode_iterations,
        )
        x = np.stack([intensity, local_edge], axis=0).astype(np.float32)

        mask_path = self.mask_dir / f"{label_index:03d}.png"
        y = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        y = (y > 0).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y).long()
