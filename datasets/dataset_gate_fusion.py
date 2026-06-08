from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GateFusionDataset(Dataset):
    def __init__(self, expert_dir, mask_dir, indices):
        expert_dir = Path(expert_dir)
        self.stage1_prob = np.load(expert_dir / "stage1_prob.npy").astype(np.float32)
        self.c1_prob = np.load(expert_dir / "c1_prob.npy").astype(np.float32)
        self.roi_mask = np.load(expert_dir / "roi_mask.npy").astype(np.float32)
        self.mask_dir = Path(mask_dir)
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label_index = int(self.indices[idx])
        array_index = label_index - 1
        p1 = np.clip(self.stage1_prob[array_index], 0.0, 1.0)
        p2 = np.clip(self.c1_prob[array_index], 0.0, 1.0)
        roi = np.clip(self.roi_mask[array_index], 0.0, 1.0)
        uncertainty1 = 1.0 - np.abs(2.0 * p1 - 1.0)
        uncertainty2 = 1.0 - np.abs(2.0 * p2 - 1.0)
        features = np.stack([p1, uncertainty1, p2, uncertainty2, roi], axis=0).astype(np.float32)

        mask_path = self.mask_dir / f"{label_index:03d}.png"
        label = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        label = (label > 0).astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(label), label_index
