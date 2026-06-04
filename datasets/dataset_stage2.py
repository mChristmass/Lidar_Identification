import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class Stage2Dataset(Dataset):
    def __init__(self, root_dir, input_items):
        self.samples = []
        self.input_items = input_items

        for name in os.listdir(root_dir):
            sample_dir = os.path.join(root_dir, name)
            if not os.path.isdir(sample_dir):
                continue

            sample = {
                item_name: os.path.join(sample_dir, f"{item_name}.npy")
                for item_name in self.input_items
            }
            sample["gt"] = os.path.join(sample_dir, "gt_mask.npy")
            meta_path = os.path.join(sample_dir, "meta.json")
            sample["meta"] = meta_path if os.path.exists(meta_path) else ""
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        feature_list = []
        for input_name in self.input_items:
            input_path = item[input_name]
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Missing input file: {input_path}")

            data = np.load(input_path).astype(np.float32)

            if input_name in {"prob", "coarse_mask"}:
                data = np.clip(data, 0.0, 1.0)
            else:
                data = (data - data.mean()) / (data.std() + 1e-6)

            feature_list.append(data)

        x = np.stack(feature_list, axis=0)

        if os.path.exists(item["gt"]):
            gt = np.load(item["gt"]).astype(np.int64)
        else:
            gt = np.empty((0,), dtype=np.int64)

        if item["meta"] and os.path.exists(item["meta"]):
            with open(item["meta"], "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            if isinstance(meta, dict):
                meta.setdefault("roi_source", "oracle")
        else:
            meta = []

        return torch.tensor(x), torch.tensor(gt), meta
