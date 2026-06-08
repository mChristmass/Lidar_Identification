from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.dataset_stage1_input_ablation import build_local_depth_edge, normalize_nonzero


class Stage2LocalEdgeDataset(Dataset):
    EXPERIMENT_INPUTS = {
        "C1": ["intensity", "local_depth_edge"],
        "C2": ["intensity", "local_depth_edge", "prob", "roi_mask"],
        "C3": ["intensity", "local_depth_edge", "local_depth_edge_roi", "prob", "roi_mask"],
    }

    def __init__(
        self,
        raw_items,
        mask_dir,
        indices,
        experiment,
        prior_dir=None,
        edge_erode_iterations=2,
    ):
        if experiment not in self.EXPERIMENT_INPUTS:
            raise ValueError(f"Unsupported experiment: {experiment}")

        self.experiment = experiment
        self.input_items = self.EXPERIMENT_INPUTS[experiment]
        self.intensity = np.load(Path(raw_items["intensity"])).astype(np.float32)
        self.depth = np.load(Path(raw_items["depth"])).astype(np.float32)
        self.mask_dir = Path(mask_dir)
        self.indices = list(indices)
        self.edge_erode_iterations = edge_erode_iterations
        self.prob = None
        self.roi_mask = None

        if experiment in {"C2", "C3"}:
            if prior_dir is None:
                raise ValueError(f"{experiment} requires Stage1 priors.")
            prior_dir = Path(prior_dir)
            self.prob = np.load(prior_dir / "prob.npy").astype(np.float32)
            self.roi_mask = np.load(prior_dir / "roi_mask.npy").astype(np.float32)

    @property
    def input_channels(self):
        return len(self.input_items)

    @property
    def uses_stage1_priors(self):
        return self.experiment in {"C2", "C3"}

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

        features = {
            "intensity": intensity,
            "local_depth_edge": local_edge,
        }
        roi_mask = np.ones_like(local_edge, dtype=np.float32)

        if self.uses_stage1_priors:
            prob = np.clip(self.prob[array_index], 0.0, 1.0)
            roi_mask = np.clip(self.roi_mask[array_index], 0.0, 1.0)
            features["prob"] = prob
            features["roi_mask"] = roi_mask
            features["local_depth_edge_roi"] = local_edge * roi_mask

        x = np.stack([features[name] for name in self.input_items], axis=0).astype(np.float32)
        mask_path = self.mask_dir / f"{label_index:03d}.png"
        y = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        y = (y > 0).astype(np.int64)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y).long(),
            torch.from_numpy(roi_mask.astype(np.float32)),
        )
