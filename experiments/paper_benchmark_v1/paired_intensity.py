"""Paired clean intensity targets for the frozen NYUv2 training split."""
import numpy as np
import torch
from PIL import Image

from .corruptions import _normalize01, stable_seed
from .corruptions_v2 import corrupt
from .data import robust_depth
from .train_mixed_control import TrainingData, training_condition


class PairedIntensityData(TrainingData):
    def __init__(self, root, seed):
        super().__init__(root, 'mixed', seed)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.asarray(Image.open(self.root / row['intensity']), np.float32)
        depth = np.asarray(Image.open(self.root / row['depth']), np.float32)
        label = np.asarray(Image.open(self.root / row['label']), np.int64)
        clean = _normalize01(image).astype(np.float32)
        family, severity = training_condition(self.seed, self.epoch,
                                              row['sample_id'], 'mixed')
        intensity, depth, valid, _ = corrupt(
            image, depth, family, severity,
            f'train/{self.epoch}/{row["sample_id"]}', self.seed)
        inputs = np.stack((intensity, robust_depth(depth, valid),
                           valid.astype(np.float32)))
        rng = np.random.default_rng(stable_seed(
            self.seed, 'train', row['sample_id'], 'flip', str(self.epoch)))
        if rng.random() < .5:
            inputs, label, clean = inputs[..., ::-1], label[:, ::-1], clean[:, ::-1]
        clean_view = inputs.copy()
        clean_view[0] = clean
        noise_active = family in ('photon_proxy', 'joint')
        return (torch.from_numpy(inputs.copy()), torch.from_numpy(label.copy()),
                torch.from_numpy(clean_view.copy()), torch.tensor(noise_active))
