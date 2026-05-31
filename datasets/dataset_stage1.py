import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from engine.stage1.normalization_stage1 import*


class LidarSegDataset(Dataset):
    def __init__(self, npy_path, mask_dir, index_list):
        """
        npy_path   : images.npy 路径
        mask_dir   : 存放 png mask 的目录
        index_list : List[int]，指定使用哪些样本
        """
        self.images = data_normalization(npy_path)  # [N, 2, 128, 128]
        self.mask_dir = mask_dir
        self.indices = index_list

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]

        # ---------- 读取图像 ----------
        image = self.images[img_idx-1]              # [2, 128, 128]
        image = torch.from_numpy(image).float()

        # ---------- 读取 mask png ----------
        mask_path = os.path.join(self.mask_dir, f"{img_idx:03d}.png")
        mask = Image.open(mask_path).convert("L")  # 灰度

        mask = np.array(mask, dtype=np.uint8)     # [128, 128]

        # 如果你的 mask 是 0/255，这一步非常重要
        if mask.max() > 1:
            mask[mask>1] = 1

        mask = torch.from_numpy(mask).long()       # [128, 128]

        return image, mask
