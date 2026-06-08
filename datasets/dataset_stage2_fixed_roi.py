import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from datasets.dataset_stage1_input_ablation import build_local_depth_edge, normalize_nonzero


def fixed_window_from_bbox(bbox, window_size, image_shape):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    image_h, image_w = image_shape
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    left = center_x - window_size // 2
    top = center_y - window_size // 2
    right = left + window_size
    bottom = top + window_size

    src_x1 = max(0, left)
    src_y1 = max(0, top)
    src_x2 = min(image_w, right)
    src_y2 = min(image_h, bottom)
    dst_x1 = src_x1 - left
    dst_y1 = src_y1 - top
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    return [src_x1, src_y1, src_x2, src_y2, dst_x1, dst_y1, dst_x2, dst_y2]


def crop_with_padding(arr, placement, window_size):
    src_x1, src_y1, src_x2, src_y2, dst_x1, dst_y1, dst_x2, dst_y2 = placement
    crop = np.zeros((window_size, window_size), dtype=arr.dtype)
    crop[dst_y1:dst_y2, dst_x1:dst_x2] = arr[src_y1:src_y2, src_x1:src_x2]
    return crop


class Stage2FixedRoiDataset(Dataset):
    def __init__(
        self,
        raw_items,
        mask_dir,
        indices,
        locator_meta_path,
        roi_size=80,
        edge_erode_iterations=2,
    ):
        self.intensity = np.load(Path(raw_items["intensity"])).astype(np.float32)
        self.depth = np.load(Path(raw_items["depth"])).astype(np.float32)
        self.mask_dir = Path(mask_dir)
        self.indices = list(indices)
        self.roi_size = int(roi_size)
        self.edge_erode_iterations = int(edge_erode_iterations)
        with Path(locator_meta_path).open("r", encoding="utf-8") as handle:
            self.locator_meta = json.load(handle)
        self.bboxes = self.locator_meta["bboxes"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label_index = int(self.indices[idx])
        array_index = label_index - 1
        full_intensity = self.intensity[array_index]
        full_depth = self.depth[array_index]
        image_shape = full_intensity.shape

        placement = fixed_window_from_bbox(
            self.bboxes[array_index],
            self.roi_size,
            image_shape,
        )
        intensity_roi_raw = crop_with_padding(full_intensity, placement, self.roi_size)
        depth_roi_raw = crop_with_padding(full_depth, placement, self.roi_size)

        intensity_roi = normalize_nonzero(intensity_roi_raw)
        local_edge_roi = build_local_depth_edge(
            depth_roi_raw,
            erode_iterations=self.edge_erode_iterations,
        )
        x = np.stack([intensity_roi, local_edge_roi], axis=0).astype(np.float32)

        mask_path = self.mask_dir / f"{label_index:03d}.png"
        full_gt = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        full_gt = (full_gt > 0).astype(np.int64)
        gt_roi = crop_with_padding(full_gt, placement, self.roi_size).astype(np.int64)

        return (
            torch.from_numpy(x),
            torch.from_numpy(gt_roi).long(),
            torch.from_numpy(full_gt).long(),
            torch.tensor(placement, dtype=torch.long),
            label_index,
        )
