import os
from sys import meta_path

import numpy as np
import torch
from torch.utils.data import Dataset
import json
#
#
# class Stage2Dataset(Dataset):
#     def __init__(self, root_dir, input_items):
#         self.samples = []
#         self.input_items = input_items
#
#         for name in os.listdir(root_dir):
#             sample_dir = os.path.join(root_dir, name)
#
#             if not os.path.isdir(sample_dir):
#                 continue
#
#             intensity_path = os.path.join(sample_dir, "intensity.npy")
#             depth_path = os.path.join(sample_dir, "depth.npy")
#             edge_path = os.path.join(sample_dir, "depth_edge.npy")
#             gt_path = os.path.join(sample_dir, "gt_mask.npy")
#             meta_path = ""
#
#             if not os.path.exists(gt_path):
#                 meta_path = os.path.join(sample_dir, "meta.json")
#                 # continue  # test集跳过。
#
#             self.samples.append({
#                 "intensity": intensity_path,
#                 "depth": depth_path,
#                 "depth_edge": edge_path,
#                 "gt": gt_path,
#                 "meta": meta_path
#             })
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         item = self.samples[idx]
#
#         feature_list = []
#
#         for input_name in self.input_items:
#             if input_name not in item:
#                 raise ValueError(f"Unsupported input item: {input_name}")
#
#             input_path = item[input_name]
#             if not os.path.exists(input_path):
#                 raise FileNotFoundError(f"Missing input file: {input_path}")
#
#             data = np.load(input_path).astype(np.float32)
#
#             # 归一化（很关键）
#             data = (data - data.mean()) / (data.std() + 1e-6)
#
#             feature_list.append(data)
#
#         # gt = np.load(item["gt"]).astype(np.int64)
#         if os.path.exists(item["gt"]):
#             gt = np.load(item["gt"]).astype(np.int64)
#         else:
#             gt = np.empty((0,), dtype=np.int64)
#
#         x = np.stack(feature_list, axis=0)  # [C,H,W]
#
#         if item["meta"]:
#             with open(item["meta"], "r", encoding="utf-8") as f:
#                 meta = json.load(f)
#         else:
#             meta = []
#
#         return torch.tensor(x), torch.tensor(gt), meta


# 与上面相同，只不过会自动拼接INPUT_ITEMS中的npy文件名称，可以做到只在config中更改通道数据即可运行
class Stage2Dataset(Dataset):
    def __init__(self, root_dir, input_items):
        self.samples = []
        self.input_items = input_items

        for name in os.listdir(root_dir):
            sample_dir = os.path.join(root_dir, name)
            if not os.path.isdir(sample_dir):
                continue

            # 自动根据 input_items 拼接路径
            sample_dict = {}
            for item_name in self.input_items:
                sample_dict[item_name] = os.path.join(sample_dir, f"{item_name}.npy")

            # gt 文件和 meta
            gt_path = os.path.join(sample_dir, "gt_mask.npy")
            if os.path.exists(gt_path):
                sample_dict["gt"] = gt_path
                sample_dict["meta"] = ""
            else:
                sample_dict["gt"] = gt_path  # 即使不存在，也保留 key
                meta_path = os.path.join(sample_dir, "meta.json")
                sample_dict["meta"] = meta_path if os.path.exists(meta_path) else ""

            self.samples.append(sample_dict)

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
            # 归一化
            data = (data - data.mean()) / (data.std() + 1e-6)
            feature_list.append(data)

        x = np.stack(feature_list, axis=0)  # [C,H,W]

        if os.path.exists(item["gt"]):
            gt = np.load(item["gt"]).astype(np.int64)
        else:
            gt = np.empty((0,), dtype=np.int64)

        # 读取 meta
        if item["meta"] and os.path.exists(item["meta"]):
            with open(item["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = []

        return torch.tensor(x), torch.tensor(gt), meta
