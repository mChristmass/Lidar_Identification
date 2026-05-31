import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
import configs.stage1.config_stage1 as conf
from configs import kfold_config as kfold
from engine.stage1.normalization_stage1 import*
from models.model_stage1 import UNet


# =========================
# 路径与参数
# =========================
MODEL_PATH = conf.MODEL_PATH
NPY_PATH = conf.Predict_PTH

# 输出目录
SAVE_MASK_NPY_PATH = conf.SAVE_LOGITS_NPY_PATH
SAVE_MASK_PNG_DIR = conf.SAVE_LOGITS_PNG_DIR

# 输入通道数，必须与训练时一致
INPUT_CHANNEL = conf.INPUT_CHANNEL

# batch size 可适当调大
BATCH_SIZE = conf.BATCH_SIZE

DEVICE = conf.DEVICE

# 是否保存 png
SAVE_PNG = True


# =========================
# Dataset
# =========================
class PredictDataset(Dataset):
    def __init__(self, npy_path):
        self.images = data_normalization(npy_path)   # [N, C, H, W]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.from_numpy(image).float()
        return image, idx

def softmax_numpy(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)


# =========================
# 预测函数
# =========================
def predict_dataset(
    model_path,
    npy_path,
    save_mask_npy_path,
    save_mask_png_dir=None,
    is_save_prob = False,
    save_prob_npy_path=None,
    input_channel=2,
    batch_size=8,
    device="cpu"
):
    # ---------- 加载数据 ----------
    dataset = PredictDataset(npy_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ---------- 检查输入通道 ----------
    sample_data0 = np.load(npy_path)
    sample_data = sample_data0.reshape(sample_data0.shape[0], 1, sample_data0.shape[1], sample_data0.shape[2])
    if sample_data.ndim != 4:
        raise ValueError(f"输入数据应为 [N, C, H, W]，当前 shape={sample_data.shape}")

    n, c, h, w = sample_data.shape
    print(f"Loaded dataset shape: {sample_data.shape}")

    if c != input_channel:
        raise ValueError(
            f"数据通道数({c})与模型输入通道数 INPUT_CHANNEL({input_channel}) 不一致"
        )

    # ---------- 加载模型 ----------
    model = UNet(in_channels=input_channel, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Model loaded successfully.")

    # ---------- 保存结果数组 ----------
    # Logits 是浮点数，且形状为 [N, 2, H, W] (假设 num_classes=2)
    num_classes = 2
    pred_logits_all = np.zeros((n, num_classes, h, w), dtype=np.float32)

    if save_mask_png_dir is not None:
        os.makedirs(save_mask_png_dir, exist_ok=True)

    # ---------- 推理 ----------
    with torch.no_grad():
        for images, indices in loader:
            images = images.to(device)

            logits = model(images)                  # [B, 2, H, W]
            logits_np = logits.cpu().numpy() # 保持 float32 类型

            for i in range(logits_np.shape[0]):
                idx = indices[i].item()
                pred_logits_all[idx] = logits_np[i]

                # ---------- 3. (可选) 如果还要保存 PNG，需在这里做 argmax ----------
                if save_mask_png_dir is not None:
                    # PNG 无法直接存储原始 Logits，通常保存预测分类结果
                    mask_class = np.argmax(logits_np[i], axis=0)  # [H, W]
                    mask_img = (mask_class * 255).astype(np.uint8)
                    img = Image.fromarray(mask_img)
                    img.save(os.path.join(save_mask_png_dir, f"{idx:03d}.png"))

    # ---------- 保存 npy ----------
    np.save(save_mask_npy_path, pred_logits_all)

    if is_save_prob:
        print("正在计算概率图 prob...")
        # 转概率 [N, 2, H, W] → softmax → 取前景通道 [N, H, W]
        prob_all = softmax_numpy(pred_logits_all, axis=1)
        prob_all = prob_all[:, 1, :, :]  # 最终形状：[N, H, W]

        np.save(save_prob_npy_path, prob_all)
        print(f"prob 形状：{prob_all.shape}")

    print(f"Logits shape: {pred_logits_all.shape}")
    return pred_logits_all


def predict_one_fold(seed, save_png=True):
    save_mask_npy_path = kfold.stage1_logits_path(seed)
    save_prob_npy_path = kfold.stage1_prob_path(seed)
    save_png_dir = kfold.stage1_png_dir(seed)
    save_mask_npy_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n===== Stage1 predict fold seed {seed} =====")
    return predict_dataset(
        model_path=kfold.as_str(kfold.stage1_model_path(seed)),
        npy_path=conf.NPY_PATH,
        save_mask_npy_path=kfold.as_str(save_mask_npy_path),
        save_mask_png_dir=kfold.as_str(save_png_dir) if save_png else None,
        is_save_prob=conf.IS_SAVE_PROB,
        save_prob_npy_path=kfold.as_str(save_prob_npy_path),
        input_channel=INPUT_CHANNEL,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )


def predict_all_folds(save_png=True):
    for seed in kfold.KFOLD_SEEDS:
        predict_one_fold(seed, save_png=save_png)


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    predict_all_folds(save_png=SAVE_PNG)
