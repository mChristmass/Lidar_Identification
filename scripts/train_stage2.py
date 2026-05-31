import torch
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage2.config_stage2 as conf
from models.model_stage2 import Stage2UNet
from datasets.dataset_stage2 import Stage2Dataset
from engine.stage2.loss_stage2 import Stage2Loss


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_metrics(logits, labels):
    """
    logits: [B, 2, H, W]
    labels: [B, H, W]
    """

    preds = torch.argmax(logits, dim=1)  # [B,H,W]

    preds = preds.view(-1)
    labels = labels.view(-1)

    TP = ((preds == 1) & (labels == 1)).sum().float()
    FP = ((preds == 1) & (labels == 0)).sum().float()
    FN = ((preds == 0) & (labels == 1)).sum().float()

    iou = TP / (TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-6)

    return iou.item(), precision.item(), recall.item(), dice.item()


def reconstruct_to_full_scale(roi_pred, meta):
    """
    roi_pred: 单张预测图 [H_roi, W_roi] (128x128)
    meta: 该 ROI 对应的元数据字典
    """
    # 1. 这里的 roi_pred 是 Tensor 还是 Numpy 取决于你的 evaluate 逻辑，建议转为 numpy
    if isinstance(roi_pred, torch.Tensor):
        roi_pred = roi_pred.cpu().numpy().astype(np.uint8)

    # 2. 提取 meta 信息
    pad_top = meta['resize_meta']['pad_top']
    pad_bottom = meta['resize_meta']['pad_bottom']
    pad_left = meta['resize_meta']['pad_left']
    pad_right = meta['resize_meta']['pad_right']
    orig_crop_h, orig_crop_w = meta['resize_meta']['orig_shape']
    full_h, full_w = meta['orig_image_shape']
    x1, y1, x2, y2 = meta['bbox_margin_xyxy']

    # 3. 裁剪掉 Padding
    target_h, target_w = meta['resize_meta']['target_shape']
    h_end = target_h - pad_bottom
    w_end = target_w - pad_right
    roi_unpadded = roi_pred[pad_top:h_end, pad_left:w_end]

    # 4. 缩放回裁剪时的原始尺寸
    roi_orig_size = cv2.resize(roi_unpadded, (orig_crop_w, orig_crop_h), interpolation=cv2.INTER_NEAREST)

    # 5. 创建全图画布并贴回
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = roi_orig_size

    return full_mask


def evaluate_full_scale(model, loader, device, mask_dir, save_dir=None): # 1. 增加 save_dir
    model.eval()
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 为了拿到全图 intensity，需要在这里预载原始 npy
    # 假设你的 config 里有原始数据的路径
    raw_intensity_all = np.load(conf.RAW_INTENSITY_PATH)

    total_tp, total_fp, total_fn = 0, 0, 0
    with torch.no_grad():
        for x, y, metas in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for i in range(preds.shape[0]):
                # --- 关键修改点：手动从 metas 字典中提取第 i 个样本的属性 ---
                meta = {
                    'mask_name': metas['mask_name'][i],
                    'orig_image_shape': [
                        int(metas['orig_image_shape'][0][i]),
                        int(metas['orig_image_shape'][1][i])
                    ],
                    'bbox_margin_xyxy': [
                        int(metas['bbox_margin_xyxy'][0][i]),
                        int(metas['bbox_margin_xyxy'][1][i]),
                        int(metas['bbox_margin_xyxy'][2][i]),
                        int(metas['bbox_margin_xyxy'][3][i])
                    ],
                    'resize_meta': {
                        'pad_top': int(metas['resize_meta']['pad_top'][i]),
                        'pad_bottom': int(metas['resize_meta']['pad_bottom'][i]),
                        'pad_left': int(metas['resize_meta']['pad_left'][i]),
                        'pad_right': int(metas['resize_meta']['pad_right'][i]),
                        'orig_shape': [
                            int(metas['resize_meta']['orig_shape'][0][i]),
                            int(metas['resize_meta']['orig_shape'][1][i])
                        ],
                        'target_shape': conf.ROI_size  # 这个是 CONFIG 里的 roi_size
                    }
                }

                # 1. 还原全图预测
                full_pred = reconstruct_to_full_scale(preds[i], meta)

                # 2. 获取全图 GT
                mask_path = os.path.join(mask_dir, meta['mask_name'])
                full_gt = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

                # 3. 计算指标
                total_tp += ((full_pred == 1) & (full_gt == 1)).sum()
                total_fp += ((full_pred == 1) & (full_gt == 0)).sum()
                total_fn += ((full_pred == 0) & (full_gt == 1)).sum()

                # 4. 执行可视化保存
                if save_dir:
                    intensity_full = raw_intensity_all[int(meta['mask_name'].split('.')[0])-1,:,:]
                    save_path = os.path.join(save_dir, f"result_{meta['mask_name']}")
                    save_stage2_visualization(intensity_full, full_gt, full_pred, meta, save_path)

    # 计算最终全图指标
    full_iou = total_tp / (total_tp + total_fp + total_fn + 1e-6)
    full_precision = total_tp / (total_tp + total_fp + 1e-6)
    full_recall = total_tp / (total_tp + total_fn + 1e-6)

    return full_iou, full_precision, full_recall

def evaluate(model, loader, device):
    model.eval()

    total_iou = 0
    total_p = 0
    total_r = 0
    total_d = 0
    n = 0

    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            iou, p, r, d = compute_metrics(logits, y)

            total_iou += iou
            total_p += p
            total_r += r
            total_d += d
            n += 1

    return (
        total_iou / n,
        total_p / n,
        total_r / n,
        total_d / n
    )

def save_stage2_visualization(intensity_full, gt_full, pred_full, meta, save_path):
    """
    绘制并保存全图尺度的结果
    """
    plt.figure(figsize=(16, 9))

    # 1. 原始强度图 + 预测叠加 (Overlay)
    plt.subplot(2, 2, 1)
    plt.imshow(intensity_full, cmap='gray')
    # 创建一个红色的透明层
    overlay = np.zeros((*pred_full.shape, 4))
    overlay[pred_full == 1] = [1, 0, 0, 0.4]  # 红色，40%透明度
    plt.imshow(overlay)
    plt.title(f"Full-Scale Prediction Overlay (Frame: {meta['mask_name']})")
    plt.axis('off')

    # 2. 误差热力图 (Error Map) - 科研汇报利器
    plt.subplot(2, 2, 2)
    error_vis = np.zeros((*pred_full.shape, 3))
    error_vis[(pred_full == 1) & (gt_full == 1)] = [0, 1, 0]  # TP: 绿色 (预测正确)
    error_vis[(pred_full == 1) & (gt_full == 0)] = [1, 0, 0]  # FP: 红色 (虚警)
    error_vis[(pred_full == 0) & (gt_full == 1)] = [0, 0, 1]  # FN: 蓝色 (漏检)
    plt.imshow(error_vis)
    plt.title("Error Analysis: TP(Green), FP(Red), FN(Blue)")
    plt.axis('off')

    # 3. 局部 ROI 放大图 (Zoom-in)
    plt.subplot(2, 2, 3)
    x1, y1, x2, y2 = meta['bbox_margin_xyxy']
    # 稍微多取一点范围以便观察上下文
    roi_view = intensity_full[max(0, y1 - 5):min(y2 + 5, 128), max(0, x1 - 5):min(x2 + 5, 128)]
    plt.imshow(roi_view, cmap='gray')
    plt.title("ROI Region Detail (1500m Target)")

    # 4. 信息面板
    plt.subplot(2, 2, 4)
    info_text = (
        f"Frame Index: {int(meta['mask_name'].split('.')[0])-1}\n"
        f"BBox (xyxy): {meta['bbox_margin_xyxy']}\n"
        f"ROI Orig Shape: {meta['resize_meta']['orig_shape']}\n"
        f"Stage 2 Status: Refinement Complete"
    )
    plt.text(0.1, 0.5, info_text, fontsize=12, family='monospace')
    plt.axis('off')
    plt.title("Metadata & Status")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()  # 重要：必须关闭以防内存溢出


def train():
    # ========= dataset =========
    train_set = Stage2Dataset(conf.TRAIN_PATH, conf.INPUT_ITEMS)
    val_set = Stage2Dataset(conf.VAL_PATH, conf.INPUT_ITEMS)
    test_set = Stage2Dataset(conf.TEST_PATH, conf.INPUT_ITEMS)

    train_loader = DataLoader(train_set, batch_size=conf.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=conf.BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=conf.BATCH_SIZE)

    # ========= model =========
    model = Stage2UNet().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_iou = 0

    # ========= training =========
    for epoch in range(conf.EPOCHS):

        model.train()
        total_loss = 0

        for x, y, meta in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ========= validation =========
        val_iou, val_precision, val_recall, val_dice = evaluate(model, val_loader, DEVICE)

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), conf.SAVE_PATH)
            print(">>> Saved best model")

        print(
            f"Epoch [{epoch + 1}/{conf.EPOCHS}]  "
            f"Loss: {total_loss:.4f}  "
            f"Val IoU: {val_iou:.3f}  "
            f"Val Precision: {val_precision:.3f}  "
            f"Val Recall: {val_recall:.3f}"
            f"Val Dice: {val_dice:.3f}"
        )

    # ========= TEST（关键） =========
    print("\nLoading best model for TEST...")

    model = Stage2UNet().to(DEVICE)

    model.load_state_dict(torch.load(conf.SAVE_PATH))
    model.eval()

    # iou, p, r, d = evaluate(model, test_loader, DEVICE)
    #
    # # 2.强制释放显存并清理垃圾回收
    # import gc
    # torch.cuda.empty_cache()
    # gc.collect()

    # 修改调用，传入保存路径
    full_iou, full_precision, full_recall = evaluate_full_scale(
        model,
        test_loader,
        DEVICE,
        conf.MASK_DIR,
        save_dir=conf.TEST_VIS_PATH  # 传入路径
    )

    print("\n===== FINAL TEST RESULT =====")
    # print(f"IoU: {iou:.4f}")
    # print(f"Precision: {p:.4f}")
    # print(f"Recall: {r:.4f}")
    # print(f"Dice: {d:.4f}")
    print("\nGLOBAL ACCURACY:")
    print(f"IoU: {full_iou:.4f}")
    print(f"Precision: {full_precision:.4f}")
    print(f"Recall: {full_recall:.4f}")




if __name__ == "__main__":

    train()
