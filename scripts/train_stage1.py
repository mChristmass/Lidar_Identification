import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.stage1.config_stage1 as conf
from datasets.dataset_stage1 import LidarSegDataset
from models.model_stage1 import UNet
from engine.stage1.visualize_stage1 import visualize_sample
from engine.stage1.loss_stage1 import SegmentationLoss
from engine.stage1.metrics_stage1 import compute_metrics


# ----------- 路径 -----------
NPY_PATH = conf.NPY_PATH
MASK_DIR = conf.MASK_DIR
SAVE_PATH = conf.SAVE_PATH
# 可视化保存路径
out_path = conf.out_path

# 固定测试集索引文件，便于消融实验
TEST_INDEX_PATH = conf.TEST_INDEX_PATH

# ----------- 参数 -----------
INPUT_CHANNEL = conf.INPUT_CHANNEL
BATCH_SIZE = conf.BATCH_SIZE
EPOCHS = conf.EPOCHS
LR = conf.LR
DEVICE = conf.DEVICE
# 类别权重
TARGET_WEIGHT = conf.TARGET_WEIGHT

# 固定随机种子，保证可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
#
#
# # ----------- 构建索引 -----------
# label_indices = sorted([
#     int(f.replace(".png", ""))
#     for f in os.listdir(MASK_DIR)
#     if f.endswith(".png")
# ])
#
# # 先固定 test 集
# if os.path.exists(TEST_INDEX_PATH):
#     test_indices = np.load(TEST_INDEX_PATH).tolist()
#     print(f"Loaded fixed test indices from: {TEST_INDEX_PATH}")
# else:
#     shuffled_indices = label_indices.copy()
#     random.shuffle(shuffled_indices)
#
#     total_num = len(shuffled_indices)
#     test_num = int(total_num * 0.1)
#
#     test_indices = shuffled_indices[-test_num:]
#     np.save(TEST_INDEX_PATH, np.array(test_indices, dtype=np.int32))
#     print(f"Saved fixed test indices to: {TEST_INDEX_PATH}")
#
# # 剩余样本用于 train/val 划分
# remain_indices = [idx for idx in label_indices if idx not in set(test_indices)]
#
# random.shuffle(remain_indices)
#
# remain_num = len(remain_indices)
# train_num = int(remain_num * 7 / 9)   # 因为剩余部分对应 90%，其中 train:val = 7:2
# val_num = remain_num - train_num
#
# train_indices = remain_indices[:train_num]
# val_indices = remain_indices[train_num:]
#
# print(f"Total samples : {len(label_indices)}")
# print(f"Train samples : {len(train_indices)}")
# print(f"Val samples   : {len(val_indices)}")
# print(f"Test samples  : {len(test_indices)}")



# ----------- 构建索引 -----------
label_indices = sorted([
    int(f.replace(".png", ""))
    for f in os.listdir(MASK_DIR)
    if f.endswith(".png")
])

label_set = set(label_indices)

# 读取固定 val / test 索引
val_indices  = np.load(conf.VAL_INDEX_PATH).astype(int).tolist()
test_indices = np.load(conf.TEST_INDEX_PATH).astype(int).tolist()

print(f"Loaded fixed val indices from : {conf.VAL_INDEX_PATH}")
print(f"Loaded fixed test indices from: {conf.TEST_INDEX_PATH}")

val_set = set(val_indices)
test_set = set(test_indices)

# 检查 val / test 是否存在重叠
overlap_val_test = val_set & test_set
if overlap_val_test:
    raise ValueError(f"Overlap between val and test: {sorted(overlap_val_test)}")

# 检查 val / test 索引是否都存在对应 mask
missing_val = sorted(val_set - label_set)
missing_test = sorted(test_set - label_set)

if missing_val:
    raise ValueError(f"Val indices not found in MASK_DIR: {missing_val}")

if missing_test:
    raise ValueError(f"Test indices not found in MASK_DIR: {missing_test}")

# 剩余样本作为 train
exclude_set = val_set | test_set
train_indices = [idx for idx in label_indices if idx not in exclude_set]

print(f"Total labeled samples : {len(label_indices)}")
print(f"Train samples         : {len(train_indices)}")
print(f"Val samples           : {len(val_indices)}")
print(f"Test samples          : {len(test_indices)}")

# ----------- DataLoader -----------
train_set = LidarSegDataset(NPY_PATH, MASK_DIR, train_indices)
val_set   = LidarSegDataset(NPY_PATH, MASK_DIR, val_indices)
test_set  = LidarSegDataset(NPY_PATH, MASK_DIR, test_indices)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)


# ----------- 模型、loss、优化器 -----------
model = UNet(in_channels=INPUT_CHANNEL, num_classes=2).to(DEVICE)

# criterion = SegmentationLoss(target_weight=TARGET_WEIGHT).to(DEVICE)
criterion = SegmentationLoss(
    target_weight=conf.TARGET_WEIGHT,
    ce_weight=conf.CE_WEIGHT,
    dice_weight=conf.DICE_WEIGHT
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR)


# ----------- 评估函数 -----------
# def evaluate(model, loader):
    # model.eval()
    #
    # total_iou = 0
    # total_p = 0
    # total_r = 0
    # n = 0
    #
    # with torch.no_grad():
    #     for imgs, labels in loader:
    #         imgs = imgs.to(DEVICE)
    #         labels = labels.to(DEVICE)
    #
    #         logits = model(imgs)
    #
    #         # if imgs.shape[1] == 3:
    #         #     valid_mask = imgs[:, 2, :, :].float()
    #         #     iou, p, r = compute_metrics(logits, labels, valid_mask)
    #         # else:
    #         iou, p, r = compute_metrics(logits, labels)
    #
    #         total_iou += iou
    #         total_p += p
    #         total_r += r
    #         n += 1
    #
    # return total_iou / n, total_p / n, total_r / n

def compute_stage1_metrics(logits, labels, threshold=0.3, valid_mask=None):
    probs = torch.softmax(logits, dim=1)[:, 1, :, :]
    preds = (probs > threshold).long()

    if valid_mask is not None:
        preds = preds * valid_mask.long()
        labels = labels * valid_mask.long()

    preds_f = preds.float()
    labels_f = labels.float()

    intersection = (preds_f * labels_f).sum(dim=(1, 2))
    pred_sum = preds_f.sum(dim=(1, 2))
    label_sum = labels_f.sum(dim=(1, 2))
    union = pred_sum + label_sum - intersection

    iou = (intersection / (union + 1e-6)).mean().item()
    precision = (intersection / (pred_sum + 1e-6)).mean().item()
    recall = (intersection / (label_sum + 1e-6)).mean().item()

    # GT coverage：GT中有多少被预测覆盖
    coverage = recall

    # 每张图预测前景像素数，可粗略反映coarse mask是否太大
    pred_area = pred_sum.mean().item()

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "pred_area": pred_area,
    }

def evaluate(model, loader, threshold=0.3):
    model.eval()

    total_iou = 0
    total_p = 0
    total_r = 0
    total_cov = 0
    total_pred_area = 0
    n = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(imgs)

            valid_mask = None
            # 如果第三通道是valid mask，可打开
            # if imgs.shape[1] >= 3:
            #     valid_mask = imgs[:, 2, :, :].float()

            metrics = compute_stage1_metrics(
                logits, labels,
                threshold=threshold,
                valid_mask=valid_mask
            )

            total_iou += metrics["iou"]
            total_p += metrics["precision"]
            total_r += metrics["recall"]
            total_cov += metrics["coverage"]
            total_pred_area += metrics["pred_area"]
            n += 1

    return {
        "iou": total_iou / n,
        "precision": total_p / n,
        "recall": total_r / n,
        "coverage": total_cov / n,
        "pred_area": total_pred_area / n,
    }


# ----------- 训练 -----------
train_losses = []
val_metrics = []

best_val_iou = -1

best_val_recall = -1
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(imgs)

        # if imgs.shape[1] == 3:
        #     valid_mask = imgs[:, 2, :, :].float()
        #     loss = criterion(logits, labels, valid_mask)
        # else:
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    # ----------------------------------------------
    # val_iou, val_precision, val_recall = evaluate(model, val_loader)
    evalue = evaluate(model, val_loader, threshold=conf.EVALUATE_threshold)
    val_iou = evalue["iou"]
    val_precision = evalue["precision"]
    val_recall = evalue["recall"]
    val_coverage = evalue["coverage"]
    val_pred_area = evalue["pred_area"]
    #------------------------------------------------

    train_losses.append(avg_loss)
    val_metrics.append((val_iou, val_precision, val_recall))

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]  "
        f"Loss: {avg_loss:.4f}  "
        f"Val IoU: {val_iou:.3f}  "
        f"Val Precision: {val_precision:.3f}  "
        f"Val Recall: {val_recall:.3f}"
        f"Val coverage: {val_coverage:.3f}"
        f"Val pred_area: {val_pred_area:.3f}"
    )

    # ---------------------------------------------------------
    # 保存验证集表现最好的模型
    # if val_iou > best_val_iou:
    #     best_val_iou = val_iou
    #     torch.save(model.state_dict(), SAVE_PATH)
    #     print(f"Best model saved. Val IoU = {best_val_iou:.3f}")

    # 保存验证集表现最好的模型

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Best model saved. Val Recall = {best_val_recall:.3f}")
    #-------------------------------------------------------------



# ----------- 载入最佳模型并在测试集上评估 -----------
print("\nLoading best model for final test evaluation...")
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()

#-------------------------------------------------------------------------
# test_iou, test_precision, test_recall = evaluate(model, test_loader)
evalue_test = evaluate(model, val_loader, threshold=conf.EVALUATE_threshold)
test_iou = evalue_test["iou"]
test_precision = evalue_test["precision"]
test_recall = evalue_test["recall"]
test_coverage = evalue_test["coverage"]
test_pred_area = evalue_test["pred_area"]
#--------------------------------------------------------------------------



print("\n===== Final Test Result =====")
print(
    f"Test IoU: {test_iou:.3f}  "
    f"Test Precision: {test_precision:.3f}  "
    f"Test Recall: {test_recall:.3f}"
)


# ----------- 测试集可视化 -----------
# out_path = r"D:\myComputer\pointsCloud\data\800label_data\HeBing\model\visualization\unet4"

os.makedirs(out_path, exist_ok=True)

num_visualize = conf.VISUALIZE_NUM
count = 0

# -------------------------------------------------------------------------------------------------------------------
# with torch.no_grad():
#     for data, label in test_loader:
#         data = data.to(DEVICE)
#
#         pred = model(data)
#         pred_mask = torch.argmax(pred, dim=1)  # [B, H, W]
#
#         for i in range(data.size(0)):
#             if count >= num_visualize:
#                 break
#
#             depth = data[i, 0].cpu().numpy()
#             intensity = data[i, 1].cpu().numpy()
#             mask = pred_mask[i].cpu().numpy()
#
#             visualize_sample(
#                 depth,
#                 intensity,
#                 mask,
#                 save_path=f"{out_path}/sample_{count}.png",
#                 title=f"Test Sample {count}"
#             )
#
#             count += 1
#
#         if count >= num_visualize:
#             break
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(DEVICE)

        pred = model(data)
        # 获取预测类别索引 [B, H, W]
        pred_mask = torch.argmax(pred, dim=1)

        for i in range(data.size(0)):
            if count >= num_visualize:
                break

            # --- 修改部分 ---
            # 假设 data 形状为 [B, 1, H, W]
            # 取出唯一的单通道 intensity 数据
            intensity = data[i, 0].cpu().numpy()

            # 由于不再有 depth 数据，我们可以传入 None 或者重复使用 intensity
            # 这取决于你的 visualize_sample 函数是如何定义的
            mask = pred_mask[i].cpu().numpy()

            visualize_sample(
                intensity=intensity,
                mask=mask,
                save_path=f"{out_path}/sample_{count}.png",
                title=f"Test Sample {count}"
            )
            # ----------------

            count += 1

        if count >= num_visualize:
            break
# ----------------------------------------------------------------------------------------------------------



print(f"\nTraining finished. Best model saved to:\n{SAVE_PATH}")