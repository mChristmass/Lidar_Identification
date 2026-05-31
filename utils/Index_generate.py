import os
import random
import numpy as np

# ==============================
# 路径
# ==============================

MASK_DIR = r"D:\myComputer\pointsCloud\data\identification\data\label\new"

# 保存索引的目录
SAVE_DIR = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices"

os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================
# 参数
# =====================================

# 多组实验随机种子
SEEDS = [42, 3407, 2025, 777, 114514]

# 划分比例
TEST_RATIO = 0.1
VAL_RATIO = 0.2

# =====================================
# 获取所有样本索引
# =====================================

mask_files = sorted([
    f for f in os.listdir(MASK_DIR)
    if f.endswith(".png")
])

# -------------------------------------
# 关键：
# mask 文件名从 001.png 开始
# 但索引需要从 0 开始
#
# 例如：
# 001.png -> index 0
# 002.png -> index 1
# ...
# -------------------------------------

all_indices = sorted([
    int(f.replace(".png", "")) - 1
    for f in mask_files
])

total_num = len(all_indices)

print(f"Total samples: {total_num}")

# =====================================
# 生成不同随机划分
# =====================================

for seed in SEEDS:

    print(f"\n========== Seed {seed} ==========")

    random.seed(seed)

    shuffled = all_indices.copy()
    random.shuffle(shuffled)

    # ---------------------------------
    # 数量
    # ---------------------------------

    test_num = int(total_num * TEST_RATIO)
    val_num = int(total_num * VAL_RATIO)

    # ---------------------------------
    # 划分
    # ---------------------------------

    test_indices = shuffled[:test_num]
    val_indices = shuffled[test_num:test_num + val_num]

    # 排序（推荐）
    test_indices = sorted(test_indices)
    val_indices = sorted(val_indices)

    # ---------------------------------
    # 保存
    # ---------------------------------

    np.save(
        os.path.join(SAVE_DIR, f"test_indices_seed{seed}.npy"),
        np.array(test_indices, dtype=np.int32)
    )

    np.save(
        os.path.join(SAVE_DIR, f"val_indices_seed{seed}.npy"),
        np.array(val_indices, dtype=np.int32)
    )

    # ---------------------------------
    # 输出信息
    # ---------------------------------

    train_num = total_num - test_num - val_num

    print(f"Train samples : {train_num}")
    print(f"Val samples   : {len(val_indices)}")
    print(f"Test samples  : {len(test_indices)}")

print("\nAll splits generated successfully.")