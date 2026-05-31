import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------- train参数 -----------
NPY_PATH = r"D:\myComputer\pointsCloud\data\identification\data\intensity.npy"
MASK_DIR = r"D:\myComputer\pointsCloud\data\identification\data\label\new"
# 固定测试集索引文件，便于消融实验
TEST_INDEX_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\3407\test_indices_seed3407.npy"
VAL_INDEX_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\3407\val_indices_seed3407.npy"
# 模型路径
SAVE_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\3407\change_loss\unet_stage1_loss3407.pth"
# train最终可视化保存路径
out_path = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\3407\change_loss\unet_stage1_loss3407"

# ----------- 参数 -----------
INPUT_CHANNEL = 1
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3
EVALUATE_threshold = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 类别权重
TARGET_WEIGHT = 5.0
CE_WEIGHT = 1.0
DICE_WEIGHT = 0.5

# 可视化
VISUALIZE_NUM = 5


# ----------- predict参数 -----------
Predict_PTH = r"D:\myComputer\pointsCloud\data\identification\data\intensity.npy"
# 模型
MODEL_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\3407\change_loss\unet_stage1_loss3407.pth"
# 输出目录
SAVE_LOGITS_NPY_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\3407\change_loss\coarse_masks\coarse_masks.npy"
SAVE_LOGITS_PNG_DIR = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\3407\change_loss\coarse_masks\png"

# 是否输出概率
IS_SAVE_PROB = True
SAVE_PROB_NPY_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\3407\change_loss\coarse_masks\prob.npy"


BATCH_SIZE = 8

