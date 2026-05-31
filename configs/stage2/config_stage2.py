import torch

# ----------- train参数 -----------
TRAIN_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\42\+probability\ROI\roi_train"
VAL_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\42\+probability\ROI\roi_val"
TEST_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\42\+probability\ROI\roi_test"

MASK_DIR = r"D:\myComputer\pointsCloud\data\identification\data\label\new"
# 模型路径
SAVE_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\42\+probability\stage2_unet42+logits.pth"
# 可视化保存路径
TEST_VIS_PATH = r"D:\myComputer\pointsCloud\data\identification\data\ROI\dif__indices\42\+probability\Final_mask"
# 原始intensity图
RAW_INTENSITY_PATH = r"D:\myComputer\pointsCloud\data\identification\data\intensity.npy"


# ----------- 参数 -----------
INPUT_ITEMS = ["intensity", "depth", "depth_edge", "prob"]
INPUT_CHANNEL = len(INPUT_ITEMS)
ROI_size = [128, 128]

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 类别权重

# 可视化
