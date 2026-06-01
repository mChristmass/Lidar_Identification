import torch

from configs import kfold_config as kfold


# ----------- train paths -----------
TRAIN_PATH = kfold.as_str(kfold.stage2_roi_root(42) / "roi_train")
VAL_PATH = kfold.as_str(kfold.stage2_roi_root(42) / "roi_val")
TEST_PATH = kfold.as_str(kfold.stage2_roi_root(42) / "roi_test")

MASK_DIR = kfold.as_str(kfold.LABEL_DIR)
SAVE_PATH = kfold.as_str(kfold.stage2_model_path(42))
TEST_VIS_PATH = kfold.as_str(kfold.stage2_vis_dir(42))
RAW_INTENSITY_PATH = kfold.as_str(kfold.RAW_ITEMS["intensity"])

# ----------- parameters -----------
INPUT_ITEMS = kfold.STAGE2_INPUT_ITEMS
INPUT_CHANNEL = len(INPUT_ITEMS)
ROI_size = list(kfold.ROI_SIZE)

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_WEIGHT = 2.0
CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
