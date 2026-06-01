import torch

from configs import kfold_config as kfold


# ----------- train paths -----------
NPY_PATH = kfold.as_str(kfold.RAW_ITEMS[kfold.STAGE1_INPUT_ITEM])
MASK_DIR = kfold.as_str(kfold.LABEL_DIR)
TEST_INDEX_PATH = kfold.as_str(kfold.index_path(3407, "test"))
VAL_INDEX_PATH = kfold.as_str(kfold.index_path(3407, "val"))
SAVE_PATH = kfold.as_str(kfold.stage1_model_path(3407))
out_path = kfold.as_str(kfold.stage1_vis_dir(3407))

# ----------- train parameters -----------
INPUT_CHANNEL = kfold.STAGE1_INPUT_CHANNEL
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3
EVALUATE_threshold = 0.30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_WEIGHT = 5.0
CE_WEIGHT = 1.0
TVERSKY_WEIGHT = 0.5
TVERSKY_ALPHA = 0.35
TVERSKY_BETA = 0.65
FBETA_BETA = 2.0
VISUALIZE_NUM = 5

# ----------- predict parameters -----------
Predict_PTH = NPY_PATH
MODEL_PATH = SAVE_PATH
SAVE_LOGITS_NPY_PATH = kfold.as_str(kfold.stage1_logits_path(3407))
SAVE_LOGITS_PNG_DIR = kfold.as_str(kfold.stage1_png_dir(3407))
IS_SAVE_PROB = True
SAVE_PROB_NPY_PATH = kfold.as_str(kfold.stage1_prob_path(3407))
