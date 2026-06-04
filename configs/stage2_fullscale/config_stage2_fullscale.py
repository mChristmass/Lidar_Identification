import torch

from configs import kfold_config as kfold


INPUT_ITEMS = ["intensity", "depth", "depth_edge", "prob", "coarse_mask", "roi_mask"]
INPUT_CHANNEL = len(INPUT_ITEMS)

BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_WEIGHT = 3.0
CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
ROI_LOSS_WEIGHT = 1.0

COARSE_THRESHOLD = 0.5
ROI_MARGIN = 8
ROI_DILATE_ITER = 0

RAW_ITEMS = kfold.RAW_ITEMS
MASK_DIR = kfold.LABEL_DIR
