import torch


BATCH_SIZE = 4
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_WEIGHT = 5.0
CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
ROI_LOSS_WEIGHT = 1.0
EDGE_ERODE_ITERATIONS = 2
