import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
PRIOR_DROPOUT = 0.30
GATE_TRAIN_RATIO = 0.80
GATE_INIT_C1_WEIGHT = 0.90
GATE_REG_WEIGHT = 0.01
