from pathlib import Path

import torch


DATA_ROOT = Path(r"D:\myComputer\pointsCloud\data\identification\data\new_data\merged")
LABEL_DIR = DATA_ROOT / "label"
INDEX_DIR = DATA_ROOT / "group_folds"
MAPPING_PATH = DATA_ROOT / "frame_mapping.csv"

RAW_ITEMS = {
    "intensity": DATA_ROOT / "intensity.npy",
    "depth": DATA_ROOT / "depth.npy",
    "local_depth_edge": DATA_ROOT / "local_depth_edge.npy",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEW_DATA_RUNS_ROOT = PROJECT_ROOT / "data" / "new_data_run0612"
RUNS_DIR = NEW_DATA_RUNS_ROOT / "run1"

NUM_FOLDS = 5
NUM_GROUP_PARTITIONS = 10
GROUP_BLOCK_SIZE = 50
FOLD_SEEDS = [42, 777, 2025, 3407, 114514]

BATCH_SIZE = 8
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
LR = 1e-3
TARGET_WEIGHT = 5.0
CE_WEIGHT = 1.0
DICE_WEIGHT = 1.0
EDGE_DROPOUT = 0.10
EDGE_ERODE_ITERATIONS = 2

INTENSITY_BASE_CHANNELS = 32
EDGE_BASE_CHANNELS = 16
LIGHT_UNET_BASE_CHANNELS = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
