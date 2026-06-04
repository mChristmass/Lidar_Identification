import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

from configs import kfold_config as kfold
from scripts.kfold_utils import tee_stdout

SEED = 42

CONFIG = {
    "raw_items": {
        "intensity": kfold.as_str(kfold.RAW_ITEMS["intensity"]),
        "depth": kfold.as_str(kfold.RAW_ITEMS["depth"]),
        "depth_edge": kfold.as_str(kfold.RAW_ITEMS["depth_edge"]),
        "prob": kfold.as_str(kfold.stage1_prob_path(3407)),
    },
    "stage1_logits_npy": kfold.as_str(kfold.stage1_logits_path(3407)),
    "mask_dir": kfold.as_str(kfold.LABEL_DIR),
    "test_index_path": kfold.as_str(kfold.index_path(3407, "test")),
    "val_index_path": kfold.as_str(kfold.index_path(3407, "val")),
    "save_root": kfold.as_str(kfold.stage2_roi_root(3407)),
    "selected_items": [
        "depth",
        "depth_edge",
        "coarse_mask",
        "intensity",
        "prob"
    ],
    # "pred": use Stage1 prediction to crop ROI.
    # "oracle": use GT mask to crop ROI for upper-bound diagnosis.
    "roi_source": "oracle", #"pred", "oracle"
    # split 比例
    "test_ratio": 0.1,
    "val_ratio": 0.1,  # 从非 test 的剩余数据里再划 val
    # ROI 参数
    "threshold": 0.30,
    "min_area": 20,
    "min_w": 4,
    "min_h": 4,
    "margin": 8,
    "max_area_ratio": 0.50,
    "roi_size": (128, 128),
    "save_vis": True,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def softmax_numpy(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-8)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    min_value = float(arr.min())
    max_value = float(arr.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    out = (arr - min_value) / (max_value - min_value)
    return (out * 255).clip(0, 255).astype(np.uint8)


def resize_with_aspect_and_pad(arr: np.ndarray, target_size: Tuple[int, int], is_mask: bool = False) -> Tuple[np.ndarray, Dict]:
    target_h, target_w = target_size
    height, width = arr.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("Empty crop encountered.")

    scale = min(target_h / height, target_w / width)
    new_h = max(1, int(round(height * scale)))
    new_w = max(1, int(round(width * scale)))

    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)

    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    meta = {
        "orig_shape": [int(height), int(width)],
        "resized_shape": [int(new_h), int(new_w)],
        "target_shape": [int(target_h), int(target_w)],
        "scale": float(scale),
        "pad_top": int(pad_top),
        "pad_bottom": int(pad_bottom),
        "pad_left": int(pad_left),
        "pad_right": int(pad_right),
    }
    return padded, meta


def read_mask_png(mask_dir: str, frame_idx: int) -> np.ndarray:
    mask_path = Path(mask_dir) / f"{frame_idx + 1:03d}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")

    return (mask > 0).astype(np.uint8)


def save_debug_visualization(save_path: Path, intensity: np.ndarray, fg_prob: np.ndarray, coarse_mask: np.ndarray, bbox_margin: List[int]) -> None:
    int_vis = normalize_to_uint8(intensity)
    prob_vis = (fg_prob * 255).clip(0, 255).astype(np.uint8)
    mask_vis = (coarse_mask * 255).astype(np.uint8)

    bbox_vis = int_vis.copy()
    x1, y1, x2, y2 = bbox_margin
    cv2.rectangle(bbox_vis, (x1, y1), (x2 - 1, y2 - 1), 255, 1)

    panel = np.concatenate([int_vis, prob_vis, mask_vis, bbox_vis], axis=1)
    cv2.imwrite(str(save_path), panel)


def build_splits(mask_dir: str, test_index_path: str, val_index_path: str, test_ratio: float, val_ratio: float):
    label_indices = sorted(
        int(name.replace(".png", "")) - 1
        for name in os.listdir(mask_dir)
        if name.endswith(".png")
    )

    if os.path.exists(test_index_path):
        test_indices = np.load(test_index_path).tolist()
    else:
        shuffled_indices = label_indices.copy()
        random.shuffle(shuffled_indices)
        test_num = int(len(shuffled_indices) * test_ratio)
        test_indices = shuffled_indices[-test_num:]
        np.save(test_index_path, np.array(test_indices, dtype=np.int32))

    remaining_indices = sorted(list(set(label_indices) - set(test_indices)))

    if os.path.exists(val_index_path):
        val_indices = np.load(val_index_path).tolist()
    else:
        shuffled_remaining = remaining_indices.copy()
        random.shuffle(shuffled_remaining)
        val_num = int(len(shuffled_remaining) * val_ratio)
        val_indices = shuffled_remaining[-val_num:]
        np.save(val_index_path, np.array(val_indices, dtype=np.int32))

    train_indices = sorted(list(set(remaining_indices) - set(val_indices)))
    return train_indices, sorted(val_indices), sorted(test_indices)


def logits_to_fg_prob(logits: np.ndarray) -> np.ndarray:
    return softmax_numpy(logits, axis=0)[1]


def prob_to_mask(fg_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (fg_prob > threshold).astype(np.uint8)


def get_main_component(binary_mask: np.ndarray, min_area: int, min_w: int, min_h: int, max_area_ratio: float):
    height, width = binary_mask.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    candidates = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < min_area or w < min_w or h < min_h:
            continue

        area_ratio = area / float(height * width)
        if area_ratio > max_area_ratio:
            continue

        candidates.append(
            {
                "label": i,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area": area,
                "area_ratio": area_ratio,
            }
        )

    if not candidates:
        return None, labels

    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates[0], labels


def expand_bbox(x: int, y: int, w: int, h: int, image_shape: Tuple[int, int], margin: int) -> List[int]:
    height, width = image_shape
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(width, x + w + margin)
    y2 = min(height, y + h + margin)
    return [x1, y1, x2, y2]


def crop_by_bbox(arr: np.ndarray, bbox: List[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return arr[y1:y2, x1:x2]


def prepare_feature_maps(base_maps: Dict[str, np.ndarray], fg_prob: np.ndarray, component_mask: np.ndarray) -> Dict[str, np.ndarray]:
    feature_maps = dict(base_maps)
    feature_maps["fg_prob"] = fg_prob.astype(np.float32)
    feature_maps["coarse_mask"] = component_mask.astype(np.uint8)
    return feature_maps


def is_mask_like_item(item_name: str) -> bool:
    return item_name in {"coarse_mask", "gt_mask"}


def build_single_roi(
    frame_idx: int,
    base_maps: Dict[str, np.ndarray],
    logits: np.ndarray,
    gt_mask: Optional[np.ndarray],
    cfg: Dict,
):
    reference = next(iter(base_maps.values()))
    height, width = reference.shape

    fg_prob = logits_to_fg_prob(logits)
    roi_source = cfg.get("roi_source", "pred")

    if roi_source == "oracle":
        if gt_mask is None:
            raise ValueError("Oracle ROI requires gt_mask.")
        coarse_mask = gt_mask.astype(np.uint8)
    elif roi_source == "pred":
        coarse_mask = prob_to_mask(fg_prob, cfg["threshold"])
    else:
        raise ValueError(f"Unsupported roi_source: {roi_source}")

    best, labels = get_main_component(
        coarse_mask,
        min_area=cfg["min_area"],
        min_w=cfg["min_w"],
        min_h=cfg["min_h"],
        max_area_ratio=cfg["max_area_ratio"],
    )
    if best is None:
        return None

    x, y, w, h = best["x"], best["y"], best["w"], best["h"]
    bbox_raw = [x, y, x + w, y + h]
    bbox_margin = expand_bbox(x, y, w, h, (height, width), cfg["margin"])
    component_mask = (labels == best["label"]).astype(np.uint8)

    feature_maps = prepare_feature_maps(base_maps, fg_prob, component_mask)
    roi_items = {}
    resize_meta = None

    for item_name in cfg["selected_items"]:
        if item_name not in feature_maps:
            raise KeyError(f"Selected item '{item_name}' is not available.")

        cropped = crop_by_bbox(feature_maps[item_name], bbox_margin)
        resized, item_resize_meta = resize_with_aspect_and_pad(
            cropped,
            cfg["roi_size"],
            is_mask=is_mask_like_item(item_name),
        )
        roi_items[item_name] = resized.astype(np.uint8 if is_mask_like_item(item_name) else np.float32)
        if resize_meta is None:
            resize_meta = item_resize_meta

    roi_gt = None
    if gt_mask is not None:
        gt_crop = crop_by_bbox(gt_mask, bbox_margin)
        roi_gt, _ = resize_with_aspect_and_pad(gt_crop, cfg["roi_size"], is_mask=True)
        roi_gt = roi_gt.astype(np.uint8)

    roi_id = f"{frame_idx:06d}_00"
    return {
        "roi_id": roi_id,
        "items": roi_items,
        "roi_gt": roi_gt,
        "meta": {
            "frame_idx": int(frame_idx),
            "mask_name": f"{frame_idx + 1:03d}.png",
            "roi_id": roi_id,
            "selected_items": list(cfg["selected_items"]),
            "orig_image_shape": [int(height), int(width)],
            "bbox_raw_xyxy": bbox_raw,
            "bbox_margin_xyxy": bbox_margin,
            "component_area": int(best["area"]),
            "component_area_ratio": float(best["area_ratio"]),
            "threshold": float(cfg["threshold"]),
            "roi_source": roi_source,
            "margin": int(cfg["margin"]),
            "roi_size": list(cfg["roi_size"]),
            "resize_meta": resize_meta,
            "has_gt": gt_mask is not None,
        },
        "debug": {
            "intensity": base_maps.get("intensity", reference),
            "fg_prob": fg_prob,
            "coarse_mask": coarse_mask,
            "bbox_margin": bbox_margin,
        },
    }


def save_roi_result(roi_result: Dict, save_dir: Path, save_vis: bool) -> None:
    roi_dir = save_dir / roi_result["roi_id"]
    ensure_dir(roi_dir)

    for item_name, item_array in roi_result["items"].items():
        np.save(roi_dir / f"{item_name}.npy", item_array)

    if roi_result["roi_gt"] is not None:
        np.save(roi_dir / "gt_mask.npy", roi_result["roi_gt"])

    save_json(roi_dir / "meta.json", roi_result["meta"])

    if save_vis:
        save_debug_visualization(
            roi_dir / "debug_vis.png",
            roi_result["debug"]["intensity"],
            roi_result["debug"]["fg_prob"],
            roi_result["debug"]["coarse_mask"],
            roi_result["debug"]["bbox_margin"],
        )


def load_raw_item_arrays(cfg: Dict) -> Dict[str, np.ndarray]:
    raw_arrays = {}
    for item_name, npy_path in cfg["raw_items"].items():
        raw_arrays[item_name] = np.load(npy_path).astype(np.float32)

    sample_count = None
    for item_name, item_array in raw_arrays.items():
        if item_array.ndim != 3:
            raise ValueError(f"{item_name} shape should be [N,H,W], got {item_array.shape}")
        if sample_count is None:
            sample_count = item_array.shape[0]
        elif item_array.shape[0] != sample_count:
            raise ValueError(f"Sample count mismatch for {item_name}, got {item_array.shape[0]}, expected {sample_count}")

    return raw_arrays


def build_rois_for_indices(
    indices: List[int],
    split_name: str,
    raw_arrays: Dict[str, np.ndarray],
    logits_all: np.ndarray,
    cfg: Dict,
) -> None:
    save_dir = Path(cfg["save_root"]) / f"roi_{split_name}"
    ensure_dir(save_dir)

    success = 0
    fail = 0

    for frame_idx in indices:
        base_maps = {item_name: item_array[frame_idx] for item_name, item_array in raw_arrays.items()}
        logits = logits_all[frame_idx]




        gt_mask = None
        if cfg.get("roi_source", "pred") == "oracle" or split_name in {"train", "val"}:
            gt_mask = read_mask_png(cfg["mask_dir"], frame_idx)

        # # 不再判断 split_name，所有模式都尝试读取 GT
        # try:
        #     gt_mask = read_mask_png(cfg["mask_dir"], frame_idx)
        # except Exception as e:
        #     print(f"Warning: Could not read mask for frame {frame_idx}: {e}")
        #     gt_mask = None



        roi_result = build_single_roi(
            frame_idx=frame_idx,
            base_maps=base_maps,
            logits=logits,
            gt_mask=gt_mask,
            cfg=cfg,
        )

        if roi_result is None:
            fail += 1
            print(f"[{split_name}] No valid ROI for frame_idx={frame_idx}")
            continue

        save_roi_result(roi_result, save_dir=save_dir, save_vis=cfg["save_vis"])
        success += 1

    print(f"[{split_name}] success={success}, fail={fail}, total={len(indices)}")


def validate_config(cfg: Dict) -> None:
    available_items = set(cfg["raw_items"].keys()) | {"fg_prob", "coarse_mask"}
    invalid_items = [item for item in cfg["selected_items"] if item not in available_items]
    if invalid_items:
        raise ValueError(f"Unsupported selected items: {invalid_items}")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    validate_config(CONFIG)

    train_indices, val_indices, test_indices = build_splits(
        mask_dir=CONFIG["mask_dir"],
        test_index_path=CONFIG["test_index_path"],
        val_index_path=CONFIG["val_index_path"],
        test_ratio=CONFIG["test_ratio"],
        val_ratio=CONFIG["val_ratio"],
    )
    print(f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    raw_arrays = load_raw_item_arrays(CONFIG)
    logits_all = np.load(CONFIG["stage1_logits_npy"]).astype(np.float32)




    if logits_all.ndim != 4 or logits_all.shape[1] != 2:
        raise ValueError(f"logits shape should be [N,2,H,W], got {logits_all.shape}")

    sample_count = next(iter(raw_arrays.values())).shape[0]
    if logits_all.shape[0] != sample_count:
        raise ValueError("Sample count mismatch between raw arrays and stage1 logits.")

    build_rois_for_indices(train_indices, "train", raw_arrays, logits_all, CONFIG)
    build_rois_for_indices(val_indices, "val", raw_arrays, logits_all, CONFIG)
    build_rois_for_indices(test_indices, "test", raw_arrays, logits_all, CONFIG)


def make_fold_config(seed: int) -> Dict:
    cfg = dict(CONFIG)
    cfg["raw_items"] = {
        "intensity": kfold.as_str(kfold.RAW_ITEMS["intensity"]),
        "depth": kfold.as_str(kfold.RAW_ITEMS["depth"]),
        "depth_edge": kfold.as_str(kfold.RAW_ITEMS["depth_edge"]),
        "prob": kfold.as_str(kfold.stage1_prob_path(seed)),
    }
    cfg["stage1_logits_npy"] = kfold.as_str(kfold.stage1_logits_path(seed))
    cfg["mask_dir"] = kfold.as_str(kfold.LABEL_DIR)
    cfg["save_root"] = kfold.as_str(kfold.stage2_roi_root(seed))
    cfg["test_index_path"] = kfold.as_str(kfold.index_path(seed, "test"))
    cfg["val_index_path"] = kfold.as_str(kfold.index_path(seed, "val"))
    return cfg


def to_zero_based(indices: List[int]) -> List[int]:
    return [int(idx) - 1 for idx in indices]


def build_one_fold(seed: int) -> None:
    cfg = make_fold_config(seed)
    log_path = kfold.stage2_dir(seed) / "build_stage2_rois.log"
    with tee_stdout(log_path):
        print(f"\n===== Build stage2 ROIs fold seed {seed} =====")
        validate_config(cfg)

        train_indices, val_indices, test_indices = kfold.load_split_indices(seed)
        train_indices = to_zero_based(train_indices)
        val_indices = to_zero_based(val_indices)
        test_indices = to_zero_based(test_indices)
        print(f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        raw_arrays = load_raw_item_arrays(cfg)
        logits_all = np.load(cfg["stage1_logits_npy"]).astype(np.float32)

        if logits_all.ndim != 4 or logits_all.shape[1] != 2:
            raise ValueError(f"logits shape should be [N,2,H,W], got {logits_all.shape}")

        sample_count = next(iter(raw_arrays.values())).shape[0]
        if logits_all.shape[0] != sample_count:
            raise ValueError("Sample count mismatch between raw arrays and stage1 logits.")

        build_rois_for_indices(train_indices, "train", raw_arrays, logits_all, cfg)
        build_rois_for_indices(val_indices, "val", raw_arrays, logits_all, cfg)
        build_rois_for_indices(test_indices, "test", raw_arrays, logits_all, cfg)


def build_all_folds() -> None:
    for seed in kfold.KFOLD_SEEDS:
        build_one_fold(seed)


if __name__ == "__main__":
    build_all_folds()
