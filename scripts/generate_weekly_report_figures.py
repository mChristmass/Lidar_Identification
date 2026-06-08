import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs import kfold_config as kfold
from datasets.dataset_stage1_input_ablation import build_local_depth_edge


OUTPUT_DIR = PROJECT_ROOT / "reports" / "weekly_2026-06-08"


def read_mean_iou(path):
    with path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["mean"]["iou"])


def make_iou_chart():
    experiments = [
        ("Intensity only", PROJECT_ROOT / "data/runs/run8/summary_stage1_only_strong.json"),
        ("Raw depth + edge", PROJECT_ROOT / "data/runs/run9/summary_stage1_input_ablation_A.json"),
        ("Local depth edge", PROJECT_ROOT / "data/runs/run9/summary_stage1_input_ablation_B.json"),
        ("Prior input C2", PROJECT_ROOT / "data/runs/run10/summary_stage2_local_edge_C2.json"),
        ("Fixed ROI", PROJECT_ROOT / "data/runs/run11/summary_stage2_fixed_roi.json"),
    ]
    labels = [label for label, _ in experiments]
    values = [read_mean_iou(path) for _, path in experiments]
    colors = ["#6B7280", "#3B82F6", "#16A34A", "#D97706", "#DC2626"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0.78, 0.85)
    ax.set_title("Five-fold mean IoU comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "iou_comparison.png", dpi=180)
    plt.close(fig)


def make_edge_comparison(frame_index=216):
    intensity = np.load(kfold.RAW_ITEMS["intensity"])[frame_index].astype(np.float32)
    depth = np.load(kfold.RAW_ITEMS["depth"])[frame_index].astype(np.float32)
    raw_edge = np.load(kfold.RAW_ITEMS["depth_edge"])[frame_index].astype(np.float32)
    local_edge = build_local_depth_edge(depth, erode_iterations=2)

    mask_path = kfold.LABEL_DIR / f"{frame_index + 1:03d}.png"
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    gt = (gt > 0).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    panels = [
        ("Intensity", intensity, "gray"),
        ("Raw depth edge", raw_edge, "magma"),
        ("Local depth edge", local_edge, "magma"),
        ("Ground truth", gt, "gray"),
    ]
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "edge_comparison_frame217.png", dpi=180)
    plt.close(fig)


def copy_roundtrip_figure():
    source = (
        PROJECT_ROOT
        / "data/runs/run7/fold_seed42/stage2/roi_roundtrip_diagnosis_test/"
        "worst_visuals/000216_00_217.png.png"
    )
    target = OUTPUT_DIR / "roi_roundtrip_loss.png"
    target.write_bytes(source.read_bytes())


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    make_iou_chart()
    make_edge_comparison()
    copy_roundtrip_figure()
    print(f"Saved report figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
