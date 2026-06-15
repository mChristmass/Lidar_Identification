import argparse
import json
from pathlib import Path

import numpy as np


DETAIL_KEYS = [
    "foreground_mean_image_iou",
    "foreground_median_image_iou",
    "foreground_iou_std",
    "background_false_positive_frame_rate",
    "background_mean_pred_area",
    "background_pixel_false_positive_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate per-fold detailed test reports for new-data experiments."
    )
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="summary_detailed_test_reports.json")
    return parser.parse_args()


def main():
    args = parse_args()
    grouped = {}
    for path in sorted(args.runs_dir.glob("fold_*/*/test_detailed_summary.json")):
        experiment = path.parent.name
        fold = path.parent.parent.name
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        grouped.setdefault(experiment, []).append({"fold": fold, **report})

    output = {}
    for experiment, rows in grouped.items():
        rows.sort(key=lambda row: int(row["fold"].split("_")[-1]))
        foreground_count = sum(row["foreground_images"] for row in rows)
        background_count = sum(row["background_images"] for row in rows)
        weighted_foreground_iou = sum(
            row["foreground_mean_image_iou"] * row["foreground_images"]
            for row in rows
        ) / max(1, foreground_count)
        total_background_fp_frames = sum(
            row["background_false_positive_frames"] for row in rows
        )
        total_background_fp_pixels = sum(
            row["background_false_positive_pixels"] for row in rows
        )
        image_pixels = 128 * 128
        output[experiment] = {
            "folds": rows,
            "fold_mean": {
                key: float(np.mean([row[key] for row in rows]))
                for key in DETAIL_KEYS
            },
            "fold_std": {
                key: float(np.std([row[key] for row in rows]))
                for key in DETAIL_KEYS
            },
            "pooled": {
                "test_images": sum(row["test_images"] for row in rows),
                "foreground_images": foreground_count,
                "background_images": background_count,
                "foreground_mean_image_iou": float(weighted_foreground_iou),
                "background_false_positive_frames": total_background_fp_frames,
                "background_false_positive_frame_rate": (
                    total_background_fp_frames / max(1, background_count)
                ),
                "background_false_positive_pixels": total_background_fp_pixels,
                "background_mean_pred_area": (
                    total_background_fp_pixels / max(1, background_count)
                ),
                "background_pixel_false_positive_rate": (
                    total_background_fp_pixels
                    / max(1, background_count * image_pixels)
                ),
            },
        }

    output_path = args.runs_dir / args.output_name
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")
    for experiment, report in sorted(output.items()):
        pooled = report["pooled"]
        print(
            f"{experiment}: image IoU={pooled['foreground_mean_image_iou']:.4f}, "
            f"background FP frames="
            f"{pooled['background_false_positive_frame_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
