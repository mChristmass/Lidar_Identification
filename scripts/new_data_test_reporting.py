import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from datasets.dataset_stage1_input_ablation import normalize_nonzero


def _safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def _display_gray(array):
    normalized = normalize_nonzero(np.asarray(array, dtype=np.float32))
    return np.clip(normalized / 1.1 * 255.0, 0, 255).astype(np.uint8)


def _display_depth(array):
    gray = _display_gray(array)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[gray == 0] = 0
    return colored


def _panel(image, title, scale=2):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(
        image,
        (image.shape[1] * scale, image.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    header = np.full((30, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([header, image])


def _error_overlay(intensity, label, prediction):
    base = cv2.cvtColor(_display_gray(intensity), cv2.COLOR_GRAY2BGR)
    true_positive = prediction & label
    false_positive = prediction & ~label
    false_negative = ~prediction & label

    color = base.copy()
    color[true_positive] = (40, 200, 40)
    color[false_positive] = (30, 30, 240)
    color[false_negative] = (240, 80, 30)
    return cv2.addWeighted(base, 0.35, color, 0.65, 0)


def save_visualization(
    output_path,
    label_index,
    intensity,
    depth,
    local_edge,
    label,
    prediction,
    row,
):
    gt_image = (label.astype(np.uint8) * 255)
    pred_image = (prediction.astype(np.uint8) * 255)
    overlay = _error_overlay(intensity, label, prediction)

    if row["is_background"]:
        metric_text = f"Background | FP pixels: {row['fp']}"
    else:
        metric_text = (
            f"IoU {row['iou']:.3f} | P {row['precision']:.3f} | "
            f"R {row['recall']:.3f}"
        )

    panels = [
        _panel(_display_gray(intensity), "Intensity"),
        _panel(_display_depth(depth), "Depth"),
        _panel(np.clip(local_edge * 255.0, 0, 255).astype(np.uint8), "Local depth edge"),
        _panel(gt_image, "Ground truth"),
        _panel(pred_image, "Prediction"),
        _panel(overlay, "TP green | FP red | FN blue"),
    ]
    first_row = np.hstack(panels[:3])
    second_row = np.hstack(panels[3:])
    title = np.full((42, first_row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        title,
        f"Frame {label_index} | {metric_text}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack([title, first_row, second_row]))


def evaluate_and_report_test_set(
    model,
    dataset,
    raw_items,
    label_dir,
    device,
    threshold,
    output_dir,
    batch_size,
    num_workers=0,
    save_visualizations=True,
):
    output_dir = Path(output_dir)
    prediction_dir = output_dir / "test_predictions"
    visualization_dir = output_dir / "test_visualizations"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if save_visualizations:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        name: np.load(Path(raw_items[name]), mmap_mode="r")
        for name in ("intensity", "depth", "local_depth_edge")
    }
    rows = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
            x = torch.stack([sample[0] for sample in samples]).to(device)
            labels = torch.stack([sample[1] for sample in samples]).numpy().astype(bool)
            probabilities = torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()
            predictions = probabilities > threshold

            for offset, (label, prediction) in enumerate(zip(labels, predictions)):
                dataset_position = start + offset
                label_index = int(dataset.indices[dataset_position])
                tp = int((prediction & label).sum())
                fp = int((prediction & ~label).sum())
                fn = int((~prediction & label).sum())
                pred_area = int(prediction.sum())
                label_area = int(label.sum())
                is_background = label_area == 0
                iou = None if is_background else _safe_ratio(tp, tp + fp + fn)
                precision = _safe_ratio(tp, tp + fp)
                recall = None if is_background else _safe_ratio(tp, tp + fn)
                dice = None if is_background else _safe_ratio(2 * tp, 2 * tp + fp + fn)
                row = {
                    "label_index": label_index,
                    "is_background": is_background,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_area": pred_area,
                    "label_area": label_area,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "dice": dice,
                }
                rows.append(row)

                Image.fromarray(prediction.astype(np.uint8) * 255).save(
                    prediction_dir / f"{label_index}.png"
                )
                if save_visualizations:
                    array_index = label_index - 1
                    save_visualization(
                        visualization_dir / f"{label_index}.png",
                        label_index,
                        arrays["intensity"][array_index],
                        arrays["depth"][array_index],
                        arrays["local_depth_edge"][array_index],
                        label,
                        prediction,
                        row,
                    )

    fieldnames = [
        "label_index",
        "is_background",
        "tp",
        "fp",
        "fn",
        "pred_area",
        "label_area",
        "iou",
        "precision",
        "recall",
        "dice",
    ]
    with (output_dir / "test_per_image_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    foreground_rows = [row for row in rows if not row["is_background"]]
    background_rows = [row for row in rows if row["is_background"]]
    foreground_ious = np.array([row["iou"] for row in foreground_rows], dtype=np.float64)
    background_fp_pixels = sum(row["fp"] for row in background_rows)
    image_pixels = int(np.prod(dataset[0][1].shape))
    summary = {
        "test_images": len(rows),
        "foreground_images": len(foreground_rows),
        "background_images": len(background_rows),
        "foreground_mean_image_iou": float(foreground_ious.mean()) if len(foreground_ious) else None,
        "foreground_median_image_iou": float(np.median(foreground_ious)) if len(foreground_ious) else None,
        "foreground_iou_std": float(foreground_ious.std()) if len(foreground_ious) else None,
        "background_false_positive_frames": sum(row["fp"] > 0 for row in background_rows),
        "background_false_positive_frame_rate": _safe_ratio(
            sum(row["fp"] > 0 for row in background_rows),
            len(background_rows),
        ),
        "background_false_positive_pixels": int(background_fp_pixels),
        "background_mean_pred_area": _safe_ratio(background_fp_pixels, len(background_rows)),
        "background_pixel_false_positive_rate": _safe_ratio(
            background_fp_pixels,
            len(background_rows) * image_pixels,
        ),
        "visualizations_saved": bool(save_visualizations),
        "visualization_dir": str(visualization_dir) if save_visualizations else None,
        "prediction_dir": str(prediction_dir),
    }
    with (output_dir / "test_detailed_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
