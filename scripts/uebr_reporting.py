import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.new_data_test_reporting import (
    _display_depth,
    _display_gray,
    _error_overlay,
    _panel,
    _safe_ratio,
)
from scripts.reliability_gate_reporting import binary_boundary, boundary_counts


def save_uebr_visualization(
    output_path,
    label_index,
    intensity,
    depth,
    edge,
    uncertainty,
    delta,
    label,
    base_prediction,
    final_prediction,
    row,
):
    overlay = _error_overlay(intensity, label, final_prediction)
    panels = [
        _panel(_display_gray(intensity), "Intensity"),
        _panel(_display_depth(depth), "Raw depth"),
        _panel(np.clip(edge * 255.0, 0, 255).astype(np.uint8), "Local depth edge"),
        _panel(np.clip(uncertainty * 255.0, 0, 255).astype(np.uint8), "Base uncertainty"),
        _panel(np.clip(delta * 255.0, 0, 255).astype(np.uint8), "Delta magnitude"),
        _panel(label.astype(np.uint8) * 255, "Ground truth"),
        _panel(base_prediction.astype(np.uint8) * 255, "Base prediction"),
        _panel(final_prediction.astype(np.uint8) * 255, "Final prediction"),
    ]
    first_row = np.hstack(panels[:4])
    second_row = np.hstack(panels[4:])
    overlay_row = _panel(overlay, "Final: TP green | FP red | FN blue")
    padding = np.full(
        (overlay_row.shape[0], first_row.shape[1] - overlay_row.shape[1], 3),
        245,
        dtype=np.uint8,
    )
    overlay_row = np.hstack([overlay_row, padding])
    title = np.full((42, first_row.shape[1], 3), 255, dtype=np.uint8)
    final_text = "BG" if row["iou"] is None else f"IoU {row['iou']:.3f}"
    cv2.putText(
        title,
        f"Frame {label_index} | {final_text} | Base-to-final {row['iou_improvement']:+.3f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack([title, first_row, second_row, overlay_row]))


def evaluate_and_report_uebr(
    model,
    dataset,
    raw_items,
    device,
    threshold,
    output_dir,
    batch_size,
    save_visualizations=True,
    is_baseline=False,
):
    output_dir = Path(output_dir)
    prediction_dir = output_dir / "test_predictions"
    base_prediction_dir = output_dir / "test_base_predictions"
    visualization_dir = output_dir / "test_visualizations"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    base_prediction_dir.mkdir(parents=True, exist_ok=True)
    if save_visualizations:
        visualization_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        name: np.load(Path(raw_items[name]), mmap_mode="r")
        for name in ("intensity", "depth", "local_depth_edge")
    }
    rows = []
    aggregate_boundary = np.zeros(4, dtype=np.int64)
    strict_boundary_intersection = 0
    strict_boundary_union = 0
    model.eval()

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            x = torch.stack([sample[0] for sample in samples]).to(device)
            labels = torch.stack([sample[1] for sample in samples]).numpy().astype(bool)
            if is_baseline:
                final_logits = model(x)
                base_logits = final_logits
                uncertainty = torch.zeros_like(x[:, 0])
                delta = torch.zeros_like(x[:, 0])
            else:
                output = model(x, return_aux=True)
                final_logits = output["logits"]
                base_logits = output["base_logits"]
                uncertainty = output["uncertainty"]
                delta = output["delta_logits"].abs().mean(dim=1)

            final_probs = torch.softmax(final_logits, dim=1)[:, 1].cpu().numpy()
            base_probs = torch.softmax(base_logits, dim=1)[:, 1].cpu().numpy()
            final_predictions = final_probs > threshold
            base_predictions = base_probs > threshold
            uncertainty = uncertainty.cpu().numpy()
            delta = delta.cpu().numpy()

            for offset, (label, base_prediction, final_prediction) in enumerate(
                zip(labels, base_predictions, final_predictions)
            ):
                position = start + offset
                label_index = int(dataset.indices[position])
                tp = int((final_prediction & label).sum())
                fp = int((final_prediction & ~label).sum())
                fn = int((~final_prediction & label).sum())
                label_area = int(label.sum())
                final_iou = None if label_area == 0 else _safe_ratio(tp, tp + fp + fn)
                base_tp = int((base_prediction & label).sum())
                base_fp = int((base_prediction & ~label).sum())
                base_fn = int((~base_prediction & label).sum())
                base_iou = None if label_area == 0 else _safe_ratio(
                    base_tp, base_tp + base_fp + base_fn
                )

                boundary_values = boundary_counts(label, final_prediction)
                aggregate_boundary += np.asarray(boundary_values, dtype=np.int64)
                matched_pred, pred_boundary, matched_label, label_boundary = boundary_values
                boundary_precision = _safe_ratio(matched_pred, pred_boundary)
                boundary_recall = _safe_ratio(matched_label, label_boundary)
                boundary_f1 = _safe_ratio(
                    2 * boundary_precision * boundary_recall,
                    boundary_precision + boundary_recall,
                )
                label_boundary_mask = binary_boundary(label)
                prediction_boundary_mask = binary_boundary(final_prediction)
                intersection = int((label_boundary_mask & prediction_boundary_mask).sum())
                union = int((label_boundary_mask | prediction_boundary_mask).sum())
                strict_boundary_intersection += intersection
                strict_boundary_union += union

                row = {
                    "label_index": label_index,
                    "is_background": label_area == 0,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_area": int(final_prediction.sum()),
                    "label_area": label_area,
                    "iou": final_iou,
                    "base_iou": base_iou,
                    "iou_improvement": 0.0 if final_iou is None else final_iou - base_iou,
                    "precision": _safe_ratio(tp, tp + fp),
                    "recall": None if label_area == 0 else _safe_ratio(tp, tp + fn),
                    "dice": None if label_area == 0 else _safe_ratio(2 * tp, 2 * tp + fp + fn),
                    "boundary_precision": boundary_precision,
                    "boundary_recall": boundary_recall,
                    "boundary_f1": boundary_f1,
                    "boundary_iou": _safe_ratio(intersection, union),
                    "mean_uncertainty": float(uncertainty[offset].mean()),
                    "mean_delta": float(delta[offset].mean()),
                }
                rows.append(row)
                Image.fromarray(final_prediction.astype(np.uint8) * 255).save(
                    prediction_dir / f"{label_index}.png"
                )
                Image.fromarray(base_prediction.astype(np.uint8) * 255).save(
                    base_prediction_dir / f"{label_index}.png"
                )
                if save_visualizations:
                    array_index = label_index - 1
                    delta_display = delta[offset]
                    delta_display = delta_display / max(float(delta_display.max()), 1e-6)
                    save_uebr_visualization(
                        visualization_dir / f"{label_index}.png",
                        label_index,
                        arrays["intensity"][array_index],
                        arrays["depth"][array_index],
                        arrays["local_depth_edge"][array_index],
                        uncertainty[offset],
                        delta_display,
                        label,
                        base_prediction,
                        final_prediction,
                        row,
                    )

    with (output_dir / "test_per_image_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    foreground_rows = [row for row in rows if not row["is_background"]]
    background_rows = [row for row in rows if row["is_background"]]
    improvements = np.asarray(
        [row["iou_improvement"] for row in foreground_rows], dtype=np.float64
    )
    matched_pred, pred_boundary, matched_label, label_boundary = aggregate_boundary
    boundary_precision = _safe_ratio(matched_pred, pred_boundary)
    boundary_recall = _safe_ratio(matched_label, label_boundary)
    background_fp_pixels = sum(row["fp"] for row in background_rows)
    image_pixels = int(np.prod(dataset[0][1].shape))
    summary = {
        "test_images": len(rows),
        "foreground_images": len(foreground_rows),
        "background_images": len(background_rows),
        "foreground_mean_image_iou": float(np.mean([row["iou"] for row in foreground_rows])),
        "foreground_mean_base_iou": float(np.mean([row["base_iou"] for row in foreground_rows])),
        "mean_image_iou_improvement": float(improvements.mean()),
        "improved_foreground_images": int((improvements > 1e-9).sum()),
        "worse_foreground_images": int((improvements < -1e-9).sum()),
        "improvement_win_rate": float((improvements > 1e-9).mean()),
        "background_false_positive_frame_rate": _safe_ratio(
            sum(row["fp"] > 0 for row in background_rows), len(background_rows)
        ),
        "background_mean_pred_area": _safe_ratio(background_fp_pixels, len(background_rows)),
        "background_pixel_false_positive_rate": _safe_ratio(
            background_fp_pixels, len(background_rows) * image_pixels
        ),
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": _safe_ratio(
            2 * boundary_precision * boundary_recall,
            boundary_precision + boundary_recall,
        ),
        "boundary_iou": _safe_ratio(strict_boundary_intersection, strict_boundary_union),
        "mean_uncertainty": float(np.mean([row["mean_uncertainty"] for row in rows])),
        "mean_delta": float(np.mean([row["mean_delta"] for row in rows])),
        "visualizations_saved": bool(save_visualizations),
        "visualization_dir": str(visualization_dir) if save_visualizations else None,
        "prediction_dir": str(prediction_dir),
        "base_prediction_dir": str(base_prediction_dir),
    }
    with (output_dir / "test_detailed_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
