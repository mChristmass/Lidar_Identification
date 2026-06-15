import csv
import json
import time
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


def binary_boundary(mask, radius=1):
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=radius)
    eroded = cv2.erode(mask, kernel, iterations=radius)
    return dilated != eroded


def boundary_counts(label, prediction, tolerance=2):
    label_boundary = binary_boundary(label)
    prediction_boundary = binary_boundary(prediction)
    kernel = np.ones((3, 3), dtype=np.uint8)
    label_band = cv2.dilate(
        label_boundary.astype(np.uint8), kernel, iterations=tolerance
    ).astype(bool)
    prediction_band = cv2.dilate(
        prediction_boundary.astype(np.uint8), kernel, iterations=tolerance
    ).astype(bool)
    matched_prediction = int((prediction_boundary & label_band).sum())
    matched_label = int((label_boundary & prediction_band).sum())
    return (
        matched_prediction,
        int(prediction_boundary.sum()),
        matched_label,
        int(label_boundary.sum()),
    )


def save_gate_visualization(
    output_path,
    label_index,
    intensity,
    depth,
    edge,
    reliability,
    filtered_edge,
    label,
    prediction,
    row,
):
    overlay = _error_overlay(intensity, label, prediction)
    panels = [
        _panel(_display_gray(intensity), "Intensity"),
        _panel(_display_depth(depth), "Raw depth"),
        _panel(np.clip(edge * 255.0, 0, 255).astype(np.uint8), "Local depth edge"),
        _panel(np.clip(reliability * 255.0, 0, 255).astype(np.uint8), "Edge reliability"),
        _panel(np.clip(filtered_edge * 255.0, 0, 255).astype(np.uint8), "Filtered edge"),
        _panel(label.astype(np.uint8) * 255, "Ground truth"),
        _panel(prediction.astype(np.uint8) * 255, "Prediction"),
        _panel(overlay, "TP green | FP red | FN blue"),
    ]
    first_row = np.hstack(panels[:4])
    second_row = np.hstack(panels[4:])
    title = np.full((42, first_row.shape[1], 3), 255, dtype=np.uint8)
    metric_text = (
        f"IoU {row['iou']:.3f}" if row["iou"] is not None
        else f"Background | FP pixels {row['fp']}"
    )
    cv2.putText(
        title,
        f"Frame {label_index} | {metric_text} | Boundary F1 {row['boundary_f1']:.3f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack([title, first_row, second_row]))


def measure_latency(model, sample, device, warmup=20, repeats=100):
    model.eval()
    sample = sample.to(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def estimate_conv_flops(model, sample):
    total = 0
    hooks = []

    def hook(module, inputs, output):
        nonlocal total
        output_tensor = output if torch.is_tensor(output) else output[0]
        batch, out_channels, out_height, out_width = output_tensor.shape
        kernel_height, kernel_width = module.kernel_size
        in_channels = module.in_channels
        groups = module.groups
        total += (
            batch
            * out_channels
            * out_height
            * out_width
            * (in_channels // groups)
            * kernel_height
            * kernel_width
            * 2
        )

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(sample)
    for handle in hooks:
        handle.remove()
    return int(total)


def evaluate_and_report_reliability_model(
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
    visualization_dir = output_dir / "test_visualizations"
    prediction_dir.mkdir(parents=True, exist_ok=True)
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
    reliability_positive_sum = 0.0
    reliability_positive_count = 0
    reliability_negative_sum = 0.0
    reliability_negative_count = 0
    model.eval()

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            samples = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            x = torch.stack([sample[0] for sample in samples]).to(device)
            labels = torch.stack([sample[1] for sample in samples]).numpy().astype(bool)
            gate_targets = torch.stack([sample[2] for sample in samples]).numpy().astype(bool)
            gate_valid = torch.stack([sample[3] for sample in samples]).numpy().astype(bool)
            if is_baseline:
                logits = model(x)
                reliability = torch.ones_like(x[:, 3])
                filtered_edge = x[:, 3]
            else:
                output = model(x, return_aux=True)
                logits = output["logits"]
                reliability = output["reliability"]
                filtered_edge = output["filtered_edge"]
            probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            predictions = probabilities > threshold
            reliability = reliability.cpu().numpy()
            filtered_edge = filtered_edge.cpu().numpy()

            for offset, (label, prediction) in enumerate(zip(labels, predictions)):
                dataset_position = start + offset
                label_index = int(dataset.indices[dataset_position])
                tp = int((prediction & label).sum())
                fp = int((prediction & ~label).sum())
                fn = int((~prediction & label).sum())
                boundary_values = boundary_counts(label, prediction)
                aggregate_boundary += np.asarray(boundary_values, dtype=np.int64)
                label_boundary_mask = binary_boundary(label)
                prediction_boundary_mask = binary_boundary(prediction)
                boundary_intersection = int(
                    (label_boundary_mask & prediction_boundary_mask).sum()
                )
                boundary_union = int(
                    (label_boundary_mask | prediction_boundary_mask).sum()
                )
                strict_boundary_intersection += boundary_intersection
                strict_boundary_union += boundary_union
                matched_pred, pred_boundary, matched_label, label_boundary = boundary_values
                boundary_precision = _safe_ratio(matched_pred, pred_boundary)
                boundary_recall = _safe_ratio(matched_label, label_boundary)
                boundary_f1 = _safe_ratio(
                    2 * boundary_precision * boundary_recall,
                    boundary_precision + boundary_recall,
                )
                positive_mask = gate_targets[offset] & gate_valid[offset]
                negative_mask = ~gate_targets[offset] & gate_valid[offset]
                reliability_positive_sum += float(reliability[offset][positive_mask].sum())
                reliability_positive_count += int(positive_mask.sum())
                reliability_negative_sum += float(reliability[offset][negative_mask].sum())
                reliability_negative_count += int(negative_mask.sum())
                label_area = int(label.sum())
                row = {
                    "label_index": label_index,
                    "is_background": label_area == 0,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "pred_area": int(prediction.sum()),
                    "label_area": label_area,
                    "iou": None if label_area == 0 else _safe_ratio(tp, tp + fp + fn),
                    "precision": _safe_ratio(tp, tp + fp),
                    "recall": None if label_area == 0 else _safe_ratio(tp, tp + fn),
                    "dice": None if label_area == 0 else _safe_ratio(2 * tp, 2 * tp + fp + fn),
                    "boundary_precision": boundary_precision,
                    "boundary_recall": boundary_recall,
                    "boundary_f1": boundary_f1,
                    "boundary_iou": _safe_ratio(boundary_intersection, boundary_union),
                    "mean_reliability": float(reliability[offset].mean()),
                }
                rows.append(row)
                Image.fromarray(prediction.astype(np.uint8) * 255).save(
                    prediction_dir / f"{label_index}.png"
                )
                if save_visualizations:
                    array_index = label_index - 1
                    save_gate_visualization(
                        visualization_dir / f"{label_index}.png",
                        label_index,
                        arrays["intensity"][array_index],
                        arrays["depth"][array_index],
                        arrays["local_depth_edge"][array_index],
                        reliability[offset],
                        filtered_edge[offset],
                        label,
                        prediction,
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
    foreground_ious = np.asarray([row["iou"] for row in foreground_rows], dtype=np.float64)
    matched_pred, pred_boundary, matched_label, label_boundary = aggregate_boundary
    boundary_precision = _safe_ratio(matched_pred, pred_boundary)
    boundary_recall = _safe_ratio(matched_label, label_boundary)
    background_fp_pixels = sum(row["fp"] for row in background_rows)
    image_pixels = int(np.prod(dataset[0][1].shape))
    summary = {
        "test_images": len(rows),
        "foreground_images": len(foreground_rows),
        "background_images": len(background_rows),
        "foreground_mean_image_iou": float(foreground_ious.mean()) if len(foreground_ious) else None,
        "foreground_median_image_iou": float(np.median(foreground_ious)) if len(foreground_ious) else None,
        "foreground_iou_std": float(foreground_ious.std()) if len(foreground_ious) else None,
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
        "boundary_iou": _safe_ratio(
            strict_boundary_intersection, strict_boundary_union
        ),
        "mean_reliability": float(np.mean([row["mean_reliability"] for row in rows])),
        "reliability_positive_mean": _safe_ratio(
            reliability_positive_sum, reliability_positive_count
        ),
        "reliability_negative_mean": _safe_ratio(
            reliability_negative_sum, reliability_negative_count
        ),
        "reliability_separation": (
            _safe_ratio(reliability_positive_sum, reliability_positive_count)
            - _safe_ratio(reliability_negative_sum, reliability_negative_count)
        ),
        "visualizations_saved": bool(save_visualizations),
        "visualization_dir": str(visualization_dir) if save_visualizations else None,
        "prediction_dir": str(prediction_dir),
    }
    with (output_dir / "test_detailed_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
