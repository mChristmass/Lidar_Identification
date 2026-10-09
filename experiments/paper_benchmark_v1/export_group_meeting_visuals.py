"""Export development-set figures for the CMX-to-current group meeting.

Inference/visualization only: neither final test set is read.
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .corruptions_v2 import FAMILIES, corrupt
from .data import PrivatePaperDataset, read_manifest, robust_depth
from .diagnose_phase1 import DiagnosticData
from .fusion_pilot_model import FusionPilotCMX
from .model import CMXPaper
from .noise_model import NoiseCMX
from .observation_model import ObservationCMX
from .reliability_model import ReliabilityCMX
from experiments.t8_transfer_statistics.model_transfer import T8TransferNet

IGNORE = 255


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def get_font(size):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                      "C:/Windows/Fonts/arial.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def color_palette(classes):
    result = np.zeros((256, 3), np.uint8)
    result[IGNORE] = (20, 20, 20)
    for index in range(classes):
        rgb = colorsys.hsv_to_rgb((index * .61803398875) % 1, .68, .95)
        result[index] = np.rint(np.asarray(rgb) * 255).astype(np.uint8)
    if classes == 2:
        result[0], result[1] = (0, 0, 0), (255, 225, 45)
    return result


def gray(value):
    array = np.nan_to_num(np.asarray(value, np.float32))
    if array.size and (array.min() < 0 or array.max() > 1):
        valid, output = np.isfinite(array) & (array > 0), np.zeros_like(array)
        if valid.any():
            low, high = np.percentile(array[valid], (1, 99))
            if high > low:
                output[valid] = np.clip((array[valid] - low) / (high - low), 0, 1)
        array = output
    array = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(array, "L").convert("RGB")


def label_image(value, classes):
    return Image.fromarray(color_palette(classes)[np.asarray(value, np.uint8)], "RGB")


def make_panel(rows, columns, row_names, destination, cell=(220, 165)):
    if len(rows) != len(row_names) or any(len(row) != len(columns) for row in rows):
        raise ValueError("Panel shape mismatch")
    left, top, gap = 250, 42, 8
    canvas = Image.new("RGB", (left + len(columns) * (cell[0] + gap),
                               top + len(rows) * (cell[1] + 42)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(columns):
        draw.text((left + index * (cell[0] + gap), 10), str(name),
                  fill="black", font=get_font(18))
    for row_index, (images, name) in enumerate(zip(rows, row_names)):
        y = top + row_index * (cell[1] + 42)
        draw.multiline_text((5, y + 3), str(name), fill="black", font=get_font(14), spacing=3)
        for column, image in enumerate(images):
            image = ImageOps.contain(image.convert("RGB"), cell, Image.Resampling.NEAREST)
            tile = Image.new("RGB", cell, (235, 235, 235))
            tile.paste(image, ((cell[0] - image.width) // 2, (cell[1] - image.height) // 2))
            canvas.paste(tile, (left + column * (cell[0] + gap), y))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)


def boundary_fraction(target):
    target = np.asarray(target)
    valid = target != IGNORE
    horizontal = (target[:, 1:] != target[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (target[1:] != target[:-1]) & valid[1:] & valid[:-1]
    return float((horizontal.sum() + vertical.sum()) / max(valid.sum(), 1))


def spread_indices(values, count):
    order = np.argsort(values)
    candidates = [int(order[0]), int(order[-1]), int(order[len(order) // 2])]
    candidates += np.linspace(0, len(values) - 1, count, dtype=int).tolist()
    result = []
    for index in candidates:
        if index not in result:
            result.append(index)
        if len(result) == count:
            break
    return result


def public_arrays(root, row, family="clean", severity="medium"):
    appearance = np.asarray(Image.open(root / row["intensity"]), np.float32)
    depth = np.asarray(Image.open(root / row["depth"]), np.float32)
    target = np.asarray(Image.open(root / row["label"]), np.int64)
    appearance, depth, valid, added = corrupt(
        appearance, depth, family, severity, row["sample_id"])
    inputs = torch.from_numpy(np.stack((appearance, robust_depth(depth, valid),
                                        valid.astype(np.float32))).copy())
    return inputs, target, added


def export_data(args, audit):
    public = read_manifest(args.nyuv2_dir / "manifests/dev.jsonl")
    scores = []
    for row in public:
        target = np.asarray(Image.open(args.nyuv2_dir / row["label"]))
        depth = np.asarray(Image.open(args.nyuv2_dir / row["depth"]))
        scores.append(boundary_fraction(target) + float((depth <= 0).mean()))
    chosen = spread_indices(scores, 4)
    rows, names = [], []
    for index in chosen:
        row = public[index]
        depth = np.asarray(Image.open(args.nyuv2_dir / row["depth"]), np.float32)
        target = np.asarray(Image.open(args.nyuv2_dir / row["label"]), np.uint8)
        rows.append([Image.open(args.nyuv2_dir / row["rgb"]).convert("RGB"),
                     Image.open(args.nyuv2_dir / row["intensity"]).convert("RGB"),
                     gray(depth), gray(depth > 0), label_image(target, 40)])
        names.append(f'NYU {row["sample_id"]}')
    make_panel(rows, ["RGB", "grayscale", "raw depth", "valid mask", "NYU40 GT"], names,
               args.output_dir / "01_data/nyuv2_processing.png")

    private = PrivatePaperDataset(args.private_dir,
                                  args.private_dir / "paper_split_v1/dev_indices.npy")
    items = [private[index] for index in range(len(private))]
    scores = [float(item["labels"].float().mean()) +
              boundary_fraction(item["labels"].numpy()) for item in items]
    chosen_private = spread_indices(scores, 6)
    rows, names = [], []
    for index in chosen_private:
        item = items[index]
        x, y = item["inputs"].numpy(), item["labels"].numpy()
        rows.append([gray(x[0]), gray(x[1]), gray(x[2]), label_image(y, 2)])
        names.append(f'frame {item["sample_id"]}')
    make_panel(rows, ["intensity", "depth", "valid mask", "binary GT"], names,
               args.output_dir / "01_data/private_processing.png", cell=(190, 190))
    audit["dataset_overview"] = {
        "nyuv2": [public[i]["sample_id"] for i in chosen],
        "private": [items[i]["sample_id"] for i in chosen_private],
    }


def export_degradation(args, audit):
    rows = read_manifest(args.nyuv2_dir / "manifests/dev.jsonl")
    scores = [boundary_fraction(np.asarray(Image.open(args.nyuv2_dir / row["label"])))
              for row in rows]
    row = rows[int(np.argmax(scores))]
    output = []
    for family in ("clean", *FAMILIES):
        x, _, added = public_arrays(args.nyuv2_dir, row, family, "heavy")
        output.append([gray(x[0]), gray(x[1]), gray(x[2]), gray(added)])
    make_panel(output, ["appearance", "depth", "valid", "added holes"],
               ["clean", *FAMILIES],
               args.output_dir / "02_degradation/heavy_families.png", cell=(190, 143))
    output = []
    for severity in ("light", "medium", "heavy"):
        x, _, added = public_arrays(args.nyuv2_dir, row, "joint", severity)
        output.append([gray(x[0]), gray(x[1]), gray(x[2]), gray(added)])
    make_panel(output, ["appearance", "depth", "valid", "added holes"],
               ["joint light", "joint medium", "joint heavy"],
               args.output_dir / "02_degradation/joint_severity.png")
    audit["degradation_sample_id"] = row["sample_id"]


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required checkpoint missing: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in state:
        raise ValueError(f"Checkpoint has no model state: {path}")
    return state["model"]


def public_models(args, group):
    root = args.runs_dir
    if group == "mixed":
        specs = [
            ("clean", lambda: CMXPaper("CMX_B2", 40, 640, 256),
             root / "mixed_control_v1/clean/last.pth"),
            ("mixed_R0", lambda: CMXPaper("CMX_B2", 40, 640, 256),
             root / "mixed_control_v1/mixed/last.pth"),
        ]
    elif group == "fusion":
        specs = [
            ("R0", lambda: CMXPaper("CMX_B2", 40, 640, 256),
             root / "mixed_control_v1/mixed/last.pth"),
            ("R3", lambda: ReliabilityCMX("R3"),
             root / "reliability_pilot_v1/R3/last.pth"),
            ("F2", lambda: FusionPilotCMX("F2"),
             root / "fusion_window_v1/F2_seed42/last.pth"),
        ]
    elif group == "repair":
        specs = [
            ("R0", lambda: CMXPaper("CMX_B2", 40, 640, 256),
             root / "mixed_control_v1/mixed/last.pth"),
            ("D1", lambda: NoiseCMX("D1"), root / "noise_window_v1/D1_seed42/last.pth"),
            ("S2", lambda: NoiseCMX("S2"), root / "repair_window_v1/S2_seed42/last.pth"),
        ]
    elif group == "observation":
        specs = [(method, lambda method=method: ObservationCMX(method),
                  root / f"observation_window_v1/nyuv2/{method}_seed42/last.pth")
                 for method in ("B0", "L", "E", "C")]
    else:
        raise ValueError(group)
    return specs


def private_models(args):
    root = args.runs_dir / "observation_window_v1/private"
    return [(method, lambda method=method: ObservationCMX(method, "CMX_B1", 2, 128, 256),
             root / f"{method}_seed42/last.pth") for method in ("B0", "L", "E", "C")]


def private_baseline_models(args):
    root = args.runs_dir / "phase1_v1/private/tune"
    t8_path = root / "T8/seed_42/best.pth"
    cmx_path = root / "CMX_B1/seed_42/best.pth"
    t8_state = torch.load(t8_path, map_location="cpu", weights_only=False)
    cmx_state = torch.load(cmx_path, map_location="cpu", weights_only=False)
    t8_channels = t8_state["args"]["t8_channels"]
    decoder_dim = cmx_state["args"]["decoder_dim"]
    del t8_state, cmx_state
    return [("T8", lambda: T8TransferNet("T0_R1_REPRO", base_channels=t8_channels,
                                           num_classes=2), t8_path),
            ("CMX_B1", lambda: CMXPaper("CMX_B1", 2, 128, decoder_dim), cmx_path)]


def infer(models, items, device):
    predictions = {}
    for name, model_factory, path in models:
        model = model_factory()
        model.load_state_dict(load_checkpoint(path), strict=True)
        model.to(device).eval()
        outputs = []
        with torch.inference_mode():
            for item in items:
                logits = model(item["inputs"][None].to(device))["logits"]
                outputs.append(logits.argmax(1)[0].cpu().numpy().astype(np.uint8))
        predictions[name] = outputs
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return predictions


def image_score(prediction, target, classes):
    target = np.asarray(target)
    valid = target != IGNORE
    if not valid.any():
        return 0.0
    if classes == 2:
        intersection = np.logical_and(prediction == 1, target == 1).sum()
        union = np.logical_or(prediction == 1, target == 1).sum()
        return float(intersection / union) if union else 1.0
    matrix = np.bincount(target[valid] * classes + prediction[valid],
                         minlength=classes * classes).reshape(classes, classes)
    union = matrix.sum(0) + matrix.sum(1) - np.diag(matrix)
    present = union > 0
    return float(np.mean(np.diag(matrix)[present] / union[present])) if present.any() else 0.0


def selected_cases(predictions, items, baseline, classes, count=3):
    scores = {name: np.asarray([image_score(pred, item["labels"].numpy(), classes)
                                for pred, item in zip(values, items)])
              for name, values in predictions.items()}
    alternatives = [name for name in predictions if name != baseline]
    stack = np.stack([scores[name] for name in alternatives])
    all_scores = np.stack(list(scores.values()))
    spread = all_scores.max(0) - all_scores.min(0)
    candidates = [int(np.argmax(stack.max(0) - scores[baseline])),
                  int(np.argmin(stack.min(0) - scores[baseline])), int(np.argmax(spread))]
    selected = []
    for index in candidates + np.argsort(spread)[::-1].astype(int).tolist():
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    return selected, scores


def export_public_group(args, group, condition, audit):
    family, severity = condition
    dataset = DiagnosticData(args.nyuv2_dir, family, severity)
    items = [dataset[index] for index in range(len(dataset))]
    predictions = infer(public_models(args, group), items, args.device)
    baseline = {"observation": "B0", "mixed": "clean"}.get(group, "R0")
    selected, scores = selected_cases(predictions, items, baseline, 40)
    model_names = list(predictions); rows, names = [], []
    for index in selected:
        item = items[index]; x = item["inputs"].numpy()
        target = item["labels"].numpy().astype(np.uint8)
        rows.append([gray(x[0]), gray(x[1]), label_image(target, 40)] +
                    [label_image(predictions[name][index], 40) for name in model_names])
        values = " ".join(f"{name}:{scores[name][index]:.3f}" for name in model_names)
        names.append(f'{item["sample_id"]}\n{values}')
    condition_name = f"{family}_{severity}"
    make_panel(rows, ["appearance", "depth", "GT", *model_names], names,
               args.output_dir / f"03_predictions/public_{group}_{condition_name}.png",
               cell=(190, 143))
    audit[f"public_{group}_{condition_name}"] = {
        "sample_ids": [items[index]["sample_id"] for index in selected],
        "per_image_miou": {name: [float(scores[name][index]) for index in selected]
                            for name in model_names},
        "selection": "largest improvement, degradation, and disagreement; dev only",
    }


def export_private_group(args, group, audit):
    dataset = PrivatePaperDataset(args.private_dir,
                                  args.private_dir / "paper_split_v1/dev_indices.npy")
    all_items = [dataset[index] for index in range(len(dataset))]
    # Empty-GT frames yield foreground-IoU extremes of 0/1 and visually empty
    # panels. They are useful for the aggregate FP metric, but not for a
    # target-segmentation qualitative comparison.
    items = [item for item in all_items if item["labels"].bool().any()]
    if len(items) < 4:
        raise RuntimeError("Fewer than four positive private dev samples")
    models = private_models(args) if group == "observation" else private_baseline_models(args)
    baseline = "B0" if group == "observation" else "T8"
    predictions = infer(models, items, args.device)
    selected, scores = selected_cases(predictions, items, baseline, 2, count=4)
    model_names = list(predictions); rows, names = [], []
    for index in selected:
        item = items[index]; x = item["inputs"].numpy()
        target = item["labels"].numpy().astype(np.uint8)
        rows.append([gray(x[0]), gray(x[1]), label_image(target, 2)] +
                    [label_image(predictions[name][index], 2) for name in model_names])
        values = " ".join(f"{name}:{scores[name][index]:.3f}" for name in model_names)
        names.append(f'frame {item["sample_id"]}\n{values}')
    make_panel(rows, ["intensity", "depth", "GT", *model_names], names,
               args.output_dir / f"03_predictions/private_{group}.png", cell=(180, 180))
    audit[f"private_{group}"] = {
        "sample_ids": [items[index]["sample_id"] for index in selected],
        "per_image_foreground_iou": {
            name: [float(scores[name][index]) for index in selected] for name in model_names},
        "selection": "largest improvement, degradation, and disagreement; dev only",
    }


def execute(args):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    required = [args.private_dir / "paper_split_v1/dev_indices.npy"]
    if not args.refresh_private:
        required.append(args.nyuv2_dir / "manifests/dev.jsonl")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required data missing:\n" + "\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "selection_manifest.json"
    if args.refresh_private and manifest.is_file():
        audit = json.loads(manifest.read_text(encoding="utf-8"))
        audit["device"] = args.device
    else:
        audit = {"scope": "frozen development sets only; no final-test access",
                 "device": args.device, "outputs": []}
    if args.refresh_private:
        for group in ("baseline", "observation"):
            print(f"VISUAL_STEP private {group}", flush=True)
            export_private_group(args, group, audit)
        audit["outputs"] = sorted(str(path.relative_to(args.output_dir)).replace("\\", "/")
                                  for path in args.output_dir.rglob("*.png"))
        save_json(manifest, audit)
        save_json(args.output_dir / "complete.json",
                  {"figures": len(audit["outputs"]), "final_test_access": False})
        print(f'VISUAL_COMPLETE figures={len(audit["outputs"])} output={args.output_dir}', flush=True)
        return
    print("VISUAL_STEP dataset_overviews", flush=True)
    export_data(args, audit)
    print("VISUAL_STEP degradations", flush=True)
    export_degradation(args, audit)
    jobs = [("mixed", ("joint", "heavy")),
            ("fusion", ("joint", "heavy")),
            ("repair", ("photon_proxy", "heavy")),
            ("repair", ("joint", "heavy")),
            ("observation", ("clean", "medium")),
            ("observation", ("joint", "heavy"))]
    for index, (group, condition) in enumerate(jobs, 1):
        print(f"VISUAL_STEP public {index}/{len(jobs)} {group} {condition[0]}_{condition[1]}",
              flush=True)
        export_public_group(args, group, condition, audit)
    for group in ("baseline", "observation"):
        print(f"VISUAL_STEP private {group}", flush=True)
        export_private_group(args, group, audit)
    audit["outputs"] = sorted(str(path.relative_to(args.output_dir)).replace("\\", "/")
                              for path in args.output_dir.rglob("*.png"))
    save_json(args.output_dir / "selection_manifest.json", audit)
    save_json(args.output_dir / "complete.json",
              {"figures": len(audit["outputs"]), "final_test_access": False})
    print(f'VISUAL_COMPLETE figures={len(audit["outputs"])} output={args.output_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyuv2-dir", type=Path,
                        default=Path("data/public_semseg/nyuv2/processed"))
    parser.add_argument("--private-dir", type=Path, default=Path("data/new_data/merged"))
    parser.add_argument("--runs-dir", type=Path, default=Path("data/paper_benchmark_runs"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/paper_benchmark_runs/group_meeting_visuals_v1"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refresh-private", action="store_true",
                        help="Regenerate only positive-target private comparison panels")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
