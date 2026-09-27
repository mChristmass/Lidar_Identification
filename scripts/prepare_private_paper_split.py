from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


FIRST_END = 390
EXPECTED_FRAMES = 1390

# These assignments were selected using acquisition provenance and label-category
# counts only. No model result was consulted. The rare double-target recording is
# kept in training; the final set consequently has only one double-target frame.
DEV_FOLDERS = {
    "depth_20260129_191730_8000frames",
    "depth_20260129_194946_2000frames",
    "depth_20260129_204454_2000frames",
}
FINAL_TEST_FOLDERS = {
    "depth_20260129_195709_8000frames",
    "depth_20260129_195059_2000frames",
    "depth_20260129_200632_10000frames",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(paths: list[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.as_posix()):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def group_id(group: str, source_folder: str) -> str:
    return f"second0612__{group}__{source_folder}"


def split_for(source_batch: str, source_folder: str) -> str:
    if source_batch == "first":
        return "train"
    if source_folder in DEV_FOLDERS:
        return "dev"
    if source_folder in FINAL_TEST_FOLDERS:
        return "final_test"
    return "train"


def source_index_lookup(source_root: Path, group: str, folder: str) -> dict[int, int]:
    path = source_root / group / folder / "source_indices.csv"
    if not path.is_file():
        return {}
    result = {}
    for row in read_csv(path):
        result[int(row["selected_index_1based"])] = int(row["source_frame_index_1based"])
    return result


def label_information(path: Path) -> tuple[int, str]:
    values = np.asarray(Image.open(path))
    unique = set(np.unique(values).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Label is not binary 0/1: {path}; values={sorted(unique)}")
    foreground = int((values > 0).sum())
    return foreground, sha256_file(path)


def build_metadata(merged_dir: Path, second_mapping_path: Path, source_root: Path) -> list[dict]:
    merged = read_csv(merged_dir / "frame_mapping.csv")
    categories = {
        int(row["frame"]): row
        for row in read_csv(merged_dir / "group_folds" / "frame_category_table.csv")
    }
    second = {int(row["merged_frame_1based"]): row for row in read_csv(second_mapping_path)}
    if len(merged) != EXPECTED_FRAMES or set(categories) != set(range(1, EXPECTED_FRAMES + 1)):
        raise ValueError("Expected exactly 1390 mapped and categorized frames")
    if set(second) != set(range(1, EXPECTED_FRAMES - FIRST_END + 1)):
        raise ValueError("Expected second0612 mapping to contain local frames 1..1000")

    source_indices: dict[tuple[str, str], dict[int, int]] = {}
    rows = []
    for expected_frame, mapping in enumerate(merged, 1):
        frame = int(mapping["merged_frame_1based"])
        if frame != expected_frame:
            raise ValueError(f"Non-contiguous merged mapping at row {expected_frame}: {frame}")
        category = categories[frame]
        label_path = merged_dir / "label" / mapping["label_png"]
        foreground, label_hash = label_information(label_path)

        if frame <= FIRST_END:
            if mapping["source_batch"] != "first":
                raise ValueError(f"Frame {frame} should belong to first batch")
            source_group = ""
            source_folder = ""
            acquisition_group = "legacy_first_unresolved"
            provenance = "batch_only_unresolved_training_only"
            raw_source_frame = ""
        else:
            local_frame = frame - FIRST_END
            detail = second[local_frame]
            if mapping["source_batch"] != "second0612":
                raise ValueError(f"Frame {frame} should belong to second0612")
            if int(mapping["source_frame_1based"]) != local_frame:
                raise ValueError(f"Outer/inner mapping mismatch at frame {frame}")
            source_group = detail["group"]
            source_folder = detail["source_folder"]
            acquisition_group = group_id(source_group, source_folder)
            provenance = "verified_source_recording"
            key = (source_group, source_folder)
            if key not in source_indices:
                source_indices[key] = source_index_lookup(source_root, *key)
            raw_source_frame = source_indices[key].get(int(detail["source_frame_1based"]), "")

        split = split_for(mapping["source_batch"], source_folder)
        category_says_background = int(category["is_background"]) == 1
        label_is_background = foreground == 0
        effective_primary_class = "background" if label_is_background else category["primary_class"]
        rows.append({
            "frame_1based": frame,
            "label_png": mapping["label_png"],
            "source_batch": mapping["source_batch"],
            "source_selected_frame_1based": mapping["source_frame_1based"],
            "source_group": source_group,
            "source_folder": source_folder,
            "source_raw_frame_1based": raw_source_frame,
            "acquisition_group_id": acquisition_group,
            "provenance_status": provenance,
            "primary_class": effective_primary_class,
            "primary_class_source_table": category["primary_class"],
            "is_occlusion": category["is_occlusion"],
            "is_background": category["is_background"],
            "is_low_quality": category["is_low_quality"],
            "is_double_target": category["is_double_target"],
            "is_tiny_target": category["is_tiny_target"],
            "foreground_pixels_current_label": foreground,
            "category_background_matches_label": int(category_says_background == label_is_background),
            "label_sha256": label_hash,
            "split": split,
        })
    return rows


def split_summary(rows: list[dict]) -> dict:
    result = {}
    for split in ("train", "dev", "final_test"):
        selected = [row for row in rows if row["split"] == split]
        result[split] = {
            "frames": len(selected),
            "acquisition_groups": len({row["acquisition_group_id"] for row in selected}),
            "verified_source_recording_frames": sum(
                row["provenance_status"] == "verified_source_recording" for row in selected
            ),
            "class_counts": dict(sorted(Counter(row["primary_class"] for row in selected).items())),
        }
    return result


def validate(rows: list[dict]) -> None:
    frames = [int(row["frame_1based"]) for row in rows]
    if frames != list(range(1, EXPECTED_FRAMES + 1)):
        raise ValueError("Metadata frames must be exactly 1..1390")
    seen = defaultdict(set)
    for row in rows:
        seen[row["acquisition_group_id"]].add(row["split"])
    leaking = {group: splits for group, splits in seen.items() if len(splits) != 1}
    if leaking:
        raise ValueError(f"Acquisition groups cross splits: {leaking}")
    for split in ("train", "dev", "final_test"):
        if not any(row["split"] == split for row in rows):
            raise ValueError(f"Empty split: {split}")
    unresolved_nontrain = [
        row["frame_1based"] for row in rows
        if row["provenance_status"] != "verified_source_recording" and row["split"] != "train"
    ]
    if unresolved_nontrain:
        raise ValueError(f"Unresolved frames outside training: {unresolved_nontrain[:10]}")


def source_hashes(merged_dir: Path, second_mapping_path: Path) -> dict[str, str]:
    paths = {
        "merged/intensity.npy": merged_dir / "intensity.npy",
        "merged/depth.npy": merged_dir / "depth.npy",
        "merged/local_depth_edge.npy": merged_dir / "local_depth_edge.npy",
        "merged/frame_mapping.csv": merged_dir / "frame_mapping.csv",
        "merged/group_folds/frame_category_table.csv": (
            merged_dir / "group_folds" / "frame_category_table.csv"
        ),
        "second0612/frame_mapping.csv": second_mapping_path,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    labels = list((merged_dir / "label").glob("*.png"))
    if len(labels) != EXPECTED_FRAMES:
        raise ValueError(f"Expected 1390 labels, found {len(labels)}")
    hashes["label_tree"] = sha256_tree(labels, merged_dir)
    return hashes


def canonical_source_hashes(hashes: dict[str, str]) -> dict[str, str]:
    """Normalize schema-v1 absolute keys so frozen splits verify across hosts."""
    suffixes = (
        "merged/intensity.npy",
        "merged/depth.npy",
        "merged/local_depth_edge.npy",
        "merged/frame_mapping.csv",
        "merged/group_folds/frame_category_table.csv",
        "second0612/frame_mapping.csv",
    )
    canonical = {}
    for original, digest in hashes.items():
        if original == "label_tree":
            key = original
        else:
            normalized = original.replace("\\", "/")
            matches = [suffix for suffix in suffixes if normalized.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Unknown or ambiguous source-hash key: {original}")
            key = matches[0]
        if key in canonical and canonical[key] != digest:
            raise ValueError(f"Conflicting hashes for canonical source {key}")
        canonical[key] = digest
    return canonical


def prepare(args) -> None:
    merged_dir = args.merged_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        output_dir / "frame_acquisition_metadata.csv",
        output_dir / "train_indices.npy",
        output_dir / "dev_indices.npy",
        output_dir / "final_test_indices.npy",
        output_dir / "split_manifest.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Frozen split already exists; use verify, not prepare: {existing[0]}"
        )

    rows = build_metadata(merged_dir, args.second_mapping.resolve(), args.source_root.resolve())
    validate(rows)
    write_csv(output_dir / "frame_acquisition_metadata.csv", rows)

    for split in ("train", "dev", "final_test"):
        indices = np.asarray(
            [int(row["frame_1based"]) for row in rows if row["split"] == split], dtype=np.int64
        )
        np.save(output_dir / f"{split}_indices.npy", indices)
        (output_dir / f"{split}_indices.txt").write_text(
            "".join(f"{value}\n" for value in indices), encoding="utf-8"
        )

    verified_train = np.asarray([
        int(row["frame_1based"]) for row in rows
        if row["split"] == "train" and row["provenance_status"] == "verified_source_recording"
    ], dtype=np.int64)
    np.save(output_dir / "verified_train_indices.npy", verified_train)
    (output_dir / "verified_train_indices.txt").write_text(
        "".join(f"{value}\n" for value in verified_train), encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "status": "frozen_prospective_from_2026-09-22_not_historically_blind",
        "index_base": 1,
        "split_policy": {
            "unit": "original recording folder; never split across train/dev/final_test",
            "legacy_first": "frames 1..390 are an unresolved indivisible training-only group",
            "selection_inputs": "acquisition provenance and label-category counts only; no model results",
            "dev_source_folders": sorted(DEV_FOLDERS),
            "final_test_source_folders": sorted(FINAL_TEST_FOLDERS),
        },
        "summary": split_summary(rows),
        "category_label_mismatches": [
            {
                "frame_1based": row["frame_1based"],
                "source_table_class": row["primary_class_source_table"],
                "effective_class": row["primary_class"],
                "foreground_pixels_current_label": row["foreground_pixels_current_label"],
            }
            for row in rows if not row["category_background_matches_label"]
        ],
        "known_limitations": [
            "All 1390 frames appeared in historical experiments; final_test is frozen now but is not a never-seen blind set.",
            "Frames 1..390 lack recoverable per-recording provenance and are restricted to training.",
            "Double-target labels are concentrated in one recording kept in training; final_test contains only one double-target frame.",
            "A new acquisition holdout is still required for a genuinely blind final confirmation.",
        ],
        "source_sha256": source_hashes(merged_dir, args.second_mapping.resolve()),
    }
    save_path = output_dir / "split_manifest.json"
    save_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    output_hashes = {}
    for path in sorted(output_dir.glob("*")):
        if path.is_file() and path.name != "frozen_output_sha256.json":
            output_hashes[path.name] = sha256_file(path)
    (output_dir / "frozen_output_sha256.json").write_text(
        json.dumps(output_hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"Prepared frozen split: {output_dir}")


def verify(args) -> None:
    output_dir = args.output_dir.resolve()
    metadata_path = output_dir / "frame_acquisition_metadata.csv"
    manifest_path = output_dir / "split_manifest.json"
    frozen_hash_path = output_dir / "frozen_output_sha256.json"
    rows = read_csv(metadata_path)
    for row in rows:
        row["frame_1based"] = int(row["frame_1based"])
    validate(rows)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_sources = source_hashes(args.merged_dir.resolve(), args.second_mapping.resolve())
    expected_sources = canonical_source_hashes(manifest["source_sha256"])
    current_sources = canonical_source_hashes(current_sources)
    if current_sources != expected_sources:
        changed = sorted(
            key for key in set(current_sources) | set(expected_sources)
            if current_sources.get(key) != expected_sources.get(key)
        )
        raise RuntimeError(f"Dataset source hashes changed after split freeze: {changed}")
    frozen = json.loads(frozen_hash_path.read_text(encoding="utf-8"))
    for name, expected in frozen.items():
        actual = sha256_file(output_dir / name)
        if actual != expected:
            raise RuntimeError(f"Frozen output changed: {name}")
    print(json.dumps(split_summary(rows), ensure_ascii=False, indent=2))
    print("VERIFY_OK dataset and frozen split hashes match")


def parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[1]
    data = project / "data" / "new_data"
    p = argparse.ArgumentParser(description="Prepare or verify the frozen private paper split")
    p.add_argument("command", choices=("prepare", "verify"))
    p.add_argument("--merged-dir", type=Path, default=data / "merged")
    p.add_argument("--second-mapping", type=Path, default=data / "second0612" / "frame_mapping.csv")
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"D:\myComputer\pointsCloud\data\origin\new\generate_label\0129"),
    )
    p.add_argument("--output-dir", type=Path, default=data / "merged" / "paper_split_v1")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        verify(args)


if __name__ == "__main__":
    main()
