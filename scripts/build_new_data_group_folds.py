import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import configs.new_data.config_new_data as conf


def read_mapping(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "merged_frame_1based",
        "source_batch",
        "source_frame_1based",
        "foreground_pixels",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Mapping is missing columns: {sorted(missing)}")
    return rows


def build_contiguous_groups(rows, block_size):
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[row["source_batch"]].append(row)

    groups = []
    for batch, batch_rows in sorted(by_batch.items()):
        batch_rows.sort(key=lambda row: int(row["source_frame_1based"]))
        for start in range(0, len(batch_rows), block_size):
            chunk = batch_rows[start : start + block_size]
            indices = [int(row["merged_frame_1based"]) for row in chunk]
            foreground_count = sum(int(row["foreground_pixels"]) > 0 for row in chunk)
            groups.append(
                {
                    "group_id": f"{batch}_{start // block_size:03d}",
                    "source_batch": batch,
                    "source_start": int(chunk[0]["source_frame_1based"]),
                    "source_end": int(chunk[-1]["source_frame_1based"]),
                    "indices": indices,
                    "frames": len(indices),
                    "foreground_frames": foreground_count,
                }
            )
    return groups


def assign_groups(groups, num_folds, seed):
    rng = np.random.default_rng(seed)
    assignments = {}
    fold_stats = [
        {"frames": 0, "foreground_frames": 0, "groups": 0}
        for _ in range(num_folds)
    ]

    by_batch = defaultdict(list)
    for group in groups:
        by_batch[group["source_batch"]].append(group)

    for batch_groups in by_batch.values():
        rng.shuffle(batch_groups)
        batch_groups.sort(
            key=lambda group: (
                group["frames"],
                abs(group["foreground_frames"] / group["frames"] - 0.5),
            ),
            reverse=True,
        )
        batch_fold_stats = [
            {"frames": 0, "foreground_frames": 0}
            for _ in range(num_folds)
        ]
        batch_foreground_ratio = sum(
            group["foreground_frames"] for group in batch_groups
        ) / max(1, sum(group["frames"] for group in batch_groups))

        for group in batch_groups:
            candidate_folds = list(range(num_folds))
            rng.shuffle(candidate_folds)

            def score(fold):
                stats = batch_fold_stats[fold]
                new_frames = stats["frames"] + group["frames"]
                new_foreground = stats["foreground_frames"] + group["foreground_frames"]
                new_ratio = new_foreground / max(1, new_frames)
                return (
                    stats["frames"],
                    abs(new_ratio - batch_foreground_ratio),
                    stats["foreground_frames"],
                )

            fold = min(candidate_folds, key=score)
            assignments[group["group_id"]] = fold
            for stats in (batch_fold_stats[fold], fold_stats[fold]):
                stats["frames"] += group["frames"]
                stats["foreground_frames"] += group["foreground_frames"]
            fold_stats[fold]["groups"] += 1

    return assignments, fold_stats


def validate_and_save(
    groups,
    assignments,
    output_dir,
    num_folds,
    num_partitions,
    metadata,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_indices = sorted(index for group in groups for index in group["indices"])

    manifest = {
        **metadata,
        "folds": [],
        "groups": [
            {
                key: value
                for key, value in group.items()
                if key != "indices"
            }
            | {"fold": int(assignments[group["group_id"]])}
            for group in groups
        ],
    }

    for fold in range(num_folds):
        test_partition_ids = {
            (fold * 2) % num_partitions,
            (fold * 2 + 1) % num_partitions,
        }
        val_partition_id = (fold * 2 + 2) % num_partitions
        test_groups = {
            group_id
            for group_id, assigned_partition in assignments.items()
            if assigned_partition in test_partition_ids
        }
        val_groups = {
            group_id
            for group_id, assigned_partition in assignments.items()
            if assigned_partition == val_partition_id
        }
        train_groups = set(assignments) - test_groups - val_groups

        split_indices = {}
        for split, selected_groups in (
            ("train", train_groups),
            ("val", val_groups),
            ("test", test_groups),
        ):
            indices = sorted(
                index
                for group in groups
                if group["group_id"] in selected_groups
                for index in group["indices"]
            )
            split_indices[split] = indices
            np.save(output_dir / f"fold_{fold}_{split}_indices.npy", np.array(indices, dtype=np.int64))

        split_sets = {name: set(values) for name, values in split_indices.items()}
        if any(
            split_sets[left] & split_sets[right]
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        ):
            raise RuntimeError(f"Fold {fold} has overlapping splits.")
        if sorted(set().union(*split_sets.values())) != all_indices:
            raise RuntimeError(f"Fold {fold} does not cover all samples.")

        manifest["folds"].append(
            {
                "fold": fold,
                "test_group_partitions": sorted(test_partition_ids),
                "validation_group_partition": val_partition_id,
                "train_samples": len(split_indices["train"]),
                "val_samples": len(split_indices["val"]),
                "test_samples": len(split_indices["test"]),
                "train_groups": len(train_groups),
                "val_groups": len(val_groups),
                "test_groups": len(test_groups),
            }
        )

    with (output_dir / "group_fold_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    with (output_dir / "group_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "group_id",
            "source_batch",
            "source_start",
            "source_end",
            "frames",
            "foreground_frames",
            "fold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    key: group[key]
                    for key in fieldnames
                    if key != "fold"
                }
                | {"fold": assignments[group["group_id"]]}
            )
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build leakage-resistant grouped folds for the merged dataset."
    )
    parser.add_argument("--mapping", type=Path, default=conf.MAPPING_PATH)
    parser.add_argument("--output-dir", type=Path, default=conf.INDEX_DIR)
    parser.add_argument("--block-size", type=int, default=conf.GROUP_BLOCK_SIZE)
    parser.add_argument("--num-folds", type=int, default=conf.NUM_FOLDS)
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=conf.NUM_GROUP_PARTITIONS,
        help="Must equal two times num-folds for 70/10/20 outer splits.",
    )
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_partitions != args.num_folds * 2:
        raise ValueError("num-partitions must equal two times num-folds.")
    rows = read_mapping(args.mapping)
    groups = build_contiguous_groups(rows, args.block_size)
    assignments, fold_stats = assign_groups(groups, args.num_partitions, args.seed)
    manifest = validate_and_save(
        groups,
        assignments,
        args.output_dir,
        args.num_folds,
        args.num_partitions,
        {
            "mapping_path": str(args.mapping),
            "sample_count": len(rows),
            "num_folds": args.num_folds,
            "num_group_partitions": args.num_partitions,
            "block_size": args.block_size,
            "seed": args.seed,
            "fold_assignment_stats": fold_stats,
            "grouping_note": (
                "Groups are contiguous windows within source_batch because the "
                "mapping does not contain finer acquisition-sequence identifiers."
            ),
        },
    )
    print(f"Saved grouped folds to {args.output_dir}")
    for fold in manifest["folds"]:
        print(
            f"Fold {fold['fold']}: "
            f"train={fold['train_samples']} "
            f"val={fold['val_samples']} "
            f"test={fold['test_samples']}"
        )


if __name__ == "__main__":
    main()
