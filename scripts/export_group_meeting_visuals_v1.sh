#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
OUTPUT="${2:-data/paper_benchmark_runs/group_meeting_visuals_v1}"
export CUDA_VISIBLE_DEVICES="${GPU}"

python -u -m experiments.paper_benchmark_v1.export_group_meeting_visuals \
  --output-dir "${OUTPUT}" \
  --device cuda
