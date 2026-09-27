#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
RUNS_DIR="${2:-data/paper_benchmark_runs/phase1_v1}"
NYUV2_DIR="${3:-data/public_semseg/nyuv2/processed}"
RAW_DIR="${4:-data/public_semseg/nyuv2/raw}"

mkdir -p "$RUNS_DIR"

echo "PHASE1_PREFLIGHT step=private_split_verify"
python -u scripts/prepare_private_paper_split.py verify

echo "PHASE1_PREFLIGHT step=private_degradation_audit"
python -u scripts/audit_private_train_degradation.py

if [[ ! -f "$NYUV2_DIR/audit.json" ]]; then
  echo "PHASE1_PREFLIGHT step=nyuv2_download_and_extract"
  bash scripts/prepare_nyuv2_paper.sh "$RAW_DIR" "$NYUV2_DIR"
else
  echo "PHASE1_PREFLIGHT step=nyuv2_existing_audit path=$NYUV2_DIR/audit.json"
  python -u -m experiments.paper_benchmark_v1.run audit \
    --dataset nyuv2 --stage tune --nyuv2-dir "$NYUV2_DIR"
fi

echo "PHASE1_PREFLIGHT step=unit_tests"
python -u -m experiments.paper_benchmark_v1.test_paper_benchmark

echo "PHASE1_PREFLIGHT step=gpu_forward_backward gpu=$GPU_ID"
CUDA_VISIBLE_DEVICES="$GPU_ID" python -u -m experiments.paper_benchmark_v1.smoke \
  --device cuda --amp

echo "PHASE1_START jobs=36 runs_dir=$RUNS_DIR"
CUDA_VISIBLE_DEVICES="$GPU_ID" python -u -m experiments.paper_benchmark_v1.launch_phase1 \
  --datasets private nyuv2 \
  --models T8 CMX_B0 CMX_B1 CMX_B2 \
  --seeds 42 777 2025 \
  --epochs 100 \
  --runs-dir "$RUNS_DIR" \
  --nyuv2-dir "$NYUV2_DIR" \
  --device cuda --amp --skip-existing

echo "PHASE1_COMPLETE runs_dir=$RUNS_DIR"

