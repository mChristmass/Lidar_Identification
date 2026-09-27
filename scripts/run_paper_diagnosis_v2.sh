#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-0}"
python -u -m experiments.paper_benchmark_v1.test_diagnosis_v2
python -u -m experiments.paper_benchmark_v1.diagnose_phase1 \
  --runs-dir data/paper_benchmark_runs/phase1_v1 \
  --output-dir data/paper_benchmark_runs/diagnosis_v2 \
  --device cuda
