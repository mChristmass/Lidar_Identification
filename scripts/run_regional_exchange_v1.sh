#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-0}"
HOURS="${2:-12}"
RUNS="${3:-data/paper_benchmark_runs/regional_exchange_v1}"
mkdir -p "$RUNS"
python -u -m experiments.paper_benchmark_v1.test_regional_exchange
python -u -m experiments.paper_benchmark_v1.run_regional_exchange --output-dir "$RUNS" --preflight-only
python -u -m experiments.paper_benchmark_v1.run_regional_exchange --output-dir "$RUNS" --budget-hours "$HOURS"
