#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
python -u -m experiments.paper_benchmark_v1.test_repair_diagnosis
python -u -m experiments.paper_benchmark_v1.diagnose_repair --device cuda
