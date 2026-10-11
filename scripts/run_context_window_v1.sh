#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${1:-0}"
python -u -m experiments.paper_benchmark_v1.test_context_window
python -u -m experiments.paper_benchmark_v1.run_context_window --device cuda --budget-hours 3
