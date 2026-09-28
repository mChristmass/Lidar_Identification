# Fusion window v1: 3–5 hour server run

This is a **single-seed exploratory screen**, with conditional paired-seed
confirmation. It uses only the frozen NYUv2 715-train/80-dev tune split.
Private final_test and official NYUv2 test are not accessed.

## Steps and decision rule

1. Re-evaluate the completed mixed R0 seed42 checkpoint on clean, heavy
   photon proxy, and heavy joint corruption. For each condition, additionally
   set intensity to zero and set depth plus its validity to zero. These are
   input-removal stress tests, not causal attribution. Reproducing the saved
   R0 curve is a required gate.
2. Train F1 and F2 from the **same clean B2 seed42 checkpoint** as R0, with
   its epoch-varying mixed augmentation, sample order, 20 epochs, batch 2,
   accumulation 4, base LR 1e-5, and final-epoch evaluation on 22 conditions.
   New scalar controls use LR 1e-3 in both candidates.
3. Select a candidate only if its mean across 21 corruptions improves by
   at least **0.5 percentage points**, or heavy joint improves by at least
   **1.0 percentage point**, with clean loss no worse than 0.5 points.
   If no candidate passes, stop. The gate is a screening heuristic, not
   statistical evidence or a final paper selection.
4. If the gate passes and at least 150 minutes remain of the 4.5-hour budget,
   run **both** R0 and the selected candidate from the clean B2 seed777
   checkpoint. Each receives 20 matched epochs and 22 evaluations. These
   results test a second seed; a third seed would still be needed for a
   final multi-seed claim.

F1 modulates CMX FFM cross-path residuals at unsupported depth locations.
F2 includes F1 and adds normalized nearby depth-feature propagation before
FFM. Both begin at the exact original CMX function; no oracle degradation
mask or label is read during inference. This does not establish novelty.

Estimated runtime based on the previous pilot: diagnostic ~10–30 min;
F1/F2 ~45–75 min each; conditional two seed777 groups another ~90–150 min.
Hence about 2–3 hours without confirmation and 4–5 hours if it runs.
Actual server throughput controls the result. The budget gates **starting**
the optional phase; it cannot forcibly end an already running training job.

## Server files

Sync these additions:

- `experiments/paper_benchmark_v1/fusion_pilot_model.py`
- `experiments/paper_benchmark_v1/diagnose_modalities.py`
- `experiments/paper_benchmark_v1/run_fusion_window.py`
- `experiments/paper_benchmark_v1/test_fusion_window.py`
- `scripts/run_fusion_window_v1.sh`

Existing `experiments/paper_benchmark_v1/` code and its CMX dependencies
must also be present. Keep the existing processed NYUv2 data, the clean B2
seed42 and seed777 `best.pth` files from phase1, and the full
`mixed_control_v1/` directory (especially `mixed/last.pth`,
`mixed/curves/*.json`, `protocol.json`, `complete.json`).
No new dataset or pretrained download.

From the repository root in the existing server environment:

```bash
mkdir -p data/paper_benchmark_runs/fusion_window_v1
nohup bash scripts/run_fusion_window_v1.sh 0 \
  >> data/paper_benchmark_runs/fusion_window_v1/launcher.log 2>&1 &
echo $!
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/fusion_window_v1 --watch --interval 60
```

One line prints per training epoch and evaluation condition. `status.json`
updates inside epochs, `decision.json` records the automatic gate, and
`complete.json` records the final completed count. Atomic checkpoints
include optimizer/scaler state and resume at the last finished epoch.
An exclusive `RUNNING.lock` prevents duplicate launches. Following a hard
kill, verify the process identified in that file no longer exists before
removing **that file alone** and restarting the same command.

Sync back `comparison.json`, `decision.json`, `complete.json`,
`diagnostic_summary.json`, group histories/results/curves, and the launcher
log for analysis. Checkpoints may stay on the server.
