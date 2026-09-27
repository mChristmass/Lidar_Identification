# Reliability pilot v1

Scope: NYUv2 frozen 715-train/80-dev split, seed 42. No official test or private
final-test access. These are candidate mechanisms, not established novel methods.

## Experiments

- R0: reuse mixed_control_v1/mixed (no training).
- R1: original CMX plus a zero-initialized 1x1 validity-to-depth additive projection.
- R2: suppress depth-to-intensity FRM residuals according to area-pooled observed
  validity. Four learned strengths are projected to [0,1] after optimizer steps.
- R3: R2 plus suppression of intensity-to-depth FRM residuals using a learned
  confidence map from observed intensity, local mean, and absolute local residual.
  Confidence is segmentation-supervised, not calibrated SNR or oracle noise labels.

All additions initially reproduce original CMX exactly. Neither branch's semantic
features are zeroed. FFM remains unchanged: this is FRM modulation, not a full
validity-masked attention implementation. R1 and R2 are parallel controls, not a
cumulative stack; only R3 includes R2. Log learned strengths to detect inactivity.

## Matched training

All candidates restart from the SAME original clean B2 seed42 checkpoint used by
mixed_control_v1 (not from its already-finetuned mixed checkpoint). Exactly 20
epochs, final checkpoint, batch 2, accumulation 4, workers 0, pin_memory False,
AdamW weight decay .01, gradient clip 1, CUDA AMP, base LR 1e-5 with polynomial
decay power .9. Same mixed TrainingData, epoch seeds, order and flips as R0.

New parameters use LR 1e-3 with the same decay in ALL candidates: starting from
zero, scalar suppression strengths would barely move at 1e-5 over this short
pilot. This is an explicit new-parameter optimization choice, not identical
parameter-wise optimization to R0. Do not attribute gains solely to architecture
without later recipe sensitivity checks. This pilot is not a final paper result.

Reference initialization, manifests, executable sources and reference curves are
hashed in protocol.json. Changing them requires a new output directory. Reference
curves must match each candidate's actual sample IDs and missingness budgets.

## Server synchronization

Sync experiments/paper_benchmark_v1/ and scripts/run_reliability_pilot_v1.sh.
Existing dependencies from the previous Git snapshot must remain installed.
No new dataset or pretrained download. Keep these existing SERVER files:

- data/public_semseg/nyuv2/processed/ (including manifests/train.jsonl and dev.jsonl)
- data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_42/best.pth
- data/paper_benchmark_runs/mixed_control_v1/protocol.json and complete.json
- data/paper_benchmark_runs/mixed_control_v1/mixed/curves/ (22 JSON files)

From the repository root in the existing training environment:

```bash
mkdir -p data/paper_benchmark_runs/reliability_pilot_v1
nohup bash scripts/run_reliability_pilot_v1.sh 0 \
  >> data/paper_benchmark_runs/reliability_pilot_v1/launcher.log 2>&1 &
echo $!
python -u -m experiments.paper_benchmark_v1.status_reliability --watch --interval 60
```

Training prints one line per epoch; evaluation prints one line per condition;
GROUP_COMPLETE reports clean mIoU and the equal-weight mean over 21 corruptions,
including the difference against R0. This is one seed, NOT five folds.
status.json updates every 25 batches and before each evaluation. Staleness means
inspect launcher.log and process status, not that failure is proven.

Restart the same command after a handled failure. Atomic checkpoints resume from
the last completed epoch with optimizer/scaler state. Existing evaluated conditions
are reused only for the same checkpoint. A root exclusive RUNNING.lock prevents
duplicate launches; after SIGKILL or power loss, verify the recorded PID/host is
no longer running before manually removing only that lock. Never delete a live lock.

Outputs: protocol.json, status.json, comparison.json, complete.json; each R1/R2/R3
contains last.pth, history.json, result.json and curves/*.json. Sync JSON and launcher
log back for analysis; large checkpoint files can stay on the server.
