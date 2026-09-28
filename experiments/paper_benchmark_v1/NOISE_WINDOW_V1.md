# Five-hour intensity-noise experiment

On the frozen NYUv2 715-train/80-dev split, all three candidates restart from
the same clean CMX-B2 seed42 checkpoint used by the completed mixed R0.
They see the same epoch-varying mixed corruption, sample ordering and paired
horizontal flips. Only the training objectives and model additions differ:

- **C1:** CMX with CE plus noisy-to-clean prediction consistency when synthetic
  photon/joint noise was applied. At inference the architecture is original CMX.
- **D1:** C1 plus a zero-initialized intensity-restoration front end trained
  against the public clean grayscale image. This isolates restoration effects.
- **D2:** D1 plus a predicted intensity reliability map supervised by the
  train-time difference between clean and corrupted intensity. At inference
  this predicted map modulates the intensity-to-depth FFM cross residual.

The corruption metadata and clean target are used **only during training**.
The inference input remains intensity, depth and observed depth validity.
These are exploratory controls, not established novel methods. The simulated
photon proxy is not a physical SPL sensor simulator.

All candidates: seed42, 20 epochs, batch 2, gradient accumulation 4, no
DataLoader workers or pinned memory, AdamW weight decay .01, base LR 1e-5,
new parameter LR 1e-3, poly decay power .9, CUDA AMP, final epoch checkpoint.
Loss: CE + 0.2 KL at temperature 2 for synthetically noisy intensity frames,
plus 5x clean-intensity MSE for D1/D2, plus 0.05x reliability BCE for D2.
For both auxiliary targets, synthetically noisy intensity frames receive
weight 1 and other frames weight 0.05 before the batch average. This avoids
the predominantly clean mixed batches overwhelming the noise target.
The loss controls are part of the treatment; compare C1→D1→D2 to isolate
effects, and compare C1 against historical mixed R0 for consistency effects.

Each candidate evaluates the same 22 frozen dev conditions and regional masks.
D2 additionally records predicted-confidence means and correlation with the
synthetic noise proxy on three dev conditions; this audit does not select a
checkpoint or alter the candidate gate.
After all three, select only if mean mIoU over 21 corruptions improves at least
0.5 percentage points **or** heavy photon/joint improves at least 1.0 point,
with clean loss no worse than 0.5 point. If a candidate qualifies and at least
150 minutes of the five-hour budget remain, train both R0 and the selected
candidate from the clean B2 seed777 checkpoint with matched augmentation and
20 epochs. This is a second-seed screen, not yet a three-seed paper result.
If none qualifies, the run ends early and writes decision.json.

Estimated from the previous pilot and the extra teacher pass: three required
groups take roughly 3–4 hours; the conditional pair may bring total time close
to five hours. The budget governs whether the optional phase starts; a running
group is not forcibly stopped. Checkpoints resume from each finished epoch.

## Server synchronization

Sync the new files in `experiments/paper_benchmark_v1/` and
`scripts/run_noise_window_v1.sh`, or update the repository to the commit
reported in chat. Keep the existing CMX dependencies and:

- `data/public_semseg/nyuv2/processed/`, including train/dev manifests
- `data/paper_benchmark_runs/phase1_v1/nyuv2_clean_medium/nyuv2/tune/CMX_B2/seed_42/best.pth`
- the matching `seed_777/best.pth`
- `data/paper_benchmark_runs/mixed_control_v1/protocol.json`,
  `complete.json`, and `mixed/curves/` (22 JSON files)

No new dataset or pretrained download is required. From the repository root:

```bash
mkdir -p data/paper_benchmark_runs/noise_window_v1
nohup bash scripts/run_noise_window_v1.sh 0 \
  >> data/paper_benchmark_runs/noise_window_v1/launcher.log 2>&1 &
echo $!
python -u -m experiments.paper_benchmark_v1.status_reliability \
  --runs-dir data/paper_benchmark_runs/noise_window_v1 --watch --interval 60
```

Status updates include group/epoch/batch. The log prints one summary per
epoch and condition. `decision.json` records the gate and remaining budget.
For analysis sync back the JSON files, curves and launcher log; the large
checkpoints may stay on the server. An exclusive `RUNNING.lock` prevents
duplicate launches; after an abrupt kill, verify that its PID is no longer
running before removing **only that lock file** and rerunning the command.
