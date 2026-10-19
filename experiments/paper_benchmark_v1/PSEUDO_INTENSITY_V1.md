# Pseudo-intensity supervision screen

Question: If true clean SPL intensity is unavailable, does using a denoised
observation as a pseudo-target help beyond the segmentation task's implicit
denoising? The pseudo-target is **not** ground-truth clean intensity. It is a
fixed, 50%-blended 3x3 binomial smoothing of each observed training image.
Private observations receive no fabricated clean target; NYUv2 uses the same
rule after its deterministic mixed corruption.

All rows use the frozen private train/dev split or NYUv2 train/dev manifest,
same per-seed source B1/B2 checkpoint, 20 epochs, final epoch selection,
private natural flip / NYUv2 mixed corruption, same optimizer and evaluation.
No final-test data are accessed. Seeds 42 and 777 are paired, not pooled
across domains. Public evaluation covers clean plus all 21 existing degraded
conditions. The key endpoints are private foreground IoU, boundary F1 and
tiny-target recall; public clean / mean corruption / heavy photon / heavy
joint mIoU.

| Method | Training / inference | Provenance |
| --- | --- | --- |
| B0 | Raw intensity and depth, segmentation CE | Existing task-control result |
| R_TASK | Learned 2913-parameter intensity residual, CE + 0.01 raw-delta L2 | Existing task-control result |
| FIXED | Fixed smoothing before CMX, segmentation CE | New training |
| PSEUDO_SUP | Same residual architecture and initialization as R_TASK, CE + 0.01 raw-delta L2 + 1.0 pseudo-target MSE | New training |

Thus B0 vs FIXED tests direct preprocessing; B0 vs R_TASK tests
task-driven correction; R_TASK vs PSEUDO_SUP isolates the added pseudo loss
under matched architecture/initialization; FIXED vs PSEUDO_SUP tests whether
learning the correction helps beyond directly applying the same target. The
pseudo weight is predeclared, not dev-tuned here. If it fails, the target or
weight may be unsuitable; a negative result does not rule out all
self-supervision. If it succeeds, it still does not prove physically accurate
recovery of clean intensity.

Run in foreground: `bash scripts/run_pseudo_intensity_v1.sh 0`.
New output: `data/paper_benchmark_runs/pseudo_intensity_v1/`. Completed groups
resume safely; do not remove `RUNNING.lock` while a process is alive. If the
server was forcibly stopped, first confirm the PID/host in that lock is gone,
then move that single lock to an archive name before rerunning. Progress is
printed at epochs 1/5/10/15/20, each completed group, and group summaries.
