# Next experiment: checkpoint and degradation diagnosis

Sync the entire experiments/paper_benchmark_v1 directory and
scripts/run_paper_diagnosis_v2.sh. Existing server data and 36 best.pth files
are required; no pretrained ImageNet downloads are needed.

Run from the server project root:

```bash
mkdir -p data/paper_benchmark_runs/diagnosis_v2
nohup bash scripts/run_paper_diagnosis_v2.sh 0 \
  > data/paper_benchmark_runs/diagnosis_v2/launcher.log 2>&1 &
echo $!
```

This performs inference, not new training. It reads only private dev and NYUv2
dev. It never overwrites phase1 histories or checkpoints. A failure exits with
a traceback in launcher.log; do not interpret old history entries as liveness.

1. Audit all 36 result/history pairs.
2. Load the actual checkpoint and its configuration; re-evaluate historical v1
   inputs in FP32 with batch size 1. Small FP32/AMP discrepancies are distinct
   from incorrect checkpoint epoch associations. Checkpoint SHA256 is recorded.
3. Evaluate clean-trained CMX-B2 seed42 on 22 v2 conditions, with a common
   corruption seed independent of initialization. Record confusion matrices in
   natural holes, added holes, valid depth and semantic boundary regions.

v2 settings: missingness removes 10/25/40 percent of originally valid pixels.
These are fixed diagnostic settings, not a claim of private-physics calibration.
Clustered holes come from ranked smooth random fields; joint assigns half its
budget to clusters and the remainder to low-intensity-biased missingness, plus
the existing Poisson/Gaussian appearance proxy. Mask cardinality is exact to
rounding. Joint severity masks need not be nested; no claim of nested joint
masks is made. Appearance is a grayscale proxy, not physical photon counts.

Outputs: history_audit.json; per-run reevaluation.json; 22 files under
b2_clean_seed42_curves; complete.json on successful completion. Region mIoU
averages classes with nonzero union in that region; inspect confusion matrices
alongside the score because region class composition differs.

Return diagnosis_v2 in full for selection of the main module. The next training
decision depends on these results: reliability routing if joint/noise dominates;
validity-constrained propagation if added-hole/boundary errors dominate. No
final-test or new architecture training is included in this diagnostic job.
