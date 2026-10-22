# Regional CA and directional exchange: 12-hour resumable queue

Five arms: B (original FRM), P (regional CA), G (directional semantic-evidence gate),
H (aggressive high-agreement-only hard gate), PG (P+G). FFM is unchanged.
Order: for each seed 42,777,2025, B/G/H/P/PG, each NYUv2 then private. All five
arms are retained for all seeds, including H irrespective of its first score.
This is 30 groups; 12 hours is a session budget, not a promise to finish all groups.
Only completed groups enter comparisons. Resume the same command to continue.

Sources are phase1 CMX_B1 private and CMX_B2 NYUv2 clean checkpoints for each seed.
Each group continues 10 epochs with AdamW, lr 1e-5, effective batch 8, workers=0,
pin_memory=False, identical per-epoch sample order and augmentation. Private uses
natural train970/dev137; public uses train715/dev80, mixed 50% clean/random corruption.
Final test labels are never read. Fixed last epoch selection, no best-dev selection.

Two probe-calibration epochs per domain/seed freeze the entire source CMX and train
eight 1x1 auxiliary heads at lr 1e-3 with labels from the training split. The resulting
heads are shared between arms and frozen during continuation. They read detached
pre-FRM features. Stage 0 is pre-interaction; later stages have earlier FRM content.
These are side-branch probes, NOT independently trained I-only/D-only networks.
Initial probe quality is measured in saved diagnostics; low-quality probes make the
gate hypothesis inconclusive, rather than proving that semantic agreement is useless.

Agreement c=1-JS(pI,pD)/ln(2); confidence r=1-entropy(p)/ln(K).
G uses D->I = rD*(c+(1-c)*(1-rI)) and the reverse analogously.
These are deterministic evidence gates, not calibrated correctness probabilities.
No consistency/KL loss forces the branches to agree. H uses c>=tau with a separate
stage threshold: training-only 75th percentile with per-class capped sampling over
24 batches. Thresholds are fixed afterwards. Record gate saturation as probe features
drift; high agreement can still reflect shared errors. H does not use r.

P applies the existing CA MLP to 4x4 regional average/max statistics, interpolates
channel weights, then blends with global CA using an alpha projected into [0,1].
Alpha starts at zero, so P initially reproduces B. G/H have a deliberate immediate
gating effect, recorded in the initial snapshot. All four FRM stages are instrumented.

Artifacts: protocol.json and source/code hashes; shared calibration checkpoints;
per-group last.pth, midpoint.pth, history.json, result.json; diagnostics at initial,
midpoint and final. Fixed samples are chosen without scores (private class/size
coverage then evenly spaced IDs; public evenly spaced IDs). Each condition includes
PNG contact sheets and NPZ raw inputs, labels, final prediction, native-resolution
probe probabilities, gates, agreement, relative correction magnitude, global CA,
regional CA grid and alpha. Dense effective CA reconstructs by bilinear interpolation
of the grid and alpha blend, avoiding multi-gigabyte duplicated tensors.
All final dev samples have quadrant/gate/correctness statistics, including private
size and class. Labels are used only for evaluation diagnostics, never inference gates.
Public evaluates clean, joint light/medium/heavy, intensity-heavy, missing-depth-heavy.
Probability/map arrays are float16 for storage, metrics are computed before casting.

Budget checks occur within training batches; partial epochs replay from the last
completed epoch with deterministic seeds. Calibration restarts if interrupted.
An evaluation/snapshot may run past the budget because it completes as a unit.
RUNNING.lock prevents duplicate jobs; after hard kill, check recorded host/PID before
manually removing only that lock. Failures write failure.json and status=failed.

Sync new files regional_exchange.py, regional_exchange_diagnostics.py,
run_regional_exchange.py, test_regional_exchange.py and scripts/run_regional_exchange_v1.sh.
Existing paper_benchmark_v1 and cmx_initial_transfer official code/data are dependencies.
No new packages beyond the existing CMX experiment environment are required.

Run from project root: bash scripts/run_regional_exchange_v1.sh 0 12
Use tee to retain visible terminal output. comparison.json includes paired seed deltas;
single-seed results are descriptive. Private acquisition groups are dependent.
