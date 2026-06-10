# Experiment Report EXP-067: BN momentum reduction (0.1 → 0.05)

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within Σdt=300s GPU-compute + total wall ≤ 600s (single H20). Baseline at experiment time: **96.45** (EXP-054, commit 86161d9). Bar: **96.55**.

## Idea & Hypothesis
**Chosen idea**: Set `momentum=0.05` (default 0.1) on all four `nn.BatchNorm2d` sites — a longer running-stat EMA window. **Mechanism**: under heavy AugMix the per-batch BN statistics are noisy; halving the update rate lowers eval-time running-stat estimation variance over the same augmented operating distribution (distinct from EXP-061's clean-data recalib, which changed the stat distribution). One of only two genuinely-untested feasible knobs left (with BN eps); chosen as the safest (static kwarg, zero wall/throughput/cudagraph risk — important given 3 recorded wall breaches + the EXP-066 multi-graph penalty).

**Hypothesis**: lower-variance eval stats raise best_test_acc to ≥ 96.55. Stated alternative (most likely): a within-noise null/mild regression, since the cosine-to-0 near-frozen tail already stabilizes running stats.

## Approach
Single-variable: added `momentum=0.05` to BasicBlock bn1 (L71), bn2 (L75), shortcut BN (L83), stem bn1 (L103). All else byte-identical to EXP-054. Params unchanged (4,299,866). No deviations from plan.

## Execution
Single clean run on idle GPU 1 (GPU 0 idle — uncontended), exit 0, no retries, 0 NaN/error. dt steady (586×8ms + 101×9ms — throughput unchanged, confirming BN momentum is compute-free and single-graph). total_seconds 589.2 < 600 (clean, no wall breach). 89 epochs.

## Results
**best_test_acc 96.15% (−0.30pp vs baseline 96.45, < 96.55 bar)** — mild regression. final_test_loss 0.2058 (> EXP-054's 0.1968). The hypothesis was **falsified**: the longer EMA window mildly hurt BOTH top-1 AND eval loss. Mechanism: with the cosine-to-0 schedule, the final epochs run at near-zero LR with near-frozen weights, so the BN running statistics under the default momentum=0.1 are ALREADY well-converged to the eval-time operating point. Lengthening the window (momentum 0.05) makes the running mean/var respond more slowly, folding slightly-staler statistics from the earlier higher-LR (different-weight) batches into the final eval estimate — a small bias that costs both calibration (loss ↑) and top-1 (↓). This is the predicted mild-regression outcome.

**Trajectory fit**: lands in the same −0.2 to −0.6pp band as the other post-plateau single-knob retunes (EXP-062 warmup 96.18, EXP-064 grad-clip 96.34, EXP-065 LS 96.17, EXP-067 96.15) — every scalar/static-hyperparameter knob is at its local optimum at the default. Confirms the project-insights High verdict that 96.45 is at/near the k=4/300s ceiling. This was the cleanest no-improvement of the recent batch — no wall breach, no caveats — a decisive closure of the BN-momentum knob.

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 96.15 < 96.55. **FAILED** (−0.30pp). (Stop at first failed condition.)
- **Necessary condition 2 — clean completion within budget**: total_seconds 589.2 < 600 ✓ (CLEAN), training_seconds 300.0 ✓, num_params 4,299,866 ✓, 0 NaN/error ✓.
- **Necessary condition 3 — no hard-constraint violation**: train.py only ✓; no new deps ✓; seed 42 ✓; evaluate() once/epoch ✓; uncontended ✓.

**Verdict: no-improvement.** Clean valid run (Σdt=300s respected, wall 589.2 < 600, zero caveats) that missed the accuracy bar (−0.30pp). The BN-momentum knob is closed on the lower side (0.05 hurts); the default 0.1 is at/above optimum for this cosine-to-0 recipe.

## Unexplored Avenues
- **BN momentum HIGHER (0.1 → 0.2)**: a SHORTER window (faster-tracking stats) — but with the frozen tail already well-estimated at 0.1, a shorter window would track the (stable) final batches slightly more tightly; plausibly near-null, marginal upside. Low value (the default is already near-optimal and faster-tracking adds noise, not signal).
- **BN eps (1e-5 → 1e-3)**: the other genuinely-untested BN knob (brainstorm-067 Idea 2); near-certain exact null but trivial and wall-safe — the last clean static-knob probe.
- **The BN-stat estimator axis is now nearly closed**: momentum-down hurts (this), clean-recalib hurts (EXP-061); only BN-eps and momentum-up remain, both near-certain nulls.

## Next Steps
1. **BN eps 1e-5→1e-3** (low confidence) — the last genuinely-untested, trivial, wall-safe static knob; closes the BN-estimator axis. (brainstorm-067 Idea 2.)
2. **Accept 96.45 as the k=4/300s ceiling** (high confidence) — every lever (augmentation both paths, capacity, LR, optimizer, EMA/SWA, normalization incl. now BN-momentum, head, batch, activation, label smoothing, throughput→epochs both directions) is closed; project-insights High states this verbatim. Remaining probes are plateau-mapping, not likely breakthroughs.
3. **Lookahead optimizer wrapper** (low confidence) — the one not-quite-covered optimizer mechanism (brainstorm-067 Idea 3), though close to the closed EMA/SWA weight-averaging family; a "new lever" attempt per the NEVER-STOP directive if cleaner probes are exhausted.
