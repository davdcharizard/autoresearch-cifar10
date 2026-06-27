# Report EXP-011: EMA weight averaging for evaluation (decay 0.995)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-011.md
- **Plan**: plans/plan-011.md
- **Log**: logs/exp-log-011.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: can evaluating an EMA of the weights — the last untried, widely-validated lever — harvest the ±0.1pp terminal eval noise and clear the +0.1pp bar without touching the schedule?

## Idea & Hypothesis

Chosen idea: keep the training recipe byte-identical and score an exponential moving average (decay 0.995, ~1.4-epoch half-life) of the weights at each per-epoch eval instead of the live weights. Rationale: EMA harvests variance the run already pays for rather than deferring progress — the mechanism that killed the previous four experiments. Hypothesis: best_test_acc ≥ 96.81 with dt ≤ 23ms, ~135–139 epochs, EMA lagging mid-schedule then converging onto/above the raw trajectory in the low-LR tail. Runner-up candidates: WARMUP_FRAC 0.15→0.08, augmentation fade-out at 85% progress.

## Approach

~12-line change in train.py: `import copy`; `EMA_DECAY = 0.995`; eager EMA copy via `copy.deepcopy(base_model).eval()` with grads off; after each `optimizer.step()` (inside the timed region) `torch._foreach_lerp_(ema_params, live_params, 0.005)` plus per-buffer `.copy_()` for the ~40 BN buffers; eval switched to `evaluator.evaluate(ema_model, device)`. No deviations from plan.

## Execution

Three runs. Runs 1–2 were infrastructure-contaminated and retried per protocol: Run 1 (89 epochs, 95.85) was time-sliced by a confirmed foreign process on GPU 0 (window-avg dt alternating 24ms/48ms); Run 2 (105 epochs, 96.01) lost its first ~166s of budget to host-side interference coinciding with a heavy GPU-1 job, with no GPU-0 compute app visible. Run 3 launched into a fully idle node and was CLEAN end-to-end: 139 epochs / 13391 steps, cumulative dt 22.4ms — byte-identical to baseline; a throughput watchdog (>30ms windowed step time) fired zero events. Notable protocol discoveries: nvidia-smi reports host-namespace PIDs (container pgrep can never match them, so PID-based contention detection is unreliable — our own Run-3 process was flagged "foreign"); contention is reliably diagnosed from pct_done deltas between step prints, not printed dt (prints sample 2% of steps). Both recorded in infra-errors.md and the exp-log.

## Results

- **Primary metric**: best_test_acc = 96.46% (baseline: 96.71, delta: −0.25pp, −0.26%)
- **Observations**: The clean run matched every predicted execution signature — EMA tax unmeasurable (139 epochs, cum dt 22.4ms vs 22.3 baseline), VRAM +18MB as computed, near-random epoch-1 eval (10.46%, the predicted init-dominated blend) recovering by epoch 5 (43.91%). The tail was a CONVERGED plateau (96.38–96.46 over epochs 130–139, final 96.43, best 96.46) — not starvation. final_test_loss improved to 0.1874 (vs ~0.20-class raw-eval baselines): the EMA genuinely produces a better-calibrated, lower-loss model.
- **Analysis**: The hypothesis failed for a reason more interesting than "EMA doesn't help": **the metric is a max-statistic, and max-statistics reward variance**. best_test_acc takes the maximum over 139 per-epoch evals that fluctuate ±0.1pp at convergence. The baseline's 96.71 is partly a lucky draw from that noise distribution; the EMA's smooth trajectory collapses the upper tail of the max while raising the mean only slightly (better test loss, −0.25pp best acc). For EMA-eval to win on this metric it must lift the MEAN by more than the variance it removes — a much higher bar than the "matches or beats" folklore, which is calibrated on final-epoch or mean-eval comparisons, not best-over-epochs. The negative is clean and definitive at this decay: identical epochs, identical throughput, single-variable attribution. It also sharpens the project's picture of the remaining headroom: the recipe's residual ±0.1pp eval noise is not harvestable by smoothing — it is part of what the current 96.71 baseline already banked.
- **Key Learning**: best_test_acc is a max over noisy per-epoch evals and therefore rewards variance; weight averaging trades variance for mean and collapses the peak (96.46 vs 96.71 at identical 139 epochs) — variance-reduction techniques are counterproductive for max-statistic metrics unless they raise the mean by more than the noise amplitude.

## Verification

- **Conditions**: condition 1 passed (clean exit, 479.9s ≤ 600); condition 2 failed (best_test_acc 96.46 < 96.81 = baseline + 0.1pp); condition 3 skipped per first-failure stop (informally compliant: 139 eval lines = 139 epochs)
- **Review Notes**: results trustworthy — metric consistent with the 139-line eval trail; throughput/params/VRAM all match predictions; Runs 1–2 correctly excluded as infrastructure-contaminated (their 89/105-epoch counts and window-dt signatures prove external interference, confirmed live for Run 1); no constraint violations
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run, metric below baseline + 0.1pp)

## Unexplored Avenues

- **Shorter-horizon EMA (decay ~0.98, ~0.35-epoch half-life)**: less lag, keeps more variance while still denoising; but the analysis suggests ANY smoothing fights the max-statistic — expected value low.
- **Eval both models and report the better (raw + EMA)**: blocked — validation at most once per epoch is a hard constraint, and alternating eval targets across epochs adds noise rather than removing it.
- **EMA as the FINAL eval only (raw evals every epoch, EMA scored once at the end)**: same once-per-epoch budget could be respected by swapping only the last eval, but best-over-epochs already captures the raw trajectory's peak; the single EMA eval would need to beat the banked max — strictly harder than this experiment.
- **Variance-INCREASING tails** (the inverted lesson): mechanisms that add benign exploration late (e.g., small constant LR floor instead of cosine-to-zero) raise eval variance at a converged mean and could push the max up — speculative, but it is the direction the max-statistic insight actually points.

## Next Steps

1. **WARMUP_FRAC 0.15 → 0.08** — the surviving runner-up from brainstorm-011: redistributes ~20s of budget from warmup to the anneal; trivial diff, clean failure mode. Confidence: low-medium.
2. **LR floor experiment (cosine to ~0.002–0.005 instead of ~0)** — directly tests the max-statistic insight: a small terminal LR keeps the converged model wandering, raising eval variance around a flat mean; the max over ~10 tail epochs samples that distribution. Confidence: low-medium (novel mechanism, cheap test).
3. **PEAK_LR 0.3 downward probe** — completes the LR bracket from below; EXP-006's healthy convergence shape argues against, but it is the last untouched base hyperparameter direction. Confidence: low.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
