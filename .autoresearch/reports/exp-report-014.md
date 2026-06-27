# Report EXP-014: Shorten LR warmup (WARMUP_FRAC 0.15 → 0.08)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-014.md
- **Plan**: plans/plan-014.md
- **Log**: logs/exp-log-014.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: is the 15% LR warmup (45s of the timed budget at sub-peak LR) wasted time that can be reallocated into the cosine anneal?

## Idea & Hypothesis

Chosen idea: single-constant change WARMUP_FRAC 0.15 → 0.08. External grounding: the NeurIPS 2024 warmup study (arXiv 2406.09405) finds warmup's only first-order benefit is tolerating the peak LR (1–5% of training typical), and EXP-010 proved this model tolerates 1.5x the current peak — so most of the 45s ramp looked like budget idling below peak. Hypothesis: the trajectory would run AHEAD of baseline from ~ep 15 and best_test_acc ≥ 96.81 at byte-identical throughput. Runners-up: WD 5e-4 → 2.5e-4, LS 0.1 → 0.05.

## Approach

1-line diff in train.py: `WARMUP_FRAC = 0.15` → `0.08`. No deviations from plan. The cosine phase needed no edit — `lr_at()` normalizes by `(1 − WARMUP_FRAC)`, so the anneal automatically stretched from 85% to 92% of the budget.

## Execution

Two runs. **Run 1** (task bm5zfzrxw) launched into a GPU-0 window that closed mid-run: the foreign job that had blocked the original launch returned, time-slicing the GPU (43% of windowed step-time samples > 30ms, sustained 48ms stretches; 92 epochs vs ~139 clean). Quarantined per the EXP-011 protocol, never analyzed. Side lesson: the live Monitor armed in a separate turn first polled an already-finished run — watchdogs must launch in the same command chain as the run. **Run 2** (task bh32axche) used a composite launcher with an inline watchdog (auto-kill on 4 consecutive slow windows) and was the cleanest run on record: 0 of 267 windows > 30ms (mean 22.4ms), 139 epochs / 13410 steps exactly on projection, total 482.9s, VRAM 1613.0MB, params 4,286,026 — every signature byte-identical to baseline as predicted.

## Results

- **Primary metric**: best_test_acc = 96.49% (baseline: 96.71, delta: −0.22pp, −0.23%); bar was 96.81
- **Observations**: Hypothesis refuted IN DIRECTION, not just magnitude. The trajectory ran BEHIND baseline through the entire mid-schedule (ep 20: 78.5 vs ~88; ep 60: 87.8 vs ~92; ep 100: 93.1 vs ~96), then converged on a flat plateau (best 96.49 first reached ep 132, final 96.45 ≈ best). The mechanism error in the brainstorm: with a TIME-KEYED cosine, shortening warmup does not "recover wasted ramp time" — it makes the LR strictly HOTTER at every progress point p < 1 (the anneal starts earlier, so q(p) = (p−W)/(1−W) shrinks and lr(p) rises for all p). The change is functionally a milder EXP-010: more total heat ⇒ deferred mid-schedule progress ⇒ the fixed-length tail cannot fully repay (−0.22pp vs EXP-010's −0.57pp at a bigger heat increase). Notably final_test_loss IMPROVED to 0.1851 (best on record), echoing EXP-011: mean/loss gains do not move the max-statistic.
- **Analysis**: This closes the schedule-shape axis from the hot side for the SECOND time, and the two probes now form a consistent dose-response on "integrated LR heat": peak +50% → −0.57pp; warmup-halving (smaller heat increase) → −0.22pp. Both refute the "wasted ramp" framing — on this recipe the 0.15-warmup/0.4-peak shape is not idling, it is pacing: the sub-peak early phase IS productive training positioned where the optimizer needs it. The external evidence (warmup = only stability insurance) failed to transfer for the same reason most external results have failed here (project-insights Medium): it was calibrated on fixed-iteration setups where shortening warmup adds usable steps at the END; under a time-keyed schedule it instead reshapes the whole LR curve upward. With hot-side probes failing symmetrically, the remaining schedule question is the COLD side (peak 0.3, or warmup 0.25) — but EXP-006's healthy convergence shape and the now two-sided heat dose-response give that low odds. Nine consecutive misses; the recipe's constants (peak, warmup, batch, WD via composition, augmentation set) are now each either measured-optimal or measured-saturated, except WEIGHT_DECAY which remains the single never-probed constant.
- **Key Learning**: Under a time-keyed (progress = elapsed/budget) cosine schedule, EVERY shape parameter is a heat parameter — shortening warmup raises LR at every subsequent instant rather than freeing budget, so "reduce wasted ramp" intuitions from fixed-iteration training invert. The 2016-style intuition that warmup time is overhead does not apply when the anneal is guaranteed to complete regardless.

## Verification

- **Conditions**: pre-condition contention sanity CLEAN (139/139 epochs, watchdog silent, post-hoc 0/267 slow windows); condition 1 FAILED (best_test_acc 96.49 < 96.81); conditions 2–3 skipped per first-failure stop (observed informally: 482.9s ≤ 600 and 139 evals = 139 epochs would have passed)
- **Review Notes**: trustworthy — the metric is consistent with the full eval trail, throughput signatures byte-identical to baseline (so the deficit is purely the schedule change), and the contaminated Run 1 was correctly quarantined rather than mixed in
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid clean run; primary-metric necessary condition not met)

## Unexplored Avenues

- **Cold-side schedule probe (PEAK_LR 0.3 at warmup 0.15, or WARMUP_FRAC 0.25)**: the only untested schedule direction; the symmetric hot-side failures imply the optimum is near current values, so expected value is low but non-zero — a cold probe would complete the two-sided bracket.
- **Decoupled shapes (e.g. keep warmup 0.15 but use a lower-power anneal like linear instead of cosine)**: changes the heat DISTRIBUTION without monotonically raising it; mechanism plausible but combinatorial and weakly informative per probe.
- **Warmup-shortening WITH compensating peak reduction (0.08 warmup + 0.35 peak)**: holds integrated heat roughly constant while still lengthening the anneal — directly tests whether the anneal-length component had any positive effect masked by the heat increase. The cleanest follow-up if the schedule axis is ever revisited.

## Next Steps

1. **WEIGHT_DECAY 5e-4 → 2.5e-4** — now the ONLY never-probed recipe constant; the four-point augmentation dose-response argues total regularization pressure is at-or-past optimum, and WD is the one component that can move pressure DOWN without touching the data pipeline. Confidence: low-medium (the curve reads "at optimum", so sign is uncertain — but it is the last unmeasured axis).
2. **Compensated schedule probe (warmup 0.08 + peak 0.35)** — isolates anneal-length from heat using this loop's mechanism insight. Confidence: low.
3. **Combination of near-misses** — EXP-012 (batch 1024) was metric-neutral at +12 epochs; pairing it with a heat-reducing tweak (e.g. sqrt-scaled LR 0.57) the brainstorm already flagged as the only batch variant with residual headroom. Confidence: low.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
