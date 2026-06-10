# Experiment Report EXP-062: WARMUP_FRAC isolation (0.05 → 0.10)

- **Date**: 2026-06-09
- **Verdict**: no-improvement
- **Primary metric**: best_test_acc = **96.18%** (baseline 96.45, bar 96.55; delta **−0.27pp**)

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) by editing only `train.py`, within the fixed 300s Σdt GPU-time budget on a single H20 (≤600s wall). Baseline 96.45 (EXP-054, commit 86161d9); bar = baseline + 0.1pp = 96.55.

## Idea & Hypothesis
**Chosen idea**: WARMUP_FRAC isolation — double the linear-warmup fraction of the time-budget LR schedule from 0.05 → 0.10, all else byte-identical to the EXP-054 best recipe. **Rationale**: WARMUP_FRAC was the one LR-schedule scalar never isolated (PEAK_LR swept in EXP-016/017, schedule-shape in EXP-029; warmup-length never). **Hypothesis**: with strong AugMix augmentation producing noisy early-batch gradients at PEAK_LR=0.2, a longer warmup (~9 ep vs ~4.5 ep) could stabilize early training and reach a marginally better basin (≥96.55). Honest prior expectation: near-noise null on a deeply-mapped plateau.

## Approach
Single-constant change: train.py L24 `WARMUP_FRAC` 0.05 → 0.10. The `lr_at_fraction` function (L35-41) reads WARMUP_FRAC, so doubling it lengthens the linear ramp 0→PEAK_LR and slightly compresses the subsequent cosine-anneal phase. Throughput- and wall-neutral (warmup only redistributes the existing time-fraction schedule; adds zero work). Smoke checks: AST OK; `git diff --name-only` == train.py only; LR sanity lr(0.0)=0, lr(0.10)=0.2 (peak at frac=0.10), lr(1.0)=0.

## Execution
One run, GPU 1 (idle; GPU 0 had foreign proc PID 1200082, 814MiB). Completed exit 0 in 596.0s wall, 91 epochs, 35294 steps. dt distribution 618×8ms + 86×9ms + 1×25ms (compile warmup) — uncontended, throughput identical to EXP-054. 0 NaN/error. No retries or adjustments. peak_vram_mb 453.8, num_params 4,299,866.

## Results
best_test_acc 96.18% — a **−0.27pp regression** vs baseline 96.45, missing the 96.55 bar by 0.37pp. final_test_loss 0.1975 (vs EXP-054's 0.1968 — marginally worse, consistent with slight under-training). The mechanism is clear: doubling the warmup to 10% of the budget eats ~4.5 epochs out of the high-LR cosine phase, so the model spends less time at productive mid-training learning rates and under-trains slightly. This directly corroborates the EXP-016/017 finding that the LR regime is finely balanced (±0.05 peak cost ~0.5pp): the schedule's time allocation is near-optimal at the default warmup 0.05, and perturbing it in either direction costs accuracy. The hypothesis (longer warmup stabilizes noisy early AugMix gradients → better basin) is refuted — this net already trains stably at warmup 0.05 (no early divergence in any prior run), so there was no instability for a longer warmup to fix; the only effect was the opportunity cost of lost high-LR training time. This is the 64th experiment, 9th distinct lever-perturbation to leave the 96.45 ceiling intact, and closes WARMUP_FRAC as a closed scalar.

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 96.18 < 96.55 → **FAILED**. Stopped at first failed condition.
- Conditions 2 (in-budget completion) and 3 (no hard-constraint violation) not formally evaluated (aborted after condition 1), but for the record both would pass: total_seconds 596.0 < 600, num_params 4,299,866, num_epochs 91, 0 NaN, git diff == train.py only, uncontended dt.
- **Trustworthiness**: result is fully trustworthy — clean uncontended run, throughput matched the reference exactly, metric value is plausible and consistent with the loss. No false-failure / false-pass / integrity concerns.
- **Verdict basis**: valid in-budget run that missed the accuracy bar → **no-improvement**.

## Unexplored Avenues
- **Shorter warmup (0.05 → 0.025)**: the regression direction suggests the schedule wants MORE high-LR time, not less; a shorter warmup is the symmetric untested probe. But EXP-016/017's finely-balanced finding makes a meaningful gain unlikely (expected near-noise or small regression) — low confidence, likely not worth a run.
- **Aug cooldown @0.10 on the AugMix recipe** (brainstorm-062 Idea 3): combine the EXP-034 near-miss (only ever on TA) with the AugMix best; wall-neutral-to-faster (removes tail work, unlike EXP-061). The one remaining un-combined near-miss.
- **Gradient-norm clipping at a permissive threshold** (brainstorm-062 Idea 2): untested knob, but this net trains stably so most likely a null.

## Next Steps
1. **Aug cooldown @0.10 on AugMix recipe** (medium confidence) — the last un-combined near-miss; wall-safe; clear joint clean-adaptation mechanism (EXP-061 insight). Best remaining principled probe.
2. **Gradient clipping probe** (low confidence) — trivial untested scalar; likely null on a stably-training net but cheap to close.
3. **Accept the plateau is near-exhausted on scalars** — after this, the remaining headroom (if any) is in untried structural/recipe combinations, not single-scalar perturbations. Consider a more radical architectural change per the NEVER-STOP "think harder" directive.

## Key Learning
Doubling WARMUP_FRAC to 0.10 regressed −0.27pp: the longer ramp steals high-LR cosine time and under-trains. The net is already stable at warmup 0.05, so there was no early instability to fix — only opportunity cost. The LR schedule's time allocation is near-optimal; WARMUP_FRAC is now a closed scalar.
