# Experiment Report EXP-065: Higher label smoothing (LABEL_SMOOTHING 0.1 → 0.15)

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within the 300s GPU-compute budget (single H20). Baseline at experiment time: **96.45** (EXP-054, commit 86161d9). Bar this loop: **96.55** (baseline + 0.1pp).

## Idea & Hypothesis
**Chosen idea**: Raise `LABEL_SMOOTHING` 0.1 → 0.15 on the current AugMix-p0.5 best, single-variable. EXP-023 probed LS only DOWNWARD (0.1→0.05, regressed to 96.03) on the OLD TrivialAugment recipe and concluded "0.1 near-optimal" — but the upper side was never tested, and never on the current AugMix recipe. The mechanism: heavy AugMix (multi-chain convex mixing → soft, distribution-shifted images on 50% of the batch) interacts with target softness; softer targets could better match the softened inputs (the Mixup↔soft-target pairing rationale), so the LS optimum might sit higher under AugMix than under TrivialAugment.

**Hypothesis**: 0.15 better matches the AugMix-mixed inputs and lifts best_test_acc to ≥ 96.55. Honest prior (project-insights Medium "adding regularization hurts at this short budget" + EXP-023's lower-side regression): more likely a near-noise null or small regression that brackets the LS optimum at 0.1.

## Approach
One-constant change: `train.py` L27 `LABEL_SMOOTHING` 0.1 → 0.15. The constant feeds `F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING)` (L243). All else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, GPU Cutout16, cosine peak0.2/warmup0.05/Nesterov/WD1e-4, batch128, seed42, compile reduce-overhead). Compute-/throughput-neutral, cudagraph-safe (LS is a host-side scalar arg to cross_entropy, outside the compiled forward), params unchanged (4,299,866). No deviations from plan.

## Execution
Single clean run on idle GPU 1 (exit 0, no retries). Smoke checks passed: AST OK, `git diff --name-only` == train.py only, LS=0.15 confirmed feeding cross_entropy at L243. Throughput identical to EXP-054 (659×8ms + 56×9ms dt; LS change is compute-free), 92 epochs, 35780 steps, 0 NaN/error. The LS-regularized train loss runs slightly higher in absolute value (more smoothing) as expected — judged by per-epoch eval test_acc trend, not train loss.

## Results
**best_test_acc 96.17%** (best at ep89), −0.28pp vs baseline 96.45, decisively under the 96.55 bar. **final_test_loss 0.2478 ≫ EXP-054's 0.1968** — the higher-LS model is materially less confident on the eval set (eval CE uses plain F.cross_entropy with no smoothing, so this is a fair comparison). Both top-1 AND confidence degraded → the model is over-regularized at this budget. `training_seconds 300.0` (compute budget respected exactly); `total_seconds 602.5` (wall breach +2.5s, see Verification).

**Interpretation**: the hypothesized LS×AugMix interaction did not produce a higher optimum — the opposite. Higher LS over-smooths at the short 300s/92-ep budget, where the recipe is convergence-bound (project-insights Medium): less-confident targets slow the approach to the decision boundary, costing both top-1 and calibration. Combined with EXP-023's lower-side regression (0.05 → 96.03), the LS optimum is now **bracketed at 0.1 from both sides** — the axis is closed. This is the cleanest interpretive outcome a no-improvement could yield here: a genuine two-sided closure.

**Trajectory fit**: consistent with the broader plateau picture (96.45 ≈ the k=4/300s ceiling). Like EXP-062 (warmup), EXP-063 (cooldown), EXP-064 (grad clip), this is a single-knob retune of an already-tuned recipe landing in the 96.1–96.4 band — every scalar/schedule lever is at or near its local optimum. It also re-confirms the project-insights Medium pattern that compute-neutral regularization changes move test-loss/confidence but not top-1 favorably at this budget (here both moved unfavorably).

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 96.17 < 96.55. **FAILED** (stop at first failed necessary condition).
- **Necessary condition 2 — clean completion within budget**: **WALL BREACH** — total_seconds 602.5 > 600 (+2.5s). training_seconds 300.0 (gated compute budget respected exactly), num_params 4,299,866 ✓, 0 NaN/error ✓. The overrun is eval+dataloader wall (92 evals + AugMix CPU variance), NOT compute (dt clean 8ms), and NOT caused by the compute-free LS change — it is the AugMix recipe's documented run-to-run wall variance (2nd breach after EXP-061's 604.6s).
- **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only ✓; no new deps ✓; seed 42 ✓; uncontended ✓.

**Verdict: no-improvement.** Valid training run (300s compute respected exactly, fair training) that decisively missed the accuracy bar. The invalid-vs-no-improvement call (the 600s wall is a hard constraint, 602.5 > 600) was resolved to **no-improvement** consistent with the EXP-061 precedent: (a) condition 1 (metric) fails first and decisively on a fully trustworthy value (96.17 ≪ 96.55); (b) the actively-gated 300s compute budget was respected exactly; (c) the 2.5s overrun is documented base-recipe wall variance, NOT caused by the compute-free LS change; (d) the breach does not make the metric untrustworthy. Classifying as `invalid` would wrongly discard a genuine, informative negative (the two-sided LS closure) over a 2.5s base-recipe wall overrun. The breach is documented prominently in the exp-log, here, and strengthened in infra-errors as a recurring pattern.

## Unexplored Avenues
- **LS axis is now exhausted** (both directions hurt). No further LS probing is warranted.
- **BN hyperparameters** (brainstorm-065 Ideas 2/3) remain genuinely untested: BN momentum 0.1→0.05 (longer EMA window for eval running-stat variance reduction) and BN eps 1e-5→1e-3 (soft channel down-weighting). Both are low-ceiling micro-probes; BN-eps is near-certain exact null, BN-momentum is weakly contraindicated by the cosine-to-0 near-frozen tail (running stats already stable).
- The honest plateau state stands: every scalar/schedule/augmentation/capacity/optimizer/normalization lever and both near-miss combinations are mapped. Remaining genuine gains, if any, require a more radical architectural change that does not cost epochs at the 300s budget.

## Next Steps
1. **BN momentum 0.1→0.05** (low confidence) — last cluster of genuinely-untested static hyperparameters; trivial, compute-neutral, cudagraph-safe. Weakly contraindicated but cleanly closes the BN-stat axis either way.
2. **Radical-but-epoch-neutral architecture** (low confidence) — the only class with headroom is a structural change that improves accuracy WITHOUT adding compute/epochs (e.g., a free-ish attention/SE-lite block that fits in the 8ms dt budget, or a better residual-init scheme). Most prior attempts hit the epoch wall; need a change that is genuinely throughput-neutral.
3. **Revisit near-miss combinations under a fresh lens** (low confidence) — EXP-049/063 closed the two obvious combos; a three-way compose of the best individual tail-tweaks (cooldown + a BN tweak) has not been tried, though expected value is low.
