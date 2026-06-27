# Report EXP-046: Anti-aliased shortcut — avg-pool the identity path at stage transitions
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-046.md
- **Plan**: plans/plan-046.md
- **Log**: logs/exp-log-046.md

## Goal

Maximize CIFAR-10 best_test_acc (%, higher is better) within the fixed 300s charged training budget by modifying train.py only. Baseline at analysis time: **96.71** @ 1990397 (EXP-006 recipe); improvement bar ≥ 96.81. Noise context (EXP-027): baseline mean ≈ 96.57, σ ≈ 0.16.

## Idea & Hypothesis

After EXP-044/045 closed the capacity class in every currency and certified the recipe as audit-complete, the program's frontier is radical-structural candidates passing the full screen stack. The chosen idea: the 2016 pad shortcut downsamples by strided slice `x[:, :, ::2, ::2]`, discarding 75% of identity-path samples and aliasing the rest, at both stage transitions. Replace the slice with `F.avg_pool2d(shortcut, 2)` — the 2×2 box filter, simplest anti-aliasing, information-preserving — keeping the channel zero-pad (EXP-020 proved that part harmless). Zero params, zero learnable state, expected ~zero dt: the ONLY candidate free in every measured currency. External anchors: ResNet-D (Bag of Tricks, knowledge/papers/bag-of-tricks-zero-gamma.md) and anti-aliased CNNs (Zhang 2019, arXiv 1904.11486), each worth +0.5–1.0 fixed-epoch with weak aug.

Hypothesis: preserving identity-path information raises the converged plateau — best_test_acc ≥ 96.81 at baseline-identical signatures. Pre-registered branches: (i) ≥ 96.81 improvement (replicate pair for 96.70–96.80); (ii) read within mean ± 0.15 (96.42–96.72) at family test_loss → gain absorbed/insufficient under TA+RE, downsample-quality class closes as the 14th external-transfer failure; (iii) GATE_KILL D0 > 26ms → kernel mispricing, invalid.

## Approach

One-line logic change in `train.py` `BasicBlock.forward`: `shortcut = shortcut[:, :, ::self.stride, ::self.stride]` → `if self.stride != 1: shortcut = F.avg_pool2d(shortcut, self.stride)`; zero-pad line and everything else untouched. Affects exactly layer2[0] and layer3[0]. CPU sanity passed all checks: params exactly 4,286,026 (zero-param confirmed); need_pad only at the two transition blocks; constant-input shortcut equivalence + random-input difference (anti-aliasing active); stride-1 blocks bit-identical; 2-step train smoke with decreasing loss. No deviations from plan.

## Execution

Single run via `/tmp/exp046_composite.sh` (exp045 script, thresholds 31→26). Dual gates cleared at poll 1 (apps=0, load=6). GATE_DECISION D0=22.5ms, projected 137 epochs, contention threshold 28.1ms — branch (iii) eliminated; avg_pool2d prices at ~zero under compile+channels_last+bf16, as expected for a stock dense-regime kernel. Pristine run end to end: all 30 windows 22.0–22.7ms, slow_streak never tripped, PROC_EXITED tick 33, rc=0, total 485.3s. No retries, no errors.

## Results

- **Primary metric**: best_test_acc 96.65 (baseline: 96.71, delta: −0.06, −0.06%)
- **Observations**: Signatures byte-identical to the baseline family — 139 epochs (baseline ~139), dt 22.0–22.7ms, startup 19.2s, peak VRAM 1613MB (baseline ~1600), params 4,286,026. Plateau: last 8 evals 96.51–96.65, final_test_loss 0.1880 — exactly the family level (~0.185). The read 96.65 = mean + 0.5σ.
- **Analysis**: This is pre-registered branch (ii) to the letter: the anti-aliasing gain is fully absorbed under TrivialAugment+RandomErasing — no plateau-level shift, no test-loss shift, no epoch/dt cost. The mechanism story mirrors the SE/SAM/LS absorption nulls: published gains measured under weak aug at fixed epochs are regularization-flavored (or supply information the heavy-aug ensemble already supplies), and the heavy-aug recipe absorbs them to a precise zero. Notably this candidate was free in EVERY currency — the first of the 14 external transfers that paid no toll anywhere — and STILL nulled. That sharpens the absorption law: it is not that external techniques fail because of deferral/dt costs; even cost-free ones add nothing once the recipe is in the heavy-aug converged regime. The shortcut-slice "defect" is evidently not a binding constraint on this network's plateau.
- **Key Learning**: Even a zero-cost, screen-stack-clean architectural fix (anti-aliased shortcut) absorbs to an exact null under the heavy-aug recipe — external fixed-epoch evidence is now 0-for-14, including the toll-free case.

## Verification

- **Conditions**: Integrity pre-condition PASSED (pristine profile, 139 epochs in [130,142], params exact, 300.0s charged, evals ≤ epochs). Condition 1 (best_test_acc ≥ 96.81) FAILED: 96.65, below the 96.70 replicate-band floor so no replicate pair was triggered. Conditions 2–3 skipped per first-failure-stop (informationally both would pass: rc=0, 485.3s ≤ 600; 139 evals ≤ 139 epochs).
- **Review Notes**: Results confirmed trustworthy — metric grepped directly from run.log of a fully integrity-checked run; no contention, no parsing ambiguity; eval contract untouched (eager base_model, Eval.evaluate() unmodified).
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure — valid, pristine result that did not meet the bar.

## Unexplored Avenues

- **Blurpool (conv-path anti-aliasing, Zhang 2019)**: the bigger dose of the same mechanism — but it requires grouped depthwise kernels (measured 2.5–3× dense, EXP-042) plus full-res stride-1 convs at transitions, pricing ≥ +3–5ms before any gain, and EXP-046 just showed the shortcut-path dose of the same mechanism absorbs to zero. The mechanism class is now evidenced-against in-regime; treat as closed unless a free variant appears.
- **Stronger shortcut filters (3×3 blur instead of 2×2 box)**: free of params but the box filter already nulled; no reason a slightly better filter escapes absorption when the simplest one shows zero slope.

## Next Steps

1. **Update the absorption law to its strongest form** (high confidence): EXP-046 proves absorption is not cost-accounting — even toll-free external techniques null. Future candidates must argue a mechanism the heavy-aug ensemble *cannot* supply (information bottlenecks, objective geometry at the optimum), not merely freedom from costs.
2. **Brainstorm-047 should probe the few remaining never-dosed categories under the tightened screen** (medium confidence): e.g., training-only structural asymmetries that vanish at eval (the model is identical at eval time; e.g., stochastic depth was never dosed — though it is regularization-flavored and likely absorbs), or revisit near-misses for combination (EXP-027 replicate machinery makes small true effects detectable only if ≥ +0.3).
3. **Treat the baseline as plausibly at the recipe's ceiling** (medium confidence): 40 consecutive non-improvements with every class measured-closed; the honest remaining moves are constructions targeting what augmentation cannot emulate.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
