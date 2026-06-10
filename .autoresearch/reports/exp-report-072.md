# Experiment Report EXP-072: AugMix `all_ops=False` — geometric-only AugMix (drop 4 photometric ops)

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within Σdt=300s GPU-compute + total wall ≤ 600s (single H20). Baseline at experiment time: **96.45** (EXP-054, commit 86161d9). Bar: **96.55**.

## Idea & Hypothesis
**Chosen idea**: Set `transforms.AugMix(all_ops=False)` at train.py L171, restricting the AugMix per-chain op pool from the torchvision-default 13 ops to the original-paper 9 (ShearX/Y, TranslateX/Y, Rotate, Posterize, Solarize, AutoContrast, Equalize) — REMOVING the 4 magnitude-photometric ops Brightness, Color, Contrast, Sharpness. Single-variable; everything else byte-identical to EXP-054. **Mechanism**: the op-SET composition is the one untested AugMix dimension (prior experiments varied chain-count EXP-052/054/055, magnitude EXP-053, mix-alpha EXP-069, coverage EXP-054/055/057, but never WHICH ops are in the menu). The photometric ops distort class-discriminative CIFAR-10 color cues; removing them could yield cleaner label-preserving diversity (the proven lever). Circumstantial support: the three color-op-inclusive policy families (TrivialAugment EXP-012, RandAugment EXP-014, AutoAugment EXP-060) all tied at 96.22 < AugMix's 96.45.

**Hypothesis**: best_test_acc within ±0.3pp of 96.45; a genuine ~25-35% upside shot at clearing 96.55 if photometric ops were net-harmful, else a small change mapping the op-set axis.

## Approach
One-line edit: `transforms.AugMix()` → `transforms.AugMix(all_ops=False)`, kept inside `RandomApply(..., p=0.5)`, all AugMix scalars fixed (severity 3, width 3, depth -1, alpha 1.0). Pre-launch smoke confirmed the op pool drops to exactly the 9 geometric/lossless ops with all 4 photometric ops absent, the transform applies cleanly to a 32×32 image, and `git diff --name-only` == train.py only. No deviations.

## Execution
Single clean run on idle GPU 1 (both GPUs idle, uncontended), exit 0, no retries, 0 NaN/error. dt steady **8ms** (normal 9-12ms jitter) — AugMix runs in the CPU dataloader workers, off the timed step, so the op-menu change is throughput-neutral as predicted. Early gate healthy: eval climbed normally (ep4 66.77%, ep5 73.51%, ep6 77.23%), tracking EXP-054 with a marginally faster early-climb. **92 epochs** (vs EXP-054's 91 — geometric-only AugMix is slightly cheaper on CPU, buying ~1 extra epoch), training_seconds 300.0, total_seconds 570.3 < 600, peak_vram 453.8 MB.

## Results
**best_test_acc 96.43% (−0.02pp vs baseline 96.45) — a virtual TIE, the SMALLEST delta of any post-EXP-054 experiment** (18 prior misses ranged −0.04 to −9.45pp; the scalar-knob band is −0.2..−0.6pp). Simultaneously, **final_test_loss dropped to 0.1911** (from EXP-054's 0.1968) — near the best-ever polish runs (GC 0.1894, grad-clip 0.1939). This is the textbook **polish-vs-top1 signature** applied to augmentation: removing the 4 photometric ops measurably improved calibration/loss but left top-1 statistically flat. The hypothesis's "photometric ops are net-harmful → cleaner diversity clears the bar" was PARTIALLY borne out on the loss axis (they were indeed mildly net-negative for loss) but NOT on top-1 (the gain did not exceed the noise band, let alone reach 96.55).

**Interpretation**: the 4 photometric ops are NOT load-bearing for top-1 — the geometric+lossless ops carry essentially all of AugMix's diversity benefit, and the photometric ops were a near-neutral (slightly loss-negative) addition. Mechanistically this also explains why AugMix (96.45) beat the color-op-heavy TA/RA/AA policies (96.22): AugMix's edge is its multi-chain MIXING, and its photometric ops were a mild drag that the mixing overcame — removing them recovers a little loss but the top-1 ceiling is set by the mixing+geometric diversity, which is unchanged. The op-set-composition axis is therefore **flat near the optimum**: even deleting a third of the op vocabulary doesn't move top-1.

**Trajectory fit**: 19th consecutive no-improvement since EXP-054. The −0.02pp virtual-tie (with improved loss) is the strongest confirmation yet that the augmentation lever is genuinely exhausted at 96.45 — the recipe is robust to even structural op-menu changes. Reinforces the High-importance "convergence-bound, not overfit-bound" and "polish-vs-top1" project verdicts: a calibration improvement (lower loss) is once again a false-positive signal for top-1.

## Verification
- **Necessary condition 1 — `best_test_acc >= 96.55`**: 96.43 < 96.55. **FAILED** (−0.12pp below bar). Stop at first failed condition.
- **Necessary condition 2 — clean completion within budget** (for completeness): training_seconds 300.0 ✓, total_seconds 570.3 < 600 ✓, num_params 4,299,866 UNCHANGED ✓, 0 NaN/error ✓, 92 ep.
- **Necessary condition 3 — no hard-constraint violation**: train.py only ✓; no new deps ✓; seed 42 ✓; evaluate() once/epoch ✓; uncontended (dt 8ms) ✓.

**Verdict: no-improvement.** Clean valid run that missed the bar by a hair (96.43, within noise of baseline). Results trustworthy — direct parse, 0 NaN, healthy trajectory. NOT invalid (no breach; aug is data-side, params unchanged) and NOT crash (real interpretable metric).

## Unexplored Avenues
- **AugMix coverage p=0.5→0.6 / severity 3→2**: the remaining untested AugMix scalars (brainstorm Ideas 2/3). Both are off already-bracketed interior optima (coverage 50% beat 35%&100%; severity 3 beat 6) → near-certain scalar-knob nulls. Low priority — this run showing the op-SET is flat near optimum makes the scalars even less likely to move top-1.
- **Per-op pruning within the geometric set**: e.g., drop only Solarize/Posterize (the harshest color-channel ops) while keeping geometric. Finer-grained, but EXP-072 shows the op-set axis is flat → vanishing upside, not worth a loop.
- **The augmentation lever is now exhausted from every angle** — policy family (TA/RA/AA/AugMix), chain-count, magnitude, mix-alpha, coverage, delivery path (CPU/GPU), AND op-set composition. Do NOT propose further augmentation-policy changes; the only conceivable aug gain would require a fundamentally new augmentation MECHANISM not in the torchvision toolbox (and at the 300s/91-epoch budget, stronger aug underfits — EXP-018/057).

## Next Steps
1. **Accept 96.45 as the comprehensively-mapped k=4/300s ceiling** (high confidence) — 19 consecutive misses; every axis closed including, now, augmentation op-set composition. The −0.02pp virtual-tie is the strongest evidence yet that the recipe is at a robust optimum. Remaining probes are pure plateau-mapping.
2. **Nesterov on→off** (low confidence) — the last untouched optimizer boolean; near-certain small regression (Nesterov is the tuned setting), a clean axis-closer that cannot destabilize. Safest remaining untested knob.
3. **SGD momentum 0.9→0.95** (low confidence) — the one untested optimizer scalar, but contraindicated (≈doubles effective LR → likely a real regression per the closed peak-LR axis EXP-016). Expected to regress, not null.
