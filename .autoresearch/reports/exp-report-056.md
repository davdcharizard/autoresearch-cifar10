# Report EXP-056: GPU-batched diverse augmentation (affine + photometric), full-coverage — the throughput unlock
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-056.md
- **Plan**: plans/plan-056.md
- **Log**: logs/exp-log-056.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the 300s Σdt budget on a single H20. Baseline = **96.45%** (EXP-054, 86161d9); bar = **96.55%**. This loop pivoted: the CPU-side augmentation-diversity lever was mapped to its wall-limited frontier (EXP-055), so this attempted the GPU-augmentation throughput unlock — move diverse augmentation off the starved 8-worker CPU dataloader onto the idle GPU to afford full coverage.

## Idea & Hypothesis
Chosen idea: add a vectorized `gpu_augment(x)` in the train loop (per-sample random affine — rotation ±12°/shear ±0.1/scale [0.9,1.1] via `affine_grid`+`grid_sample` — plus photometric brightness ±0.1 / contrast [0.85,1.15]), full-coverage, and REMOVE the CPU `RandomApply([AugMix()])` (lighten CPU to crop+flip). Reasoning: augmentation diversity is the only lever that lifts top-1 here, it's CPU-wall-limited, and the GPU is idle ~6-7ms/step; EXP-003 (Cutout CPU→GPU, +0.58pp) is direct precedent. Hypothesis: dt stays within ~1-2ms of 8ms (epochs ≥ ~80), wall < 600s, and full-coverage GPU diversity ≥ the 50%-subset CPU AugMix → best_test_acc ≥ 96.55.

## Approach
Three edits to train.py: new `gpu_augment` function (after `cutout_batch`), wired into the loop before `cutout_batch`, and CPU AugMix removed. Magnitudes kept gentle (EXP-053: aug magnitude is interior-optimal). affine_grid/grid_sample are core torch (no new dep). Smoke: gpu_augment ~0.52ms/batch standalone, output (128,3,32,32) finite float32 channels_last; num_params unchanged.

## Execution
- **Run 1 (GPU 0) ABORTED — infra contention** (NOT a code/research failure): another user's process (PID 194920, 1.4GB, 85% util) landed on GPU 0 after launch + CPU load avg ~8 (competing `pdflatex`) starved the dataloader → wall/Σdt ≈ 10× (384s wall vs 36s Σdt at step 2450), projecting ~3200s ≫ 600s. Aborted (TaskStop); the known shared-node hazard (goal-learnings: fair runs require an idle GPU).
- **Run 2 (GPU 1, idle) — clean**: exit 0, 390.5s wall, uncontended. Early gate passed: dt steady ~9ms (+1ms gpu_augment premium), wall/Σdt 1.39× (healthy), projected ~78 epochs. Completed at **84 epochs** / 32,621 steps. No NaN.

## Results
- **Primary metric**: best_test_acc **95.39%** (baseline 96.45, delta **−1.06pp**) — a LARGE regression, below even the pre-AugMix 96.00 (EXP-012 era). final 95.27%; final_test_loss **0.2240** (≫ EXP-054's 0.1968).
- **Observations**:
  - dt steady ~9ms (635×9ms) — the GPU-aug premium is only ~1ms; **the throughput unlock works as designed** (cheap, full-coverage feasible). num_epochs 84 (vs 91 baseline). Wall 390.5s, dt-bound (CPU no longer starves). peak_vram 452.9 MB unchanged.
  - The ~7-epoch loss cannot explain −1.06pp — the dominant cause is the augmentation POLICY.
- **Analysis**: Hypothesis REJECTED, and informatively so. The GPU-augmentation INFRASTRUCTURE is validated (cheap ~1ms dt, correct, full-coverage, uncontended) — the failure is the POLICY, not the mechanism. `gpu_augment` applied FIVE simultaneous distortions (rotate AND shear AND scale AND brightness AND contrast), stacked, to 100% of images every step, with NO clean-image convex mixing. This is far harsher than the augmentations that WORK here: TrivialAugment applies ONE random op per image (→ 96.22, EXP-012), and AugMix convex-mixes augmented chains with the ORIGINAL clean image (bounding the distribution shift) on a SUBSET (→ 96.45, EXP-054). Stacking all ops compounds into a large per-image distribution shift; the high test loss (0.224) is the signature of a model fighting an over-distorted training distribution it can't reconcile with the clean test set. The result localizes the problem precisely: the GPU path is the right direction, but the policy must BOUND the per-image shift (single-op selection, or clean-image mixing, and/or stochastic <100% coverage).
- **Key Learning**: The GPU-augmentation throughput unlock is feasible and cheap (~1ms dt premium, full-coverage, 84 ep) — but a naive full-coverage STACK of affine+photometric (5 simultaneous distortions, no clean-image mix) is far too harsh → 95.39 (−1.06pp). Working augmentations bound the per-image shift (TA: 1 op; AugMix: convex clean-mix).

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.55) **FAILED** — 95.39 (−1.06pp). Remaining skipped (informationally pass: 390.5s < 600, params 4,299,866, no NaN, scope train.py-only, no new deps, seed 42).
- **Review Notes**: Trustworthy — Run 2 clean and uncontended (GPU 1 solo; verified via nvidia-smi), dt steady, regression far exceeds the noise band. Run 1's contention was correctly identified and excluded (relaunched on idle GPU). No integrity/scope concerns. The regression is a genuine policy effect, not measurement.
- **Verdict**: no-improvement
- **Verdict Basis**: necessary Cond 1 failed; primary metric 95.39 regressed −1.06pp below baseline. Valid run, no constraint violation → no-improvement.

## Unexplored Avenues
- **GPU AugMix-style clean-image mixing**: `out = m*clean + (1-m)*augmented`, m~Beta, on the GPU-augmented batch — replicates AugMix's shift-bounding property (the likely reason CPU AugMix worked and this didn't). Highest-value next step; reuses the validated gpu_augment infra.
- **GPU TrivialAugment-style single-op selection**: per image, pick ONE random op (rotate OR shear OR scale OR brightness OR contrast) at random magnitude, instead of stacking all five. Matches the 96.22 TA recipe's structure, on GPU at full coverage.
- **Stochastic coverage (RandomApply-style p<1) on the GPU stack**: apply the full gpu_augment to only a random ~30-50% of each batch (like EXP-054's p=0.5), leaving the rest crop+flip only.
- **Gentler/fewer ops**: drop photometric, keep only mild affine; or shrink magnitudes further.

## Next Steps
- **GPU AugMix-style clean-mix (or single-op TA-style) on the validated gpu_augment infra**: the infrastructure works; bound the per-image shift via convex clean-mixing or one-op-per-image. This is the direct fix for the EXP-056 failure mode and the real test of the throughput unlock. (high)
- **Stochastic coverage p≈0.5 on the GPU stack**: cheap variant — mirror the proven EXP-054 subset structure but full-strength on the GPU subset, since GPU cost is no longer the constraint. (medium)
- **If GPU-aug policies stall too, the lever is genuinely exhausted**: revert to consolidating 96.45 (replicate) and accept the k=4/300s ceiling, or attempt a more radical architecture change. (low)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
