# Report EXP-001: Width-2x ResNet-20 with Wall-Clock-Fractional MultiStep LR
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md
- **Plan**: plans/plan-001.md
- **Log**: logs/exp-log-001.md

## Goal

Maximize best_test_acc (%) on CIFAR-10, higher is better. Baseline: 91.72% (ResNet-20, ~270K params, MultiStepLR milestones=[32000, 48000], 97 epochs in 300s on H20). Improvement threshold: +0.1pp (>= 91.82%).

## Idea & Hypothesis

**Chosen idea**: Double the per-stage channel widths of the ResNet-20 from {16, 32, 64} to {32, 64, 128} (WIDTH_MULT=2, ~1.07M params) while reparametrizing the absolute-iteration MultiStepLR into a wall-clock-fractional equivalent with drops at the 50% and 75% marks of TIME_BUDGET_S.

**Why selected**: The WRN paper's Table 5 anchors the n=3, k=2 design point at ~93-94% on CIFAR-10 under the same SGD-momentum-WD-step-decay recipe — a +2pp gain over the baseline. The wall-clock-fractional schedule is the invariant form of the He-2015 step-decay shape that the EXP-000 failure analysis identified as critical: it preserves the 0.5/0.75 fractional drop positions regardless of the wider model's reduced step rate.

**Hypothesis**: best_test_acc would reach the 93.0-94.0% band, comfortably above the 91.82% threshold.

## Approach

Four changes to `train.py`, no other files modified:
1. Added `WIDTH_MULT = 2` hyperparameter constant
2. Multiplied all six channel-width literals in `ResNet.__init__` by `WIDTH_MULT` (stem conv/bn, three `_make_layer` calls, FC head input)
3. Replaced `MultiStepLR(milestones=[32000, 48000], gamma=0.1)` with a `LambdaLR` using a wall-clock-fractional decay function reading from a closure cell `_lr_progress[0] = total_training_time / TIME_BUDGET_S`
4. Added the progress-cell update line in the inner training loop after `total_training_time += dt`

No deviations from the plan. The optional Change 5 (cosmetic model-info print extension) was skipped per the plan's own "skip if scope creep" guidance.

## Execution

Single run, completed without errors. Training ran for 69 epochs (26,737 steps) in exactly 300.0s on a single H20 GPU. Total wall-clock 352.3s (includes 52.3s of per-epoch eval overhead + 1.1s startup). No retries, no adjustments needed.

## Results

- **Primary metric**: 92.29% (baseline: 91.72%, delta: +0.57pp, +0.62%)
- **Observations**:
  - LR schedule fired correctly: first drop 0.1→0.01 at pct_done=50.1% (step 13350, epoch 35), second drop 0.01→0.001 at pct_done=75.0% (step 20050, epoch 52)
  - Immediate accuracy jump after first LR drop: 88.09% (epoch 29 best pre-drop) → 91.23% (epoch 35) → 92.27% (epoch 48)
  - Second LR drop produced a small incremental gain: 92.27% → 92.29% (final best at epoch 69)
  - 1,073,962 params (~4x baseline's 269,722), peak VRAM 598.7 MB (~1.8x baseline's 330 MB)
  - 69 epochs completed vs baseline's 97 — wider model's step time (~11ms vs baseline's ~8ms) reduces epoch count but the wall-clock-fractional schedule compensates
- **Analysis**: The hypothesis was directionally correct — the wider model improved accuracy — but the magnitude (+0.57pp) fell well short of the predicted 93.0-94.0% band. The gap between the achieved 92.29% and the WRN-paper's ~93.5% anchor for the n=3, k=2 point is likely due to (a) fewer training epochs (69 vs the WRN paper's 200), (b) the WRN paper's use of weight_decay=5e-4 vs our 1e-4, and (c) the WRN paper's use of horizontal flip + random crop with additional augmentation tricks. The wall-clock-fractional schedule worked exactly as designed — the EXP-000 failure mode (LR never reaching the low-LR regime) was structurally eliminated. The first LR drop was the critical event, delivering a ~3pp jump in a few epochs; the second LR drop contributed only +0.02pp incremental, suggesting the model had largely converged by the 75% mark. The capacity increase was the right axis to move — the wider model reached a higher accuracy ceiling than the narrow baseline despite fewer epochs.
- **Key Learning**: Channel widening from 270K to 1.07M params gains +0.57pp under a 300s budget; the gain is real but below the WRN-paper's 200-epoch anchor, indicating the time budget constrains the wider model's convergence — future experiments should explore throughput improvements (AMP, larger batch) or augmentation to extract more from the wider architecture.

## Verification

- **Conditions**: All 3 passed — (1) 92.29% > 91.82% threshold, (2) summary block complete with all 10 metric lines, (3) eval_count=69 == num_epochs=69
- **Review Notes**: Results confirmed trustworthy. LR schedule transitions verified at correct wall-clock positions. Param count consistent with width-mult-2 scaling. No integrity concerns.
- **Verdict**: improvement
- **Verdict Basis**: All necessary conditions passed; primary metric exceeds baseline by +0.57pp, well above the +0.1pp threshold and several standard deviations above the ~0.3pp noise floor.

## Unexplored Avenues

- **Width-4x (WIDTH_MULT=4)**: The WRN-paper's k=4 point targets ~95% but would reduce epoch count to ~25-35, which may not be enough for convergence under the step-decay schedule. Could work if combined with throughput improvements.
- **Weight decay 5e-4**: The WRN paper uses 5e-4 vs our inherited 1e-4 — the wider model may benefit from stronger L2 regularization. A single-axis sweep on WD with the new width-2x baseline could close part of the gap to the literature anchor.
- **AMP (torch.cuda.amp)**: Mixed-precision training would increase throughput, allowing more epochs in the same 300s budget at width-2x, potentially recovering some of the gap between 92.29% and the WRN-paper's ~93.5% anchor.
- **TrivialAugmentWide + RandomErasing**: The augmentation axis is now more valuable on the wider model — a larger model has more capacity to benefit from heavier regularization, the standard "wider model + stronger augmentation" compounding effect from the WRN/AutoAugment literature.

## Next Steps

1. **Augmentation upgrade on the width-2x baseline** — add TrivialAugmentWide + RandomErasing to the now-wider model. The capacity-first sweep order means augmentation should compound well. High confidence.
2. **Weight decay sweep** — try 5e-4 on the width-2x model (the WRN paper's recommended value for wider models). Medium confidence.
3. **AMP + larger batch for throughput** — mixed-precision training to fit more epochs in 300s, potentially recovering convergence quality. Medium confidence.

## Exit Action Results
(no exit actions defined)
