# Report EXP-002: k=3 Width + Dynamic T_max + CutMix
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Log**: logs/exp-log-002.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better). Baseline: 94.03% (EXP-001, commit 30b6e12).

## Idea & Hypothesis

Widen to k=3 ({48,96,192}, 2.4M params), add dynamic T_max calibration from epoch 1 timing, replace CutOut with CutMix. Hypothesis: 95-96% from more capacity + fixed T_max + better augmentation.

## Approach

WIDTH_MULT=3, CutMix(alpha=1.0, p=0.5) replacing CutOut, dynamic T_max computed after epoch 1 completion. All other settings preserved from EXP-001.

## Execution

Single run, completed without crashes. However, the dynamic T_max calibration failed catastrophically: epoch 1 took 21.6s (inflated by torch.compile JIT), leading to T_max=10. Actual steady-state was ~4.8s/epoch with 62 epochs total. The cosine schedule completed at epoch ~15, leaving LR at minimum for 47 epochs.

## Results

- **Primary metric**: 94.09% (baseline: 94.03%, delta: +0.06%, below 0.1% threshold)
- **Observations**: Despite k=3 having 2.25x more capacity than k=2, the broken T_max completely negated the benefit. The model peaked at 94.09% around epoch 15 and degraded to 90.53% — the worst best/final gap yet (3.56%). The T_max calibration measured epoch 1 at 21.6s, not realizing torch.compile's first-epoch JIT overhead inflates timing by ~4.5x.
- **Analysis**: This experiment confirms that T_max alignment is critical — even with significantly more capacity, wrong T_max prevents the model from leveraging it. The k=3 model at 62 epochs with proper T_max would have had ample training time. CutMix's effect cannot be evaluated due to the confounding T_max issue.
- **Key Learning**: Never calibrate T_max from epoch 1 when torch.compile is active — epoch 1 includes JIT compilation overhead (~4.5x slower than steady-state). Calibrate from epoch 2 or average epochs 2-3.

## Verification

- **Conditions**: FAILED at condition 3 (94.09% < 94.13% threshold)
- **Review Notes**: Failure is real — caused by broken T_max calibration, not measurement error.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric did not exceed baseline + 0.1% threshold.

## Unexplored Avenues

- **Fix calibration: use epoch 2+ timing**: The core idea (k=3 + dynamic T_max + CutMix) is sound. The failure was purely in the calibration implementation. Using epoch 2 or average of epochs 2-3 for timing would give accurate T_max.
- **Static T_max from EXP-001 data**: Since k=2 got 78 epochs and k=3 got 62, we can directly use T_max=57 (62 - warmup) for k=3 without dynamic calibration.

## Next Steps

1. **k=3 with fixed T_max=57 + CutMix** (high confidence): Same experiment but with correct static T_max based on the 62 epochs we now know k=3 achieves. This isolates the width+CutMix effect cleanly.
2. **Fix dynamic calibration (epoch 2 based)** (medium confidence): Useful infrastructure but less critical than getting k=3 results first.

## Exit Action Results
