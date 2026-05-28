# Report EXP-004: Nesterov Momentum + Label Smoothing 0.1
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md
- **Log**: logs/exp-log-004.md

## Goal

Maximize best_test_acc (%) on CIFAR-10, higher is better. Baseline: 93.33% (EXP-003). Threshold: >= 93.43%.

## Idea & Hypothesis

**Chosen idea**: Add nesterov=True to SGD and label_smoothing=0.1 to cross_entropy. Recipe polish.

**Hypothesis**: Expected 93.5-93.8% (+0.2-0.5pp) from improved gradient estimation and confidence calibration.

## Approach

Two keyword argument changes to train.py: `nesterov=True` on SGD, `label_smoothing=0.1` on F.cross_entropy. No other changes.

## Execution

Single run, 65 epochs (25,126 steps) in 300.0s. Total 363.6s. No errors.

## Results

- **Primary metric**: 93.28% (baseline: 93.33%, delta: -0.05pp)
- **Observations**:
  - 4 fewer epochs than EXP-003 (65 vs 69) — per-step time increased from ~11.3ms to ~11.9ms
  - Nesterov improved the high-LR phase: best pre-drop was 89.69% (vs EXP-003's 81.15%)
  - But the model peaked at epoch 60 (93.28%) and then *declined* to 93.11% by epoch 65
  - The epoch-count reduction combined with the earlier peak resulted in a net loss vs baseline
- **Analysis**: The hypothesis failed. While Nesterov clearly helped convergence speed in the high-LR phase (+8.5pp better pre-drop accuracy), the combination with label smoothing had two negative effects: (1) per-step overhead cost 4 epochs, and (2) the model peaked earlier and lower. The epoch loss is likely the dominant factor — EXP-003 gained +0.34pp in its final 4 epochs (93.33% was the last-epoch result). Label smoothing makes training harder without adding the data diversity that augmentation provides, so the model reaches a lower ceiling in fewer epochs. The Nesterov overhead (extra buffer ops per step) contributed to the epoch loss.
- **Key Learning**: In the time-budgeted regime (300s), per-step overhead matters — 4 fewer epochs cost more than the convergence-quality gain from Nesterov + label smoothing. Future experiments should prefer changes that are throughput-neutral or throughput-positive.

## Verification

- **Conditions**: Condition 1 FAILED — 93.28% < 93.43%. Conditions 2-3 skipped.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric below baseline.

## Unexplored Avenues

- **Nesterov alone** (without label smoothing): may retain the convergence benefit without the LS-induced ceiling reduction
- **Label smoothing with lower value** (0.05): less aggressive, may not reduce the ceiling as much
- **Throughput-first approach**: AMP or batch-size scaling to get more epochs, then add Nesterov

## Next Steps

1. **AMP (torch.cuda.amp) for throughput** — the binding constraint is epochs; more epochs = more accuracy. High confidence this is the right axis.
2. **Nesterov alone** (without label smoothing) — isolate the Nesterov effect. Medium confidence.
3. **Batch size 256 + LR scaling** — alternative throughput path. Medium confidence.

## Exit Action Results
(no exit actions defined)
