# Report EXP-003: Weight Decay 5e-4 on Width-2x Augmented Baseline
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Log**: logs/exp-log-003.md

## Goal

Maximize best_test_acc (%) on CIFAR-10, higher is better. Baseline: 92.92% (EXP-002). Threshold: >= 93.02%.

## Idea & Hypothesis

**Chosen idea**: Increase WEIGHT_DECAY from 1e-4 to 5e-4, aligning with the WRN paper's standard for wider CIFAR-10 models.

**Hypothesis**: best_test_acc would reach 93.1-93.5% (+0.2-0.6pp) through stronger L2 regularization compounding with augmentation.

## Approach

Single constant change: `WEIGHT_DECAY = 1e-4` → `WEIGHT_DECAY = 5e-4` in train.py. All other settings unchanged.

## Execution

Single run, 69 epochs (26,608 steps) in 300.0s. Total 355.4s. No errors or adjustments.

## Results

- **Primary metric**: 93.33% (baseline: 92.92%, delta: +0.41pp, +0.44%)
- **Observations**:
  - Pre-LR-drop accuracy lower than EXP-002 (81.15% vs 86.45% at epoch 34) — stronger WD makes high-LR phase harder
  - Post-second-LR-drop: dramatic convergence from 91.53% to 93.33% — the second drop delivered +1.8pp (vs EXP-002's +0.52pp, EXP-001's +0.02pp)
  - Accuracy climbed steadily through the final 5 epochs (92.99→93.33), suggesting the model was still improving at budget end
  - Peak VRAM, step time, epoch count all unchanged from EXP-002
- **Analysis**: The hypothesis was confirmed — WD=5e-4 lands at 93.33%, within the predicted 93.1-93.5% band. The most striking finding is the escalating importance of the second LR drop: EXP-001 (no aug, WD=1e-4) +0.02pp → EXP-002 (aug, WD=1e-4) +0.52pp → EXP-003 (aug, WD=5e-4) +1.8pp. Stronger regularization (both augmentation and WD) creates a larger optimization gap in the second LR plateau that the 0.001 LR phase closes. The total gain from BASE is now +1.61pp (91.72→93.33%), with the WRN paper's WRN-16-2 anchor of 93.2% now exceeded. The model is still improving at epoch 69 — more training time would likely push further.
- **Key Learning**: WD=5e-4 adds +0.41pp and dramatically amplifies the second LR drop's contribution (+1.8pp vs +0.52pp), confirming WD and augmentation create synergistic regularization pressure. The model is still converging at budget end — throughput improvements could unlock more gains.

## Verification

- **Conditions**: All 3 passed
- **Verdict**: improvement
- **Verdict Basis**: 93.33% > 93.02% threshold; +0.41pp above baseline.

## Unexplored Avenues

- **Nesterov momentum**: Free +0.1-0.3pp, orthogonal to WD.
- **Label smoothing 0.1**: Another regularization axis.
- **AMP + channels_last**: More epochs in budget — the model was still improving at epoch 69.
- **WD schedule (cosine decay)**: Instead of constant WD=5e-4, decay WD during training.
- **Batch size 256 + LR 0.2**: Linear scaling to increase throughput.

## Next Steps

1. **Nesterov momentum** — free orthogonal improvement, one keyword change. High confidence.
2. **AMP for throughput** — more epochs when the model is still improving at budget end. Medium-high confidence.
3. **Label smoothing 0.1** — third regularization axis after augmentation and WD. Medium confidence.

## Exit Action Results
(no exit actions defined)
