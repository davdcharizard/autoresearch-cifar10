# Report EXP-000: Training Recipe Modernization
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md
- **Plan**: plans/plan-000.md
- **Log**: logs/exp-log-000.md

## Goal

Maximize best_test_acc (%) on CIFAR-10, higher is better. Baseline: 91.72% (ResNet-20, 272K params, MultiStepLR, 97 epochs in 300s time budget). Improvement threshold: +0.1pp (i.e., >= 91.82%).

## Idea & Hypothesis

**Chosen idea**: Training Recipe Modernization — keep ResNet-20 architecture unchanged but overhaul the training recipe with cosine annealing LR, Cutout augmentation, label smoothing, and Nesterov momentum.

**Why selected**: The baseline LR schedule appeared fundamentally mismatched — MultiStepLR milestones at steps 32K/48K were designed for 64K-step training, but only ~38K steps complete in 300s. Cosine annealing naturally adapts to actual training duration. Cutout and label smoothing are individually proven to add 0.5-1pp each on CIFAR-10 and compound when combined.

**Hypothesis**: These changes would improve best_test_acc from 91.72% to approximately 93-94%, primarily driven by the LR schedule fix.

## Approach

Five changes to `train.py`, no architecture modifications:
1. Replaced `MultiStepLR(milestones=[32000, 48000], gamma=0.1)` with epoch-based `CosineAnnealingLR(T_max=200)`, moving `scheduler.step()` from per-batch to per-epoch
2. Added `Cutout(n_holes=1, length=16)` transform after normalization
3. Added `label_smoothing=0.1` to `F.cross_entropy`
4. Enabled `nesterov=True` on SGD optimizer
5. Removed `MAX_STEPS = 64000` cap — time budget is sole stopping criterion

No deviations from the plan.

## Execution

Single run, completed without errors. Training ran for 91 epochs (35,215 steps) in 300s on a single H20 GPU. No retries, no adjustments needed. The script completed normally and printed the full summary block.

## Results

- **Primary metric**: 88.79% (baseline: 91.72%, delta: -2.93pp, -3.19%)
- **Observations**: LR at epoch 90 was still 0.058 — barely decayed from the initial 0.1. Training loss plateaued at ~0.79-0.81 from epoch ~60 onward, never converging. The model completed fewer epochs (91 vs 97 baseline) due to slight Cutout overhead. Peak VRAM unchanged at 330MB.
- **Analysis**: The hypothesis was wrong in a critical way. The brainstorm analysis correctly identified that the second MultiStepLR milestone (48K) never fires, but incorrectly concluded the schedule was "fundamentally mismatched." In fact, the first step-drop at step 32,000 (epoch ~82) DID fire and was critical — it dropped LR from 0.1 to 0.01, enabling the final convergence phase. CosineAnnealingLR with T_max=200 and only 91 epochs completed barely reached LR=0.058 — never providing the low-LR regime needed. Meanwhile Cutout and label smoothing increased training difficulty, requiring even more aggressive LR decay to compensate.
- **Key Learning**: LR schedule T_max must match actual training duration; T_max=200 with 91 actual epochs keeps LR too high throughout training and prevents convergence.

## Verification

- **Conditions**: Condition 1 (best_test_acc >= 91.82%) FAILED — 88.79% is 2.93pp below baseline. Conditions 2-3 skipped.
- **Review Notes**: Results confirmed trustworthy — training completed normally, metrics are plausible, no integrity concerns.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric significantly below baseline; no hard constraint violations.

## Unexplored Avenues

- **Cosine annealing with T_max=100 (matched to actual epochs)**: The core cosine annealing idea is sound but T_max was miscalibrated. With T_max~100, the LR would reach near-zero by end of training, providing the low-LR convergence phase the model needs. This is the most direct fix.
- **Step-based cosine annealing instead of epoch-based**: Using per-step cosine scheduling with T_max matched to actual step count (~35K-38K) would provide smoother LR decay and is how many modern recipes implement it.
- **Isolate each change**: The 5 simultaneous changes make attribution impossible. Testing Cutout, label smoothing, or Nesterov individually on top of the baseline (keeping the working MultiStepLR) would reveal which techniques actually help at this model scale.
- **Warm restarts (CosineAnnealingWarmRestarts)**: Could cycle the LR multiple times within 300s, letting the model re-explore the loss landscape and potentially converge better than a single cosine decay.

## Next Steps

1. **Fix cosine T_max to match actual epoch count (~100)** — high confidence. This is the most likely root cause; the underlying idea of cosine annealing is well-supported, the parameter was just miscalibrated.
2. **Keep baseline MultiStepLR and add only Cutout + label smoothing** — high confidence. Isolates regularization improvements from LR schedule changes, avoiding the T_max miscalibration issue entirely.
3. **WideResNet architecture (WRN-16-4) with properly tuned LR** — medium confidence. Architectural capacity increase with a correctly calibrated schedule could yield large gains, but introduces more variables.

## Exit Action Results
