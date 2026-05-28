# Report EXP-018: Stochastic Depth (DropPath) on BasicBlock
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Log**: logs/exp-log-018.md

## Goal

Maximize best_test_acc (%) on CIFAR-10, direction higher is better. Baseline: 95.57% (EXP-015, commit 626e9d1). Improvement threshold: >95.67% (baseline + 0.1pp).

## Idea & Hypothesis

Stochastic depth (DropPath) was chosen as a regularization technique orthogonal to the existing input-space augmentation stack (TrivialAugmentWide + RandomErasing) and output regularization (label smoothing 0.2). The hypothesis: randomly dropping entire residual branches during training would provide complementary regularization by preventing co-adaptation between blocks, yielding +0.1-0.2pp improvement to ≥95.67%. Drop rates linearly increase from 0 (shallowest) to 0.1 (deepest block), following Huang et al. 2016 guidance.

## Approach

Four changes to train.py on the WIDTH_MULT=4 ResNet-20 (9 BasicBlocks, ~4.3M params):
1. Added `DROP_PATH_RATE = 0.1` hyperparameter
2. Added `drop_path_rate` parameter to `BasicBlock.__init__()` with instance storage
3. Modified `BasicBlock.forward()` — per-sample Bernoulli mask on residual branch during training, scaled by 1/keep_prob for expected-value preservation; identity during eval
4. Modified `ResNet.__init__()` and `_make_layer()` to compute and distribute linearly spaced drop rates (0.011 to 0.1) across all 9 blocks

No deviations from plan. Implementation was straightforward — the existing clean separation between residual and shortcut paths made DropPath insertion natural.

## Execution

Single run on H20 GPU. Training completed without errors: 92 epochs in 300.0s. Throughput stable at 16-17ms/step (unchanged from baseline). First LR drop at epoch 46 (84%→93%), second at epoch 73 (93%→95.24%). Final phase showed a convergence plateau around 94.9-95.2%, peaking at 95.24%.

## Results

- **Primary metric**: 95.24% (baseline: 95.57%, delta: -0.33pp, -0.35%)
- **Observations**: Final-phase training losses (~1.08-1.10) were notably higher than typical baseline runs, indicating under-fitting from excessive regularization. The accuracy plateau in the polish phase (75-100%) was unusually flat and low, suggesting the model could not fully converge with the combined regularization burden.
- **Analysis**: The hypothesis was disproven. Stochastic depth at p_L=0.9 does not provide complementary regularization — it compounds with the existing stack (TrivialAugmentWide + RandomErasing + label smoothing 0.2 + WD=5e-4) to push total regularization past the optimal point. The shallow 9-block architecture has limited capacity, and the existing regularizers are already near saturation. This is consistent with the CutMix failure (EXP-010) where stacking another regularizer also degraded accuracy.
- **Key Learning**: The regularization stack is near saturation for this shallow network — additional structural regularizers (DropPath, CutMix) hurt rather than help; gains must come from capacity, optimization, or data efficiency improvements.

## Verification

- **Conditions**: Condition 1 failed (95.24% < 95.67%); Conditions 2-3 passed
- **Review Notes**: Results confirmed trustworthy — metric degradation is genuine, not an artifact. Throughput unchanged (92 epochs, same as expected), full summary block printed, eval count matches epoch count.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 0.33pp below baseline; verification condition 1 failed

## Unexplored Avenues

- **Much lower DropPath rate (p_L=0.02-0.05)**: The 0.1 rate was calibrated for deeper networks (50-100+ blocks). For a 9-block network, survival probability 0.99-0.98 per block might provide subtle regularization without the under-fitting observed here. However, given the regularization saturation signal, the expected gain is marginal.
- **DropPath with reduced existing regularization**: Replacing RandomErasing with DropPath (rather than stacking) could maintain total regularization budget while shifting from input-space to architecture-space regularization. The CutMix/Mixup failures suggest input-space replacement is risky.

## Next Steps

1. **Capacity/architecture improvements** (high confidence): The regularization stack is saturated — further accuracy gains likely require increased model capacity (deeper or wider architecture that still fits in 300s budget) or fundamentally different architectural elements (attention, improved residual connections).
2. **Optimization improvements** (medium confidence): Learning rate schedule refinements (e.g., OneCycleLR, warmup restarts) or optimizer changes (AdamW, LAMB) could extract more from existing capacity.
3. **Knowledge distillation** (medium confidence): Using a larger teacher model to provide soft targets could improve accuracy without changing model capacity or adding regularization.
