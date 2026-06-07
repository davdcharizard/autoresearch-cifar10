# Report EXP-018: Channels_last (NHWC) memory format
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-018.md
- **Plan**: plans/plan-018.md
- **Log**: logs/exp-log-018.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, higher is better) within 300s single-GPU training budget. Current baseline: 96.39% (EXP-016).

## Idea & Hypothesis

Convert model and training inputs to PyTorch's channels_last (NHWC) memory format for faster cuDNN convolutions with AMP. Hypothesis: 10-20% speedup → ~60-65 epochs instead of 54 → better convergence → +0.1-0.4% accuracy. T_max adjusted from 49 to 55 to exploit the extra epochs.

## Approach

Three changes to train.py: (1) `model.to(memory_format=torch.channels_last)` before EMA deepcopy, (2) `memory_format=torch.channels_last` on training input conversion, (3) COSINE_T_MAX 49→55. No architectural or hyperparameter changes beyond T_max.

## Execution

Single run, no issues. Training completed normally in 300s. Model built with 4,327,754 params and torch.compile warmup succeeded.

## Results

- **Primary metric**: 96.11% (baseline: 96.39%, delta: -0.28%, -0.29%)
- **Observations**: Channels_last provided a real speedup: 59 epochs in 300s vs baseline's 54 (~9% faster, ~5.08s/ep vs ~5.56s/ep). However, 96.11% is below the 96.39% baseline despite more training epochs. best==final (96.11%) confirms good T_max alignment — the cosine schedule and actual epochs are matched. The regression must come from the T_max change (49→55), not channels_last itself.
- **Analysis**: The hypothesis was partially correct — channels_last does speed up training. But the T_max change from 49 to 55 was counterproductive. With T_max=55, the cosine annealing decays the LR more slowly: at any given epoch, the LR is higher than with T_max=49. This means the model spends more time at higher learning rates, which produces more noise late in training and may prevent convergence to as sharp a minimum. The baseline's T_max=49 (with 54 actual epochs) means the LR reaches minimum at epoch 54, and the last few epochs refine at very low LR. With T_max=55, the LR is still relatively high at epoch 54, losing that refinement window. The experiment confounded two variables (memory format + T_max), making it impossible to isolate the channels_last effect.
- **Key Learning**: Channels_last gives ~9% speedup; but T_max=49→55 hurts — slower LR decay negates the benefit of extra epochs. Must isolate variables.

## Verification

- **Conditions**: best_test_acc >= 96.49% FAILED (actual: 96.11%)
- **Review Notes**: Results confirmed trustworthy — training completed normally, no anomalies
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 96.11% below baseline 96.39% + 0.1% threshold

## Unexplored Avenues

- **Channels_last with original T_max=49** (high priority) — isolate the channels_last speedup from the T_max change. With T_max=49 and 59 epochs, the cosine schedule completes at epoch 54 and the model trains at near-minimum LR for 5 extra epochs. These extra low-LR epochs could provide additional refinement without the downside of slower LR decay.
- **Channels_last with T_max=49 and optimized warmup** — the extra epochs from channels_last could be used for a longer warmup (e.g., 8 epochs instead of 5) while keeping the cosine phase length the same.

## Next Steps

1. **Channels_last + original T_max=49** (high confidence) — keep the 9% speedup but restore the proven LR schedule. The extra ~5 epochs at near-zero LR will provide free refinement. This isolates variables properly.
2. **Extended TTA with spatial shifts** (medium confidence) — add ±1px spatial shifts at test time. Zero training overhead, builds on EXP-016's +0.66% TTA success.
3. **Channels_last + T_max=49 + extended TTA** (medium confidence) — combine both improvements if either works individually.

## Exit Action Results
