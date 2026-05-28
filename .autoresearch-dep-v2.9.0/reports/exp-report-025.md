# Report EXP-025: Gradient Centralization
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md
- **Plan**: plans/plan-025.md
- **Log**: logs/exp-log-025.md

## Goal
Maximize best_test_acc on CIFAR-10. Baseline: 96.46% (EXP-020). Direction: higher is better. Threshold: >96.56% (baseline + 0.1pp).

## Idea & Hypothesis
Apply gradient centralization (GC) to all conv/linear weight gradients — subtract the per-output-channel mean from each gradient tensor before the optimizer step. Sourced from Yong et al. 2020 (ECCV). Hypothesis: GC would improve generalization by constraining weight updates to be mean-free, improving loss landscape smoothness, yielding +0.1-0.3pp with zero throughput cost.

## Approach
Inserted `scaler.unscale_(optimizer)` after backward pass, followed by a GC loop over all parameters with `grad.dim() > 1` (conv weights: 4D, linear weights: 2D). For each, subtracted the mean across all dims except dim 0 (output channel) using in-place `sub_()`. BN parameters (1D) and biases automatically excluded by the dim filter. No hyperparameter changes.

## Execution
Single run, completed within 300s budget. 96 epochs at 16ms/step — 3 fewer epochs than baseline's 99. The GC gradient loop added ~0.5ms per step, accumulating to ~9s lost over 18500 steps. Wider test accuracy oscillations observed throughout training (e.g., dips to 78.69% at epoch 34, 77.92% at epoch 22) compared to baseline's tighter convergence. Model converged steadily in the final phase with consecutive new bests from epochs 88-94, peaking at 96.49% at epoch 94.

## Results
- **Primary metric**: 96.49% (baseline: 96.46%, delta: +0.03pp, +0.03%)
- **Observations**: Two unexpected findings: (1) GC was NOT zero throughput cost — the per-parameter gradient loop added ~0.5ms/step, costing 3 epochs. (2) GC induced noticeably wider test accuracy oscillations throughout training, suggesting it modifies the loss landscape geometry in a way that makes SGD paths less stable. The model still converged in the final low-LR phase, but the oscillations consumed training capacity that could have produced a higher final accuracy.
- **Analysis**: The hypothesis was partially validated — GC did not cause instability or crash, and the final accuracy was marginally above baseline (+0.03pp). However, the benefit was too small to overcome the 3-epoch throughput cost and wider oscillations. At 96.46% with the regularization stack near saturation, GC's weight-space regularization effect was marginal — the model is already well-regularized and the additional constraint didn't provide meaningful additional generalization. The per-step overhead, while small (~3%), matters in this throughput-constrained regime.
- **Key Learning**: Gradient centralization adds non-trivial per-step overhead (~0.5ms) from Python-level gradient iteration, and its weight-space regularization is marginal when the model is already well-regularized with augmentation + WD + label smoothing.

## Verification
- **Conditions**: Condition 1 FAILED (96.49% < 96.56% threshold). Conditions 2-3 PASSED (clean completion, 96 evals for 96 epochs).
- **Review Notes**: Results confirmed trustworthy — the 96.49% is plausible given the convergence trajectory observed. No parsing errors or stale output.
- **Verdict**: no-improvement
- **Verdict Basis**: Verification condition 1 failed — primary metric +0.03pp above baseline, below the 0.1pp improvement threshold.

## Unexplored Avenues
- **GC with throughput compensation**: The 3-epoch cost could be eliminated by implementing GC as a custom C++/CUDA optimizer extension rather than a Python loop. If throughput were preserved at 99 epochs, the +0.03pp might grow to +0.1pp.
- **Selective GC (conv-only, skip linear)**: Applying GC only to conv layers (4D) and skipping the final linear layer might reduce overhead and change the regularization balance.
- **GC combined with Nesterov momentum**: GC + Nesterov could compound — GC constrains the direction, Nesterov improves the step quality. Both are optimizer-level changes.

## Next Steps
- **Nesterov momentum (revisited)** (medium confidence): Zero-cost single parameter change. Failed in EXP-004 in a completely different context (no AMP, no batch 256, no cosine). Worth retrying in current setup.
- **Alternating flip augmentation** (medium confidence): From airbench96. Zero throughput cost, deterministic balanced exposure. Risk of being in noise floor like augmentation swaps (EXP-022).
- **Deeper architecture NUM_BLOCKS=4** (low-medium confidence): ResNet-26 with +33% params. Risk of throughput regression (~74 epochs) but provides fundamentally more capacity.

## Exit Action Results
- Log cleanup: Deleted run.log from repo root. `*.log` already in .gitignore.
