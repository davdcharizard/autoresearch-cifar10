# Report EXP-017: Mixup α=0.2 Replacing RandomErasing
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md
- **Plan**: plans/plan-017.md
- **Log**: logs/exp-log-017.md

## Goal

Maximize best_test_acc (%) on CIFAR-10. Baseline: 95.57% (EXP-015, commit 626e9d1). Improvement threshold: best_test_acc > 95.67% (baseline + 0.1pp). Direction: higher is better.

## Idea & Hypothesis

Replace RandomErasing(p=0.25, scale=(0.02, 0.2)) with Mixup α=0.2 applied at batch level. Rationale: EXP-010 (CutMix α=1.0) failed from over-regularization when stacked on existing augmentation — this addresses both failure modes by (1) replacing instead of stacking and (2) using mild α=0.2 (λ concentrated near 0/1, median ~0.85). Literature (Zhang et al. 2018, Yu et al. 2021, Galdran et al. 2021) reports 0.2-0.5pp gains on CIFAR-10 ResNets. Hypothesis: best_test_acc improves by 0.1-0.4pp to ≥95.67%.

## Approach

Four edits to train.py: (1) Added `MIXUP_ALPHA = 0.2` hyperparameter; (2) Removed `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` from augmentation pipeline; (3) Pre-created `beta_dist = torch.distributions.Beta(0.2, 0.2)` before training loop; (4) Added batch-level mixup in training loop — sample λ from Beta(0.2, 0.2), clamp max(λ, 1-λ), permute batch, mix inputs, construct one-hot labels with label smoothing 0.2 baked in, mix soft targets, compute manual soft-target cross-entropy loss replacing `F.cross_entropy`. No deviations from plan.

## Execution

Single local run on H20 GPU. Training completed 96 epochs in 300.0s (18,620 steps). No errors, crashes, or adjustments. Loss converged smoothly. Total wall time 408.9s including startup and evaluation.

## Results

- **Primary metric**: 95.53% (baseline: 95.57%, delta: -0.04pp)
- **Observations**: Test accuracy peaked at 95.53% mid-training then oscillated in late epochs (95.26-95.49%). Final test accuracy 95.33% with test loss 0.4466. Throughput unchanged (~16ms/step), VRAM 865.2 MB consistent with prior experiments. The late-epoch oscillation pattern differs from the baseline stack (TrivialAugmentWide+RandomErasing) which showed more stable convergence in the polish phase.
- **Analysis**: Hypothesis not supported. Mixup α=0.2 as a replacement for RandomErasing produced 0.04pp lower accuracy than the baseline with RandomErasing. The result suggests that RandomErasing's per-sample occlusion regularization is more effective than mild cross-sample interpolation for this specific model/budget combination. The late-epoch oscillation indicates mixup may destabilize the low-LR polish phase — consistent with Yu et al. 2021 (Mixup Without Hesitation) finding that mixup hurts fine convergence. The accuracy trajectory (peak mid-training, declining afterward) contrasts with the baseline where accuracy steadily improves through the polish phase.
- **Key Learning**: RandomErasing provides more effective regularization than mild Mixup (α=0.2) for WIDTH_MULT=4 ResNet-20 at 96 epochs; cross-sample interpolation destabilizes the late-training polish phase where per-sample occlusion does not.

## Verification

- **Conditions**: Condition 1 (best_test_acc > 95.67%) FAILED — actual 95.53%, 0.14pp short of threshold. Condition 2 (full summary block) PASSED. Condition 3 (eval ≤ num_epochs) PASSED.
- **Review Notes**: Results confirmed trustworthy — metrics consistent with training dynamics, no anomalies.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 95.53% did not exceed 95.67% threshold; in fact 0.04pp below the 95.57% baseline.

## Unexplored Avenues

- **Mixup Without Hesitation (mWh)**: Phase out mixup during the low-LR polish phase (last 25% of budget), using standard augmentation for fine convergence. Directly addresses the late-epoch oscillation observed. Yu et al. 2021 show mWh outperforms full-duration mixup.
- **Mixup + RandomErasing at reduced strengths**: Instead of replacing RandomErasing, combine Mixup α=0.1 with RandomErasing(p=0.15) — both at reduced intensity to avoid the over-regularization that caused EXP-010's failure while getting both regularization signals.
- **Higher α (0.3-0.4) with mWh scheduling**: More aggressive interpolation during exploration phase may produce stronger regularization signal, with mWh preventing late-training destabilization.

## Next Steps

1. **Stochastic Depth (DropPath) on BasicBlock** (medium confidence): Throughput-neutral structural regularizer orthogonal to all existing augmentation. p_max=0.1 for 9 blocks. Brainstorm-017 candidate #2 — untried, different regularization axis.
2. **Full state_dict EMA with β=0.995** (medium confidence): Lower β than EXP-014's 0.999 for better tracking of ~98-epoch trajectory. In-place tensor updates to minimize throughput cost. Goal-learnings notes idea "not exhausted."
3. **Test-time augmentation with horizontal flip** (medium confidence): hlb-CIFAR10 reports ~0.2pp gain from TTA with flip. Zero training cost — only modifies evaluation. Simple and orthogonal to all training regularization.

## Exit Action Results
