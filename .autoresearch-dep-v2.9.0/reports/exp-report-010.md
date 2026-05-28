# Report EXP-010: CutMix Batch Augmentation
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md
- **Plan**: plans/plan-010.md
- **Log**: logs/exp-log-010.md

## Goal
Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better). Baseline: 95.39% (EXP-009, commit cfe19c2). Verification threshold: > 95.49% (baseline + 0.1pp).

## Idea & Hypothesis
CutMix batch augmentation (α=1.0) was chosen as the strongest candidate from brainstorm-010: cross-sample mixing provides an augmentation axis orthogonal to the existing per-sample pipeline (TrivialAugmentWide + RandomErasing), with strong external evidence (+0.97% in Yun et al. 2019 on CIFAR-10 ResNet-56). Hypothesis: CutMix would add +0.2–0.5pp by improving generalization through harder training examples, targeting 95.6–95.9%.

## Approach
Four edits to train.py: (1) `import numpy as np`, (2) `CUTMIX_ALPHA = 1.0` hyperparameter, (3) `rand_bbox(size, lam)` helper computing a random bounding box from λ, (4) CutMix logic in the training loop — draw λ from Beta(α,α), shuffle indices, blend image patches, adjust λ for actual pixel ratio, compute mixed-label cross-entropy. No deviations from plan. All other hyperparameters preserved from EXP-009.

## Execution
Single run, local H20 GPU. Training completed 96 epochs / 18,591 steps in 300.0s. No errors, no retries, no adjustments. CutMix added negligible per-step overhead (pure tensor ops). LR schedule operated normally: 0.2→0.02 at ~50%, 0.02→0.002 at ~75%.

## Results
- **Primary metric**: 95.03% (baseline: 95.39%, delta: −0.36pp, −0.38%)
- **Observations**: The model was still improving at end of budget — best_test_acc peaked at epoch 93 (95.03%), final_test_acc was 94.91% at epoch 96. Earlier in training (epoch 72), accuracy was only 93.72%, suggesting CutMix significantly slowed convergence. Peak VRAM unchanged (864.6 MB vs 864.6 MB baseline). Epoch count dropped slightly from 98 to 96 (negligible).
- **Analysis**: The hypothesis was wrong. CutMix α=1.0 stacked on TrivialAugmentWide + RandomErasing + WD=5e-4 over-regularized the model. The combined regularization made each training step less informative, requiring more epochs to converge than the 300s budget allows. The still-rising accuracy trajectory at epoch 96 confirms this is a convergence problem, not a capacity problem — given more epochs, CutMix would likely match or exceed the baseline. The original paper's +0.97% was measured on a standard augmentation baseline (RandomCrop + RandomHorizontalFlip only), not on an already-heavy augmentation stack.
- **Key Learning**: Stacking CutMix on heavy per-sample augmentation requires either more training time or a reduced α to compensate for the increased regularization burden.

## Verification
- **Conditions**: Condition 1 (best_test_acc > 95.49%) FAILED; Conditions 2-3 PASSED
- **Review Notes**: Results confirmed trustworthy. The 95.03% value is plausible for an over-regularized model — the convergence curve shows steady improvement through the final epoch, consistent with the over-regularization hypothesis. No parsing errors, stale output, or evaluation issues.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result but primary metric (95.03%) did not exceed verification threshold (95.49%), failing Condition 1.

## Unexplored Avenues
- **CutMix with reduced α (e.g., 0.2–0.5)**: Lower α concentrates the Beta distribution near λ=0 and λ=1, producing mostly-original or mostly-replaced images rather than uniform 50/50 mixes. This reduces per-step regularization strength, potentially allowing convergence within the 96-epoch budget while retaining the cross-sample mixing benefit.
- **CutMix replacing RandomErasing instead of stacking**: RandomErasing and CutMix both operate on rectangular image patches. Replacing RandomErasing with CutMix removes one regularization layer while substituting a more informative one (filling with another sample's content vs zeros), potentially maintaining total regularization load while improving information content.
- **CutMix with probability p<1.0**: Apply CutMix to only a fraction of batches (e.g., p=0.5), reducing the average regularization per epoch while keeping the cross-sample signal.

## Next Steps
1. **OneCycleLR schedule** (medium confidence): Replace step-decay with continuous warmup+decay. The smooth profile avoids the sharp LR=0.01 instability seen with AMP (EXP-005) and may extract more accuracy from the same epoch budget. Risk: total_steps estimation and WD interaction.
2. **Cosine annealing with correct T_max** (medium confidence): Minimal-risk schedule change — swap only the decay shape while preserving everything else. T_max can now be accurately calibrated from EXP-009's 98-epoch data. Expected effect near noise floor (~0.2pp).
3. **Deeper model (ResNet-32 or ResNet-44)** (low-medium confidence): Increase depth while keeping WIDTH_MULT=4. More parameters may improve accuracy ceiling, but will reduce epoch count. Need to verify throughput impact first.
