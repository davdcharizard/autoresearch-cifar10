# Report EXP-028: ResNet-26 (NUM_BLOCKS=4)
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md
- **Plan**: plans/plan-028.md
- **Log**: logs/exp-log-028.md

## Goal
Maximize best_test_acc on CIFAR-10. Baseline: 96.46% (EXP-020). Threshold: >96.56%.

## Idea & Hypothesis
Increase model depth from NUM_BLOCKS=3 (ResNet-20) to NUM_BLOCKS=4 (ResNet-26), adding 33% more parameters (4.3M→5.8M). Hypothesis: the capacity increase would break the ~96.5% ceiling, gaining +0.2-0.5pp despite ~22% throughput loss.

## Approach
Changed `NUM_BLOCKS = 4` and `ESTIMATED_EPOCHS = 80` (recalibrated cosine schedule). No other changes.

## Execution
Single run, 75 epochs in 300s at 20-21ms/step. Model trained stably but converged slower in early epochs due to more parameters. Peak VRAM increased to 1103MB (from 865MB). Model was still improving at the final epoch (best=final at 96.31%).

## Results
- **Primary metric**: 96.31% (baseline: 96.46%, delta: -0.15pp)
- **Observations**: The deeper model achieved 96.31% in 75 epochs — below the baseline's 96.46% in ~96 epochs. The 22% epoch reduction (75 vs 96) outweighed the 33% capacity increase. This is the same throughput trap pattern seen with SE blocks (EXP-011/012) and pre-activation (EXP-021). The model was still improving at epoch 75 (best=final), confirming it was undertrained.
- **Analysis**: The hypothesis was wrong — depth increase does not break through the ceiling because the throughput cost creates an undertrained model. The EXP-007 width scaling precedent (+0.38pp with 22% epoch loss) succeeded because it was at a lower baseline (94.44%) where each pp was easier to gain. At 96.46%, the marginal value of each additional pp is much higher, and the throughput tax of 21 fewer epochs is more damaging. The model needs BOTH more capacity AND the same epoch count — which is impossible in 300s without a throughput improvement.
- **Key Learning**: At 96.46%, any capacity increase that costs >10% throughput will fail because the model at this level needs both capacity AND epochs. The WIDTH_MULT=4 + NUM_BLOCKS=3 configuration appears optimal for the 300s budget.

## Verification
- **Conditions**: Condition 1 FAILED (96.31% < 96.56%). Conditions 2-3 PASSED.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 0.15pp below baseline.

## Unexplored Avenues
- **NUM_BLOCKS=4 with batch 384/512 to recover throughput**: Larger batch could reduce per-epoch time, partially compensating for the per-step increase. But larger batch may hurt generalization.
- **Asymmetric depth (e.g., 3-4-5 blocks per stage)**: More depth in later stages where feature maps are smaller and conv is cheaper, maintaining throughput better.
- **NUM_BLOCKS=4 with reduced WIDTH_MULT=3**: Trade width for depth — keep total params similar but shift capacity toward depth.

## Next Steps
- **Nesterov + reflect padding** (medium confidence): Stack Nesterov (+0.06pp) with reflect-padded RandomCrop — two orthogonal zero-cost changes targeting different axes.
- **Reduced label smoothing (0.15 or 0.1)** (low-medium confidence): Current 0.2 may be slightly over-regularizing at this accuracy level. Reducing might allow tighter final convergence.
- **Different random seed or data ordering** (low confidence): The ~96.5% result from multiple experiments might be seed-dependent. A different seed might reveal whether we're at a true ceiling or a local minimum.

## Exit Action Results
- Log cleanup: Cleaned .log files from repo root.
