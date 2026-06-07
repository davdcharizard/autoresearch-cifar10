# Report EXP-004: k=4 Width
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md

## Goal
Maximize CIFAR-10 test accuracy. Baseline: 94.80% (EXP-003).

## Idea & Hypothesis
Continue width scaling to k=4 ({64,128,256}, 4.3M params). Hypothesis: 95.0-95.5%.

## Approach
WIDTH_MULT: 3→4, COSINE_T_MAX: 57→49. Two-line change.

## Execution
Single run, 58 epochs, no issues. T_max=49 was close to actual (58 minus warmup = 53).

## Results
- **Primary metric**: 95.25% (baseline: 94.80%, delta: +0.45%)
- **Observations**: best/final gap only 0.09% (95.25% vs 95.16%) — excellent T_max alignment. 58 epochs is slightly more than predicted 54, contributing to the small gap. Width scaling trend: k=1→k=2: +1.93%, k=2→k=3: +0.77%, k=3→k=4: +0.45%. Clear diminishing returns.
- **Key Learning**: Width scaling continues to work at k=4 but with diminishing returns (+0.45% vs +0.77% at k=3). VRAM still only 538MB. Further width increases possible but gains will be small. May need to explore other dimensions (depth, augmentation, architecture).

## Verification
- **Conditions**: All passed (95.25% >= 94.90%)
- **Verdict**: improvement

## Unexplored Avenues
- k=6 or k=8 width (diminishing returns likely < 0.3%)
- ResNet-32 at k=4 (add depth)
- Pre-activation ResNet blocks
- Squeeze-and-Excitation attention
- Knowledge distillation from larger model
- AutoAugment / TrivialAugment

## Next Steps
1. **k=6 or k=8 with depth increase** (medium confidence): Push width further while adding depth. Diminishing returns but may squeeze out 0.2-0.4%.
2. **Attention mechanisms (SE blocks)** (medium confidence): Add channel attention to better utilize the wide features.
3. **TrivialAugment** (medium confidence): State-of-the-art augmentation, zero hyperparameters.

## Exit Action Results
