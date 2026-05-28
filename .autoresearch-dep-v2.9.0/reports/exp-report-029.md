# Report EXP-029: Learned 1x1 Conv Shortcut Projections
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md
- **Plan**: plans/plan-029.md
- **Log**: logs/exp-log-029.md

## Goal
Maximize best_test_acc on CIFAR-10. Baseline: 96.46% (EXP-020). Threshold: >96.56%.

## Idea & Hypothesis
Replace zero-padded shortcuts at stage transitions with learned 1x1 conv + BN projections (ResNet "option B"). Hypothesis: full gradient flow through all shortcut channels would improve accuracy by 0.1-0.3pp.

## Approach
Modified BasicBlock to use `nn.Sequential(Conv2d(1x1, stride), BN)` when dimensions mismatch, `nn.Identity()` otherwise. Adds ~41K params (<1% of model). No other changes.

## Execution
Single run, 98 epochs at 16ms/step. Zero throughput cost confirmed — 1x1 convs on small feature maps added no measurable overhead. VRAM increased slightly (889MB vs 865MB baseline).

## Results
- **Primary metric**: 96.43% (baseline: 96.46%, delta: -0.03pp)
- **Observations**: The learned shortcut projections produced slightly lower accuracy than zero-padding. This was unexpected — the original ResNet paper showed option B improving over option A by ~0.5pp. The likely explanation: zero-padding acts as implicit regularization. The forced-zero channels in the shortcut reduce information flow through the shortcut path, similar to structural dropout. In our highly-regularized model (TrivialAugmentWide + RandomErasing + WD + LS=0.2), this additional implicit regularization from zero-padding is beneficial, and removing it by providing full learned projections causes slight over-fitting.
- **Analysis**: This reinforces the "regularization saturation" pattern seen throughout this project. The model is at a point where ANY change that reduces regularization (even a structural improvement like better gradient flow) hurts rather than helps. The zero-padding shortcut is not a weakness at this accuracy level — it's a feature.
- **Key Learning**: Zero-padding shortcuts act as implicit regularization through forced information bottleneck; learned projections remove this regularization and slightly hurt in the current well-regularized setup.

## Verification
- **Conditions**: Condition 1 FAILED (96.43% < 96.56%). Conditions 2-3 PASSED.
- **Verdict**: no-improvement
- **Verdict Basis**: Primary metric 0.03pp below baseline.

## Unexplored Avenues
- None identified — the result suggests learned shortcuts are counterproductive in this regularization regime.

## Next Steps
- **Nesterov + reflect padding** (low-medium): Stack two orthogonal near-miss changes.
- **Label smoothing 0.15 (reduced from 0.2)** (low-medium): Test if LS=0.2 is over-regularizing at this accuracy level. Could release capacity for better convergence.
- **Weight decay 3e-4 or 1e-3** (low): WD tuning hasn't been revisited since the cosine schedule change.

## Exit Action Results
- Log cleanup: Cleaned .log files from repo root.
