# Report EXP-025: Zero-init residual (BN2 gamma=0)
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md
- **Plan**: plans/plan-025.md
- **Log**: logs/exp-log-025.md

## Goal
Maximize CIFAR-10 test accuracy within 300s. Baseline: 96.39%.

## Idea & Hypothesis
Zero-init the last BN gamma in each BasicBlock so residual branches start as identity.

## Approach
Added `nn.init.zeros_(m.bn2.weight)` for all BasicBlocks after weights_init.

## Execution
Single clean run. 57 epochs.

## Results
- **Primary metric**: 96.08% (baseline: 96.39%, delta: -0.31%)
- **Analysis**: Zero-init residual had no meaningful effect on this shallow 9-block ResNet-20. The technique is designed for deeper networks (50+ layers) where gradient flow through the skip connection is critical. At 20 layers, gradient flow is not the bottleneck and the initialization matters less. The result falls within the ~0.3% run-to-run variance band.
- **Key Learning**: Zero-init residual (Bag of Tricks) has no effect on shallow ResNet-20; the technique benefits deeper networks where gradient flow is a bottleneck.

## Verification
- **Conditions**: best_test_acc >= 96.49% FAILED (96.08%)
- **Verdict**: no-improvement

## Unexplored Avenues
- k=5 width (~6.7M params, ~42 epochs) with full modern recipe (EMA, CutMix, TTA)

## Next Steps
1. **k=5 width multiplier** (medium confidence) — untried intermediate between k=4 (54 ep) and k=6 (32 ep). With the full modern recipe, ~42 epochs might be enough for 6.7M params.

## Exit Action Results
