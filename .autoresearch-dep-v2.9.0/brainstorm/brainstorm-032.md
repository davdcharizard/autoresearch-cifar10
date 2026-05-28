# Brainstorm EXP-032
**Created**: 2026-05-28
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search — alternating flip is from airbench96 (previously reviewed in brainstorm-024).

## Experimental History Review

- **Current best**: 96.56% (EXP-031) — Nesterov + reflect padding. New baseline.
- **Validated strategy**: Orthogonal stacking works. Combining changes on different axes compounds past the noise floor.
- **Untried on current baseline**: Alternating flip augmentation (temporal augmentation pattern axis), WD tuning, LR tuning, label smoothing tuning.

## Candidate Ideas

### 1. Alternating Flip Augmentation
**Summary**: Replace stochastic `RandomHorizontalFlip()` with deterministic alternating flip: flip ALL images in even epochs, no flip in odd epochs. Remove `transforms.RandomHorizontalFlip()` from the transform pipeline and add `inputs = inputs.flip(-1)` after GPU transfer when `epoch % 2 == 0`. This guarantees equal exposure to both orientations across consecutive epochs.

**Reasoning**: From airbench96 recipe. Different axis from Nesterov (optimizer) and reflect padding (data quality) — this targets augmentation temporal pattern. Random flip gives 50% chance per image per epoch; alternating flip guarantees balanced exposure. Zero throughput cost (GPU tensor flip is near-free). The TTA already uses horizontal flip, so the model is trained to be flip-equivariant — alternating flip reinforces this more systematically.

**Sources**: airbench96 (alternating flip), brainstorm-024 (previously considered), EXP-019 (TTA works because model learns flip-equivariant features)

**Estimated Effort**: low — ~5 lines of code

**Risk Assessment**: Low. Augmentation swaps have been at the noise floor (EXP-022: +0.07pp), but this isn't a swap — it's a replacement of random with deterministic temporal pattern. The interaction with TrivialAugmentWide is unknown. Worst case: no improvement or slight regression. Zero throughput cost.

### 2. Weight Decay 3e-4 (reduced from 5e-4)
**Summary**: Reduce WD from 5e-4 to 3e-4. The current model may be slightly over-regularized at 96.56% with Nesterov + reflect padding already improving quality.

**Reasoning**: WD=5e-4 was set at 93.33% baseline (EXP-003). At 96.56% with additional regularization, reducing WD could allow slightly larger weights, potentially improving the model's discriminative capacity for hard examples.

**Sources**: EXP-003 (WD=5e-4 added), goal-learnings § Patterns (regularization saturation)

**Estimated Effort**: low — single constant

**Risk Assessment**: Medium. WD=5e-4 is deeply validated — reducing it risks regression.

## Chosen Idea
**Selected**: Alternating Flip Augmentation

**Why this idea**: Adds a third orthogonal axis to the stack (augmentation pattern on top of optimizer + data quality). Zero throughput cost. The deterministic temporal pattern is a genuinely different mechanism from random flip.

**Hypothesis**: Replacing RandomHorizontalFlip with deterministic alternating flip will improve best_test_acc by 0.1-0.15pp (to 96.66-96.71%) by guaranteeing balanced exposure to both image orientations, reinforcing the flip-equivariant features that TTA exploits.
