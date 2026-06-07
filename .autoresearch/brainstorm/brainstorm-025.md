# Brainstorm EXP-025
**Created**: 2026-05-29
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- He et al. 2019 "Bag of Tricks for Image Classification with Convolutional Neural Networks" — recommends initializing the last BN gamma=0 in each residual block. This makes the residual branch output zero initially, so the block starts as identity. Reported to improve convergence and final accuracy for ResNets.

## Experimental History Review

- **26 experiments**, baseline 96.39%, nine consecutive failures (017-024)
- All hyperparameter tuning exhausted: WD, CutMix prob, label smoothing, gradient clipping — none improved over baseline
- Architecture changes that ADD components failed (SE blocks), but this idea CHANGES initialization — zero overhead
- Key distinction from EXP-017 (SE blocks): zero-init residual doesn't add parameters or per-step overhead. It only changes how the network starts training.

## Candidate Ideas

### 1. Zero-Init Residual (BN2 gamma=0)
**Summary**: Initialize `bn2.weight` (gamma) to 0 in each BasicBlock. With gamma=0, the BN2 output is zero, making the residual branch contribute nothing initially. The network starts as a series of identity mappings and gradually learns residual corrections. After `self.apply(self._weights_init)`, add a loop: `for m in self.modules(): if isinstance(m, BasicBlock): nn.init.zeros_(m.bn2.weight)`.

**Reasoning**: This is a well-known technique from "Bag of Tricks" (He et al. 2019). The mechanism: at initialization, each block outputs `F.relu(0 + shortcut(x)) = F.relu(shortcut(x))`, which is closer to identity than a random residual. This makes early training more stable — gradients flow through the shortcut without interference from random residual noise. The model can then gradually increase BN gamma to learn useful residual corrections. This is particularly valuable for our setup because: (1) we only have ~54 epochs, so efficient early training matters, (2) CutMix creates noisy targets that benefit from stable identity-like initialization, (3) the technique adds zero parameters and zero per-step overhead.

**Sources**: He et al. 2019 "Bag of Tricks", widely adopted in modern ResNet implementations

**Estimated Effort**: low — add 3 lines after `self.apply(self._weights_init)`

**Risk Assessment**: Very low. The network can learn to increase gamma from 0 — BN gamma is a trainable parameter. Worst case: no improvement (the model learns the same solution regardless of init). Cannot cause regression since the initialization only affects the starting point, not the training dynamics.

## Idea Evaluation

This is the only remaining well-evidenced technique that hasn't been tried. It's a proven "Bag of Tricks" recommendation with zero overhead.

## Chosen Idea
**Selected**: Zero-Init Residual (BN2 gamma=0)

**Why this idea**: Well-evidenced from He et al. 2019, zero overhead, addresses early training efficiency which is critical in our 54-epoch budget. Different mechanism from all prior failed experiments.

**Hypothesis**: Zero-init residual branches will improve early training efficiency by starting from identity, allowing the model to converge faster and reach best_test_acc ≥ 96.49%.
