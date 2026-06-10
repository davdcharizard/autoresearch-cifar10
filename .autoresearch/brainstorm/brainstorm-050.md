# Brainstorm EXP-050
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing CIFAR architecture context** (`knowledge/papers/wide-residual-networks.md`)
  The current `28/56/112` anchor is already the locally validated capacity point under this fixed budget, so EXP-050 should avoid more width and focus on no-parameter levers.
- **Existing PyTorch throughput context** (`knowledge/references/pytorch-throughput-tools.md`)
  Preserving the proven FP32 compile/channels-last path remains important; candidate ideas should be low overhead and must explicitly verify the step-21000 first LR drop.
- **Existing ResNet initialization context** (`knowledge/references/resnet-zero-init-residual.md`)
  Residual BN initialization is a possible no-overhead lever, but local EXP-028 evidence makes aggressive identity-bias initialization risky.

No new external search was needed. The strongest evidence for the next move is local: EXP-047 left a clean attribution gap for mild ColorJitter because the run missed the critical first LR drop under GPU contention.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; a new improvement must reach at least `94.07%` under the +0.10 percentage-point rule.
- The current anchor is `STAGE_WIDTHS=(28, 56, 112)`, batch size 128, LR 0.1, momentum 0.9, weight decay 2e-4, first LR drop at step 21000, reflection crop padding, label smoothing 0.05, FP32 compile, and channels-last.
- EXP-049 showed that decoupling `2e-4` weight decay weakens the anchor, so optimizer decay semantics should not be the next isolated lever.
- Recent clean failures also make scalar LR changes, weight-decay bracketing, BN momentum lowering, time-budget cosine, sparse EMA, and mild RandAugment lower priority.
- EXP-047 tried mild ColorJitter but ended at 20,321 steps and never reached `lr: 0.0100`, so it is a weak scientific attribution against photometric augmentation.
- The goal-learnings protocol finding now explicitly says missing the 21k first LR drop invalidates clean anchor comparisons. A retry is justified only if GPU conditions are clean enough to reach step 21000.

## Candidate Ideas

### 1. Clean Mild ColorJitter Retry
**Summary**: Re-run the EXP-047 transform change under clean GPU conditions: add `transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)` after `RandomHorizontalFlip()` and before `ToTensor()`, while preserving every other anchor setting.

**Reasoning**: EXP-047 produced a numeric no-improvement, but the run missed the step-21000 first LR drop because of severe GPU contention. That makes the result poor evidence about whether mild photometric augmentation can improve the current anchor. A clean retry directly resolves this attribution gap, modifies only `train.py`, has a simple implementation, and can be classified cleanly if it reaches `lr: 0.0100`.

**Sources**: `reports/exp-report-047.md`; goal-learnings protocol finding on missed LR drop; current `train.py` transform pipeline.

**Estimated Effort**: low

**Risk Assessment**: The expected improvement probability is modest because EXP-044 RandAugment failed cleanly at 93.83%. The main risk is again throughput loss; the plan must check GPUs before launch and treat a missed first LR drop as weak attribution or no-improvement with a caveat.

### 2. Partial Residual-Branch BN Scale Initialization
**Summary**: Initialize each residual block's second BatchNorm scale to a small positive value such as `0.1`, rather than the default learned scale of 1.0 or the fully zeroed residual branch used in EXP-028.

**Reasoning**: Full zero-gamma initialization undertrained badly in EXP-028, but a partial scale may retain early residual learning while slightly stabilizing block updates. This is a no-parameter, near-zero-overhead initialization change that stays within `train.py`.

**Sources**: `knowledge/references/resnet-zero-init-residual.md`; `reports/exp-report-028.md`; `BasicBlock.bn2` in `train.py`.

**Estimated Effort**: low

**Risk Assessment**: Local negative evidence is strong: the full zero-gamma variant reached only 91.74%. A partial value might avoid the worst slowdown, but it may still delay useful representation learning under the fixed 300s training budget.

### 3. Milder RandAugment Magnitude Probe
**Summary**: Test a lower-overhead policy augmentation sibling by changing the clean EXP-044 setting from `RandAugment(num_ops=1, magnitude=5)` to `magnitude=3`, preserving the same insertion point after crop/flip.

**Reasoning**: EXP-044 reached 93.83% with a valid first LR drop, which is below threshold but closer to the anchor than the confounded ColorJitter run. A lower magnitude could reduce distortion while preserving some generalization benefit, and it would still be a one-line `train.py` transform change.

**Sources**: `reports/exp-report-044.md`; current `train.py` transform pipeline.

**Estimated Effort**: low

**Risk Assessment**: This is likely a low-ceiling retry of a clean failure. It also adds transform overhead, and a milder policy may simply have too little effect to clear the 94.07% threshold.

## Idea Evaluation

The clean ColorJitter retry has the clearest scientific value because EXP-047's failure was confounded by a missed schedule milestone. It is not a blind repeat: the new experiment's core verification is whether the exact same photometric intervention can complete under clean GPU conditions and reach the first LR drop. If it still fails below 94.07% after reaching `lr: 0.0100`, the augmentation family can be deprioritized more confidently.

Partial residual BN scale initialization is cheap and distinct, but EXP-028 makes the mechanism risky. It might be worth testing later, yet it does not resolve a known attribution gap.

Lower-magnitude RandAugment is also cheap, but EXP-044 was already a clean no-improvement. Retuning its magnitude has weaker evidence than resolving the ColorJitter confound, especially given the current threshold.

## Chosen Idea
**Selected**: Clean Mild ColorJitter Retry

**Why this idea**:
It is the most disciplined next experiment because a prior run tested the idea under invalid schedule conditions. A clean run can either rescue mild photometric augmentation or close the gap with much stronger evidence, while preserving the current architecture, optimizer, schedule, and evaluation harness.

**Hypothesis**:
Under clean GPU conditions that reach the step-21000 first LR drop, mild ColorJitter will improve color/illumination robustness without enough overhead to disrupt the anchor, producing `best_test_acc >= 94.07%`.
