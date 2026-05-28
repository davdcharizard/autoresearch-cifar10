# Brainstorm EXP-016
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **cifar10-airbench / hlb-CIFAR10 speedrun recipes** (https://github.com/KellerJordan/cifar10-airbench, https://github.com/tysam-code/hlb-CIFAR10)
  Achieves 96% in ~27s on A100. Key techniques beyond what we already use: **BN momentum 0.6** (vs default 0.1) for faster running-stat convergence in short training, triangular/OneCycleLR schedule, 12-pixel Cutout. Label smoothing 0.2 matches our current setup.

- **Mixup: Beyond Empirical Risk Minimization** (https://arxiv.org/pdf/1710.09412)
  Alpha in [0.1, 0.4] is optimal; α=0.2 adds ~0.3-0.5% on PreAct ResNet-18. Benefits more pronounced on larger models and longer training. Near-zero throughput cost (linear interpolation of inputs/labels).

- **94% on CIFAR-10 in 3.29 Seconds** (https://arxiv.org/html/2404.00498v2)
  Documents the speedrun lineage. OneCycleLR (triangular) strongly preferred over step decay. SGD + Nesterov with very high peak LR. Confirms BN momentum tuning is a key ingredient for short-budget training.

## Experimental History Review

- **Current best**: 95.57% (EXP-015, label smoothing 0.2 standalone, commit 626e9d1)
- **Trajectory**: BASE 91.72 → EXP-001 92.29 → EXP-002 92.92 → EXP-003 93.33 → EXP-005 94.44 → EXP-007 94.82 → EXP-009 95.39 → EXP-015 95.57 (8 improvements out of 17 experiments)
- **What worked**: Width scaling (+capacity), AMP (+throughput/epochs), augmentation (TrivialAugmentWide+RE), WD=5e-4, batch 256 with LR scaling, label smoothing 0.2. Common thread: throughput-neutral or throughput-positive changes that add regularization or capacity.
- **What failed**: SE blocks (throughput cost exhausted, count=2), parameter-only EMA (BN mismatch), full state_dict EMA β=0.999 (too conservative for ~92 epochs), CutMix α=1.0 (over-regularization when stacked), torch.compile (no speedup on H20), shifted LR drops (worse than 0.5/0.75), Nesterov+LS=0.1 on width-2x (per-step overhead cost epochs).
- **Key pattern**: The (0.5, 0.75) MultiStepLR schedule is HIGH IMPORTANCE and validated as near-optimal. Any schedule replacement must provide a comparably low LR regime in the final ~15% of training.
- **Key pattern**: ~98 epochs in 300s budget at current config. Throughput-to-accuracy conversion remains strong — each additional epoch contributes.
- **Untried directions**: BN hyperparameter tuning (momentum), Mixup (α=0.2 replacing RandomErasing), OneCycleLR/triangular schedule, gradient clipping, different weight initialization.

## Candidate Ideas

### 1. Higher BN Momentum (0.5)
**Summary**: Increase BatchNorm momentum from the PyTorch default of 0.1 to 0.5. In PyTorch, BN momentum controls the running statistics update: `running_mean = (1 - momentum) * running_mean + momentum * batch_mean`. Default 0.1 gives an effective averaging window of ~10 batches; raising to 0.5 shrinks this to ~2 batches, making running stats track the current model state much more closely. This is a post-construction hyperparameter change applied to all BN layers — a 3-line loop after model creation.

**Reasoning**: The cifar10-airbench speedrun recipe uses BN momentum 0.6 as a key ingredient for short-budget training. With only ~98 epochs and rapid LR changes (drops at 50%/75% of budget), the default slow-tracking running stats lag behind the actual batch statistics, creating a train-eval mismatch that degrades test accuracy. The effect is most pronounced: (1) during early training when parameters change rapidly under high LR, and (2) right after LR drops when the loss surface geometry shifts. Higher momentum ensures eval-time BN statistics better reflect the current model state. This is completely orthogonal to all existing regularization (augmentation, label smoothing, weight decay) and has zero throughput cost.

**Sources**: cifar10-airbench (BN momentum 0.6), hlb-CIFAR10 speedrun recipe, arxiv 2404.00498v2 (94% in 3.29s paper). No prior experiments have touched BN hyperparameters.

**Estimated Effort**: low — 3-line change after model creation, zero runtime overhead

**Risk Assessment**: If momentum is too high, running stats become noisy (too responsive to individual batch fluctuations), potentially degrading eval accuracy. Using 0.5 (vs speedrun's 0.6) as a slightly conservative choice for our setup which differs from the speedrun (we have heavier augmentation and different LR schedule). Worst case: slight accuracy degradation (~0.1-0.2pp), easily identifiable.

### 2. Mixup α=0.2 Replacing RandomErasing
**Summary**: Replace RandomErasing(p=0.25, scale=(0.02, 0.2)) with Mixup (α=0.2) in the training loop. Mixup creates convex combinations of pairs of training examples and their labels: `x_mixed = λ*x_i + (1-λ)*x_j, y_mixed = λ*y_i + (1-λ)*y_j` where λ ~ Beta(α, α). This provides cross-sample regularization in input+label space — a different dimension than label smoothing (output-only) and TrivialAugmentWide (input-only per-sample).

**Reasoning**: Mixup α=0.2 is well-supported by literature (original paper shows optimal α in [0.1, 0.4]). EXP-010 showed CutMix α=1.0 over-regularizes when stacked on existing augmentation; replacing RandomErasing (rather than stacking) keeps total regularization load manageable. Mixup has near-zero throughput cost (a linear interpolation per batch). The combination of Mixup + label smoothing has been shown to work well in practice.

**Sources**: Mixup paper (arxiv 1710.09412), EXP-010 lesson (don't stack, replace), EXP-015 report Next Steps (item 1)

**Estimated Effort**: medium — requires modifying the training loop to generate mixed inputs/labels per batch, removing RandomErasing from the transform pipeline

**Risk Assessment**: Over-regularization is the primary risk even when replacing RandomErasing, since we still have TrivialAugmentWide + label_smoothing=0.2 + WD=5e-4. The Mixup label mixing may interact unexpectedly with label smoothing (both modify the target distribution). CutMix failure (EXP-010) at α=1.0 is a cautionary signal, though α=0.2 is much more conservative and Mixup is less aggressive than CutMix. Worst case: ~0.3pp regression similar to CutMix.

### 3. OneCycleLR (Triangular Schedule)
**Summary**: Replace the wall-clock-fractional MultiStepLR schedule with PyTorch's OneCycleLR using a triangular (linear) annealing policy. The schedule ramps LR from a low value to peak LR over ~20-30% of training, then linearly decays to near-zero. This is the schedule used by all major CIFAR-10 speedrun recipes and consistently outperforms step decay in short-budget training.

**Reasoning**: The speedrun community (cifar10-airbench, hlb-CIFAR10, davidcpage/cifar10-fast) universally uses OneCycleLR over step decay. The smooth decay provides a gradually decreasing LR that may allow finer convergence in the final epochs compared to the abrupt drops in our current schedule. Our MultiStepLR spends 25% of training at LR=0.002 (final plateau), while OneCycleLR spends progressively more time at lower LRs, potentially finding a better minimum.

**Sources**: cifar10-airbench, hlb-CIFAR10, arxiv 2404.00498v2, OneCycleLR experiments (hd10.dev blog)

**Estimated Effort**: medium — replace the scheduler construction and remove the wall-clock progress tracking; requires choosing max_lr, div_factor, final_div_factor, pct_start parameters

**Risk Assessment**: HIGH. The current MultiStepLR schedule is validated as HIGH IMPORTANCE in goal-learnings — "Baseline MultiStepLR first drop at step 32K is critical for convergence." Replacing it is a direct contradiction of a high-importance pattern. The EXP-000 failure (CosineAnnealingLR with wrong T_max) and EXP-006 failure (shifted drops) show schedule changes are risky. The speedrun recipes use very different setups (different model widths, batch sizes, total epochs, sometimes different optimizers), so their schedule may not transfer directly. Multiple hyperparameters need tuning simultaneously (max_lr, warmup fraction, min_lr).

## Idea Evaluation

**Evidence strength**: BN momentum has direct evidence from the speedrun community (cifar10-airbench uses 0.6 specifically for short-budget training). Mixup has strong literature support but the interaction with existing heavy regularization is uncertain. OneCycleLR has strong community evidence but contradicts our own high-importance pattern.

**Mechanism clarity**: BN momentum has the clearest mechanism — running stats lag creates a train-eval distribution mismatch, and higher momentum directly fixes this. Mixup's mechanism is well-understood (vicinal risk minimization) but its interaction with label smoothing adds uncertainty. OneCycleLR's mechanism is clear but its advantage over our well-tuned step decay is harder to predict.

**Expected impact**: All three target ~0.1-0.3pp. BN momentum is harder to predict (could be 0 or 0.3pp). Mixup has literature support for ~0.3pp. OneCycleLR has the highest ceiling but also the highest floor risk.

**Risk profile**: BN momentum is zero-cost with graceful failure (slight degradation). Mixup requires replacing an existing augmentation component. OneCycleLR contradicts high-importance patterns and has the most hyperparameters to tune.

**Feasibility**: BN momentum is the simplest (3-line change, no hyperparameter tuning beyond the momentum value). Mixup requires training loop modification. OneCycleLR requires scheduler replacement and parameter selection.

BN momentum wins on mechanism clarity, risk profile, and feasibility. It's a completely novel direction (no prior BN experiments), has zero throughput cost, and is directly evidenced by the most successful CIFAR-10 speedrun recipes. The conservative choice of 0.5 (vs speedrun's 0.6) accounts for our heavier regularization and different schedule.

## Chosen Idea
**Selected**: Higher BN Momentum (0.5)

**Why this idea**:
Zero throughput cost, minimal code change, clear causal mechanism (faster BN running stat convergence reduces train-eval mismatch in short training), and directly evidenced by the speedrun recipes that achieve 96% on CIFAR-10. This is a completely untried dimension — no prior experiment has touched BN hyperparameters. The risk profile is excellent: worst case is a slight degradation that's easily attributed and reversed.

**Hypothesis**:
Increasing BN momentum from 0.1 to 0.5 will improve test accuracy by 0.1–0.3pp (to ~95.67–95.87%) by making BatchNorm running statistics more responsive to the current model state, reducing the distribution mismatch between training (which uses batch statistics) and evaluation (which uses running statistics). The effect should be most visible in the per-epoch eval accuracy right after LR drops, where the old running stats lag behind the rapidly changing loss surface.
