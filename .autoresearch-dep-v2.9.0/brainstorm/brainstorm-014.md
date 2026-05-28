# Brainstorm EXP-014
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **hlb-CIFAR10 (github.com/tysam-code/hlb-CIFAR10)** (https://github.com/tysam-code/hlb-CIFAR10)
  Achieves 95.79% in ~110s on A100 using EMA with β tuned per epoch count (~80 epochs), label smoothing 0.2, and patch-based cutmix (size 10). Key insight: EMA is effective at short epoch budgets when properly implemented with full model state, and label smoothing at 0.2 (not 0.1) can work when paired with appropriate augmentation.

- **airbench (arxiv 2404.00498)** (https://arxiv.org/abs/2404.00498)
  Reaches 96% in 46.3s on A100 using test-time augmentation (TTA), derandomized horizontal flipping, and aggressive per-sample techniques. TTA is not feasible for us (evaluator in read-only prepare.py), but confirms that augmentation-side interventions have significant headroom.

- **EMA dynamics paper (arxiv 2411.18704)** (https://arxiv.org/abs/2411.18704)
  Analyzes EMA as implicit regularization — smoothing SGD noise in parameter space. For models with BatchNorm, EMA must include BN running statistics (buffers), not just nn.Parameter tensors. Our EXP-013 failure confirmed this: parameter-only EMA caused BN mismatch suppressing accuracy to 94.98%.

- **Mixup (Zhang et al. 2018, ICLR)** (https://arxiv.org/abs/1710.09412)
  Original paper uses α=1.0 for CIFAR-10 with 200-epoch training. For shorter schedules (~100 epochs), lower α (0.1-0.4) is recommended to avoid slow convergence. Cross-sample interpolation smooths decision boundaries and provides regularization orthogonal to per-sample augmentation.

- **Mixup Without Hesitation (arxiv 2101.04342)** (https://arxiv.org/abs/2101.04342)
  Proposes disabling mixup in later training epochs to solve the slow convergence problem. In short-budget regimes like ours (~98 epochs), this insight is valuable — mixup early for regularization, disable late for sharp convergence.

- **SWA (Izmailov et al. 2018, UAI)** (https://arxiv.org/abs/1803.05407)
  Stochastic Weight Averaging: +0.4% on CIFAR-10 with ResNet-164 using cyclic LR. Now in torch.optim.swa_utils. Requires cyclic or constant LR in the averaging phase — fundamentally changes the LR schedule. More complex than EMA and may conflict with our validated wall-clock-fractional MultiStepLR.

## Experimental History Review

**Current best**: 95.39% (EXP-009, commit cfe19c2) — batch 256 + LR 0.2 + 5-epoch warmup. 98 epochs in 300s at 16ms/step.

**Trajectory** (7 improvements across 15 experiments): BASE 91.72 → EXP-001 92.29 (width-2x) → EXP-002 92.92 (augmentation) → EXP-003 93.33 (WD) → EXP-005 94.44 (AMP) → EXP-007 94.82 (width-4x) → EXP-009 95.39 (batch 256).

**Key failed approaches**:
- SE blocks (EXP-011/012, count 2): ~9ms/step overhead exhausted — intrinsic to SE computation
- Parameter-only EMA (EXP-013): BN buffer mismatch — idea is sound, implementation was broken
- CutMix α=1.0 (EXP-010): over-regularizes when stacked on existing augmentation pipeline
- torch.compile (EXP-008): zero speedup on H20 for this model size
- Shifted LR schedule (EXP-006): (0.5, 0.75) is near-optimal, earlier drops hurt
- Nesterov + label_smoothing=0.1 (EXP-004): per-step overhead cost epochs, peaked lower

**Key patterns**:
- Wall-clock-fractional (0.5/0.75) schedule is near-optimal — do not change
- ~98 epochs in 300s at batch 256 — throughput ceiling reached
- Throughput-to-accuracy conversion is the primary improvement driver
- Augmentation and WD stack synergistically
- Any per-step overhead directly reduces epoch count (the binding constraint)

**Untried approaches**:
- Full state_dict EMA (fix for EXP-013's BN bug — explicitly listed as unexplored avenue)
- Mixup α<1.0 replacing RandomErasing (instead of stacking as EXP-010 did)
- Increased depth (NUM_BLOCKS=4/5) — untested but throughput concern
- Label smoothing at 0.2 (0.1 failed in EXP-004 with Nesterov; different α untried standalone)
- GhostBatchNorm / virtual batch normalization

## Candidate Ideas

### 1. Full State Dict EMA (β=0.999)
**Summary**: Maintain an exponential moving average of the FULL model state_dict (including BatchNorm running_mean/running_var buffers) and swap it in for evaluation. This is the direct fix for EXP-013's BN mismatch failure. Implementation uses `model.state_dict()` / `load_state_dict()` instead of `named_parameters()`, ensuring all tensors (parameters AND buffers) are tracked in the shadow copy.

**Reasoning**: EXP-013 proved the EMA mechanism works — late-training recovery to 94.98% after LR drops showed EMA provides genuine smoothing once SGD and EMA weights converge. The failure was purely implementation: BN running_mean/running_var are buffers excluded from `named_parameters()`, causing eval-time mismatch. Full state_dict EMA eliminates this entirely. The hlb-CIFAR10 reference achieves 95.79% using EMA, confirming EMA is effective at short epoch counts. Zero per-step throughput overhead (torch.no_grad EMA update is negligible vs forward/backward pass).

**Sources**: EXP-013 report (reports/exp-report-013.md § Unexplored Avenues), EMA dynamics paper (arxiv 2411.18704), hlb-CIFAR10 (github.com/tysam-code/hlb-CIFAR10), goal-learnings § Failed Approaches (parameter-only EMA entry)

**Estimated Effort**: low — 3 localized changes to train.py (init shadow dict, update loop, eval swap), identical structure to EXP-013 but using state_dict instead of named_parameters

**Risk Assessment**: Very low risk. The only change from EXP-013 is using state_dict (includes buffers) instead of named_parameters (excludes buffers). Worst case: EMA provides no improvement over the baseline (β too high or too low for 98 epochs), resulting in no-improvement verdict. No throughput cost, no destabilization risk. The mechanism is well-understood and the fix is surgical.

### 2. Mixup α=0.2 Replacing RandomErasing
**Summary**: Replace RandomErasing(p=0.25) with Mixup (α=0.2) as the secondary augmentation. Mixup interpolates pairs of training images and their labels: `x = λ*x_i + (1-λ)*x_j`, `y = λ*y_i + (1-λ)*y_j` where λ~Beta(α,α). This requires switching from hard-label cross_entropy to a soft-label loss (linear combination of per-class losses). Using α=0.2 (low, per short-schedule recommendation) and REPLACING rather than stacking (learning from EXP-010's over-regularization).

**Reasoning**: CutMix α=1.0 stacked on top of existing augmentation over-regularized (EXP-010: 95.03% vs 95.39%). The lesson: cross-sample augmentation must replace, not stack, and use lower α. Mixup α=0.2 is softer than CutMix α=1.0, provides cross-sample regularization orthogonal to TrivialAugmentWide, and smooths decision boundaries. The Mixup Without Hesitation paper suggests disabling in later epochs solves slow convergence — but for α=0.2 the convergence penalty should be minimal.

**Sources**: Mixup paper (arxiv 1710.09412), Mixup Without Hesitation (arxiv 2101.04342), EXP-010 report (CutMix failure analysis), goal-learnings § Failed Approaches (CutMix entry)

**Estimated Effort**: medium — requires implementing mixup in the training loop (lambda sampling, image interpolation, soft-label loss modification), removing RandomErasing from transforms, and potentially adjusting WD

**Risk Assessment**: Moderate risk. α=0.2 is conservative but untested on this exact setup. The soft-label loss change adds complexity. If convergence is slower than expected, 98 epochs may not be enough (same mechanism that hurt CutMix). Throughput impact should be minimal (mixup is a simple tensor operation). Worst case: no-improvement similar to EXP-010.

### 3. Cosine Annealing with Warm Restarts (SGDR)
**Summary**: Replace the wall-clock-fractional MultiStepLR with CosineAnnealingWarmRestarts using T_0 calibrated to the actual epoch count (~98 epochs). Use 2 cycles: T_0=49 epochs (half the budget), T_mult=1 (equal-length cycles). Each cycle: LR decays from 0.2→near-zero via cosine, then restarts. This explores multiple basins of attraction instead of the single-basin convergence of step decay.

**Reasoning**: The current (0.5, 0.75) step schedule is validated as near-optimal for step decay, but cosine restarts are a fundamentally different schedule family. SGDR (Loshchilov & Hutter 2017) showed improved generalization over fixed cosine by allowing the optimizer to escape sharp minima via warm restarts. With T_0 properly calibrated to actual training duration (unlike EXP-000 which miscalibrated T_max), this avoids the prior failure mode. Two half-budget cycles give the model two chances at finding good basins.

**Sources**: SGDR paper (Loshchilov & Hutter 2017, ICLR), EXP-000 failure (cosine T_max miscalibration), goal-learnings § Patterns (wall-clock schedule is near-optimal for step decay)

**Estimated Effort**: low — replace the LambdaLR scheduler with CosineAnnealingWarmRestarts, compute T_0 from epoch estimate

**Risk Assessment**: Moderate-to-high risk. The current MultiStepLR (0.5/0.75) is validated as near-optimal (EXP-006 confirmed shifting drops hurts). Warm restarts is a fundamentally different schedule family, so prior validation doesn't transfer. Risk of the restart disrupting convergence at the critical mid-training point. The epoch count (~98) may be too low for multiple cycles to converge. Wall-clock-fractional timing makes T_0 estimation approximate. Worst case: regression similar to EXP-000 or EXP-006.

## Idea Evaluation

**Evidence strength**: Full state_dict EMA has the strongest evidence by far. EXP-013 demonstrated the mechanism works (late-training recovery to 94.98%), and the failure root cause is precisely identified (BN buffers excluded). The fix is surgical and well-understood. hlb-CIFAR10 confirms EMA works at short epoch counts. Mixup has good literature support but no direct evidence in our setup — CutMix (related technique) failed at α=1.0, and while α=0.2 replacing instead of stacking is a different approach, it's untested. SGDR has theoretical appeal but our prior schedule experiments (EXP-000, EXP-006) show the current schedule is sensitive to changes.

**Mechanism clarity**: EMA: crystal clear — average all model state (params + BN buffers), evaluate with smoother weights. Mixup: clear — interpolate samples and labels, replace erasing with cross-sample regularization. SGDR: less clear for our regime — the restart mechanism's benefit depends on having enough epochs per cycle for convergence, which is uncertain at ~49 epochs per cycle.

**Expected impact**: EMA: +0.1-0.3pp expected (hlb-CIFAR10 uses EMA to reach 95.79%, and our late-training recovery in EXP-013 suggests genuine smoothing benefit once BN is fixed). Mixup: +0.1-0.3pp possible but uncertain. SGDR: unpredictable — could help or hurt.

**Risk profile**: EMA has the safest failure mode (no throughput cost, worst case is no-improvement). Mixup requires more code changes and has moderate risk. SGDR risks regression by abandoning a validated schedule.

**Feasibility**: EMA is the easiest to implement (3 localized changes, identical structure to EXP-013). Mixup requires training loop modification and loss function change. SGDR is easy to implement but harder to calibrate.

## Chosen Idea
**Selected**: Full State Dict EMA (β=0.999)

**Why this idea**:
Strongest evidence across all criteria. EXP-013 proved the EMA mechanism provides genuine smoothing (late-training recovery), and the failure was a precisely identified implementation bug (BN buffers excluded from parameter-only shadow). The fix — using `model.state_dict()` instead of `named_parameters()` — is surgical and well-understood. Zero throughput overhead preserves the 98-epoch budget. hlb-CIFAR10 confirms EMA effectiveness at short epoch counts. Lowest risk of all candidates with the clearest causal mechanism.

**Hypothesis**:
Full state_dict EMA (β=0.999) will achieve best_test_acc > 95.49% by providing implicit regularization through weight smoothing, with the BN buffer inclusion eliminating the eval-time mismatch that suppressed EXP-013 to 94.98%. The EMA-averaged model will show stable accuracy improvement from epoch 1 (unlike EXP-013's suppressed early phase), with the largest gains appearing after LR drops when SGD weights stabilize and EMA smoothing is most beneficial.
