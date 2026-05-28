# Brainstorm EXP-013
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **"Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits"** (https://arxiv.org/html/2411.18704v1)
  EMA consistently outperforms SGD baseline on CIFAR-10 ResNet in both accuracy and loss. EMA introduces implicit regularization that reduces the need for LR decay, smooths training trajectory, and improves generalization. Benefits include robustness to noisy labels, prediction consistency, and better calibration.

- **"94% on CIFAR-10 in 3.29 Seconds on a Single GPU" (Keller Jordan, 2024)** (https://arxiv.org/abs/2404.00498)
  Achieves 96.04% on CIFAR-10 with scaled architecture (128/512 channels), 12-pixel Cutout, and 40 epochs. Test-time augmentation (horizontal flip + crops) further improves accuracy. Demonstrates that wider models + strong augmentation + sufficient epochs is the path to 96%+.

- **"mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)** (https://arxiv.org/pdf/1710.09412)
  Alpha in [0.1, 0.4] is the optimal range; larger alpha causes underfitting. Mixup prefers smaller weight decay (1e-4 vs 5e-4) — its regularization effect partially overlaps with WD. Acts as a data-dependent regularizer that smooths the loss landscape.

- **"Using mixup as regularization and tuning hyper-parameters for ResNets"** (https://arxiv.org/pdf/2111.11616)
  When using Mixup on CIFAR-10 with ResNets, reducing WD from 5e-4 to 1e-4 is recommended to avoid over-regularization. Benefits scale with network depth/capacity.

## Experimental History Review

**Trajectory** (7 improvements over 14 experiments): BASE 91.72 → EXP-001 92.29 (width-2x) → EXP-002 92.92 (augmentation) → EXP-003 93.33 (WD=5e-4) → EXP-005 94.44 (AMP) → EXP-007 94.82 (width-4x) → EXP-009 95.39 (batch 256 + LR scaling)

**Current best**: 95.39% (EXP-009, commit cfe19c2). Threshold for improvement: >95.49%.

**What worked**: Capacity increases (width scaling), throughput improvements (AMP, batch doubling), regularization (TrivialAugmentWide + RandomErasing + WD=5e-4), wall-clock-fractional LR schedule.

**What failed**:
- SE blocks exhausted (count=2, EXP-011/012): ~9ms/step overhead intrinsic to computation, not format-related
- CutMix α=1.0 (EXP-010): over-regularizes when stacked on existing augmentation in ~96-epoch budget
- torch.compile (EXP-008): zero speedup on H20 for small ResNet+AMP
- Shifted LR (EXP-006): earlier drops reduce accuracy ceiling
- Nesterov+label_smoothing (EXP-004): per-step overhead cost epochs

**Key patterns**:
- ~98 epochs in 300s at current config (batch 256, WIDTH_MULT=4)
- Wall-clock-fractional schedule (0.5/0.75) is near-optimal
- Throughput-to-accuracy conversion is the primary improvement driver
- CutMix over-regularized — any new regularizer must replace, not stack

**Untried gaps**: EMA of model weights, Mixup with low alpha (replacing RandomErasing to avoid over-regularization), increased depth, test-time augmentation, Cutout replacing RandomErasing.

## Candidate Ideas

### 1. EMA of Model Weights (Polyak Averaging)
**Summary**: Maintain an exponential moving average of model parameters during training with decay β=0.999. At evaluation time, use the EMA weights instead of the raw SGD weights. Implementation: shadow copy of parameters updated each step as `ema_param = β * ema_param + (1-β) * param`. Before each evaluation call, swap EMA weights in; after eval, swap original weights back. Zero throughput cost — one extra multiply-add per parameter per step is negligible compared to forward/backward pass.

**Reasoning**: EMA is a pure post-processing technique that averages out SGD noise in the final weights, acting as implicit regularization without reducing training capacity or throughput. The 2024 paper (arxiv 2411.18704) shows consistent improvement over SGD baseline on CIFAR-10 ResNet. Since our primary bottleneck is epoch count (~98 epochs in 300s), any technique with zero throughput cost has outsized value. EMA was the top recommendation from EXP-012's next steps (high confidence).

**Sources**: arxiv 2411.18704 (EMA dynamics paper), exp-report-012.md § Next Steps, goal-learnings § Patterns (throughput-to-accuracy conversion)

**Estimated Effort**: low — ~20 lines of code, no hyperparameter interaction with existing training pipeline

**Risk Assessment**: Low risk. Worst case: EMA weights perform identically to raw weights (no-improvement). β=0.999 is the standard choice; too low (0.99) would average too aggressively and hurt early training, too high (0.9999) would barely differ from raw weights. The technique is purely additive — it cannot degrade training dynamics since training uses original weights.

### 2. Mixup (α=0.2) Replacing RandomErasing
**Summary**: Replace RandomErasing augmentation with Mixup (α=0.2). For each batch, sample λ ~ Beta(0.2, 0.2), shuffle batch indices, and compute mixed inputs as `x_mix = λ*x + (1-λ)*x[shuffled]` with targets `y_mix = λ*y_onehot + (1-λ)*y_onehot[shuffled]`. This requires switching from hard labels + cross_entropy to soft labels with a KL or soft CE loss. Additionally reduce WD from 5e-4 to 2e-4 to account for Mixup's own regularization effect (literature recommends 1e-4, but our model uses strong augmentation already).

**Reasoning**: EXP-010 showed CutMix α=1.0 over-regularized when stacked on existing augmentation. The fix is (a) use a gentler cross-sample method (Mixup vs CutMix), (b) use much lower alpha (0.2 vs 1.0), and (c) replace rather than stack — remove RandomErasing to keep total regularization budget neutral. Literature confirms α∈[0.1,0.4] is optimal and that Mixup prefers lower WD.

**Sources**: arxiv 1710.09412 (Mixup paper), arxiv 2111.11616 (Mixup + ResNet tuning), exp-report-010.md (CutMix over-regularization), goal-learnings § Failed Approaches (CutMix)

**Estimated Effort**: medium — requires modifying loss computation (soft labels), removing RandomErasing, adding batch shuffle logic, tuning WD

**Risk Assessment**: Medium risk. The α=0.2 + WD reduction should avoid the over-regularization that killed EXP-010, but the loss function change (soft labels) and WD change introduce two moving parts. If accuracy drops, it's unclear which change caused it. The soft-label cross-entropy may interact with AMP (FP16) differently than hard labels.

### 3. Test-Time Augmentation (Horizontal Flip)
**Summary**: At evaluation time, run each test image through the model twice — once normally and once horizontally flipped — and average the logits before argmax. This is the simplest form of test-time augmentation (TTA). Implementation: in the evaluator's evaluate function... but `prepare.py` is read-only. Instead, implement TTA within `train.py` by wrapping the evaluation call — compute predictions on original images and flipped images, average logits, then compute accuracy manually rather than using `evaluator.evaluate()`.

**Reasoning**: The Keller Jordan 2024 paper notes TTA improves CIFAR-10 accuracy. Horizontal flip TTA is essentially free in terms of implementation complexity and adds only ~2x evaluation time (evaluation is a small fraction of total time). However, this approach modifies how we measure the metric rather than how we train the model.

**Sources**: arxiv 2404.00498 (Keller Jordan 2024), standard TTA practice

**Estimated Effort**: medium — need to reimplement evaluation within train.py since prepare.py is read-only

**Risk Assessment**: High risk of being reward hacking. The goal intends to improve model quality through training improvements, not through evaluation-time tricks. TTA doesn't improve the actual model weights — it improves the measurement. A different benchmark or evaluation protocol would not retain the benefit. Additionally, the hard constraint says `prepare.py` is read-only, and reimplementing evaluation logic in `train.py` to circumvent this is a scope violation in spirit. Finally, `evaluator.evaluate()` is the ground truth measurement — any custom evaluation could diverge from it.

## Idea Evaluation

**Evidence strength**: EMA has the strongest direct evidence — a 2024 paper specifically studying EMA dynamics on CIFAR-10 ResNet shows consistent improvement, and it was the #1 recommendation from the most recent experiment analysis. Mixup has solid evidence from the original paper but our specific situation (already heavy augmentation + WD=5e-4) adds uncertainty about the net effect. TTA has evidence but is problematic for this goal.

**Mechanism clarity**: EMA has the clearest mechanism — it averages out SGD noise in final weights, acting as free regularization without affecting training dynamics. Mixup's mechanism is also clear (smoother decision boundaries via convex combinations) but has known interactions with existing regularization that require careful tuning. TTA's mechanism is evaluation-time ensembling, which doesn't improve the model itself.

**Expected impact**: EMA is expected to yield +0.1-0.3pp (modest but reliable). Mixup could yield more (+0.3-0.5pp) if tuned correctly but has higher variance. TTA could yield +0.2-0.5pp but is borderline reward hacking.

**Risk profile**: EMA has the safest failure mode — it literally cannot hurt training (training uses original weights). Mixup changes the loss landscape and could cause degradation. TTA risks being classified as invalid due to reward hacking concerns.

**Feasibility**: EMA is the simplest — ~20 lines, no hyperparameter interactions. Mixup requires loss function changes and WD retuning. TTA requires reimplementing evaluation.

**Verdict**: EMA dominates on evidence, mechanism, risk, and feasibility. Its expected impact is modest, but the near-zero risk of degradation and zero throughput cost make it the clear choice. If EMA succeeds, Mixup becomes the natural next experiment.

## Chosen Idea
**Selected**: EMA of Model Weights (Polyak Averaging)

**Why this idea**:
EMA has the strongest evidence (2024 dynamics paper + multiple prior studies showing consistent improvement on CIFAR-10 ResNet), the clearest mechanism (implicit regularization via weight averaging with zero throughput cost), and the safest risk profile (cannot degrade training since it only affects evaluation weights). It was the top recommendation from the most recent experiment (EXP-012) with high confidence. In a throughput-constrained regime (~98 epochs), techniques with zero per-step overhead have outsized value.

**Hypothesis**:
Adding EMA with β=0.999 will improve best_test_acc from 95.39% to ~95.5-95.7% by smoothing out SGD noise in the final weights, providing implicit regularization that complements the existing augmentation+WD pipeline without any throughput cost. The EMA model should show consistent improvement over raw weights across all evaluation epochs, with the gap widening as training progresses and weight oscillations increase near LR drops.
