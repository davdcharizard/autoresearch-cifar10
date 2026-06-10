# Brainstorm EXP-010
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external search this loop. The candidate techniques (SiLU/Swish activation, per-channel input
  standardization, Mixup) are textbook and already grounded in the project's own history + standard CIFAR practice.
  Knowledge base (`.autoresearch/knowledge/README.md`) is empty. The binding evidence here is internal
  (the 96.0 plateau, the augmentation-as-lever result, the exhausted-axes record).

## Experimental History Review

Current best **96.00%** (EXP-003, commit f59de56): k=4 WideResNet {64,128,256} 4.3M + 1×1 projection shortcuts +
Cutout(16) GPU-vectorized + bf16/channels_last + time-fraction cosine (peak 0.2, 5% warmup) + Nesterov + label
smoothing 0.1, seed 42, ~77 epochs, converged (final_test_loss 0.204).

**SIX axes now exhausted** (11 experiments, 5 improvements):
- Width/capacity — k=4 sweet spot; k=6 (EXP-004) and compiled k=5 (EXP-009, 94.21) both regress via the epoch wall.
- Regularization — Cutout big win (EXP-002/003); WD 5e-4 marginal (EXP-005). Near-saturated.
- Weight-averaging — EMA 95.97 (EXP-006), redundant with cosine-to-0.
- Training-length/throughput — compile bought 77→89 epochs, acc 95.92 (EXP-007): **converged past ~77, more epochs
  don't help** (this is the KEY fact — the model is generalization-bound, not optimization-bound).
- Channel-attention — SE 95.86 (EXP-008), not channel-gating-limited.
- Compiled-capacity-scaling — EXP-009, refuted the "compile re-opens capacity" hope.

Genuinely-untried gaps (all low-ceiling on a converged plateau):
1. **Activation function** (ReLU → SiLU/Swish) — the nonlinearity axis, completely untried, ~zero epoch cost.
2. **Per-channel input standardization** — the code uses `std=(1,1,1)` (train.py L152-155), i.e. it mean-subtracts
   but does NOT divide by per-channel std. Standard CIFAR practice divides by (0.247,0.243,0.261). Untried, free.
3. **Complementary augmentation** (Mixup/CutMix) — a different regularizer mechanism than Cutout's occlusion.

Blocked/closed: TTA & ensembling (eval harness frozen, one model, ≤1 eval/epoch); more capacity (axis closed);
more epochs (converged).

## Candidate Ideas

First-principles framing: the model is **converged** at 77 epochs (EXP-007 proved more epochs don't help) and sits
at the capacity sweet spot. So test accuracy is **generalization-bound at fixed capacity**, not optimization- or
capacity-bound. The levers that can still move it are ones that improve the generalization of the *same-size,
same-budget* model: a better-conditioned nonlinearity, better-conditioned inputs, or a complementary regularizer
that doesn't cost more epochs than the budget allows.

### 1. SiLU (Swish) activation in place of ReLU
**Summary**: Replace the three `F.relu` sites — two in `BasicBlock.forward` (post-bn1, post-residual) and one in
`ResNet.forward` (stem) — with `F.silu` (x·sigmoid(x)). k=4 and the full recipe otherwise byte-identical. Optionally
add the validated `torch.compile(reduce-overhead)` enabler if the extra sigmoid measurably cuts epochs, to keep the
test fair (same de-risking EXP-008 used for SE) — decided at plan time from a quick dt check.

**Reasoning**: SiLU is a smooth, non-monotonic activation with consistent small gains over ReLU on image
classifiers (the default in EfficientNet). Mechanism: nonzero gradient for small negatives + smoothness → better
optimization landscape and marginally better generalization, at fixed capacity. It is the one architectural axis
never touched here, orthogonal to all six exhausted axes, and nearly free (elementwise, no params, far cheaper than
SE's FC layers — so unlike SE it should barely dent the epoch count).

**Sources**: Swish/SiLU literature (Ramachandran 2017; EfficientNet); codebase `train.py:89-92,127`; EXP-009 report
Next Steps #1 ("best next experiment"); goal-learnings (six exhausted axes). project-insights § Medium (compile
enabler usable if needed).

**Estimated Effort**: low — 3-line swap (+ optional 2-line compile enabler).

**Risk Assessment**: Most likely a sub-0.2pp delta within the noise band (no-improvement) — activation swaps are
typically small on compact CIFAR ResNets, especially with BN already smoothing the landscape. Safe failure mode
(no crash). Minor epoch cost from the sigmoid; mitigated by the optional compile enabler. Low ceiling, but the
cleanest and best-evidenced of the remaining probes.

### 2. Per-channel input standardization (fix `std=(1,1,1)`)
**Summary**: Change the `Normalize` std from `(1,1,1)` to the standard CIFAR-10 per-channel std
`(0.2470, 0.2435, 0.2616)` so inputs are divided by their channel std (currently only mean-subtracted). k=4 and
everything else fixed.

**Reasoning**: Proper input standardization is textbook and currently disabled by a deliberate
"faithful-to-original-paper" choice (train.py comment L155). The goal explicitly invites modernizing the setup and
treats data handling as fair game. Mechanism: unit-variance inputs better condition the first conv's gradients and
the initial optimization. Free — no epoch cost, a one-tuple change.

**Sources**: standard CIFAR normalization constants; codebase `train.py:152-163`; goal statement ("modernize the
training setup… data augmentation is fair game").

**Estimated Effort**: low — one-line constant change.

**Risk Assessment**: Likely near-neutral: a BatchNorm immediately follows conv1, so it largely re-normalizes
whatever scale the inputs have — the benefit of input std-scaling is mostly absorbed downstream. Could even be
within noise either direction. Safe, free, but low expected impact for exactly that reason.

### 3. Mixup augmentation (complementary regularizer)
**Summary**: Add Mixup (Zhang 2018): per batch, sample λ~Beta(α,α), mix inputs `x = λx + (1−λ)x_perm` and apply the
mixed-target loss `λ·CE(y) + (1−λ)·CE(y_perm)`, vectorized on-GPU like Cutout. α≈0.2. Stacks with or replaces Cutout.

**Reasoning**: Mixup is a strong, well-evidenced CIFAR regularizer with a *different* mechanism than Cutout
(convex interpolation of examples/labels vs occlusion) — and augmentation has been this project's most productive
lever after capacity (Cutout: +0.58/+0.52, EXP-002/003). It improves generalization without adding capacity.

**Sources**: Mixup (Zhang et al. 2018, ICLR); goal-learnings § Patterns (augmentation is a live lever); EXP-002/003.

**Estimated Effort**: low-medium — ~10 lines (Beta sample, batch permute, mixed loss), GPU-vectorized.

**Risk Assessment**: Mixup typically needs MORE epochs to pay off (it slows convergence by making the task harder),
a poor fit for the converged 77-epoch budget — the dominant risk is under-convergence pushing acc down, the same
trap that sank heavy capacity. EXP-005 also found the regularization axis near-saturated. Medium risk of a soft
regression. Higher ceiling than ideas 1–2 *if* it converges, but the budget makes that doubtful.

## Idea Evaluation

**Evidence strength**: All three are textbook. SiLU (idea 1) has the most directly-relevant evidence (consistent
small gains on conv classifiers) AND was pre-identified as the best next probe in the EXP-009 report. Std-norm
(idea 2) is textbook-correct but its benefit is specifically the kind BN absorbs. Mixup (idea 3) has the strongest
raw accuracy evidence but in a *longer-budget* regime that doesn't match ours.

**Mechanism clarity**: SiLU — clear, small. Std-norm — clear, but the mechanism is largely neutralized by the
immediate BN. Mixup — clear and strong, but the same mechanism that helps (harder task → better generalization)
costs convergence speed we can't afford.

**Expected impact**: On a converged, generalization-bound plateau all three are low-ceiling. SiLU is the most
likely to yield a *clean, fair* signal (cheap, doesn't fight the budget). Mixup has the highest ceiling but the
highest chance of a budget-driven regression. Std-norm is the lowest expected impact (BN absorption).

**Risk profile**: SiLU and std-norm fail gracefully and cheaply. Mixup risks a soft regression via under-convergence.

**Feasibility**: All low-effort. SiLU and std-norm are trivial; Mixup is slightly more code.

**Conclusion**: SiLU (idea 1) is the best lead — cleanest mechanism, best-matched evidence, cheapest, fair test,
and the pre-registered next step. Std-norm is a reasonable near-free follow-up; Mixup is deferred as budget-risky.

## Chosen Idea
**Selected**: SiLU (Swish) activation in place of ReLU (Idea 1)

**Why this idea**:
The model is converged and generalization-bound at the capacity sweet spot, so the only live levers are ones that
improve generalization of the same-size/same-budget model. SiLU is the one architectural axis never tried, has the
best-matched evidence (consistent small gains on conv nets), is nearly free (elementwise — unlike SE it won't
epoch-starve), fails gracefully, and was pre-identified as the best next experiment in the EXP-009 report. It gives
the cleanest fair test of the remaining options.

**Hypothesis**:
Swapping ReLU→SiLU on the converged k=4 recipe will keep ~75–77 epochs (negligible per-step cost; compile enabler
added if a dt check shows otherwise) and yield a small generalization gain, lifting `best_test_acc` above the 96.10
bar (expected ~96.1–96.3%). If it lands within ~0.2pp of 96.0, that confirms the nonlinearity axis is also
non-binding for this model and narrows the remaining search to input/regularization micro-tweaks before declaring
convergence.
