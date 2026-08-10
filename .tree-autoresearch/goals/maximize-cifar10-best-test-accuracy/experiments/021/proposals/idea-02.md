# Proposal: Hard-Batch Single-View Supervised Contrastive Auxiliary Loss

## Intervention

Add a fixed, training-only supervised contrastive term to EXP004's unchanged
main cross-entropy objective. Reuse the final 256-dimensional pooled feature
that already feeds `fc`; do not run a second backbone view or add a projection
head. On eligible hard-label batches, optimize

```text
L_total = L_CE + 0.05 * L_SupCon(T=0.1)
```

where `L_CE` is exactly the inherited clean cross-entropy and `L_SupCon` uses
all other same-class examples in the current batch as positives. On CutMix
batches, retain the exact inherited area-weighted two-target CE and set the
auxiliary contribution to zero. Preserve every other part of EXP004: WRN-16-4,
front-loaded CutMix, drop-path schedule, charged-time LR schedule, period-two
clean-tail SAM, Nesterov SGD, weight decay, BF16 autocast, channels-last
layout, batch size 256, seed 42, 300-second charged budget, and once-per-epoch
evaluation. Only `train.py` changes.

Use these fixed constants, with no coefficient, temperature, phase, or
attachment sweep:

```python
SUPCON_WEIGHT = 0.05
SUPCON_TEMPERATURE = 0.1
SUPCON_SCOPE = "hard_batches"
SUPCON_FEATURE_DIM = 256
```

The model's ordinary `forward` must continue to return only `[B, 10]` logits.
An explicit training-only option may return `(logits, pooled_features)` from
the same backbone pass. The frozen evaluator must always use the default
logit-only path. The feature is the post-final-BatchNorm, post-ReLU,
adaptive-average-pooled 256-vector immediately before `fc`, so it is the exact
representation consumed by the deployed classifier.

## Exact SupCon Objective

For hard-label batch targets `y` and pooled features `h`, compute normalized
FP32 features and a full same-batch similarity matrix:

```python
z = F.normalize(h.float(), dim=1)
similarity = z @ z.T / SUPCON_TEMPERATURE
```

For anchor `i`, let `A(i)` be every batch index other than `i`, and let `P(i)`
be indices in `A(i)` with `y_p == y_i`. Use the preferred many-positive
supervised contrastive form:

```text
L_i = -1/|P(i)| * sum[p in P(i)]
      log(exp(sim(i,p)) / sum[a in A(i)] exp(sim(i,a)))
L_SupCon = mean[i with |P(i)| > 0] L_i
```

Mask the diagonal to negative infinity before FP32 `logsumexp`; never include
the anchor itself as a positive or denominator element. Similarities lie in
`[-10, 10]` at temperature 0.1 after normalization, and the FP32 reduction
avoids fragile BF16 exponentials. Singleton-class anchors remain valid
negatives for other anchors but do not contribute an anchor loss. If no
anchor has a positive, return `h.float().sum() * 0.0`, producing an exact
finite differentiable zero rather than a NaN or a divide-by-zero.

At batch size 256 with approximately balanced CIFAR-10 shuffling, an anchor
normally has about 25 same-class examples including itself, hence about 24
positives. Under an independent class approximation, an anchor's probability
of having no other same-class example is `0.9**255`, roughly `2.1e-12`;
nevertheless the implementation must handle singleton and all-zero-positive
cases exactly rather than relying on this rarity.

## Projection Choice

Use no projection head. The original NeurIPS method uses a disposable
projection head, two augmented views, long contrastive training, and a
separate linear-classifier stage
(`experiments/021/papers/supervised-contrastive-learning.md`). Those choices
make sense when the contrastive objective is the primary pretraining loss, but
they are not cost-free or directly validated for this joint CE, fixed-time
setting. Even a small MLP introduces extra parameters, optimizer and SAM
snapshot semantics, and multiple tiny kernels whose launch cost matters more
than memory on the H20. It also permits the projection space to absorb much of
the contrastive geometry without guaranteeing that the classifier's input
changes usefully.

Applying the term directly to normalized pooled features makes the proposal a
clean test of whether same-class batch geometry improves the deployed
representation. The ICML analysis finds that CE and SupCon share a
simplex-like class-collapse optimum under its assumptions
(`experiments/021/papers/dissecting-supervised-contrastive-learning.md`), which
partly reduces concern that direct application imposes an incompatible
endpoint. It does not eliminate optimization conflict: the no-projection
choice is a deliberate efficiency and attribution tradeoff, and a negative
result would reject this direct-feature formulation rather than projection-
head SupCon in general.

## Hard-Label and CutMix Semantics

Cross-entropy remains active on every batch. SupCon is active if and only if
`targets_b is None` in the inherited loss branch:

- Before 75% charged progress, a CutMix-selected batch keeps
  `lam * CE(logits, targets_a) + (1-lam) * CE(logits, targets_b)` and receives
  no SupCon term. A non-selected batch uses hard CE plus SupCon.
- At and after 75% charged progress, CutMix is disabled by EXP004's strict
  `< CUTMIX_END` gate, so every batch uses hard CE plus SupCon.

Do not assign a mixed image its original class, choose the dominant CutMix
class, duplicate it under both labels, or define fractional positive masks.
Each would add an unvalidated target policy and can pull an ambiguous mixed
feature toward the wrong class cluster. Exclude every applied CutMix batch,
even in the rare case of a zero-area patch, because the proposal's eligibility
is defined by the intervention event and `targets_b` state, not by a
post-hoc image-content exception.

This scope supplies the auxiliary loss on an expected 62.5% of primary
optimization steps: roughly half of the first 75% plus all of the final 25%.
It preserves EXP004's successful mixed-label CE exactly while using ordinary
random crops and flips to provide variation among same-class single views.
There is no paired view of the same image, so the proposal tests class
compactness rather than the instance-view invariance central to the original
SupCon protocol.

## SAM Semantics

On ordinary steps, backpropagate the combined objective once. On every
scheduled clean-tail SAM step, both the first and perturbed passes must
recompute the same combined CE-plus-SupCon objective:

1. Run the normal forward once, compute `L_CE + 0.05 * L_SupCon`, and
   backpropagate it.
2. Construct EXP004's radius-0.05 perturbation from that joint gradient.
3. Restore the saved CUDA RNG state, disable BatchNorm running-stat tracking,
   and run the existing perturbed backbone replay on the same inputs.
4. Recompute both CE and SupCon from the perturbed logits/features, then
   backpropagate their combined loss.
5. Restore BatchNorm flags and exact parameter snapshots before the sole
   Nesterov update.

Using auxiliary loss only in the first pass would perturb for one objective
and update with the sharpness gradient of another; using it only in the second
would be equally incoherent. The two-pass rule preserves plain SAM semantics
for the new joint objective. SupCon consumes no RNG, so EXP004's CUDA replay
continues to reproduce the stochastic-depth mask and preserve the future RNG
stream. Because there is no projection head, the SAM parameter inventory,
snapshot inventory, optimizer groups, and model parameter count remain
exactly 2,748,890.

## Temperature and Weight Rationale

Fix temperature at 0.1 because that is the primary SupCon paper's reported
default, not because it was selected on this test set. Fix the auxiliary
weight at 0.05 to make SupCon a perturbative representation bias rather than a
replacement classifier objective. A randomly initialized normalized batch
has a raw contrastive loss on the order of `log(255) = 5.54`; weight 0.05
therefore contributes roughly 0.28 loss units initially, around one tenth of
an approximately `log(10)` CE loss. The contribution can become relatively
stronger as CE falls, which is intentional but also the main dose risk.

Do not normalize the weight adaptively by observed loss or gradient norm,
ramp it from test behavior, or tune it after a preflight or metric run. Those
choices would turn one falsifiable auxiliary-objective experiment into a
search. The production audit should report raw and weighted auxiliary loss so
the realized scale can be interpreted after the accuracy verdict, but those
diagnostics must not alter the fixed coefficient.

## Compute and Fixed-Budget Fit

One FP32 `256 x 256` similarity matrix contains 65,536 values (0.25 MiB), and
the dominant pairwise operation is `O(B^2 D) = 256^2 * 256`, about 16.8 million
multiply-accumulate terms per auxiliary evaluation. Masks and log-probability
matrices add only a few more sub-megabyte buffers. This is small beside a
WRN-16-4 backbone pass and uses abundant H20 memory, but backward through the
pairwise matrix and several reduction kernels is not free. Moreover, every
period-two late SAM pulse evaluates it twice. With EXP004's observed 25,560
steps and 2,449 SAM pulses, the expected dose is roughly 15,975 primary
SupCon evaluations plus 2,449 perturbed replays, about 18,400 total.

All auxiliary work must remain inside the inherited charged interval. Do not
precompute embeddings, use a memory bank, add a second image transform, or
move pairwise work outside the timer. No second backbone forward is introduced
beyond EXP004's already validated SAM replay. The intended overhead is below
5% and should retain at least 24,000 optimizer steps; this is a hypothesis to
measure accuracy-blind, not an assumption.

## Correctness and Accuracy-Blind Preflight

Before the sole metric run, perform deterministic CPU FP32 tests that verify:

- the default candidate forward has bitwise-identical logits to EXP004 and
  the feature-return path yields `[B, 10]` logits plus `[B, 256]` pooled
  features whose logits equal `fc(features)`;
- the vectorized objective matches a small explicit-loop reference on labels
  containing multiple positives, singleton classes, all-one-class samples,
  and a batch with no positive anchors;
- self-similarities are excluded, positives are exactly same-label non-self
  entries, permutation of the batch leaves the scalar loss unchanged, and
  feature rescaling before normalization leaves it unchanged within tolerance;
- auxiliary-only backward produces finite nonzero gradients in the pooled
  representation and backbone, while the differentiable-zero edge case is
  finite and has exact zero gradient;
- clean batches combine CE and `0.05 * SupCon`, while applied CutMix batches
  execute the unchanged two-target CE and make zero SupCon calls; and
- constructing and evaluating the loss consumes no CPU or CUDA random draws.

Then run one decisive accuracy-blind BF16/channels-last preflight on physical
GPU 0, with the evaluator and test-loader access replaced by guards that
raise. Check finite clean and CutMix forward/backward/update behavior, the
joint loss decomposition, exact CutMix exclusion, and a real scheduled SAM
step. For SAM, verify a radius-0.05 joint-gradient perturbation, distinct
perturbed CE and SupCon values, identical first/replay stochastic masks, one
BatchNorm running-buffer update, exact parameter restoration before the
optimizer step, one optimizer update, and unchanged future CUDA RNG state.

Benchmark at least five alternating-order paired parent/candidate rounds after
warmup using the inherited expected time mixture: 37.5% early CutMix without
SupCon, 37.5% early hard with SupCon, 12.5% late hard without a SAM replay, and
12.5% late hard with a two-pass SAM update and SupCon on both passes. Include
data mixing, loss construction, backward, SAM machinery, diagnostics, and
optimizer work inside synchronized timings, and reset equivalent
model/optimizer/RNG state for each comparison. The first complete numeric gate
is decisive; retry only an exception, failed assertion, or malformed result
before the gate is emitted.

Proceed only if correctness passes, parent timing drift is at most 4%, median
weighted candidate/parent latency is at most 1.05, no paired ratio exceeds
1.08, projected exposure is at least 24,000 steps and 124 epochs, projected
total runtime remains below 600 seconds, and peak-allocation growth is finite.
VRAM is reported but is not itself a rejection gate. Do not alter temperature,
weight, precision, eligibility, or matrix formulation in response to timing.

## Production Diagnostics and Integrity Contract

The startup config and final audit should make the exact mechanism observable:

- temperature 0.1, weight 0.05, direct pooled-feature attachment, no
  projection, hard-batch-only scope, feature dimension 256, and FP32
  similarity/reduction precision;
- primary SupCon calls, perturbed-replay SupCon calls, CutMix skips, and the
  identity `primary_calls == num_steps - cutmix_applied_batches` plus
  `replay_calls == sam_applied_batches`;
- valid-anchor count, zero-positive anchor count, all-zero-positive batch
  count, total positive pairs, and mean positives per valid anchor;
- aggregate or synchronization-safe sampled raw SupCon loss, weighted SupCon
  contribution, and main CE, separated for primary and SAM replay calls; and
- inherited CutMix and SAM exposure, first SAM step/progress, evaluator calls,
  completed epochs/steps, charged and total time, peak VRAM, unchanged model
  parameters, and the complete required metric summary.

Do not add a per-step synchronization solely for diagnostics. Read diagnostic
scalars only after EXP004's existing synchronization, or maintain bounded
device accumulators and transfer them after training; include whichever method
is chosen in the latency preflight. Do not retain per-step features or
similarity matrices. Require exit 0, finite losses, exactly one evaluation per
completed epoch, 299.5-301.0 charged seconds, total runtime below 600 seconds,
at least 24,000 steps, complete summaries, and exact counter identities.

Evaluation must remain isolated: auxiliary features or similarities cannot be
used for test-time normalization, k-nearest-neighbor prediction, classifier
refitting, ensembling, checkpoint selection, or any additional validation.
`best_test_acc` remains the frozen evaluator's accuracy on the ordinary main
logits.

## Evidence, Expected Effect, and Falsification

Khosla et al. provide direct evidence that normalized, many-positive
same-class contrastive objectives can improve classification representations
(`experiments/021/papers/supervised-contrastive-learning.md`). The proposed
same-batch matrix and temperature follow that mechanism, while avoiding an
extra backbone view under the fixed compute budget. EXP004's final-equals-best
95.40% and lower final loss show a stable CE/SAM solution, so the plausible
benefit is not recovery from failed fitting; it is a modest shift toward
tighter within-class features and larger inter-class margins during the same
trajectory.

The evidence transfer is materially incomplete. The source method uses two
views, a projection MLP, much longer training, and a separate linear probe,
whereas this proposal uses one view, no projection, joint CE, CutMix exclusion,
and late SAM. Graf et al. further show that CE and SupCon can share the same
ideal class geometry (`experiments/021/papers/dissecting-supervised-contrastive-learning.md`),
so the auxiliary term may add only redundant optimization pressure. Strong
CutMix, drop path, and SAM may already supply enough regularization; direct
feature SupCon may collapse useful within-class variation; and the extra
pairwise work may trade away more examples than it helps. These caveats make
the experiment exploratory but sharply interpretable.

The parent metric is 95.40%, so the formal prediction is:

> Hard-batch, single-view direct-feature SupCon at temperature 0.1 and weight
> 0.05 will preserve at least 24,000 EXP004 optimizer steps and produce
> `best_test_acc >= 95.50%` in the one fixed-seed run.

An expected effect is approximately +0.10 to +0.30 points: enough to reach
95.50-95.70 if the class-compactness bias complements SAM, but not assumed to
clear the 95.61 global best. A result at or above 95.71 supplies stronger
goal-wide evidence because it clears that best at the required 0.10-point
resolution. A valid completed result below 95.50 falsifies this exact
single-view, no-projection, hard-only dose. A failure of the accuracy-blind
latency gate rejects it as infeasible under the fixed budget without making an
accuracy claim. Neither outcome licenses a temperature, coefficient,
projection, phase, or seed retry within EXP021.

## Estimated Effort and Risk

**Estimated effort: medium.** The model change is small, but the masked
many-positive loss, zero-positive behavior, SAM replay equivalence, evaluator
isolation, and fixed-budget diagnostics require focused tests.

**Risk: medium-high.** The mechanism has strong literature support in its
original protocol but weak direct support for this one-view joint-CE transfer.
Its compute should be manageable, yet SAM amplifies the pairwise cost exactly
in the clean tail where the auxiliary is always active. The fixed low weight,
CutMix exclusion, no-projection simplification, strict preflight, and single
falsifiable run bound the risk without disguising it.
