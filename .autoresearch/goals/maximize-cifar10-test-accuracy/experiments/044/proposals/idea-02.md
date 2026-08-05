# Proposal: Fixed CIFAR-10 Per-Channel Training Standardization

## Recommendation

Rank last and do not advance by default. The treatment is a clean one-line
experiment, but its prior is strongly negative. Change only the training
transform's standard deviation from `(1, 1, 1)` to the rounded CIFAR-10
training-set population values `(0.2470, 0.2435, 0.2616)`. The immutable
evaluator in `prepare.py` continues to use `(1, 1, 1)`, so this is necessarily
train-only standardization rather than a matched modern preprocessing recipe.
That train/eval scale mismatch is a more serious objection than the possible
benefit from equalizing training-channel variance.

There is also direct offline adverse evidence. The neighboring v2.9.0 run
`../v2.9.0-opus-4-6/.autoresearch-dep-v2.9.0/reports/exp-report-030.md`
tested the same tuple as a sole training-transform change and fell from 96.46%
to 94.67% at unchanged throughput. Its architecture is not the current
accepted pooled-head WRN, so it is not dispositive, but a 1.79-point regression
is a much stronger local prior than generic claims that canonical
standardization is beneficial. If selected despite this evidence, allow one
tightly qualified score only, with no statistic, LR, stem, or evaluator rescue.

## Exact Fixed Treatment

At accepted commit `a7c42dc`, change exactly this line inside
`make_train_transform`:

```python
mean, std = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
```

The values are the four-decimal rounded full-training-set per-channel pixel
population standard deviations paired with the existing rounded means. They
are fixed prospectively and loaded as constants; do not compute statistics at
runtime. The frequently copied alternative `(0.2023, 0.1994, 0.2010)` uses a
different convention and is not an allowed fallback. The rounding and
estimator convention are not uniquely privileged by this learner, which is
another reason to treat this as low-confidence rather than as a correction.

Keep the transform order exact:

```text
RandomCrop -> RandomHorizontalFlip -> EarlyRandAugment (when active)
-> ToTensor -> Normalize(existing mean, new std)
```

Thus for an augmented tensor `x`, accepted input is `x - mean` and candidate
input is `(x - mean) / std`, independently by channel. Preserve the means,
crop/flip policies, RandAugment N1/M5 configuration and PIL fill, temporal
cutoffs, mixup placement and alpha, all RNG, loader behavior, model, loss,
optimizer, schedule, evaluation cadence, and summary. Do not modify
`prepare.py`; doing so violates the goal.

## Scale, Initialization, BatchNorm, and Optimizer Semantics

The candidate preserves model parameter and buffer initialization byte for
byte. With stem weights `W`, however, its initial training stem function is

```text
Conv(W, diag(1/std) * (x - mean)),
```

which is functionally equivalent on training inputs to rescaling each stem
input-channel slice of `W` by `1/std`. It is therefore not function-preserving
at initialization even though the stored weights and RNG are unchanged.

The stem has no preceding normalization, but `layer1[0].bn1` immediately
normalizes its output. Training-mode BatchNorm approximately absorbs a common
positive input scale: scaling both the stem activation and its backward path
causes the BN Jacobian to counter-scale. Accordingly, the simplistic claim
that dividing inputs by about four creates a four-times larger effective stem
learning rate is not generally correct here. The unequal RGB factors, BN
epsilon, finite-batch statistics, ReLU/shortcut structure, weight decay, and
optimization transients prevent exact invariance, but much of the intended
conditioning benefit may be absorbed.

The immutable evaluator creates the sharper non-invariance. Candidate BN
running statistics are learned from standardized training inputs, while test
images remain only mean-centered. A scalar change in stem weights or BN affine
parameters cannot remove the ratio between the training and evaluation input
scales because it acts on both paths. Evaluation therefore presents roughly
one-quarter-scale, channel-skewed stem activations relative to the statistics
accumulated during training. Initial evaluation behavior is accepted before
training, but the learned function and BN buffers diverge immediately.

SGD remains coupled-decay Nesterov with LR `0.2 -> 0.002`, momentum `0.9`, and
matrix decay `5e-4`. No gradient is manually rescaled: autograd supplies the
gradient induced by the changed training function, decay is then added in the
accepted parameter units, and momentum buffers accumulate those changed
directions normally. The relative balance of data gradient, decay, and
momentum can therefore change even where BN makes the forward activations
nearly scale-invariant. LR compensation, stem-specific LR/decay, alternative
initialization, frozen BN, or recalibrated BN would be separate multivariable
experiments and are forbidden rescues.

Kaiming initialization preserves variance relative to its input variance; it
does not itself guarantee that unit-variance pixels improve this already
BatchNorm-mediated network. There is no current result diagnosing RGB variance
imbalance as the remaining 5.52% error source. This mechanism is thus
plausible only as a small channel-conditioning change and is more likely
neutralized by BN or overwhelmed by evaluation mismatch.

## Distinction From EXP043 Gradient Centralization

EXP043 changed the post-backward update by projecting every convolution data
gradient and removed useful common-mode directions, scoring 93.88% at 129.81
passes. This proposal performs no projection, mean subtraction of gradients,
norm operation, hook, or optimizer edit. It deterministically rescales input
channels before the forward pass, after which raw autograd gradients are used
unchanged. It alters the training function, stem parameterization, and BN
statistics rather than deleting gradient directions. EXP043 therefore does
not directly test this mechanism, but its negative result supplies no reason
to expect another broad change to convolution learning geometry to help.

## Falsifiable Hypothesis and Risks

The hypothesis is: if equalized RGB training variance improves useful stem
conditioning more than BatchNorm absorbs it and more than the fixed evaluator
scale mismatch harms it, then the exact tuple will retain at least 127 data
passes and raise fixed-seed `best_test_acc` from 94.48% to at least 94.58%.

Principal risks are the structurally mismatched train/eval distributions; an
effect largely absorbed by the first pre-activation BN; altered BN running
statistics, ReLU gates, coupled-decay balance, and momentum trajectory; an
arbitrary choice among common CIFAR statistic conventions; and a strong prior
exact-treatment regression in the neighboring run. The unchanged graph makes
throughput risk small, but a sole fixed-seed score cannot estimate average
effect size.

## Semantic, Worker, and RNG Qualification

Use an ignored evaluator-free preflight against `git show a7c42dc:train.py`.
Print measurements before assertions and require:

- `train.py` is the sole production file changed and the tuple is the sole
  semantic diff; `prepare.py` remains hashed and unchanged with eval std ones;
- initial named parameters, buffers, bytes, count `1,003,482`, optimizer groups
  and options, post-construction CPU/CUDA RNG states, schedule, loss, temporal
  constants, and evaluator call sites match accepted exactly;
- from identical augmented pixels, candidate tensors equal independently
  computed `(accepted_tensor / std)` channelwise within fixed FP32 tolerance,
  with unchanged dtype, shape, targets, sample order, and no source mutation;
- crop, flip, and EarlyRandAugment decisions and their worker RNG states match
  accepted for fixed multi-worker fixtures; Normalize consumes no RNG;
- under the production multiprocessing context with eight persistent workers,
  accepted and candidate loaders replay identical target/order and inverse-
  normalized augmented pixels across at least two exhausted epochs;
- the shared RandAugment flag flips only after a fully exhausted iterator, the
  clean tail contains no prefetched RandAugment leakage, and worker-private
  RandAugment isolation remains exact;
- deterministic early-mixup and hard-label steps are finite and reproducible;
  accepted/candidate mix coefficients, permutations, RNG, and labels match,
  while normalized inputs, logits, gradients, updates, and BN statistics differ
  nontrivially as intended.

A semantic failure closes the implementation. Correct only a verifier defect;
do not change statistics, means, transform placement, BN handling, optimizer,
or evaluator normalization.

## H20 Timing and Exposure Gate

Although the transform executes in workers and uses the same Normalize kernel,
qualify the full path. On one idle H20, run four counterbalanced accepted and
candidate active-loader windows in both early and hard regimes, with at least
20 warmup steps and at least 50 measured complete steps per arm/window. Preserve
production H2D, LR writes, mixup when active, forward, loss, finite check,
backward, coupled Nesterov update, and synchronization. Also record four full
exhausted-epoch loader wall times per arm because batch retrieval occurs before
the inner-loop training timer. Require population CV at most 5% for each series,
no worker stall/outlier, and projected total wall time below 600 seconds.

Using medians, compute:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms  + 0.35 / accepted_hard_ms)
projected_passes = 130.304 * retention
```

Proceed only if retention is at least `127 / 130.304 = 0.9746439096`, projected
passes are at least 127, and active-loader median epoch time regresses by no
more than 5%. A stable miss closes feasibility; do not rerun windows, lower the
floor, move normalization to CUDA, or alter worker settings.

## Sole Score and One-Score Closure

After both gates pass, reconfirm accepted baseline 94.48% at `a7c42dc`, an idle
single NVIDIA H20, local CIFAR-10, frozen `prepare.py`, exact reviewed diff, and
no stale log. Run once at seed 42:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit zero, one finite summary, 300.0-300.1 counted seconds, wall time
under 600 seconds, 1,003,482 parameters, correct one-way transitions, unique
every-fifth plus final evaluations, and no traceback, worker, OOM, evaluator,
or non-finite error. Record steps, `steps * 256 / 50000` passes, VRAM,
best/final accuracy, final loss, and best-final gap.

Primary success is solely `best_test_acc >= 94.58%`. Final accuracy at least
94.45% and final loss at most 0.2456 are corroboration only and cannot rescue a
miss or veto a primary success. At least 127 realized passes is required for a
normal-exposure mechanism claim. A valid lower-exposure score still counts and
cannot be rerun.

A normal-exposure score below 94.58% closes this exact fixed training-only
standardization and immediate normalization rescues: no alternate std tuple,
extra precision, recomputed statistics, partial interpolation toward one,
mean change, transform relocation, LR/decay compensation, stem rescaling,
BN recalibration, matched evaluator edit, different seed, or rerun. Success
supports only this exact tuple under the immutable evaluator, not a broader
claim that canonical normalization is universally superior. Invalid execution
permits correction only for an independently proven infrastructure or harness
defect; it never permits replacing a valid score.

## Final Evaluation

**Evidence and reasoning: 1/5.** The constants and intervention are precise,
but the evaluator mismatch, approximate BN scale invariance, lack of a current
error diagnosis, and exact-treatment neighboring regression dominate.

**Potential impact: 1/5.** It is nearly free and could in principle alter RGB
conditioning, but clearing +0.10 is unlikely and a material regression is
credible.

**Overall:** executable as a one-score closure, but reject in favor of any
candidate with a mechanism compatible with the immutable evaluation pipeline.
