# Proposal: Early Drop-Path on the Added Stage-3 Block

## Summary

Keep the accepted EXP-027 model and data recipe exactly: stage depths
`(2,2,3)`, widths `[32,64,128]`, early worker-safe
`RandAugment(num_ops=1,magnitude=5)`, batch-shared alpha-0.2 mixup, FP32
SGD/Nesterov, and the 300-second time-aligned schedule. During only the first
65% of counted training, apply per-example stochastic depth with probability
`p=0.05` to the residual branch of `layer3[2]`, the third and newly accepted
8x8 block. At 65%, disable it exactly with mixup. Evaluation always uses the
full block, and every hard-tail training example uses the exact accepted full
model with no mask generation or residual rescaling.

This targets a specific interaction rather than adding general regularization.
The extra block alone reached 94.15% but had 0.2782 test loss, while early
RandAugment alone reached 94.12%; their exact composition reached the accepted
94.32% and improved the deeper model's final loss to 0.2523. A small drop-path
rate may stop the added block from becoming a brittle early shortcut while
preserving all of its capacity for the final 35% hard-label refinement.

The alternative label-smoothed bridge is weaker. It has no scored local result,
prior reviews explicitly downgraded it for consuming the validated hard-label
tail, and the accepted first 65% already combines mixup soft targets with image
augmentation. Retain it as an unselected idea; do not combine it with drop-path.

## Exact Mechanism

The accepted third stage contains three `PreActBlock(128,128,stride=1)`
modules. Modify only `layer3[2]`. Its accepted computation is:

```text
shortcut = x
r = conv2(relu(bn2(conv1(relu(bn1(x))))))
y = shortcut + r
```

For training steps with counted progress `< 0.65`, replace only the addition
with per-example inverted drop-path:

```text
keep = 0.95
mask shape = [batch, 1, 1, 1]
mask_i = 1 if rand_i >= 0.05 else 0
y_i = x_i + r_i * mask_i / keep
```

Use one mask value per example, shared across all 128 channels and 8x8 spatial
positions. Do not use elementwise dropout, channel dropout, per-block masks,
or a depth-dependent schedule. Do not skip computation of the residual branch;
the treatment is a regularizer, not a conditional-compute optimization. The
`1/0.95` factor preserves the residual branch expectation during active
training.

The probability is exactly `0.05`; it is not tuned after timing or accuracy.
Only `layer3[2]` has nonzero probability. All other six residual blocks
remain accepted and deterministic. The model topology, initialization,
parameters, buffers, optimizer membership, and convolution/linear cost remain
exactly accepted: **987,098 trainable parameters** and **119,981,312 MACs per
image** before the tiny mask multiply.

At progress `>= 0.65`, take an explicit `p == 0.0` guard before mask allocation
and return `x + r` directly. This guard must consume no RNG and perform no
division or multiplication. `model.eval()` must also return `x + r` regardless
of the configured early probability, so every validation uses the full
accepted model even before the temporal cutoff.

Tie the transition to the existing per-step `use_mixup` predicate: while
`use_mixup` is true, use `p=0.05`; at the first hard-label step, set the block
to `p=0.0` once and log the transition. Preserve EXP-027's RandAugment
exhausted-iterator transition exactly. RandAugment may therefore finish the
already-created epoch shortly after 65%, as accepted, but stochastic depth and
mixup both stop at the exact counted-time branch.

## RNG and Semantic Controls

Adding a mask to the default CUDA RNG would shift future mixup coefficients
and permutations, turning this into a training-stream reroll. Use a dedicated
CUDA `torch.Generator` for drop-path, seeded exactly once with fixed seed
`28028`. Constructing and seeding it must not alter the global CPU or CUDA RNG
states. Pass it only to the third block's mask helper. Do not call
`torch.manual_seed` or `torch.cuda.manual_seed` for this stream, and do not
change seed 42 for model, sampler, crop/flip, or mixup.

The fixed experiment-local seed is preregistered and never varied. It isolates
the intervention; it is not seed optimization. Active drop-path advances only
this private generator. Evaluation and `p=0` hard-tail calls do not advance it
or any global generator.

An evaluator-free semantic preflight must prove:

- source and runtime constants are exactly `p=0.05`, cutoff `0.65`, and private
  seed `28028`;
- the initial model state, parameter identities, optimizer groups, and
  post-construction CPU/CUDA global RNG states are bitwise identical to the
  accepted EXP-027 oracle;
- `layer3[2]` alone receives drop-path and its identity shortcut is unchanged;
- with `p=0` in train mode, logits, loss, every gradient, one optimizer update,
  and global/private RNG states are bitwise identical to accepted;
- eval mode is likewise bitwise identical and RNG-free even if the stored
  probability is `0.05`;
- active mode is deterministic when the private generator state is restored,
  changes only the third residual contribution, leaves global CPU/CUDA RNG
  untouched, advances the private state, produces finite gradients, and has
  empirical drop frequency within `[0.04,0.06]` over a sufficiently large
  synthetic sample;
- a forced dropped sample returns its identity shortcut exactly, while a kept
  sample uses exactly `r/0.95`, with a `[B,1,1,1]` mask;
- after the one-way cutoff, repeated training forwards consume no private mask
  RNG and are bitwise accepted;
- EXP-027 transform order, worker-private RandAugment RNG, sampler stream,
  exhausted-iterator flag semantics, and clean-tail replay remain unchanged.

The preflight must stub `prepare.Eval` before import and must not construct the
real evaluator, inspect test data, report accuracy, or write `run.log`.

## Distinction From EXP-006

EXP-006 applied `F.dropout(p=0.10)` inside **all six** residual branches of the
shallower `(2,2,2)` model, after the second BN/ReLU and before `conv2`. That
independently zeroed activation elements feeding every final branch convolution
and scored 93.52%, 0.55 points below its baseline despite normal exposure.

This proposal is narrower in four material ways:

- it targets only the added `layer3[2]` block whose standalone high test loss
  supplies a specific regularization rationale;
- it drops an entire residual contribution per example after `conv2`, leaving
  internal feature computation and the identity path intact;
- `p=0.05` is half EXP-006's rate and applies to one of seven blocks rather
  than all residual transformations;
- it operates on the accepted deeper-plus-RandAugment interaction, a model not
  available in EXP-006.

This distinction makes the experiment interpretable, not safe. It still stacks
a third early regularizer with mixup and RandAugment. A negative run at normal
exposure should be read as evidence that even targeted masking disrupts the
successful interaction, and should close adjacent drop probabilities rather
than invite `p=0.025` or `p=0.10` retries.

## Why Not a Label-Smoothed Bridge

A bridge such as epsilon-0.05 smoothing from 65% to 85% would be cheap and
would not overlap mixup in time. Its evidence is nevertheless weaker:

- no label-smoothing treatment has a scored local positive result;
- the EXP-013 blind review scored transition smoothing only 5.5/10 for evidence
  and 5.0/10 for impact, specifically because it repeats negative soft-target
  directions and shortens the useful hard-label tail;
- mixup cutoff experiments at 50% and 75% bracket 65%, showing that the balance
  between soft targets and clean refinement is locally sensitive;
- stronger mixup and other additive regularizers regressed, while EXP-027's
  improvement came from image invariance interacting with capacity, not from
  extending soft targets.

Drop-path also has weak direct evidence, but it is better targeted to the one
component whose generalization changed across EXP-011 and EXP-027. Do not add
label smoothing to this run, use it as a fallback after observing the score,
or shorten the accepted hard-label phase.

## Matched H20 Timing Gate

The mask is low cost but lies inside the counted GPU path. Benchmark accepted
and candidate in one evaluator-free process on exactly one NVIDIA H20. Use
identical accepted model states, optimizer states, fixed pinned host inputs and
targets, and independent initially equal global training RNG streams. The
candidate additionally has its fixed private mask stream.

Time the full production body: nonblocking copies, LR/group writes, zero-grad,
mixup coefficient and permutation where applicable, forward/loss, finite guard,
backward, SGD/Nesterov step, and final synchronization. Warm each path for 25
steps. Measure three continuing 50-step windows per path in balanced order for:

- early mode at 50% progress: accepted mixup versus candidate mixup plus active
  `p=0.05` drop-path;
- hard-tail mode at 80% progress: accepted versus candidate with exact `p=0`
  guard, both using hard-label loss.

Use each regime's median window mean, require population CV at most 5%, and
compute `weighted_ms = 0.65 * early_ms + 0.35 * tail_ms`. Define retention as
`accepted_weighted_ms / candidate_weighted_ms` and projected passes as
`133.00736 * retention`, calibrated to EXP-027's realized accepted exposure.

Proceed to scoring only if all semantic checks pass, losses/updates are finite,
every CV is at most 5%, throughput retention is at least **0.977**, and
projected exposure is at least **130.0 passes**. The pass floor is binding
(`130 / 133.00736 = 0.97739`). Do not lower it or repeat a stable timing miss.
This gate protects the accepted exposure regime; it does not predict accuracy.

## One-Run Decision Rule

After a passing preflight, require a `train.py`-only production diff and run
exactly once with no stale log:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, one H20, 987,098 parameters, 300.0-300.1 counted seconds, total
below 600 seconds, finite loss, no more than one evaluation per epoch, exactly
one mixup/drop-path transition at the first progress-65% step, one later
exhausted-epoch RandAugment transition with accepted lag bounds, and a complete
summary. Compute realized passes as `num_steps * 256 / 50000`.

The current baseline is 94.32%, so acceptance requires exactly
`best_test_acc >= 94.42%` in the sole valid run. A score of 94.41% is a formal
no-improvement. Do not rerun seed 42, mask seed 28028, or try another
probability, cutoff, block, mask granularity, or label-smoothed fallback.

If realized exposure is at least 130 passes and accuracy is below 94.42%, close
this targeted stochastic-depth mechanism; lower accuracy or higher final loss
supports additive over-regularization, while unchanged accuracy says the small
mask is ineffective. If realized exposure is below 130, the score still counts
and cannot be rerun, but the mechanism is operationally inconclusive. Timeout,
non-finite loss, wrong RNG semantics, wrong transition, or missing summary is a
crash, not permission for a rescue run.

## Risks

- **Compounded regularization:** mixup, RandAugment, and drop-path all operate
  early. EXP-006 makes this the dominant risk even at the narrower scope.
- **Damage to the accepted interaction:** the third block may need every early
  update to absorb RandAugment invariances; dropping it can weaken the exact
  synergy that raised accuracy to 94.32%.
- **Weak effect:** only 5% masking on one low-resolution branch may fall below
  fixed-seed evaluation resolution and fail to clear the 0.10-point margin.
- **BatchNorm mismatch:** the branch's BNs execute on all samples even when its
  output is dropped. This is standard residual drop-path behavior, but training
  statistics reflect features not used by every sample.
- **No compute saving:** the residual branch is always evaluated; mask overhead
  can only reduce exposure, hence the strict timing gate.
- **Single local seed:** the private stream is necessary for attribution but one
  mask realization cannot establish variance. The no-reroll rule remains
  binding.

## Evidence

- `experiments/027/04-analysis.md`: the accepted `(2,2,3)` plus early
  RandAugment composition scored 94.32%, 94.22% final, 0.2523 loss, and 133.01
  passes.
- `experiments/011/04-analysis.md`: the same third block alone scored 94.15%
  with 0.2782 final loss, motivating regularization of that block rather than
  the whole network.
- `experiments/026/04-analysis.md`: early RandAugment alone scored 94.12% with
  exact worker-RNG isolation, establishing that EXP-027 is an interaction.
- `experiments/006/04-analysis.md`: broad p=0.10 early block dropout scored
  93.52% at normal exposure, defining the main failure mode and the required
  distinction.
- `02-system-understanding.md`: backpropagation is about 74% of counted time and
  the model nearly interpolates its hard tail, so a useful change must improve
  generalization at near-zero overhead.
- `knowledge/papers/time-matters-regularization.md`: early-only regularization
  can retain benefits after removal, supporting an exact full-model tail.
- `knowledge/papers/label-smoothing.md` and `experiments/013/01-idea-review.md`:
  smoothing can reduce confidence but local evidence opposes stacking or
  extending soft targets into the accepted hard-label window.
