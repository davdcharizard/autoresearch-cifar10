# Proposal: Weaker Alpha-0.1 Batch-Shared Mixup

## Recommendation

Starting exactly from accepted commit `67c8e98`, change only
`MIXUP_ALPHA = 0.2` to `MIXUP_ALPHA = 0.1`. Keep the batch-shared scalar
coefficient, the 65% counted-time cutoff, the `(2,2,3)` WRN, early worker-safe
`RandAugment(num_ops=1, magnitude=5)`, batch 256, FP32 SGD/Nesterov, time-based
LR curve, seed 42, data pipeline, and evaluator unchanged.

This is a low-confidence closure experiment, not a monotonic extrapolation from
the alpha-0.4 failure. The accepted alpha 0.2 may already be the local optimum,
and the duration evidence actively warns that weaker regularization can hurt.
The value of the experiment is its unusually clean attribution: it closes the
only unmeasured side of batch-shared mixup strength while preserving the full
stage-3 contribution that EXP-030 showed must not be masked.

## Evidence and Rationale

EXP-027 established the current 94.32% baseline by composing the extra
low-resolution block with early RandAugment. It retained the prior alpha-0.2
batch-shared mixup window and scored 94.22% final accuracy with 0.2523 final
loss at 133.00736 passes. EXP-030 then applied narrow p=0.05 drop-path only to
the added block and regressed to 93.91% best / 93.86% final with 0.2887 loss at
132.72064 passes. That normal-exposure miss says to preserve the full residual
transformation rather than add or tune another feature mask.

Alpha 0.1 takes the remaining non-stacking route: weaken the already-present
input/target interpolation while leaving model participation and the clean
tail intact. There is a plausible interaction-specific reason to test it now.
The alpha-0.4 score was run on the earlier `(2,2,2)` learner without
RandAugment, whereas the accepted learner is deeper and already receives early
image invariance. It is possible, but not demonstrated, that this composition
needs less interpolation softness.

The contrary evidence is stronger and must remain explicit:

- alpha 0.4 scored 93.57%, showing that stronger interpolation was harmful but
  not implying that every weaker value is better;
- ending alpha-0.2 mixup at 50% scored 93.91%, while extending it to 75%
  scored 93.82%, locally bracketing the accepted duration and showing that the
  regularization/refinement response is not monotonic;
- independent per-example alpha-0.2 coefficients scored 93.79%, so batch-level
  coefficient coherence is protected;
- EXP-027 improved both accuracy and the deeper model's loss, providing no
  direct symptom that the accepted composition is over-regularized.

The local mixup knowledge supports convex input/target interpolation as a
low-cost CIFAR generalizer, but does not identify alpha 0.1 as superior. The
proposal therefore predicts only a possible small boundary-quality gain and
precommits that one normal-exposure miss closes immediate mixup-strength
tuning.

## Exact Intervention

The production diff must be exactly one constant line in `train.py`:

```python
MIXUP_ALPHA = 0.1  # accepted: 0.2
```

No helper, control-flow, logging, seed, cutoff, or loss change is allowed. In
particular, preserve the accepted operation:

```python
mix = distribution.sample()
permutation = torch.randperm(inputs.size(0), device=inputs.device)
mixed_inputs = mix * inputs + (1.0 - mix) * inputs[permutation]
loss = mix * CE(outputs, targets) + (1.0 - mix) * CE(outputs, targets[permutation])
```

The coefficient remains one FP32 CUDA scalar shared by the whole batch. Do not
sample per example, clamp it, replace it with `max(mix, 1-mix)`, use a private
generator, force deranged pairs, or change pairing. Mixup remains active only
for pre-step progress `<0.65`; the final 35% uses the exact accepted hard-label
path. RandAugment retains its exhausted-iterator cutoff, so it may finish the
already-created epoch shortly after the exact mixup transition as in EXP-027.

Everything else remains accepted: 987,098 trainable parameters, stage depths
`(2,2,3)`, widths `[32,64,128]`, Kaiming initialization, batch 256, LR
`0.2 -> 0.002`, 5% warmup, momentum 0.9, Nesterov, matrix-only `5e-4` decay,
crop/flip plus early N1/M5 RandAugment, eight persistent workers, seed 42,
300 counted seconds, and at most one evaluation per epoch.

## Distribution Semantics

For a symmetric `Beta(a,a)` distribution, both alpha values have mean 0.5;
weakening alpha changes concentration around the endpoints, not mean mixture
orientation. The variance rises from
`1 / (4 * (2*0.2 + 1)) = 0.178571` to
`1 / (4 * (2*0.1 + 1)) = 0.208333`.

The exact theoretical mass illustrates the intended treatment:

| Event | Beta(0.2,0.2) | Beta(0.1,0.1) |
|---|---:|---:|
| `0.2 <= lambda <= 0.8` | 21.46% | 12.06% |
| `lambda <= 0.1 or lambda >= 0.9` | 67.34% | 81.28% |
| `lambda <= 0.01 or lambda >= 0.99` | 41.96% | 64.06% |

A coefficient near zero is still a near-clean example after pair orientation
is swapped: image and target weighting remain aligned. Thus alpha 0.1 presents
many more collectively near-endpoint batches while preserving the expected
0.5 coefficient and the accepted batch-level coherence. It is weaker in
typical interpolation severity, not biased toward the original target.

## Exact RNG Trajectory

Keep the accepted global seed policy unchanged:
`torch.manual_seed(42)` and `torch.cuda.manual_seed(42)`. Construct the CUDA
concentration tensors and `Beta` distribution in the same location. Model
initialization, DataLoader creation, and the pre-training CPU/CUDA RNG states
must match the `67c8e98` oracle because changing a tensor value does not itself
draw randomness.

After the first mixed step, CUDA trajectory identity is neither expected nor a
valid gate. `torch.distributions.Beta.sample()` is implemented through
concentration-dependent gamma sampling/rejection. Changing alpha changes both
the returned coefficient and potentially how the global CUDA stream is
consumed. The following device-local `torch.randperm` therefore also differs,
and later coefficients/permutations follow an alpha-defined trajectory even
under the same seed. This is intrinsic to the one-constant treatment, exactly
as documented for EXP-005; it is not seed rerolling.

Do not introduce a private generator, cache accepted draws, remap uniforms, or
attempt to realign permutations. Those would change the sampling algorithm and
create a second intervention. The scored result represents the deterministic
seed-42 alpha-0.1 process as a whole. No alternate seed or rerun is permitted.

The worker-side trajectory remains protected. Main-process CUDA Beta and
permutation draws cannot advance DataLoader worker CPU streams. EXP-027's
worker-private RandAugment RNG swapping continues to restore crop/flip state,
and `MIXUP_ALPHA` is never read in worker code. Sampler/worker behavior is
therefore unchanged by construction even though the main CUDA mixup stream
diverges.

## Semantic and Distribution Preflight

Use a fail-closed evaluator-free harness and an independent
`git show 67c8e98:train.py` oracle. It must never construct test data or call
evaluation. Before scoring, require all of the following:

- the production diff against `67c8e98` contains only `MIXUP_ALPHA = 0.2` to
  `0.1`, and `prepare.py` is unchanged;
- candidate and accepted initial model state, topology, parameter count,
  optimizer groups, learning-rate function, loader constants, and
  post-construction CPU/CUDA RNG states match exactly;
- before sampling, candidate concentration tensors equal CUDA FP32 0.1 and the
  accepted oracle's equal CUDA FP32 0.2; both are symmetric scalar `Beta`
  distributions;
- in an isolated fixed-seed distribution check of at least 100,000 draws,
  candidate values are finite and in `[0,1]`, mean lies in `[0.495,0.505]`,
  variance in `[0.203,0.214]`, central mass in `[0.115,0.127]`, and endpoint
  mass for `<=0.1 or >=0.9` in `[0.806,0.820]`;
- the same check shows alpha 0.1 has higher variance and endpoint mass, and
  lower `[0.2,0.8]` mass, than separately seed-reset alpha 0.2 draws;
- restoring the same pre-call CUDA RNG state reproduces candidate lambda,
  permutation, mixed inputs, paired targets, loss, gradients, and one optimizer
  update exactly, while the post-call candidate and accepted CUDA RNG states
  are intentionally allowed and expected to differ;
- candidate `mixup_batch` returns one scalar coefficient, applies it to every
  image and both target losses consistently, and leaves the hard-label path
  bitwise identical to accepted under cloned state/RNG;
- strict progress probes prove mixup remains active below 0.65 and disabled at
  and above 0.65, with no change to the LR values or one-way transition.

Run distribution checks in a disposable process or restore every saved RNG
state; they must not contaminate the scored process. Never inspect evaluator
outputs during preflight.

For workers, exact one-line source scope is the primary proof: require
byte-identical `EarlyRandAugment`, `make_train_transform`, DataLoader setup,
shared-byte control, and exhausted-iterator transition against `67c8e98`.
Reuse the established marker/private-RNG/clean-tail replay semantic check if
available. A new loader timing benchmark is unnecessary because no CPU-worker
code, transform, batch shape, or consumer pace changes.

## Throughput Gate

The tensor shapes and training graph are unchanged, but alpha-dependent Beta
rejection could have a tiny sampling-cost difference and the accepted exposure
has only about three passes of headroom over 130. Measure rather than assume.

On one idle NVIDIA H20, compare independent accepted and candidate modules from
equal model/optimizer snapshots. For early mixup and hard labels separately,
use fixed pinned host batches, include nonblocking H2D copies, LR writes,
zero-grad, the production Beta draw/permutation/interpolation when applicable,
forward, paired loss, finite guard, backward, SGD/Nesterov step, and final CUDA
synchronization. Use fresh deterministic fixtures per replicate, at least 20
warmups, and three balanced windows of at least 50 steps per arm. Print every
window and derived metric before enforcing gates.

Require finite values and population CV at most 5% in every arm. From regime
median step times compute the exact fixed-time retention:

```text
retention =
    (0.65 / candidate_early_ms + 0.35 / candidate_hard_ms) /
    (0.65 / accepted_early_ms + 0.35 / accepted_hard_ms)
projected_passes = 133.00736 * retention
```

Proceed only if retention is at least `0.9774` and projected passes are at
least 130.0. Do not lower the gate or repeat a stable miss. Hard-path timing
should be equal within measurement noise; any material difference indicates a
harness or scope error. This timing protects the intended operating regime but
does not predict accuracy.

## Sole Scored Run and Decision Rule

After all preflight checks pass, reconfirm the stored baseline is 94.32% at
`67c8e98`, confirm one idle H20 and local CIFAR-10, remove stale `run.log`, and
run exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, one complete finite summary, 300.0-300.1 counted seconds, total
below 600 seconds, 987,098 parameters, one mixup transition at the first step
with progress at least 65%, one later exhausted-epoch RandAugment transition,
unique every-fifth-epoch evaluations plus the final partial epoch, and no
traceback, OOM, worker, or non-finite error. Record realized passes as
`num_steps * 256 / 50000`. A valid completed score below 130 passes still
counts and may not be rerun, but the strength mechanism is operationally
inconclusive because it left the protected exposure regime.

The objective succeeds only when `best_test_acc >= 94.42%`, exactly 0.10
percentage points above the 94.32% baseline. Pre-register two separate
corroboration signals:

- `final_test_acc >= 94.32%` supports a stable endpoint rather than an isolated
  maximum over sparse evaluations;
- `final_test_loss <= 0.2523` supports preserved or improved boundary quality.

Neither corroboration condition may override the primary metric. A valid
94.42% best score is an objective improvement even if endpoint/loss
corroboration fails, but the report must call that mechanism evidence fragile.
Conversely, better final accuracy or loss cannot rescue a best score below
94.42%.

## One-Run Family Closure

If a valid normal-exposure run scores below 94.42%, retain alpha 0.2 and close
the immediate batch-shared mixup-strength family. Do not try alpha 0.05, 0.15,
0.25, 0.3, another seed, a different cutoff, coefficient symmetrization,
private-RNG realignment, or per-example sampling. Alpha 0.4 already closed the
stronger side; alpha 0.1 closes the weaker side; alpha 0.2 remains the calibrated
point between them.

If alpha 0.1 succeeds, it may replace alpha 0.2 on the frontier based on the
primary objective and hard constraints, but the result still does not license
adjacent alpha tuning without a new independent mechanism. The single score
tests this exact fixed-seed treatment, not the continuous optimum of alpha.

Timeout, malformed output, semantic contamination, or wrong transition is a
crash and should be analyzed as such. A weak but valid accuracy is never a
reason to rerun.

## Risks

- **Likely under-regularization:** 81.28% of alpha-0.1 draws lie outside
  `[0.1,0.9]`; many early batches become nearly clean/swapped, potentially
  losing the interpolation bias that made EXP-002 successful.
- **Negative local neighborhood:** every scored perturbation of mixup strength,
  duration, or coefficient correlation has regressed, so expected upside is
  below that of a genuinely new architecture or optimizer mechanism.
- **Alpha-dependent trajectory:** the CUDA coefficient/permutation stream
  diverges by design. One seed cannot distinguish treatment effect from its
  particular deterministic realization, and the no-reroll rule is binding.
- **Interaction uncertainty:** the deeper-plus-RandAugment learner could need
  either less interpolation because it has more augmentation, or exactly the
  accepted alpha to regularize its extra capacity. Existing results do not
  identify the sign.
- **Maximum-over-evaluations noise:** the required gain is ten additional
  correct test examples. Endpoint and loss corroboration reduce narrative
  overclaiming but cannot establish multi-seed robustness.

## Falsifiable Hypothesis

If alpha-0.2 interpolation is slightly too strong for the accepted
deeper-plus-early-RandAugment learner, then changing only the batch-shared
symmetric Beta concentration to 0.1 will retain at least 130 projected passes,
raise fixed-seed `best_test_acc` from 94.32% to at least 94.42%, and retain
`final_test_acc >=94.32%` without worsening final loss above 0.2523.

A valid normal-exposure score below 94.42% rejects weaker alpha 0.1 as a useful
standalone refinement and closes adjacent strength tuning. Endpoint and loss
are preregistered mechanism corroboration only; the goal verdict remains solely
the primary metric plus hard constraints.

## Local Evidence

- `experiments/027/04-analysis.md`: accepted `(2,2,3)` plus early RandAugment,
  alpha-0.2 learner scored 94.32% best, 94.22% final, 0.2523 loss, and 133.00736
  passes.
- `experiments/030/04-analysis.md`: targeted early drop-path scored 93.91% with
  worse 0.2887 loss at 132.72064 passes, so preserve full residual
  participation and stop mask tuning.
- `experiments/005/04-analysis.md`: alpha 0.4 scored 93.57% at normal exposure;
  it also established that alpha-dependent Beta sampling changes later CUDA
  permutations despite fixed seed 42.
- `experiments/015/04-analysis.md`: per-example alpha-0.2 coefficients scored
  93.79% at normal exposure, protecting batch-shared coherence.
- `experiments/004/04-analysis.md` and `experiments/020/04-analysis.md`: 50% and
  75% cutoffs both regressed, protecting the 65% duration.
- `02-system-understanding.md`: compute is binding, memory and I/O are not, and
  generalization/boundary quality limits the accepted learner.
- `knowledge/papers/mixup.md`: convex image/target interpolation is a validated
  low-overhead CIFAR generalizer; the local record supplies the treatment-specific
  evidence and constraints.
