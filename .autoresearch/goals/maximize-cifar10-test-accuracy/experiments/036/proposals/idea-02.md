# Proposal: Scaled Pooled-Feature Residual MLP Head

## Summary

Preserve the accepted `(2,2,3)` WRN backbone, final `BN -> ReLU`, global
average pooling, and `128 -> 10` linear classifier, but refine the pooled
128-vector immediately before the classifier with one small residual MLP:

```python
pooled = F.adaptive_avg_pool2d(out, 1).flatten(1)
refinement = fc2(F.relu(fc1(pooled)))
pooled = pooled + 0.1 * refinement
return self.fc(pooled)
```

The two new projections are bias-free `128 -> 64` and `64 -> 128` linear
layers. Both receive the existing `WideResNet._weights_init` Kaiming-normal
initialization from a fixed, isolated CPU seed `36036`. There is no BatchNorm,
dropout, alternate activation, learnable scale, auxiliary loss, schedule
change, or optimizer override. This adds exactly 16,384 trainable parameters,
raising the accepted count from 987,098 to 1,003,482, and exactly 16,384
matrix multiply-accumulates per image before the final classifier.

The hypothesis is that the accepted early-RandAugment/deeper WRN has learned a
useful 128-dimensional spatial representation whose class boundary is still
restricted by a single affine map. A small nonlinear residual remapping can
improve class separation without paying for another spatial transform or
discarding the accepted pooled features. Fixed scale `0.1` makes the branch
nonzero and trainable from the first backward while keeping its initial
perturbation deliberately subordinate to the accepted direct path.

## Evidence and Distinction From Closed Treatments

- The accepted run at `67c8e98` scores 94.32% from 133.00736 data passes, but
  finishes at 0.2523 test loss after nearly interpolating the hard-label tail.
  Generalization and boundary quality, rather than memory or input throughput,
  are limiting (`02-system-understanding.md`). The accepted head is only one
  `128 -> 10` affine map after pooling.
- EXP-012's rank-64 **spatial** residual bottleneck scored 93.74% at 135.49
  passes. That result closes `128 -> 64 -> 64 -> 128` convolutional refinement
  on the 8x8 map; it does not test a post-pooling class-boundary remapping.
  This proposal leaves every spatial feature and stage interface untouched,
  uses two tiny dense kernels rather than three sequential convolutions, and
  adds 16,384 MACs/image versus EXP-012's 3,407,872. The distinction is
  placement and objective, not a claim that low rank itself is newly proven.
- EXP-014's exact-zero residual endpoints scored 93.88% at 142.81 passes. That
  treatment showed that the accepted schedule benefits from active random
  residual features at startup. This proposal therefore does **not** zero the
  second projection or use an exact-neutral gate. Both new matrices are
  Kaiming initialized, the fixed multiplier is nonzero, and both layers must
  have nonzero gradients on backward one. The `0.1` multiplier controls the
  perturbation without recreating a branch that must open over multiple steps.
- EXP-027 showed that capacity can become useful only in composition with the
  accepted early image invariance. A pooled head is a cheap way to test a new
  capacity placement on that representation. Conversely, EXP-010/011/012 and
  the later gate ablations warn that extra capacity often overfits or destroys
  useful dense semantics; this is an exploratory mechanism, not an expected
  automatic gain from parameter count.

No external search is needed or allowed for this local-only session. The
proposal is grounded in the accepted source, the system profile, and EXP-012,
EXP-014, EXP-017, and EXP-027 reports.

## Exact Production Change

Add fixed constants near the accepted architecture constants:

```python
POOLED_HEAD_WIDTH = 64
POOLED_HEAD_SCALE = 0.1
POOLED_HEAD_INIT_SEED = 36036
```

In `WideResNet.__init__`, construct and initialize the entire accepted model,
including `self.fc`, exactly as it is now. Only after `self.apply(self._weights_init)`
has completed, register the new branch inside a restoring CPU RNG fork:

```python
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(POOLED_HEAD_INIT_SEED)
    self.pooled_head = nn.Sequential(
        nn.Linear(widths[2], POOLED_HEAD_WIDTH, bias=False),
        nn.ReLU(),
        nn.Linear(POOLED_HEAD_WIDTH, widths[2], bias=False),
    )
    self.pooled_head.apply(self._weights_init)
```

This order is mandatory. It preserves all accepted common tensor values and
the global post-construction CPU RNG state; creating the branch before the
accepted `.apply` would shift the classifier initialization and confound the
test. Seed `36036` is derived from the experiment ID and preregistered before
any score. It selects the unavoidable initialization of new parameters, not a
reroll; there is no alternate head seed.

In `forward`, keep the accepted backbone and pooling operations byte-for-byte,
then apply the exact residual expression before `self.fc`. Use ordinary
out-of-place `F.relu`/`nn.ReLU`, FP32, and no branch diagnostics in production.

The existing optimizer comprehension must discover the new parameters without
special cases. Both bias-free two-dimensional matrices belong exactly once to
the existing `decay_params` group and receive `WEIGHT_DECAY=5e-4`, the same LR,
momentum, and Nesterov update as every other matrix. There are no new
`no_decay_params`, parameter-group LRs, warmups, freezes, or clipping rules.

Everything else remains accepted: seed 42, batch 256, FP32, `(2,2,3)` stages,
RandAugment N1/M5 through the first exhausted epoch at or after 65%,
batch-shared alpha-0.2 mixup before 65%, the 0.2-to-0.002 time cosine,
continuous matrix decay, loader, evaluator, and evaluation cadence.

## Preregistered Semantic Preflight

Use an ignored, evaluator-free verifier; it may not modify tracked production
or consume the single scored run. Fail closed before timing if any item fails:

1. Compare the candidate against an independent `git show 67c8e98:train.py`
   oracle. Assert that `prepare.py` is unchanged and the production diff is
   limited to the three head constants, branch construction, topology logging,
   and the two forward lines needed for refinement.
2. Instantiate accepted and candidate models from cloned initial CPU RNG
   states. Assert all 987,098 common parameters and buffers are byte-identical,
   their names/shapes/dtypes match, and the post-construction global CPU RNG
   states are byte-identical. Independently reconstruct the branch under seed
   36036 and require byte-exact equality for its two matrices.
3. Assert the exact topology `Linear(128,64,bias=False) -> ReLU ->
   Linear(64,128,bias=False)`, the fixed scale `0.1`, exactly 16,384 new
   parameters, and 1,003,482 total parameters. Reject any normalization,
   dropout, bias, extra activation, learnable scale, or alternate head path.
4. On deterministic finite synthetic FP32 inputs, use hooks to prove that the
   branch input equals the accepted flattened post-`BN/ReLU/GAP` vector and
   that the classifier input equals
   `pooled + 0.1 * pooled_head(pooled)`. Require finite logits and an initially
   finite, nonzero branch contribution. Prove candidate logits equal an
   independently evaluated expression using the same candidate modules.
   Exact candidate/accepted logit identity is neither expected nor required.
5. With cloned model/input/target and RNG states, prove the accepted direct
   path `candidate.fc(pooled)` is byte-identical to accepted logits. Prove the
   new forward consumes no CPU or CUDA RNG and therefore leaves accepted
   mixup lambda/permutation and post-forward RNG semantics unchanged.
6. Inspect optimizer groups by parameter identity. Every trainable parameter
   must occur exactly once; both head matrices must be in the matrix-decay
   group at `5e-4`; accepted grouping, LR, momentum, and Nesterov values must
   remain exact.
7. Execute finite early-mixup and hard-label production updates from cloned
   states. On the first backward, require finite nonzero gradients for both
   head matrices and finite gradients for the accepted classifier/backbone.
   After the step, require both new matrices to move and all optimizer states
   to be finite. This explicitly rules out the delayed opening that makes an
   exact-zero endpoint inappropriate here.
8. Prove the alpha-0.2 batch-shared loss, 65% mixup boundary, worker-safe
   RandAugment cutoff, LR samples, fixed seed 42, once-per-epoch evaluation,
   finite-loss guard, time accounting, and 600-second outer timeout semantics
   are unchanged. Guard all evaluator/test-data access during preflight.

A semantic failure closes only this exact implementation and must not be
repaired by changing width, scale, activation, bias, initialization, or seed
inside EXP036.

## Preregistered H20 Timing Gate

After semantic checks, run a disposable, evaluator-free timing process on one
NVIDIA H20. Time balanced/interleaved accepted and candidate **complete
production update bodies** separately for the early-mixup and hard-label
regimes: zeroing, optional mixup/permutation, forward, exact loss, backward,
and SGD step. Use the same resident batch inputs/targets, common weights,
accepted hyperparameters, and cloned pre-arm RNG states. Include the candidate
head and its optimizer work; exclude data loading and evaluation because their
production source is exact.

- At least 20 warmup updates per model/regime.
- Three or more measured windows of at least 50 updates per model/regime.
- CUDA events with synchronization at window boundaries.
- Coefficient of variation at most 5% for every model/regime series.
- Finite outputs, losses, gradients, and optimizer states throughout.
- Candidate peak allocated memory below 2,048 MiB.

Let `a_mix`, `a_hard`, `c_mix`, and `c_hard` be the medians in seconds/update.
Compute the time-weighted exposure retention exactly as:

```text
retention = (0.65 / c_mix + 0.35 / c_hard) \
          / (0.65 / a_mix + 0.35 / a_hard)
projected_passes = 133.00736 * retention
```

Score only if `retention >= 0.9774` **and** `projected_passes >= 130.0`.
This guards the accepted representation's demonstrated normal-exposure regime
and is stricter than a generic architecture feasibility floor. Because model
initialization and forward are RNG-free, timing instability or a stable miss
is not rerun. The unchanged loader does not receive a new timing gate.

## Sole Scored Experiment and Verdict

If and only if all preflight gates pass, remove stale `run.log` and run exactly
once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Use seed 42 and one H20. Do not rerun a valid score, tune the head, inspect the
test set outside the frozen evaluator, or add an evaluation. Audit exit zero,
one finite summary, 300 counted seconds, less than 600 total seconds, no more
than one evaluation per epoch, the single mixup transition, worker-safe
RandAugment transition, exact 1,003,482 parameter count, and at least 130
realized data passes.

- **Improvement:** `best_test_acc >= 94.42%` (the required +0.10 points over
  accepted 94.32%). Accept only this condition.
- **Corroborating mechanism evidence:** preregister `final_test_acc >= 94.32%`
  and `final_test_loss <= 0.2523`. These strengthen a positive interpretation
  but cannot rescue a primary-metric miss or veto a valid primary success.
- **No improvement:** any valid score below 94.42%, even if loss improves.
- **Crash/invalid:** missing summary, nonfinite values, protocol violation,
  realized exposure below 130 passes, or timeout. Diagnose but do not rerun a
  valid completed score.

## Risks, Interpretation, and Closure

- A nonlinear pooled head can memorize class boundaries more easily and may
  worsen the already visible train/test gap. Weight decay, the residual direct
  path, and scale 0.1 limit but do not remove this risk.
- Global average pooling has already discarded spatial layout, so the MLP
  cannot recover missing spatial evidence; it can only remap channel
  co-occurrences. A failure would be consistent with the backbone rather than
  classifier geometry being the remaining limiter.
- Tiny GEMMs can be launch-bound on H20 despite negligible arithmetic. The
  measured complete-body gate, not MAC counting, decides feasibility.
- The fixed branch seed and scale define one optimization trajectory. They are
  chosen prospectively for attribution and are not evidence that nearby values
  would improve.
- The treatment perturbs initial logits intentionally. This avoids the locally
  failed zero-opening geometry but weakens function-preserving attribution; the
  common-state/direct-path checks isolate the perturbation exactly.

A valid >=130-pass score below 94.42 closes the immediate pooled-feature
residual MLP neighborhood: do not follow with width 32/96/128, scales
0.05/0.2/1.0, biases, GELU/SiLU, zero-final initialization, a learnable scale,
another head seed, or head-specific LR/decay without a new independently
supported mechanism. It does not close all classifier changes, normalized
logits, or other representation objectives. A pre-score semantic or throughput
failure closes only this exact `128 -> 64 -> 128`, ReLU, bias-free, scale-0.1,
seed-36036 design. A valid success may be accepted as the new baseline without
post-hoc head modification.
