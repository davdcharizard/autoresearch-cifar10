# Proposal: Reset Only BatchNorm Running Statistics at the 80% Boundary

## Decision and testable hypothesis

Reset every `BatchNorm2d` module's `running_mean`, `running_var`, and
`num_batches_tracked` exactly once, after the accepted 80% strong-phase evaluation and
weak-loader reconstruction but before the first weak batch. Preserve BN affine weights
and biases, model parameters, gradients, ordinary SGD and all momentum buffers, LR,
weight decay, data order and transforms, timer, and evaluator. The intended mechanism is
to discard activation moments accumulated under RandAugment plus CutMix so that later
clean evaluation uses moments estimated only from crop/flip hard-label weak views.

The formal accuracy hypothesis is that weak-only running statistics improve clean
normalization enough for the fixed seed-42 run to raise `best_test_acc` from the current
94.15% frontier to at least 94.25%. The expected signature would be unchanged training
loss, weights, exposure, and switch accuracy, followed by better first-weak and late
evaluation NLL/top-1. This is a clean evaluator-state hypothesis rather than an optimizer
or representation hypothesis.

The proposal has a severe prior weakness that must be treated as load-bearing, not hidden
by the attractive distribution-shift story. Installed PyTorch 2.9.1 uses mini-batch
statistics in training mode, so running statistics cannot affect training logits,
gradients, parameter updates, or optimizer state. With the model's fixed BN momentum
`0.1`, the inherited strong-stat contribution after `k` weak batches is exactly
`0.9**k`: 0.1216 after 20, 0.00970 after 44, 0.00118 after 64, and
`1.43e-18` after the accepted loader's complete 390-batch weak epoch. The first
post-switch evaluation occurs only after that complete epoch. Thus the accepted method
already performs an exponential weak-stat refresh long before any scored look. Unless
finite-precision behavior leaves a measurable evaluator difference after 390 batches,
the intervention is an effective no-op at every eligible metric observation and should
be rejected in preflight rather than consume a scored run.

## Exact `train.py`-only implementation

Add a narrowly typed helper:

```python
@torch.no_grad()
def reset_batch_norm_running_stats(model):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()
            count += 1
    return count
```

Call it exactly once in the existing one-way switch block, after the current switch
evaluation, successful strong-worker shutdown, and weak-loader construction, and before
`randaugment_enabled = False` makes the transition complete and before the next training
iterator is created. Extend the existing provenance line with
`bn_running_stats_reset: 19`. The accepted width-2 ResNet-20 contains 19 `BatchNorm2d`
modules; a different count is an implementation failure.

Using `module.reset_running_stats()` would be semantically equivalent, but explicit buffer
operations make the mutation surface auditable. Do not change `module.momentum`, `eps`,
`track_running_stats`, training/eval mode, BN affine tensors, or the optimizer. In
particular, do not switch to cumulative averaging, add a calibration pass, reset selected
layers, reset repeatedly, or move the reset closer to an evaluation after observing the
erasure result. Each would be a different intervention.

The only intended state delta immediately after the call is:

- 19 running means become exact zero;
- 19 running variances become exact one;
- 19 batch counters become exact zero;
- all named parameters, BN affine gradients, non-BN buffers, optimizer groups and 59
  momentum buffers, and CPU/CUDA RNG states remain bitwise unchanged.

The call is outside counted GPU-step timing and introduces no recurring work. Keep the
accepted 19 evaluation opportunities and at-most-once-per-epoch behavior unchanged; this
candidate cannot justify an extra early weak evaluation because that would alter the
maximum-over-looks metric and manufacture the only window in which the reset is large.

## Why this differs from momentum reset and SWA BN refresh

This is not EXP032's optimizer-momentum reset. An SGD momentum buffer directly determines
the next parameter displacement, so EXP032 changed the weak-tail weight trajectory even
though it left parameters fixed at the boundary; the effect decayed directionally and
produced a valid but worse 93.89% result. BN running buffers are excluded from SGD and are
not used to normalize training-mode activations. Their reset leaves the complete weight
trajectory unchanged and can affect only eval-mode normalization until ordinary EMA
updates overwrite it. EXP032 therefore supplies no positive or negative causal result for
this idea, beyond showing that the weak boundary is already sensitive and that inherited
state is not automatically stale.

This is also not the BN refresh used with EXP018's SWA model. PyTorch's `update_bn`
resets running statistics on a *frozen averaged parameter state*, temporarily sets BN
momentum to `None`, and makes a no-gradient pass so counters implement a cumulative
average over the calibration corpus. That operation is necessary because SWA installs
weights whose activation distribution does not match the online buffers. EXP018 spent
1,624 weak batches recalibrating the installed averaged model; its 93.85% result rejected
that weight average/window, not BN recalibration in isolation. The present idea keeps
online weights moving, retains default exponential momentum 0.1, and performs no separate
calibration. Consequently the final buffers emphasize roughly the most recent 44 weak
batches, just as the accepted control already does, rather than estimate a population
average for fixed weights.

## Exact preflight and authorization gate

Preflight must use ignored experiment artifacts and no CIFAR-10 test evaluation. The
purpose is first to prove scope and safety, then to test whether any intervention survives
until the first legal evaluator look. A demonstrable no-op is a valid negative result.

1. **Static and state-scope audit.** Compile, Ruff, format, pre-commit, and diff checks must
   show that only `train.py` changed and that the helper/call occurs exactly once at the
   registered boundary. Construct the accepted model and optimizer, take one real update
   so all momentum buffers exist, clone state, invoke the candidate helper, and require
   the exact 57-buffer reset above. Require 19 BN modules, 1,073,962 parameters, identical
   model parameter bytes, gradients, optimizer state/groups, non-BN buffers, logits in
   both train and eval mode before the call versus after restoring the original buffers,
   and unchanged CPU/CUDA RNG. Calling the helper twice in the controller is an error,
   not an idempotence feature.

2. **Installed-semantics proof.** Inspect/execute the installed `_BatchNorm.forward`
   behavior and verify that all 19 modules have `momentum == 0.1`,
   `track_running_stats is True`, and use per-batch statistics in training mode. On an
   analytic one-channel example, prove the mean and unbiased-variance recurrences for
   accepted and reset arms and show `num_batches_tracked` does not alter the update factor
   when momentum is non-`None`. Require the accepted/reset difference after `k` identical
   batches to equal `0.9**k` times its post-reset initial difference within FP32-aware
   tolerance.

3. **Immutable weak-corpus replay.** Start from one copied model/optimizer state after the
   registered 200-record strong corpus and replay a newly registered 390-record weak-hard
   corpus corresponding to one complete production loader epoch. The two arms receive
   byte-identical batches and initial parameter/optimizer/RNG state; only the candidate BN
   buffers and counters are reset. Record every BN buffer delta at steps 0, 1, 2, 5, 10,
   20, 44, 64, 100, 200, and 390. Require identical finite training logits, losses,
   gradients, parameter updates, parameters, non-BN buffers, and momentum buffers at each
   checkpoint. If CUDA nondeterminism prevents bitwise equality, a same-process shadow
   recurrence must isolate buffer-only differences, while paired relative parameter and
   update errors must remain below `1e-6`; candidate-only class concentration or any
   nonfinite state vetoes the idea.

4. **First-look evaluator-survival test.** At step 390, freeze the shared weights and
   compare accepted versus reset-buffer eval logits on a preregistered, label-blind set of
   at least 5,000 weak-transformed CIFAR-10 *training* images. Record per-layer maximum
   absolute/relative running-mean and running-variance differences, logit relative L2 and
   maximum absolute difference, top-1 disagreement count, and margin distribution. The
   reset is authorized for production only if all of these hold: at least one BN buffer
   differs in FP32 after 390 updates; relative logit L2 is at least `1e-5`; at least 0.10%
   (five of 5,000) top-1 predictions disagree; training weights/optimizer remain invariant;
   and no candidate eval logit is nonfinite. These are minimal effect-survival gates, not
   evidence of improved test accuracy. Labels must not be used to tune a reset variant.

5. **Cadence and runtime proof.** Simulate the accepted state machine and require the reset
   remains after the boundary evaluation, the first weak evaluator look remains after 390
   weak batches, exactly 19 unique evaluation epochs including terminal are produced, and
   no evaluation is inserted to catch the transient. A bounded H20 timing check need only
   show the one-time reset/rebuild transaction is below 50 ms and complete-run projection
   remains below 540 seconds; recurring step timing must be unchanged within noise.

The effect-survival thresholds are intentionally modest. A candidate that cannot alter
five of 5,000 training-image predictions at the actual first-look cadence has no credible
mechanism to flip the ten CIFAR-10 test images needed for +0.10 point. Given the analytic
`1.43e-18` inherited coefficient, preflight is expected to fail because buffers/logits
will converge exactly or far below these floors. Do not lower the floor, evaluate early,
inspect test labels, change BN momentum, or add cumulative refresh as a rescue.

## Conditional fixed-seed production and verdict

Only if every preflight gate passes, query the moving baseline immediately before the run,
confirm exactly one idle NVIDIA H20, preserve only the reviewed `train.py` diff and the
user-owned `data/`, remove stale completed logs, and execute exactly once at seed 42:

```bash
uv run train.py > run.log 2>&1
```

Supervise without streaming output and kill before 600 seconds. Require exit zero, ten
finite summary fields, 300.0 counted training seconds, total below 600 seconds, exactly
1,073,962 parameters, one 80% transition, eight stopped workers, exactly one 19-module BN
reset, hard weak targets, 45-55% strong CutMix, 19 evaluations on 19 unique epochs with a
terminal look, and no model/data/LR/optimizer/evaluator drift. Compare switch accuracy to
89.73%, first weak accuracy to 93.16%, final NLL to 0.1934, exposure to 26,898 steps, and
best/final accuracy to 94.15%; these explain mechanism but cannot override the metric.

- **Improvement:** all integrity checks pass and `best_test_acc >= moving baseline + 0.10`
  (currently at least 94.25%).
- **No improvement:** the single valid run finishes below that threshold, even if NLL or
  one checkpoint improves.
- **Invalid:** preflight reveals a scope, state, cadence, safety, or implementation fault,
  or production violates a hard condition. Failure of the preregistered survival gate is
  an informative preflight rejection recorded with `NaN`, not permission to score a
  practically null intervention.

No reroll or rescue is allowed. In particular, do not change momentum to cumulative or a
smaller exponential value, reset only deep layers, retain variance, move the reset late,
repeat it before evaluation, add a no-grad calibration pass, or compose it with EMA/SWA.

## Risks and evidence assessment

- **Effective no-op at legal evaluations — very high.** One weak epoch suppresses the
  inherited strong-stat contribution by approximately `7e17`; FP32 recurrence is likely
  to erase the distinction completely.
- **Misleading distribution-shift intuition — high.** Strong augmentation does change
  activations, but baseline EMA already adapts within tens of batches, whereas legal
  evaluation occurs hundreds of batches later.
- **Reset prior can be temporarily worse — medium.** Zero mean/unit variance is not an
  estimate of weak activations. An early look would be corrupted, which is why adding one
  is forbidden; training itself remains unaffected.
- **No optimization or representation upside — high.** Unlike momentum reset, this cannot
  improve the learned weights because train-mode BN ignores running buffers.
- **Calibration mismatch — medium.** Online exponential buffers follow drifting weights
  and emphasize recent batches; this is much less principled than cumulative refresh of a
  frozen SWA model, but it is also exactly what the accepted recipe already does.
- **Single-seed metric noise — medium.** A bare 94.25% would be protocol-valid but only ten
  images and would not establish generality. Seed rerolls are prohibited.
- **Implementation/runtime risk — low.** The mutation is one-time, allocation-free, and
  outside counted updates; scope and exact counts are easy to audit.

Overall, the idea is maximally isolated but has low expected value. Its strongest outcome
may be a rigorous negative result before production: default-momentum BN already forgets
the strong phase before the evaluator can observe the reset. A future BN-stat experiment
would need a genuinely distinct mechanism—such as a reviewed fixed-weight cumulative
calibration policy—not a later or partial retry of this exact boundary reset.

## Sources

- Installed PyTorch 2.9.1 `_BatchNorm.forward`: training mode uses batch statistics;
  fixed non-`None` momentum keeps exponential factor 0.1 independently of the counter.
- Installed PyTorch 2.9.1 `torch.optim.swa_utils.update_bn`: resets buffers, temporarily
  uses cumulative averaging (`momentum=None`), performs a no-grad loader pass, and restores
  module momenta.
- `experiments/010/03-execute.md` and `experiments/010/04-analysis.md`: accepted 390 batches
  per epoch, 80% transition, first weak 93.16%, final/best 94.15%, NLL 0.1934, 26,898
  steps, and 19 evaluations.
- `experiments/018/04-analysis.md`: eight-state SWA plus 1,624 cumulative BN-refresh batches
  produced 93.85%, below its 94.02% online model.
- `experiments/032/04-analysis.md`: full optimizer-momentum reset changed weak-tail updates
  but reached only 93.89%, distinguishing optimizer state from evaluator-only BN buffers.
- `01-definition.md`, `03-experiment-learnings.md`, `04-results.tsv`, current accepted
  `train.py`, and immutable `prepare.py` evaluator protocol, read for EXP037.
