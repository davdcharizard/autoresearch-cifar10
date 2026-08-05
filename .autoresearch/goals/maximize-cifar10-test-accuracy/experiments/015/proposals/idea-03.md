# Proposal: Ten-Percent Early-Window SAM, Then Accepted SGD

## Recommendation

Conditionally test one short, fixed sharpness-aware minimization (SAM) window:
use non-adaptive SAM with `SAM_RHO = 0.05` while pre-step counted-time
`progress < 0.10`, then use the exact accepted SGD/Nesterov update for the
remaining 90% of the 300-second training budget. Preserve the accepted
WRN-16-2, FP32 arithmetic, batch size 256, time-based LR schedule, weight decay,
alpha-0.2 mixup through 65%, hard-label tail, data augmentation, seed, loader,
and evaluation cadence.

This is a high-upside but medium-low-evidence candidate. It should advance only
if the cheaper per-example-mixup and gradient-geometry candidates do not offer
a cleaner expected-value case. A scored run is justified only if a local
preflight demonstrates at least 94% projected exposure retention and exact
state semantics. Do not extend the SAM window, tune rho, or combine SAM with
another new treatment after seeing the result.

## Diagnosis And Rationale

The accepted run reaches 94.07% after 141.9 data passes and near-zero late
training loss. EXP-003 through EXP-009 show that more target/feature
regularization, altered late decay, altered LR floor, and extra reduced-precision
exposure all regress. EXP-010 and EXP-011 show only small gains from extra
low-resolution capacity, while EXP-013 and EXP-014 show that late averaging and
identity-biased initialization do not clear the 94.17% threshold. The remaining
problem is plausibly the geometry of the learned solution rather than inability
to fit or insufficient raw exposure.

SAM directly changes the update objective: instead of minimizing only the loss
at the current weights, it uses the gradient at a nearby adversarial point in
weight space. The intended effect is to avoid solutions whose loss changes
sharply under small parameter perturbations. Restricting SAM to an early window
is motivated by the saved `Time Matters in Regularizing Deep Networks` note:
early interventions can leave a persistent generalization effect while later
ordinary optimization restores convergence. This proposal does not claim that
the note specifically validates SAM or rho 0.05; those remain the speculative
parts of the treatment.

The 10% boundary is deliberately much earlier than the 65% mixup boundary.
Full-run SAM would roughly halve the number of minibatches and optimizer updates
inside the fixed time budget. A 10%-of-time window confines that cost to about
30 counted seconds, overlaps only the already-regularized mixup phase, and
leaves about 270 seconds of the exact accepted optimizer path, including the
entire hard-label tail. A longer 15-25% window would sacrifice too many
high/medium-LR updates without local evidence that partial SAM needs that much
time.

Fix rho at `0.05`. This is a moderate, conventional non-adaptive SAM radius for
image classification and avoids creating a radius sweep. Do not scale rho by
parameter magnitude, learning rate, layer, time, or gradient norm beyond SAM's
single global normalization. Do not use ASAM, LookSAM, stochastic SAM, or a
rho schedule; each would be a distinct experiment.

## Exact Training Rule

Add only:

```python
SAM_RHO = 0.05
SAM_END_FRACTION = 0.10
SAM_EPS = 1e-12
```

Use the existing pre-step
`progress = min(total_training_time / TIME_BUDGET_S, 1.0)`. A step is a SAM
step iff `progress < SAM_END_FRACTION`. Therefore the first step begins in SAM
mode and the first step whose pre-step counted time is at least 30 seconds uses
ordinary accepted SGD. The final SAM step may cross the boundary because a
training step is atomic; do not split or discard it.

For every SAM step:

1. Transfer one accepted minibatch, set the accepted time-based LR, and sample
   mixup exactly once. Construct `mixed_inputs`, `targets_a`, `targets_b`, and
   scalar `mix` once and reuse those exact tensors for both forward passes.
2. Zero gradients and run the ordinary accepted mixed-input forward and mixed
   cross-entropy at current parameters `w`. Require finite first-pass loss and
   call backward.
3. Compute one global FP32 L2 norm over every non-`None` raw data gradient:
   `g_norm = sqrt(sum_p sum(p.grad**2))`. Weight decay and momentum are not part
   of this perturbation direction because neither has been applied yet. Require
   a finite, strictly positive norm.
4. Under `torch.no_grad()`, compute the scalar
   `scale = SAM_RHO / (g_norm + SAM_EPS)` and perturb every parameter with a
   gradient by `e_p = scale * p.grad`, giving `p <- p + e_p`. Before adding,
   clone each original parameter value and retain both the original and `e_p`.
   The aggregate perturbation norm must be rho to normal floating-point
   tolerance.
5. Preserve the BatchNorm running state produced by the first forward. Snapshot
   every `running_mean`, `running_var`, and `num_batches_tracked` after the first
   forward and before the second. Zero the first gradients, then run a second
   forward and the identical mixed loss at `w + e`; require finite loss and
   backward once.
6. In an exception-safe `finally` restoration path, restore all BatchNorm
   buffers from the post-first-forward snapshots and restore every parameter
   with `p.copy_(saved_original)`. Do not restore by subtracting `e_p`, because
   FP32 add-then-subtract is not guaranteed bitwise reversible. The second
   forward still uses current-batch BN statistics for its gradients, but its
   duplicate running-statistic update must not persist.
7. Call the existing `optimizer.step()` exactly once at restored `w`, using the
   second-pass gradients. Consequently Nesterov momentum advances once, and
   the accepted parameter-group policy plus PyTorch SGD's coupled weight decay
   are applied once to the original, unperturbed parameters. Synchronize and charge
   both forward/backward passes, norm calculation, perturbation, buffer copies,
   restoration, and the one optimizer step to counted training time.

Log the second SAM loss because it is the loss whose gradient drives the
optimizer update. Optionally accumulate the first loss separately for one
end-of-run diagnostic, but do not change the existing progress display or use
either loss to alter control flow beyond finite-value failure. Do not resample
augmentation, mixup coefficient, or permutation between the two passes.

For every step at or after the 10% boundary, execute the accepted code path
exactly: one zero-grad, one forward/loss/backward, and one `optimizer.step()`.
No SAM buffers, perturbation tensors, gradient norms, second pass, or BatchNorm
snapshot work may occur in this path. Emit one transition line with step,
counted seconds, progress, and LR, and print the final number of SAM updates.

## Why BatchNorm Must Be Handled Explicitly

The model contains BatchNorm in every pre-activation block and before the
classifier. A naive SAM implementation performs two training-mode forwards and
therefore advances running means, variances, and counters twice for each
minibatch. That changes evaluation calibration and makes the candidate partly a
BatchNorm-update-frequency experiment.

Disabling running statistics for the second pass by changing module momentum is
not exact: counters can still advance, temporary module configuration can leak,
and special handling can alter the second pass's normalization. Snapshotting
the post-first-forward buffers and restoring them after the second forward is
unambiguous. It preserves exactly one accepted running-stat update per observed
minibatch while allowing both training-mode passes to normalize by their own
batch statistics. The restore must copy into existing buffers rather than
replace registered tensor objects.

The perturbation restore has the same requirement. Never replace an
`nn.Parameter`, never rebuild the optimizer, and never expose perturbed weights
to evaluation. If either the second forward or backward raises, restore weights
and BatchNorm buffers before propagating the exception.

## Exposure And Budget Accounting

Let `r` be the measured median SAM-step time divided by accepted-step time in a
matched H20 preflight. Because SAM is active for a fixed 10% of counted wall
time rather than a fixed number of steps, projected exposure retention is:

```text
retention = 0.90 + 0.10 / r
projected_passes = 141.9 * retention
```

At `r = 2.0`, retention is 95.0% and projected exposure is about 134.8 passes.
At `r = 2.2`, retention is 94.55% and exposure is about 134.2 passes. The early
30 seconds would contain roughly half as many optimizer updates and unique
augmented minibatches as accepted training, followed by 270 seconds at accepted
throughput. This loss is real: the extra SAM pass reuses the same augmented
images and does not count as additional data exposure.

The treatment also maps fewer optimizer updates to the same wall-time LR
trajectory. It therefore spends fewer updates in warmup and near the LR peak.
This cannot be eliminated without changing the time-based schedule and is the
main experimental confound. The hypothesis is specifically that the flatness
signal from early SAM outweighs roughly 5% lower total exposure and the reduced
early update density; a failure rejects that joint tradeoff, not SAM in an
unlimited-compute setting.

## Evaluator-Free Semantic Preflight

Before timing or scoring, replace `prepare.Eval` with a fail-closed dummy before
importing `train.py`, so no evaluation data or accuracy is constructed or
observed. On cloned models and optimizers with fixed synthetic or training-only
minibatches, require all of the following:

1. Accepted and candidate models have identical topology, initialization,
   parameter count 691,674, parameter groups, state-dict keys, dtypes, and
   initial CPU/CUDA RNG states.
2. One instrumented SAM step samples mixup once and leaves CPU/CUDA RNG states
   equal to an ordinary one-forward step after mixup sampling. The model has no
   stochastic forward layers, so the second pass must consume no random values.
3. Before the second forward, the global perturbation is finite and its L2 norm
   is within `1e-5` relative and `1e-7` absolute tolerance of 0.05. Every
   perturbed tensor must equal its saved original plus its recorded `e_p`.
4. Immediately before `optimizer.step()`, every parameter is bitwise equal to
   its pre-perturbation value, all parameter and registered-buffer object IDs
   are unchanged, and the retained gradients are the finite second-pass
   gradients.
5. BatchNorm `running_mean`, `running_var`, and `num_batches_tracked` after a
   complete SAM step are bitwise equal to a control clone stopped after exactly
   the same first forward. This proves that only one persistent BN update
   occurred.
6. Wrap or instrument the optimizer to prove `step()` is called exactly once.
   Verify each momentum buffer is created/advanced once and weight decay is
   evaluated at restored parameters, not at `w + e`.
7. Inject a deliberate second-forward exception and separately a
   second-backward exception. In both cases all parameters and BN buffers must
   be restored before the error escapes.
8. Force `progress >= 0.10` on matched clones and require the candidate ordinary
   branch to produce bitwise-equal loss, gradients, parameters, optimizer state,
   BN buffers, and RNG state to the accepted step. Confirm no SAM temporary is
   allocated and the transition log occurs exactly once in a short boundary
   simulation.
9. Require finite `[256, 10]` logits, finite losses, finite gradients, finite
   parameters, and no OOM throughout.

Any semantic failure aborts the candidate. Do not repair a failing preflight by
excluding difficult layers from the norm, freezing BatchNorm, using live second
pass buffers, or changing rho.

## Matched Throughput Preflight And Abort Gates

Benchmark the full production timed body for accepted ordinary steps and SAM
steps on one H20, using cloned initial model/optimizer states, fixed pinned-host
inputs/targets, matched RNG streams, and a balanced interleaved ordering. Warm
each path for at least 25 complete steps, then collect at least four 50-step
windows per path. Include nonblocking host transfer, LR assignment, mixup
sampling/permutation, zeroing, all forwards/loss guards/backwards, global norm,
perturb/restore, BatchNorm snapshot/restore, optimizer update, and final CUDA
synchronization exactly where production charges it.

Report every window mean, median step times, population CV/mean, ratio `r`,
retention from `0.90 + 0.10 / r`, and projected passes from the accepted 141.9.
Launch the single scored run only if:

- both paths have timing CV/mean no greater than 5%;
- projected retention is at least 94.0% and projected exposure is at least
  133.4 passes;
- `r` is no greater than 2.50;
- the semantic suite passes without qualification;
- all states stay finite FP32 and peak allocation stays below 5 GiB; and
- a short loop crossing 10% shows ordinary-step timing returns to within 5% of
  the accepted median.

The retention and ratio gates are mathematically related but both are retained
for auditability. A ratio above 2.5 indicates implementation overhead beyond
the intended two-pass method and fails before spending a scored run. These
gates may inspect timing and training-only state, never evaluator output.

## Scored-Run Invariants

- Modify only `train.py`; add no package, checkpoint, remote call, download, or
  evaluator modification.
- Use one NVIDIA H20, seed 42, FP32, batch 256, the unchanged model and parameter
  count, existing augmentation, alpha-0.2 mixup through 65%, hard-label tail,
  LR/warmup/floor, SGD/Nesterov momentum, parameter-group weight decay, loader,
  maximum steps, and evaluation cadence.
- Count all SAM work inside `total_training_time`. Evaluations remain excluded
  exactly as accepted and occur at most once per epoch.
- Run exactly once with
  `timeout 600s uv run train.py > run.log 2>&1`; do not reroll or adjust rho or
  the window based on the result.
- Require one SAM transition near 30 counted seconds, no SAM update afterward,
  one mixup transition near 195 counted seconds, a 300-second training summary,
  total wall time below 600 seconds, and no non-finite/error trace.
- Remove `run.log` after the experiment is analyzed.

## Expected Impact And Decision Quality

SAM has a plausible route to a greater-than-0.10-point improvement because it
targets a qualitatively different failure mode from capacity, mixup strength,
dropout, schedule endpoints, precision, averaging, and initialization. Unlike
the recent near-miss architecture changes, it explicitly changes local loss
geometry while leaving 90% of counted time and the entire clean-label tail to
the accepted optimizer.

The case is nevertheless weaker than its conceptual upside suggests. The
repository contains no local SAM result or saved SAM paper, rho 0.05 is not
calibrated to this exact network, and early SAM compounds the existing mixup
regularization. EXP-005 and EXP-006 warn that additional regularization can
harm this already well-regularized model. More importantly, SAM buys its signal
by discarding roughly half of the unique minibatches and optimizer updates in
the critical first 30 seconds. The time-based LR schedule makes that exposure
loss particularly expensive around warmup and peak LR.

Thus the upside justifies one budget only as a deliberate moonshot after many
cheap local neighborhoods have been exhausted, and only behind the strict
preflight gates. It does not justify a sweep, a longer window, or a rescue run.
If a lower-cost candidate has comparable mechanistic evidence, prefer that
candidate first.

## Falsifiable Hypothesis And Interpretation

One fixed-seed H20 run using rho-0.05 non-adaptive SAM only while pre-step
counted progress is below 10%, followed by exact accepted SGD, will retain at
least 133.4 projected passes in preflight, complete within 300 counted and 600
wall seconds, and raise `best_test_acc` from 94.07% to at least 94.17%.

A valid score below 94.17% is a no-improvement and closes this exact
10%-window/rho-0.05 policy. Do not infer that full-run SAM fails; the result
tests the fixed-budget joint tradeoff between early sharpness-aware gradients
and fewer high-LR updates. If realized exposure falls below 133.4 passes despite
a passed stable preflight, retain the formal accuracy verdict but classify the
mechanism as operationally confounded. A crash, restoration failure, duplicate
BN-state update, post-boundary SAM step, missing final summary, extra evaluation,
or wall timeout invalidates the run rather than motivating an in-place rescue.

## Evidence

- `01-definition.md`: fixed 300-second counted budget, single H20, fixed seed,
  `train.py`-only scope, one evaluation per epoch, and 94.17% acceptance target.
- `03-experiment-learnings.md`: accepted mild early mixup helps, while stronger
  target/feature regularization and several late/initialization treatments fail.
- `04-results.tsv`: accepted exposure is 141.9 passes; extra exposure alone did
  not help, while lower-exposure capacity runs remained close but sub-threshold.
- `experiments/002/04-analysis.md`: the 65% temporal mixup boundary and long
  ordinary hard-label tail are locally validated and must remain unchanged.
- `knowledge/papers/time-matters-regularization.md`: regularization effects can
  concentrate early and persist after returning to ordinary optimization; this
  motivates the short window but does not specifically validate SAM.
- `train.py`: current time-based scheduling, BN-heavy pre-activation WRN,
  optimizer grouping, mixup implementation, and counted-time boundaries define
  the exact integration and its risks.
