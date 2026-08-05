# Proposal: Short-Horizon Post-Mixup EMA

## Summary

Keep the complete EXP-002 training path unchanged and add a short-horizon
exponential moving average of trainable parameters during the settled portion of
the hard-label tail. Initialize the shadow parameters at 75% of counted training
time, update them every 10 optimizer steps with decay 0.99, and use the EMA view
only for the budget-exhausted final evaluation. Existing non-final evaluations
continue to use the live model. This retains EXP-002's early alpha-0.2 mixup,
65% switch to hard labels, WRN-16-2, optimizer, time-based cosine schedule,
augmentation, loader, seed, and evaluation cadence.

The intervention is intentionally narrower and more endpoint-biased than the
70%-start, 0.995-decay EMA considered before EXP-002. EXP-002's accuracy was
still best at the final evaluation, so a long average can lag useful late
progress. The proposed 0.99 decay at a 10-step cadence has an effective horizon
of roughly 1,000 optimizer steps, half that earlier design, while the final-only
EMA evaluation preserves the live model's sparse test history through the last
regular evaluation.

## Limiter Diagnosis

EXP-002 reaches 94.07% after 27,735 steps and 141.9 dataset passes. Its final
accuracy equals its best accuracy, and the hard-label tail drives training loss
near zero. Additional fitting capacity is therefore not the obvious need. The
remaining opportunity is a small generalization refinement: suppress
minibatch-order and SGD-iterate noise around the endpoint without weakening the
validated early mixup or consuming meaningful image exposure.

The local note for *When, Where and Why to Average Weights?* reports that a
carefully selected averaging window can mildly improve generalization at low
cost, but that early, under-trained iterates should be excluded. The local
*Time Matters in Regularizing Deep Networks* note supports keeping mixup in its
validated early critical period and removing it for late clean-label
convergence. Starting EMA at 75% respects both findings: it begins 30 counted
seconds after mixup is disabled and averages only the mature hard-label
trajectory.

This is a modest-headroom proposal, not a claim that EMA fixes optimization.
The cosine schedule already reaches a low terminal learning rate and EXP-002
ends at its best, so the expected gain is only about 0.10-0.30 percentage
points. The design specifically limits the two main downside mechanisms from
the previous EMA review: trajectory lag and loss of a late live-model peak.

## Exact Intervention

### Fixed constants

Add the following constants to `train.py`:

```python
EMA_START_FRACTION = 0.75
EMA_UPDATE_EVERY = 10
EMA_DECAY = 0.99
```

Do not change `MIXUP_ALPHA = 0.2`, `MIXUP_END_FRACTION = 0.65`, any model or
optimizer setting, the LR schedule, batch size, seed, transforms,
`EVAL_EVERY = 5`, or the stopping rules.

### Initialization and updates

Build a stable list of all trainable `Parameter` objects once after model
creation. After each `optimizer.step()`, while still inside the timed training
step and before the existing CUDA synchronization:

1. If EMA is not initialized and the step's precomputed time progress is at
   least 0.75, clone each live parameter into a detached FP32 shadow tensor.
   Log the counted time, epoch, and global step exactly once. Copy
   initialization avoids bias correction.
2. Once initialized, interpolate every tenth subsequent optimizer step:

```python
with torch.no_grad():
    torch._foreach_lerp_(ema_parameters, model_parameters, 1.0 - EMA_DECAY)
```

3. Increment and log an `ema_updates` counter. Do not update shadows during
   evaluation and do not derive any EMA choice from test results.

Using the progress measured before the current batch makes the gate deterministic
under the existing timer. Placing initialization and interpolation before
`torch.cuda.synchronize()` charges their GPU work to the 300-second training
budget. If the foreach primitive is unavailable in the installed PyTorch,
perform the identical `shadow.lerp_(parameter, 0.01)` operation in a
`torch.no_grad()` loop; do not change the constants.

At EXP-002 throughput, the final 25% contains about 6,900 optimizer steps and
roughly 690 EMA updates. Decay 0.99 per sampled update is approximately decay
0.999 per optimizer step, with an e-folding horizon of
`10 / (1 - 0.99) = 1,000` steps, or about 11 counted seconds. The shadow's
initial 75%-time value should contribute about `0.99**690`, below 0.1%, at the
endpoint. Thus the final EMA is local to the low-LR endpoint rather than a
uniform average over the entire hard-label tail.

### Batch-normalization buffers

Average every trainable parameter, including batch-normalization affine scale
and bias, but exclude all buffers: `running_mean`, `running_var`, and
`num_batches_tracked`. These buffers are nonlinear running statistics and must
not be interpolated as weights.

For EMA evaluation, retain the live model's latest BN buffers. They cover the
same post-mixup hard-label data distribution as the short-horizon EMA weights,
and the approximately 1,000-step parameter window limits their mismatch.
Do not run an extra BN recalibration pass: it would add a new data pass outside
the training intervention, complicate attribution, and add runtime. If this
experiment fails with otherwise normal throughput and convergence, BN mismatch
remains a plausible failure mechanism for a separate experiment.

### Evaluation semantics

Preserve exactly one evaluator call whenever the existing condition
`epoch % EVAL_EVERY == 0 or budget_exhausted` is true:

- At ordinary every-fifth-epoch evaluations, evaluate the live model exactly as
  EXP-002 does, even after EMA has initialized.
- At the single budget-exhausted final evaluation, evaluate the EMA parameter
  view if initialized; otherwise fall back to the live model.
- Never evaluate live and EMA models in the same epoch.

For the final EMA evaluation, under `torch.no_grad()` clone the live parameters
to temporary backup tensors and copy the EMA shadows into the existing
`Parameter` objects. Call `evaluator.evaluate(model, device)` once inside a
`try` block, then restore every live value in `finally`. Copy tensor values;
do not replace parameter objects or load a second module, because the optimizer
must retain its original references and momentum state.

`best_acc` continues to be the maximum of the one model evaluated at each
scheduled point. Consequently it contains the baseline-compatible live history
through the last non-final evaluation and one EMA candidate at the endpoint.
This avoids spending all late evaluations on a lagging average while still
giving EMA an uncontaminated final measurement. `final_test_acc` and
`final_test_loss` refer to the EMA final evaluation. Log `eval_model=live` or
`eval_model=ema` on every evaluation and include `ema_start_step`,
`ema_updates`, and `final_eval_model` in the final summary.

## Expected Impact and Cost

The primary hypothesis is that the final live weights contain small residual
iterate noise around a solution whose generalization was established by early
mixup. A short endpoint EMA should reduce that noise while tracking the useful
late hard-label drift closely enough to reach `best_test_acc >= 94.17%`, the
required +0.10-point improvement over 94.07%. A result in the 94.17-94.37%
range is plausible; a larger gain would be welcome but is not assumed.

The 691,674-parameter model needs about 2.8 MiB for FP32 shadows and another
2.8 MiB for the temporary final-evaluation backup. This is negligible against
EXP-002's 1,094 MiB peak. Roughly 690 fused interpolations over 0.69 million
values add no forward or backward pass and should reduce the 27,735-step count
by less than 1%. The work is explicitly included in the counted timer, while
the final swap and restore occur during excluded evaluation time. Total runtime
should remain close to EXP-002's 341.2 seconds and well below 600 seconds.

## Evidence

- `experiments/002/04-analysis.md`: early alpha-0.2 mixup plus a 35% hard-label
  tail improved 93.38% to 94.07%; final accuracy equaled best accuracy, so the
  accepted baseline still benefits from late clean-label progress.
- `04-results.tsv`: 94.07% at commit `eb08811` is the moving baseline, making
  94.17% the pre-registered improvement threshold.
- `knowledge/papers/weight-averaging.md`: carefully windowed averaging can
  mildly improve generalization at minimal memory and implementation cost, but
  should exclude early under-trained iterates.
- `knowledge/papers/time-matters-regularization.md`: early regularization can
  retain its benefit after removal, supporting EMA only after the validated
  mixup-to-hard-label transition.
- `experiments/002/01-idea-review.md`: the prior 0.995/10-step EMA had a genuine
  lag and EMA-only-evaluation risk. The present 0.99/10-step horizon is half as
  long and only replaces the final evaluation.

## Risks and Controls

- **Small statistical headroom:** cosine annealing and the stable final score
  may already suppress most iterate noise. Pre-register 94.17% and do not tune
  start, decay, cadence, or seed from intermediate test accuracy.
- **Trajectory lag:** EXP-002 improves through its final evaluation. The
  approximately 1,000-step horizon, 75% start, and live non-final evaluations
  make this risk materially smaller than the earlier 2,000-step design.
- **BN mismatch:** EMA affine parameters use live BN running buffers. The short
  window and shared hard-label distribution limit mismatch; no unbudgeted
  recalibration is allowed in this experiment.
- **Parameter-swap corruption:** use in-place copies under `torch.no_grad()` and
  mandatory `try/finally` restoration. Add a smoke assertion that live
  parameters exactly equal their backups after an EMA evaluation.
- **Uncharged or excessive overhead:** place EMA updates before the existing
  synchronization, record the update count and final step count, and require at
  least 95% of EXP-002's realized exposure: 26,348 steps, equivalent to about
  134.9 dataset passes.
- **Metric ambiguity:** label every evaluation as live or EMA. The final metric
  is valid because only one model is evaluated per epoch; do not make a second
  final live-model evaluator call.

## Verification

1. Confirm the experiment diff changes only `train.py`; architecture, mixup,
   optimizer, schedule, data pipeline, seed 42, budget, and evaluator remain
   unchanged.
2. Run a short CUDA smoke test that crosses the EMA gate, performs at least two
   EMA updates, verifies shadows are detached and finite, verifies BN buffers
   are absent from the shadow list, and verifies final swap/restoration leaves
   all live parameters bitwise equal to their backups.
3. Confirm EMA initializes exactly once at 75% counted progress (allowing one
   batch of timing granularity), begins only after the 65% mixup switch, and
   reports the pre-registered decay and update cadence.
4. Run the required single-H20 command with a 600-second kill guard and output
   redirected to `run.log`.
5. Verify evaluations occur every fifth epoch plus the final partial epoch,
   never more than once per epoch. Verify all non-final evaluation records say
   `live` and exactly one budget-exhausted record says `ema`.
6. Require a complete summary with approximately 300 counted training seconds,
   total runtime below 600 seconds, finite loss, at least 26,348 optimizer
   steps, and the EMA metadata fields.
7. Classify as an improvement only if `best_test_acc >= 94.17%`. Interpret a
   normal-throughput 93.97-94.16% result as insufficient EMA headroom or mild
   lag, a larger regression with a normal live curve as possible EMA/BN
   mismatch, and a step-count shortfall as an implementation-cost failure.

