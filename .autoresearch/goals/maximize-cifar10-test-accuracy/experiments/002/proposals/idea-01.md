# Proposal: Late-Phase Exponential Weight Averaging

## Summary

Keep the successful EXP-001 WRN-16-2, data pipeline, optimizer, batch size, and
time-aligned learning-rate schedule unchanged, and add a late-phase exponential
moving average (EMA) of trainable parameters. Begin averaging when 70% of the
300-second counted training budget has elapsed, update the shadow parameters
once every 10 optimizer steps with decay 0.995, and use the EMA parameters for
the existing sparse evaluations after averaging has begun. Before that point,
evaluate the live model as EXP-001 did.

This is an isolated, `train.py`-only generalization intervention. It preserves
seed 42, persistent DataLoader workers, evaluation every fifth epoch plus the
final epoch, and the one-H20 execution contract. It does not change training
examples, gradients, optimizer state, LR, weight decay, architecture, or the
frozen evaluator.

## Limiter Diagnosis

EXP-001 raised `best_test_acc` from 91.54% to 93.38% and completed 28,540 steps
and 147 epochs in 342.5 seconds total. Its last evaluations were tightly
clustered: the model peaked at 93.38% at epoch 145 and finished at 93.34%. This
shows that the WRN and cosine schedule reliably enter a good late basin, but the
reported model remains one noisy SGD iterate. The remaining opportunity is not
obviously more capacity or more training exposure; it is to reduce late-iterate
variance without perturbing the proven optimization trajectory.

The saved distillation of Ajroldi, Orvieto, and Geiping, *When, Where and Why to
Average Weights?*, reports that averaging a carefully selected portion of the
trajectory can mildly improve generalization at low cost, while including early
under-trained iterates can bias the result. EXP-001 provides an unusually clean
case for this intervention because its final accuracy is stable and its LR is
already in the descending part of the cosine schedule.

## Exact Intervention

### Start time and averaging rule

- Define `EMA_START_FRACTION = 0.70`, `EMA_UPDATE_EVERY = 10`, and
  `EMA_DECAY = 0.995`.
- Immediately after the first `optimizer.step()` for which
  `total_training_time / TIME_BUDGET_S >= 0.70`, clone every trainable model
  parameter into a detached FP32 shadow tensor. Initialization by copying the
  live weights avoids startup bias and requires no bias-correction term.
- After initialization, update the shadows after every tenth optimizer step:

```python
with torch.no_grad():
    torch._foreach_lerp_(ema_parameters, model_parameters, 1.0 - EMA_DECAY)
```

  If avoiding a private-looking API is preferred, the equivalent per-parameter
  `ema.lerp_(parameter, 0.005)` loop is acceptable; the foreach operation is
  preferred because it launches far fewer CUDA kernels.
- The sampling-adjusted e-folding window is approximately
  `10 / (1 - 0.995) = 2,000` optimizer steps. At EXP-001 throughput this is
  about 21 counted seconds. Starting at 210 seconds leaves roughly 8,000 live
  steps and about four effective EMA windows, so the initial 70%-time iterate
  contributes little at the end while the average still spans much more than a
  single minibatch-scale fluctuation.

The start point is deliberately time-based rather than epoch- or step-based.
It remains aligned with the fixed budget if the small EMA cost changes
throughput. At 70% progress, the existing cosine schedule has already reduced
the LR substantially from its 0.2 peak, excluding the early trajectory where
weights move between basins. A 0.995 decay tracks the continuing late drift
more closely than a uniform average over all remaining iterates while still
smoothing the noisy endpoint.

### Batch-normalization-safe evaluation

Average trainable parameters, including BN affine scale and bias, but do not
average or overwrite BN buffers (`running_mean`, `running_var`, and
`num_batches_tracked`). Those statistics are nonlinear summaries of activation
distributions and should not be treated as ordinary weights.

At each already-scheduled evaluation after EMA initialization:

1. Under `torch.no_grad()`, clone the current live parameters as a temporary
   backup and copy the EMA shadow values into the existing parameter objects.
2. Call the frozen `evaluator.evaluate(model, device)` exactly once. It puts the
   model in evaluation mode, so the current live BN running statistics are read
   but cannot be updated.
3. In a `try/finally` block, copy the backed-up live values into the same
   parameter objects before training can resume.

This approach keeps the live model's current BN buffers, which were accumulated
over the same late training distribution as the nearby EMA weights. It avoids
an unbudgeted BN recalibration pass and avoids averaging incompatible running
statistics. Restoring values into the original `Parameter` objects preserves
the optimizer's references and momentum buffers. The `finally` restoration is
required so an evaluation exception cannot silently leave EMA values installed
for subsequent SGD updates. The next epoch's existing `model.train()` restores
training mode.

Before 70% progress, use the live model at sparse evaluation points. After 70%,
evaluate only the EMA view, not both live and EMA models, so the constraint of
no more than one validation per epoch remains satisfied. `best_test_acc` is
updated from whichever single model is evaluated at that point.

## Why These Hyperparameters

Averaging from the beginning would mix weights learned at high LR and visibly
lower accuracy into the final model. Conversely, starting only in the final few
seconds would not collect enough independent iterates. The 70% gate captures
the final 90 counted seconds, after representation learning is mature but with
enough time for the shadow state to forget its initialization.

Updating every 10 steps is a compute-aware approximation to per-step EMA. A
decay of 0.995 at that cadence is approximately a per-step decay of 0.9995 and
therefore emphasizes the final low-LR portion without collapsing to the last
iterate. This should be more robust under a monotonically decaying LR than
uniform SWA, whose center could lag behind the converged endpoint. These values
are fixed before the run; there is no result-conditioned window selection.

## Budget and Overhead

WRN-16-2 has roughly 0.69 million parameters, so one FP32 EMA copy uses about
2.8 MB. A temporary live-parameter backup during evaluation uses another
approximately 2.8 MB. This is negligible beside EXP-001's 1,092 MiB peak and
the H20's roughly 98 GB capacity.

The EMA performs one fused interpolation over 0.69 million values every 10
steps. It adds no forward or backward pass and should consume well below 1% of
training work. Its time is naturally charged to the 300-second training timer,
so even unexpected overhead cannot violate the counted budget; the observable
cost would only be a small reduction from EXP-001's 28,540 steps. Sparse
evaluation adds two device-to-device parameter copies but no second evaluator
call. Persistent workers remain enabled, so total runtime should remain close
to EXP-001's 342.5 seconds and safely below the 10-minute limit.

## Hypothesis and Success Criteria

The interpretable hypothesis is that EXP-001's 93.34-93.38% late plateau
contains small SGD iterate variance that is not useful for classification.
Late EMA should preserve the learned representation and current BN calibration
while reducing that variance, improving the best evaluated test accuracy by at
least 0.10 percentage points.

The experiment succeeds if a fresh run on one NVIDIA H20:

- reports `best_test_acc >= 93.48%`, compared with the 93.38% moving baseline;
- completes approximately 300 seconds of counted training and under 600 seconds
  total;
- retains fixed seed 42 and modifies only `train.py`;
- evaluates at most once per epoch and produces the complete summary.

Record final and best accuracy, loss, step and epoch counts, peak VRAM, EMA
initialization time/step, and number of EMA updates. If accuracy stays within
93.28-93.47% while step count remains near EXP-001, the result suggests that
late iterate variance was already too small or that live BN statistics are not
well matched to averaged weights. A material drop paired with an ordinary live
training curve would instead implicate too long an EMA window; a later start or
smaller effective window would be the appropriate follow-up. A large step-count
drop would indicate implementation overhead rather than rejection of the
generalization hypothesis.

## Risks and Controls

- **BN mismatch:** Averaged weights can induce slightly different activations
  from the live model. Retaining the current late-stage BN buffers is the
  lowest-cost coherent choice, but a poor result may warrant a later experiment
  with an explicitly budgeted BN-statistics recalibration. This proposal does
  not add such a pass.
- **Trajectory lag:** EMA can trail a still-improving model. The late 70% start
  and roughly 2,000-step effective window favor the endpoint over early late-
  phase weights.
- **Training corruption during evaluation:** Swapping tensor values is safe only
  if exact live values are restored. Use original `Parameter` objects,
  `torch.no_grad()`, and `try/finally`; never load a replacement module into the
  optimizer.
- **Hidden throughput cost:** Use decimated foreach updates and log the update
  count. Compare total steps with 28,540; the wall-clock schedule and stop guard
  remain authoritative.
- **Metric fishing:** Evaluate only at the existing cadence and select no decay,
  start time, or seed based on intermediate test results.

## Verification

1. Confirm one NVIDIA H20 is visible, seed remains 42, and no file except
   `train.py` is changed for the experiment implementation.
2. Log the EMA initialization step/time and the final number of EMA updates.
3. Run `uv run train.py > run.log 2>&1` and stop it if total wall time exceeds
   10 minutes.
4. Verify evaluations remain every fifth epoch plus the budget-exhausted final
   epoch, with only one evaluator invocation at each point.
5. Confirm BN buffers are excluded from EMA and that live parameters are
   restored after a mid-training EMA evaluation.
6. Require a complete summary with approximately 300 training seconds and
   `best_test_acc >= 93.48%` for an improvement verdict.

## Evidence

- `experiments/001/04-analysis.md`: the proven WRN-16-2 reached 93.38%, finished
  at 93.34%, used sparse evaluation and persistent workers, and completed in
  342.5 seconds total.
- `experiments/002/papers/weight-averaging.md`: checkpoint averaging can mildly
  improve generalization at low implementation and memory cost, but its window
  should exclude early under-trained iterates.
