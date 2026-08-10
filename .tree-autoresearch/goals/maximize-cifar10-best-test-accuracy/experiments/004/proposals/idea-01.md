# Proposal: Sparse Late State EMA on the Validated CutMix WRN

## Summary

Add a sparse exponential moving average (EMA) of the complete inference state
to the accepted EXP-002 recipe. Begin averaging at 75% of charged training
time, exactly when CutMix turns off and drop path starts annealing, and update
once every 32 optimizer steps. After EMA activation, use the EMA model for the
single allowed evaluation at each epoch boundary; do not also evaluate the
online model. Average trainable parameters and floating-point BatchNorm buffers
together, copy integer buffers, and perform no separate BatchNorm recalibration
pass.

This is a deliberately minimal generalization intervention. Architecture,
data order, CutMix RNG streams, optimizer, LR schedule, drop-path schedule,
batch size, BF16 execution, and seed 42 remain identical to EXP-002. All EMA
updates occur inside the existing charged step timer.

## Diagnosis and Feasibility

EXP-002 is the correct base: it improved the WRN parent from 94.62% to 95.23%
with 27,950 steps, 144 epochs, 300 seconds of charged training, 467.1 seconds
total runtime, 1,178.9 MiB peak VRAM, and 2,748,890 parameters. Its best and
final accuracies were 95.23% and 95.19%, only 0.04 points apart. The small gap
means late checkpoint selection noise is not a large observed failure mode,
but it also shows that a stable late trajectory exists from which a modest
weight-space averaging gain is plausible.

EXP-003 found no reliable benefit from changing CutMix probability and drop
path strength. Its selected single-run maxima fell by 0.14-0.29 points on
confirmation, so another narrow regularization scalar search would mostly
revisit measurement variance. EMA is qualitatively different: it reduces
parameter-trajectory variance without another forward/backward pass and does
not change the examples or gradients seen by the online model.

Classical SWA normally samples a constant- or cyclic-LR tail and then
recomputes BatchNorm statistics. That exact recipe is a poor fit here. The
accepted model uses a wall-clock cosine decay, and an additional train-loader
forward pass for BatchNorm recalibration would either consume unbudgeted work
or displace training. Instead, maintain a state EMA during the already clean,
low-LR final quarter. This preserves the accepted schedule and supplies
BatchNorm state without an extra data pass.

The storage cost is negligible on the 98 GB H20: an FP32 copy of 2.75M
parameters is about 10.5 MiB, plus small BatchNorm buffers and module overhead.
The expected peak allocation increase is approximately 11-15 MiB, leaving the
run near 1.2 GiB. At 32-step cadence, the final quarter of an EXP-002-like run
produces about 218 EMA samples. Each sample performs only linear memory
operations over the model state, and no extra network execution.

## Mechanism

### Start time

Set `EMA_START = 0.75`. The training loop already computes
`progress = total_training_time / TIME_BUDGET_S` before each step. The first EMA
sample is taken after `optimizer.step()` on the first cadence-eligible step for
which `progress >= EMA_START`.

This boundary is intentional:

- CutMix is active only while `progress < 0.75`, so every EMA sample comes from
  clean-supervision updates.
- Drop path is constant through 0.75 and then decays to zero, so the averaged
  state tracks the settling phase rather than combining the strongly
  regularized early trajectory with the final predictor.
- At progress 0.75 the cosine LR is approximately 0.034 and decays to 0.002,
  leaving enough movement for averaging to span nearby solutions while keeping
  them in one late basin.

### Cadence and decay

Set `EMA_UPDATE_EVERY = 32` optimizer steps and `EMA_DECAY = 0.99`. Sample only
after the optimizer update. Use a short bias-free ramp for the actual decay on
the `k`-th EMA sample:

```python
if k == 1:
    ema_state.copy_(online_state)
else:
    decay = min(EMA_DECAY, (k - 1) / k)
    ema_state.lerp_(online_state, 1.0 - decay)
```

For samples 1 through 100 this is exactly an arithmetic average of the sparse
trajectory points. Thereafter it becomes a 0.99 EMA, preventing the earliest
tail checkpoints from retaining excessive weight. With about 218 samples, the
first checkpoint's final coefficient is roughly
`0.01 * 0.99**118`, or 0.3%, while the effective late window remains about 100
sparse samples (roughly 3,200 optimizer steps). This is better matched to the
75-second tail than a conventional per-step decay such as 0.9999, which would
retain too much of the initialization when started this late.

Do not average optimizer momentum, gradients, RNG state, CutMix counters, or
the LR/drop-path schedules. EMA is an inference-state estimator only.

### BatchNorm state and evaluation

Maintain one no-gradient `ema_model` with the same architecture on GPU. At each
EMA sample:

1. Apply the same ramped EMA update to all model parameters.
2. Apply the same update to floating-point buffers, specifically BatchNorm
   `running_mean` and `running_var`.
3. Copy non-floating buffers, specifically `num_batches_tracked`, from the
   online model.

This produces a self-contained state for evaluation without a second pass over
training data. Averaging BatchNorm moments is an approximation, because the
moments are not recomputed under the averaged weights, but it is preferable to
using stale online buffers or spending a material part of the fixed budget on
recalibration. The EMA begins in a stable late regime, which limits this
mismatch.

Before EMA activation, evaluate the online model as EXP-002 does. After the
first EMA sample, evaluate only `ema_model`. Log the evaluation source as
`online` or `ema`. The evaluator still runs exactly once at each epoch boundary
and once after a partial final epoch through the existing logic; there is never
an online-plus-EMA double evaluation. `best_acc`, `test_acc`, and `test_loss`
therefore refer to whichever single deployable model was evaluated in that
epoch, and the final evaluation is the EMA model.

## Minimal `train.py` Implementation

1. Add `import copy` and constants:

   ```python
   EMA_START = 0.75
   EMA_UPDATE_EVERY = 32
   EMA_DECAY = 0.99
   ```

2. After constructing the online model and before `t_start_training`, create
   `ema_model = copy.deepcopy(model)`, call `ema_model.requires_grad_(False)`,
   and put it in evaluation mode. Its initial contents are only storage; the
   first eligible sample explicitly copies the complete online state.
3. Build parallel lists of online/EMA parameters, floating buffers, and integer
   buffers once during setup. Assert equal lengths, names, shapes, dtypes, and
   devices so state-order mistakes fail before the timed run.
4. Add `ema_updates = 0` and `ema_start_step = None`. Immediately after
   `optimizer.step()`, if `progress >= EMA_START` and
   `step % EMA_UPDATE_EVERY == 0`, update the shadow state under
   `torch.no_grad()`. Because the current loop increments `step` later, either
   move `step += 1` before this condition or use
   `(step + 1) % EMA_UPDATE_EVERY == 0`; document the convention and use it
   consistently.
5. On the first eligible sample use exact `copy_` for parameters and all
   buffers, record the start step, and set `ema_updates = 1`. On later samples,
   compute `decay = min(0.99, ema_updates / (ema_updates + 1))`, update
   parameters and floating buffers with `lerp_(online, 1 - decay)`, copy
   integer buffers, and increment the counter.
6. Use `torch._foreach_lerp_` for the FP32 parameter lists and, separately, for
   compatible floating-buffer lists to reduce Python and kernel-launch
   overhead. Fall back to a simple no-grad loop only if a deterministic smoke
   test shows foreach incompatibility. Integer buffers use `copy_`.
7. Keep the EMA update before the existing `torch.cuda.synchronize()` and
   `dt = time.time() - t0`. This charges every copy/lerp operation to the
   300-second training budget. Do not create or refresh EMA state between the
   synchronization and time accounting.
8. At evaluation, choose
   `eval_model = ema_model if ema_updates > 0 else model`, call
   `evaluator.evaluate(eval_model, device)` exactly once, and include
   `source=ema` or `source=online` in the existing line. Do not swap weights
   into the online model; keeping two modules avoids accidental corruption of
   optimizer-owned parameters.
9. Add setup logging for the three EMA constants and terminal logging for
   `ema_updates`, `ema_start_step`, and the final evaluation source. Preserve
   every required summary key unchanged.

No other training behavior changes. In particular, keep `CUTMIX_PROB=0.5`,
`MAX_DROP_PATH=0.08`, their shared 0.75 transition, the dedicated seed-42
CutMix generators, SGD/Nesterov settings, BF16 channels-last path, once-per-
epoch evaluation, and global seed 42.

## Expected Charged Overhead

An EXP-002-like run should take approximately 27,000-28,000 steps, leaving
roughly 6,800-7,000 steps and 210-220 EMA samples after 75% progress. One update
reads the approximately 10.5 MiB online parameters, reads and writes the EMA
parameters, and touches small buffers. The aggregate data movement is only
single-digit GiB over the last 75 seconds, negligible relative to H20 memory
bandwidth. Framework launch overhead is the larger concern; foreach operations
should keep the charged cost below 0.5% and reduce exposure by fewer than about
150 steps. There is no additional forward, backward, data-loading, or
evaluation pass.

The total wall-clock runtime should stay near EXP-002's 467 seconds because the
number of evaluator calls is unchanged. The 600-second outer timeout remains
appropriate.

## Evidence

- `experiments/004/papers/stochastic-weight-averaging.md` reports that averaging
  later SGD trajectory points improves generalization across residual models
  on CIFAR and adds almost no computation. It also identifies LR behavior and
  BatchNorm statistics as coupled implementation choices, motivating the
  adapted late-state EMA rather than an unmodified classical SWA recipe.
- `experiments/002/04-analysis.md` establishes the parent at 95.23%, with a
  clean final quarter, only 0.04 points between best and final accuracy, and
  ample memory headroom. The proposal preserves every validated mechanism in
  that run.
- `experiments/003/04-analysis.md` shows that selected scalar-search maxima did
  not confirm and recommends EMA or model averaging as a qualitatively
  different next direction. The proposal makes one preregistered EMA choice,
  avoiding another selection-heavy grid.
- `experiments/004/00-navigate.md` selects EXP-002 specifically because its
  CutMix gain is validated and its failed child rules out only narrow
  probability/drop-path tuning, not orthogonal trajectory averaging.

## Risks and Mitigations

- **BatchNorm mismatch (medium):** averaged running moments are not exactly the
  moments induced by averaged weights. Mitigate by starting only in the stable
  clean tail and averaging the moments with the same coefficients. Do not hide
  this risk with an uncharged recalibration pass.
- **Insufficient trajectory diversity (medium):** cosine LR is already low in
  the final quarter, unlike classical SWA's constant/cyclic tail. Starting at
  0.75 captures LR from about 0.034 to 0.002; starting later would likely leave
  too little diversity. Do not change the validated LR schedule in this first
  EMA test.
- **EMA lag (low-medium):** decay 0.99 can lag a rapidly improving model. The
  arithmetic-average ramp and late start avoid cold-start bias, while the
  effective 100-sample window emphasizes the later half of the tail.
- **Metric visibility (controlled):** evaluating only EMA after activation
  means late online checkpoints are not measured. This is necessary to respect
  one evaluation per epoch and makes the intervention's deployable result the
  metric. Early online evaluations preserve a sanity baseline, and final logs
  identify the source explicitly.
- **Small true headroom (medium-high):** EXP-002's best/final gap is only 0.04
  points, so averaging may produce less than the required 0.10-point gain. The
  intervention is still worth one run because it has low cost and attacks the
  EXP-003-confirmed trajectory variance through a different mechanism.
- **Measurement variance (medium):** EXP-003 showed 0.14-0.29 point selection
  reversals. Use one preregistered configuration and judge its single fixed-seed
  result against 95.33%; do not retry or choose among EMA decays based on test
  accuracy.
- **Implementation ordering (low):** a mismatch between named state lists could
  silently corrupt EMA. Setup assertions and a deterministic helper smoke test
  must verify exact correspondence before the full run.

## Tests and Verification

### Deterministic helper smoke tests

Before the full run, exercise the EMA updater on a tiny module containing a
linear layer and BatchNorm:

1. Verify the first sample exactly copies parameters, running mean/variance,
   and `num_batches_tracked` from online state.
2. Set known scalar online states and verify sample 2 uses decay 0.5, sample 3
   uses 2/3, and sample 101 caps at 0.99.
3. Verify floating states are averaged, integer buffers are copied, EMA tensors
   have `requires_grad=False`, and EMA parameters are absent from the optimizer.
4. Verify an EMA forward in eval mode is finite and does not change its
   BatchNorm buffers.

### Full experiment

1. Confirm physical GPU 0 is the approximately 98 GB NVIDIA H20, then launch
   exactly once with
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
2. Confirm the log reports `EMA_START=0.75`, cadence 32, maximum decay 0.99, an
   EMA start near 75% progress, approximately 210-220 updates, and `source=ema`
   for all subsequent evaluations.
3. Confirm each epoch contains at most one evaluator call; there must be no
   online/EMA paired evaluation and no BatchNorm recalibration pass.
4. Confirm charged training time remains approximately 300 seconds, total time
   is below 600 seconds, throughput remains close to the parent's 27,950 steps,
   and the complete required summary prints.
5. Confirm peak VRAM increases only modestly from the parent's 1,178.9 MiB and
   that no NaN, Inf, traceback, CUDA error, or state-order assertion occurs.
6. Confirm the implementation diff modifies only `train.py`, retains seed 42,
   and leaves `prepare.py` and evaluation untouched.

## Testable Hypothesis

Sparse late state EMA will move the accepted EXP-002 predictor toward a wider,
lower-variance late optimum without materially reducing optimizer exposure.
The preregistered prediction is `best_test_acc >= 95.33%`, at least 0.10 points
above the 95.23% parent, with an expected range of 95.33-95.50%, fewer than 150
lost optimizer steps, and less than 20 MiB additional peak VRAM. A result below
95.33% is a no-improvement even if test loss improves; it should not trigger a
decay/cadence retry selected on the same test metric.

