# Proposal: Time-Constant Late EMA on the Full CutMix + SAM Parent

## Summary

Add a no-gradient exponential moving average of EXP-004's complete inference
state while preserving its online WRN, full CutMix dose, independent-image
stream, drop path, and period-two clean-tail SAM unchanged. Start EMA at charged
progress 0.75, after each completed optimizer update and only after any SAM
perturbation has been restored. Update every 32 optimizer steps with a decay
derived from elapsed charged time rather than a fixed per-update scalar.

Define the EMA e-folding time as 18.75 seconds: one quarter of the 75-second
clean tail. Four time constants fit between progress 0.75 and 1.0, so the first
EMA sample has only `exp(-4) = 1.8%` residual weight at the nominal end. This is
a single preregistered operating point, not a decay/cadence search.

After EMA activation, evaluate only the EMA model at each natural epoch
boundary. Before activation, evaluate the online model. This retains exactly
one evaluator call per epoch and makes the final reported model the averaged
one without sacrificing any CutMix or SAM training work.

## Why This Is the Right Bottleneck

The current limiter is detectable generalization gain, not memory. EXP-004
reaches 95.40% with 2,748,890 parameters, 25,560 updates, and only 1,190.5 MiB
on a 97,871 MiB H20. Its two children preserved throughput but failed: DLB lost
0.12 points after halving new-image introduction, while substituting manifold
mixup for one quarter of CutMix gained only 0.01 and worsened loss.

Late checkpoints still move materially. EXP-006's final four evaluations span
0.15 points even though their losses span only 0.0011, and earlier selected
runs reversed by 0.14-0.29 points on confirmation. EMA directly targets this
late-iterate variation while retaining every validated data and optimizer dose.
Unlike an ensemble or extra regularization pass, it costs only sparse linear
updates to a shadow state.

The SWA reference reports CIFAR residual-network gains from averaging later SGD
trajectory points and connects them to wider optima. It also warns that LR and
BatchNorm handling matter. The EMA scaling reference shows that decay and
update frequency must be specified as one effective horizon. A charged-time
constant is therefore better than importing a familiar `0.999` value into a
wall-clock loop whose realized step count changes under SAM.

## Preregistered EMA Dynamics

Add constants:

```python
EMA_START = 0.75
EMA_UPDATE_EVERY = 32
EMA_TAIL_TIME_CONSTANTS = 4.0
EMA_TAU_S = (1.0 - EMA_START) * TIME_BUDGET_S / EMA_TAIL_TIME_CONSTANTS
```

With `TIME_BUDGET_S=300`, `EMA_TAU_S=18.75`. EXP-004 completed 4,898 tail
steps, so cadence 32 should produce about 153 samples. At the observed tail
rate, samples are roughly 0.49 seconds apart, only 2.6% of the time constant;
this is fine enough to approximate continuous EMA while avoiding per-step
parameter traffic.

On the first eligible cadence point, copy the online parameters and all buffers
exactly into the shadow and record `ema_last_training_time`. For subsequent
samples, derive decay from the actual charged-time interval:

```python
delta_s = total_training_time - ema_last_training_time
decay = math.exp(-delta_s / EMA_TAU_S)
ema.lerp_(online, 1.0 - decay)
ema_last_training_time = total_training_time
```

The product of decays over an interval is `exp(-sum(delta_s) / tau)`. Thus the
effective horizon is stable if EMA overhead removes steps, if SAM steps are
twice as slow, or if cadence later changes for engineering reasons. Cadence
controls overhead and discretization only; it does not redefine the averaging
horizon.

Use `progress` and `next_step` already computed before the forward. A step is
eligible when `progress >= 0.75` and `next_step % 32 == 0`. Perform the EMA
sample after `optimizer.step()` and after SAM restoration. Keep it before the
existing CUDA synchronization and `dt` calculation so all shadow-update work
is charged. The timestamp may omit the current step's few milliseconds, but
this bounded one-step offset is negligible relative to 18.75 seconds and is
consistent at every sample.

Do not average gradients, Nesterov momentum, SAM snapshots, RNG states, CutMix
counters, or schedules. The shadow is an inference-state estimator only.

## Minimal Implementation

1. Add `import copy`. After constructing the parent model, create
   `ema_model = copy.deepcopy(model)`, call `requires_grad_(False)` and
   `eval()`, and keep it on the same GPU/channels-last layout. Deep copy consumes
   no model RNG draws. Construct the optimizer, SAM parameter list, and SAM
   snapshots from the online model only.
2. Build name-keyed parallel lists for online/EMA parameters, floating buffers,
   and integral buffers. Assert identical names, shapes, dtypes, devices, and
   memory formats before charged training starts.
3. At the first eligible sample, use `torch._foreach_copy_` for parameters and
   floating buffers and `copy_` for integral buffers. Set `ema_updates=1`,
   `ema_first_step`, `ema_first_progress`, and `ema_last_training_time`.
4. At later samples, use `torch._foreach_lerp_` separately for parameters and
   compatible floating buffers. Copy integral buffers. Update counters and the
   cumulative start coefficient by multiplying it by the realized decay.
5. Do not swap EMA weights into the online model. Keeping two modules prevents
   accidental optimizer/SAM ownership of shadow parameters and removes restore
   complexity.

The online training path must otherwise remain byte-for-byte equivalent to
EXP-004: same model, optimizer, BF16/channels-last path, LR, CutMix generators
and decisions, drop masks, SAM schedule, RNG replay, BN suppression, and one
online optimizer update per batch.

## BatchNorm State

Average `running_mean` and `running_var` with the same elapsed-time coefficient
as parameters. Copy `num_batches_tracked` from the online model at every EMA
sample. BatchNorm affine weights are ordinary parameters and are averaged with
the rest.

This is an approximation to recomputing statistics under the EMA weights, but
it keeps weights and inference moments on the same smoothed trajectory. Copying
only the latest online moments would pair an approximately 18.75-second-lagged
model with statistics dominated by the latest few batches. Leaving the
shadow's initial moments would be invalid.

Do not perform a BatchNorm recalibration pass. A full extra traversal would be
material work outside the validated parent dose; charging it would displace
training, while excluding it would violate the budget's intent. The state-EMA
design is self-contained and requires no extra images or forward pass.

During SAM, the existing second pass continues to disable online BatchNorm
tracking. EMA sampling occurs only after the sole optimizer update and observes
the once-updated online buffers. The EMA model always stays in eval mode and
never updates its own buffers through inference.

## Evaluation Under the Once-Per-Epoch Rule

At each existing evaluation site choose exactly one model:

```python
eval_model = ema_model if ema_updates > 0 else model
eval_source = "ema" if ema_updates > 0 else "online"
test_loss, test_acc = evaluator.evaluate(eval_model, device)
```

Log `source={eval_source}` on the existing evaluation line. Never evaluate
both models in one epoch. Before 0.75, online evaluations provide normal
training sanity checks; after activation, every metric measures the deployable
EMA state. `best_acc` may span early online and late EMA checkpoints, but the
final accuracy/loss necessarily come from EMA. This is one model evaluation
per natural epoch, not an ensemble or evaluator change.

## Cost and Budget Compatibility

An FP32 copy of 2.75M parameters is about 10.5 MiB. Including BatchNorm buffers
and module metadata, expect 11-15 MiB additional allocation, taking peak VRAM
from 1,190.5 MiB to roughly 1.20 GiB. This is immaterial on the H20.

Approximately 153 sparse samples each read the online state and read/write the
shadow, totaling only a few GiB of device traffic across 75 seconds. Foreach
operations avoid one launch per tensor. Expected charged overhead is below
0.5%, with fewer than 100 lost optimizer steps and at least 25,450 total steps.
There is no extra model forward, backward, data batch, SAM pulse, or evaluator
call, so total runtime should remain near EXP-004's 457.3 seconds and below the
600-second outer limit.

## Instrumentation

Add the EMA constants to the setup config and print a final audit line with:

- update count, first and last one-based step, and first/last progress;
- first and last charged timestamps and total EMA elapsed seconds;
- mean/min/max sample interval and mean/min/max realized decay;
- cumulative coefficient remaining on the first sample;
- final online-to-EMA FP32 parameter L2 distance and relative L2 distance;
- final evaluation source.

Accumulate intervals, decay extrema, and decay sum as CPU scalars already
available in the update path. Do not call `.item()` on model tensors during
charged training. Compute parameter distance once after training under
`torch.no_grad()`; it is diagnostic, not part of optimization or selection.
Expected invariants are about 153 updates, first progress near 0.75, about 75
seconds total span, and a first-sample coefficient near 0.018.

## Expected Effect and Hypothesis

The parent is 95.40%, and formal success begins at 95.50%. The stronger,
mechanism-level hypothesis is:

> A charged-time-scaled EMA over the full clean SAM tail will reach
> `best_test_acc` of 95.70-95.95% (+0.30 to +0.55 points), preserve the parent's
> complete CutMix/SAM dose, retain at least 25,450 updates, and add less than
> 20 MiB peak VRAM.

The expected effect exceeds measured tail oscillation because the proposed
benefit is not merely checkpoint selection: averaging should place the
predictor nearer the center of the late flat basin identified by SAM/SWA. This
is still uncertain. SAM already reduces sharpness, the cosine tail may not
contain enough diverse solutions, and the observed 0.15-point oscillation alone
does not guarantee a 0.30-point gain.

One fixed-seed run decides the proposal. Do not search EMA start, cadence, or
time constant against test accuracy. Below 95.50% is no-improvement; 95.50-
95.69% passes the formal tree gate but falsifies the preregistered >=0.30-point
effect hypothesis.

## Risks

- **SAM/EMA redundancy:** SAM already found a flatter final solution, so EMA may
  average nearly identical weights or blur a good endpoint.
- **Decaying-LR diversity:** classical SWA often uses constant/cyclic LR; the
  parent's 0.034-to-0.002 tail may not explore enough of a basin.
- **BatchNorm approximation:** averaged buffers need not equal statistics under
  averaged weights. A regression with large EMA/online distance would implicate
  this mismatch, but no post-hoc recalibration is allowed in the metric run.
- **Lag:** an 18.75-second horizon can underweight very late improvement. Four
  tail time constants bound the start-state residue without collapsing EMA into
  the final iterate.
- **Evaluation visibility:** late online checkpoints are deliberately not
  measured. Dual evaluation would violate cadence and introduce model selection.
- **Small true signal:** prior tail ranges and run reversals are 0.14-0.29
  points. A nominal gain near the threshold may be indistinguishable from
  protocol variance even when formally accepted.
- **State-order corruption:** positional list mismatches could silently average
  wrong tensors. Name/shape/dtype assertions and deterministic tests are
  mandatory.

## Verification

1. On a tiny Linear+BatchNorm model, verify first-sample exact copy; elapsed-
   time decay values; foreach parameter/float-buffer lerp; integer-buffer copy;
   no EMA gradients; and optimizer exclusion.
2. Simulate two update cadences over the same elapsed time and verify the
   product of decays equals `exp(-elapsed / 18.75)` within floating tolerance.
3. On full WRN BF16/channels-last, verify EMA samples only post-restore,
   post-optimizer weights on SAM steps, while CUDA replay, one BN update,
   perturbation norm 0.05, exact restore, and one momentum update remain intact.
4. Verify one evaluator call per epoch and source switches once from online to
   EMA after the first sample; no BN recalibration or second evaluation occurs.
5. Confirm physical GPU 0 is the 97,871 MiB H20, then run once with
   `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
6. Confirm approximately 300 charged seconds, under 600 total seconds, complete
   CutMix ratio near 0.5, SAM ratio near 0.5 from progress 0.75, at least 25,450
   steps, finite EMA diagnostics, complete summary, and `best_test_acc` against
   both the 95.50 formal gate and 95.70 effect hypothesis.
7. Verify only `train.py` changed, seed 42 and evaluator stayed fixed, no
   dependency was added, and transient logs are removed after analysis.

