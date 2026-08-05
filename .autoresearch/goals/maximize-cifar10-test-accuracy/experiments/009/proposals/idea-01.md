# Proposal: Isolated BF16 Autocast at the Accepted Batch Size

## Summary

Train the accepted WRN-16-2 recipe under CUDA BF16 autocast while retaining
FP32 parameters, gradients, SGD state, input interpolation, and test-time
evaluation. Keep batch size 256 and every accepted optimization and
regularization constant unchanged. The treatment is therefore the BF16
training compute path and the additional optimizer/data exposure it buys
inside the fixed 300-second budget, rather than a simultaneous large-batch
optimization experiment.

This is the scientifically strongest form of the seed. A local evaluator-free
H20 feasibility benchmark measured approximately 14.7% higher image throughput
for BF16 at batch 256 than for FP32 at batch 256. Increasing BF16 batch size to
512 added only about 1.5% image throughput beyond BF16 batch 256 while halving
the number of optimizer updates per image and changing BatchNorm statistics,
SGD noise, mixup grouping, and the plausible LR policy. The negligible
occupancy gain does not justify those confounds. EXP-009 should score exactly
one fixed-seed configuration: BF16, batch 256, peak LR 0.2.

## Limiter Diagnosis

The accepted EXP-002 path reaches 94.07% after 27,735 batch-256 updates, or
141.9 dataset-equivalent passes, in 300 counted seconds. It uses only about
1.1 GiB of the H20's 97.9 GiB and processes roughly 23.7k training images per
counted second. Memory capacity is plainly not limiting. Uniform WRN-16-3
widening is compute-limited: its measured throughput was only 56.8% of width 2
and its calibrated exposure projection was about 80.6 passes. Larger FP32
batches recovered less than 6% throughput for that wider model.

Accuracy still improved through EXP-002's hard-label tail. Conversely, six
follow-ups that changed regularization or suppressed late optimization all
regressed at normal exposure: CutMix, shorter or stronger mixup, block dropout,
late decay removal, and cosine-to-zero. The evidence now fixes alpha 0.2,
the 65% mixup cutoff, continuous selective `5e-4` decay, and the 0.002 LR floor.
The remaining plausible limiter is useful optimization and data exposure per
counted second, not VRAM or an insufficiently strong regularizer.

The H20 has native BF16 tensor cores, and the repository's PyTorch 2.9.1 CUDA
build can invoke them through `torch.autocast`. BF16 retains FP32's exponent
range, so it does not need FP16 dynamic loss scaling. Autocast leaves model
parameters and optimizer state in FP32, executes eligible convolutions in
BF16, and promotes cross entropy to FP32. A local semantic smoke check observed
FP32 parameters, FP32 gradients, BF16 logits, and FP32 loss. This is a narrow
way to buy more steps without changing model capacity or the accepted training
recipe.

## Proposal-Development Feasibility Evidence

The following was an **unscored feasibility benchmark**, not an accuracy run.
It did not construct `Eval`, load or inspect the test set, write `run.log`, or
produce a metric. In a separate process on one idle NVIDIA H20, it used the
production WRN-16-2, selective Nesterov SGD, alpha-0.2 mixup path, synthetic
CUDA images and labels, 30 warmup steps, and five synchronized 100-step timing
windows. BF16 and FP32 used FP32 master weights and the same batch-256 graph
apart from autocast. Repeated endpoint measurements controlled for timing
drift.

| Configuration | Median step | Images/s | Synthetic 300s passes |
|---|---:|---:|---:|
| FP32, batch 256 | 9.98-9.99 ms | 25.62-25.66k | 153.7-154.0 |
| BF16, batch 256 | 8.70-8.71 ms | 29.40-29.43k | 176.4-176.6 |
| BF16, batch 512 | 17.14 ms | 29.88k | 179.3 |

The repeated batch-256 ratio is about 1.147x. Calibrating that ratio to
EXP-002's authoritative 141.9 realized passes predicts approximately 162.8
passes and 31.8k optimizer steps for BF16, about 20.9 extra passes and 4.1k
extra updates. `MAX_STEPS = 64000` remains far away. The synthetic absolute
projection is optimistic because it excludes the loader, so only the matched
ratio and the calibrated projection should guide feasibility; realized steps
in the scored run remain authoritative.

Batch 512 increases BF16 image throughput by only about 1.5% while doubling
step time. That means it would complete roughly half as many optimizer updates
for almost the same number of images. It would also use a single beta sample
for twice as many examples, alter every `randperm`, change BatchNorm's batch
statistics and gradient noise, and force an arbitrary choice among unchanged,
linear-scaled, or square-root-scaled LR. It is rejected before scoring. Do not
use the timing result to combine BF16 with batch or LR changes in EXP-009.

## Falsifiable Hypothesis

BF16 tensor-core execution at batch 256 will preserve the accepted model and
regularization behavior closely enough while increasing realized exposure by
at least 10%. Because the accepted hard-label tail continues to benefit from
nonzero updates, approximately 15% more updates distributed across the same
counted-time LR curve will improve representation fitting and clean-label
margin refinement, reaching `best_test_acc >= 94.17%` within the same 300
counted seconds.

The mechanism is falsified by any valid scored run below 94.17%, regardless of
lower loss or higher throughput:

- At least 10% more exposure with worse accuracy and loss means BF16 numerical
  error or excess update count outweighs the throughput benefit.
- At least 10% more exposure with similar loss but sub-threshold accuracy means
  exposure is no longer the primary limiter.
- Less than 10% more realized exposure indicates loader or full-loop overhead
  absorbed most of the synthetic gain. The run remains a valid test of the
  deployed BF16 treatment, but it has weak evidence for the exposure mechanism.
- A non-finite loss or unsupported deterministic kernel is an implementation or
  feasibility failure, not evidence about accuracy; do not switch to FP16,
  GradScaler, batch 512, or a different LR inside the same experiment.

One fixed-seed scored run is sufficient. Do not reroll or repeat a near miss.

## Exact Scored Treatment

Start from accepted commit `eb08811`. Add a named BF16 training dtype near the
other constants:

```python
TRAIN_AUTOCAST_DTYPE = torch.bfloat16
```

Print the dtype once at setup for auditability. Restructure only the existing
per-batch forward/loss block so mixup interpolation remains FP32 and the model
forward plus cross entropy execute inside autocast:

```python
if use_mixup:
    train_inputs, targets_a, targets_b, mix = mixup_batch(
        inputs, targets, mixup_distribution
    )
else:
    train_inputs = inputs

with torch.autocast(
    device_type=device.type,
    dtype=TRAIN_AUTOCAST_DTYPE,
    enabled=device.type == "cuda",
):
    outputs = model(train_inputs)
    if use_mixup:
        loss = mix * F.cross_entropy(outputs, targets_a) + (
            1.0 - mix
        ) * F.cross_entropy(outputs, targets_b)
    else:
        loss = F.cross_entropy(outputs, targets)
```

Keep `loss.backward()` and `optimizer.step()` outside the autocast context.
Do not add `GradScaler`: BF16 has FP32-like exponent range, the local smoke
check is finite without scaling, and scaling would add a second treatment.
Do not cast the model or optimizer state to BF16. Do not wrap evaluation in
autocast; the frozen evaluator must continue evaluating the FP32 master weights
through its unchanged FP32 path.

Preserve all of the following exactly:

- `BATCH_SIZE = 256`, `LR = 0.2`, `MIN_LR = 0.002`, 5% warmup, momentum 0.9,
  Nesterov, and continuous `5e-4` decay on matrix weights only.
- WRN-16-2 and its 691,674 trainable parameters.
- Alpha-0.2 batchwise mixup through 65% counted time and the 35% hard-label
  tail.
- Time-based LR scheduling, `MAX_STEPS`, crop/flip transforms, loader settings,
  seed 42, evaluator, and every-fifth-epoch evaluation cadence.
- The finite-loss check, CUDA synchronization inside counted step time, and all
  summary metrics.

No LR scaling applies because batch size is unchanged. Peak LR 0.2 and the
0.002 floor are validated parts of the accepted recipe. More steps at each
point on the counted-time schedule are the intended mechanism, not a reason to
divide LR by the measured speedup.

## Determinism and RNG Accounting

Autocast itself consumes no random numbers. With batch size, seed, loader,
mixup branch, and operation order unchanged, an equal-length step prefix uses
the same sequence shape for data shuffles, crop/flip transformations, beta
samples, and `randperm` calls. The faster run intentionally completes more
steps and epochs, so it consumes a longer prefix of those deterministic
streams. Sparse evaluations add no training RNG operations.

The numerical trajectory is not bitwise identical: BF16 convolution inputs
and activations have fewer mantissa bits, eligible H20 kernels differ from
FP32, parameter values diverge from the first update, and a faster step rate
maps a given batch index to an earlier point on the wall-time LR schedule.
Those are intrinsic parts of the BF16-throughput treatment. Parameters,
gradients, optimizer momentum buffers, weight-decay arithmetic, mixup
interpolation, loss, and evaluation remain FP32. Keep the existing seed and
perform no repeat or seed selection.

The existing combination of `cudnn.benchmark = True` and
`cudnn.deterministic = True` must remain unchanged. The preflight must confirm
the selected BF16 path is finite under those exact flags; do not relax
determinism to obtain a faster kernel.

## Preflight Gate Before the One Scored Run

The development measurement above selected the proposal, but execution should
still verify the final production diff in a fresh, evaluator-free process.
This is a feasibility-only preflight and must not load the CIFAR-10 test set,
call `Eval.evaluate`, write `run.log`, or report an accuracy.

1. Confirm exactly one NVIDIA H20 and PyTorch 2.9.1 with CUDA BF16 autocast.
2. Stub module-scope `prepare.Eval` before importing the final `train.py`, as in
   the prior EXP-006 preflight, so no test loader is constructed.
3. On synthetic `[256, 3, 32, 32]` CUDA inputs, benchmark the actual FP32 and
   final BF16 mixup training steps with their real optimizer groups. Warm up at
   least 25 steps, then use at least three synchronized 50-step windows and
   compare medians. Use balanced order or endpoint repeats.
4. Require finite FP32 loss, finite BF16 loss, logits shape `[256, 10]`, FP32
   parameters/gradients/optimizer state, BF16 logits, and FP32 loss.
5. Require BF16 image throughput at least **1.10x** FP32 and a calibrated
   projection of at least **156.1 passes** (`141.9 * 1.10`). Require no OOM and
   keep peak allocation far below H20 capacity.

If either throughput gate fails, do not consume the scored run; return to idea
selection. Do not rescue the proposal by benchmarking multiple layouts,
batches, precision modes, or LR values. The observed 1.147x ratio gives enough
margin that the verification gate should pass without tuning.

## Full-Run Verification

After the production preflight passes:

1. Confirm `git diff` changes only `train.py`, with the autocast treatment and
   one setup log; confirm `prepare.py` is untouched.
2. Remove stale `run.log`, then execute exactly once with
   `timeout 600s uv run train.py > run.log 2>&1`.
3. Require one NVIDIA H20, exit code 0, finite losses, a complete final summary,
   approximately 300 counted training seconds, and no more than 600 total
   seconds.
4. Confirm the model remains WRN-16-2 with 691,674 parameters, batch 256, and
   BF16 training autocast. Confirm the evaluator remains FP32.
5. Confirm mixup disables exactly once near 195 counted seconds with LR about
   0.0612. Confirm evaluation occurs at most once per epoch and only at the
   accepted every-fifth-epoch/final cadence.
6. Record best/final accuracy, final loss, steps, epochs, peak VRAM, total wall
   time, transition step, and realized passes as
   `num_steps * 256 / 50_000`. Compare steps and passes to EXP-002's 27,735 and
   141.9 and to the calibrated 162.8-pass projection.

Success requires `best_test_acc >= 94.17%`. Final loss below EXP-002's 0.2432
and final accuracy near best support a stable gain but cannot replace the
primary threshold. More evaluations caused by completing more epochs are
permitted because they remain at most one per epoch; their time is excluded
from the frozen 300-second training budget, and the 600-second total cap remains
authoritative.

## Risks and Interpretation

- **BF16 rounding harms a small-network optimum:** Tensor-core convolution
  commonly accumulates in FP32, but BF16 activations have only seven explicit
  mantissa bits. Extra exposure may not recover a noisier optimization path.
- **More updates are not automatically better:** The time schedule is fixed,
  so a 14.7% faster loop performs about 14.7% more SGD updates at each time/LR
  region. This increases integrated update opportunity and may overshoot even
  though EXP-002 benefited from its late nonzero floor.
- **Loader overhead reduces realized speedup:** Synthetic timing excludes
  crop/flip workers and host-to-device transfer. Eight persistent workers make
  a large collapse unlikely, but realized passes, not synthetic projection,
  decide whether the exposure mechanism occurred.
- **Evaluation remains FP32:** This is intentional and evaluator-consistent.
  A gain therefore reflects FP32 master weights learned through BF16 training,
  not lower-precision test-time prediction or metric manipulation.
- **Autocast coverage is operation-specific:** BatchNorm and cross entropy may
  execute in FP32 while convolutions use BF16. That mixed policy is the stable
  PyTorch primitive being tested; do not manually force individual modules to
  BF16.
- **Sparse evaluation observes more epochs:** Faster training creates several
  extra legal evaluation points. `best_test_acc` is the frozen primary metric,
  so these are part of the treatment's realized trajectory, not justification
  for changing cadence.

## EXP-009 Recommendation

Advance **isolated BF16 autocast at batch 256** as a high-confidence feasibility,
medium-confidence accuracy candidate. It directly targets the measured
compute/exposure limiter, preserves every accepted hyperparameter, requires no
dependency or evaluator change, and has a locally measured 14.7% throughput
upside. Its numerical effect means the experiment is not pure additional-data
exposure, but it is substantially cleaner than BF16 plus a batch/LR change.

Do not score BF16 batch 512 in this experiment. Its measured 1.5% incremental
image-throughput gain is too small to compensate for the optimization and
BatchNorm confounds. If isolated BF16 passes feasibility but fails accuracy at
normal realized exposure, reject the BF16 path for this recipe rather than
post-hoc changing the batch, GradScaler, layout, precision, or LR.

## Evidence

- `experiments/002/04-analysis.md`: the accepted alpha-0.2 recipe reached
  94.07% with 27,735 steps / 141.9 passes, final equal to best, and continued
  progress through the hard-label tail.
- `experiments/006/04-analysis.md`: added block regularization regressed at
  normal exposure; its matched preflight establishes the evaluator-free
  production-code timing pattern.
- `experiments/006/proposals/idea-01.md`: WRN-16-3 retained 56.8% throughput,
  projected only 80.6 calibrated passes, and gained little from larger batches,
  showing the H20 bottleneck is model compute rather than memory capacity.
- `experiments/008/04-analysis.md`: zero-floor annealing preserved 142.5 passes
  but regressed to 93.80%, evidence that useful late updates remain rather than
  needing a quieter endpoint.
- `03-experiment-learnings.md` and `04-results.tsv`: all six post-EXP-002
  regularization/late-optimization variants failed, motivating an orthogonal
  throughput lever.
- Local EXP-009 proposal-development checks: FP32 batch 256 measured
  9.98-9.99 ms/step, BF16 batch 256 measured 8.70-8.71 ms/step, and BF16 batch
  512 measured 17.14 ms/step on one H20; a semantic check observed FP32
  parameters/gradients, BF16 logits, and FP32 loss.
