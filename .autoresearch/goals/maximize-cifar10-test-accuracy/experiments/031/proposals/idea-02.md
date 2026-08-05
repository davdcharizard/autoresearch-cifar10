# Proposal: End-to-End FP32 Channels-Last Training

## Summary

Run the accepted `67c8e98` EXP-027 learner in PyTorch's
`torch.channels_last` memory format while keeping its logical model, FP32
storage dtype, optimizer, data stream, augmentation, schedule, seed, and
evaluator unchanged. Convert the initialized model once after it reaches the
H20, request channels-last layout during each pinned host-to-device image
copy, and place an idempotent channels-last conversion at the start of
`WideResNet.forward` so the frozen evaluator's ordinary contiguous NCHW input
is handled explicitly. No autocast, compilation, fusion, precision switch,
or hyperparameter change belongs in this experiment.

This is a systems treatment with one falsifiable mechanism: if cuDNN's NHWC
FP32/TF32 convolution paths reduce full forward/backward step time by at least
2%, the fixed 300-second schedule will execute at least 2% more steps using the
same sampler, augmentation, Beta, and permutation decisions for an equal-length
prefix. The accepted H20 path is convolution-bound:
forward/backward consume about 98% of an isolated step, convolutional stages
dominate forward time, while H2D is only about 0.6%. The extra layout work is
therefore small enough to measure, but a speedup is not assumed. CIFAR's small
32x32 tensors and batch 256 may make NHWC neutral or slower. Only balanced
full-body timing can advance the treatment.

## Exact Production Treatment

Start from accepted commit `67c8e98`. Preserve direct seed-42 construction so
initial random values and the 987,098-parameter `(2,2,3)` WRN remain accepted.
Change setup to convert the already initialized module before optimizer
construction:

```python
model = WideResNet(STAGE_BLOCKS, WIDEN_FACTOR, NUM_CLASSES).to(
    device=device,
    memory_format=torch.channels_last,
)
```

Construct the two accepted SGD parameter groups only after this conversion.
This avoids any ambiguity about optimizer references while preserving exactly
the accepted name/shape-based decay split.

At the first line of `WideResNet.forward`, add:

```python
x = x.contiguous(memory_format=torch.channels_last)
```

For an already channels-last training tensor this is an allocation-free
identity. For the frozen evaluator's `[N,C,H,W]` contiguous input, it performs
the required layout conversion before the first convolution without changing
shape, indexing, values, or dtype. Keep the existing adaptive pool and
`view(out.size(0), -1)`; the `[N,128,1,1]` channels-last result is viewable as
`[N,128]` in the installed PyTorch 2.9.1 environment.

Change only the image transfer in the counted training body:

```python
inputs = inputs.to(
    device,
    non_blocking=True,
    memory_format=torch.channels_last,
)
targets = targets.to(device, non_blocking=True)
```

The CPU loader continues to emit ordinary contiguous NCHW tensors. The
requested device layout must be verified at runtime. `mixup_batch` remains
unchanged; indexing and elementwise interpolation are required to preserve the
channels-last stride. Both hard and mixed tensors must reach the model as
FP32 channels-last tensors. Do not add a second explicit conversion after
mixup.

Log the training memory format once for auditability. Do not change model
classes, layer order, parameter values, BatchNorm behavior, loss, optimizer,
LR writes, mixup/RandAugment transitions, loader construction, evaluation
cadence, or summary.

## FP32 and cuDNN Scope

"FP32" here means the accepted storage and API precision: input, parameters,
activations, gradients, momentum buffers, logits, and loss remain
`torch.float32`. Do not use BF16/FP16, autocast, GradScaler, manual casts, or
reduced-precision optimizer state.

The installed accepted environment already reports cuDNN convolution FP32
precision as TF32 (`torch.backends.cudnn.allow_tf32=True` and
`torch.backends.cudnn.conv.fp32_precision="tf32"`). Preserve that state
exactly; neither enable nor disable TF32. Preserve
`cudnn.benchmark=True` and `cudnn.deterministic=True`. Channels-last may make
cuDNN choose a different deterministic convolution implementation and a
different FP32 accumulation order. That intrinsic numerical trajectory is the
treatment, not a precision-policy change.

## Evaluator Compatibility

Do not edit `prepare.py` or wrap/replace `Eval`. Its loader transfers an
ordinary contiguous NCHW FP32 tensor and calls `model(inputs)`. The forward
entry conversion is the sole compatibility bridge and must work in both train
and eval modes.

An evaluator-free preflight must simulate this exact contract with synthetic
contiguous `[256,3,32,32]` input under `model.eval()` and
`torch.inference_mode()`. Require a finite `[256,10]` FP32 output, no parameter
or buffer mutation, and bitwise-equal output between (a) the simulated frozen
evaluator input and (b) the same logical input preconverted to channels-last,
because both reach the same model layout before convolution. Also require the
candidate's predictions and loss to be numerically close to an independent
accepted oracle as described below. Evaluation time is outside the 300-second
counted budget; its conversion cost cannot justify modifying cadence.

## Fixed-Seed Numerical Trajectory

Layout conversion consumes no random numbers. For an equal-length step prefix,
the sampler, crop/flip, isolated RandAugment worker streams, batch-shared beta
draws, permutations, and seed-42 global CPU/CUDA streams must have the same
states and draws as accepted. A faster run intentionally consumes a longer
prefix inside 300 seconds; this is exposure, not a seed reroll.

Do not claim bitwise accepted/candidate training identity. Different legal
cuDNN kernels can change reduction order, so logits, gradients, the first SGD
update, and all later weights may differ in low FP32 bits from step one even
under deterministic execution. The correct standard is:

- identical logical initial state values, topology, parameter names/shapes,
  optimizer algorithm/groups, data values, stochastic decisions, and dtype;
- bounded accepted/candidate numerical differences at initialization and one
  update;
- exact candidate self-replay from restored model, optimizer, and RNG state;
- no run or seed repetition based on the resulting accuracy.

## Semantic Gate

Before timing, use an ignored evaluator-free harness that stubs `prepare.Eval`
before importing either module. Load the accepted implementation independently
from `git show 67c8e98:train.py`; do not manufacture the accepted arm by
converting the candidate back to contiguous format.

Require all of the following:

1. Direct seed-42 construction occurs before memory conversion. Named
   `state_dict` shapes, dtypes, and logical values are byte-equal after copying
   candidate 4D tensors to ordinary contiguous CPU storage. CPU and CUDA RNG
   states after construction/conversion equal accepted.
2. Parameter count is exactly 987,098. Every convolution weight and every
   four-dimensional candidate activation observed by hooks is channels-last;
   one-dimensional BN tensors and two-dimensional FC tensors retain their
   natural layouts. All parameters, gradients, buffers, logits, losses, and
   momentum buffers are FP32.
3. Optimizer class, Nesterov/momentum values, LR, parameter names per group,
   matrix-only `5e-4` decay split, and initially empty state are accepted.
   After one update, momentum keys/shapes/dtypes and the logical SGD formula
   remain accepted.
4. Pinned NCHW H2D conversion preserves every input value and produces the
   required device stride. `mixup_batch` returns the same targets,
   permutation, coefficient, and logical values as accepted from restored RNG,
   while its image result remains channels-last. Hard input does likewise.
5. On fixed eval input, accepted/candidate initial logits satisfy
   `torch.testing.assert_close(rtol=2e-4, atol=2e-5)`, have identical argmaxes,
   and cross entropy differs by at most `2e-5`. These bounds allow reduction
   order but reject a transposition or semantic mistake.
6. On one restored training step, every gradient is finite and present exactly
   where accepted. The concatenated candidate gradient and SGD-update vectors
   each have relative L2 error at most `1e-3` versus accepted. Do not require
   bitwise cross-layout equality.
7. Repeating the candidate step from identical model/optimizer/RNG snapshots
   produces bitwise-equal loss, gradients, post-step state, and RNG state.
   Preserve the accepted cuDNN benchmark/deterministic/TF32 flags during this
   check.
8. Source/runtime audits show no change to seed, dtype, batch, architecture,
   optimizer, schedule, transforms, loader, mixup, RandAugment cutoff, time
   accounting, evaluator cadence, or frozen files. The preflight never loads
   test data or writes `run.log`.

Any semantic failure aborts this exact idea. Do not loosen a numerical bound
after seeing it; diagnose only harness mistakes that cannot affect production.

## Balanced Full-Body H20 Timing Gate

Time the independent accepted and candidate production bodies on the one idle
NVIDIA H20. Absolute convolution microbenchmarks are insufficient. Each timed
step begins before pinned host-to-device input/target copies and ends after the
real synchronized SGD/Nesterov update. Include LR/group writes, zero-grad,
beta sampling and permutation in mixup mode, forward, cross entropy, finite
guard, backward, optimizer step, and final `torch.cuda.synchronize()`.

Benchmark both regimes:

- early path at 50% counted progress, including accepted batch-shared mixup;
- hard-label path at 80% counted progress.

For every arm/window, restore fresh logically identical model and optimizer
snapshots, identical pinned host fixtures, and identical global RNG state.
Warm each relevant layout/path for at least 25 steps after restoration so
cuDNN benchmarking and allocations are outside measurement. Measure at least
three continuing 50-step windows per arm and regime in a symmetric balanced
order such as `A-C-C-A-A-C`; use wall-clock timing around the full synchronized
body. Record and print every window before enforcing gates.

For each arm/regime, take the median window mean and require population CV at
most 3%. Because the regimes occupy fixed time rather than fixed step shares,
compute:

```text
accepted_rate   = 0.65 / accepted_mixup_ms + 0.35 / accepted_hard_ms
candidate_rate  = 0.65 / candidate_mixup_ms + 0.35 / candidate_hard_ms
speedup         = candidate_rate / accepted_rate
projected_passes = 133.00736 * speedup
```

Proceed to the sole score only if all semantic checks and CV gates pass,
`speedup >= 1.0200`, and `projected_passes >= 135.67`. This preregisters a
material gain of at least 2% rather than spending a score on harmless layout
churn. It is plausible on the H20 because almost the entire step is
forward/backward convolution and H2D is under 1%, but it is intentionally not
guaranteed. A stable timing miss closes exact end-to-end channels-last without
a score. Do not try channels-last weights only, inputs only, alternate
conversion placement, a different cuDNN flag, autocast, compile, or fusion as
a rescue.

No real-loader timing is needed: CPU transforms, workers, batch, and loader are
unchanged, and the only changed boundary cost is included from pinned host
input through synchronization. The accepted run's 345.3-second wall time also
leaves ample margin for the few additional out-of-budget evaluations created
by a faster run.

## One-Run Decision and Closure

After passing preflight, require a `train.py`-only production diff, frozen
`prepare.py`, local CIFAR-10, exactly one NVIDIA H20, no stale log, and execute
exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, one finite summary, 300.0-300.1 counted seconds, total below
600 seconds, 987,098 parameters, FP32 state, unique at-most-once-per-epoch
evaluations, the accepted mixup and exhausted-epoch RandAugment transitions,
and realized exposure `num_steps * 256 / 50000 >= 135.67` passes. Record peak
VRAM but impose no memory gate beyond successful H20 execution.

The current baseline is 94.32%, so formal improvement requires
`best_test_acc >= 94.42%`. Separately report whether predetermined
`final_test_acc >= 94.32%` corroborates a gain when faster exposure creates
more legal best-of-trajectory evaluation opportunities, along with final loss
relative to accepted 0.2523 and the projected/realized evaluation counts.
These corroboration signals cannot override the primary objective.

One valid miss closes this exact channels-last treatment. Do not rerun seed 42
or tune layout placement, conversion timing, benchmark flags, batch, LR,
precision, or evaluation cadence. If the scored run realizes less than 135.67
passes, it still counts and cannot be rerun; classify the exposure mechanism
as operationally failed. A timeout, non-finite loss, wrong layout/precision,
invalid transition, or missing summary is a crash, not permission for a rescue
variant.

## Expected Outcome and Risks

- **Expected H20 throughput:** uncertain but plausibly 2-8% faster if cuDNN
  selects better NHWC forward and backward kernels. The 2% gate is the minimum
  useful effect; measured full-body timing is authoritative.
- **Numerical divergence:** same FP32 values and stochastic stream do not imply
  the same optimization path. Kernel accumulation differences can erase any
  benefit, just as BF16's larger exposure did not improve accuracy, although
  this proposal does not reduce storage precision.
- **Exposure may not be the limiter:** EXP-027 is generalization-limited and
  nearly interpolates the hard tail. More passes are a weak accuracy mechanism,
  so the proposal is lower-confidence than a justified learning intervention.
- **Layout conversion may dominate:** small CIFAR tensors may not amortize the
  NCHW-to-channels-last transfer. Including pinned H2D and the evaluator-safe
  forward guard in timing prevents hiding this cost.
- **Evaluation trajectory changes:** the evaluator is logically unchanged but
  executes candidate layout kernels. Low-bit logit differences and extra
  evaluation epochs are intrinsic, hence the final-accuracy corroboration.

## Evidence

- `02-system-understanding.md`: accepted EXP-027 completes 133.007 passes;
  forward/backward are about 24%/74% of step time, H2D about 0.6%, and memory
  is not binding on the H20.
- `experiments/027/04-analysis.md`: accepted `(2,2,3)` plus early RandAugment
  scored 94.32% best, 94.22% final, 0.2523 loss in 300.0 counted / 345.3 total
  seconds with 987,098 parameters.
- `experiments/009/04-analysis.md`: BF16 produced substantially more exposure
  but lost accuracy, proving that throughput is not itself an accuracy result
  and motivating FP32 plus a strict one-run closure.
- `experiments/010/01-idea-review.md` and `experiments/023/01-idea-review.md`:
  prior systems ideas were rejected without a material full-production timing
  opportunity; the same standard applies here.
- Installed local stack: PyTorch 2.9.1+cu128, CUDA 12.8, cuDNN 9.10.2, one
  NVIDIA H20; no package, network, or remote service is required.
