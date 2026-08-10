# Proposal: FP32 Channels-Last Training

## Proposal and hypothesis

Apply one physical-layout intervention to the accepted EXP010 recipe: initialize the
batch-128 width-2 postactivation ResNet-20 exactly as accepted, move it to the H20,
then restride its four-dimensional parameters to `torch.channels_last`. Transfer every
four-dimensional training input with the same memory format inside the existing counted
step. Logical tensor dimensions remain NCHW and all arithmetic stays FP32 under the
installed default TF32/cuDNN settings.

The measured workload spends 2.408 ms in model forward and 8.220 ms in model backward,
97.57% of its GPU-stage time; backward alone is 75.46%. PyTorch's official memory-format
tutorial documents channels-last propagation through CUDA convolution and BatchNorm when
both model and input use the layout. The testable hypothesis is that NHWC-compatible
kernels reduce complete synchronized step time by at least 3%, increasing the accepted
26,898-step fixed-budget exposure to at least 27,705 updates, without changing batch noise,
capacity, optimizer logic, data, or evaluation semantics, and that the extra exposure plus
the legal kernel-numerics change raises seed-42 `best_test_acc` from 94.15% to at least
94.25%.

This is principally an exposure hypothesis. Official headline gains are strongest for
reduced precision and larger images, so a neutral or slower result on tiny 32x32 FP32
activations is entirely plausible and must be decided by local end-to-end timing.

## Exact `train.py` changes

Keep model construction and initialization order unchanged, then convert layout after the
ordinary device move and before optimizer construction:

```python
model = ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULTIPLIER).to(device)
model = model.to(memory_format=torch.channels_last)
num_params = sum(p.numel() for p in model.parameters())

optimizer = optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
```

Change only the input transfer in the counted training interval:

```python
inputs = inputs.to(
    device,
    non_blocking=True,
    memory_format=torch.channels_last,
)
targets = targets.to(device, non_blocking=True)
```

At the beginning of `ResNet.forward`, add:

```python
x = x.to(memory_format=torch.channels_last)
```

For training this must be a same-pointer, zero-allocation no-op because the input was
already converted and charged after `t0`. It is needed for the immutable evaluator, whose
contiguous test inputs are transferred in `prepare.py`; their one boundary restride occurs
outside the training counter and lets evaluation exercise the same logical model without
editing `prepare.py`. No operator-specific permutations or rescue conversions are allowed.

Because a faster batch-128 candidate completes more epochs and the metric is a maximum,
replace the open-ended dense-tail evaluator trigger with exactly 19 elapsed-progress
thresholds: the accepted early `(0.2, 0.4, 0.6, 0.7)` looks plus 15 points evenly spaced
from 0.8 through 1.0 inclusive. Keep the existing single evaluator call, threshold-advance
loop, final evaluation, and at-most-once-per-epoch behavior. A preflight simulation must
prove the measured candidate schedule yields 19 unique evaluation epochs including the
terminal epoch. This is a fairness control matching EXP010, not a training mechanism.

Everything else remains accepted: width multiplier 2, 1,073,962 parameters, batch 128,
postactivation block order, Option-A transitions, Kaiming initialization, global average
pooling, ordinary SGD momentum 0.9, all-parameter decay `1e-4`, LR `0.1` through 80%, the
`0.01`-to-`1e-4` weak-tail cosine, N1/M7 plus probability-0.5 alpha-1 CutMix, persistent
workers, seed 42, 300 counted seconds, and the unmodified `Eval.evaluate()` implementation.
Do not enable autocast, BF16/FP16, compilation, cuDNN benchmark/determinism changes,
explicit TF32 changes, fused SGD, larger batches, or an architecture/data modification.

## Evidence and relationship to prior results

- `02-system-understanding.md` localizes 75.46% of GPU-stage time to convolution/BN
  backward and another 22.11% to forward, while visible launch/synchronization overhead is
  only 0.034 ms. Layout attacks the measured limiter rather than a small Python stage.
- `knowledge/references/pytorch-channels-last.md`, distilled from the official PyTorch
  tutorial, establishes that channels-last changes physical strides while preserving
  logical NCHW dimensions and is supported by CUDA convolution and BatchNorm. It also
  warns that unsupported operators can introduce hidden conversions and that FP32 CIFAR
  speed must be measured.
- EXP013 showed that fresh paired timing can overturn a serial throughput estimate and
  that changing epoch length can bias `best_test_acc` through extra evaluations. Hence
  seven alternating fresh pairs and exactly 19 looks are load-bearing.
- EXP003/029 showed that nominally small per-step helpers can cost 1.97-6.7% exposure.
  Input restriding and hidden layout repairs are therefore included in complete-step timing.
- EXP016's width-3 BF16 failure does not test this point: this proposal keeps width 2 and
  FP32, introduces no autocast, and requires production-distribution numerical safety.
- EXP020/022/028 optimizer failures are avoided by retaining ordinary SGD and its exact
  state path. EXP034/036/037 representation failures are avoided because logical weights,
  padding, activations, and functions are unchanged; only physical strides and legal CUDA
  reduction order differ.
- EXP029 developed but deferred channels-last in brainstorming; it never changed code,
  ran timing, or produced a metric. EXP038 is therefore the first actual layout experiment,
  not a retry of a measured failure.

## Construction and layout preflight

Use an experiment-local controller; no diagnostic hooks or profiler calls enter production.

1. Construct accepted and candidate models from seed 42. Before layout conversion require
   identical named state, CPU/CUDA RNG hashes, parameter count, and optimizer configuration.
   After conversion require logical parameter/buffer values bitwise equal and unchanged RNG.
2. Require all 19 four-dimensional Conv weights to be channels-last contiguous and retain
   their logical shapes. BN vectors and the Linear matrix must retain accepted layouts.
   After one hard and one CutMix update, every Conv gradient and corresponding momentum
   buffer must also be channels-last contiguous.
3. Require a candidate `[128,3,32,32]` input stride of `(3072,1,96,3)`, identical logical
   values, and a same-pointer no-op at the forward boundary. For an evaluator-like
   contiguous `[256,3,32,32]` CUDA input, require exactly one boundary allocation/restride
   and finite `[256,10]` logits.
4. Hook all Conv, BN, and residual-block outputs. Require channels-last propagation at
   32x32, 16x16, and 8x8, and explicitly inspect the two strided-slice/F.pad Option-A
   shortcuts, post-add tensors, adaptive pool output, flattening, and classifier input.
5. Profile warmed hard and probability-target steps with shapes and stacks. Reject repeated
   `aten::contiguous`, `_to_copy`, `clone`, or permutation repairs between convolutions.
   Record conversion counts and forward/backward CUDA time by stage. A changed cuDNN
   algorithm is expected; a hidden per-block format shuttle is not.

Any state/RNG/value drift at conversion, broken `.view`, incorrect optimizer-state layout,
or repeated hidden repair is an implementation no-go. Do not patch individual blocks as a
fallback.

## Control-qualified numerical safety

Reuse without regeneration the registered EXP022 200-batch strong corpus (94 hard/106
CutMix, SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`)
and EXP028 64-batch weak corpus (SHA-256
`ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`). Verify schemas,
tensor hashes, target ranks/sums, ordering, and unchanged file bytes before and after use.

First run two accepted/accepted calibrations under production-default backend flags. They
must pass frozen finite-state, BN-counter, class-concentration, loss, gradient, and update
ceilings before candidate comparisons have authority; this incorporates EXP035's lesson
that a noisy or zero denominator cannot veto only the candidate. Then replay independent
accepted/candidate arms from identical logical state over all 200 strong batches at LR 0.1
and 64 weak batches along the accepted cosine tail.

Require exact 264 BN updates, positive running variances, finite parameters/gradients/
momentum/logits/loss, no candidate-only persistent >95% predicted-class share, terminal
strong and weak loss EMA each no more than 1.5x control, whole gradient and update norms no
more than 5x the matching control and preceding-window median, and candidate maximum update
below 25% of parameter norm. On matched initial hard and CutMix batches additionally require
logit cosine >=0.999, relative logit L2 <=0.02, loss ratio in `[0.98,1.02]`, and gradient/
update norm ratios in `[0.90,1.10]`. All ratios must use denominator-safe absolute-plus-
relative definitions and report the two control floors.

This is a catastrophic numerical screen, not a demand for bitwise trajectories and not an
accuracy proxy. Unlike EXP037, no long-horizon generic divergence is used as a mechanism
survival gate: layout propagation and measured kernel time directly establish this idea's
mechanism.

## Seven-pair timing gate

After one explicitly unscored conditioning process, run seven alternating fresh-process
control/candidate pairs on one idle H20. Rotate order so both arms appear first. Each arm
restores identical logical model/optimizer state, uses byte-identical pinned CPU batches,
warms 100 complete steps, then measures at least 1,000 complete synchronized steps in the
registered 40% strong-hard, 40% strong-CutMix, 20% weak-hard weighting.

Timing begins before H2D and includes candidate input restride, target transfer, zero-grad,
forward, cross-entropy, backward, ordinary SGD, and final synchronization. Record each
pair's mean/median/p95, CV, images/s, peak allocation, input/weight/gradient/momentum
strides, conversion counts, and CUDA-event transfer/forward/loss/backward/update stages.

Production is authorized only if:

- weighted candidate/control mean step ratio is `<=0.9700`, every pair is `<1.0`, per-arm
  trial-mean CV is <=2%, and ratio CV is <=2%;
- `floor(26_898 / weighted_ratio) >= 27_705` projected updates, with candidate weighted p95
  no slower than control p95;
- candidate forward+backward time is lower than control and at least 90% of the absolute
  CUDA-stage saving comes from those two stages, consistent with the measured limiter;
- layout/profiler and numerical gates still pass, peak allocation is <700 MiB and no more
  than 64 MiB above control, and a conservative full-run wall projection is <540 seconds.

A stable speedup below 3% fails the declared exposure premise. Do not move conversion before
`t0`, lower the gate, enable reduced precision, add batch scaling, or combine another idea.
The actual production update count is an integrity check, not permission to rerun timing.

## Production verification and decision rule

If and only if all preflight gates pass:

1. Confirm the moving baseline is 94.15% at `7c1e7d8`, hence the improvement threshold is
   94.25%; confirm exactly one idle ~97,871 MiB H20.
2. Require the tracked diff from the baseline to contain only `train.py` and only the model,
   input, forward-boundary, and fixed-19-look changes described above. Pass compile, Ruff,
   format, pre-commit, and `git diff --check`.
3. Simulate the measured epoch timing and require exactly 19 unique evaluation epochs,
   including the terminal epoch and never more than one evaluation per epoch. Measure the
   evaluator-like contiguous-input boundary and project total wall below 540 seconds.
4. Remove stale completed logs and run exactly once at seed 42 with
   `uv run train.py > run.log 2>&1` under the 600-second supervisor. No reroll or alternate
   layout point is allowed.
5. Require exit zero, approximately 300 counted seconds, total below 600, ten finite summary
   fields, 1,073,962 parameters, one ~80% strong-to-weak switch, eight stopped strong
   workers, 45-55% CutMix, hard weak targets, and exactly 19 unique evaluations.
6. Require at least 27,705 actual updates to support the exposure hypothesis. Record actual
   strong/weak steps, image slots, step ratio, peak memory, switch accuracy, first weak
   accuracy, final/best accuracy, final NLL, and total wall time against EXP010.
7. `best_test_acc >=94.25%` with all integrity conditions is improvement. A complete lower
   result is no-improvement and is reverted without reroll. A metric pass below the exposure
   floor is reported honestly but falsifies the registered mechanism and requires adversarial
   analysis; it does not authorize a second run. Any scope, layout, safety, timing, evaluator,
   or wall failure is invalid and blocks production.

## Risks and expected impact

The largest risk is no FP32 speedup: tiny CIFAR tensors, default TF32, Option-A slicing/pad,
and input restriding may erase any NHWC kernel advantage. A second risk is that 3% more
updates—about 807 accepted-scale decisions—is too little to move generalization by 0.10
points. Different cuDNN reductions also change the seed-42 trajectory, so a result is the net
channels-last implementation effect rather than a bitwise exposure counterfactual. Finally,
the +0.10 gate is only ten test images; one fixed-seed pass is protocol-valid but weak causal
evidence.

The upside is unusually clean: if the timing gate passes, the candidate directly accelerates
the 97.57% model-compute region while preserving the accepted model, batch-128 noise,
curriculum, optimizer, parameter count, and FP32 semantics. Expected metric impact is modest,
with a point estimate near 94.25% and a plausible valid-run range of roughly 94.0-94.4%; the
proposal is valuable only because feasibility and exposure can be established before spending
the single scored run.

## Sources

- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/01-definition.md`
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md`
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/knowledge/references/pytorch-channels-last.md`
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/013/04-analysis.md`
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/029/proposals/idea-02.md`
- PyTorch official memory-format tutorial: https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html
