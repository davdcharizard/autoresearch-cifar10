# Proposal: FP32 Channels-Last ResNet-20

## Intervention and falsifiable hypothesis

Run the accepted width-2 ResNet-20 and its CUDA image tensors in PyTorch
`torch.channels_last` memory format while preserving FP32/default-TF32 arithmetic,
batch 128, architecture, parameter values, SGD, schedule, N1/M7 plus probability-0.5
alpha-1 CutMix, the hard weak tail, seed, timer, and evaluator. The intervention
changes strides and eligible cuDNN kernel selection, not tensor dimensions or the
model equation.

EXP-013 measured forward plus backward at 97.57% of GPU-stage time, with backward
alone at 75.46%; loader wait, transfer, loss, optimizer, and visible launch gaps
are too small to fund meaningful exposure. PyTorch documents channels-last support
for cuDNN Conv and BatchNorm, but its strongest published gain is for reduced-
precision Tensor-Core ResNet-50. This FP32, batch-128, 32x32 model may gain nothing.

**Hypothesis:** channels-last reduces the weighted synchronized full-step mean by
at least 3%, increasing fixed-budget exposure from 26,898 to at least 27,700
updates without meaningful numerical drift, and the extra accepted-recipe updates
raise `best_test_acc` from 94.15% to at least 94.25%. There is no proposed
representation or regularization mechanism. Failure of either the speed gate or
the accuracy threshold falsifies this operating point.

## Exact `train.py` implementation

1. After ordinary Kaiming initialization, move the model with
   `model.to(device=device, memory_format=torch.channels_last)` **before** creating
   SGD. Four-dimensional convolution weights become channels-last; BN vectors and
   the linear matrix retain their natural layouts. Parameter names, values,
   ordering, and count remain exactly 1,073,962.
2. In the timed training step, replace the image transfer with
   `inputs.to(device=device, non_blocking=True,
   memory_format=torch.channels_last)`. Keep targets unchanged. The transfer and
   restriding cost therefore counts against the same 300 seconds.
3. At entry to `ResNet.forward`, use
   `x = x.to(memory_format=torch.channels_last)` on CUDA. This is a no-op for
   already converted training inputs and converts the contiguous CUDA tensors
   supplied by the read-only `Eval` harness. It keeps evaluator code and logits
   semantics unchanged while ensuring one declared layout throughout inference.
4. Do not call `permute`, change public NCHW shapes, enable autocast, set
   `cudnn.benchmark`, alter TF32/determinism flags, compile the model, change batch
   size, or combine the layout with wider channels. EXP-016 already vetoed the
   numerically distinct full-forward BF16 width-3 path; this proposal stays FP32.

Preflight hooks must verify every Conv2d weight and every four-dimensional input
and output of Conv/BN/ReLU is channels-last in the candidate. The Option-A
slice/pad shortcut and residual add must propagate that layout. Any repeated
fallback to contiguous format, implicit permutation between blocks, or operator
error rejects this implementation rather than inviting graph rewrites.

## Semantic and numerical gates

Use one accepted initialization/state and an immutable 200-batch production
corpus (80 strong-hard, 80 strong-CutMix, 40 weak-hard), hashed before either arm.
The contiguous control and channels-last candidate must consume byte-identical
values and targets in the same order.

- Assert identical architecture, parameter/state keys and values, RNG state,
  optimizer groups, schedules, target formats, and parameter count. Converting an
  input back to contiguous must be bitwise equal to its source; conversion must
  consume no RNG.
- In eval mode, compare at least 20 fixed batches including evaluator-sized N=256.
  Require finite logits, `max_abs(logit_delta) <= 1e-3`, normalized logit RMSE
  `<=1e-4`, cross-entropy difference `<=1e-4`, at least 99.9% argmax agreement,
  and no parameter or BN-buffer mutation.
- From independently restored model/SGD state, compare hard and soft one-step
  training. Require finite losses, gradients, momentum, parameters, and buffers;
  loss difference `<=1e-4`; global gradient and update relative-L2 differences
  `<=1e-3`; no trainable tensor above `5e-3`; and BN running-stat relative-L2
  differences `<=1e-3`. Record, but do not require, bitwise equality because
  cuDNN reduction order may change.
- Continue both arms for all 200 exact-corpus steps. Require no nonfinite state,
  no candidate-only prediction concentration above 95%, candidate terminal loss
  EMA within 10% of control, and no BN counter or target-path divergence. Serialize
  the first threshold event before asserting.

These are equivalence/safety bounds, not tuning objectives. A failure blocks
timing and production; do not loosen tolerances or switch algorithms after seeing
the result.

## Real-loader H20 timing and exposure gate

On exactly one idle 97,871-MiB H20, run five alternating fresh-process
control/candidate pairs with the actual eight-worker strong and weak loaders.
Condition every arm for 100 complete steps, then measure at least 1,000 synchronized
steps. Measure strong hard/probability-target and weak hard paths separately and
combine them 40/40/20. Include iterator wait, nonblocking H2D/layout conversion,
forward, loss, backward, SGD, and synchronization; also record CUDA-event transfer,
forward, backward, and update components. Keep cuDNN/TF32 settings identical and
discard no slow candidate trial.

Proceed only if all hold:

- weighted candidate/control full-step mean `<=0.970`, every paired ratio
  `<=0.985`, and both trial-mean CVs `<=1.5%`;
- candidate p95 full-step time `<=0.99x` paired control p95;
- candidate combined forward-plus-backward mean `<=0.970x` control, demonstrating
  that any gain comes from the measured model bottleneck rather than loader noise;
- `floor(26,898 * control_mean / candidate_mean) >=27,700` projected updates,
  with the same 80/20 time allocation;
- candidate peak allocation below 700 MiB, finite state throughout, and a
  conservative end-to-end projection below 540 seconds including evaluation;
- projected and dry-run evaluation counts equal EXP-010's 19 unique epochs. If
  faster epoch crossings would add a test look, pre-register a fixed 19-point
  elapsed-progress evaluation schedule for both timing validation and production;
  never accept a max over more looks.

The 3% floor is intentionally stronger than “not slower.” Additional exposure has
not yet been shown to improve this recipe, so a marginal kernel win does not
justify the sole fixed-seed accuracy run. EXP-013's stable fresh pairs superseded
a misleading serial estimate, and EXP-023 showed paired ratios can predict actual
exposure within 1.5%; the same alternating-pair discipline applies here.

## Production verification

Only after all gates pass, run the exact candidate once at seed 42 via
`uv run train.py > run.log 2>&1`. Require exit zero, 300.0 counted training seconds,
total below 600 seconds, finite standard summary fields, 1,073,962 parameters,
at least 27,700 updates, one 80% loader switch with eight workers stopped,
approximately 50% strong-phase CutMix, hard weak-tail targets, and exactly 19
unique at-most-once-per-epoch evaluations.

Record switch accuracy versus 89.73%, first weak accuracy versus 93.16%, best and
final accuracy, final NLL versus 0.1934, steps, epochs, stage timings, total time,
and peak memory. Acceptance requires `best_test_acc >=94.25%` with every integrity
condition satisfied. A switch below 87.08% diagnoses numerically induced strong
underfit but cannot stop or rerun the production trajectory.

## Risks and abort criteria

- Official evidence says the largest GPU gains occur with FP16/Tensor Cores; FP32
  CIFAR kernels are small and may be unchanged or slower on H20.
- The input restride and unsupported Option-A/padding operators can erase Conv/BN
  gains or cause hidden format conversions. Any propagation break or timing miss
  aborts before accuracy.
- Layout is mathematically semantic-preserving but not bitwise-preserving: cuDNN
  may choose different kernels and reduction orders. The paired numerical gates
  bound this unavoidable confound; EXP-016 makes unbounded precision drift
  unacceptable.
- More steps are the only accuracy hypothesis. If exposure rises but accuracy does
  not, do not attribute the miss to insufficient capacity or add width within this
  experiment.
- Faster epochs can create extra dense-tail evaluations and inflate a max metric.
  The fixed-19-look guard is mandatory, even though evaluation is outside the
  training timer.
- A bare 0.10-point pass is only ten CIFAR-10 examples and one seed; it is
  protocol-valid but weak causal evidence.

Any semantic, layout-propagation, numerical, timing, exposure, evaluation-count,
hardware, completion, or accuracy failure retires this exact FP32 channels-last
candidate. No BF16, larger model, compiler, alternate batch, evaluator change, or
kernel-setting fallback is allowed.

## Sources

- PyTorch, *Channels Last Memory Format in PyTorch*:
  https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html —
  Conv/BatchNorm support, explicit model/input conversion, fallback-conversion
  risk, and strongest gains in reduced precision.
- `experiments/013/04-analysis.md` — measured stage costs, alternating-pair timing
  precedent, and evaluation-look control for throughput changes.
- `experiments/016/04-analysis.md` — BF16 numerical veto and reason to isolate
  FP32 layout from precision/capacity.
- `experiments/023/04-analysis.md` — H20 width timing and prediction-to-production
  exposure accuracy.
- `02-system-understanding.md`, `03-experiment-learnings.md`, and `04-results.tsv`
  — accepted bottleneck, protocol findings, failures, and 94.15% frontier.
