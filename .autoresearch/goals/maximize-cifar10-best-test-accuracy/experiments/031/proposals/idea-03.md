# Proposal: FP32 Channels-Last with Exactly 19 Evaluations

## Intervention and falsifiable hypothesis

Run the accepted EXP-010 width-2 ResNet-20 and CUDA image tensors in
`torch.channels_last` physical memory format. Preserve logical NCHW shapes,
FP32/default-TF32 arithmetic, batch 128, parameter values, architecture, ordinary
momentum SGD, all-parameter decay `1e-4`, N1/M7 plus probability-0.5 alpha-1
CutMix through 80%, the LR-0.01 weak hard tail, seed 42, workers, timer, and the
read-only `Eval.evaluate()` implementation.

EXP-013 attributes 22.11% of GPU-stage time to forward and 75.46% to backward,
making convolution/BN layout one of the few remaining ways to add optimizer
updates without changing batch noise or model capacity. PyTorch documents
channels-last propagation through CUDA Conv2d and BatchNorm, but its strongest
GPU examples use reduced precision and larger images. Tiny 32x32 FP32 kernels may
be neutral or slower.

**Hypothesis:** channels-last cuts synchronized full-step time by at least 3%,
increasing exposure from 26,898 to at least 27,705 updates in 300 seconds, and the
extra accepted-recipe updates raise `best_test_acc` from 94.15% to at least
94.25%. Layout has no intrinsic representation or regularization mechanism. A
timing miss blocks production; an exposure pass followed by lower accuracy
falsifies the extra-updates hypothesis.

## Exact implementation

Preserve accepted initialization before restriding:

```python
model = ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULTIPLIER).to(device)
model = model.to(memory_format=torch.channels_last)
num_params = sum(p.numel() for p in model.parameters())
optimizer = optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
```

Moving to CUDA first and restriding second leaves constructor/Kaiming RNG order
unchanged; SGD is created only after parameters have final storage. In the counted
step, replace only the image transfer with:

```python
inputs = inputs.to(
    device, non_blocking=True, memory_format=torch.channels_last
)
```

Targets stay unchanged and transfer/restride cost remains inside `t0..synchronize`.
At the start of `ResNet.forward`, execute
`x = x.to(memory_format=torch.channels_last)`. This must be a same-pointer no-op
for training inputs and the single explicit conversion for contiguous CUDA test
inputs supplied by the immutable evaluator. Keep Option-A slice/pad, residual
adds, adaptive average pooling, `.view`, and classifier unchanged.

Do not enable autocast/BF16/FP16, compilation, channels-last-3d, fused SGD,
`cudnn.benchmark`, deterministic-mode changes, explicit TF32 changes, larger
batches, architecture changes, or a second accuracy mechanism. Record and require
identical control/candidate values for cuDNN version, benchmark/deterministic
flags, deterministic-algorithm state, matmul/cudnn TF32 flags, dtype, and device.

## Fixed comparable evaluation opportunities

EXP-010 used 19 unique evaluations. Faster steps can create extra tail epochs and
therefore extra chances in the maximum metric. Add `evaluation_count = 0` and
allow ordinary nonterminal checkpoint/dense-tail evaluation only while
`evaluation_count < 18`; always reserve one final evaluation when
`training_done`. Increment only after a completed evaluator call and assert at the
summary that exactly 19 unique epochs were evaluated, including the terminal
epoch. The four existing 20/40/60/70% checkpoints retain priority and validation
remains at most once per epoch.

This cap does not modify test batches, logits, loss, argmax, or model state. It
only suppresses surplus nonterminal tail looks. Dry-run accepted and candidate
epoch/progress traces from timing must both yield at least 18 natural nonterminal
opportunities, exactly 19 executed evaluations after the cap, and one terminal
look. If the candidate would yield fewer than 19, production is blocked rather
than filling opportunities with additional mid-epoch evaluations. No test result
may tune which epochs are kept.

## Initialization, parity, and layout propagation gates

Use disposable diagnostics only; production retains no hooks/profiler.

1. Independently construct seed-42 control/candidate models. Require identical
   parameter/state names, ordering, logical values, buffers, post-construction
   CPU/CUDA RNG hashes, and exactly 1,073,962 parameters. Layout conversion must
   consume no RNG or change a value when compared in logical contiguous order.
2. Require every four-dimensional Conv weight to be channels-last contiguous and
   channel-stride one; BN vectors and Linear tensors retain natural layouts. After
   one update, every Conv gradient and created SGD momentum buffer must preserve
   the parameter's channels-last stride.
3. Require a candidate `[128,3,32,32]` input stride `(3072,1,96,3)`, bitwise source
   values after conversion back to contiguous, and identical pointer across the
   forward-boundary no-op. An evaluator-like contiguous `[256,3,32,32]` CUDA input
   must incur exactly one boundary restride and produce finite `[256,10]` logits
   without mutating BN state.
4. Hook inputs/outputs of all 19 Conv and 19 BN operations plus nine block outputs.
   Every non-ambiguous 4-D activation at 32x32, 16x16, and 8x8 must remain
   channels-last. Inspect both stride-2 shortcut slice/pad tensors and post-adds;
   verify 1x1 pooled layout and unchanged `.view` produce the correct classifier
   feature values.
5. Profile warmed hard and probability-target full steps. Apart from the declared
   H2D/input restride and documented allocations, reject repeated
   `aten::contiguous`, `_to_copy`, clone, permute, or layout-repair calls between
   convolutions. Record operator names, shapes, strides, self-CUDA time, and
   conversion counts. Endpoint strides alone do not prove no hidden fallback.

Any value/RNG/optimizer-layout mismatch, broken view, propagation loss, or repeated
repair rejects the exact implementation. Do not patch individual operators with
manual permutations as a rescue.

## Exact-corpus numerical safety

Materialize and hash one immutable production corpus before either arm: 200 strong
post-transform batches (100 hard, 100 resolved CutMix) plus 64 weak hard batches.
Both fresh arms start from identical logical model/SGD/RNG state and consume exact
tensors/targets in the same order. No CIFAR test evaluation is allowed here.

- On reset hard and soft batches require finite logits/loss/gradients, logit and
  loss agreement at `rtol=1e-5, atol=1e-5`, nonzero-gradient cosine `>=0.99999`,
  per-parameter relative-L2 gradient error `<=1e-4`, and BN-buffer relative error
  `<=1e-4` after one update.
- Replay all 264 batches with accepted LR/momentum/decay. Require no nonfinite
  state, candidate-only >95% predicted-class concentration, missed/reordered
  corpus item, BN-counter disagreement, or RNG drift. Strong and weak terminal
  loss EMA must each be `<=1.10x` control; gradient/update-norm p95 ratios must
  remain in `[0.90,1.10]`.
- In eval mode compare at least 20 fixed non-test batches including N=256. Require
  finite logits, maximum absolute delta `<=1e-3`, normalized RMSE `<=1e-4`, loss
  delta `<=1e-4`, at least 99.9% argmax agreement, repeatability, and immutable
  state.

Long trajectories need not remain bitwise equal: different legal cuDNN FP32
reduction order is part of the net layout effect. Threshold failure blocks timing;
do not force another algorithm or relax tolerance.

## Profiler-backed fresh paired timing gate

After safety passes, confirm one idle 97,871-MiB H20 and run **seven**
counterbalanced fresh-process control/candidate pairs. Restore identical logical
model/optimizer state and backend flags. Each arm receives 100 untimed warmups and
at least 1,000 measured synchronized full steps with registered strong-hard,
strong-CutMix, and weak-hard paths weighted 40/40/20. Timing begins before pinned
nonblocking H2D and includes candidate layout conversion, target transfer,
zero-grad, forward, loss, backward, ordinary SGD, and final synchronize.

Also record CUDA-event transfer/forward/loss/backward/update components, iterator
wait in a production eight-worker loader probe, peak allocation, strides, and
profiler conversion counts. Separately time evaluator-like N=256 inference with
contiguous CUDA inputs entering the forward boundary (100 warmups, 500 forwards).

Authorize production only if all hold:

- weighted candidate/control full-step mean `<=0.9700`, every paired ratio `<1.0`,
  both trial-mean CVs `<=2%`, and candidate weighted p95 no slower than control;
- candidate combined forward+backward mean `<=0.9700x` control, confirming gain in
  the measured bottleneck rather than loader variance;
- `floor(26,898 / weighted_ratio) >=27,705` projected updates;
- candidate peak allocation `<650 MiB` and at most 32 MiB above control;
- evaluator mean ratio `<=1.10`, evaluator CV `<=2%`, exactly one boundary
  conversion per batch, and projected total wall time `<540s`;
- all parity/layout/profiler gates still pass, and timing-derived epoch traces pass
  the exact-19-evaluation simulation.

EXP-013 showed fresh pairs can overturn serial estimates; EXP-029 showed only
1.97% extra step cost is enough to fail exposure attribution. Symmetrically, a
stable gain below 3% is insufficient here because extra exposure has no validated
accuracy mechanism. Do not exclude restride, add candidate-only warmup, drop a
slow pair, or introduce precision/compiler/batch fallbacks.

## Production verification and verdict

Only after every gate passes, run the exact candidate once at seed 42 via
`uv run train.py > run.log 2>&1` on the sole idle H20. Require exit zero, 300.0
counted seconds, total below 600 seconds, finite standard summary, exactly
1,073,962 parameters, at least 27,705 updates, one 80% switch with eight workers
stopped, 45-55% strong CutMix, hard weak-tail targets, memory within the timing
gate, and exactly 19 unique evaluations including the final epoch.

Record switch accuracy versus 89.73%, first weak versus 93.16%, best/final
accuracy, final NLL versus 0.1934, strong/weak steps and image slots, epochs,
evaluation epochs, component timings, total time, and VRAM. Acceptance requires
all integrity/exposure conditions and `best_test_acc >=94.25%`.

- Timing below the 3% gain or any preflight failure: `invalid`, no scored run.
- Complete run below 94.25% with at least 27,705 steps: `no-improvement`; extra
  accepted-recipe updates did not improve generalization.
- Accuracy at/above 94.25% but below the registered exposure floor: timing-
  confounded/invalid for this hypothesis, not a channels-last exposure win.
- Passing accuracy, exposure, completion, runtime, and integrity: `improvement`.

No reroll, threshold relaxation, layout-boundary change, BF16, width increase,
evaluation substitution, or same-experiment combination is allowed.

## Risks and evidence limits

- Official channels-last gains emphasize reduced precision; FP32 CIFAR kernels
  and the input restride may erase or reverse any benefit.
- Option-A slicing/padding and residual adds may force hidden conversions even if
  Conv/BN endpoints appear channels-last.
- Different cuDNN algorithms alter rounding, so the result is a net layout run,
  not a bitwise counterfactual with only more steps. EXP-016 motivates strict
  numerical safety even though this candidate stays FP32.
- More updates alone may not help and can over-refine the same solution; EXP-030's
  lower train loss with worse NLL reinforces that exposure is not automatically
  generalization.
- The 19-look cap prevents max-metric inflation but can skip surplus late epochs;
  this is necessary measurement control and must be reported.
- A 0.10-point pass is ten test examples at one fixed seed and should not be
  overstated as a precise effect size.

## Sources

- `knowledge/references/pytorch-channels-last.md` and the official PyTorch
  channels-last tutorial linked there.
- `experiments/010/04-analysis.md` — accepted recipe and 19-look anchors.
- `experiments/013/04-analysis.md` — stage costs, fresh-pair and evaluation-count
  lessons.
- `experiments/016/04-analysis.md` — numerical safety precedent.
- `experiments/023/04-analysis.md` — paired timing-to-exposure accuracy on H20.
- `experiments/029/04-analysis.md` and prior channels-last proposals — small-cost
  exposure sensitivity and hardened layout/profiler design.
- `experiments/031/01-brainstorm.md`, `02-system-understanding.md`,
  `03-experiment-learnings.md`, and `04-results.tsv` — current context.
