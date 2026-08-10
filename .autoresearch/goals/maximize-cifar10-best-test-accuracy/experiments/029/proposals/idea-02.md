# Proposal: FP32 / Default-TF32 Channels-Last Training

## Decision and hypothesis

Test a pure physical-layout intervention on the accepted EXP010 recipe. Keep logical tensors NCHW, batch 128, width-2 ResNet-20, FP32 parameters/activations, default backend and TF32 settings, ordinary momentum SGD, the complete N1/M7 + probability-0.5 alpha-1 CutMix curriculum, LR schedule, and fixed evaluator. Convert initialized Conv2d weights and every model input to `torch.channels_last`, allowing cuDNN to select NHWC kernels for the measured convolution/BN-dominated workload.

The system profile assigns 22.11% of GPU-stage time to forward and 75.46% to backward; channels-last can therefore add exposure without changing batch noise or optimizer decisions. The hypothesis is that end-to-end synchronized candidate steps are at least 3% faster, projecting at least 27,705 steps versus the accepted 26,898 in 300 counted seconds, and that the resulting run reaches `best_test_acc >= 94.25%` versus the 94.15% frontier. This is an exposure hypothesis, not an intrinsic accuracy mechanism; official guidance is strongest for reduced precision and larger tensors, so tiny 32x32 FP32/default-TF32 kernels may show no gain.

## Exact implementation and scope

Construction order is part of the intervention:

```python
model = ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULTIPLIER).to(device)
model = model.to(memory_format=torch.channels_last)
num_params = sum(p.numel() for p in model.parameters())
optimizer = optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
```

This preserves the accepted CPU constructor and Kaiming draw order, performs ordinary device transfer first, then restrides only 4-D tensors, and constructs the optimizer only after parameters have their final storage layout. Conversion must consume no CPU or CUDA RNG and must leave all logical parameter/buffer values bitwise identical when compared in contiguous logical order.

Inside the existing counted region, replace only the training input transfer:

```python
inputs = inputs.to(
    device, non_blocking=True, memory_format=torch.channels_last
)
```

Targets remain unchanged. CPU transforms, CutMix, workers, and their NCHW tensors remain unchanged; moving conversion before `t0` or into workers would make candidate work free or alter data behavior.

At the start of `ResNet.forward`, use:

```python
x = x.to(memory_format=torch.channels_last)
```

For training this must be a no-allocation, same-data-pointer no-op. It is required because the immutable `Eval.evaluate()` transfers contiguous NCHW test batches to CUDA; the boundary conversion makes evaluation use the same layout without editing `prepare.py`. Keep the remainder of the graph, including Option-A slice/pad shortcuts, adaptive average pooling, `.view`, and classifier, unchanged.

Do not enable autocast/BF16/FP16, compilation, fused SGD, larger batches, cuDNN benchmark/determinism changes, explicit TF32 changes, architectural changes, gradient centralization, or any timing rescue. Capture `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, benchmark/deterministic flags, deterministic-algorithm state, dtype, and device in both arms and require equality; “default-TF32” means preserving installed defaults, not forcing a preferred value.

## Initialization, layout, and operator preflight

Use disposable diagnostics only; production must not retain hooks or profiler work.

1. Reset seed 42 and construct accepted/candidate models independently. Require equal parameter names/order/count (`1,073,962`), equal logical parameters and buffers, and identical post-construction CPU/CUDA RNG-state hashes. Conversion must not advance either RNG.
2. Require each 4-D Conv weight to be channels-last contiguous with channel stride 1; 1-D BN and 2-D Linear tensors retain values/shapes. After one step, require every 4-D weight gradient and SGD momentum buffer to preserve channels-last layout.
3. Require a candidate `[128,3,32,32]` input stride `(3072,1,96,3)`, unchanged shape/values, and identical data pointer across the forward-boundary conversion. Feed a contiguous evaluator-like `[256,3,32,32]` CUDA tensor and require exactly one boundary restride, finite `[256,10]` logits, and unchanged BN buffers in evaluation mode.
4. Hook all 19 Conv outputs, all 19 BN outputs, and nine block outputs. Every non-ambiguous 4-D activation at 32x32, 16x16, and 8x8 must remain channels-last. Directly inspect both stride-2 shortcut slice/pad outputs and post-add tensors. Verify adaptive pooling and unchanged `.view` produce correct classifier features despite the ambiguous 1x1 spatial layout.
5. Profile warmed hard-target and probability-target full steps with shapes/stacks. Apart from the declared input restride/H2D and documented Option-A/pooling allocations, reject repeated `aten::contiguous`, `aten::_to_copy`, `clone`, or permutation repairs between convolutions. Record convolution algorithm/operator names, shapes, strides, self-CUDA time, and conversion counts for both arms; layout endpoints alone are insufficient because a later Conv can hide an intermediate fallback.

Any RNG drift, value mismatch, layout fallback, broken `.view`, optimizer-state mismatch, or repeated hidden repair is a no-go. Do not patch individual operators with permutations as a rescue.

## Exact-corpus numerical safety

Materialize once and hash an immutable production-distribution corpus before comparing arms: 200 post-transform strong batches with exact N1/M7 pixels and their resolved hard/CutMix targets, balanced near 50/50, plus 64 weak hard batches. Confirm all eight workers shut down. Both fresh arms must load the same tensors and begin from identical logical model/optimizer/RNG states; no CIFAR-10 test evaluation is allowed in preflight.

On matched initial hard and soft batches, require finite logits/loss/gradients, logits and loss within `rtol=1e-5, atol=1e-5`, and per-parameter nonzero-gradient cosine at least `0.99999` with relative-L2 error at most `1e-4`. Replay the entire corpus with accepted LR/decay/momentum. Record losses, class histograms, BN counters, gradient/update norms, parameter/state finiteness, and RNG hashes. Reject candidate-only predicted-class share above 95%, nonfinite state, skipped/repeated corpus items, BN-counter disagreement, terminal strong or weak loss-EMA above 1.10x control, or gradient/update-norm p95 outside `[0.90,1.10]` of control. Longer-run parameter bitwise equality is not required: different legal cuDNN FP32 reduction order is the experimental numerical effect.

## Fresh paired full-step timing gate

After safety passes, confirm one idle H20 and run seven alternating fresh-process control/candidate pairs. Each process restores identical logical model/optimizer state and backend flags, uses the same persisted CPU batches, performs 100 untimed warmups, then times at least 1,000 complete synchronized steps. Timing starts before pinned CPU H2D and includes candidate restride, target transfer, zero-grad, FP32/default-TF32 forward, hard-or-soft CE, backward, ordinary SGD, and final synchronize. Measure strong hard, strong soft, and weak hard paths in their registered 40/40/20 production weighting. Record pair means/medians/p95, CV, peak allocation, strides, and profiler conversion counts.

Authorize production only if all are true:

- weighted candidate/control mean ratio `<= 0.9700`, every paired ratio `<1.0`, and per-arm trial-mean CV `<=2%`;
- weighted candidate p95 no slower than control p95;
- `floor(26_898 / weighted_ratio) >= 27_705` projected steps;
- candidate peak allocation below 650 MiB and no more than 32 MiB above control;
- all numerical/layout/operator gates remain satisfied.

Separately time evaluator-like batch-256 inference with contiguous CUDA input entering the model boundary, 100 warmups and 500 forwards per trial. Require candidate/control mean `<=1.10`, CV `<=2%`, and a conservative projected total wall time below 540 seconds. A sub-3% training gain fails the systems premise even if statistically stable; do not enable reduced precision, change batch size, exclude restride time, lower the threshold, or combine another idea.

## Evaluator implications and production verification

Channels-last changes neither logical logits nor `Eval.evaluate()` ground truth, batch 256, loss, or argmax. It can, however, change both kernel numerics and epoch count. Because `best_test_acc` is a maximum and EXP010 used 19 evaluations, a faster run must not gain extra test-set looks. Add a protocol guard that preserves the four elapsed checkpoints, allows dense-tail epoch evaluations only while reserving one final look, and performs exactly 19 unique evaluations including the terminal epoch. Simulate accepted 69-epoch and plausible faster schedules: each must produce four early looks, at most once per epoch, exactly 19 total looks, and a terminal evaluation. This guard is measurement control only; do not use evaluator results in any preflight or tune the schedule after seeing accuracy.

After every gate passes, execute exactly once at seed 42 with `uv run train.py > run.log 2>&1`; no reroll. Require exit zero, one H20, ten unique finite summary fields, 300.0 counted seconds, total below 600 seconds, 1,073,962 parameters, at least 27,705 steps, memory within the timing gate, one 80% augmentation transition, eight strong workers stopped, 45–55% strong CutMix, hard weak targets, and exactly 19 unique evaluations including final. Compare switch accuracy to 89.73%, first weak to 93.16%, final/best to 94.15%, NLL to 0.1934, steps to 26,898, memory to 598.7 MiB, and total time to 330.7 seconds. Record actual image slots and strong/weak step counts.

`best_test_acc >=94.25%` with all integrity conditions is a valid improvement. A valid run below 94.25% is no-improvement and must be reverted without reroll. Accuracy above the gate but fewer than 27,705 steps remains a formal metric improvement only if all protocol conditions pass, but it falsifies the registered exposure attribution. Safety/layout/timing/evaluation-count failure is invalid and blocks production; fix only a demonstrable implementation defect within the exact scope. No threshold relaxation, alternate layout boundary, precision change, candidate combination, or same-experiment rescue is allowed.

## Evidence limits

Channels-last may lose on tiny FP32 convolutions because official headline gains emphasize reduced precision, input restride can erase kernel savings, and Option-A slicing/padding may trigger fallback. Different cuDNN algorithms also make the production trajectory numerically non-identical, so any result is the net implementation effect rather than a bitwise extra-exposure counterfactual. EXP013 showed that stable throughput must be established with fresh pairs and that more exposure alone is not known to improve generalization. Finally, the +0.10 gate is only ten test images under one fixed seed; a bare pass is protocol-valid but weak causal evidence.

## Sources

- `knowledge/references/pytorch-channels-last.md` and the linked official PyTorch memory-format tutorial.
- `experiments/013/04-analysis.md` and `experiments/013/00-paired-timing.md` for fresh-pair and evaluation-count lessons.
- `01-definition.md`, `02-system-understanding.md`, `03-experiment-learnings.md`, `04-results.tsv`, `train.py`, and `prepare.py`, read through EXP028.
