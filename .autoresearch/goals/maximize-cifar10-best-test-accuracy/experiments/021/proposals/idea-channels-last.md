# Proposal: FP32 Channels-Last Training and Evaluation

## Decision

Test a pure physical-memory-layout change on the complete accepted EXP-010 recipe. Keep logical tensor dimensions `[N,C,H,W]`, every model/data/optimizer/schedule value, and FP32 numerics. Store convolution weights and 4D activations in `torch.channels_last` (NHWC physical stride order) and require a measured fixed-budget speedup before a production run.

The exact production changes are:

```python
def forward(self, x):
    out = x.to(memory_format=torch.channels_last)
    out = F.relu(self.bn1(self.conv1(out)))
    # remainder unchanged
```

```python
model = ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULTIPLIER).to(
    device=device,
    memory_format=torch.channels_last,
)
```

```python
inputs = inputs.to(
    device,
    non_blocking=True,
    memory_format=torch.channels_last,
)
```

The training-loop conversion occurs inside the existing counted `t0` region. The model-boundary conversion is a no-allocation no-op for already channels-last training inputs and converts the fixed evaluator's contiguous CUDA inputs without modifying `prepare.py`. Do not use `permute`; PyTorch channels-last preserves NCHW logical shape and represents NHWC storage through strides.

## Hypothesis

The H20 spends 97.6% of measured GPU-stage time in model forward/backward, principally 19 Conv/BN/ReLU blocks. Official PyTorch guidance recommends converting both model and every input so cuDNN can propagate channels-last and avoid repeated layout repairs. NVIDIA notes that Tensor Core convolution performance relies on NHWC layouts; current FP32 convolution may use TF32-capable kernels under unchanged backend defaults.

**Hypothesis:** full FP32 channels-last propagation will reduce synchronized weighted strong/weak step time by at least 3%, increase fixed-budget exposure from 26,898 to at least 27,705 updates without changing the accepted stochastic/data policy, and reach at least 94.25% best test accuracy, with a point prediction of **94.28%**.

This is primarily a systems/exposure hypothesis, not a new representation method. At the 3% gate it adds roughly 807 updates, 103,000 presented images, and about two dataset passes. Because EXP-010 finished at its best, additional high-LR exploration and weak-tail refinement may help. They may also be redundant or harmful; exposure alone has not been causally validated here.

## Exact Scope

Preserve:

- width-2 postactivation ResNet-20, Option-A shortcuts, global average pooling, classifier, and exactly 1,073,962 parameters;
- seed 42, constructor draw order, logical parameter values, FP32 dtype, loss, gradients, optimizer state, and evaluation dtype;
- batch 128, N1/M7, alpha-1 CutMix probability 0.5 through 80%, and hard weak crop/flip tail;
- ordinary SGD momentum 0.9, coupled all-parameter decay `1e-4`, and one optimizer group;
- `lr=0.1` through 80%, the `0.01` step, cosine to `1e-4`, timer, synchronization, maximum steps, and summary;
- DataLoader/collator worker-side NCHW transforms and RNG, worker lifecycle, and fixed `Eval.evaluate()`.

The only protocol guard added alongside the layout change is an explicit cap of
19 production evaluations, equal to EXP-010. Preserve the four existing budget
checkpoints. During the weak tail, evaluate at epoch boundaries only while fewer
than 18 looks have occurred, and always reserve/perform the nineteenth look at
`training_done`. For EXP-010's original 69-epoch trajectory this is behaviorally
unchanged; if channels-last completes an extra epoch, it skips the penultimate
tail look instead of increasing the maximum-selection opportunity count.

Do not add autocast/BF16/FP16, compilation, larger batches, cuDNN benchmark/determinism/TF32 flag changes, fused SGD, graph changes, data conversion in workers, or any accuracy mechanism. Do not combine channels-last with ECA or another candidate as a timing rescue.

## Why Conversion Occurs at These Boundaries

Worker transforms and CutMix remain ordinary contiguous CPU NCHW so their values, region semantics, target formats, and RNG stream stay unchanged. The combined `Tensor.to(device, non_blocking=True, memory_format=...)` performs device/layout transfer inside counted training work. Moving the restride into workers or before `t0` would make candidate-specific work free and change collator behavior.

`prepare.py` transfers test tensors to CUDA without a memory-format argument and cannot be modified. The single `x.to(memory_format=...)` at the model boundary therefore guarantees channels-last evaluation inputs. On the training path, assert this call returns the same storage pointer because the input already has the required strides.

Convert the model after normal seed-42 construction and before optimizer construction. No random number is consumed; logical values remain identical while 4D Conv weight strides change. Building the optimizer afterward ensures its parameter references and lazily created momentum buffers use the final layout.

Use `.to(memory_format=torch.channels_last)`, not `.contiguous(...)`, because official PyTorch documentation notes ambiguous dimensions such as `H=W=1`; explicit `to` assigns meaningful channels-last strides. Retain the current pooled `.view` only after a functional gate proves it works with the installed version.

## Expected Graph Behavior

The accepted graph is favorable to propagation: Conv2d, BatchNorm2d, ReLU, pointwise residual addition, and padding support or preserve channels-last. Same-shape shortcuts already share the input layout. At transition blocks, `[:, :, ::2, ::2]` creates a strided shortcut view and `F.pad` must restore a channels-last dense result before addition. Adaptive average pooling creates `[N,128,1,1]`, whose layout is formally ambiguous, but the current flattening view should still produce logical `[N,128]` classifier input.

Official PyTorch guidance warns that an unsupported operator may silently return contiguous output, causing downstream conversions. Passing correctness is therefore insufficient: preflight must prove propagation and profile conversions. A convolution can restore channels-last after an accidental fallback, hiding intermediate overhead from endpoint-only checks.

## Layout and Correctness Gates

Run disposable CPU and H20 diagnostics with hooks; add no hooks or layout logs to production:

1. Construct control/candidate from reset seed 42. After viewing candidate tensors in contiguous order, require every parameter and buffer value bitwise equal, identical parameter count/order, and identical post-construction CPU/CUDA RNG states.
2. Require every 4D Conv weight to be channels-last contiguous with channel stride one; 1D BN and 2D Linear tensors remain semantically unchanged. After one step, require every 4D weight gradient and corresponding SGD momentum buffer to preserve channels-last.
3. For `[128,3,32,32]`, require candidate input stride `(3072,1,96,3)`, unchanged logical shape, and same data pointer before/after the forward-boundary no-op conversion.
4. Hook all 19 Conv outputs, all 19 BN outputs, and all nine block outputs. Every 4D output at 32x32, 16x16, and 8x8 must report `is_contiguous(memory_format=torch.channels_last)`. Directly inspect both transition slice/pad paths and require the padded shortcut and post-add result to be channels-last.
5. Require adaptive pooling plus the unchanged `.view` to produce finite `[128,10]` logits without copying/reordering classifier features incorrectly.
6. Feed contiguous CUDA inputs through the exact model boundary used by `Eval.evaluate`; require conversion to channels-last, finite logits, and the same class dimension. Verify repeated evaluation-mode calls do not mutate BN buffers.
7. Compare paired initial FP32 control/candidate hard and probability-target logits, losses, input gradients, and parameter gradients. Different cuDNN algorithms/reduction order need not be bitwise equal; require finite values, logits/loss `rtol<=1e-5, atol<=1e-5`, gradient relative-L2 `<=1e-4`, and cosine similarity `>=0.99999` for every parameter tensor with nonzero gradient.
8. Run at least four identical optimizer steps on persisted hard/soft batches. Require finite state, unchanged RNG, no class concentration above 95% candidate-only, and logical parameter/state alignment to the expected diverging FP32 tolerance; this is a numerical sanity gate, not an accuracy proxy.
9. Profile a warmed full step. Apart from the declared input transfer/restride, reject any visible `aten::contiguous`, `aten::_to_copy`, or clone attributable to intermediate 4D activation layout repair. Document unavoidable Option-A/pooling copies separately rather than silently accepting repeated conv-to-conv conversions.
10. Compile, Ruff, pre-commit, scope, worker lifecycle, target-format, and static evaluator-call checks must pass.

Any failed layout, numerical, RNG, optimizer-state, or hidden-conversion gate is a no-go. Do not patch individual layers with ad hoc permutations or relax precision after observing failure.

## Paired Timing and Exposure Gate

Confirm one idle 97,871 MiB H20. Persist a fixed set of post-transform CPU batches before either arm so control/candidate see identical N1/M7/CutMix values and hard/soft targets. Run **seven alternating fresh-process pairs** to control H20 algorithm/cache drift. For each arm:

1. restore identical logical model/optimizer state and unchanged backend flags;
2. run 100 untimed warmup steps so cuDNN selects and warms layout-specific kernels;
3. time at least 2,000 complete synchronized steps, including pinned CPU H2D, candidate layout conversion, zero-grad, FP32 forward/loss/backward, SGD, and final synchronization;
4. measure strong hard/soft mixture and weak hard batches separately;
5. record mean, median, p95, CV, peak allocation, actual input/weight/output strides, and profiler conversion counts.

Compute an 80/20 weighted candidate/control mean ratio from strong and weak paths. Require:

- weighted ratio `<=0.9700`, a material at least 3.09% throughput gain;
- every one of the seven paired weighted ratios `<1.0` and CV of trial means `<=2%` per arm;
- candidate weighted p95 `<=1.0x` control p95;
- projected steps `floor(26,898 * control_mean / candidate_mean) >=27,705`;
- peak allocation `<650 MiB` and no more than 32 MiB above control;
- no intermediate conversion veto and finite state throughout.

Benchmark evaluation separately with the fixed evaluator-like batch 256, contiguous CUDA input entering the boundary conversion, 100 warmups, and 500 forwards per trial. Require candidate/control inference mean `<=1.10`, CV `<=2%`, and conservative total-wall projection below 540 seconds with exactly 19 projected evaluation passes.

The 3% gate is deliberately stronger than mere non-regression. Layout has no direct accuracy mechanism; a sub-3% gain does not justify spending the one fixed-seed accuracy run on ordinary numerical trajectory variation. If the gate misses, retire this exact FP32 layout point. Do not increase batch size, enable autocast, move conversion outside the timer, or reduce evaluations to rescue it.

## Evaluation-Cap Interaction

Keep the single existing `if checkpoint_due or dense_tail_due or training_done` evaluator branch, but add an `evaluation_count` guard: four early elapsed-budget checkpoints remain unchanged; weak-tail epoch boundaries are eligible only while `evaluation_count < 18`; `training_done` always triggers the reserved final look. No preflight uses the CIFAR-10 test evaluator. Production therefore performs at most once per epoch and exactly 19 total looks on a complete run, matching EXP-010 even if faster steps add an epoch.

This guard prevents a systems speedup from improving `best_test_acc` partly by increasing the number of sampled checkpoints. It must be tested against a simulated EXP-010 69-epoch schedule and a faster 70-epoch schedule: both yield 19 unique evaluation epochs, both include the terminal epoch, and the 69-epoch control schedule is unchanged. Do not evaluate a contiguous control model in production.

The model-boundary conversion makes each fixed evaluator pass test the same channels-last model state without changing ground-truth data, logits semantics, or `Eval.evaluate()` source. `best_test_acc`, final loss, and prediction count remain the fixed harness outputs.

## Production Success and Veto Criteria

After every gate passes, run once at seed 42 with required `run.log` redirection and no retry.

Require exit zero, ten finite unique summary fields, 300.0 counted seconds, total below 600 seconds, 1,073,962 parameters, at least 27,705 steps, memory within gate, one switch near 80%, eight workers stopped, 45-55% strong CutMix, hard weak targets, and exactly 19 unique evaluation epochs including the terminal epoch.

Formal improvement requires `best_test_acc >=94.25%` over EXP-010's 94.15%. Compare switch 89.73%, first weak 93.16%, final/best 94.15%, NLL 0.1934, 26,898 steps, 598.7 MiB, 330.7 seconds, and 19 evaluations. Record actual images processed and strong/weak update counts to test the exposure mechanism.

- **Accuracy and all integrity gates pass:** improvement; accept channels-last as part of the recipe.
- **Valid accuracy below 94.25% despite exposure floor:** no-improvement; more FP32 updates did not improve this trajectory. Revert without reroll.
- **Accuracy pass below 27,705 steps:** formal metric clears the goal, but the registered exposure mechanism failed; do not attribute the gain to speed.
- **Layout fallback, numerical safety failure, timing miss, wall projection miss, duplicate evaluation, crash, or timeout:** veto/invalid as appropriate. Fix only implementation defects while preserving exact scope; no fallback layout combination.

## Risks and Evidence Limits

- **FP32 kernel ceiling:** official large-model gains often emphasize FP16/Tensor Cores or different hardware; tiny CIFAR convolutions may already use efficient NCHW/TF32 kernels.
- **Input restride cost:** every CPU NCHW batch must reach NHWC storage inside counted work; conversion can erase convolution gains.
- **Unsupported-layout fallback:** slice, pad, pooling, or another operator may cause hidden contiguous conversions between convolutions.
- **Small-kernel behavior:** ResNet-20 has shallow, small spatial kernels; launch overhead and algorithm choice can dominate theoretical layout benefits.
- **Numerical divergence:** different cuDNN kernels alter FP32 reduction order. A gain/loss is the net layout implementation, not a mathematically identical extra-exposure counterfactual.
- **Exposure may not help:** accepted accuracy is generalization-limited; EXP-013 showed throughput alone was not enough to authorize larger batches, and accepted final-equals-best does not prove extra passes raise the ceiling.
- **Evaluation cadence displacement:** if speed adds an epoch, the fixed 19-look guard skips one penultimate weak-tail look to preserve a terminal evaluation; a narrow transient peak at the skipped epoch would be unseen.
- **Wall-time growth:** extra weak epochs add excluded-from-training but included-in-wall evaluator passes.
- **Single-seed resolution:** the required 0.10 point is ten CIFAR-10 images; a bare pass is protocol-valid but weak causal evidence.

## Sources

- [PyTorch Channels Last Memory Format tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html): convert model and every input, stride semantics, propagation, supported operators, and fallback warning.
- [PyTorch Conv2d weight memory-format documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.convert_conv2d_weight_memory_format.html): cuDNN NHWC strategy and conversion-overhead caveats.
- [NVIDIA Optimizing Convolutional Layers guide](https://docs.nvidia.com/deeplearning/performance/pdf/Optimizing-Convolutional-Layers-User-Guide.pdf): NHWC/Tensor Core convolution guidance.
- `TASK.md`, `prepare.py`, `train.py`, and goal definition/system understanding/results/learnings through EXP-020.
