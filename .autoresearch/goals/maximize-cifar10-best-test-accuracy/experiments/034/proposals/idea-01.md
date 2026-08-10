# Proposal: End-to-End FP32 Channels-Last Training

## Decision and testable hypothesis

Test a pure physical-memory-layout change on the accepted EXP010 recipe. Preserve logical NCHW shapes, batch 128, the width-2 postactivation ResNet-20, all 1,073,962 logical parameter values, FP32/default-TF32 arithmetic, ordinary momentum SGD, the 80% N1/M7 plus probability-0.5 alpha-1 CutMix phase, the hard weak tail, elapsed-time LR schedule, seed 42, and the immutable evaluator. Convert the initialized model's four-dimensional tensors and every transferred model input to `torch.channels_last` so supported CUDA convolution and BatchNorm kernels can propagate NHWC physical layout.

This directly targets the measured system limiter: accepted forward is 22.11% and backward is 75.46% of GPU-stage time, with convolution/BN backward dominating the latter; transfer, loss, SGD, and launch gaps together have little ceiling. Unlike batch scaling, channels-last does not reduce update count, change gradient-noise scale, or require LR scaling. Unlike an architecture or optimizer intervention, it preserves the logical forward graph and mathematical update rule. The exposure hypothesis is that the complete synchronized step, including NCHW-pinned-host to channels-last-CUDA transfer, becomes at least 3% faster. That projects at least `floor(26_898 / 0.97) = 27_730` optimizer steps in 300 counted seconds. The accuracy hypothesis is that this additional exposure lets the single fixed-seed run reach the current required threshold, obtained at execution as moving baseline plus 0.10 percentage points (currently 94.25% from 94.15%).

This is deliberately a two-link hypothesis: layout must accelerate this exact tiny FP32 workload, and extra same-batch-size exposure must improve generalization. Neither link is established locally. Official PyTorch evidence says CUDA Conv2d and BatchNorm support channels-last propagation, but its strongest performance examples use reduced precision and larger activations. A neutral or slower result on 32x32 FP32 tensors is therefore plausible and must be rejected before production.

## Exact `train.py`-only implementation

Preserve the accepted constructor and initialization order, then restride after ordinary device transfer and before optimizer construction:

```python
model = ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULTIPLIER).to(device)
model = model.to(memory_format=torch.channels_last)
num_params = sum(p.numel() for p in model.parameters())

optimizer = optim.SGD(
    model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)
```

The CPU constructor and Kaiming draw sequence remain bit-for-bit accepted. The second `.to` changes only strides/storage of eligible four-dimensional tensors and occurs before SGD can create state. It must consume no CPU or CUDA RNG and must preserve every logical parameter and buffer value exactly when compared after conversion to ordinary contiguous layout.

Replace only the existing training-input transfer, leaving the target transfer unchanged:

```python
inputs = inputs.to(
    device, non_blocking=True, memory_format=torch.channels_last
)
```

This line remains after `t0`, so the production timer charges both DMA and any restride. CPU datasets, transforms, forkserver workers, default collate/CutMix, and their NCHW tensors remain untouched. Moving the conversion before `t0`, preconverting cached batches, or changing worker output format would make the systems comparison invalid.

`prepare.py` cannot be edited and `Eval.evaluate()` transfers contiguous NCHW test tensors. Add one boundary normalization at the start of `ResNet.forward`:

```python
x = x.contiguous(memory_format=torch.channels_last)
```

For production training input this must be an allocation-free, identical-data-pointer no-op. For evaluator input it performs exactly one CUDA restride before the stem, after which layout propagates. Do not change the remaining stem, 19 convolutions/BNs/ReLUs, Option-A slice/pad shortcuts, residual additions, adaptive average pooling, existing `.view`, or classifier. In particular, do not “repair” individual fallback operators with permutations or extra copies after seeing profiling.

Faster epochs could otherwise add test-set looks to a maximum-over-checkpoints metric. Add `MAX_EVALUATIONS = 19` and a local `evaluation_count`. Keep the existing four elapsed checkpoints and dense-tail due condition, but allow a nonterminal evaluation only while `evaluation_count < MAX_EVALUATIONS - 1`; always allow the terminal evaluation, increment once inside the single combined evaluation block, and assert `evaluation_count == MAX_EVALUATIONS` before the final summary. In pseudocode:

```python
evaluation_due = checkpoint_due or dense_tail_due or training_done
evaluation_allowed = training_done or evaluation_count < MAX_EVALUATIONS - 1
if evaluation_due and evaluation_allowed:
    test_loss, test_acc = evaluator.evaluate(model, device)
    evaluation_count += 1
    ...
...
assert evaluation_count == MAX_EVALUATIONS
```

This preserves at-most-once-per-epoch behavior, reserves the nineteenth look for the terminal model, and makes a faster candidate conservative by suppressing surplus dense-tail looks rather than exploiting them. It does not change `Eval`, test data, batch size, loss, argmax, or ground truth. Simulate the state machine over the accepted 69-epoch history and projected 69-75-epoch histories before any scored run; each must yield four early looks, 19 unique evaluation epochs total, and exactly one terminal look. Do not expose intermediate accuracy to any gate or revise cadence after seeing it.

Do not combine autocast, TF32/backend changes, `torch.compile`, fused SGD, a batch-size or LR change, architecture edits, data-policy edits, or any other optimization. Record rather than force `torch.backends.cuda.matmul.allow_tf32`, `torch.backends.cudnn.allow_tf32`, `torch.backends.cudnn.benchmark`, `torch.backends.cudnn.deterministic`, deterministic-algorithm state, dtype, and PyTorch/cuDNN versions; accepted and candidate diagnostics must match all of them.

## Semantic and memory-format preflight

Use disposable ignored diagnostics only; production `train.py` must retain no hooks, profiler calls, synthetic inputs, or diagnostic counters beyond the evaluation-count guard.

1. In fresh seed-42 processes, construct an accepted model and a candidate model through the accepted CPU initialization path. Require identical parameter/buffer names, order, shapes, dtypes, count, and logical values. Hash CPU and CUDA RNG state immediately before and after conversion and require no change. Require conversion back to contiguous logical order to be bitwise equal to the accepted state.
2. Require all 19 four-dimensional Conv weights to be channels-last contiguous and to have channel stride one; one-dimensional BN state and the two-dimensional FC weight retain accepted logical layout/value. After a real hard and a real soft-target step, require every four-dimensional Conv gradient and SGD momentum buffer to be channels-last contiguous, finite, and attached to the expected parameter.
3. Require a candidate `[128,3,32,32]` training input stride of `(3072,1,96,3)`, unchanged shape/value, and the same data pointer before and after the forward-boundary call. Feed an evaluator-like contiguous `[256,3,32,32]` CUDA tensor in eval mode; require exactly one boundary allocation/restride, finite `[256,10]` logits, unchanged BN buffers, and no second conversion on repeated already-channels-last input.
4. With disposable forward/pre-forward hooks, inspect inputs and outputs of every Conv2d, BatchNorm2d, and BasicBlock, both stride-2 shortcut slice/pad outputs, and both post-add tensors. Every unambiguous four-dimensional activation at 32x32, 16x16, and 8x8 must remain channels-last. Explicitly verify that the ambiguous 1x1 adaptive-pool output and unchanged `.view` produce the accepted `[N,128]` classifier features and `[N,10]` logits.
5. On at least four immutable real production batches covering strong-hard, strong-CutMix, and weak-hard targets, compare untrained accepted/candidate logits using `torch.testing.assert_close(rtol=1e-3, atol=1e-4)`, logit cosine at least 0.99999, and relative logit L2 at most `1e-3`. Require relative loss error at most `1e-3`, aggregate nonzero-gradient cosine at least 0.9999, and relative aggregate gradient L2 at most `5e-3`. These tolerances allow legal cuDNN reduction-order differences while rejecting a wrong graph, wrong values, or a silent dtype change.

Abort before timing on any RNG drift, logical-value mismatch, unexpected parameter/state layout, broken residual/add/pool/view behavior, nonfinite tensor, tolerance failure, repeated boundary copy, or unsupported-path layout loss. No operator-specific repair is permitted inside EXP034; a failure answers that end-to-end channels-last is not cleanly supported by the accepted graph.

## Bottleneck and hidden-conversion profiling

Re-profile accepted and candidate full training steps on the sole idle H20 using fresh processes and identical pinned immutable batches. Warm cuDNN with at least 100 hard and 100 soft steps before collection. Use `torch.profiler` CPU+CUDA activities with shapes and a bounded active window, and retain a Chrome trace plus summarized operator/kernel tables in ignored experiment artifacts. Separate forward, CE, backward, and SGD ranges exactly as in the system-understanding probe.

For the accepted arm, confirm that model backward remains the dominant stage (at least 65% of synchronized GPU-stage time) and that Conv/BN backward kernels explain most of it; otherwise the 2026-08-06 diagnosis is stale and timing may proceed only after the discrepancy is explained. For the candidate, record Conv/BN forward/backward kernel names and times, activation strides at stage boundaries, and counts/self-CUDA time for `aten::contiguous`, `_to_copy`, `clone`, `copy_`, `permute`, and layout-transform kernels. The one declared training H2D/restride is allowed. No additional per-layer repair may appear, and candidate layout-transform CUDA time outside that boundary must be below 0.5% of full-step CUDA time. Candidate Conv+BN backward time should decrease; if total steps improve while this component does not, diagnose the actual source rather than claiming the proposed mechanism.

Profiling is explanatory and a hidden-conversion veto, not the throughput estimator: profiler overhead and kernel attribution are not used to project exposure. The fresh paired synchronized wall test below is the authorization gate.

## Immutable-corpus numerical trajectory gate

Materialize once from the accepted seed-42 loaders and persist a SHA-256-addressed CPU corpus before comparing layouts: the first 200 actual post-transform strong batches with their already-resolved hard/CutMix targets (require 40-60% soft targets, do not rebalance them), followed by 64 weak hard-target batches. Preserve order and exact float/target bytes, record shapes/dtypes, and prove all eight materialization workers shut down. This avoids the known forkserver error of assuming fresh processes replay worker augmentation from seed alone. No CIFAR-10 test evaluation is allowed during preflight.

Replay the corpus in two fresh processes from identical logical model/optimizer states under accepted LR 0.1, momentum 0.9, and all-parameter decay `1e-4`; the only arm difference is memory format and its charged input transfer. Record every step's loss, predicted-class histogram, logit RMS/max, aggregate gradient norm, aggregate parameter-update norm, BN counters, and parameter/momentum/BN finiteness. Record checkpoints at steps 1, 2, 3, 5, 10, 20, 50, 100, 200, and after the 64 weak records.

Reject candidate-only predicted-class concentration above 95%; any nonfinite parameter, buffer, gradient, loss, logit, or optimizer state; skipped/repeated records; unequal BN batch counters; terminal strong or weak loss EMA above 1.10x control; or candidate/control aggregate update-norm p95 outside `[0.90, 1.10]` and maximum outside `[0.85, 1.15]`. Also reject candidate/control logit-RMS ratio p95 outside `[0.90,1.10]` or any step outside `[0.75,1.25]`. Longer-run parameter equality is not required: legal cuDNN kernels can change rounding and the local benchmark does not force deterministic CUDA. These are broad collapse/implementation gates, not a demand for bitwise recurrence.

## Fresh paired end-to-end timing gate

Only after semantic, layout, profiling, and trajectory gates pass, verify exactly one idle NVIDIA H20 with approximately 97,871 MiB. Run seven fresh-process pairs with balanced order (`C-A`, `A-C`, alternating, with the seventh order preregistered before launch). Each process restores identical logical seed-42 model/optimizer state and backend flags, pre-pins the same immutable CPU corpus, performs at least 100 untimed full-step warmups, and then measures at least 1,000 complete synchronized steps. Do not reuse CUDA contexts across arms.

Timing begins before nonblocking pinned-host H2D and includes input layout selection/restride, target transfer, LR assignment, `zero_grad`, FP32/default-TF32 forward, hard or probability-target cross-entropy, backward, ordinary SGD, and final `torch.cuda.synchronize()`. Use the registered production mix—40% strong hard, 40% strong soft, 20% weak hard—in every measured arm, and report stratum as well as weighted totals. Capture synchronized wall time as the decision metric and CUDA events as diagnostic decomposition. Record mean, median, p95, CV, clocks/utilization, peak allocation/reservation, and all pairwise candidate/control ratios. Do not remove outliers post hoc.

Authorize one production run only if all of the following hold:

- geometric-mean and median candidate/control full-step ratios are both `<=0.9700`;
- the paired-ratio 95% bootstrap upper bound is `<=0.9850`, at least six of seven pairs favor the candidate, and each arm's trial-mean CV is `<=2.5%`;
- candidate weighted p95 is no slower than control p95, and no stratum is more than 2% slower;
- `floor(26_898 / geometric_mean_ratio) >= 27_730` projected fixed-budget steps;
- candidate peak allocation is below 4 GiB, no OOM/workspace retry occurs, and profiler conversion gates still hold;
- separate evaluator-like batch-256 timing with contiguous NCHW CUDA inputs, 100 warmups and at least 500 forwards per arm, has candidate/control mean `<=1.10`, CV `<=2.5%`, and a conservative complete-run projection below 540 seconds.

The 3% gate is load-bearing because exposure is the only proposed accuracy mechanism. A stable 1-2% gain is informative but insufficient to justify a scored run. Do not lower the threshold, exclude the charged transfer, change precision or batch size, force backend flags, or add another optimization as a rescue. Any externally contaminated timing pair detected from independent utilization/clock evidence invalidates the entire suite; at most the whole seven-pair suite may be restarted after documenting and clearing that infrastructure cause—never delete only an unfavorable pair.

## Fixed-seed production and acceptance

Immediately before production, query the moving baseline from `04-results.tsv` with the project index helper; do not rely on the value copied into this proposal. Confirm the experiment branch differs from the integration baseline only in tracked `train.py`, no completed/stale log remains, and exactly one H20 is idle. Then run exactly once at seed 42 with the required command and no reroll:

```bash
uv run train.py > run.log 2>&1
```

Supervise without streaming the log and kill at 600 seconds. Require exit zero; ten unique finite summary fields; exactly 300.0 counted training seconds; total below 600 seconds; 1,073,962 parameters; one 80% augmentation transition; all eight strong workers stopped; 45-55% strong CutMix; integer-only weak targets; no backend/dtype drift; 19 evaluation lines on 19 unique epochs with one terminal evaluation; and no evaluation more than once per epoch. Compare actual steps to 26,898 accepted and the registered 27,730 projection, switch accuracy to 89.73%, first weak accuracy to 93.16%, final NLL to 0.1934, peak allocation to 598.7 MiB, and total time to 330.7 seconds.

`best_test_acc >= moving_baseline + 0.10` with all hard protocol conditions is an improvement; one decimal test image corresponds to 0.01 point, so the comparison uses the printed hundredth-percent metric without hidden rounding. A valid lower result is no-improvement and is reverted without reroll. If production completes with fewer than 27,730 steps, the timing-to-exposure premise is falsified and must be reported even if accuracy improves; fewer than the accepted 26,898 steps specifically contradicts the systems mechanism, but metric validity still follows the goal's hard constraints rather than retroactively discarding a scored outcome. A crash, nonnumeric summary, extra/missing evaluation, worker/target/backend drift, wrong hardware, timing violation, or out-of-scope change is invalid.

## Risks and evidence limits

- **Tiny FP32 kernels may not benefit.** Official channels-last speedups are shape-, operator-, dtype-, and hardware-dependent; H20 tensor-core advantages are more naturally exposed by reduced precision, which this experiment forbids.
- **Option-A can break propagation.** Spatial slicing followed by channel padding, residual in-place addition, adaptive pooling, or `.view` can trigger a fallback or hidden repair even if Conv/BN support NHWC generally. The layout hooks and profiler veto make this observable.
- **Transfer can erase kernel savings.** Training begins with pinned NCHW host batches, so the candidate pays its layout selection inside every counted step. Timing a preconverted CUDA tensor would overstate production benefit.
- **Numerics are not bitwise invariant.** Different legal cuDNN algorithms/reduction orders can produce a different seed-42 trajectory. The experiment measures the net layout implementation, not a deterministic extra-steps counterfactual; immutable-corpus checks only veto gross divergence.
- **Exposure may not improve accuracy.** EXP013 found extra image throughput at larger batch but rejected its update tradeoff, and no local experiment proves that roughly 3% more same-batch-size updates raises this already noisy frontier. The full fixed-seed run is necessary.
- **Maximum-over-checkpoints is bias-prone.** Faster epochs would create extra dense-tail looks. The fixed 19-look cap is part of measurement integrity, not an accuracy technique, and may conservatively skip intermediate faster-candidate epochs.
- **One seed is weak causal evidence.** A bare +0.10 pass is protocol-valid but only ten additional correct test images and does not establish a population-average gain. Seed rerolls are forbidden.

## Sources

- `knowledge/references/pytorch-channels-last.md` and its linked official PyTorch memory-format tutorial: logical NCHW semantics, Conv2d/BatchNorm support, and operator-dependent fallback risk.
- `02-system-understanding.md`: 22.11% forward, 75.46% backward, 0.61% transfer, and 1.67% reset/update attribution on the accepted workload.
- `03-experiment-learnings.md` and `project-notes/project-insights.md`: fresh paired timing, exact worker-materialized corpora, nominal-overhead risk, and evaluation-count bias.
- `experiments/010/04-analysis.md`: accepted 94.15%, 26,898 steps, 19 unique looks, 89.73% switch accuracy, 93.16% first weak accuracy, 0.1934 final NLL, 598.7 MiB, and 330.7 seconds.
- `experiments/013/00-paired-timing.md` and `experiments/013/02-plan-review.md`: fresh-process timing and maximum-over-look integrity.
- `experiments/029/proposals/idea-02.md`: earlier deferred channels-last specification, tightened here using EXP029-033 protocol findings.
- `01-definition.md`, `04-results.tsv`, current accepted `train.py`, and immutable `prepare.py` evaluator implementation, read through EXP033.
