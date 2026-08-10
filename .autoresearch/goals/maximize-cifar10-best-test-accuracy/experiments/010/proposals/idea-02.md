# EXP-010 Proposal: CUDA BF16 Autocast for Width-2 Throughput

## Proposal

Run only the width-2 model's training forward pass and cross-entropy loss under CUDA BF16 autocast:

```python
optimizer.zero_grad()
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
loss.backward()
optimizer.step()
```

Keep model parameters, gradients, BatchNorm state, SGD momentum buffers, and weight-decay computation in FP32. Do not use `GradScaler`. Keep evaluation outside autocast so the unchanged `Eval.evaluate()` method runs the FP32 model exactly as it does for the accepted baseline.

The intervention targets fixed-time optimizer exposure, not memory capacity. On the accepted width-2 recipe, FP32 training completed 27,143 steps in 300 seconds and peaked at 93.55%. BF16 is useful only if the H20 executes enough eligible convolution work faster to add a meaningful number of full SGD updates without losing the accepted model's numerical behavior.

## Accepted Baseline and Local Diagnosis

The moving baseline is EXP-007:

- Best test accuracy: 93.55%; EXP-010 must reach at least 93.65%.
- Width-2 post-activation ResNet-20 with 1,073,962 parameters.
- Batch 128, 27,143 updates, 71 reported epochs, 300.0 counted seconds, 333.0 total seconds, and 598.7 MB peak allocation.
- N1/M7 RandAugment through 80% of counted time, followed by the deterministic crop/flip-only weak tail.
- SGD at LR 0.1, momentum 0.9, all-parameter coupled decay `1e-4`, and hard-label cross-entropy.
- Final three evaluations were essentially flat around 93.5%, so speed alone is not guaranteed to improve the late solution.

EXP-007 proved that additional representation capacity was worth a 29.2% update loss. EXP-008 and EXP-009 then bracketed weight decay: `5e-4` underfit the strong phase, while removing BN/bias decay fit harder without improving generalization. EXP-010 therefore preserves the exact accepted statistical recipe and asks whether the H20 can execute more of it in 300 seconds.

The baseline mean full-run step time is approximately `300 / 27,143 = 11.05 ms`. EXP-007's paired synthetic FP32 timing was 10.928 ms and predicted actual exposure within 2.5%, establishing that a paired local timing ratio is a useful feasibility signal.

## H20 Feasibility

The current node reports:

```text
GPU: NVIDIA H20
Memory: 97,871 MiB
Compute capability: 9.0
PyTorch: 2.9.1+cu128
CUDA runtime: 12.8
torch.cuda.is_bf16_supported(): True
```

Hopper-class compute capability 9.0 supports BF16 Tensor Core execution. The width-2 stage channels 32/64/128, batch 128, and fixed 32x32 shapes are multiples favorable to Tensor Core kernels. Convolutions dominate the wider model more than they did the original 0.27M-parameter baseline, making mixed precision more plausible now than before EXP-007.

There are important reasons not to assume a large gain:

- cuDNN FP32 convolution may already use accelerated TF32 paths on this environment.
- ResNet-20 remains shallow and launch-heavy; BatchNorm, ReLU, residual adds, pooling, optimizer work, transfers, Python, and the mandatory per-step synchronize do not all become twice as fast.
- Autocast performs eligibility checks and may cast/copy weights or activations. Its caching helps, but the context and casts have overhead.
- The first 3-channel stem and small late feature maps may not saturate Tensor Cores.

For these reasons, the proposal requires a paired local speedup gate rather than relying on peak H20 specifications.

Relevant primary documentation:

- PyTorch AMP documentation explains that autocast selects lower precision for eligible operations and keeps numerically sensitive operations in FP32: <https://docs.pytorch.org/docs/stable/accelerator/amp.html>.
- PyTorch's AMP recipe notes that convolutions and linear layers can be faster in FP16/BF16 while reductions often require FP32 range: <https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html>.
- NVIDIA's support matrix lists BF16 support for compute capability 9.0: <https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1040/pdf/TensorRT-Support-Matrix-Guide.pdf>.

## FP32 Master and Optimizer Semantics

The model must be constructed exactly as today with `.to(device)`, not `.to(torch.bfloat16)`. Its parameter storage remains FP32. `torch.autocast` temporarily chooses dtypes per eligible operation; it does not replace the FP32 leaf parameters with persistent BF16 master weights.

The intended precision boundaries are:

- **FP32 persistent state:** all model parameters, parameter `.grad` tensors, BatchNorm affine parameters, BatchNorm running mean/variance, SGD momentum buffers, and optimizer weight-decay arithmetic.
- **Autocast-selected forward/backward work:** eligible convolutions and linear operations use BF16; backward kernels use the dtype selected for their corresponding forward operations.
- **Numerically sensitive loss:** CUDA autocast's policy handles cross-entropy/reductions at the safe eligible precision; the returned scalar loss is expected to be FP32 and must be asserted in preflight.
- **FP32 evaluation:** evaluator calls occur after the autocast context has exited and use the same FP32 parameter/buffer storage.

After `loss.backward()`, gradients accumulated into FP32 leaf parameters should be FP32. Ordinary SGD then initializes/updates FP32 momentum buffers from those gradients, adds coupled `1e-4` decay to FP32 parameters, and writes FP32 updates. This avoids BF16 weight stagnation, where small late updates would round away if parameters themselves were stored in BF16.

No `GradScaler` is proposed. BF16 has the same exponent width/dynamic range as FP32 and is much less prone to underflow than FP16. A scaler would add state, unscale checks, potential skipped steps, and synchronized overhead to an intervention whose purpose is throughput. If unscaled BF16 produces non-finite loss or gradients in preflight, the candidate fails rather than silently adding a scaler.

## Exact `train.py` Scope

Starting from accepted commit `8faf0f3`, modify only `train.py`:

1. After selecting `device`, require a CUDA BF16-capable device:

   ```python
   if device.type != "cuda" or not torch.cuda.is_bf16_supported():
       raise RuntimeError("EXP-010 requires CUDA BF16 support")
   ```

2. Wrap only `outputs = model(inputs)` and `loss = F.cross_entropy(...)` in the BF16 autocast context shown above.
3. Keep `loss.backward()` and `optimizer.step()` outside the context, as recommended by AMP guidance.
4. Optionally print one startup line such as `Training autocast: cuda/bfloat16`; do not add per-step dtype checks or logging to the timed region.

Do not change:

- Model architecture, initialization, width multiplier, parameter count, or parameter storage dtype.
- Batch size, LR values, 80/20 elapsed-time schedule, momentum, decay, or optimizer parameter grouping.
- RandAugment strength/duration, weak transform, DataLoader, workers, switch lifecycle, or seed.
- Loss definition, labels, timer boundaries, per-step synchronization, evaluation cadence, evaluator, or summary schema.
- Memory format, TF32 settings, `torch.compile`, `zero_grad(set_to_none=True)`, fused optimizer flags, batch size, or GPU-resident augmentation.
- Evaluation precision: never surround `evaluator.evaluate()` with autocast.

This scope makes the full-run difference attributable to mixed-precision training plus the extra updates it enables.

## Mandatory Paired Preflight

Run all diagnostics from disposable `/tmp` scripts in fresh processes. Do not edit tracked files or consume the full training budget during proposal/planning. Confirm the only visible H20 is idle before each benchmark.

### Precision-state and numerical parity check

Create two identical width-2 models from one cloned state dict, with identical SGD optimizers and fixed pinned input/target batches. Run FP32 and BF16-autocast forward/backward paths without stepping, then verify:

- Every parameter and floating-point BatchNorm running mean/variance remains `torch.float32`; integer `num_batches_tracked` buffers remain integer.
- Every populated parameter gradient is `torch.float32`.
- BF16 loss is `torch.float32`, finite, and within 2% relative error of FP32 loss over at least 20 seeded random batches.
- Flattened BF16-versus-FP32 gradient cosine similarity is at least 0.99 on every batch; gradient norm ratio stays within `[0.90, 1.10]`.
- Logit cosine similarity after casting BF16 output to FP32 is at least 0.995; mean absolute logit error is reported.

Then perform one optimizer step for each path and assert every SGD momentum buffer is FP32 and finite. Run 200 additional disposable BF16 steps on a fixed seeded batch stream; require finite losses/gradients throughout, no loss spike above 2x the paired FP32 loss at the same step, and no skipped updates (there is no scaler).

These gates are deliberately tolerant of BF16 rounding but reject a materially different gradient field before a fixed-seed accuracy run.

### Synchronized throughput benchmark

Benchmark the exact code inside the accepted `t0`/`dt` interval, including pinned CPU-to-GPU nonblocking copies, `optimizer.zero_grad()`, forward/loss, backward, SGD step, and `torch.cuda.synchronize()`.

For each of FP32 and BF16:

1. Start from identical fresh model/optimizer state and a reusable pinned batch of shape `[128, 3, 32, 32]`.
2. Execute 100 untimed warm-up steps so cuDNN and autocast caches are warm.
3. Execute 500 synchronized timed steps; report aggregate mean, median, p95, samples/s, and peak allocation.
4. Repeat five paired trials, alternating order (`FP32,BF16` then `BF16,FP32`) to reduce thermal/order bias.
5. Use the median of the five aggregate trial means as `t_fp32` and `t_bf16`. Require trial coefficient of variation below 3% for each path.

Calibrate projected full-run exposure using the paired ratio and accepted actual steps:

```text
speedup = t_fp32 / t_bf16
projected_bf16_steps = floor(27,143 * speedup)
projected_bf16_samples = 128 * projected_bf16_steps
```

All throughput gates must pass:

- `speedup >= 1.15`, equivalently BF16 mean time at most 86.96% of paired FP32.
- `projected_bf16_steps >= 31,215`, at least 15% more than EXP-007.
- `projected_bf16_samples >= 3,995,520`, versus EXP-007's 3,474,304.
- BF16 p95 step time below FP32 aggregate mean, avoiding a gain driven only by a thin fast tail.
- Peak allocation below 1 GB and no unexpected allocation growth versus FP32 greater than 10%.

At the minimum gate, batch 128 still uses 390 batches per full epoch, so the run projects about 80 full-data equivalents, roughly 24,972 high-LR steps and 6,243 weak-tail steps. EXP-007 had about 69.6 full-data equivalents, 21,714 plateau steps, and 5,429 tail steps.

If the paired gate fails, do not run the full experiment and do not add channels-last, compilation, fused SGD, or a larger batch to rescue it. Those are separate candidates.

### Loader and total-wall guard

EXP-004 measured the N1/M7 loader at 165.5-175.8 batches/s. The 1.15x gate projects about 104 BF16 GPU steps/s, so the existing workers should retain comfortable headroom. Still benchmark at least 1,000 real strong batches and the accepted strong-to-weak switch at batch 128 in a fresh process. Require:

- Strong loader throughput at least 1.20x projected BF16 GPU batch rate.
- Weak loader throughput at least projected BF16 GPU batch rate.
- One switch under five seconds with all eight old workers stopped.
- Conservative projected total runtime below 540 seconds, including expected additional dense-tail evaluations.

## Fixed-Time Semantics

The existing timer starts after a batch is yielded and wraps input transfer, model/loss, backward, optimizer step, and synchronization. Autocast overhead and every BF16/FP32 kernel therefore count toward the same 300 seconds. Extra steps are legitimate accelerator throughput, not exempted work.

Compilation is not involved. No BF16 warm-up is moved into the full script outside the timer. cuDNN/autocast first-step effects remain counted exactly as FP32 first-step effects are counted today; the separate diagnostic warm-up exists only to obtain a stable go/no-go estimate.

The time-based schedule remains at 80/20. If BF16 is faster, it performs more updates at `lr=0.1` under N1/M7 and more updates in the weak cosine tail. This is the intended mechanism, but it also increases cumulative optimization and coupled decay exposure per wall-clock phase. At a 1.15x speedup, integrated LR/decay application rises about 15%; EXP-008's underfitting result makes this a real statistical risk even though the scalar remains `1e-4`.

More completed epochs also produce more dense-tail evaluations under the accepted policy. The metric definition permits one evaluation per epoch, and all evaluator calls remain outside training time but inside the 600-second total timeout. The loader/total-wall guard accounts for this.

## Hypothesis and Expected Impact

**Hypothesis:** CUDA BF16 autocast will reduce synchronized width-2 step time by at least 15%, increase full-run exposure from 27,143 to at least 31,215 updates without non-finite or materially misaligned gradients, and raise `best_test_acc` from 93.55% to at least 93.65% under unchanged FP32 evaluation.

Expected best accuracy is 93.50-93.85%, with 93.65-93.75% the plausible successful band if the throughput and parity gates pass. The upside is denser high-LR representation learning plus roughly 800 additional weak-tail updates at the minimum gate. The ceiling is modest because EXP-007's final trajectory was already flat; extra steps can refine the solution but do not introduce a new representation or regularizer.

A small regression is plausible. BF16 rounds convolution activations and backward signals, changes the exact optimization trajectory, and increases the number of high-LR/weight-decay applications. Mixed-precision noise might regularize, but it might also erase small residual corrections or destabilize BatchNorm/residual additions. The preflight rules out gross numerical mismatch, not a generalization shift.

## Risks

- **Insufficient H20 speedup.** FP32 may already use TF32 Tensor Cores, while the small model remains launch/normalization/synchronization-bound. This is the highest feasibility risk and is resolved before a full run.
- **BF16 mantissa precision.** BF16 has only seven explicit fraction bits; small activation or gradient differences can be rounded even though dynamic range is safe.
- **BatchNorm/residual sensitivity.** Repeated residual additions and BatchNorm statistics may accumulate quantization differences. Persistent running buffers remain FP32, but their observed batch inputs come from a mixed-precision path.
- **More updates are not automatically better.** EXP-007's late accuracy was flat. More high-LR steps could overfit or increase cumulative coupled decay rather than improve the optimum.
- **No gradient scaler.** This is correct for the BF16 range and keeps the test isolated, but any observed non-finite gradient is a hard failure rather than dynamically skipped.
- **Autocast policy/version dependence.** PyTorch 2.9.1 and cuDNN decide operation dtypes. Record versions and observed output/loss/gradient dtypes in preflight; do not assume every op is BF16.
- **Evaluation/train precision mismatch.** Training is mixed precision while scoring is FP32. This is intentional to preserve the evaluator, but a model may adapt to BF16 rounding in a way that does not transfer perfectly to FP32 inference.
- **Loader bottleneck and wall time.** A large speedup can approach the worker-side RandAugment ceiling; loader waits are outside counted training but inside total runtime.
- **Seed-stream divergence.** Faster training consumes more shuffled/augmented batches before 300 seconds, inherently changing the later CPU RNG stream. Seed remains 42 and no reroll is permitted.

## Confound Controls

- Start from accepted commit `8faf0f3` with all-parameter decay restored to `1e-4`.
- Change only the CUDA BF16 capability check, training forward/loss autocast context, and optional single startup log in `train.py`.
- Preserve exact model, optimizer, loader, transforms, schedule, phase switch, evaluator, seed, and timing code.
- Keep model/evaluator FP32 and verify optimizer state dtype explicitly in preflight.
- Do not combine AMP with channels-last, compilation, larger batch, scaler, fused optimizer, or `set_to_none=True`.
- Run one fixed-seed full experiment only after all preflight gates pass; do not retry a valid run in FP16 or with a scaler.

## Verification Plan

No full training run is part of proposal development. If selected:

1. Confirm the moving baseline is 93.55% and improvement requires at least 93.65%.
2. Confirm exactly one idle H20, compute capability 9.0, approximately 98 GB VRAM, and BF16 support in the installed PyTorch build.
3. Pass every paired numerical-parity, FP32-state, synchronized-throughput, loader, lifecycle, and total-wall gate above.
4. Verify the tracked diff modifies only `train.py` and matches the exact autocast scope; no evaluator call may occur inside autocast.
5. Run syntax compilation, Ruff, pre-commit, and assert 1,073,962 FP32 parameters and 390 loader batches.
6. Remove stale logs and execute once as `uv run train.py > run.log 2>&1` under the 600-second supervisor.
7. Require exit zero, one complete finite ten-field summary, about 300 counted seconds, total below 600 seconds, and unchanged parameter count.
8. Require exactly one `randaugment->base` switch near 80.0%, eight stopped workers, unique evaluation epochs, and terminal evaluation aligned with the summary epoch.
9. Require at least 31,215 actual steps and 3,995,520 presented samples for throughput mechanism success. Record actual speedup against EXP-007, epochs, switch step, tail updates, peak VRAM, and total time.
10. Require `best_test_acc >= 93.65%` for an improvement verdict. Compare strong switch accuracy/loss EMA, first weak checkpoint, tail slope, best/final gap, and final NLL against EXP-007.
11. Remove `run.log` after analysis.

## Decision Rules

- **Preflight no-go:** speedup below 1.15, projected steps below 31,215, dtype-state failure, poor gradient alignment, non-finite values, or total-time risk means do not run the full experiment.
- **Accept:** accuracy at least 93.65%, all integrity checks pass, and at least 31,215 updates complete. BF16 autocast becomes part of the accepted width-2 recipe.
- **Accuracy failure with throughput success:** reject BF16 for this statistical recipe. More updates or mixed-precision noise did not improve top-1; do not add FP16/scaling post hoc.
- **Accuracy gain below the throughput gate:** the metric may formally improve, but the proposed throughput mechanism is unsupported. Review timing integrity and report BF16 as a numerical intervention rather than an efficiency success.
- **Numerical failure:** non-finite loss/gradient, FP32-state mutation, or gross trajectory divergence is invalid/no-go. Revert to pure FP32; do not silently enable a scaler.
- **Runtime, lifecycle, scope, or evaluator-precision failure:** invalid. Revert to accepted EXP-007 and diagnose without changing seed or combining optimizations.
