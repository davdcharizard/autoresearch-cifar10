# Proposal: Time-Normalized Wide-Batch BF16 ResNet-20

## Summary

Replace the baseline's step-number LR milestones with an LR schedule driven by measured training-time progress, while moving the tiny ResNet-20 workload to a better H20 operating point: widen every stage by 2x, use batch size 512, keep parameters in FP32 but run convolutional work under BF16 autocast, and use channels-last memory format. The concrete first trial uses a 5% time warmup to peak LR 0.20 followed by cosine decay to 1% of peak over the remaining 95% of the 300-second training budget.

This is one coupled intervention. The schedule makes every run spend meaningful time learning, refining, and converging regardless of how many updates the new kernel mix completes. The wider model converts otherwise idle GPU capacity into representation quality, while the larger batch and BF16/channels-last execution make that extra capacity affordable within the fixed training time.

## Baseline Diagnosis

The parent BASE achieves `best_test_acc = 91.51%` with 34,435 updates over 89 epochs and only 330.1 MiB peak allocated VRAM. At batch 128, that is about 4.41 million presented examples, 8.71 ms of measured training work per update, and 14.7k images/s on average.

The current `MultiStepLR` milestones are 32,000 and 48,000 steps. Under the observed rate, the first drop occurs around 279 seconds, or 92.9% of the 300-second training budget, and the second milestone is unreachable. Thus the effective schedule is approximately 93% at LR 0.1 and 7% at LR 0.01, with no genuine final convergence phase. Any throughput or architecture change would shift these phase boundaries again because the scheduler is indexed by update count rather than the resource the benchmark fixes: elapsed training time.

The memory result is equally diagnostic. Physical GPU 0 is an NVIDIA H20 with 97,871 MiB and compute capability 9.0, while the baseline allocates about 0.34% of that memory. ResNet-20's 272k parameters and batch 128 provide too little work per launch for this GPU. The first experiment should spend some of that unused capacity on a stronger model and more efficient, larger kernels rather than merely trying to execute more tiny FP32 updates.

## Mechanism

### Wall-clock-aware optimization phases

Compute LR from `p = min(total_training_time / TIME_BUDGET_S, 1)` before every update:

```python
def learning_rate(progress):
    if progress < 0.05:
        return PEAK_LR * (0.10 + 0.90 * progress / 0.05)
    cosine_progress = (progress - 0.05) / 0.95
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return PEAK_LR * (0.01 + 0.99 * cosine)
```

This starts at 0.02, reaches 0.20 after about 15 training seconds, then decays smoothly to 0.002 near 300 seconds. Warmup limits the transient created by batch 512 and the wider network. Cosine decay gives the run a long middle optimization phase and a real low-LR refinement phase, independent of achieved steps/s. Unlike an epoch schedule, it also remains stable if batch size changes the number of updates per epoch.

### Better H20 utilization with useful capacity

- Set `BATCH_SIZE = 512`. This makes each launch four times larger, reducing the kernel-launch-dominated character of 32x32 convolutions. It is large enough to improve occupancy but still gives 97 full updates per CIFAR-10 epoch, avoiding the very low update count of batch 1024 or larger.
- Add a `WIDTH = 2` channel multiplier and use stages 32/64/128 instead of 16/32/64. The classifier input changes from 64 to 128. Convolutional parameters grow by roughly 4x, to about 1.1M, which is still tiny for the H20 but offers a direct accuracy lever rather than spending all throughput gains on repeated passes through the same under-capacity model.
- Move the model and image batches to `torch.channels_last`. Convolutional weights remain ordinary trainable FP32 parameters; only memory layout changes.
- Wrap forward pass and cross-entropy in `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Keep the optimizer state and master parameters in FP32. BF16 does not require `GradScaler`, avoids FP16's narrower exponent-range risk, and is natively supported on compute capability 9.0.
- Enable `torch.backends.cudnn.benchmark = True`, since all training images have the same 32x32 shape and the larger batch makes kernel selection more consequential.
- Preserve SGD momentum 0.9, weight decay `1e-4`, existing augmentation, shortcut semantics, seed 42, and one evaluation at each epoch boundary. Keeping these fixed makes the proposal's mechanism interpretable.

Do not add `torch.compile` in this first trial. Compilation is lazy, the first training graph is reached after the training timer starts, and alternating train/eval modes can create additional graph work. For this small network, there is no measured evidence that saved Python overhead will repay compilation and guard/recompile costs inside the charged 300 seconds. It remains an ablation only after the batch/width operating point is known.

## Concrete Implementation

1. Import `math`; add `WIDTH = 2`, `BATCH_SIZE = 512`, `PEAK_LR = 0.20`, `WARMUP_FRACTION = 0.05`, and `MIN_LR_RATIO = 0.01`. Remove `MAX_STEPS` as a binding limit, or set it high enough that time remains the sole normal termination condition.
2. Parameterize `ResNet`'s stem and stages from `base_channels = 16 * WIDTH`: stem/base, base/2x base, and 2x/4x base, with `fc` consuming `4 * base_channels`.
3. Set `torch.backends.cudnn.benchmark = True`; create the model with `.to(device, memory_format=torch.channels_last)`.
4. Initialize SGD at `PEAK_LR * 0.10`. Delete `MultiStepLR`. Immediately before each forward pass, set every optimizer param group's LR from the time-progress function above.
5. Transfer inputs with `inputs.to(device, non_blocking=True, memory_format=torch.channels_last)` and targets normally. Run model forward and cross-entropy under BF16 autocast, then use ordinary `loss.backward()` and `optimizer.step()`.
6. Retain the existing `torch.cuda.synchronize()` timing boundary so `total_training_time` continues to measure completed GPU training work. Log `p` and the calculated LR. Stop when the accumulated training time reaches `TIME_BUDGET_S`; never validate inside an epoch more than once.
7. Evaluate through the unchanged `Eval.evaluate`. Autocast need not wrap evaluation; the model parameters remain FP32 and the evaluator remains ground truth.

## Expected Benefit

The minimum success condition is `best_test_acc >= 91.61%`. A reasonable first-run target is 92.0-92.8%.

The highest-confidence gain is schedule repair: the baseline currently has only about 21 seconds below LR 0.1 and never reaches its intended second decay. Smooth time normalization ensures the last part of every run actually converges. Width 2 supplies additional representational headroom that CIFAR-10 can use, while batch 512 plus BF16/channels-last should recover much of the extra compute cost by issuing denser tensor-core-friendly convolution work. Even if update count falls, the experiment can expose a similar or larger number of examples to a materially stronger model and should finish at a much lower effective LR.

The expected peak memory remains only a few GiB: activation memory scales approximately with batch times width (about 8x before BF16, about 4x after BF16), while parameter memory grows about 4x. This remains far below 97,871 MiB and is justified by a direct quality hypothesis.

## Evidence

- `train.py` and the BASE metrics provide direct benchmark evidence for the schedule mismatch and extreme underutilization: 34,435 observed steps cannot reach the 48,000 milestone, and 330.1 MiB is negligible on the available H20.
- *Time Matters in Regularizing Deep Networks* argues that training interventions have phase-dependent effects and that late interventions cannot repair poor early dynamics. Its experiment distillation at `experiments/001/papers/time-matters-regularization.md` supports allocating deliberate early and late phases rather than allowing hardware-dependent step throughput to determine them accidentally.
- The mixed-sample survey at `experiments/001/papers/mixed-sample-analysis.md` identifies augmentation as an effective regularizer, but that is orthogonal to the present bottleneck and introduces another variable. It is a good later branch after establishing a sound time schedule and compute operating point.
- `experiments/001/papers/regmixup.md` explicitly notes the extra forward-work tradeoff under this fixed-time benchmark, while `experiments/001/papers/shakedrop.md` notes higher risk for a shallow network and short schedule. These make a conventional width increase plus efficient arithmetic a more defensible first capacity experiment than RegMixup's extra clean/mixed objectives or stochastic residual scaling.
- `experiments/001/papers/stackmix.md` reports gains from complementary mixing but changes spatial geometry and may reduce throughput, another reason to defer it until the basic device operating point is measured.

## Risks and Mitigations

- **Large-batch generalization or under-optimization:** batch 512 provides 4x fewer updates per presented example. Peak LR 0.20 uses conservative square-root scaling from the baseline instead of linear scaling to 0.40, and warmup limits instability. If training loss remains high late, batch 256 is the first fallback.
- **Width costs more throughput than AMP recovers:** width 2 is roughly 4x convolutional FLOPs. The proposal should record examples/s, steps, epochs, and parameter count. If presented examples collapse below the baseline's 4.41M with no accuracy gain, keep the time schedule and BF16 path but revert to width 1 or use width 1.5 with integer channels 24/48/96.
- **BF16 numerical or kernel regression:** Hopper supports BF16, but tiny CIFAR kernels may not all speed up. Compare achieved images/s and loss finiteness. If BF16 is neutral or slower, retain batch/width/schedule and disable autocast; channels-last should also be separable in an ablation.
- **BatchNorm behavior changes:** larger batches produce more stable but fewer BN updates. Keep BN momentum unchanged initially; batch 512 still yields 97 updates per full epoch.
- **The timing schedule lags by one update:** LR uses completed elapsed time because the next update's duration is unknown. With expected per-update times in milliseconds, this error is negligible. Clamp progress to `[0, 1]`.
- **Outer runtime exceeds ten minutes:** data-loader waiting and validation are excluded from `training_seconds` but included in `total_seconds`. Persistent workers may be enabled to reduce repeated epoch startup, but do not change the timer semantics. Enforce the required 600-second outer timeout and classify an overrun as failure.

## Ablation Order

If the combined run fails or gives an ambiguous result, use this order; each ablation keeps seed 42 and gets one legitimate run, never seed rerolls:

1. **Schedule only:** width 1, batch 128, FP32 contiguous, time warmup/cosine. This isolates repair of the baseline's unreachable milestones.
2. **Execution operating point:** width 1, batch 512, BF16 channels-last, time schedule. This measures throughput and large-batch effects without extra capacity.
3. **Batch balance:** width 2 at batch 256 versus 512. Prefer 256 if update starvation dominates despite lower throughput.
4. **Capacity:** width 1 versus width 2 at the better batch size. This isolates accuracy gained per unit of device time.
5. **Arithmetic/layout:** disable channels-last and BF16 one at a time. Keep only components with measured throughput or accuracy value.
6. **Compile, later only:** after shapes and batch are fixed, measure a pre-warmed `torch.compile` variant against the retained eager configuration and ensure compilation is accounted for consistently.

## Verification

1. Confirm the required device before execution. Physical GPU 0 must report `NVIDIA H20`, approximately 97,871 MiB, and compute capability 9.0. Launch only with `CUDA_VISIBLE_DEVICES=0`.
2. Run exactly once under the outer limit: `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
3. Require a complete final summary and `training_seconds` near 300 seconds, `total_seconds < 600`, finite losses, and no crash. Confirm `num_params` is about 4x baseline and inspect `peak_vram_mb` for the expected low-single-digit-GiB range.
4. Confirm logs show warmup completion near 5% of measured time and monotonically decreasing LR thereafter, with LR near 0.002 at termination. This verifies schedule behavior independently of step count.
5. Confirm exactly one `eval ep` record per completed epoch and no extra mid-epoch evaluations.
6. Compare `num_steps`, `num_epochs`, and approximate presented examples (`num_steps * 512`) with BASE's 34,435 steps and 4.41M examples. Throughput is diagnostic; the verdict remains solely `best_test_acc`.
7. The proposal is an improvement only if `best_test_acc >= 91.61%`. Report `final_test_acc` as a convergence check: a large best/final gap suggests excessive late noise or a schedule implementation error.
8. Remove `run.log` after analysis as required by the goal protocol.
