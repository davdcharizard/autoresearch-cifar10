# H20 Throughput Dividend: BF16 GPU Pipeline + 2x-Width PreAct ResNet-20

## Proposal Summary

Use the H20's currently idle compute and memory capacity to train a materially stronger, but still small, CIFAR residual network within the same 300 seconds. Move the entire 50,000-image training set to the GPU as `uint8`, perform the existing random crop and horizontal flip in vectorized Torch operations, train channels-last under BF16 autocast with a batch size of 512, and use `torch.compile` only after a guarded untimed compilation warm-up. Spend the resulting throughput dividend on a 2x-width preactivation ResNet-20 (stage widths 32/64/128, approximately 1.08M parameters).

This is a joint systems-and-capacity intervention, not a claim that raw step count alone improves accuracy. The baseline is an extremely small 269,722-parameter FP32 model trained at batch 128. On an NVIDIA H20 with 97,871 MiB and compute capability 9.0, its kernels and batches are unlikely to occupy much of the accelerator. Larger batches, BF16 Tensor Core execution, channels-last convolution layouts, and removal of per-example CPU transforms should make enough compute available to train a wider model at comparable or higher image throughput.

The end-to-end timeout must be handled explicitly. The baseline spends 300 seconds in its counted training region but 595.4 seconds total because it evaluates after every one of 99 epochs. A faster pipeline could complete more epochs and therefore make total runtime *worse*. The proposal caps evaluation at five epoch-boundary checkpoints, including the terminal checkpoint. This remains within the rule of no more than one validation per epoch and creates room for compilation while keeping total runtime below ten minutes.

## Baseline Diagnosis

- Accuracy: 91.67% best test accuracy.
- Work completed: 38,525 optimizer steps, batch size 128, or about 4.93M presented examples across 99 epochs.
- Model: post-activation ResNet-20 with 269,722 parameters and 330.1 MB peak VRAM.
- Counted training: 300 seconds, approximately 7.8 ms per step and 16.4k images/s.
- End-to-end time: 595.4 seconds, leaving essentially no margin under the ten-minute failure threshold.
- The baseline's small batch, FP32 execution, CPU/Python transforms, host-to-device copy, and synchronization on every optimizer step are poor matches for a 98 GB Hopper-class accelerator.
- The original step milestones are also misaligned with the observed time horizon: step 32,000 occurs around 83% of the run, while step 48,000 is never reached. Any batch-size or throughput change makes fixed step milestones still less meaningful.

The bottleneck for this idea is therefore useful model capacity per fixed active-training second, subject to the separate end-to-end timeout. Merely accelerating the existing 0.27M-parameter model may saturate its representational/generalization limit. Merely widening it in FP32 at batch 128 may sacrifice too many examples. The proposed combination exchanges system efficiency for capacity.

## Mechanism

1. **BF16 plus channels-last increases convolution efficiency.** H20 Tensor Cores are well matched to BF16 and channel counts divisible by eight. BF16 has a much wider exponent range than FP16, so plain SGD does not require a gradient scaler. Keeping model weights in FP32 while autocasting convolutional forward/backward operations preserves optimizer precision.
2. **A larger batch exposes parallelism.** Batch 512 provides four times the examples per launch and makes the small 32x32 convolutions less launch-bound. The learning rate is scaled accordingly, with warm-up to control the changed optimization transient.
3. **GPU-resident augmentation removes the input path from the critical loop.** CIFAR-10 occupies only about 150 MB as GPU `uint8`. Batched crop/flip, conversion, and normalization avoid per-sample PIL/Python work and repeated PCIe copies without changing augmentation semantics.
4. **Compilation reduces Python and kernel-launch overhead.** A fixed batch shape, fixed 32x32 resolution, and static residual model are favorable for `torch.compile`. A warm-up before the training timer is legitimate because compilation is explicitly excluded from the training budget; it must not perform a real optimizer update.
5. **Preactivation and width turn efficiency into accuracy.** Full preactivation improves gradient flow through identity paths, while widths 32/64/128 increase the convolutional parameter budget by roughly 4x. The result remains tiny relative to H20 capacity and should have substantially more representational headroom than ResNet-20 at widths 16/32/64.
6. **Elapsed-time scheduling makes the optimization horizon invariant.** Warm-up and cosine decay based on counted active-training time reach a low learning rate regardless of the number of steps produced by compilation or batch-size changes. This follows the time-horizon concern identified in the SGDR paper note.

## Exact Proposed Changes to `train.py`

### Model

- Replace `BasicBlock` with a full preactivation block:
  - `BN(in) -> ReLU -> 3x3 conv(stride)`.
  - `BN(out) -> ReLU -> 3x3 conv`.
  - Use the untouched identity for same-shape blocks.
  - At the first block of stages 2 and 3, use a bias-free 1x1 projection with stride 2 on the preactivated signal.
- Use a 3x3 stem convolution with 32 output channels and no stem BN/ReLU; the first block supplies preactivation.
- Use three blocks per stage with widths 32, 64, and 128, preserving ResNet-20 depth.
- Add final `BN(128) -> ReLU -> global average pool -> Linear(128, 10)`.
- Retain Kaiming initialization. The expected parameter count is approximately 1.084M, about 4.0x baseline but still modest for this device.
- Move the model to `torch.channels_last` before constructing the compiled wrapper.

### Data and augmentation

- Construct CIFAR-10 without a per-example transform and read `train_set.data` and `train_set.targets` once.
- Copy images to the GPU as an NHWC/NCHW `uint8` tensor and targets as `int64`; do not materialize all 50,000 images as float.
- At each epoch, create one seeded GPU `randperm` and consume 97 full batches of 512, matching the baseline's `drop_last=True` behavior closely (49,664 examples versus 49,920 baseline examples per full epoch).
- For each batch:
  - Zero-pad by four pixels while still `uint8`.
  - Generate one `(y, x)` crop offset per image in `[0, 8]`.
  - Use `unfold` views plus batched advanced indexing to select 32x32 crops without a Python loop.
  - Apply one independent horizontal-flip mask per image.
  - Convert to float, divide by 255, subtract the existing per-channel mean, and make the result channels-last contiguous.
- Preserve precisely the baseline's augmentation family: random crop with zero padding and random horizontal flip. Do not add RandAugment, Mixup, CutMix, label smoothing, or other regularization in the first experiment.

### Precision and compilation

- Enable `torch.backends.cudnn.benchmark = True` and `torch.set_float32_matmul_precision("high")`.
- Run the forward and loss under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`; keep parameters, BatchNorm state, loss reduction, and optimizer state in FP32. Do not use `GradScaler` for BF16.
- Wrap only the training model with `torch.compile(model, mode="reduce-overhead")`. Evaluate the eager underlying module so test-time compilation cannot recur at every checkpoint.
- Before starting `t_start_training`, execute one fixed-shape dummy forward/backward to trigger training-graph compilation, synchronize, clear gradients, and restore the saved pristine model state (especially BatchNorm running buffers). Create or recreate the optimizer after this warm-up. No parameter update and no real training example may occur outside the 300-second budget.
- Catch a compile/warm-up failure and fall back to the restored eager model rather than losing the run. Also fall back if compilation alone consumes more than roughly 120 seconds, preserving enough margin under the ten-minute total timeout.

### Optimization and time accounting

- Use batch size 512.
- Use SGD with momentum 0.9 and weight decay `5e-4`; use base learning rate `0.4`, consistent with linear scaling from 0.1 at batch 128.
- Warm the LR linearly from 0.04 to 0.4 during the first 5% of counted training time, then cosine-decay it to 0.004 at 300 seconds. Compute progress from active training seconds, not optimizer steps or epoch count.
- Use standard hard-target cross-entropy in the first experiment.
- Replace unconditional per-step `torch.cuda.synchronize()` with synchronized timing chunks of at most eight batches. Accumulate actual wall time for training segments, including augmentation and optimizer work, while excluding only compilation and evaluation. When less than two seconds remain, synchronize and check after every batch to limit budget overshoot to approximately one step.
- Keep a fixed seed of 42 for CPU and CUDA. This is reproducibility, not seed selection.

### Evaluation and timeout control

- Evaluate at no more than five epoch boundaries: the first completed epoch at or after 20%, 40%, 60%, and 80% of active-training time, plus the terminal model after the budget ends.
- Track the last evaluated epoch and never evaluate twice for the same epoch. The terminal partial epoch receives the terminal evaluation; a completed prior epoch has a different epoch identifier.
- Synchronize before pausing the training timer and before evaluation. Validation time must not enter `training_seconds` but must remain visible in `total_seconds`.
- Pass the eager underlying module to the unchanged `Eval.evaluate()` method. Continue reporting `best_test_acc` over all performed evaluations.
- Abort compilation/fall back early enough that the complete run is expected to remain below 600 seconds. With only five validation passes, there should be substantially more end-to-end headroom than the baseline's 99 passes.

## Hypothesis and Expected Benefit

**Primary hypothesis:** the H20-aware pipeline will maintain at least the baseline's approximately 16k image/s while training a roughly 4x larger preactivation network, and the added capacity plus a schedule that actually anneals will raise `best_test_acc` from 91.67% to at least 92.2% in one 300-second run.

A reasonable first-run target range is 92.3-93.2%, a gain of 0.6-1.5 percentage points. The lower end assumes the wider model processes only roughly the baseline's 4.9M examples; the upper end requires both increased sample throughput and stable large-batch optimization. The minimum success threshold remains 91.77% under the goal definition.

Expected systems outcomes are:

- 80-130 equivalent full-data epochs within 300 active seconds despite the wider model.
- Approximately 1.08M parameters.
- Peak VRAM likely below 5 GB, including the resident dataset, batch activations, compiler workspace, and optimizer state; even a several-fold underestimate remains well below 97,871 MiB.
- Five or fewer test evaluations and enough removed evaluation overhead to complete comfortably within ten minutes, including compilation.

These are predictions to measure, not acceptance criteria. In particular, a lower optimizer-step count is not itself a failure because batch size changes; processed examples and completed data epochs are the comparable work measures.

## Risks and Mitigations

- **Large-batch generalization or under-training.** Batch 512 produces fewer parameter updates per image. Warm-up and time-based cosine decay mitigate instability, while 512 is conservative relative to the H20's capacity. If accuracy is flat despite high throughput, test batch 256 before abandoning the architecture.
- **The 4x wider model consumes the entire throughput gain.** This is acceptable if it preserves roughly 100 effective epochs and improves accuracy. Log images/s and examples processed. If fewer than 75 epochs complete, reduce the base width to 24 (24/48/96, approximately 0.61M parameters) before removing the systems changes.
- **`torch.compile` has high startup cost or a graph/runtime failure.** The reduced evaluation count provides total-time margin. Use a guarded warm-up and eager fallback; do not use `max-autotune` in the first experiment because its compile latency is unnecessary risk.
- **Compiled training and eager evaluation diverge.** Both wrappers share the same underlying parameters and buffers. Synchronize before evaluation and call `model._orig_mod` (or retain an explicit eager-module reference).
- **Vectorized crop changes augmentation semantics.** Pad the raw uint8 image with zeros before conversion/normalization, use uniform integer offsets 0-8 independently per image, and apply Bernoulli(0.5) flips. Add local assertions for shape, dtype, and value range during setup, outside timed training.
- **BF16 changes numerical behavior.** Keep BatchNorm and optimizer state FP32 and use BF16 rather than FP16 to avoid loss scaling. If loss becomes non-finite, automatically mark the run invalid rather than changing the seed.
- **Sparse validation misses a transient best checkpoint.** Cosine annealing makes late checkpoints the most relevant; checkpoints at four intermediate budget fractions still detect gross regressions. The dramatic timeout reduction is worth the small loss of temporal resolution.
- **Timing undercounts asynchronous GPU work.** Synchronize at each timing-chunk boundary and before evaluation/final reporting. Chunked synchronization removes per-step serialization without exempting actual GPU execution from the 300 seconds.
- **The proposal combines several changes, weakening attribution.** The pieces form one resource-exchange hypothesis, but follow-up ablations are required. The first run intentionally excludes unrelated augmentation and regularization so it does not become an untraceable kitchen-sink recipe.

## Confound Controls and Follow-up Ablations

- Keep dataset, train/test split, evaluator, seed 42, crop/flip distribution, number of blocks, loss, and one-GPU protocol fixed.
- Report active training seconds, total seconds, compile/startup seconds, processed examples, effective epochs, optimizer steps, parameter count, and peak VRAM. Batch-size changes make steps alone misleading.
- Parameterize the LR by active time so faster compilation/kernel behavior cannot silently change where decay occurs.
- Do not combine this first run with the separately motivated Mixup, RandAugment, label-smoothing, or weight-averaging ideas documented in the experiment paper notes.
- If the combined run improves, run a throughput-only control using the original width-16 post-activation model under the new pipeline, then a width ablation (base 24 versus 32). This separates benefits from scheduling/system changes and capacity.
- If the combined run fails, first inspect loss stability, effective epochs, and images/s. Disable `torch.compile` only if it fails or is slower; reduce batch to 256 for an optimization failure; reduce width to 24 for a compute-budget failure. Do not reroll the seed.

## Fixed-Budget Feasibility

The resident dataset is negligible relative to the H20's 98 GB VRAM, and the proposed 1.08M-parameter model is still small. The relevant feasibility risk is the ten-minute end-to-end cap. Baseline total time can be decomposed approximately as 300 seconds counted training plus 295 seconds of startup, data-loop, and 99 evaluations. Reducing validation to five calls should recover on the order of several minutes, enough for a guarded compile warm-up and final reporting. Compilation stays outside `training_seconds` as required but remains inside the ten-minute wall-clock check.

The active-training timer must include GPU augmentation, forward/backward, optimizer work, and chunk synchronization. It must exclude only the same categories named by the task: startup/compilation and validation. This prevents the GPU-resident pipeline from obtaining extra training by moving previously timed work outside the timer.

## Suggested First Experiment Scope

Run the full coherent intervention once with base width 32, batch 512, BF16 autocast, channels-last, vectorized GPU crop/flip, guarded `torch.compile(mode="reduce-overhead")`, 5% warm-up plus time-based cosine decay, and at most five evaluations. Keep hard-label cross-entropy and omit every advanced regularizer. This single run answers the high-value question: can H20-specific efficiency pay for a 4x-parameter preactivation model without reducing the amount of useful CIFAR-10 training completed in 300 seconds?

Success requires `best_test_acc >= 91.77%`, a valid numeric summary, `training_seconds` approximately 300 seconds, and `total_seconds < 600`. A result above 92.2% with at least roughly 80 effective epochs supports the hypothesis strongly. A faster run with unchanged accuracy indicates capacity or optimization, not throughput, is limiting and motivates batch/width ablation. A slow run below 75 effective epochs indicates the first correction should be width 24, not removal of the entire pipeline.

## Relevant Experiment Notes

- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/sgdr.md`: motivates a smooth schedule expressed in the actual training horizon rather than unreachable fixed step milestones.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/mixup.md` and `label-smoothing.md`: both may later regularize the wider model, but are excluded initially to control confounding and avoid slowing fixed-budget convergence.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/randaugment.md`: explicitly flags host-side transform overhead; the first run instead preserves baseline augmentation semantics while moving crop/flip to the GPU.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/weight-averaging.md`: potentially complementary after the throughput-capacity hypothesis is validated, but excluded because BatchNorm-statistics handling would complicate the first test.
