# Idea 04 — Throughput maximization to fit far more training into the 300s budget

## Summary

The metric is `best_test_acc` under a **fixed 300s training-time budget**, and the baseline
ResNet-20 never finishes its own LR schedule (`MultiStepLR` milestones at steps 32000/48000,
but only ~37k steps fit in 300s — confirmed by the diagnosis and by reading the schedule in
`train.py:145-147`). The model is also tiny (~0.27M params, 32×32 inputs) and the H20 is
massively underutilized (330 MB / 98 GB, fp32, batch 128, ~8.0 ms/step). This proposal is the
**orthogonal throughput lever**: keep the SAME ResNet-20 architecture, push steps/sec as high as
possible (larger batch + bf16 autocast + channels_last + cuDNN autotune + torch.compile +
reduced per-step overhead + better dataloader), then **re-tune the LR schedule to the new, much
larger achievable step budget** so the model actually converges within 300s.

The causal chain to the metric: more throughput → more steps/epochs in 300s → a cosine/one-cycle
schedule that actually completes → better convergence → higher `best_test_acc`. The dominant risk
is large-batch generalization loss, which is mitigated by the linear LR scaling rule + warmup
(Goyal et al. 2017) and by keeping the batch in a moderate range (256–512), not extreme.

This is deliberately a **convergence-enablement** change, not an architecture change. Its biggest
single contribution may simply be **fixing the never-completed schedule** — which is achievable
even at the current batch size and is the lowest-risk part of the proposal.

## What it targets (named limiter)

Diagnosis limiter: the baseline is **compute/throughput-starved and schedule-mismatched** — the
300s budget buys only ~37k steps but the LR schedule is written for 64k steps, so the LR never
decays to the low-LR regime where ResNets do most of their accuracy gain. Two coupled levers:
1. **Throughput** (steps in 300s) — raise it so more of the schedule runs.
2. **Schedule match** — make the schedule complete within whatever step count actually fits.

Both live entirely in `train.py` and touch no frozen code.

## Budget / timing accounting — honest analysis (READ THIS FIRST)

This is the part that must be legitimate. From `train.py:155-203`:

- `total_training_time` is the budget variable. It accumulates **only** `dt = time.time() - t0`,
  where `t0` is set at the **first line inside the per-step for-body** (`train.py:167`) and `dt`
  is measured **after** `torch.cuda.synchronize()` (`train.py:178-180`). The loop runs
  `while total_training_time < TIME_BUDGET_S` (`train.py:162`).
- Therefore the budget counts: H2D copy + forward + backward + optimizer.step + scheduler.step +
  the `synchronize()`. It does **NOT** count: dataloader batch-fetch latency (that happens at the
  top of the next `for` iteration, before `t0`), `evaluator.evaluate` time, or anything before
  `t_start_training`.
- `startup_seconds = t_start_training - t_start` is reported separately and is NOT in the budget.

Implications for what is legitimate:

1. **torch.compile and cuDNN autotune must be warmed up BEFORE the timed loop.** torch.compile
   compiles lazily on the first forward/backward; cuDNN `benchmark=True` autotunes on first sight
   of each input shape. If these happen inside the loop, the first few `dt` values balloon and
   correctly consume budget. The legitimate fix is to run a handful of **warmup
   forward+backward steps on a dummy batch (or the first real batch) before `t_start_training`**,
   then `torch.cuda.synchronize()`, then start the loop. This time lands in `startup_seconds`,
   which the goal explicitly excludes ("excluding startup/compilation", 01-definition.md:8). This
   is honest: it is genuinely compilation/autotune, not training. It is the standard way these
   benchmarks are run. **We must NOT remove or weaken the in-loop `torch.cuda.synchronize()` +
   `dt` timing** — that is the budget meter itself; tampering with it would game the accounting.

2. **The per-step `torch.cuda.synchronize()` (train.py:178) cannot simply be deleted.** It is what
   makes `dt` a true wall-clock measure of that step's GPU work. Removing it would let GPU work
   "leak" past the timer and undercount the budget — that is exactly the kind of timing game we
   must avoid. So we KEEP the per-step sync and per-step `dt` accounting unchanged. The throughput
   win must come from making each *honestly-timed* step cheaper or doing more useful work per step
   (bigger batch), NOT from hiding step time.

3. **`loss.item()` every step (train.py:183) forces a GPU→CPU sync** and is only used for a
   logging EMA. Because we already `synchronize()` right before it, the extra stall is small, but
   we can still reduce it: keep the loss on-GPU and only `.item()` it inside the `if step % 50`
   logging branch. This is a real, legitimate per-step saving and does not touch the budget meter.

Net: the legitimate throughput gains are (a) larger batch (more images per honestly-timed step),
(b) bf16 + channels_last + TF32 + compile making each step's GPU work faster, (c) warming
compile/autotune into startup, (d) trimming `.item()`/Python overhead, (e) a faster dataloader so
the loop is not stalled between steps (dataloader stalls are outside the meter but still waste wall
clock and reduce epochs completed within the 10-min hard cap). None of these touch `prepare.py`
or the budget meter.

## Exact `train.py` changes

All edits are in `train.py` only. Grouped by concern.

### 1. Global perf flags (top of `main`, after seeding, ~train.py:108-111)
```python
torch.backends.cudnn.benchmark = True              # autotune convs for fixed 32x32 shapes
torch.backends.cuda.matmul.allow_tf32 = True       # TF32 on the fc/matmul path
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")         # TF32 for fp32 matmuls
```
Fixed input shape (32×32, drop_last=True already on the train loader, train.py:135) makes
`cudnn.benchmark` a pure win with no reshuffling cost.

### 2. Hyperparameter block (train.py:18-24)
- `BATCH_SIZE = 256` (start), with `LR` scaled by the linear rule. Baseline LR 0.1 at batch 128 →
  `LR = 0.1 * (256/128) = 0.2`. (Treat 512/LR 0.4 as a follow-up only if 256 is stable.)
- Remove the `MAX_STEPS = 64000` early-exit dependence on a fixed count; drive everything off a
  **fraction-of-budget** schedule instead (see §5). Keep a generous `MAX_STEPS` guard well above
  what 300s can reach so the `while` exits on time, not steps.

### 3. Model + data to channels_last + bf16 (train.py:138, 166-176)
```python
model = ResNet(NUM_BLOCKS, NUM_CLASSES).to(device, memory_format=torch.channels_last)
...
inputs = inputs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
...
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
loss.backward()
```
**bf16, not fp16** → no `GradScaler` needed (bf16 has fp32 exponent range), simpler and robust.
BatchNorm and the final reductions stay in fp32 under autocast automatically. channels_last must
be applied to BOTH model and inputs to actually hit NHWC cuDNN kernels (per PyTorch perf guide).

### 4. torch.compile + warmup (before `t_start_training`, train.py:155)
```python
model = torch.compile(model)            # default mode; reduces kernel-launch overhead
# warmup: compile + cudnn autotune OUTSIDE the timed loop
model.train()
warm_x = torch.randn(BATCH_SIZE, 3, 32, 32, device=device).to(memory_format=torch.channels_last)
warm_y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        l = F.cross_entropy(model(warm_x), warm_y)
    l.backward()
    optimizer.step()
torch.cuda.synchronize()
# reset optimizer/model state touched by warmup is unnecessary for BN/weights at this scale,
# but to be clean: re-init model weights OR build optimizer AFTER warmup on a fresh model.
```
**Important correctness note:** the warmup does 3 real optimizer steps on random data, which
perturbs weights/BN stats. Cleanest fix: construct the model, `torch.compile` it, run warmup on a
**throwaway copy or with `torch.no_grad` forward-only passes** to trigger compile, then build the
optimizer and reset seeds for the real run. Forward-only warmup compiles the inference graph but
NOT the backward graph, so to warm the backward path do the warmup on the real model and then
**re-initialize weights** via `model.apply(ResNet._weights_init)` and reset BN running stats
before the timed loop. The planner should pick the clean variant; the key invariant is: timed loop
starts from a properly initialized model with compile+autotune already done.

### 5. LR schedule matched to the achievable step budget (train.py:145-147, 176)
The core fix. Replace `MultiStepLR(milestones=[32000,48000])` with a **OneCycle / cosine schedule
sized to the steps that actually fit in 300s.** We do not know that count a priori, so estimate it
and verify:
- Measure once: at batch 256 + bf16 + channels_last + compile on H20, expect a large steps/sec
  increase vs baseline's ~125 steps/s. Conservatively assume we reach the **full dataset many
  times**. Set `total_sched_steps` from a quick calibration: run the warmup, time ~50 real steps to
  get `ms/step`, compute `est_steps = int(TIME_BUDGET_S / (ms_step/1000) * 0.97)` (3% safety
  margin), and build the schedule for `est_steps`.
- Use `optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=est_steps,
  pct_start=0.15, anneal_strategy="cos", div_factor=10, final_div_factor=100)`. The 15% warmup
  implements the Goyal et al. warmup; cosine anneal guarantees the LR reaches near-zero by the time
  budget ends, which is exactly what the baseline fails to do.
- Alternatively, `CosineAnnealingLR(T_max=est_steps)` plus a short linear warmup. OneCycle is
  preferred because it bundles warmup + anneal and is well-matched to short budgets (Smith 2018).
- Keep `scheduler.step()` per optimizer step (train.py:176), unchanged in position.

### 6. Dataloader efficiency (train.py:129-136)
```python
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    persistent_workers=True, prefetch_factor=4,
)
```
`NUM_WORKERS=8` (from prepare.py). `persistent_workers=True` avoids re-spawning workers every
epoch (with larger batches there are more, shorter epochs, so respawn overhead matters more).
`prefetch_factor=4` keeps the GPU fed so the between-step dataloader stall (outside the budget
meter but inside the 10-min wall cap) stays small. This does not affect the budget meter but
protects the wall-clock hard limit.

### 7. Reduce per-step Python overhead (train.py:183-200)
- `optimizer.zero_grad(set_to_none=True)` (cheaper than zeroing).
- Move `loss.item()` and the EMA/logging math **inside** the `if step % 50 == 0` branch so the
  GPU→CPU sync for logging happens 1/50 as often. The budget meter (`dt`) is unchanged; this just
  removes a redundant per-step stall.

## Reasoning with cited pointers

- **Schedule never completes (primary, highest-confidence win).** `train.py:145-147` sets decay
  milestones at 32k/48k steps; the diagnosis says only ~37k steps fit in 300s, so the second decay
  (and most of the low-LR fine-convergence) never happens. ResNet CIFAR accuracy depends heavily on
  reaching the low-LR phase (He et al. 2015, ResNet CIFAR recipe). A cosine/one-cycle schedule
  sized to the real step count fixes this even at batch 128 — this alone is plausibly worth a
  meaningful fraction of the needed +0.1pp and is the safest part of the proposal.
- **Larger batch amortizes launch/Python overhead on a launch-bound tiny model.** ResNet-20 is
  ~0.27M params with many small conv kernels; at batch 128 the H20 is at <1% memory and the step is
  dominated by kernel-launch + Python overhead, not FLOPs (consistent with the "massively
  underutilized" diagnosis and ~8 ms/step for such a small model). Doubling/quadrupling the batch
  raises images-per-honestly-timed-step roughly linearly until compute-bound — directly increasing
  throughput without gaming the meter.
- **Linear LR scaling + warmup preserves generalization at larger batch.** Goyal et al. 2017,
  "Accurate, Large Minibatch SGD" — scale LR linearly with batch and use a warmup; validated to
  hold SGD accuracy up to large batches. We apply LR = 0.1 × (B/128) with OneCycle's built-in
  warmup (`pct_start=0.15`).
- **bf16 + channels_last + TF32 + cuDNN benchmark are the standard CNN throughput stack.** PyTorch
  Performance Tuning Guide (docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html):
  channels_last is "meant to be used in conjunction with AMP to further accelerate CNNs"; AMP gives
  "up to 3x" on Volta+; `cudnn.benchmark=True` autotunes for fixed shapes;
  `set_float32_matmul_precision('high')`/`allow_tf32` accelerate the fp32 matmul path. H20 has
  ~296 BF16 TFLOPS vs ~74 TF32 TFLOPS (Hopper), so bf16 is the strong throughput path on this GPU
  — though note this model is launch-bound, so the bf16 *compute* win is secondary to the
  *kernel-fusion* win from compile and the *amortization* win from batch size.
- **torch.compile reduces launch overhead and its cost is startup, not budget.** Confirmed by the
  timing analysis above: compile is lazy on first call; warming it before `t_start_training` keeps
  it in `startup_seconds`, which 01-definition.md:8 explicitly excludes from the budget. For a
  launch-bound tiny CNN, kernel fusion is one of the largest available per-step savings.

Sources:
- [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Accurate, Large Minibatch SGD (Goyal et al. 2017)](https://arxiv.org/abs/1706.02677)
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [H20 BF16/TF32 throughput figures (third-party listing — verify)](https://www.burncloud.com/gpu-catalog/H20.html)

## Estimated effort

**Low–Medium** relative to one experiment loop. All edits are localized in `train.py`: perf flags
(4 lines), channels_last + autocast (a few lines), compile + warmup (one block — the trickiest
part is the clean weight re-init after backward-graph warmup), OneCycle schedule sized from a quick
calibration, dataloader kwargs, and the `.item()` logging move. The main iteration cost is one
calibration/tuning pass to right-size `est_steps` and confirm batch/LR stability.

## Risk assessment (worst case)

1. **Large-batch generalization loss (primary risk).** If LR is mis-tuned, batch 256–512 can lose
   1–2pp vs batch 128. Mitigation: conservative batch (256 first), linear LR scaling + OneCycle
   warmup, and the fallback of keeping batch 128 but still fixing the schedule (which captures much
   of the upside at near-zero generalization risk). Worst case: a single run shows no improvement;
   no budget is gamed, run still completes.
- **bf16 numerics on BatchNorm/residuals.** Low risk — autocast keeps BN and loss reductions in
  fp32; bf16's fp32 exponent range avoids the overflow issues that plague fp16, so no GradScaler.
  Worst case: tiny accuracy noise, addressed by autocasting only the conv/matmul regions (default
  autocast already does the right thing).
- **torch.compile warmup contaminating the timed run.** If the warmup leaves perturbed weights/BN
  and we forget to re-init, accuracy drops. Mitigation: explicit `model.apply(_weights_init)` +
  BN running-stat reset after warmup, started from a fixed seed. This is a correctness checklist
  item, not a fundamental risk.
- **torch.compile failing/recompiling on shape changes.** `drop_last=True` (train.py:135) fixes the
  train batch shape, so no recompiles in the loop; the only variable-shape risk is eval, but eval
  runs the frozen `Eval.evaluate` on the (compiled) model with batch 256 fixed (prepare.py:24-30) —
  one extra compile, absorbed at first eval. If compile is flaky, fall back to eager + bf16 +
  channels_last (still a throughput win).
- **Throughput gain smaller than hoped because the model is launch-bound, not compute-bound.** The
  bf16/TF32 *compute* speedup may be modest; the real wins are batch-size amortization + compile
  fusion + schedule completion. Honest framing: if compile underdelivers, the schedule fix + batch
  amortization still stand on their own.

## Expected accuracy estimate with justification

Baseline 91.57%. I expect **92.0–93.0%** best_test_acc, i.e. roughly **+0.4 to +1.4pp**, with the
**most-confident component being the schedule fix** (a completed cosine/one-cycle decay typically
adds several tenths to >1pp over a schedule frozen mid-high-LR for ResNet CIFAR recipes). The
throughput gains (compile + bf16 + channels_last + bigger batch) convert into more completed
cosine cycles / more effective epochs, compounding the schedule benefit. This comfortably clears
the +0.1pp bar **if** large-batch LR tuning holds. Honest caveat: if batch must stay at 128 to
preserve generalization, the gain narrows toward the schedule-fix-only floor (~+0.3–0.6pp), which
still passes. The dominant uncertainty is the batch/LR interaction, not the engineering.
