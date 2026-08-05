# Proposal: Throughput-Balanced WRN-16-3

## Summary

Scale the successful EXP-001 pre-activation WRN from width factor 2 to 3 and
raise the batch from 256 to 384 so the wider convolutions can use the H20
efficiently. Scale only the peak learning rate with the batch, from 0.2 to 0.3,
and retain the proven 5%-warmup/time-cosine schedule, 0.002 floor, optimizer,
augmentation, weight decay, and evaluation policy. This is a controlled test of
whether modestly more representation capacity can improve the 93.38% baseline
without sacrificing too much data exposure inside the fixed 300-second budget.

## Limiter Diagnosis

EXP-001 completed 28,540 updates and 147 epochs at about 24.4k images/s, reached
93.38%, and ended at 93.34%. Its late smoothed training loss was approximately
0.005, while test accuracy plateaued near 93.3%. The model therefore converges
reliably but still has a generalization/representation gap. It also allocates
only 1.1 GiB on a 97.9 GiB H20, leaving ample room to trade memory and some
throughput for capacity.

Width is not automatically beneficial under a time budget: WRN-16-3 has about
2.25 times the convolutional work of WRN-16-2, so retaining batch 256 could lose
too many image exposures. Conversely, jumping to batch 512 could reduce the
number of statistically distinct optimizer updates and require a more aggressive
LR extrapolation. Batch 384 is the middle point: it increases work per launch and
activation occupancy by 50%, while reducing each epoch from 195 to 130 updates.
The experiment tests width and its necessary throughput compensation together,
not width in an artificially inefficient batch regime.

## Exact Intervention

Make only these hyperparameter changes in the current `train.py`:

- `WIDEN_FACTOR = 3`, producing stage widths 48/96/192 while retaining depth 16,
  the existing pre-activation blocks, projection shortcuts, initialization, and
  final BN/ReLU/pooling/classifier.
- `BATCH_SIZE = 384`. Keep `drop_last=True`, shuffling, pinned memory,
  `NUM_WORKERS`, and especially `persistent_workers=True` unchanged. The latter
  is required by the EXP-001 timeout finding.
- `LR = 0.3`, the linear batch-scaled equivalent of 0.2 at batch 256. Keep
  `WARMUP_FRACTION = 0.05`, `MIN_LR = 0.002`, and the exact measured-training-time
  cosine function. The unchanged absolute floor intentionally restores a clean
  low-LR convergence tail rather than scaling the entire schedule upward.

Keep all other behavior fixed:

- SGD, momentum 0.9, Nesterov, selective weight decay `5e-4`, and no decay on BN
  affine parameters or biases.
- Crop, horizontal flip, normalization, hard-label cross entropy, and seed 42.
- FP32 execution, cuDNN benchmarking/determinism settings, and
  `zero_grad(set_to_none=True)`.
- `EVAL_EVERY = 5` plus the budget-exhausted final evaluation, so evaluation is
  never more frequent than once per epoch.
- The current `MAX_STEPS` safety ceiling; the 300-second measured training-time
  budget should remain the active stop condition.

Do not add dropout, label smoothing, mixup, Cutout, EMA, AMP, or compilation in
this run. Some are plausible follow-ups, but adding one here would prevent a
clean decision about the capacity/throughput tradeoff and could obscure whether
the wider network itself improved the ceiling.

The resulting model should contain approximately 1,549,530 trainable parameters,
versus roughly 0.69 million for WRN-16-2. Assert or log that parameter count to
catch an accidental architecture change before the full run.

## Why This Configuration

The Wide Residual Networks evidence already saved for this goal supports using
width as an effective CIFAR capacity lever. EXP-001 validated the moderate end of
that mechanism: WRN-16-2 plus a budget-aligned schedule improved the original
baseline by 1.84 percentage points. This proposal takes the smallest integer
width step from that working point rather than changing depth or jumping to a
large WRN.

Batch 384 is coupled to the width change for hardware reasons. Wider feature maps
increase arithmetic intensity, while the larger batch amortizes kernel launches
and input transfer. The H20 should have no memory difficulty: scaling EXP-001's
activation footprint by approximately 1.5 for width and 1.5 for batch suggests
roughly 2.5 GiB peak allocation, still less than 3% of available VRAM. The useful
question is therefore images/second and completed epochs, not capacity or OOM
risk.

The time-based schedule is intentionally retained. A step- or epoch-based policy
would confound this test because width and batch both change the realized step
count. Scaling the peak LR from 0.2 to 0.3 preserves the usual LR-per-sample ratio
for the 50% larger batch; the 15-second warmup bounds early instability, and the
unchanged 0.002 floor preserves the low-LR behavior that produced a final score
within 0.04 points of EXP-001's best.

## Hypothesis and Success Criteria

The testable hypothesis is that WRN-16-3's added representational capacity,
combined with batch 384's throughput compensation, will improve
`best_test_acc` from 93.38% to at least 93.48% while preserving enough image
exposure for the time-cosine schedule to converge. A reasonable target range is
93.55-93.90%.

The run is successful only if all of the following hold:

- A fresh final summary reports `best_test_acc >= 93.48%` (the required
  +0.10-point improvement at the harness's two-decimal precision).
- It uses one NVIDIA H20, approximately 300 seconds of counted training time,
  and completes in less than 10 minutes total.
- Only `train.py` differs from the accepted EXP-001 state; `prepare.py`, the
  evaluator, dependencies, and seed remain untouched.
- Validation occurs at most once in any epoch.

## Budget Expectations and Diagnostics

Before the full run, perform only an untimed one-batch forward/backward smoke
test if needed to verify shapes and memory. Then remove stale `run.log` and run
the prescribed single full experiment with all output redirected.

Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`,
`total_seconds`, `num_epochs`, `num_steps`, parameter count, and peak VRAM. Also
derive average image throughput as `num_steps * 384 / training_seconds` and
dataset-equivalent passes as `num_steps * 384 / 50_000`.

A useful throughput floor is about 13.3k images/s, equivalent to 80 dataset
passes in 300 seconds. This is not a validity condition, but it separates two
failure mechanisms:

- **At least 80 passes, stable loss, but accuracy below 93.48%:** added width did
  not improve generalization enough. Retain WRN-16-2 and test EMA, early mixup,
  or mild label smoothing rather than widening further.
- **Fewer than 80 passes and accuracy regresses:** the width/batch combination is
  compute-limited under 300 seconds. A later throughput study may try batch 512
  or acceleration, but this result should not motivate WRN-16-4.
- **Early non-finite or persistently high loss:** LR 0.3 is too aggressive for
  this model despite warmup. The clean follow-up is the same WRN-16-3/batch-384
  configuration with LR 0.25; do not reinterpret it as a capacity failure.
- **Accuracy exceeds threshold despite fewer than 80 passes:** capacity is
  valuable enough to offset lower exposure, supporting a later controlled
  regularizer on WRN-16-3.

## Risks and Controls

- **Throughput loss:** width triples neither memory nor performance linearly,
  but it increases convolution work substantially. Batch 384 is the bounded
  utilization adjustment; do not dynamically alter batch size during the run.
- **Large-batch optimization:** batch 384 yields fewer updates per epoch. Linear
  peak-LR scaling and the existing warmup compensate, while the unchanged cosine
  floor retains late refinement. Batch 512 is deliberately avoided.
- **Overfitting:** a wider model could widen the already visible train/test gap.
  This experiment isolates that question. If it fails after sufficient passes,
  the next intervention should regularize the proven width-2 baseline or add one
  lightweight regularizer to width 3, not stack several methods here.
- **Attribution:** batch size and peak LR change with width, but they are the
  minimum paired changes needed to keep the hardware and optimizer regime
  comparable. Architecture, data augmentation, optimizer family, regularization,
  schedule shape, seed, and evaluator remain fixed.
- **Wall-clock compliance:** retain persistent loader workers and sparse
  evaluation. Kill and classify the run as failed if total elapsed time exceeds
  10 minutes.

## Evidence

- `experiments/001/04-analysis.md`: WRN-16-2 achieved 93.38%, 28,540 steps, 147
  epochs, 1,092 MiB peak VRAM, and a final score only 0.04 points below its best.
- `experiments/001/papers/wide-residual-networks.md`: wider residual networks are
  an established capacity/compute tradeoff for CIFAR classifiers.
- `03-experiment-learnings.md`: preserve the successful moderate-WRN,
  time-aligned-cosine pattern and persistent workers.
- `04-results.tsv`: the moving baseline is 93.38%, making 93.48% the minimum
  accepted score for EXP-002.

## Estimated Effort

Low. The implementation is three constant changes plus parameter/throughput
logging, followed by one 300-second run. No new code path, dependency, evaluator
change, or additional full-budget calibration run is required.
