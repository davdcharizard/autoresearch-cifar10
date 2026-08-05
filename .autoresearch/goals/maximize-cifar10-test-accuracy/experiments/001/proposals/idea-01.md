# Proposal: Time-Aligned Pre-Activation WRN-16-2

## Summary

Replace the post-activation, thin ResNet-20 with a CIFAR-style pre-activation
Wide ResNet of depth 16 and width multiplier 2 (WRN-16-2), increase the batch
size from 128 to 256, and replace step-count milestones with a short warmup plus
cosine decay driven by measured training-time progress. This is a deliberately
moderate first widening step: it raises representational capacity substantially
without spending the 300-second budget on a very large WRN that cannot complete
enough passes over the data.

## Limiter Diagnosis

The baseline reaches only 91.54% despite performing 38,254 optimizer steps and
99 reported epochs in 300 seconds. It uses 330 MiB on an H20 with roughly 98 GB
available, so memory is not a limiting resource. The two immediate limitations
are model capacity/use of compute and a schedule that is misaligned with the
actual stop condition:

- ResNet-20 has only 272,474 parameters and narrow 16/32/64-channel stages. The
  H20 spends the budget executing many tiny convolutions rather than learning
  richer features from a moderately wider model.
- The current `MultiStepLR` decays at steps 32,000 and 48,000. At the observed
  throughput, the first decay occurs after about 251 seconds (83.7% of the
  budget), and the second is never reached. Thus nearly the entire run uses LR
  0.1 and never receives the intended low-LR convergence phase.
- The baseline already sees approximately 98 dataset-equivalent passes
  (`38,254 * 128 / 50,000`). More steps in the same thin architecture are less
  likely to help than spending some throughput on width while retaining enough
  data passes for convergence.

## Proposed Model

Implement a true CIFAR pre-activation residual network in `train.py`:

- Stem: one 3x3 convolution, 3 to 16 channels, stride 1, no bias.
- Three groups with output widths 32, 64, and 128 (width multiplier 2).
- Two residual blocks per group. The first block of groups 2 and 3 downsamples
  with stride 2. This gives depth `6 * 2 + 4 = 16` under WRN counting.
- Each block uses `BN -> ReLU -> 3x3 conv -> BN -> ReLU -> 3x3 conv`, followed
  by residual addition, with no post-addition activation.
- When shape changes, use a learned 1x1 projection with the block stride. Feed
  the pre-activated tensor to the projection, matching the standard
  pre-activation WRN transition rather than zero-padding channels.
- Apply a final `BN -> ReLU`, global average pooling, and a 128-to-10 linear
  classifier.
- Retain Kaiming initialization and initialize batch-normalization scale to 1
  and bias to 0. Do not add block dropout in this first run; under the short
  budget it adds another hyperparameter and may slow fitting.

WRN-16-2 is expected to have about 0.69 million parameters, roughly 2.5 times
the baseline, while reducing residual blocks from nine to six. It captures the
wide-network benefit reported by Zagoruyko and Komodakis without the compute
cost of WRN-28-10. The shallower graph also limits sequential kernel-launch
overhead, which matters for 32x32 feature maps.

## Training Configuration

Use the existing crop-and-horizontal-flip pipeline and deterministic seed 42.
Change only the following training settings:

- `BATCH_SIZE = 256` to improve H20 occupancy and amortize Python/kernel-launch
  overhead. The loader still drops the final incomplete batch, giving 195
  updates per full epoch.
- SGD with momentum 0.9 and Nesterov enabled.
- Peak LR 0.2, linearly scaled from the baseline's 0.1 for the doubled batch.
- Weight decay 5e-4, the conventional WRN CIFAR setting; do not decay batch-norm
  affine parameters or biases. Construct two optimizer parameter groups so
  only convolution and linear weights receive weight decay.
- No AMP and no `torch.compile` in this experiment. Either may be valuable
  later, but compilation startup inside the timed loop and numerical changes
  would confound the first architecture result.

The no-decay parameter group is a small but important detail: a fivefold higher
weight decay applied indiscriminately to batch-normalization scales can impede
short-budget optimization.

## Wall-Clock Learning-Rate Schedule

Remove `MultiStepLR`. Before each optimizer update, derive the LR from the
training time already consumed:

```python
progress = min(max(total_training_time / TIME_BUDGET_S, 0.001), 1.0)
warmup = 0.05
if progress < warmup:
    lr = 0.2 * progress / warmup
else:
    cosine_progress = (progress - warmup) / (1.0 - warmup)
    lr = 0.002 + 0.5 * (0.2 - 0.002) * (
        1.0 + math.cos(math.pi * cosine_progress)
    )
for group in optimizer.param_groups:
    group["lr"] = lr
```

Use a nonzero initial warmup value (for example `0.2 / 100`) on the first step
so the first update is not a no-op. Five percent corresponds to 15 measured
training seconds, and the LR is guaranteed to enter its convergence phase even
if widening changes steps per second dramatically. The 0.002 floor preserves a
small amount of learning at the end without an abrupt drop. Because the timer
measures synchronized training work, schedule progress and the enforced budget
share the same clock.

## Expected Budget Behavior

The doubled batch and wider convolutions should make each update slower, but
they should use the H20 much more efficiently. A conservative target is at
least 45 dataset-equivalent epochs in 300 seconds; 60-90 is plausible if the
larger kernels improve utilization. Peak allocated memory should remain well
below 2 GiB, far under the observed hardware limit. Validation remains exactly
once after each completed or budget-truncated epoch, and the existing outer
time guard remains unchanged. `MAX_STEPS` can be retained as a safety ceiling,
but wall-clock progress, not `MAX_STEPS`, controls the schedule.

## Hypothesis and Success Criteria

The wider pre-activation representation should reduce under-capacity error,
while the time-based cosine policy gives the model substantially more
low-learning-rate optimization than the baseline. The primary hypothesis is
that the run achieves at least 92.0% `best_test_acc`, exceeding the 91.54%
baseline and the required 91.64% improvement threshold, while completing in
under 10 minutes on one H20.

Record `num_epochs`, `num_steps`, parameter count, and peak VRAM along with the
accuracy. A result below 91.64% with fewer than roughly 45 dataset passes would
point to insufficient optimization time rather than disproving width; that
would motivate a smaller width, BF16, or a larger batch in the next experiment.
A result below threshold despite 60 or more passes would instead implicate the
LR/regularization choices and favor tuning peak LR or weight decay before
scaling capacity further.

## Risks and Controls

- **Too few useful updates:** Widening may reduce throughput more than expected.
  WRN-16-2, rather than WRN-28-10, bounds this risk, and batch 256 improves
  utilization. The wall-clock schedule still completes even at low throughput.
- **Peak LR 0.2 instability:** Linear scaling can fail at small epoch counts.
  The 15-second warmup limits early divergence. Loss and first-epoch accuracy
  should be checked for a clear instability signal.
- **Over-regularization:** Weight decay 5e-4 may be too strong for a short run.
  Excluding normalization/bias parameters and avoiding dropout reduce this
  risk. A follow-up can isolate 1e-4 versus 5e-4 if accuracy is close.
- **Attribution:** Architecture, batch size, and schedule all change together.
  They form one coherent compute-budget intervention, but the run cannot assign
  gains among them. The logged throughput and epoch count will identify which
  component deserves the next controlled experiment.

## Evidence

- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/001/papers/wide-residual-networks.md`:
  shallower, wider residual networks improve CIFAR accuracy and computational
  efficiency relative to thin very-deep networks.
- `.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/001/papers/time-matters-regularization.md`:
  regularization timing matters under a finite budget, supporting a schedule
  that reserves an explicit late convergence phase.
- `.autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv` and the
  baseline run summary: 91.54%, 38,254 steps, 99 epochs, and 330 MiB peak VRAM.
- `train.py`: the current second milestone is unreachable within the measured
  300-second run, and the model remains the original narrow 16/32/64 ResNet-20.
