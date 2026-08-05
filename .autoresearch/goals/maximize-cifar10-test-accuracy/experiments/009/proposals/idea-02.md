# Proposal: Selective 160-Channel Final Stage

## Summary

Replace the accepted WRN-16-2 topology `[32, 64, 128]` with the explicitly
stage-wise topology `[32, 64, 160]`. The stem, both 32x32 residual blocks, both
16x16 residual blocks, batch size, optimizer, learning-rate schedule, mixup
schedule, augmentation, seed, and evaluation cadence remain unchanged. Only
the two 8x8 residual blocks and their classifier interface become wider.

This is materially different from the rejected uniform WRN-16-3 candidate.
Uniform width 3 changes every residual stage to `[48, 96, 192]`, including the
expensive high-resolution stages, and measured only 56.8% of accepted-path
throughput. The proposed topology leaves those high-resolution stages exactly
at `[32, 64]` and makes a smaller 128-to-160 increase only where feature maps
are 8x8. It adds 39.0% parameters but only 17.0% convolution/linear forward
MACs, versus uniform width 3's 124.0% parameter and 120.6% MAC increases.

The proposal is conditional on a matched, unscored H20 preflight. Proceed to
the single scored run only if the candidate retains at least 85% of accepted
WRN-16-2 image throughput and its ratio-calibrated projection is at least 120
dataset-equivalent passes in 300 seconds.

## Diagnosis and Rationale

The accepted 94.07% recipe is no longer limited by an obvious nearby
regularization or endpoint-schedule defect. CutMix, a shorter mixup phase,
stronger mixup, residual-branch dropout, late weight-decay removal, and a
cosine-to-zero endpoint all regressed while retaining roughly normal exposure.
Those results fix alpha-0.2 mixup through 65%, continuous `5e-4` matrix decay,
and the 0.002 learning-rate floor for this experiment.

Capacity remains an orthogonal lever, but EXP-006 established that memory is
not the relevant constraint. WRN-16-2 uses about 1.1 GiB of a 97.9-GiB H20;
the binding constraint is how much useful optimization fits into 300 counted
seconds. Uniform WRN-16-3 has 1,549,530 parameters and about 2.21 times the
accepted model's convolution/linear MACs. Its matched FP32 microbenchmark took
21.972 ms/step versus 12.488 ms/step, retained only 56.8% throughput, and
projected just 80.6 calibrated passes versus EXP-002's realized 141.9. That is
why a scored uniform-width run was correctly rejected.

Selective final-stage widening addresses the same representation hypothesis
with a different compute allocation. CIFAR stages operate at 32x32, 16x16,
and 8x8. A new convolutional weight in the final stage is reused over only 64
spatial positions, versus 256 or 1,024 earlier. Adding channels there therefore
buys substantially more learned parameters per forward MAC. The final stage
also consumes the most abstract features and directly supplies the global pool
and classifier, making it a plausible place for modest extra class-separating
capacity.

The target width is 160 rather than 192 for three reasons:

- 160 is a 32-channel multiple, preserving regular tensor shapes on the H20.
- It sits halfway between the accepted 128 channels and uniform width 3's 192,
  adding 32 channels or 25%; this is large enough to test capacity without
  reproducing width 3's compute allocation.
- An affine estimate from the measured width-2 and uniform-width-3 timings
  predicts about 13.83 ms/step, 90.3% retained throughput, and 128.1 calibrated
  passes. This estimate is only a prior; the matched preflight is authoritative.

## Exact Topology and Cost

Use a pre-activation WRN with two blocks per stage and these channels:

| Component | Accepted WRN-16-2 | Candidate |
|---|---:|---:|
| Stem | 16 | 16 |
| Stage 1, 32x32 | 32, 32 | 32, 32 |
| Stage 2, 16x16 | 64, 64 | 64, 64 |
| Stage 3, 8x8 | 128, 128 | 160, 160 |
| Pooled classifier input | 128 | 160 |

The first stage-3 block changes from `64 -> 128` to `64 -> 160`, including its
3x3 convolution, 1x1 projection shortcut, second 3x3 convolution, and second
BatchNorm. The second stage-3 block changes from `128 -> 128` to `160 -> 160`.
Final BatchNorm and the linear classifier change to 160 channels. No extra
block, auxiliary head, or bottleneck is introduced.

Exact trainable parameter counts are:

- accepted `[32, 64, 128]`: 691,674;
- candidate `[32, 64, 160]`: 961,562, an increase of 269,888 or 39.0%;
- rejected uniform `[48, 96, 192]`: 1,549,530, or 2.24 times accepted.

Counting convolution and linear multiply-accumulates for one 32x32 image:

- accepted: 101,106,944 MACs;
- candidate: 118,343,232 MACs, an increase of 17,236,288 or 17.0%;
- uniform width 3: 223,004,544 MACs, or 2.21 times accepted.

The candidate therefore carries 62.1% of uniform width 3's parameters but only
53.1% of its forward MACs. Activation and optimizer memory will rise, but the
accepted path's roughly 1.1-GiB peak leaves this far below the H20 capacity.
Throughput, not VRAM, remains the feasibility criterion.

## Hypothesis and Falsification

The additional 8x8-stage channels will improve class-level representation
without cutting accepted-path exposure by more than 15%. With the validated
mixup/hard-label recipe unchanged, the candidate is predicted to retain at
least 120 passes and reach `best_test_acc >= 94.17%`, the required 0.10-point
gain over 94.07%.

The mechanism is falsifiable:

- `best_test_acc >= 94.17%` in a valid run supports selective final-stage
  capacity as a better fixed-time compute allocation.
- A valid score from 94.07% through 94.16% is a near miss but still a formal
  no-improvement; do not accept it or rerun the seed.
- A valid score below 94.07% with at least 120 realized passes and finite,
  stable loss rejects the claim that this extra semantic capacity is useful
  under the accepted optimization recipe.
- Failure of either preflight throughput gate falsifies feasibility before a
  result run. Do not weaken the gates or try widths 144, 176, or 192 after
  seeing timings.
- Divergent or non-finite training would show that the unchanged optimization
  recipe does not transfer. It would not justify changing LR inside this same
  experiment because that would destroy attribution.

## Implementation

Make the stage topology explicit rather than continuing to print a misleading
single widen factor. In `train.py`:

1. Replace `WIDEN_FACTOR = 2` with `STAGE_WIDTHS = (32, 64, 160)`.
2. Change `WideResNet.__init__` to accept `stage_widths`, validate that it has
   three positive integers, and use those values directly for `layer1`,
   `layer2`, `layer3`, final BatchNorm, and `fc`.
3. Instantiate `WideResNet(NUM_BLOCKS, STAGE_WIDTHS, NUM_CLASSES)`.
4. Print an unambiguous label such as
   `WRN-16 stages=[32,64,160] | params: 961,562`.

Do not introduce a generic topology abstraction beyond this small change. The
candidate must retain the same `PreActBlock`, projection-shortcut behavior,
weight initialization, and forward order as accepted.

Preserve the complete EXP-002 optimization and data recipe:

- batch 256 and `drop_last=True`;
- FP32 forward/backward with no autocast, compile, channels-last, or fused-SGD
  change;
- SGD, momentum 0.9, Nesterov, peak LR 0.2, 5% warmup from 0.002, and
  counted-time cosine decay to the 0.002 floor;
- `5e-4` decay only for convolutional and linear matrices, with no decay on BN
  and bias parameters;
- one device-resident `Beta(0.2, 0.2)` mixup coefficient per batch through 65%
  counted time, then hard-label cross entropy;
- crop, flip, CIFAR mean subtraction, seed 42, persistent workers, evaluation
  every fifth epoch plus the terminal epoch, and the finite-loss guard.

This isolation is important: BF16, larger batches, LR scaling, or stronger
augmentation may be valuable independent experiments, but combining them here
would prevent attributing any result to selective capacity.

## Matched Unscored H20 Preflight

Run one feasibility-only microbenchmark in a separate process. It must not load
the CIFAR test set, instantiate or call `Eval`, report accuracy, or write
`run.log`. It must benchmark exactly two configurations: accepted
`[32, 64, 128]` and candidate `[32, 64, 160]`. Do not perform a width sweep.

For each configuration, one at a time on the same NVIDIA H20:

1. Use synthetic CUDA inputs shaped `[256, 3, 32, 32]`, random class targets,
   the production parameter grouping, FP32 SGD/Nesterov, and the real alpha-0.2
   mixup forward/loss/backward/optimizer path.
2. Warm up 25 complete steps, synchronizing afterward so cuDNN selection and
   allocator startup are excluded.
3. Measure three windows of 50 complete steps. Synchronize only at each window
   boundary and use median window seconds per step.
4. Record finite loss, logits shape, exact parameter count, peak allocated
   memory, median step time, images/second, and coefficient of variation across
   the three timing windows.
5. Delete the first model and optimizer and clear their references before
   constructing the second. Keep batch, precision, loss, optimizer, and timing
   procedure identical.

Compute candidate throughput retention as
`accepted_median_step_ms / candidate_median_step_ms`. Calibrate projected
realized passes against the authoritative accepted run rather than trusting the
synthetic absolute rate:

```text
projected_passes = 141.9 * throughput_retention
```

Proceed to the scored run only if every preregistered gate passes:

- exactly one NVIDIA H20 is visible and used;
- both logits have shape `[256, 10]`, both losses remain finite, and no OOM or
  runtime error occurs;
- parameter counts are exactly 691,674 and 961,562;
- timing-window coefficient of variation is at most 5% for each path;
- candidate image throughput is at least 85% of accepted throughput; and
- calibrated candidate exposure is at least 120 passes in 300 seconds.

The two throughput thresholds deliberately demand a much smaller exposure loss
than rejected uniform width 3. The observed width-2/width-3 timing endpoints
give a prior of 90.3% retention and 128.1 passes for this 17%-MAC increase, so
the gate is demanding but plausible. If timing variability alone exceeds 5%,
repeat the same exact preflight once; do not change its width, batch, precision,
or thresholds. If either throughput gate fails on a stable measurement, stop
and mark this proposal infeasible without consuming the scored run.

## Scored Run and Verification

After a passing preflight:

1. Confirm the worktree diff modifies only `train.py`, `prepare.py` is
   unchanged, and all differences are the explicit stage-width implementation.
2. Run a no-evaluation shape/parameter smoke check and require `[256, 10]`,
   961,562 parameters, finite loss, and one successful optimizer step.
3. Remove any stale `run.log` and launch exactly one result run with
   `timeout 600s uv run train.py > run.log 2>&1`.
4. Require exit code 0, a complete final summary, approximately 300 counted
   training seconds, no more than 600 total seconds, and no more than one test
   evaluation in any epoch.
5. Confirm mixup disables exactly once near 195 counted seconds and the 35%
   hard-label tail retains the accepted 0.002 LR floor and continuous matrix
   decay.
6. Record best/final test accuracy, final test loss, steps, epochs, parameters,
   peak VRAM, and realized passes as `num_steps * 256 / 50_000`.

At the timing prior, the run should complete about 25,000 optimizer steps, 128
epochs, and 128 passes. These are diagnostics, not validity requirements; the
actual fixed-time run remains authoritative. Accept only a valid score of at
least 94.17%. A lower score is not rescued by parameter count, lower loss, or a
favorable intermediate evaluation.

## Risks and Interpretation

- **The final stage is not the capacity bottleneck.** Extra semantic channels
  may duplicate already sufficient features while the unchanged early stages
  bottleneck information. A stable negative run at adequate exposure rejects
  this allocation, not all possible architecture changes.
- **Exposure loss still dominates.** The MAC estimate omits BN, activation,
  optimizer, and kernel-shape effects. The matched preflight measures the real
  training-step cost before the result run; realized full-run passes provide
  the final interpretation.
- **Abrupt 64-to-160 transition.** The learned projection already handles the
  accepted 64-to-128 transition, but a wider fan-out may be harder to optimize.
  Kaiming initialization and pre-activation remain unchanged; do not silently
  add a bottleneck or tune LR to compensate.
- **More capacity can overfit.** Alpha-0.2 early mixup and continuous decay are
  retained because they are the validated regularizers. Worse final test loss
  despite finite training would indicate that 160 channels add variance rather
  than useful margin.
- **Preflight optimism.** Synthetic data omits DataLoader overhead. Ratio
  calibration against EXP-002 reduces this bias, but a valid scored run remains
  valid even if realized passes are below the projection.
- **Single-seed noise near the threshold.** Seed 42 and one scored run are
  mandatory. A 94.16% result is not an improvement and must not trigger a
  reroll.

## Evidence

- `knowledge/papers/wide-residual-networks.md`: width is an effective CIFAR
  representation lever when compute is allocated efficiently.
- `experiments/001/04-analysis.md`: WRN-16-2 plus time-aligned cosine raised
  accuracy to 93.38%, realized about 146 passes, and used only about 1.1 GiB.
- `experiments/002/04-analysis.md`: alpha-0.2 mixup through 65% reached the
  accepted 94.07% with 141.9 passes and final accuracy equal to best.
- `experiments/006/proposals/idea-01.md`: uniform WRN-16-3 measured 56.8% of
  accepted throughput and about 80.6 calibrated passes, while batch scaling
  recovered less than 6%; this proposal avoids widening the costly stages.
- `03-experiment-learnings.md` and `04-results.tsv`: six post-EXP-002 changes
  regressed at normal exposure, supporting a controlled architecture probe
  that preserves the accepted optimization and regularization recipe.
