# Proposal: Mixup-Regularized WRN-16-3 With Batch-Scaled SGD

## Summary

Scale the validated EXP-002 model from WRN-16-2 to WRN-16-3 while retaining
early alpha-0.2 mixup and the 35% hard-label tail. Pair the wider model with
batch 384 and a linearly scaled `0.3 -> 0.003` peak/floor learning-rate range.
This is a focused test of whether the regularization that raised accuracy from
93.38% to 94.07% now makes additional representation capacity useful under the
same 300-second counted-training budget.

## Diagnosis and Hypothesis

EXP-001 established that moderate width is productive on this H20, but its
WRN-16-2 already approached zero training loss. Width alone was therefore a
poorly matched next step at that point: it would add fitting capacity while
reducing data exposure. EXP-002 changed that tradeoff. Early mixup directly
regularizes the high-LR fitting phase, then the final 35% uses clean labels for
margin refinement; it gained 0.69 percentage points and finished at its best
accuracy despite completing 141.9 passes. The current model also uses only
about 1.1 GiB of a roughly 97 GiB H20, so memory is not the limiting resource.

The hypothesis is that the validated temporal mixup schedule prevents the
extra WRN-16-3 capacity from merely memorizing, while width 3 raises the feature
ceiling enough to offset its lower image throughput. A successful run should
reach `best_test_acc >= 94.17%`, with a plausible expected range of 94.2-94.6%.
The result is especially convincing if final accuracy remains within 0.10
points of best after the hard-label tail.

## Exact Configuration

Change only these constants in the accepted EXP-002 `train.py`:

- `WIDEN_FACTOR = 3`, giving stage widths 48/96/192 and 1,549,530 trainable
  parameters at unchanged depth 16.
- `BATCH_SIZE = 384`.
- `LR = 0.3`.
- `MIN_LR = 0.003`.

Preserve all other behavior exactly:

- `NUM_BLOCKS = 2`, pre-activation blocks, projection shortcuts, initialization,
  global pooling, and classifier structure.
- Nesterov SGD with momentum 0.9, selective `5e-4` decay on matrix/tensor
  weights, and no decay on BN or bias parameters.
- Five-percent time warmup followed by cosine decay keyed to counted training
  seconds.
- One device-resident `Beta(0.2, 0.2)` scalar and one permutation per batch for
  progress `< 0.65`, followed by unchanged hard-label cross entropy.
- Seed 42, crop/flip augmentation, persistent workers, pinned memory,
  `drop_last=True`, evaluation every fifth epoch plus the budget-ending epoch,
  and all existing finite-loss, transition, progress, and summary logging.

Batch 384 is the smallest proportional batch increase from width 2 to width 3.
It gives wider kernels more work per launch and amortizes loader/optimizer
overhead without jumping to a very large batch that could erase useful SGD
noise. Peak LR 0.3 follows the same linear scaling already embodied by batch
256/LR 0.2; scaling the floor from 0.002 to 0.003 preserves the full schedule's
100:1 ratio rather than changing the late optimization regime independently.
Thus batch and LR form one utilization/optimization pairing required by the
width change, while mixup duration and strength remain controlled.

## Throughput Risk and Smoke Gate

Convolutional work grows roughly quadratically with width in most residual
blocks, so abundant VRAM does not imply abundant time. Batch 384 may improve
H20 utilization, but WRN-16-3 can still complete materially fewer passes than
EXP-002's 141.9. Too little exposure would make a negative result ambiguous
between insufficient optimization and ineffective capacity.

Before the full run, benchmark both the accepted WRN-16-2/batch-256 path and
the candidate WRN-16-3/batch-384 path in the same process and environment:

1. Use synthetic 32x32 CUDA tensors, the real optimizer, and the active mixup
   loss path, which is the slower phase and occupies 65% of training.
2. Run 20 untimed warmup steps, then 100 timed forward/backward/optimizer steps
   with CUDA synchronization for each configuration.
3. Compute projected 300-second passes as
   `300 / seconds_per_step * batch_size / 50_000`.
4. Proceed only if the candidate projects at least 80 dataset-equivalent passes
   and at least 60% of the matched WRN-16-2 image throughput. Also require
   finite loss, logits shape `[batch, 10]`, and no CUDA OOM.

The 80-pass absolute floor is deliberately below EXP-002's exposure but still
leaves about 10,417 optimizer updates at batch 384. The 60% matched-relative
floor protects against a misleading synthetic absolute projection while
rejecting a width step whose compute loss dominates any plausible capacity
gain. Record both timings and do not relax either threshold after observing the
candidate. The full-run interpretation must use realized
`num_steps * 384 / 50_000`, not the smoke projection.

## Expected Impact and Decision Rules

- **At least 94.17%, complete run:** improvement; regularized width is productive
  under the fixed budget and becomes the new baseline.
- **94.07-94.16% with at least 80 realized passes:** near-flat; added capacity is
  not worth its compute at this resolution, even though it is not a catastrophic
  regression.
- **Below 94.07% with at least 80 passes and stable loss:** reject the capacity
  hypothesis. Return to WRN-16-2 and tune mixup duration/strength or add a
  low-overhead averaging method.
- **Regression with fewer than 80 realized passes:** classify the configuration
  as compute-limited under 300 seconds; do not infer that width cannot help in a
  different throughput regime.
- **Non-finite loss early in warmup:** the scaled LR is unstable. The clean
  follow-up would retain width 3/batch 384 and test peak/floor `0.2/0.002`, not
  modify mixup or reroll the seed.

Expected VRAM remains only a few GiB, far below capacity. The primary risk is
fewer useful updates and passes, followed by larger-batch generalization loss;
both are observable through the smoke benchmark, realized exposure, loss
trajectory, and final-versus-best accuracy.

## Verification

1. Confirm the diff changes only `train.py` and only the four declared
   constants; `prepare.py`, evaluator logic, seed, and evaluation cadence remain
   untouched.
2. Run formatting/lint checks and a CUDA forward/backward smoke step; confirm
   parameter count 1,549,530 and model label `WRN-16-3`.
3. Execute and record the matched throughput gate above before launching the
   full experiment.
4. Remove stale `run.log`, then run exactly once on one H20 with
   `timeout 600s uv run train.py > run.log 2>&1`.
5. Require exit code 0, a complete final summary, `training_seconds` about
   300, `total_seconds <= 600`, and validation no more than once per epoch.
6. Confirm exactly one mixup-disable transition near 65% counted time and that
   subsequent training uses hard-label cross entropy.
7. Extract `best_test_acc`, `final_test_acc`, `final_test_loss`, `num_steps`,
   `num_epochs`, `peak_vram_mb`, and realized passes. Success requires
   `best_test_acc >= 94.17%`; use final accuracy and exposure to distinguish a
   stable capacity gain from a transient maximum or throughput-confounded null.

## Sources

- `knowledge/papers/wide-residual-networks.md`: shallow-wide residual networks
  are an effective CIFAR capacity allocation.
- `knowledge/papers/mixup.md`: convex sample/target interpolation regularizes
  classifiers and improves CIFAR generalization.
- `experiments/001/04-analysis.md`: WRN-16-2 improved the initial baseline and
  used only 1.1 GiB, validating moderate width and the hardware headroom.
- `experiments/002/04-analysis.md`: early alpha-0.2 mixup plus a 35% hard-label
  tail reached 94.07% with 141.9 passes and final accuracy equal to best.
