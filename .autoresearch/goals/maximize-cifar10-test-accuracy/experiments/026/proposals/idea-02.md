# Proposal: Batch-128 With a Proportionally Scaled LR Curve

## Summary

Retain the accepted WRN-16-2, crop/flip pipeline, batch-shared alpha-0.2
mixup through 65% counted time, optimizer, and evaluator, but reduce batch size
from 256 to 128 and halve the complete learning-rate curve from
`0.2 -> 0.002` to `0.1 -> 0.001`. Double `MAX_STEPS` from 64,000 to
128,000 only to keep that safety cap from becoming easier to hit after the
batch change. Run one fixed-seed score only if a matched preflight projects at
least 120 dataset-equivalent passes.

This is the most defensible smaller-batch operating point because batch 128 / LR
0.1 is the same LR-per-sample scaling from which the accepted batch 256 / LR
0.2 treatment was derived. Keeping LR 0.2 at batch 128 would simultaneously
double step aggressiveness and stochastic-gradient diffusion, making a one-run
failure difficult to interpret. Batch 384 / LR 0.3 points in the less promising
direction: fewer optimizer updates and lower batch noise in exchange for image
throughput, while EXP-009 and EXP-016 already show that substantially more
fixed-time exposure alone does not improve this model.

## Diagnosis

The accepted model reaches 94.07% after about 141.9 data passes and finishes
with near-zero training loss. Twenty-three scored or feasibility-rejected
follow-ups have not displaced it. More exposure is not the limiter: BF16 reached
159.07 passes at 93.81%, and the fixed-MAC `[1,2,3]` model reached 171.70 passes
at 93.82%. The closest positive changes instead alter representation or
optimization geometry, suggesting that the remaining gap is a generalization
boundary.

Smaller batches offer a controlled change to that boundary. At equal image
exposure, batch 128 gives approximately twice as many optimizer decisions,
noisier individual gradient and BatchNorm estimates, and twice as many
batch-shared mixup draws over groups half as large. Halving LR preserves the
first-order LR-per-sample scale and approximately preserves accumulated SGD
diffusion per example, so the hypothesis is specifically about finer update
granularity and smaller-batch stochastic regularization, not an unbounded noise
increase. Momentum's example-domain horizon and mixup-coefficient refresh do
necessarily change; they are part of the batch operating point and must be
reported rather than hidden as exact invariants.

One useful structural control is exact here: with `drop_last=True`, CIFAR-10
produces 195 batches of 256 or 390 batches of 128, and both epochs contain
49,920 examples. Epoch boundaries, discarded examples per epoch, and the
every-fifth-epoch evaluation cadence therefore retain the same example-domain
meaning. The warmup, cosine, and mixup cutoff remain keyed to counted seconds,
as required by the fixed-time objective.

## Exact Intervention

Change only these constants in accepted `train.py`:

```python
BATCH_SIZE = 128       # accepted: 256
LR = 0.1               # accepted: 0.2
MIN_LR = 0.001         # accepted: 0.002
MAX_STEPS = 128000     # accepted: 64000; nonbinding safety-cap rescale
```

Do not change `NUM_BLOCKS`, `WIDEN_FACTOR`, momentum, Nesterov, weight decay,
warmup fraction, cosine formula, mixup alpha/cutoff/law, transforms, loader
flags, evaluation cadence, seed, numerics, or evaluator calls. In particular,
allow no momentum retune, 0.002-floor fallback, gradient accumulation, or
intermediate LR after seeing preflight or score results. The 0.001 floor retains
the locally necessary nonzero late update
amplitude while keeping the *entire* candidate LR curve exactly half of the
accepted curve.

`MAX_STEPS` is not a second optimization lever. Leaving it at 64,000 would
halve the cap's example-domain allowance and could stop a fast batch-128 run at
163.84 passes before 300 counted seconds. Doubling it preserves the original
cap in processed examples and guarantees that normal completion remains
time-budgeted.

## Semantic Preflight

Before any scored run, use a separate local preflight process and require all of
the following:

- the production diff contains only the four constant changes above;
- the model remains WRN-16-2 with 691,674 trainable parameters and finite
  `[128, 10]` logits, loss, and gradients on one forward/backward step;
- `len(train_loader) == 390`, `drop_last=True`, persistent workers, crop/flip,
  seed 42, optimizer groups, momentum 0.9, Nesterov, and `5e-4` matrix decay are
  unchanged;
- candidate LR is exactly one half of accepted LR at progress
  `0, 0.025, 0.05, 0.50, 0.65, 1.0`, including `0.1` peak and `0.001` floor;
- mixup is one batch-shared `Beta(0.2, 0.2)` scalar per update before 65%, then
  the unchanged hard-label loss; the transition is still driven only by
  counted training seconds;
- `MAX_STEPS=128000` cannot bind under the timing projection, and validation is
  still at most once per epoch.

Do not require the accepted and candidate training RNG streams or updates to be
identical after batching begins. Different batch partitions, shuffle-iterator
consumption, mixup refresh frequency, and BatchNorm statistics are intended
consequences of the intervention. Model construction before iteration must
still be seed-42 reproducible.

## Matched Timing Gate

Benchmark accepted batch 256 and candidate batch 128 in the same process on the
same H20, with separate identically initialized models and optimizers. Use
production FP32 forward/backward/Nesterov steps, device-resident synthetic CIFAR
tensors, the real batch-shared mixup branch, and the real hard-label branch.
Give each configuration and branch at least 20 unmeasured warmups, randomize
measurement order with a private timing RNG, synchronize CUDA around each step,
and collect enough samples for branch CV at or below 5%.

Because the transition is keyed to counted time, compute each configuration's
weighted image throughput as
`0.65 * batch_size / mixup_time + 0.35 * batch_size / hard_time`. Project
candidate passes relative to the accepted 141.9-pass run:

```text
candidate_passes = 141.9 * candidate_images_per_second / accepted_images_per_second
candidate_steps  = candidate_passes * 50_000 / 128
```

Proceed only if projected candidate exposure is at least **120.0 passes**, all
timing CVs are at most 5%, `candidate_steps < 128000`, and the loader smoke test
produces stable `[128, 3, 32, 32]` batches without threatening the 600-second
wall limit. The 120-pass floor retains 84.6% of accepted image exposure while
still projecting at least 46,875 updates, about 1.69 times EXP-002's update
count. This preserves both sides of the proposed tradeoff. A lower projection
would confound smaller-batch stochasticity with severe underexposure and must be
classified as a feasibility failure without scoring. Do not lower the gate.

## One-Run Rule and Decision

After all preflight gates pass, remove stale `run.log` and execute exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Use seed 42 and one H20. Do not rerun a valid completion, reroll the seed, or
adapt batch size, peak LR, floor LR, momentum, or the exposure gate in EXP-026.
Success requires a fresh complete summary, 300 counted training seconds, total
wall time below 600 seconds, no more than one evaluation per epoch, and
`best_test_acc >= 94.17%`.

Record best/final accuracy and loss, steps, epochs, realized passes
`num_steps * 128 / 50_000`, mixup transition time/step, wall time, and peak VRAM.
A valid score below 94.17% rejects this exact batch-128 / `0.1 -> 0.001`
operating point; it does not authorize an LR retry. A valid improvement supports
smaller-batch update granularity as useful under the fixed-time budget, with the
stated caveat that BatchNorm and batch-shared mixup stochasticity co-vary.

## Hypothesis

If the accepted WRN's remaining error is limited by its batch/update operating
point rather than example exposure, batch 128 with an exactly halved
`0.1 -> 0.001` time-cosine LR curve will retain at least 120 projected passes,
complete at least 46,875 optimizer steps in 300 counted seconds, and increase
one fixed-seed `best_test_acc` from 94.07% to at least 94.17%.

## Local Evidence

- `experiments/001/04-analysis.md`: batch 256 / LR 0.2 established the accepted
  schedule family and ample H20 memory headroom.
- `experiments/002/04-analysis.md`: temporal alpha-0.2 mixup reached 94.07% at
  about 141.9 passes and remains the protected training law.
- `experiments/009/04-analysis.md` and `experiments/016/04-analysis.md`: 159.07
  and 171.70 passes scored only 93.81% and 93.82%, so exposure alone is not a
  sufficient direction.
- `03-experiment-learnings.md`: retain the 65% mixup cutoff, FP32 numerics,
  nonzero LR floor, continuous weight decay, and batch-shared coefficients.
- `project-notes/project-insights.md`: H20 throughput is shape-sensitive and
  must be measured rather than inferred from arithmetic or memory headroom.
