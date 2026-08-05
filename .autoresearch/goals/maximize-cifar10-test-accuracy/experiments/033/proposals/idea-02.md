# Proposal: Batch 512 With a Fully Scaled LR Curve

## Decision Summary

Treat batch 512 as one indivisible large-batch optimizer operating point, not as
an optimization-equivalent speed trick. It is worth a sole fixed-seed score only
if direct H20 timing shows at least **1.10x** the accepted complete-step image
rate, corresponding to at least **146.308096 dataset-equivalent passes** in 300
counted seconds. A smaller speedup would discard roughly half the accepted
optimizer, BatchNorm, and batch-shared-mixup decisions for too little exposure
upside. The local prior is deliberately cautious: several faster treatments
regressed, and no result yet shows that coarser large-batch decisions help this
generalization-limited model.

## Exact Production Change

Start from accepted commit `67c8e98` and change exactly four constants in
`train.py`:

```python
BATCH_SIZE = 512       # accepted: 256
LR = 0.4               # accepted: 0.2
MIN_LR = 0.004         # accepted: 0.002
MAX_STEPS = 32000      # accepted: 64000
```

The step cap remains image-equivalent: both configurations permit at most
`16,384,000` processed images. Doubling both LR endpoints is the predetermined
linear batch-size scaling of the complete accepted curve; it preserves the
accepted 100:1 peak-to-floor ratio, 5% warmup boundary, and time-cosine shape.
With mean-reduced loss, twice the LR and about half as many steps per image
approximately preserve first-order cumulative gradient displacement and
coupled-decay strength per image exposure. This is only a scaling rationale,
not a semantic-equivalence claim: Nesterov state, noise scale, BN statistics,
mixup refresh frequency, dropped examples, and the number of model updates all
change intentionally.

Preserve every other accepted component: `(2,2,3)` stage depth and width 2;
987,098 FP32 parameters; seed 42; initialization; SGD with momentum 0.9,
Nesterov, and matrix-only `5e-4` weight decay; alpha-0.2 batch-shared mixup
through 65% of counted time; worker-private one-op magnitude-5 RandAugment until
the first exhausted epoch ending at or after 65%; crop/flip/normalization;
loader flags; counted-time accounting; and every-fifth-plus-final evaluation.
Do not add accumulation, alter momentum/decay/warmup, change the evaluator, or
combine this with any other intervention.

## Exposure and Update Tradeoff

The accepted run completed 25,978 batch-256 updates and 133.00736 passes. At
equal exposure, batch 512 performs approximately half as many updates, BN
updates, Beta draws, and CUDA permutations. At the minimum 1.10x image-rate
gate, it projects:

```text
projected_passes = 133.00736 * 1.10 = 146.308096
projected_steps  = 146.308096 * 50000 / 512 = 14287.9
```

Thus the candidate gains at least 10% image exposure but still makes about 45%
fewer optimizer decisions than accepted. Each active mixup coefficient also
covers twice as many examples, and each BN estimate uses twice as many images.
Larger kernels may exploit the H20 better, but the lower decision count may
weaken stochastic regularization or coarsen the late hard-label refinement.

Epoch packing changes from `195 * 256 = 49,920` to
`97 * 512 = 49,664` images, so 336 rather than 80 examples are dropped from
each permutation. Every-fifth-epoch evaluation occurs after 248,320 candidate
images rather than 249,600 accepted images. This 0.513% change plus greater
throughput can create more evaluation opportunities; report the unique
evaluation count and best-final gap, but do not replace the registered primary
metric with an endpoint rule.

## Fail-Closed Semantic Gate

Use an ignored, experiment-local verifier and an independent
`git show 67c8e98:train.py` oracle. Before any comparative GPU timing, require:

- the tracked production diff is exactly the four constant changes above, with
  frozen `prepare.py` and evaluator behavior;
- accepted and candidate construction produce byte-equal model state and
  construction-time CPU/CUDA RNG states, with exactly 987,098 trainable FP32
  parameters; optimizer group membership, decay, momentum, and Nesterov flags
  are identical, with only the intended exactly doubled initial LR differing;
- candidate logits are `[512, 10]`, and a production-faithful FP32
  forward/loss/backward/Nesterov update is finite with safe H20 peak allocation;
- the loader has 97 batches, yields exactly 49,664 images, uses `drop_last`,
  pinning, eight persistent workers, prefetch 2, forkserver, and the exact
  accepted transform order;
- candidate LR is exactly twice the accepted LR at counted-time progress
  `0, 0.025, 0.05, 0.5, 0.65, 1.0`, including the `0.4` peak and `0.004` floor;
- a paired batch-512 active/crop-flip-only worker oracle proves identical
  sampler order, crop/flip decisions, and private base-transform RNG; after a
  fully exhausted active iterator, disabling RandAugment yields exact clean-tail
  replay with no marker leak and the shared flag never re-enables;
- source/control-flow checks preserve 300-second accounting, at-most-once
  evaluation per epoch, one mixup transition, and one exhausted-epoch
  RandAugment transition, whose scored lag must lie in `[0, 97)` steps.

Exact post-iterator RNG or optimization identity with batch 256 is neither
required nor expected: packing, worker assignment, Beta/permutation call count,
BN state, and optimizer state intentionally diverge. Abort without repair on
any scope, finite-value, memory, LR, loader, cutoff, or worker-isolation failure.

## Fail-Closed Throughput and Wall Gates

Time accepted batch 256 and candidate batch 512 in one local process on one
otherwise idle NVIDIA H20. For both the active mixup path and clean hard-label
path, use at least three balanced paired replicates per arm, at least 20 warmup
and 50 measured updates per replicate, alternating arm order. Recreate the
deterministic model, optimizer, pinned-host inputs/targets, and private timing
RNG fixtures for every arm/window. Measure the complete production
`t0`-through-`torch.cuda.synchronize()` body: nonblocking copies, LR writes,
zeroing, real Beta/permutation/mixup work, forward, loss, backward, and Nesterov
step. Emit every raw window, median, CV, peak VRAM, and projection before any
assertion.

Because the two paths occupy fixed fractions of counted **time**, compute:

```text
accepted_rate  = 0.65 * 256 / accepted_mixup_step_s
               + 0.35 * 256 / accepted_hard_step_s
candidate_rate = 0.65 * 512 / candidate_mixup_step_s
               + 0.35 * 512 / candidate_hard_step_s
retention       = candidate_rate / accepted_rate
projected_passes = 133.00736 * retention
projected_steps  = projected_passes * 50000 / 512
```

Advance only if every value is finite, every arm/path CV is at most 5%,
`retention >= 1.10`, `projected_passes >= 146.308096`, and
`14287 <= projected_steps < 32000`. Never rerun a stable miss, lower the gate,
or choose batch 384/1024 after seeing the result.

Separately exercise fresh real batch-512 loaders in active and inactive phases
at the measured candidate consumer pace. Require at least three stable complete
epochs per phase, exactly 97 finite batches and 49,664 images per epoch, correct
one-way cutoff behavior, no worker error or starvation, and CV at most 5% in
each phase. Use the accepted 345.3-second wall result and projected candidate
epoch/evaluation counts to form both differential and conservative absolute
wall projections; require each below 500 seconds. Print raw loader windows and
both wall projections before assertions. Any stable feasibility miss ends the
experiment without scoring or same-run rescue.

## Sole Scored Run and Verdict Rules

After all gates pass, verify the 94.32 baseline at `67c8e98`, one idle H20,
local CIFAR-10, frozen non-`train.py` production files, and absence of stale
`run.log`. Run exactly once, offline and locally:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Do not inspect interim accuracy to alter or repeat the run. Validity requires
exit 0, one finite final summary, `training_seconds` in `[300.0, 300.1]`, total
wall under 600 seconds, 987,098 parameters, at least 146.308096 realized passes,
`14287 <= num_steps < 32000`, unique every-fifth-plus-final evaluations, and
exactly one ordered mixup/RandAugment transition with RandAugment lag below one
97-step epoch.

The registered success criterion is solely
`best_test_acc >= 94.42%`, a 0.10-point gain over 94.32%. Report
`final_test_acc` versus 94.22, `final_test_loss` versus 0.2523, best-final gap,
evaluation count, realized passes/steps, epochs, transition times/lags, peak
VRAM, and counted/total time as corroboration rather than additional vetoes.
A valid result below 94.42 is no-improvement and must not be rerun.

A semantic or stable feasibility failure closes this exact implementation
without a score. A valid score closes exactly batch 512 with the
`0.4 -> 0.004` time-based LR curve and 32,000 image-equivalent cap. Neither
outcome authorizes an adjacent batch size, altered floor/peak ratio, momentum,
warmup, accumulation, seed, cutoff, or post-hoc combined treatment.

## Falsifiable Hypothesis

If batch 512 improves complete-body H20 image rate by at least 10% and the
fully doubled LR curve preserves useful optimization despite roughly 45% fewer
Nesterov/BN/mixup decisions, then the sole seed-42 run will realize at least
146.308096 passes in 300 counted seconds and raise `best_test_acc` from 94.32%
to at least 94.42%. Failure of a preregistered semantic/feasibility gate, or a
valid score below 94.42%, falsifies this exact operating-point proposal.

## Evidence and Risk Register

- EXP027 established the accepted 94.32% best, 94.22% final, 0.2523 loss,
  133.00736 passes, 25,978 steps, 345.3-second wall, and 1096.3 MiB peak VRAM.
- EXP029 showed batch 128 retained under 90.22% accepted image throughput and
  could not preserve its preregistered joint update/exposure regime; it does not
  predict batch 512 accuracy but supports fail-closed direct timing.
- EXP009, EXP016, and EXP028 reached 159.07, 171.70, and 159.10 passes yet
  regressed, so exposure alone is not a sufficient accuracy mechanism.
- Forward/backward consume about 98% of isolated step time and the H20 has
  roughly 98.9% allocation headroom, making batch size a plausible utilization
  lever while leaving large-batch generalization as the dominant risk.
- H20 CNN speed is shape-sensitive; measured throughput, not MAC or memory
  arguments, determines feasibility.
- Batch-shared mixup coherence was locally useful, but batch 512 doubles the
  number of examples sharing each coefficient. Larger BN batches and a longer
  example-domain momentum horizon are inseparable risks of this treatment.
