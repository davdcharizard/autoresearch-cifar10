# Proposal: Batch 512 With a Fully Scaled LR Curve

## Verdict Up Front

This is a weakly evidenced but executable fixed-time operating-point test. Score
it only if batch 512 delivers at least **1.10x** the accepted complete-body image
rate on the H20, projecting at least **146.308096 dataset-equivalent passes** in
300 counted seconds. Below that gate, the candidate gives up roughly half of the
optimizer, BatchNorm, and batch-shared-mixup decisions without enough measured
exposure upside to justify a score. A passing preflight makes one fixed-seed run
worthwhile because the intervention is the large-batch counterpart to EXP-029,
uses otherwise idle memory, and jointly tests better kernel utilization plus a
linearly scaled SGD operating point. It is not a claim that exposure alone is
beneficial.

## Exact Intervention

Starting from accepted commit `67c8e98`, change only these constants in
`train.py`:

```python
BATCH_SIZE = 512       # accepted: 256
LR = 0.4               # accepted: 0.2
MIN_LR = 0.004         # accepted: 0.002
MAX_STEPS = 32000      # accepted: 64000
```

`MAX_STEPS` is a safety-cap rescale, not an optimization lever: both caps allow
16,384,000 processed images. Preserve `(2,2,3)`, FP32, seed 42, the exact
initialization, SGD with momentum 0.9/Nesterov, matrix-only `5e-4` weight decay,
5% warmup and time-cosine formula, alpha-0.2 batch-shared mixup through 65%, the
worker-private early RandAugment policy and exhausted-iterator cutoff, loader
flags, and every-fifth-plus-final evaluation. Do not add gradient accumulation,
retune momentum/decay/warmup, retain the old LR floor, or alter the evaluation
cadence.

Doubling both LR endpoints is the exact linear-scaling counterpart of the
accepted batch-256 curve and the symmetric direction to EXP-029's batch-128 /
half-LR treatment. With mean-reduced loss, twice the LR and approximately half
as many steps per example preserve first-order cumulative SGD displacement and
coupled weight-decay shrinkage per data pass. They do not preserve momentum's
example-domain horizon, gradient noise, BatchNorm statistics, or mixup refresh
frequency; those are intended components of this indivisible operating point.

## Fixed-Time Mechanism and Risks

At equal image exposure, batch 512 makes about half as many full-model updates
as batch 256. Each update averages twice as many samples, each BatchNorm estimate
uses twice as many images, and one batch-shared Beta coefficient applies to
twice as many examples. The doubled LR approximately preserves drift and the
usual linear-scaling diffusion argument, while the number and temporal spacing
of decisions still change. Larger, less noisy decisions may stabilize training,
but they may also remove useful stochastic regularization and coarsen the late
hard-label refinement that produced the accepted 94.32% boundary.

The positive case is hardware utilization, not memory relief. The accepted run
uses only 1096.3 MiB of a 97,871 MiB H20, and training compute is dominated by
forward/backward. Batch 512 may amortize launches and improve convolution kernel
efficiency enough to process materially more images. Local evidence makes a
small speedup insufficient: BF16 reached 159.07 passes at 93.81%, fixed-MAC
redistribution reached 171.70 at 93.82%, and late freezing reached 159.10 at
93.99%. Those interventions are not batch/LR equivalents, but they establish
that exposure alone does not overcome degraded optimization or representation.
The 1.10x gate therefore requires a real hardware upside before accepting the
large-batch stochasticity risk.

## Epoch, Evaluation, and RNG Consequences

`drop_last=True` changes a full epoch from `195 * 256 = 49,920` images to
`97 * 512 = 49,664`; the candidate drops 336 rather than 80 examples per
permutation. Every fifth epoch therefore evaluates after 248,320 candidate
training images versus 249,600 accepted, only 0.513% more frequently in the
example domain. Extra evaluations may still occur if higher throughput creates
more epochs. They are outside the counted 300 seconds but inside the 600-second
wall limit and create more opportunities for `best_test_acc`; success must
therefore be corroborated by the predetermined final accuracy condition below.
No epoch may be evaluated twice.

Require only construction-time RNG identity to the accepted source. After the
first iterator begins, exact trajectory identity is neither expected nor
desirable:

- batch packing and the larger dropped tail change which examples receive
  updates and when epoch permutations advance;
- half as many Beta samples and CUDA `randperm` calls occur per processed image,
  with each shared coefficient covering twice as many examples;
- BatchNorm statistics and optimizer states intentionally diverge;
- worker task assignment changes, so crop/flip and private RandAugment streams
  cannot be compared sample-for-sample with batch 256.

The RandAugment isolation contract must still hold at batch 512. Validate it
against a paired batch-512 crop/flip-only oracle with identical sampler order,
worker assignment, and base-transform RNG: consume a complete active iterator,
disable only after exhaustion, then require exact clean-tail replay and no
RandAugment marker leak. The shared active flag must never re-enable. Mixup must
disable on the first update whose counted-time progress is at least 65%; early
RandAugment must disable at the first exhausted epoch ending at or after that
boundary, with transition lag in `[0, 97)` steps.

## Fail-Closed Semantic Gate

Use an ignored local verifier with an independent
`git show 67c8e98:train.py` oracle. Before GPU timing, require:

- the complete production diff contains exactly the four constant changes;
- accepted and candidate model state, parameter groups, construction-time CPU
  and CUDA RNG states are byte-equal, with 987,098 trainable parameters;
- candidate logits have shape `[512, 10]`, one FP32 forward/backward/Nesterov
  update is finite, and peak allocation is safely below available H20 memory;
- `len(train_loader) == 97`, exactly 49,664 images are yielded, `drop_last`,
  persistent workers, pinning, prefetch, transforms, and worker context match;
- candidate LR is exactly twice accepted at progress
  `0, .025, .05, .5, .65, 1`, including the `0.4` peak and `0.004` floor;
- the paired batch-512 worker cutoff/replay test above passes without evaluator
  or test-set construction;
- source/control-flow checks prove unchanged budget accounting, at-most-once
  per-epoch evaluation, one mixup transition, and one exhausted RandAugment
  transition.

Abort without repair on any semantic, finite-value, scope, memory, worker, or
RNG-isolation failure. Do not infer a failure from intended post-iteration RNG
or update divergence.

## Fail-Closed Timing and Wall Gate

Benchmark accepted batch 256 and candidate batch 512 in the same local process
on one idle H20. For both the early mixup path and hard-label path, run at least
three balanced paired replicates with at least 20 warmup and 50 measured steps
per arm. Recreate deterministic model, optimizer, pinned-host inputs/targets,
and private timing RNG fixtures for every arm/window; alternate arm order. Time
the complete production `t0`-through-`torch.cuda.synchronize()` body, including
nonblocking H2D copies, LR writes, zeroing, the real mixup sample/permutation,
forward, loss, backward, and Nesterov step. Print every raw window, median, CV,
peak VRAM, and derived projection **before** enforcing assertions.

For each arm compute:

```text
image_rate = 0.65 * batch_size / mixup_step_s
           + 0.35 * batch_size / hard_step_s
retention = candidate_image_rate / accepted_image_rate
projected_passes = 133.00736 * retention
projected_steps = projected_passes * 50000 / 512
```

Proceed only if all timing values are finite, every regime/arm CV is at most
5%, `retention >= 1.10`, `projected_passes >= 146.308096`, and
`14,287 <= projected_steps < 32,000`. The step lower bound is redundant up to
rounding but makes the intended decision regime explicit. Never rerun stable
timing, lower the throughput gate, or select batch 384/1024 from the result.

Separately exercise fresh real batch-512 loaders in active and inactive
RandAugment phases at the measured candidate consumer pace. Require three
stable complete epochs, exactly 97 finite batches / 49,664 images, worker
cutoff correctness, and loader CV at most 5%. Project excluded loader stall and
evaluation/boundary wall cost using the accepted 345.3-second wall result and
the candidate's projected epoch/evaluation counts; require a conservative total
below 500 seconds. Abort on data starvation, worker error, or a wall projection
at or above 500 seconds even though loader waiting is excluded from counted
training time.

## Sole Score and Closure

After every gate passes, confirm baseline 94.32 at `67c8e98`, one idle H20,
local CIFAR-10, frozen `prepare.py`/evaluator, and only `train.py` modified.
Remove stale `run.log` and execute exactly once:

```bash
timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, one finite final summary, 300.0-300.1 counted seconds, total
under 600 seconds, 987,098 parameters, at least 146.308096 realized passes,
`14,287 <= num_steps < 32,000`, correct ordered transitions, and unique
every-fifth-plus-final evaluations. Do not react to interim accuracy and do not
rerun a valid completion.

Success requires both:

- `best_test_acc >= 94.42%`, the goal's +0.10-point margin over 94.32%;
- `final_test_acc >= 94.32%`, so extra evaluation opportunities cannot be the
  sole support for the result.

Record final loss versus 0.2523, best-final gap, realized passes and steps,
epochs/evaluation count, transition times/lags, peak VRAM, and counted/total
time. A valid miss closes exactly batch 512 with the `0.4 -> 0.004` curve and
32,000 cap. It does not isolate batch size from LR, BN, momentum horizon, mixup
refresh, dropped examples, or evaluation opportunity, and it does not authorize
an adjacent batch, LR, floor, momentum, or warmup retry.

## Hypothesis

If batch 512 improves H20 complete-body image rate by at least 10% and the
linearly doubled LR preserves useful per-example optimization despite fewer,
larger stochastic decisions, then the fixed-seed run will complete at least
146.308096 passes and raise `best_test_acc` from 94.32% to at least 94.42%, with
`final_test_acc >= 94.32%`. Otherwise the exact large-batch operating point is
closed after one stable feasibility result or one valid score.

## Local Evidence

- `experiments/027/03-execute.md` and `04-analysis.md`: accepted 94.32% best,
  94.22% final, 0.2523 loss, 133.00736 passes, 25,978 steps, 345.3 seconds wall,
  and only 1096.3 MiB VRAM.
- `experiments/029/03-execute.md` and `04-analysis.md`: the symmetric batch-128
  direction lost more than 9.78% image rate; its full-body timing protocol and
  fail-before-print mistake define the verification requirements here.
- `02-system-understanding.md`: forward/backward dominate counted time, memory
  and wall time are not binding, and evaluation/boundaries cost 44.2 seconds.
- `03-experiment-learnings.md`: exposure-only and update-reallocation treatments
  have repeatedly regressed; preserve full gradients, FP32, the 65% mixup
  cutoff, continuous decay, and the accepted early invariance/depth interaction.
- `project-notes/project-insights.md`: H20 CNN speed is shape-sensitive, so the
  batch-512 advantage must be directly timed rather than inferred from MACs or
  memory capacity.
