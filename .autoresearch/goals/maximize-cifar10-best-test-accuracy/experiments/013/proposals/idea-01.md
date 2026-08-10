# Idea: Batch-256 Fixed-Time Training with Linear LR Scaling

## Summary

Raise the accepted EXP-010 training batch from 128 to 256 and scale the complete
learning-rate curve by exactly two: `0.1 -> 0.2`, `0.01 -> 0.02`, and
`1e-4 -> 2e-4`. Keep momentum `0.9`, coupled all-parameter weight decay
`1e-4`, model, augmentation, CutMix, elapsed-time phase schedule, loader
lifecycle, evaluator, seed, and every other setting unchanged.

This is one coherent batch-scaling intervention, not a free batch-size sweep.
The linear LR rule is fixed before the accuracy result and applies to all three
LR levels. It approximately preserves parameter displacement and coupled-decay
shrinkage per dataset pass when the number of updates per pass halves. A paired
H20 diagnostic identifies batch 256 as the throughput knee: relative to batch
128, it processes 28.44% more images in the synthetic 300-second projection,
whereas batch 512 adds only another 5.1% of images and discards nearly half of
batch 256's remaining optimizer updates.

The candidate is deliberately a net fixed-time optimization test. Larger
batches reduce update count and gradient noise while increasing examples seen;
the scaled LR addresses first-order under-optimization but cannot make the two
stochastic trajectories equivalent.

## Diagnosis and Local Evidence

The current accepted frontier is EXP-010 at 94.15%. Its width-2 postactivation
ResNet-20, N1/M7 plus p=0.5 alpha-1 CutMix strong phase, and hard weak tail
completed 26,898 updates and processed about 3.44M batch slots in 300 counted
seconds. It switched at a healthy 89.73% clean checkpoint and finished at its
best with 0.1934 NLL. EXP-011 and EXP-012 show that adding strong-phase
suppression can cross the 87.08% underfit marker, so the accepted data recipe
and postactivation architecture must remain intact.

The EXP-013 system decomposition measures only 0.145 ms median real-loader wait
and 0.034 ms visible launch/synchronization gap, but 10.628 ms in model forward
plus backward. Backward alone is 8.220 ms, or 75.46% of the GPU-stage time.
Peak allocation is just 598.7 MB on a 97,871 MiB H20. A larger batch therefore
targets unused model-kernel parallelism and memory capacity rather than loader,
loss, optimizer, or Python overhead.

Batch 256 also has a useful exact data invariant. With 50,000 training examples
and `drop_last=True`:

```text
batch 128: floor(50,000 / 128) = 390 batches = 49,920 images/pass
batch 256: floor(50,000 / 256) = 195 batches = 49,920 images/pass
```

Both discard exactly the final 80 positions of each shuffled permutation. The
candidate changes batch grouping and update frequency, not the nominal number
of images in a complete dataset pass.

## Measured Batch-Size Knee

A serial H20 synthetic diagnostic measured the accepted model and optimizer at
three batch sizes. It used the complete synchronized training step and projects
the work possible in 300 seconds from the measured mean:

| Batch | Mean step | Projected steps | Projected images | Dataset passes | Peak allocation |
|---:|---:|---:|---:|---:|---:|
| 128 | 11.733 ms | 25,568 | 3.273M | 65.45 | 598.7 MB |
| 256 | 18.270 ms | 16,420 | 4.204M | 84.07 | 1,120.2 MB |
| 512 | 34.762 ms | 8,630 | 4.419M | 88.37 | 2,163.2 MB |

Batch 256 takes 1.5571x as long per update but carries twice as many images. It
therefore retains 64.22% of updates and raises image throughput by 28.44%. Batch
512 retains only 52.56% of batch 256's updates for 5.11% more images. That
defines batch 256 as the measured knee and excludes both an unmeasured midpoint
and a batch-512 accuracy gamble.

The absolute synthetic projection is deliberately conservative relative to
EXP-010's real 26,898 updates. Calibrating only the paired timing ratio to that
accepted exposure gives:

```text
26,898 / 1.5571 = about 17,274 candidate updates
17,274 * 256 = about 4.422M batch slots
4.422M / 49,920 = about 88.58 dataset passes
```

Planning should treat 16.4k-17.3k updates, 4.20M-4.42M image slots, and roughly
84-89 passes as the expected production range. The timing diagnostic cannot
predict accuracy, loader scheduling, or the exact mix of partial epochs.

## Exact Candidate

Starting from accepted commit `7c1e7d8`, change only these constants in
`train.py`:

```python
BATCH_SIZE = 256
LR = 0.2
ANNEAL_START_LR = 0.02
MIN_LR = 2e-4
```

No other source line should change. In particular, retain:

- width-2 postactivation ResNet-20 with Option-A shortcuts, 1,073,962
  parameters, FP32 execution, and existing initialization;
- N1/M7 crop/flip strong views and alpha-1 CutMix on probability 0.5 of strong
  batches, followed by crop/flip hard-label views;
- the 80% elapsed-time phase boundary and LR hold, the existing 10x LR drop at
  the boundary, and elapsed-time cosine tail;
- ordinary SGD, momentum 0.9, all-parameter coupled decay `1e-4`, no Nesterov,
  and the existing zero-gradient behavior;
- eight persistent forkserver workers, pinned memory, shuffling, `drop_last`,
  explicit strong-worker shutdown, garbage collection, and weak-loader rebuild;
- checkpoints `(0.2, 0.4, 0.6, 0.7)`, dense tail evaluation at most once per
  epoch, fixed `Eval`, seed 42, 300-second counted budget, 600-second timeout,
  and summary schema.

Do not add gradient accumulation, warmup, mixed precision, compilation, fused
SGD, channels-last, a second LR branch, or a batch-size-dependent CutMix rule.
Do not choose between LR 0.1 and 0.2 after seeing accuracy. Such changes define
different experiments.

## Learning-Rate and Integrated-Update Semantics

At batch 256 there are exactly half as many updates per complete 49,920-image
pass. With a locally stable average gradient and after momentum's transient, a
single SGD update contributes approximately `lr * gradient / (1 - momentum)`.
Doubling LR therefore approximately preserves accumulated gradient displacement
per pass:

```text
390 updates/pass * 0.1 = 39 learning-rate units/pass at batch 128
195 updates/pass * 0.2 = 39 learning-rate units/pass at batch 256
```

The same logic applies to the `0.01 -> 0.02` tail and, for consistency, the
`1e-4 -> 2e-4` cosine endpoint. The 10x discontinuity at the 80% boundary is
preserved exactly. With coupled weight decay unchanged, the first-order shrink
per pass is also approximately preserved because the product of update count,
LR, and decay remains the same.

Keeping LR at 0.1 would retain only 64.22% of accepted-like update displacement
over the fixed 300 seconds despite processing 28.44% more images. It would test
under-optimization and larger batches simultaneously. Linear scaling instead
turns the additional 28.44% dataset passes into approximately 28.44% more
integrated first-order displacement, matching the exposure mechanism.

This rule is not equivalence. Momentum remains 0.9, so its ten-update memory
spans twice as many images. Gradient covariance and CutMix batch composition
change, and finite-step curvature can make LR 0.2 less stable. Changing momentum
to `sqrt(0.9)`, adding warmup, or scaling weight decay separately would add
unvalidated degrees of freedom. They are intentionally excluded.

The no-warmup decision is also predeclared. This is only a 2x batch/LR change on
a residual model with BatchNorm, not the multi-node extreme-batch regime that
motivated warmup in large-minibatch literature. Warmup would consume part of
the already short 80% plateau and introduce another schedule parameter. Any
early instability is a genuine failure of this exact candidate, not grounds for
a rescue run.

## Expected Phase Exposure

Because both LR and augmentation phases are functions of counted seconds, the
projected batch-256 work divides approximately 80/20:

| Projection basis | Strong updates | Weak updates | Strong passes | Weak passes |
|---|---:|---:|---:|---:|
| Absolute 16,420-step projection | about 13,136 | about 3,284 | about 67.4 | about 16.8 |
| Ratio-calibrated 17,274 steps | about 13,819 | about 3,455 | about 70.9 | about 17.7 |

EXP-010 completed about 68.9 total passes, so the candidate should see roughly
15-20 additional passes while taking about 9.6k-10.5k fewer optimizer steps.
That is the central trade: extra data/augmentation exposure and better device
utilization versus fewer stochastic parameter updates.

## RNG, Batch, and Target Semantics

Retain the single process seed 42 and the existing `cutmix_collate` unchanged.
The `torch.random.fork_rng(devices=[])` scope must still contain both the p=0.5
gate and torchvision CutMix operation, restoring worker CPU RNG afterward so
the collator's extra draws do not leak into later worker transforms. The strong
loader alone may return either hard targets `[256]` or probability targets
`[256, 10]`; the weak loader must return only integer targets `[256]` and retain
the current assertion. Cross-entropy must accept both forms without a custom
loss or target cast.

The realized mixed-batch fraction should remain near 50%, but the candidate
executes only about 64% as many strong batch gates. Each mixed batch has twice
as many samples and CutMix pairings. Consequently, the number of mixed image
slots should rise with total image exposure even though the number of CutMix
decisions falls. Log the existing `cutmix_batches/strong_batch_count` provenance
without adding timed-path diagnostics.

Seed equality does not imply samplewise trajectory equality. Batch grouping
changes the CutMix permutations; different worker scheduling and the number of
samples consumed per collate can change crop, flip, RandAugment, and CutMix
draws. The sampler's nominal epoch still covers 49,920 positions, but pairings
and transforms differ. The result is the net fixed-seed batch-training method,
not a paired-minibatch causal comparison. Do not reseed, realign RNG streams, or
reroll a mechanically valid run.

## Mandatory Implementation and Functional Gates

Before any full run:

1. Confirm the diff from `7c1e7d8` contains exactly the four constant changes
   above and only tracked `train.py` is modified.
2. Compile and lint `train.py`; construct the model and require exactly
   1,073,962 trainable parameters and unchanged state-dict keys/shapes.
3. Require both strong and weak loaders to have length 195, batch tensors
   `[256, 3, 32, 32]`, `drop_last=True`, and exactly 49,920 consumed image slots
   in one complete iterator.
4. Across at least 1,000 strong batches in a disposable loader-only audit,
   require finite tensors, both target ranks, a realized mixed fraction in
   `[0.45, 0.55]`, probability-target rows summing to one within tolerance, and
   no soft target from the weak loader.
5. Run one candidate optimizer step with a hard target and one from a cleanly
   reconstructed state with a probability target. Require finite losses,
   gradients, parameters, and logits at LR 0.2.
6. Statically and dynamically prove exactly one strong-to-weak transition,
   eight stopped workers, no validation more than once per epoch, and no change
   to timer boundaries.

Any target, loader-length, numerical, lifecycle, or scope failure is a planning
defect and blocks the full run. Do not reduce the batch or LR to repair it
inside EXP-013.

## Mandatory Paired H20 Timing Gate

The serial diagnostic establishes feasibility but is not the final gate. On one
otherwise idle NVIDIA H20 near 97,871 MiB, benchmark the actual reviewed
candidate and accepted control in fresh processes. Use five paired trials with
alternating order. For each model, warm at least 100 complete steps and measure
500 synchronized steps using its production batch size, pinned host inputs,
nonblocking H2D, alternating hard and probability targets, ordinary FP32
forward, cross-entropy, backward, SGD, and terminal synchronization. Recreate
model and optimizer state for every trial.

Record trial mean, median, p95, images/s, loss finiteness, and peak allocation.
Use the median of the five trial means and require all of these:

- trial-mean CV below 3% for each batch size;
- batch-256 image throughput at least 1.20x batch-128 throughput, equivalently
  candidate/control mean-step ratio at most 1.6667;
- ratio-calibrated projection from EXP-010 of at least 16,139 updates and
  4.131M image slots, preserving at least 120% of its image exposure;
- candidate p95 image throughput at least 1.15x control p95 image throughput;
- candidate peak allocation below 1,500 MB and no numerical failure.

The observed 1.5571 step ratio, 28.44% image-throughput gain, and 1,120.2 MB
peak pass these gates with useful margin. If the reviewed implementation fails
the 20% image-exposure gate, do not proceed: its only intended systems benefit
is too small to justify losing roughly one third of optimizer updates. Do not
relax the gate or substitute batch 512.

## Mandatory Real-Loader and Wall Gates

Synthetic tensors do not prove that eight workers can sustain 256-sample
N1/M7+CutMix batches. Benchmark the exact production strong loader and the weak
loader after persistent-worker warmup, without changing their options. Measure
at least 500 delivered batches across iterator boundaries and record median/p95
iterator wait, batches/s, worker health, target ranks, and realized mixing.

Require:

- strong-loader delivery at least 1.25x the candidate GPU consumption rate
  implied by paired timing; the weak loader must meet the same gate;
- p95 iterator wait no more than 20% of candidate mean GPU-step time;
- a 500-step integrated production-path check to have
  `wall_time / counted_step_time <= 1.05`, with finite losses and all workers
  alive until explicit shutdown;
- rebuilding the weak loader to stop exactly eight strong workers and create no
  leaked process;
- predicted end-to-end runtime below 540 seconds, leaving at least 60 seconds
  before the mandatory timeout.

The evaluator remains fixed at batch 256 and the model is unchanged, so one
evaluation pass should cost the same as EXP-010. Training epochs become shorter
in counted time because there are only 195 updates per pass. The absolute and
ratio-calibrated projections imply about 16.8-17.7 weak passes. Including the
evaluation at the 80% switch, the terminal partial pass, and four earlier
checkpoints, expect about 22-23 total evaluator calls versus EXP-010's 19.

Planning must measure one fixed-evaluator pass and project wall time explicitly:

```text
projected_total = 300s counted training
                + measured loader/iterator wall gap
                + startup and loader-rebuild cost
                + projected_eval_count * measured_eval_pass_seconds
```

Use the conservative upper evaluation count implied by measured candidate step
time, and require the projection below 540 seconds. The accepted run finished in
330.7 seconds with 19 evaluations, so three or four additional unchanged-model
passes are expected to remain far below 600 seconds, but that expectation does
not replace the real-loader and evaluator measurements.

## Testable Hypothesis

**Primary hypothesis:** batch-256 training at an exactly 2x-scaled LR curve will
use H20 parallelism to process at least 20% more image slots while preserving
approximately the accepted integrated update and decay displacement per dataset
pass, improving `best_test_acc` from 94.15% to at least **94.25%** under the
otherwise unchanged EXP-010 protocol.

The measured exposure prior is 16.4k-17.3k optimizer steps, 4.20M-4.42M image
slots, 84-89 dataset passes, and 1.12 GB peak allocation. A plausible success
range is 94.25-94.45%. A larger prediction is not justified because larger
batches reduce stochastic regularization and update count, and the accepted
recipe is already near a one-seed plateau.

Mechanism-supporting diagnostics are a final strong checkpoint at or above the
87.08% underfit marker, first weak checkpoint near or above EXP-010's 93.16%,
roughly 50% realized CutMix batches, final NLL near or below 0.1934, and a rising
late trajectory. These explain the result but cannot override the primary
threshold. Only `best_test_acc >= 94.25%` with all integrity conditions is an
improvement.

## Failure Mechanisms and Interpretation

- **Too few stochastic updates.** The candidate keeps only about 64% as many
  optimizer decisions. Even with 28% more examples, fewer opportunities to
  change direction can limit fitting or terminal refinement.
- **Strong-phase underfit.** Larger batches average away gradient diversity, and
  N1/M7 plus CutMix is already demanding. A switch checkpoint below 87.08%
  would show that linear LR scaling did not compensate for the update loss.
- **LR-0.2 instability or sharpness.** Linear scaling is first-order. Curvature,
  BatchNorm dynamics, and momentum can make the doubled step overshoot or select
  a sharper solution. Nonfinite preflight blocks launch; finite but worse
  accuracy is a valid no-improvement, not grounds for LR tuning.
- **Reduced implicit regularization.** Larger batches have lower gradient noise.
  Training accuracy can rise while test NLL/accuracy worsen even if exposure
  and strong fit pass.
- **Momentum horizon mismatch.** Momentum 0.9 spans twice as many image slots at
  batch 256. This may smooth useful short-scale gradient variation or delay
  adaptation at the 80% switch.
- **Fewer CutMix decisions.** The same 50% gate applies to fewer, larger batches.
  More mixed image slots do not recreate the diversity of many independently
  sampled boxes and permutations.
- **BatchNorm statistic change.** Per-batch moments use twice as many examples
  and are less noisy. This may help evaluation or remove beneficial stochastic
  regularization, especially across the strong-to-weak distribution shift.
- **Dense-tail wall amplification.** Shorter 195-step epochs trigger more
  excluded-time evaluator passes. They do not consume the 300-second training
  budget but can increase wall time; the explicit projection and 540-second gate
  prevent a timeout.
- **Loader saturation.** N1/M7 transformations scale with image count. If eight
  workers cannot feed the faster image path, counted training may still finish
  but wall time rises and the throughput premise becomes misleading.
- **Single-seed uncertainty.** The formal 0.10-point gate is ten test images.
  One valid run estimates the net declared method, not a precise expected effect;
  never reroll or select another LR after the result.

## Verification and Decision Rule

After every gate passes, run exactly once on the confirmed idle H20:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
```

Require exit 0, finite summary fields, approximately 300 counted seconds, total
time below 600 seconds, one augmentation transition, eight clean worker exits,
unique evaluation epochs, no more than one evaluation per epoch, hard/soft
target provenance, exactly 1,073,962 parameters, batch-size-consistent steps,
and no unreviewed tracked diff. Preserve seed 42 and do not retry a mechanically
valid run.

- **Improvement:** accept only if `best_test_acc >= 94.25%` and every integrity
  condition passes.
- **Valid no-improvement:** revert all four constants if the correct run finishes
  below 94.25%; classify it using strong fit, first weak accuracy, NLL, steps,
  images, and timing.
- **Mechanical failure before timed learning:** repair only an implementation or
  environment defect that does not change the candidate, rerun the failed gate,
  and document it. Any LR, momentum, batch, augmentation, or schedule change
  requires a newly reviewed experiment.

## Attribution

The candidate bundles batch 256 with the mathematically predeclared 2x LR curve
because those settings jointly define standard linear batch scaling. Everything
else is fixed. The full run therefore attributes its net result to increased
batch/image throughput under approximately pass-preserving update magnitude,
including the unavoidable changes in update noise, BatchNorm moments, CutMix
pairings, worker RNG trajectory, and evaluation count.

It cannot isolate image exposure from reduced gradient noise or prove that LR
0.2 is independently optimal. It also cannot transfer batch 512's small extra
throughput to an accuracy claim. No post-result tuning, fallback LR, reroll, or
batch escalation is permitted.
