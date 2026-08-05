# Proposal: Isolated WRN-16-3 Capacity Increase

## Summary

Increase only `WIDEN_FACTOR` from 2 to 3 in the accepted EXP-002 training
recipe. This changes the model from WRN-16-2 (stage widths 32/64/128, 691,674
parameters) to WRN-16-3 (48/96/192, 1,549,530 parameters) while retaining
batch 256, the `0.2 -> 0.002` time-based cosine schedule, selective `5e-4`
weight decay, and alpha-0.2 mixup through 65% of counted training.

The required matched, unscored H20 preflight has now rejected this candidate:
width 3 retained only 56-57% of width-2 image throughput and projects about
80 realized-pass equivalents after calibration to EXP-002, below the proposed
60% and 85-pass feasibility gates. A batch-size sweep recovered less than 6%
throughput. The isolated capacity hypothesis is scientifically crisp, but its
fixed-budget compute tradeoff is poor enough that it should not consume the
EXP-006 result run.

This is deliberately different from the earlier unexecuted width-3 proposal,
which coupled width 3 to batch 384 and LR 0.3. Batch and LR scaling would change
SGD noise, optimizer-step count, mixup draws, and batch-normalization statistics
at the same time as capacity. Holding them fixed makes width the only treatment;
its unavoidable throughput cost is part of what this fixed-time experiment is
meant to measure.

## Diagnosis

The current 94.07% baseline already has a strong optimization recipe. WRN-16-2
plus time-aligned cosine improved the original result from 91.54% to 93.38%,
and early mild mixup with a hard-label tail then improved it to 94.07%. Three
subsequent augmentation changes all regressed despite normal exposure:
CutMix scored 93.72%, a 50% mixup cutoff scored 93.91%, and alpha-0.4 mixup
scored 93.57%. This makes further nearby augmentation tuning less attractive.

Capacity is the clearest untested orthogonal lever. The accepted model uses
only about 1.1 GiB on a 97.9-GiB H20, and the Wide Residual Networks result
supports allocating CIFAR compute to wider, relatively shallow residual
networks. The limiting resource is not memory, however: it is useful data
exposure in 300 counted seconds. EXP-002 completed 27,735 batch-256 updates, or
141.9 dataset-equivalent passes. Width 3 has about 2.24 times as many parameters
and substantially more convolutional work. It may use the H20 more efficiently,
but it will still reduce optimizer steps and passes.

The actual question is therefore not "does a larger network fit?" It plainly
does. The question is whether the representation gain from width 3 is larger
than the convergence loss from lower exposure under the frozen 300-second
budget.

## Hypothesis

The validated alpha-0.2 mixup phase will regularize the additional WRN-16-3
capacity, and the final 35% hard-label phase will refine its clean-label
decision boundary. If the candidate retains at least roughly 60% of the
accepted path's image throughput, its richer features will offset the lower
pass count and produce `best_test_acc >= 94.17%`.

This mechanism is falsifiable:

- A complete run at or above 94.17% supports useful capacity under the fixed
  budget.
- A complete run below 94.07% with stable finite loss rejects WRN-16-3 as a
  better fixed-budget allocation, even if the proximate cause is its lower
  exposure. That compute-capacity tradeoff is intrinsic to the treatment.
- A regression accompanied by early loss instability would instead falsify
  the assumption that the unchanged LR transfers to width 3 and would motivate
  an LR-only follow-up, not a batch-size change in this experiment.

## Exact Code Scope

Starting from accepted commit `eb08811`, change exactly one line in `train.py`:

```python
WIDEN_FACTOR = 3
```

Preserve all other behavior exactly:

- `NUM_BLOCKS = 2`, so depth remains 16 and only width changes.
- `BATCH_SIZE = 256`, `LR = 0.2`, `MIN_LR = 0.002`, 5% warmup, and cosine
  progress driven by counted training seconds.
- SGD with momentum 0.9, Nesterov, and `5e-4` decay only on convolutional and
  linear weights.
- One device-resident `Beta(0.2, 0.2)` scalar per batch before 65% progress,
  then the unchanged hard-label path.
- Seed 42, crop/flip transforms, persistent workers, evaluator, evaluation
  cadence, finite-loss check, and logging.

The expected parameter count is 1,549,530. Do not change batch size to 384 or
scale LR to 0.3: width does not invoke the linear batch-size rule, and those
changes would prevent clean attribution.

## Unscored Throughput Preflight

Use one feasibility-only microbenchmark before the full experiment. It must
not load the CIFAR test set, call `Eval.evaluate`, report an accuracy, or write
`run.log`; therefore it is not an experiment result run. Run it in a separate
process so its models, optimizer state, CUDA state, and RNG consumption cannot
affect the later seed-42 training process.

Benchmark WRN-16-2 and WRN-16-3 in the same process using synthetic
`[256, 3, 32, 32]` CUDA inputs, random class targets, the real SGD parameter
groups, and the real alpha-0.2 mixup loss path. For each width:

1. Warm up 25 complete forward/backward/optimizer steps, including mixup and a
   CUDA synchronization, so cuDNN selection and allocator startup are excluded.
2. Measure three windows of 50 complete steps, synchronizing at each window
   boundary. Use the median seconds per step, not the fastest window.
3. Record finite loss, logits shape, peak allocated memory, median step time,
   images/second, relative candidate throughput, and projected 300-second
   passes: `300 / median_step_seconds * 256 / 50_000`.
4. Delete benchmark models after recording the values. Do not tune the model,
   batch, or gate based on observed timings.

Proceed to the scored run only if all predeclared gates pass:

- the device is one NVIDIA H20;
- both paths have finite loss and logits shape `[256, 10]`;
- no CUDA OOM occurs;
- width 3 retains at least 60% of width 2 image throughput; and
- width 3 projects at least 85 dataset-equivalent passes in 300 seconds.

The relative gate controls synthetic-benchmark bias because loader overhead is
common to both configurations. The absolute gate rejects a configuration with
too little optimization exposure to be a sensible accuracy bet. Based on the
2.24x parameter increase but likely better utilization of wider convolutions,
a reasonable prior is 60-75% retained throughput, or about 85-106 projected
passes relative to EXP-002's 141.9 realized passes. This estimate is uncertain;
that uncertainty is exactly why the preflight is required.

If either throughput gate fails, do not consume the sole result run. Mark this
proposal infeasible for EXP-006 and select a lower-overhead candidate. The gate
is a planning check, not an opportunity to weaken thresholds or try several
width/batch combinations.

### Observed Preflight Results

The local single-H20 preflight failed both gates. In a matched batch-256 run
with 10 warmup steps and 50 timed full SGD steps, WRN-16-2 took 12.488 ms/step
and WRN-16-3 took 21.972 ms/step. Width 3 therefore retained only 56.8% of
width-2 step and image throughput. The naive synthetic projections were 123.0
passes for width 2 and 69.9 for width 3. Correcting the relative ratio against
EXP-002's authoritative 141.9 realized passes predicts about 80.6 passes for
width 3 (`141.9 * 0.568`), still below the 85-pass gate and representing an
approximately 43% exposure cut.

A follow-up synthetic batch sweep corroborated that this is model compute, not
an easily repaired occupancy problem:

| Configuration | Images/s | Projected passes |
|---|---:|---:|
| WRN-16-2, batch 256 | 20,629 | 123.8 |
| WRN-16-2, batch 512 | 21,502 | 129.0 |
| WRN-16-3, batch 256 | 11,626 | 69.8 |
| WRN-16-3, batch 512 | 12,153 | 72.9 |
| WRN-16-3, batch 768 | 12,298 | 73.8 |

Moving width 3 from batch 256 to 768 recovers only 5.8% image throughput and
would additionally change optimizer-step count, batch-normalization statistics,
SGD noise, and the appropriate LR. It does not rescue the capacity tradeoff and
would make the scored test less attributable. These unscored measurements are
far enough below the gates, and mutually consistent enough, that repeating a
longer microbenchmark would not change the decision.

## Full Run and Verification

Only after a passing preflight:

1. Confirm `git diff -- train.py` contains only the one-line width change and
   `prepare.py` is unchanged.
2. Confirm the model prints `WRN-16-3`, 1,549,530 parameters, and uses one H20.
3. Remove any stale `run.log` and execute exactly once with
   `timeout 600s uv run train.py > run.log 2>&1`.
4. Require exit code 0, `training_seconds` approximately 300, total wall time
   no more than 600 seconds, a complete final summary, and no more than one
   evaluation per epoch.
5. Confirm mixup disables exactly once near 195 counted seconds (65%) and the
   remaining training uses hard labels.
6. Record `best_test_acc`, `final_test_acc`, `final_test_loss`, `num_steps`,
   `num_epochs`, `peak_vram_mb`, and realized dataset passes computed as
   `num_steps * 256 / 50_000`.

Success requires at least 94.17%, the current 94.07% baseline plus the mandated
0.10 percentage-point margin. Final-versus-best accuracy and test loss should
show whether any gain is stable. Compare realized passes to both EXP-002's
141.9 and the preflight projection, but do not dismiss a valid negative run
solely for lower exposure: the objective is accuracy inside this time budget.

## Risks and Interpretation

- **Compute dominates capacity:** This is the principal risk. The feasibility
  gate avoids an obviously poor allocation, while the scored run measures the
  real tradeoff including data loading and evaluation boundaries.
- **Unchanged LR is suboptimal:** Width alone does not mathematically require
  LR scaling, so LR 0.2 is the cleanest controlled choice. Divergent or erratic
  early loss would justify a later LR-only experiment; stable loss plus a lower
  score would reject this fixed recipe.
- **More capacity overfits:** Early mixup is already validated on WRN-16-2 and
  is more likely to make added capacity useful than an unregularized width
  trial. Worse test loss despite adequate training exposure would indicate
  that width adds no generalization benefit here.
- **Microbenchmark optimism:** Synthetic inputs omit the loader. The matched
  ratio and conservative absolute pass gate mitigate this, and realized passes
  remain the authoritative diagnostic.
- **Run variance:** Retain seed 42 and perform one scored run. Do not reroll a
  near miss.

## Why This Should Be EXP-006

Before measurement, this was a reasonable EXP-006 candidate: the search had
three consecutive augmentation regressions, moderate width was part of the
largest accepted gain, and the machine has exceptional memory headroom. The
one-line change also preserved the best regularization recipe and offered clean
attribution.

After measurement, it should not be EXP-006. Width 3 misses both feasibility
gates and cuts expected exposure by about 43%; batch 512-768 recovers less than
6% image throughput while introducing optimizer confounds. Although a scored
run could still surprise, the capacity gain would need to overcome roughly 61
fewer dataset passes than EXP-002. That is a weaker bet than a low-overhead
orthogonal method such as carefully windowed weight averaging. Do not weaken
the gates after observing them, do not combine width 3 with batch 384/LR 0.3,
and do not spend the single EXP-006 result run on this proposal.

## Evidence

- `knowledge/papers/wide-residual-networks.md`: shallower, wider residual
  networks are effective CIFAR representations and can allocate compute more
  efficiently than very deep thin networks.
- `experiments/001/04-analysis.md`: WRN-16-2 plus a time-aligned schedule raised
  accuracy from 91.54% to 93.38%, completed about 146 passes, and used only
  about 1.1 GiB of H20 memory.
- `experiments/002/04-analysis.md`: alpha-0.2 mixup through 65% reached 94.07%
  with 141.9 passes and final accuracy equal to best, validating the recipe to
  retain around a width change.
- `03-experiment-learnings.md` and `04-results.tsv`: CutMix, earlier mixup
  removal, and stronger mixup all regressed at normal exposure, while model
  capacity beyond width 2 remains untested in a scored run.
- Local EXP-006 feasibility measurements: matched full-SGD synthetic timing
  found 12.488 ms/step for width 2 and 21.972 ms/step for width 3; a batch sweep
  showed at most 12,298 images/s and 73.8 synthetic projected passes for width
  3, insufficient to pass the preregistered throughput gate.
