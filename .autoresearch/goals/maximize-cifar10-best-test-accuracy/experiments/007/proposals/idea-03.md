# EXP-007 Proposal: Batch 256 with Noise-Scale-Matched Learning Rates

## Proposal

Double the physical training batch from 128 to 256 on the accepted EXP-004 recipe and linearly scale every learning-rate level by 2x:

```python
BATCH_SIZE = 256
LR = 0.2
ANNEAL_START_LR = 0.02
MIN_LR = 2e-4
```

Preserve everything else: the 269,722-parameter ResNet-20, N1/M7 RandAugment through the first 80% of counted time, the deterministic switch to crop/flip-only inputs, hard-label cross-entropy, SGD momentum 0.9, weight decay `1e-4`, elapsed-time phase boundary, synchronized step timing, evaluation policy, worker lifecycle, seed 42, and fixed evaluator.

This is one mechanically coupled intervention, not an unconstrained batch/LR search. A batch-only change at `lr=0.1` would halve the leading-order SGD noise scale and cumulative parameter movement per data epoch, confounding accelerator throughput with an intentionally colder optimizer. Exact linear LR scaling preserves `lr / batch` at all schedule levels and is the most defensible first operating point. No fallback LR is chosen after observing the run.

**Measured preflight verdict:** the LR rule remains theoretically defensible, but physical batch 256 is not defensible for a full fixed-time EXP-007 run in the current eager synchronized implementation. A local synthetic benchmark measured 13.040 ms at batch 256, projecting only 23,006 updates, 60.0% of EXP-004's actual update count, for 5.89M samples. The corresponding batch-128 control measured 7.515 ms and projected 39,921 updates/5.11M samples. Batch 256 therefore sacrifices 42.4% of synthetic-control updates for only 15.3% more synthetic-control sample exposure. This fails the tightened gates below, so this proposal is a documented no-go unless a separately reviewed throughput intervention first changes the measured operating point.

## Local Baseline and Opportunity

The moving baseline is accepted EXP-004:

- `best_test_acc`: 92.30%; EXP-007 must reach at least 92.40%.
- 38,358 synchronized optimizer updates in 300.0 counted seconds.
- 99 reported epochs, 269,722 parameters, 330.1 MB peak VRAM, and 340.7 seconds total.
- Batch 128 presents `38,358 * 128 = 4,909,824` augmented samples, approximately 98.35 complete 49,920-sample training epochs.
- Mean synchronized step time is `300 / 38,358 = 7.82 ms`, or about 16,366 samples/s.
- The N1/M7 worker pipeline sustained 165.5-175.8 batches/s in EXP-004, while the synchronized GPU loop consumed about 127.9 batches/s.

The H20 has roughly 98 GB VRAM, so doubling this tiny model's activation batch is memory-trivial. The opportunity is to amortize kernel launch, Python, synchronization, and host-to-device costs across twice as many examples. The constraint is parameter-update exposure: even perfect linear compute scaling would halve the number of SGD decisions. EXP-003 already showed that losing only 6.7% of fixed-budget steps can erase the benefit of a plausible regularizer, and Hoffer et al. identify update count, not physical batch alone, as a central cause of the large-batch generalization gap.

The measured batch-256 point uses only 591.1 MB peak memory, confirming that VRAM is not the limiter. Compute scaling is: batch 256 takes 1.735x as long per update as batch 128 (`13.040 / 7.515`) while carrying 2x the samples. That yields only a 1.153x image-throughput gain. Against EXP-004's actual 38,358 updates and 4.91M samples, the projection is 40.0% fewer updates and 20.0% more samples. Neither comparison offers a large enough sample dividend to offset the update-count warning confidently.

## Learning-Rate Rule

Use exact linear scaling across the full accepted schedule:

| Phase | Batch 128 baseline | Batch 256 proposal |
|---|---:|---:|
| 0-80% RandAugment plateau | 0.1 | 0.2 |
| Tail entry at >80% | 0.01 | 0.02 |
| Tail minimum | 0.0001 | 0.0002 |

Smith and Le approximate SGD noise scale as `g ~= lr * N / batch`. The proposal preserves the batch-dependent factor exactly:

```text
0.1 / 128 = 0.2 / 256 = 0.00078125
0.01 / 128 = 0.02 / 256
0.0001 / 128 = 0.0002 / 256
```

At equal sample exposure, batch 256 performs half as many updates at twice the LR, also preserving first-order cumulative gradient and decoupled interpretation of the accepted 10x plateau-to-tail step. Goyal et al. successfully use linear LR scaling to address large-minibatch optimization. This repository's 2x increase is much smaller than their distributed setting; therefore no new warm-up is proposed. Warm-up would shorten the locally validated high-LR exploration regime and add a second schedule intervention.

Momentum remains 0.9. Changing momentum would alter both the effective averaging horizon and the noise scale. Weight decay remains `1e-4`: PyTorch SGD includes it in the gradient, so at equal sample exposure, half as many steps at twice the LR gives approximately equal cumulative shrinkage. More H20 sample throughput will increase total shrinkage, but adjusting weight decay preemptively would make the first result unidentifiable.

Primary sources:

- Hoffer, Hubara, and Soudry, *Train longer, generalize better* (NeurIPS 2017), find that large-batch generalization depends strongly on the number of parameter updates and propose Ghost BatchNorm as a separate mitigation: <https://arxiv.org/abs/1705.08741>.
- Goyal et al., *Accurate, Large Minibatch SGD*, motivate exact linear LR scaling and warm-up for much larger batch increases: <https://arxiv.org/abs/1706.02677>.
- Smith and Le, *A Bayesian Perspective on Generalization and Stochastic Gradient Descent*, derive the approximate `lr * N / batch` noise scale used here: <https://arxiv.org/abs/1710.06451>.

These sources justify the rule and the risk controls; they do not guarantee improvement for this fixed-time CIFAR run.

## Ghost BatchNorm Decision

Exclude Ghost BatchNorm from EXP-007.

Physical batch 256 changes each BatchNorm statistic from 128 to 256 examples, which reduces normalization noise and may weaken implicit regularization. Ghost BatchNorm with virtual batch 128 could preserve that aspect of EXP-004. It is not necessary in the first test for four reasons:

1. This is only a 2x batch increase, not the extreme regime for which Ghost BatchNorm is most compelling.
2. Implementing ghost statistics requires replacing every `nn.BatchNorm2d` or splitting each forward into two microbatches. The former changes model semantics and adds custom reduction overhead; the latter largely gives up physical-batch kernel utilization.
3. It would bundle a normalization intervention with batch/LR scaling, so a gain or loss could not be attributed to the throughput operating point.
4. Update-count loss remains even if BatchNorm noise is restored. The preflight gates attack the more immediate fixed-budget risk directly.

If a future systems intervention makes plain batch 256 pass the tightened throughput gates but it then regresses with stable optimization, Ghost BatchNorm becomes a justified *new* experiment. It must not be enabled adaptively within EXP-007. At the currently measured 23,006-update point, Ghost BatchNorm is not a rescue: it can restore batch-128 normalization statistics but cannot restore the missing 15,352 update decisions relative to EXP-004.

## Exact Throughput Microbenchmark and Measured Result

The required disposable synthetic synchronized-step benchmark has been run locally. It did not edit tracked files or launch a full training experiment. Its measured evidence is now the controlling feasibility result:

| Configuration | Mean step | Projected 300s updates | Projected samples | Relative updates | Relative samples | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Width 16, batch 128 | 7.515 ms | 39,921 | 5.11M | reference | reference | not limiting |
| Width 16, batch 256 | 13.040 ms | 23,006 | 5.89M | -42.4% | +15.3% | 591.1 MB |

EXP-004's actual full-run reference was 38,358 updates, slightly below the synthetic batch-128 projection. That close agreement (4.1% projection error) is good enough to treat the batch-256 projection as actionable. Even if batch 256 enjoyed the same favorable error direction, it would remain far below the tightened update gate.

The reproducible benchmark specification remains below so future throughput changes can be compared on the same basis. Any rerun must use a disposable, untracked `/tmp` process so its RNG and optimizer updates cannot affect a fixed-seed experiment.

### GPU timed-region benchmark

Benchmark the exact current `ResNet`, hard-label mean cross-entropy, SGD momentum 0.9/weight decay `1e-4`, CPU-pinned input tensors, nonblocking transfer, forward, backward, optimizer step, and terminal `torch.cuda.synchronize()`. This matches the code inside the current `t0`/`dt` interval. Augmentation is deliberately excluded because DataLoader work occurs before `t0` in the harness and is benchmarked separately.

Run both configurations:

- Control: batch 128, LR 0.1.
- Candidate: batch 256, LR 0.2.

For each configuration:

1. Confirm the only visible H20 is idle, set seed 42, and instantiate a fresh model/optimizer.
2. Allocate a reusable pinned CPU image batch of shape `[B, 3, 32, 32]` and integer targets; use fixed random values, not downloaded training examples.
3. Execute 100 warm-up training steps to settle cuDNN algorithm selection and allocator state.
4. Execute 500 synchronized timed steps and compute aggregate mean, median, p95, and samples/s.
5. Repeat five times with fresh model state, alternating order `128,256,256,128` across trials to limit temperature/order bias.
6. Report the median of the five aggregate means as `t128` and `t256`; also report worst trial and p95. Reject non-finite loss or greater than 5% timing coefficient of variation as an unstable diagnostic.

Compute conservative fixed-budget projections:

```text
projected_updates_256 = floor(300 / t256)
projected_samples_256 = 256 * projected_updates_256
sample_speedup = (256 / t256) / (128 / t128)
```

### Tightened hard go/no-go gates

The measured result shows the previous 65%-update/30%-sample proposal was too permissive on updates and still unmet on samples. In light of Hoffer et al.'s explicit update-count warning and EXP-003's local sensitivity to a 6.7% step loss, all of these stricter gates must pass before a full batch-256 run:

- At least 75% of EXP-004's actual updates: `projected_updates_256 >= 28,769`.
- `t256 <= 10.43 ms`, the timing equivalent of the 75% update floor.
- At least 50% more sample exposure than EXP-004: `projected_samples_256 >= 7,364,864`.
- Relative sample-throughput speedup at least 1.40x against the same-process batch-128 control.
- Peak allocated memory below 2 GB, finite loss, and timing coefficient of variation below 5%.

The observed point fails every performance gate: 23,006 updates, 13.040 ms, 5.89M samples, and 1.153x relative sample throughput. It passes only memory/stability. Therefore **do not launch the full EXP-007 training run** from the current code.

The 75% update floor still concedes one quarter of the accepted optimizer trajectory; linear LR scaling is the reason that concession may be tolerable. In exchange, the 50% sample gate requires a meaningful number of additional independently augmented views. A 42.4% update loss for only 15.3% more samples is the wrong side of this trade.

### Worker and total-wall benchmark

Run this second-stage benchmark only after a future GPU benchmark passes every tightened gate; the current measured point does not, so no additional preflight can authorize a full run. In a second fresh process, instantiate the exact EXP-004 strong and weak transforms and real forkserver DataLoaders at batch 256. After two warm-up epochs, time at least 1,000 strong batches and 400 weak batches, then exercise the actual strong-to-weak shutdown/rebuild path once.

Record batches/s, images/s, switch duration, worker PIDs, and whether all old workers terminated. Estimate total runtime as:

```text
gpu_batch_rate = 1 / t256
loader_limited_training_wall = 300 * max(1, gpu_batch_rate / strong_loader_batch_rate)
predicted_epochs = projected_updates_256 / 195
predicted_tail_evals = ceil(0.20 * predicted_epochs)
projected_total = startup + loader_limited_training_wall + switch + 2s * (4 + predicted_tail_evals + 1)
```

The two-seconds-per-evaluation allowance is conservative relative to EXP-004's observed 40.7 seconds of total non-training time across 25 evaluations plus startup/switch. Require:

- Strong loader throughput at least the larger of 70 batches/s or 80% of projected GPU batch rate.
- Weak loader throughput at least projected GPU batch rate.
- One switch in less than five seconds, exactly eight old workers stopped, and no live old PID.
- `projected_total <= 540 seconds`, preserving one minute of supervisor margin.

If a hard gate fails, do not run the full experiment and do not silently add workers, AMP, channels-last, compilation, or GPU augmentation. Those are different interventions.

## Expected Training Exposure and Schedule Behavior

Batch 256 gives 195 full batches per epoch and still presents exactly 49,920 samples per complete epoch, identical to batch 128. The accepted time schedule remains authoritative:

- RandAugment N1/M7 and `lr=0.2` through the batch that first crosses 240 counted seconds.
- Immediate epoch break, evaluation, verified shutdown of the strong workers, and weak-loader rebuild at the same 80% boundary.
- Weak crop/flip inputs at `lr=0.02` followed by cosine decay to `2e-4` over the final 60 counted seconds.

At the measured operating point, the run projects 23,006 updates, 5,889,536 samples, and 118.0 full-data equivalents. The time-based split yields only about 18,405 high-LR plateau updates and 4,601 low-LR tail updates, versus approximately 30,686 plateau and 7,672 tail updates in EXP-004. The model would see about 20% more samples than the actual accepted run, but receive 40% fewer parameter updates in both phases.

At the tightened gate, the minimum becomes 28,769 updates, 7,364,864 samples, 147.5 full-data equivalents, about 23,015 plateau updates, and 5,754 tail updates. Even this exchanges 25% of parameter-update decisions for 50% more augmented-image exposure. The measured implementation falls 5,763 updates and 1.48M samples short.

Keep `MAX_STEPS=64,000`; it cannot bind unless batch-256 step time falls below 4.7 ms. Keep the evaluation logic unchanged. More completed epochs will produce more dense-tail evaluations, but the accepted best-final gap has been small and the projected total-time gate protects the supervisor limit.

## Hypothesis and Expected Accuracy

**Original testable hypothesis, now rejected by preflight:** batch 256 with exact 2x LR scaling would retain enough updates while exposing substantially more N1/M7 views to improve `best_test_acc` from 92.30% to at least 92.40%. The measured point does not retain enough updates or add enough samples to support that mechanism.

If run unchanged despite the no-go result, a tighter evidence-based expectation is 91.95-92.35%, with failure to reach the 92.40% acceptance threshold more likely than improvement. Linear scaling preserves approximate diffusion *temperature* (`lr/batch`) but not diffusion *length*: Hoffer et al. find weight distance grows with update count and empirically associate the generalization gap with too few updates. Here the optimizer receives roughly 15,000 fewer direction changes than EXP-004. The 15.3% synthetic sample gain is too small to make a positive accuracy expectation defensible.

Only if a separately reviewed throughput change first satisfies the tightened gates would the prior 92.35-92.60 candidate range become reasonable. Such a change is outside this isolated proposal.

## Risks

- **Update-count generalization gap.** Even noise-scale-matched LR cannot manufacture missing SGD decisions. The measured 23,006 updates are only 60.0% of EXP-004, directly activating the primary-literature warning. This is now a demonstrated blocker, not a hypothetical risk.
- **LR 0.2 instability.** Exact scaling may be too aggressive from initialization. The disposable benchmark checks finite optimization but not generalization. A valid unstable full run is a no-improvement; do not add warm-up or retry.
- **Reduced BatchNorm noise.** Statistics over 256 images may generalize differently. Ghost BatchNorm is explicitly deferred to preserve isolation and cannot correct the measured update-count deficit.
- **RandAugment worker bottleneck.** Batch 256 doubles images requested per iterator yield while N1/M7 is CPU/PIL-heavy. Loader wait is outside `training_seconds` but inside total wall time; benchmark and the 540-second projection gate prevent a timeout.
- **Fewer tail optimizer updates.** At the measured point, the tail lasts the same 60 counted seconds but has only about 60% as many updates. Scaling tail LR preserves approximate movement/noise, yet fine-grained convergence and update-driven diffusion still lose roughly 3,071 tail decisions.
- **More dense-tail evaluations.** Higher epoch throughput increases evaluation count and best-metric opportunities. Evaluation policy remains unchanged rather than selectively reduced, and all epochs must remain unique.
- **Integrated weight decay increases if throughput is superlinear.** More scaled-LR updates increase cumulative shrinkage. Adjusting weight decay would add a confound; diagnose train loss after the fixed run instead.
- **Seed stream changes.** A different batch partition necessarily changes augmentation and shuffle consumption. Seed 42 remains fixed and no reroll is allowed; exact effect size cannot be separated from the inherent batch change in one run.
- **Microbenchmark optimism.** Repeated synthetic batches omit augmentation and iterator variability. The separate real-loader measurement and conservative total-time projection cover this gap; final acceptance uses actual run metrics.

## Confound Controls

- Change only `BATCH_SIZE`, `LR`, `ANNEAL_START_LR`, and `MIN_LR` in tracked `train.py`; optional descriptive startup logging may report them without changing behavior.
- No model, BatchNorm, loss, optimizer type, momentum, weight decay, augmentation, phase boundary, worker count, evaluation, timer, seed, or evaluator changes.
- No AMP, TF32 tuning, channels-last, compilation, gradient accumulation, Ghost BatchNorm, or GPU-resident data.
- Use one fixed-seed full run only after preflight passes. A failed valid run is not retried with a new LR or seed.
- Derive sample exposure as `num_steps * 256` and full-data equivalents as `num_steps / 195`; do not compare optimizer steps alone.
- Retain the exact crossing-batch and loader-switch behavior from EXP-004 and verify one `randaugment->base` switch near 80.0%.

## Implementation Sketch

The intended tracked behavioral diff is only:

```diff
-BATCH_SIZE = 128
-LR = 0.1
-ANNEAL_START_LR = 0.01
-MIN_LR = 1e-4
+BATCH_SIZE = 256
+LR = 0.2
+ANNEAL_START_LR = 0.02
+MIN_LR = 2e-4
```

The current loader factory, switch helper, schedule formula, and training loop already adapt to the new constants. Do not change `drop_last=True`: both 390x128 and 195x256 present exactly 49,920 samples per complete epoch.

## Full-Run Verification

No full training run is part of proposal development, and the measured preflight says not to select the current operating point for execution. If a separately reviewed systems change later improves the same benchmark enough to pass all tightened gates:

1. Confirm moving baseline 92.30% and acceptance threshold 92.40%.
2. Confirm exactly one idle NVIDIA H20 with approximately 98 GB VRAM.
3. Pass every GPU, loader, lifecycle, exposure, and projected-total preflight gate above.
4. Verify tracked diff changes only `train.py` and exactly the four constants.
5. Run syntax compilation, Ruff, pre-commit, parameter-count assertion (269,722), and loader-length assertion (195).
6. Remove stale logs and launch once as `uv run train.py > run.log 2>&1` under a 600-second supervisor.
7. Require exit zero, one finite ten-field summary, approximately 300 counted seconds, total below 600 seconds, and unchanged parameter count.
8. Require exactly one strong-to-weak switch near 80%, eight stopped workers, and no evaluation epoch duplicated.
9. Require actual `num_steps >= 28,769` and actual sample exposure `num_steps * 256 >= 7,364,864`; the measured preflight does not support launching a run expected to meet these gates.
10. Require `best_test_acc >= 92.40%` for improvement. Report best/final gap, final loss, epochs, synchronized step time, samples/s, VRAM, and total time against EXP-004.
11. Remove `run.log` after analysis.

## Decision Rules

- **Accept:** accuracy at least 92.40%, every integrity condition passes, at least 28,769 updates complete, and sample exposure reaches 7,364,864. Batch 256 with scaled rates becomes the moving recipe.
- **Accuracy failure with gate success:** reject this batch/LR operating point. The result is evidence that missing updates or changed batch statistics outweigh extra sample exposure; Ghost BatchNorm may be separately brainstormed, not retrofitted.
- **Accuracy gain with update/sample gate failure:** the formal metric may improve, but the predeclared mechanism is unsupported. Review timing integrity before accepting and do not claim throughput-equivalent scaling.
- **Preflight failure (current verdict):** 13.040 ms, 23,006 projected updates, and 5.89M samples fail the tightened gates. Do not spend the full run; record this operating point as fixed-time infeasible and return to an orthogonal candidate.
- **Runtime/lifecycle failure:** invalid. Revert to EXP-004 and diagnose mechanics without changing LR or seed in the failed experiment.
