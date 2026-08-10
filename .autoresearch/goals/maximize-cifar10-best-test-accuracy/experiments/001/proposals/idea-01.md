# Idea: Budget-Aligned Cosine SGD with Nesterov and a Gated Averaging Extension

## Summary

Replace the step-indexed `MultiStepLR` with a single monotone cosine cycle driven by measured training time, and enable Nesterov momentum. The baseline assumes a 64,000-step horizon, but the measured 300-second run reaches only 38,525 steps: it crosses the first milestone at step 32,000, spends only about 6,525 steps at `lr=0.01`, and terminates 9,475 steps before the planned `lr=0.001` milestone. Consequently, the nominal three-phase schedule is compressed into `0.1` for about 83% of the actual budget and `0.01` for the remaining 17%, with no low-learning-rate convergence phase.

The proposed schedule uses elapsed counted training time as the source of truth, so it always traverses the complete annealing curve even if throughput changes. A first controlled experiment should test only the time-aware cosine schedule and Nesterov. EMA or SWA is a plausible follow-up, but it should not be enabled in the first run because BatchNorm handling and the already tight 595.4-second total runtime create distinct risks and would obscure whether horizon alignment itself caused the result.

## Limiter Diagnosis

- Baseline `best_test_acc` is 91.67% after 38,525 optimizer steps.
- The step budget implied by runtime is about 128.4 steps/s. Step 32,000 occurs at roughly 249 seconds and step 48,000 would occur around 374 seconds, outside the fixed 300-second training budget.
- Thus, the effective schedule is badly mismatched to the actual optimization horizon. It preserves a high learning rate for most of the run and never reaches its intended final refinement regime.
- This is a more direct fixed-budget limiter than adding regularization alone: label smoothing may improve generalization, but it does not repair the missing convergence phase. It should remain disabled in the first test to control the optimizer/schedule comparison.
- Total runtime is 595.4 seconds even though counted training time is 300 seconds, presumably because per-epoch evaluation is excluded from the training budget. Any extension that adds a data pass or extra evaluations risks crossing the 10-minute failure threshold.

## Mechanism

For each optimizer update, compute progress from training time already consumed:

```text
p = clamp(total_training_time / TIME_BUDGET_S, 0, 1)
lr(p) = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1 + cos(pi * p))
```

Set the optimizer learning rate immediately before the forward/backward/update for that batch. `total_training_time` contains only previously completed timed updates, so it is available without synchronization or forecasting and is consistent with the harness's counted budget. The last completed updates will be close to `MIN_LR` regardless of whether the run completes 35,000 or 45,000 steps.

Use one monotone cycle, not SGDR restarts. Restarts are useful for longer anytime or snapshot-ensemble settings, but reheating late in a single 300-second run would spend scarce final updates moving away from the converged basin. The cosine shape gives a long, smooth transition and a genuine terminal refinement phase without discontinuous milestone choices.

Enable SGD Nesterov momentum with the existing `momentum=0.9` and `weight_decay=1e-4`. Nesterov uses the same parameter state and has negligible runtime or memory overhead. It provides a modest anticipatory correction while the learning rate is high and is a standard fit for a smoothly annealed SGD trajectory. Do not introduce inverse momentum cycling in the first run; keeping momentum fixed isolates two simple changes and avoids a second schedule whose useful range is not established for this model.

## Exact Proposed Changes to `train.py`

1. Add `import math`.
2. Replace the schedule constants with:

   ```python
   MAX_LR = 0.1
   MIN_LR = 1e-4
   MOMENTUM = 0.9
   ```

   Keep `BATCH_SIZE=128`, `WEIGHT_DECAY=1e-4`, `MAX_STEPS=64000`, the model, data pipeline, augmentation, loss, seed, and evaluation cadence unchanged.
3. Construct SGD with `lr=MAX_LR`, `momentum=MOMENTUM`, `weight_decay=WEIGHT_DECAY`, and `nesterov=True`.
4. Remove `MultiStepLR` and its `scheduler.step()` call.
5. At the start of each timed batch, before `optimizer.zero_grad()`, calculate elapsed-budget progress and assign the cosine value to every optimizer parameter group:

   ```python
   progress = min(total_training_time / TIME_BUDGET_S, 1.0)
   lr = MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (
       1.0 + math.cos(math.pi * progress)
   )
   for group in optimizer.param_groups:
       group["lr"] = lr
   ```

6. Continue logging the current parameter-group learning rate. No additional synchronization, validation, dependency, model copy, or data-loader pass is required.
7. Do not add warmup in the first experiment. The unchanged baseline already trains stably from `lr=0.1`; warmup would reduce useful high-learning-rate work and add another tunable confound. If the loss shows early instability after enabling Nesterov, a follow-up can linearly ramp from `0.02` to `0.1` over the first 3% of counted time, then remap the cosine over the remaining 97%.

At the baseline's measured throughput, the approximate cosine learning rates would be:

| Counted time | Approx. step | Cosine LR | Baseline LR |
|---:|---:|---:|---:|
| 0 s (0%) | 0 | 0.1000 | 0.1000 |
| 75 s (25%) | 9,631 | 0.0854 | 0.1000 |
| 150 s (50%) | 19,263 | 0.0501 | 0.1000 |
| 225 s (75%) | 28,894 | 0.0147 | 0.1000 |
| 249 s (83%) | 32,000 | about 0.0070 | 0.0100 |
| 300 s (100%) | 38,525 | 0.0001 | 0.0100 |

This deliberately exchanges some mid-run high-LR exploration for a much longer annealing and refinement interval. That is the central experimental question.

## Optional EMA/SWA Extension and BatchNorm Handling

Weight averaging should be tested only after the schedule result is known. If added, it must not naively average every item in `state_dict`: BatchNorm running means and variances are data-dependent buffers, while `num_batches_tracked` is an integer counter and must never be arithmetically averaged.

The lowest-risk EMA extension is:

- Maintain a detached copy of model parameters only, initialized from the online model when elapsed progress reaches 50%.
- Update the shadow parameters after every optimizer step with a time-aware decay, e.g. `decay = exp(-dt / 5.0)` for a five-second averaging time constant. A time-based decay remains stable if step throughput changes.
- Copy BatchNorm buffers (`running_mean`, `running_var`, and `num_batches_tracked`) from the online model into the EMA model rather than averaging them. Evaluate only the EMA model once per epoch after EMA begins, so the once-per-epoch validation rule is preserved. This uses endpoint BN statistics, which are not exact statistics for the EMA weights, but avoids an unbudgeted calibration pass and is reasonable when the EMA time constant keeps shadow and online weights close.

For strict BN correctness, reserve the last approximately one training epoch of the 300-second budget for a no-gradient BN-statistics refresh after swapping in averaged parameters. Reset BN running statistics, run augmented training batches in `train()` mode under `torch.no_grad()`, measure this refresh wall time, and include it in `total_training_time`; do not perform it after the fixed budget as ostensibly free work. Perform the normal final evaluation only after refresh. This is methodologically cleaner for either EMA or late SWA, but it sacrifices optimizer updates and may increase end-to-end time through loader overhead. Given the baseline's 595.4-second total, it is not recommended until evaluation/runtime margin has been measured.

SWA is less attractive as the first averaging variant because a monotonically decaying cosine trajectory does not maintain a long constant/high learning-rate tail from which broad, equally weighted samples are naturally drawn. If tested, average parameters only over the final 15-20% of elapsed budget at a fixed interval and use the in-budget BN refresh above.

## Hypothesis and Expected Benefit

**Primary hypothesis:** aligning the entire decay to the measured 300-second horizon will improve `best_test_acc` from 91.67% to at least 91.77%, with a plausible result around 92.0-92.4% (+0.3 to +0.7 percentage points), because the model will receive roughly 8,000 rather than zero updates below `lr=0.01` and will finish near a local minimum instead of terminating mid-schedule.

**Secondary hypothesis:** Nesterov will provide a small complementary gain or faster progress during the first half of the cosine cycle without changing throughput materially. The combined change should still produce approximately 38,000 optimizer steps and 300 seconds of counted training.

If the first result improves, an EMA follow-up may add roughly 0.05-0.2 percentage points by reducing late-trajectory variance. That expected gain is smaller and less certain than the schedule correction, especially with BatchNorm and only a short horizon.

## Risks and Failure Modes

- **Premature annealing / underfitting:** cosine drops below the baseline's `0.1` immediately and reaches about `0.05` halfway through the run. If ResNet-20 needs a longer high-LR phase, accuracy may plateau early. A next test should use a 10-20% flat `0.1` hold followed by cosine over the remaining budget rather than returning to absolute step milestones.
- **Nesterov interaction:** Nesterov changes optimization dynamics at the same time as the scheduler. If the combined run fails, rerun the exact cosine schedule with standard momentum before rejecting time alignment.
- **Noisy time progress:** individual batch duration jitter slightly perturbs learning-rate spacing by step, but the schedule remains smooth in wall-clock time and has no extra synchronization. This is preferable under a wall-clock objective.
- **Final-step overshoot:** the last update begins just below the time limit and may complete after 300 seconds, as in the baseline. Its LR is already near the floor, so the tiny overshoot cannot prevent annealing completion.
- **EMA BatchNorm mismatch:** copying online buffers is approximate, while averaging buffers is invalid. A full recalibration is correct but costs time. Do not claim an EMA result without recording which policy was used.
- **Timeout:** adding EMA model evaluation, an extra BN pass, or extra validation could push the 595.4-second run beyond the 10-minute cap. The first experiment deliberately adds none of these.
- **Best-vs-final metric:** cosine is expected to favor the final epochs. Report both best and final accuracy; a large gap may indicate late overfitting or schedule decay that is too aggressive.

## Confound Controls

- Preserve seed 42; do not reroll or select seeds.
- Preserve architecture, initialization, batch size, data order behavior, transforms, normalization, loss, weight decay, maximum steps, and evaluation cadence.
- Make no label smoothing, Mixup, RandAugment, EMA, SWA, compilation, precision, or data-loader changes in the first experiment.
- Verify that step count and peak VRAM remain close to baseline. A substantial throughput shift would mean the comparison is not purely optimization-policy based.
- Use the same single H20 and required command/output protocol. Compare against 91.67% and require at least 91.77% to count as an improvement.
- Inspect the logged LR near 25%, 50%, 75%, and termination to verify that elapsed-time scheduling, not a step-count fallback, controlled the run.

## Fixed-Budget Feasibility

The first experiment is effectively cost-free: one cosine calculation and assignment per step plus Nesterov's modified SGD update add negligible compute and no meaningful VRAM. It introduces no dependency and modifies only `train.py`. It leaves the fixed 300-second counted training loop and once-per-epoch evaluation intact, and should retain roughly the baseline's 38,525 steps and end-to-end runtime. `MAX_STEPS=64000` remains a safety cap but should not bind.

The optional EMA costs one shadow parameter copy (about 272,474 parameters) and one linear interpolation per parameter per step, which is small but not literally free. A strict BN refresh must be charged to the 300-second budget and may threaten the total-runtime cap; therefore it is excluded from the first experiment.

## Suggested First Experiment Scope

Run exactly one controlled variant:

- elapsed-budget monotone cosine from `0.1` to `1e-4` over the full 300 seconds;
- SGD momentum `0.9` with `nesterov=True`;
- all other baseline behavior unchanged;
- no warmup, restart, momentum cycling, label smoothing, EMA, or SWA.

Accept the proposal if `best_test_acc >= 91.77%`, the summary is valid, counted training remains approximately 300 seconds, total runtime remains below 600 seconds, and throughput/VRAM are materially unchanged. If it improves, retain cosine and test EMA separately. If it does not, first disambiguate schedule from Nesterov with cosine-only; then test a 15% flat high-LR hold followed by elapsed-time cosine, rather than adding multiple regularizers at once.

## Evidence

- `TASK.md`: fixed-time protocol, only-`train.py` scope, single-GPU and 10-minute constraints.
- `train.py`: current SGD, fixed step milestones `[32000, 48000]`, per-step timing, and BatchNorm architecture.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/01-definition.md`: metric, constraints, and 0.1-point acceptance threshold.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/sgdr.md`: cosine annealing and strong anytime behavior on CIFAR.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/weight-averaging.md`: averaging works best with annealing and requires explicit BatchNorm-state handling.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/001/papers/label-smoothing.md`: plausible near-zero-cost regularizer intentionally held out of the first optimization experiment.
