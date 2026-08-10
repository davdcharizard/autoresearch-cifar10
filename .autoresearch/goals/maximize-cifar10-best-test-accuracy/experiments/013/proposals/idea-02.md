# Proposal: Uniform Late Weak-Tail Endpoint Averaging

## Decision

Preserve the complete accepted EXP-010 recipe and add one isolated model-selection mechanism: a uniform arithmetic average of online model parameters sampled at weak-tail epoch endpoints from 90% of counted training onward. Continue training the online model and its original SGD momentum state; use the average only for the one already-scheduled evaluation at each eligible endpoint.

This experiment does not add an evaluation, alter the optimizer trajectory, recalibrate BatchNorm, ensemble predictions, introduce a new learning-rate phase, or adapt the averaging window after observing results. The fixed policy is:

```python
AVERAGE_START_FRACTION = 0.90
```

An endpoint is eligible exactly when `not randaugment_enabled` and the progress computed from charged training time is at least `0.90`. The first eligible endpoint initializes the average and therefore evaluates the unchanged online parameter values. Each subsequent eligible endpoint updates the uniform average before evaluation. The terminal partial epoch is included when eligible because it is already an evaluation endpoint.

The primary hypothesis is that centering the final annealed trajectory will raise `best_test_acc` from 94.15% to at least the 94.25% acceptance threshold. The point prediction is **94.30%**, with an expected modest range of 94.25-94.40% if the mechanism works. A lower valid result is no-improvement, with no rerun, start-window adjustment, or switch to EMA.

## Local Rationale

EXP-010 is the accepted frontier: width-2 postactivation ResNet-20, all-parameter coupled decay `1e-4`, alpha-1 CutMix on 49.77% of N1/M7 plateau batches, then a hard-label weak crop/flip tail. It reached 94.15%, finished at its best, completed 26,898 steps in 300.0 counted seconds, and retained a healthy 89.73% strong-phase checkpoint. EXP-012 likewise finished at its best after a strong weak-tail recovery, although its full-preactivation model missed the gate. These runs indicate that the late tail is productive and should not be shortened or regularized more strongly.

The accepted schedule already supplies the condition under which the weight-averaging literature is most plausible: a low-learning-rate annealed trajectory. From 80% to 100%, LR falls by cosine from `0.01` to `1e-4`; at 90% it is approximately `0.00505`. Starting at 90% excludes the rapid initial adaptation after RandAugment and CutMix are removed while retaining roughly 2,690 updates and about seven epoch endpoints on EXP-010's 69-epoch trajectory. Starting at 80% would average nonstationary domain-transition weights; starting at 95% would leave only three or four highly correlated samples. The selected 90% boundary is fixed before the run and is not tunable within EXP-013.

This is endpoint Polyak-style averaging, not classic SWA training. There is no constant or restarted SWA LR, no per-step shadow update, no optimizer-state averaging, and no second inference pass. It attacks terminal iterate noise without adding to the already-sensitive strong-view underfit.

## Exact Average

Let `theta_j` be every trainable model parameter at the `j`-th eligible endpoint. Maintain detached FP32 CUDA tensors with:

```text
theta_avg_1 = theta_1
theta_avg_j = theta_avg_(j-1) + (theta_j - theta_avg_(j-1)) / j
```

In implementation, clone every item from `model.parameters()` at the first endpoint. At later endpoints increment `average_count`, then apply `average.lerp_(param, 1.0 / average_count)` under `torch.no_grad()`. Include all convolution weights, all BatchNorm affine weights and biases, and classifier weight and bias. Preserve model iteration order and use `zip(..., strict=True)` with one-time shape, dtype, device, and count assertions.

Do not construct a second `nn.Module`, create replacement `nn.Parameter` objects, or load a state dict. Original parameter identities, gradients, optimizer parameter-group references, and SGD momentum buffers must remain unchanged. The live online values are copied to preallocated detached backups before an averaged evaluation, averaged values are copied in place into the same parameters, and online values are restored in a `finally` block.

The online model always supplies the next training step. Thus averaging cannot feed back into optimization except through its explicitly charged, tiny elapsed-time cost.

## BatchNorm State Policy

Average BatchNorm `weight` and `bias` because they are trainable parameters. Do **not** average or reset any BatchNorm buffers:

- `running_mean` and `running_var` remain the current online endpoint's weak-tail estimates;
- integer `num_batches_tracked` remains the current online value;
- evaluator mode prevents those buffers from changing during the averaged evaluation.

The evaluated object is therefore `averaged parameters + current online BN buffers`. Log this explicitly as `average_bn_policy=current_online_buffers`.

This is an acknowledged approximation. Arithmetic averages of running moments do not produce the activation moments of an averaged nonlinear network, while averaging `num_batches_tracked` is meaningless. A BatchNorm recalibration pass is forbidden: it would consume extra training data, alter loader and RNG state, add model-dependent wall time, and turn this into a second mechanism. Forwarding a shadow model on every batch is also forbidden because it would materially reduce fixed-budget exposure.

The policy is locally defensible because all averaged endpoints occur well after the 80% switch and use nearby online statistics accumulated solely on the weak crop/flip distribution. A mismatch remains a predeclared failure mechanism, not permission for a post-hoc buffer policy change.

## Evaluation Semantics

Retain the existing `checkpoint_due`, `dense_tail_due`, and `training_done` branch so `evaluator.evaluate()` remains reachable at most once per epoch. The evaluated model is fixed by endpoint progress:

- before 90%, evaluate the online model exactly as today;
- from 90% onward, if the weak loader is active, update the average and evaluate the averaged parameters exactly once using current online BN buffers;
- never evaluate both online and averaged parameters in the same epoch;
- never compute or log an unreported online tail accuracy for comparison.

For an eligible endpoint:

1. Initialize or update the average from the online parameters and charge that work to training time.
2. Copy the online parameter values into the preallocated backup.
3. Copy the averaged values into the existing parameters.
4. Invoke `evaluator.evaluate(model, device)` once.
5. Restore every online parameter in `finally`, even if evaluation raises.
6. Update `best_acc` from that single result and log `model=average`, `average_count`, and the BN policy.

The final summary's `final_test_acc` and `final_test_loss` describe the final averaged evaluation when the terminal endpoint is eligible. Online parameters are restored afterward, but no second final evaluation or checkpoint selection is allowed. This deliberately replaces late online model-selection opportunities; it does not inflate them.

## Timing, RNG, and Memory Semantics

Average initialization/update is candidate-specific training computation and must count against the 300-second budget. Synchronize immediately before and after that operation, measure elapsed wall time, add it to `total_training_time`, and then recompute `progress` and `training_done` before evaluation. Eligibility is decided from the pre-update endpoint progress; if the charged update crosses 100%, that endpoint is still evaluated once and training then terminates.

Backup, temporary swap, evaluator execution, and restoration are validation preparation/execution and remain outside `total_training_time`, matching the existing evaluator treatment. They still count toward the 600-second total runtime. Log per-endpoint charged update time and cumulative charged averaging time.

The averaging, copy, and `lerp_` operations are deterministic and consume no random numbers. They must leave CPU RNG, the active CUDA RNG state, loader RNG progression, and CutMix gating unchanged. Do not instantiate a shadow model because constructor initialization would consume RNG. Preflight must capture CPU and CUDA RNG states immediately before helper operations and prove exact equality afterward.

Memory remains negligible relative to the H20:

- averaged parameters: `1,073,962 * 4` bytes, about 4.10 MiB;
- online backup: another approximately 4.10 MiB;
- no duplicate gradients, optimizer state, activations, or BatchNorm buffers.

Expected persistent overhead is about 8.20 MiB plus tensor metadata, taking the accepted 598.7 MB peak only to roughly 607-615 MB.

## Preserved Recipe

Apart from averaging state, helper operations, charged timing, and evaluation provenance logs, keep `train.py` behavior unchanged:

- width-2 postactivation ResNet-20 with parameter-free Option-A shortcuts and 1,073,962 trainable parameters;
- seed 42 and current Kaiming initialization;
- batch 128, strong N1/M7 worker transforms, alpha-1 CutMix probability 0.5 only in the strong phase, and ordinary hard targets in the weak phase;
- the exact 80% strong-to-weak switch, deterministic shutdown of eight persistent workers, and weak-loader reconstruction;
- SGD, momentum 0.9, coupled all-parameter decay `1e-4`, `lr=0.1` through 80%, then `0.01` cosine annealing to `1e-4`;
- current timer boundaries and per-step synchronization except for the additional explicitly charged average update;
- fixed `Eval.evaluate()`, early checkpoint tuple, dense weak-tail cadence, terminal evaluation, maximum-step guard, and final summary schema.

Do not combine averaging with compilation, larger batches, zero-gamma, architecture changes, EMA, per-step averaging, BN recalibration, prediction ensembling, or any LR modification.

## Preflight Correctness Gates

All gates are mandatory before a full GPU run:

1. Static scope: only the reviewed `train.py` diff; `prepare.py`, dependency files, data, architecture, optimizer, transform, schedule, and evaluator call count are unchanged.
2. Model identity: exactly 1,073,962 parameters; helper setup does not change parameter object IDs, optimizer group membership, or momentum-buffer object IDs/values.
3. Arithmetic: two and three synthetic known parameter snapshots yield the expected FP32 uniform means within exact operation-order tolerance; `average_count` begins at one and increases once per eligible endpoint.
4. Restoration: after initialize/update, backup, swap, and restore, every online parameter is bitwise equal to its pre-swap value. Shapes, dtypes, devices, and finite status all match.
5. Buffers: all BN buffers remain bitwise equal through averaging and swap/restore; `num_batches_tracked` is never copied into average state. An eval-mode smoke call must not mutate them.
6. RNG: CPU and CUDA RNG states are bitwise identical before and after helper operations; no new module initialization or loader iteration occurs.
7. Training continuity: one normal optimizer step after swap/restore matches a control continuation from the same model and optimizer state, including momentum and BN updates.
8. Evaluation structure: the source has one evaluator call in the existing per-epoch branch, and a synthetic progress trace proves no epoch can produce both online and averaged evaluation.
9. Lifecycle: average state cannot initialize while `randaugment_enabled`; the existing strong loader still shuts down all eight workers once near 80% and weak targets remain one-dimensional.

A failure blocks execution until the implementation defect is corrected. It does not authorize changing the 90% window, averaging rule, BN policy, or accepted training recipe.

## H20 Timing Gates

On one idle 97,871 MiB H20, benchmark the exact final helper implementation using the accepted model tensors:

- 20 warmups and at least 100 synchronized repetitions of average update alone;
- 20 warmups and at least 100 synchronized repetitions of backup + averaged swap + online restore;
- report median, p95, projected totals for eight endpoints, and peak allocated memory;
- verify finite tensors and bitwise restoration within the timed harness.

Require projected charged average initialization/update overhead for eight endpoints `<=0.25s`, projected excluded validation-preparation overhead `<=1.0s`, and peak allocated memory `<700 MB`. Require a conservative full-run total projection below 540 seconds. Any timing-gate failure is a no-go; do not move state to CPU, reduce tensor coverage, update less often, or change the start fraction as an unreviewed fallback.

The full run should retain at least 97% of EXP-010's exposure: `num_steps >= 26,091`. This is a mechanism-integrity floor, not an alternate accuracy criterion. Because projected averaging cost is tiny, a lower count indicates timing or implementation contamination that must be reported.

## Full-Run Verification and Decision Rule

Run once at seed 42 under the required one-H20 supervisor with all output redirected to `run.log`. Do not retry a valid no-improvement.

Require:

- exit code zero, all ten finite summary fields, 300.0 counted training seconds, and total runtime below 600 seconds;
- `best_test_acc >= 94.25%` for improvement over the 94.15% moving baseline;
- exactly 1,073,962 trainable parameters, peak allocation below 700 MB, and at least 26,091 steps;
- one augmentation switch near 80%, eight workers stopped, no soft target after the switch, and the expected approximately 50% realized strong-phase CutMix rate;
- unique evaluation epochs and no more than one evaluator invocation per epoch;
- first averaged evaluation at an eligible weak endpoint with pre-update progress at least 90%, monotonic unit increments in `average_count`, and at least five averaged endpoints for a mechanism-valid test;
- cumulative charged averaging time consistent with preflight and included in `training_seconds`;
- online parameter restoration and the declared `current_online_buffers` policy in logs.

Compare the 80% checkpoint, first weak checkpoint, late averaged trajectory, final NLL, best/final gap, steps, epochs, runtime, and VRAM with EXP-010. These are mechanism diagnostics only. The formal verdict is determined by the primary accuracy threshold after integrity gates pass.

- **At least 94.25%:** improvement; accept only if all protocol and averaging-integrity gates pass.
- **Below 94.25% with at least five valid averaged endpoints:** no-improvement; correlated weights, trajectory lag, or BN mismatch outweighed the mechanism. Revert without reroll.
- **Fewer than five averaged endpoints:** score the metric normally but record inadequate averaging exposure; do not widen the window in the same experiment.
- **Lower NLL without top-1 gain:** no-improvement; prior decay experiments show NLL cannot replace the stated metric.
- **Crash, timeout, duplicate evaluation, uncharged update, restoration/RNG/buffer failure, or worker leak:** invalid implementation. Fix only the protocol fault and rerun the unchanged predeclared policy.

## Failure Mechanisms

- **Trajectory lag:** a uniform mean from 90% centers earlier, higher-LR parameters and may underperform EXP-010's still-improving final online iterate.
- **Checkpoint correlation:** epoch endpoints separated by roughly 390 updates may be too similar for averaging to move into a meaningfully wider solution.
- **Nonlinear interpolation:** even individually good endpoints need not share a low-loss linear basin; their arithmetic mean can lose class-margin accuracy.
- **BatchNorm mismatch:** current online moments are not exact moments for averaged convolution and affine weights, and the resulting error can exceed a small averaging gain.
- **Lost online peak opportunities:** replacing late online evaluations can hide an online model that would have set a new best; evaluating both is nevertheless forbidden.
- **Insufficient samples:** runtime variation may yield fewer than the expected seven endpoint samples, weakening the test.
- **Implementation feedback:** incomplete restoration or changed parameter identity would silently train from averaged values or detach optimizer state, invalidating attribution.
- **Accounting error:** performing average updates for free would slightly increase candidate compute; explicit synchronization and charging are required.
- **Single-seed resolution:** the required 0.10 point is ten CIFAR-10 examples. A bare threshold pass is protocol-valid but weak causal evidence and must not trigger confirmation rerolls.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`: primary metric, 94.25% moving gate, one-H20 fixed-time protocol, one evaluation per epoch, only-`train.py` scope, and fixed-seed rule.
- `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`: backward-dominated step cost, large memory headroom, accepted exposure, and late averaging as an open question.
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md` and `04-results.tsv`: accepted long-plateau/weak-tail/CutMix recipe and failed stronger regularization or preactivation changes.
- `goals/maximize-cifar10-best-test-accuracy/experiments/010/04-analysis.md`: accepted 94.15%, 26,898-step, final-equals-best trajectory and exact recipe provenance.
- `goals/maximize-cifar10-best-test-accuracy/experiments/012/04-analysis.md`: compute-neutral architecture near miss, strong-phase underfit, and productive terminal weak-tail recovery.
- `goals/maximize-cifar10-best-test-accuracy/knowledge/papers/weight-averaging.md`: mild checkpoint-averaging gains, complementarity with annealing, and explicit BatchNorm-state caveat.
- `train.py`: current model, timer, phase switch, loader lifecycle, dense-tail evaluation branch, and parameter/buffer structure.
