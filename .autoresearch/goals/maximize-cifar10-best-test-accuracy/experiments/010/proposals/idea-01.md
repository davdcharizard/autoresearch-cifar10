# Idea: Late Checkpoint Parameter Averaging on the Accepted Width-2 Recipe

## Summary

Preserve the accepted EXP-007 width-2 model and complete N1/M7-through-80%-then-weak-tail recipe, and add only a uniform parameter average over late weak-tail evaluation endpoints. Begin averaging when elapsed training progress first reaches 90%. At each eligible epoch endpoint, update a running average of all trainable parameters, temporarily swap those averaged values into the existing model, and use the epoch's single allowed evaluator call on that averaged model. Restore the live online parameters immediately afterward so SGD continues on the original trajectory and optimizer momentum is never altered.

Do not average BatchNorm buffers. During averaged-model evaluation, retain the current online model's weak-tail `running_mean`, `running_var`, and integer `num_batches_tracked`. This is an explicit approximate BN policy: late averaged weights are paired with the most recent weak-distribution statistics. It avoids invalid arithmetic on running statistics and avoids an additional training-data recalibration pass that would cost time and change the experiment.

The proposal targets the nearly flat late EXP-007 trajectory: best accuracy was 93.55%, final accuracy was 93.49%, and the final three evaluations were 93.48%, 93.51%, and 93.49%. Averaging several nearby low-LR solutions may move toward the center of the local basin and reduce parameter noise without strengthening regularization during the successful strong-view fit phase.

## Diagnosis

The local evidence narrows the useful intervention class:

- EXP-007 established the accepted frontier at 93.55% with the width-2 model, 27,143 steps, 71 epochs, 300.0 counted seconds, 333.0 total seconds, 598.7 MB peak VRAM, and 1,073,962 parameters.
- Its 80% strong-view phase and weak hard-label cosine tail both worked. Accuracy plateaued rather than visibly climbing at termination, so simply extending the same terminal direction is not the clearest remaining lever.
- EXP-008 increased coupled decay to `5e-4`; it improved NLL but suppressed fitting and reduced best accuracy to 93.38%.
- EXP-009 removed decay from BN affine parameters and biases; it fit harder but worsened NLL and reached only 93.52%. Together, these failures support retaining all-parameter `1e-4` decay and moving away from norm-pressure tuning.
- The accepted tail already traverses a smooth elapsed-time cosine from `0.01` to `1e-4`. Weight-averaging evidence indicates that late averaging is most useful when combined with annealing, making the existing tail a natural trajectory to summarize rather than redesign.

The goal is not to ensemble predictions or increase evaluation opportunities. It is to replace each late evaluated parameter point with a single uniform weight-space average while preserving one evaluation per epoch and the same fixed test harness.

## Mechanism

Let `theta_j` be the online trainable parameters at the `j`-th eligible weak-tail endpoint. Maintain the uniform checkpoint average:

```text
theta_avg_1 = theta_1
theta_avg_j = theta_avg_(j-1) + (theta_j - theta_avg_(j-1)) / j
```

The online model continues SGD from `theta_j`; it is never replaced by the average for training. Uniform endpoint averaging can suppress idiosyncratic displacement along a noisy, low-learning-rate trajectory and favor a flatter central solution. Sampling once per epoch provides checkpoints separated by roughly 390 updates, avoiding the high correlation and per-step overhead of averaging every optimizer update.

This is checkpoint parameter averaging, not prediction ensembling, EMA, or classic SWA training:

- no extra model inference is performed;
- no restart or constant SWA learning-rate phase is introduced;
- no per-step shadow update is added;
- no online optimizer state is averaged;
- each included endpoint has equal weight;
- the accepted elapsed-time cosine schedule remains unchanged.

The literature basis is `.autoresearch/goals/maximize-cifar10-best-test-accuracy/knowledge/papers/weight-averaging.md`, which reports mild generalization gains from late averaging and a favorable interaction with annealing. `.autoresearch/goals/maximize-cifar10-best-test-accuracy/knowledge/papers/sgdr.md` supports the smooth cosine trajectory already present. Neither source guarantees a gain for this short, shallow CIFAR run; the proposal is a concrete test of that mechanism at the locally validated endpoint.

## Exact Averaging Window

Add:

```python
AVERAGE_START_FRACTION = 0.90
```

An endpoint is eligible only when all conditions hold:

```python
not randaugment_enabled and progress >= AVERAGE_START_FRACTION
```

This excludes every strong-view checkpoint and approximately the first half of the weak tail. On the EXP-007 trajectory, 90-100% spans about 2,714 optimizer steps, roughly seven full CIFAR epochs, and should yield about 7-8 averaged checkpoints including the terminal partial epoch. At 90% progress, the accepted schedule is near `lr=0.00505`; it then decays smoothly to `1e-4`.

The first eligible endpoint initializes the average and is evaluated unchanged because its average count is one. Every later eligible endpoint updates the average before evaluation. Include the final partial epoch endpoint if training terminates mid-epoch; it is already a required terminal evaluation point and its inclusion is predeclared rather than selected from accuracy.

Starting at 80% would average the first weak checkpoints, which are still rapidly adapting from strong augmentation and are materially below the accepted peak. Starting only at 95% would leave approximately 3-4 samples, limiting the variance-reduction mechanism. The 90% boundary is a fixed compromise between burn-in and sample count, not a value selected after observing EXP-010 accuracy.

## Exact Parameter Semantics

Average every trainable `nn.Parameter` in model iteration order:

- convolution weights;
- BatchNorm affine weights and biases;
- classifier weight and bias.

All accepted parameters are FP32. Keep the average in detached FP32 CUDA tensors with identical shapes and devices. Do not create new `nn.Parameter` objects and do not replace entries in the module or optimizer. The optimizer must retain references to the original online parameters and its original momentum buffers.

Suggested state:

```python
averaged_params = None
online_param_backup = None
average_count = 0
```

At the first eligible endpoint:

```python
with torch.no_grad():
    averaged_params = [param.detach().clone() for param in model.parameters()]
    online_param_backup = [torch.empty_like(param) for param in model.parameters()]
average_count = 1
```

At subsequent eligible endpoints:

```python
average_count += 1
weight = 1.0 / average_count
with torch.no_grad():
    for average, param in zip(averaged_params, model.parameters(), strict=True):
        average.lerp_(param, weight)
```

The `lerp_` update is algebraically the online uniform mean and avoids allocating a full temporary model. Assert the parameter-list lengths and shapes once when the state is initialized.

## BatchNorm Buffer Policy

BatchNorm state must be divided into parameters and buffers:

- `weight` and `bias` are learned parameters and are included in `averaged_params`.
- `running_mean` and `running_var` are data-dependent estimates, not ordinary learned weights; do not average them.
- `num_batches_tracked` is an integer counter; arithmetic averaging is semantically invalid.

During the temporary averaged-parameter evaluation, leave all BatchNorm buffers untouched at the current online endpoint. By this stage, the model has trained only on the weak crop/flip distribution for roughly half the tail or more, so the live buffers reflect the evaluation-relevant input regime. The averaged parameters also come only from nearby late weak-tail checkpoints, limiting their mismatch with endpoint buffers.

This policy is approximate: the live buffers are not exact moments produced by `theta_avg`. The approximation is preferable here to the alternatives:

- Averaging BN running buffers does not reconstruct activation moments for averaged weights and mishandles `num_batches_tracked`.
- Recomputing BN statistics requires resetting buffers and running weak training data through the averaged model. That extra pass is a model-dependent training operation, changes loader/RNG state, costs wall time, and complicates the accepted worker lifecycle.
- Maintaining a second averaged model and forwarding each training batch through it would approximately track buffers but nearly doubles forward work and destroys fixed-time exposure.

No BN recalibration pass is allowed in this experiment. Log the selected policy as `average_bn_policy=current_online_buffers` so analysis does not mistake it for exact SWA BN reconstruction.

## One-Evaluation-Per-Epoch Implementation

Retain the current `checkpoint_due`, `dense_tail_due`, and `training_done` predicates. They continue to produce at most one evaluator branch per epoch. Inside that single branch:

1. Determine whether this endpoint is averaging-eligible.
2. If eligible, update/initialize the parameter average.
3. Copy online parameters into the preallocated backup.
4. Copy averaged parameters into the existing model parameters.
5. Call `evaluator.evaluate(model, device)` exactly once.
6. Restore online parameters from the backup in a `finally` block.
7. Update `best_acc` from that single result and log `model=average`, `average_count`, and the BN policy.

If the endpoint is not eligible, call the evaluator once on the online model exactly as EXP-007 and log `model=online`. Never evaluate both online and averaged parameters at the same epoch. The resulting tail protocol is:

- strong and early checkpoints: online model, one pass when already scheduled;
- weak tail below 90%: online model, one pass per completed epoch;
- weak tail at or above 90%: averaged model, one pass per completed/terminal epoch.

Suggested swap structure:

```python
with torch.no_grad():
    for backup, param in zip(online_param_backup, model.parameters(), strict=True):
        backup.copy_(param)
    for param, average in zip(model.parameters(), averaged_params, strict=True):
        param.copy_(average)

try:
    test_loss, test_acc = evaluator.evaluate(model, device)
finally:
    with torch.no_grad():
        for param, backup in zip(
            model.parameters(), online_param_backup, strict=True
        ):
            param.copy_(backup)
```

The `finally` restoration is mandatory so an evaluator error cannot silently leave averaged parameters attached to the live optimizer trajectory. Parameter objects, gradients, optimizer parameter groups, and momentum buffers remain unchanged; only FP32 values are copied temporarily.

The final summary's `final_test_acc` and `final_test_loss` refer to the final averaged evaluation even though online parameters are restored afterward. This is intentional and must be recorded. No checkpoint file or second final evaluator call is needed.

## Fixed-Time Accounting and Overhead

The accepted timer counts synchronized batch H2D/forward/backward/update work and excludes evaluation. Treat the running-average update as candidate-specific training work and charge it to `total_training_time`:

```python
torch.cuda.synchronize()
t_average = time.time()
# initialize or update averaged_params
torch.cuda.synchronize()
average_update_dt = time.time() - t_average
total_training_time += average_update_dt
```

After charging the update, recompute `progress` and `training_done` before evaluation/logging. This prevents averaging from receiving free algorithmic compute outside the 300-second budget. The backup, temporary swap, evaluator, and restoration are validation preparation/execution and remain excluded from counted training, matching the existing evaluator policy; all still count against the 600-second total wall limit.

Memory overhead is bounded:

- averaged parameters: `1,073,962 * 4` bytes, about 4.10 MiB;
- online backup: another about 4.10 MiB;
- persistent total: about 8.20 MiB plus Python tensor metadata.

Against EXP-007's 598.7 MB peak, expected peak allocation is approximately 607-615 MB, far below H20 capacity. No extra optimizer or gradient state is added.

Average/update and copy overhead occurs only about 7-8 times. It should be negligible relative to 300 seconds, but EXP-003 showed that apparently small operations can matter. Before a full run, benchmark the exact width-2 parameter-list operations on the idle H20:

- 20 warmups and 100 timed repetitions for average update only;
- 20 warmups and 100 timed repetitions for backup + swap + restore;
- synchronize around each timed region;
- report mean, p95, total projected endpoint overhead, and peak allocation.

Feasibility gates:

- projected charged average-update overhead across eight endpoints `<=0.25s`;
- projected validation-preparation overhead across eight endpoints `<=1.0s`;
- peak allocation `<700 MB`;
- no parameter-identity change, shape mismatch, non-finite value, or optimizer-state mutation;
- a one-step equivalence smoke test must show that after swap/restore the online parameter tensors are bitwise equal to their pre-swap backups.

If any gate fails, do not launch the full run. Do not move averaging to CPU as an unreviewed workaround; host transfers would add a different timing mechanism.

Expected full-run exposure remains approximately EXP-007's 27.1k steps. Charged averaging may remove at most a few updates; pre-register a material-overhead warning below 26,500 steps (more than about 2.3% below EXP-007), while recognizing ordinary node timing as in EXP-008/009.

## Preserved Accepted Recipe

Outside the averaging state and single evaluator-branch logic, keep `train.py` unchanged:

- width multiplier 2, post-activation ResNet-20, raw Option-A shortcuts, and exactly 1,073,962 trainable model parameters;
- current Kaiming initialization and seed 42;
- N1/M7 RandAugment through exactly 80% of counted training;
- existing strong-loader break, evaluation, deterministic shutdown of eight persistent forkserver workers, `gc.collect()`, and weak crop/flip loader reconstruction;
- hard-label `F.cross_entropy`, batch size 128;
- ordinary SGD momentum 0.9 and all-parameter coupled weight decay `1e-4`;
- `lr=0.1` through 80%, then the step to `0.01` and elapsed-time cosine to `1e-4`;
- early checkpoints `(0.2, 0.4, 0.6, 0.7)`, dense weak-tail evaluation, terminal evaluation, and fixed `Eval.evaluate()`;
- maximum-step guard, timing boundaries except explicit charged averaging, log redirection, and ten-field final summary.

Do not add EMA, per-step averaging, SWA LR changes, additional validation, BN recalibration, prediction ensembling, altered decay, mixed precision, or checkpoint selection based on observed accuracy.

## Attribution

The online SGD path remains the accepted EXP-007 path apart from the tiny, explicitly charged elapsed-time cost:

- averaging uses no random operation, so it does not intentionally change the data stream;
- online parameter values are restored bitwise after each evaluation;
- optimizer momentum and parameter identity are untouched;
- data, architecture, loss, regularization, and LR formulas are fixed;
- only the evaluated late parameter point changes from online endpoint to uniform endpoint average.

The primary comparison is necessarily cross-run against EXP-007's 93.55%. Evaluating both models in EXP-010 would violate the one-pass-per-epoch constraint and create extra opportunities for `best_test_acc`. Therefore EXP-010 cannot directly decompose its result into online-versus-average accuracy within the same run. Log average count and parameter distance diagnostics only if they do not require synchronization or extra data passes; do not add online evaluations.

Because `best_test_acc` takes the maximum over scheduled epochs, the intervention also changes which parameter sequence is observed after 90%. This is the intended model-selection mechanism, not a hidden confound. The evaluation count and epoch schedule remain the same or differ only if charged averaging removes a final partial update/epoch.

## Hypothesis and Expected Impact

**Primary hypothesis:** uniform averaging of width-2 weak-tail endpoints from 90% onward will raise `best_test_acc` from 93.55% to at least 93.65%, with a plausible range of 93.65-93.85% (+0.10 to +0.30 points), by reducing late-trajectory parameter noise and moving toward a flatter solution without weakening strong-view fitting.

**Secondary predictions:**

- final averaged accuracy should be at least as stable as EXP-007's final three online values and may narrow the best/final gap;
- final fixed-evaluator NLL may fall below EXP-007's 0.2196 if averaging improves basin centrality, but NLL alone cannot satisfy the top-1 goal;
- 7-8 averaged endpoints should be collected, all from the weak phase at progress >=90%;
- parameter count stays 1,073,962 and peak VRAM remains below 700 MB;
- optimizer steps remain near 27.1k and total runtime remains near EXP-007's 333 seconds, comfortably below 600 seconds.

The expected gain is modest. The final EXP-007 checkpoints are correlated and already nearly flat, so averaging may simply reproduce their common solution. A 0.10-point improvement is the minimum useful outcome and is not assumed from lower NLL alone.

## Failure Modes

- **Correlated checkpoints:** epoch endpoints may be too close in weight space for averaging to change generalization materially.
- **Nonlinear path / early-point bias:** even after 90%, uniform weight interpolation can cross a worse region or overweight earlier, higher-LR points relative to the final solution.
- **BatchNorm mismatch:** current endpoint buffers are only approximate statistics for averaged affine/conv parameters; this can erase a small averaging gain or worsen NLL.
- **Too few samples:** a faster or shorter run may provide fewer than the expected 7 endpoints. Require at least 5 averaged evaluations for a mechanism-valid result; otherwise record insufficient averaging exposure even if the run itself is protocol-valid.
- **Online restoration bug:** failure to restore exact values would silently train from the average and invalidate attribution. Bitwise smoke tests and `finally` restoration are mandatory.
- **Optimizer-state mismatch:** replacing Parameter objects or loading a separately constructed model can detach momentum state. Only in-place `copy_` on existing parameters is allowed.
- **Unbudgeted compute:** excluding average updates from `total_training_time` would give candidate-specific work for free. Charge update time explicitly.
- **Extra evaluation:** evaluating online and average models in the same epoch violates the requested design and inflates selection opportunities.
- **Endpoint count changes:** charged overhead or node timing may remove a final epoch, slightly reducing evaluation opportunities versus EXP-007.
- **Metric noise:** 0.10 points is ten CIFAR-10 examples. Run one fixed seed with no retries or adaptive start-window changes.

## Implementation Sketch

1. Add `AVERAGE_START_FRACTION = 0.90` and averaging state counters.
2. Add small helper functions that initialize/update averages, swap averaged values into existing parameters, and restore online values. Helpers operate only under `torch.no_grad()` and assert aligned parameter lists.
3. In the existing single evaluator branch, initialize/update the average only after the weak loader is active and progress is at least 90%.
4. Charge only initialization/update time to `total_training_time`; recompute progress afterward.
5. Swap parameters, run exactly one evaluation, and restore in `finally`.
6. Preserve online evaluation before averaging begins.
7. Extend the existing eval log with `model`, `average_count`, and `average_bn_policy`; do not change final summary keys.
8. Add static assertions for average/backup tensor count, shape, dtype, device, and detached status.

## Verification

### Preflight

- Confirm moving baseline 93.55%, making 93.65% the acceptance threshold.
- Confirm exactly one idle H20 with approximately 98 GB VRAM and no compute process.
- Confirm no stale log and only the reviewed `train.py` diff.
- Run Python compilation, Ruff, pre-commit, model-shape/parameter checks, and averaging helper unit smoke tests.
- Verify accepted optimizer has one group with all parameters at `weight_decay=1e-4`.
- Verify swap/restore leaves each parameter bitwise equal and leaves every optimizer momentum tensor unchanged.
- Benchmark average/copy overhead and require the gates above.
- Statistically inspect code paths to prove `evaluator.evaluate` is reachable only once per loop epoch.

### Eventual Execution

Run exactly once under the required supervisor and redirection:

```bash
CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1
```

### Post-run

- Require `best_test_acc >= 93.65%` for improvement; otherwise record no-improvement without rerun.
- If accuracy passes, require exit 0, ten unique finite summary keys, approximately 300 counted seconds, and total time below 600 seconds.
- Require `num_params == 1073962`, one augmentation switch at 80%, eight stopped workers, and weak training resumed.
- Require unique evaluator epochs and no more than one evaluator call in any epoch.
- Require the first averaged evaluation at progress >=90%, no averaged evaluation while RandAugment is active, monotonic `average_count`, and at least 5 averaged checkpoints.
- Require logs to identify online versus averaged evaluation and `current_online_buffers` BN policy.
- Compare steps, epochs, evaluation count, runtime, VRAM, final NLL, best/final gap, and trajectory with EXP-007.
- Confirm charged average overhead is recorded and within the preflight projection.
- Do not inspect or select an unreported online tail metric; none should be computed.
- Preserve seed 42, do not retry a valid run, and remove `run.log` after analysis.

## Decision Rule

- **Improvement:** accept only at `best_test_acc >=93.65%` with all integrity, timing, evaluation-count, and averaging-state checks passing.
- **No improvement with valid averaging:** revert averaging and retain EXP-007; correlated checkpoints or BN mismatch outweighs the mechanism at this operating point.
- **No improvement with fewer than five averaged checkpoints:** classify the metric normally but record insufficient averaging exposure; redesigning the window requires a new experiment.
- **Lower NLL without top-1 gain:** formal verdict remains no-improvement, consistent with EXP-008's warning that NLL and top-1 can diverge.
- **Material step loss below 26,500:** treat overhead/timing as a confound and revert; do not make averaging free after seeing the result.
- **Invalid restoration, duplicate evaluation, BN recalibration, worker leak, crash, or timeout:** invalidate the run, fix only the protocol fault, and rerun the same predeclared setting.

## Evidence

- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/01-definition.md`: metric, only-`train.py` scope, one-H20 fixed-time protocol, evaluation cap, and no-seed-hacking rule.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/007/04-analysis.md`: accepted 93.55% width-2 frontier, 27,143 steps, 15 weak-tail evaluations, flat terminal trajectory, runtime, VRAM, and parameter count.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/008/04-analysis.md`: stronger decay lowered NLL but underfit and lost top-1.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/009/04-analysis.md`: selective decay fit harder but worsened NLL and did not beat top-1; averaging is the recommended orthogonal follow-up.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/010/01-brainstorm.md`: late-trajectory diagnosis and checkpoint-averaging seed.
- `train.py`: accepted width-2 architecture, all-parameter `1e-4` decay, N1/M7 lifecycle, elapsed cosine tail, one evaluator branch, timer, and summary semantics.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/knowledge/papers/weight-averaging.md`: late averaging, mild generalization gains, annealing complementarity, and BN-state warning.
- `.autoresearch/goals/maximize-cifar10-best-test-accuracy/knowledge/papers/sgdr.md`: cosine-annealed trajectory basis and CIFAR relevance.
