# Proposal: Late Weak-Tail Online Parameter EMA

## Decision

Preserve the complete accepted EXP-010 model and training recipe, and maintain a detached FP32 exponential moving average of every trainable parameter only during the final half of the weak cosine tail. Use that EMA parameter set, paired with the current online BatchNorm buffers, for the single already-scheduled evaluation at each eligible epoch endpoint. Restore online parameters immediately afterward so SGD and momentum always continue on the original trajectory.

Pin the policy before execution:

```python
EMA_START_FRACTION = 0.90
EMA_HALF_LIFE_EPOCHS = 1.0
EMA_DECAY = 0.5 ** (1.0 / len(train_loader))
```

Both accepted strong and weak loaders have 390 batches (`50_000 // 128` with `drop_last=True`), so `EMA_DECAY` is approximately `0.998224`. Its half-life is exactly 390 EMA updates, and its asymptotic mean parameter age is approximately 562 steps, or 1.44 epochs. The decay is derived from one local epoch rather than selected from an accuracy sweep.

Start EMA after the optimizer step of the first weak batch whose **pre-step counted progress** is at least 90%. Initialize the shadow by cloning the resulting online parameters; do not initialize it to zero and do not apply bias correction. Every later eligible optimizer step updates the shadow. Before the shadow exists, evaluate the online model. Once it exists, evaluate only EMA parameters. Never evaluate online and EMA weights in the same epoch.

The hypothesis is that smoothing the coherent, low-LR terminal trajectory will raise `best_test_acc` from the 94.15% frontier to at least 94.25%, with a point prediction of **94.30%**, while retaining at least 99% of EXP-010's optimizer exposure.

## Why This Window and Decay

EXP-010's accepted path spends 80% of counted time at `lr=0.1` with N1/M7 and p=0.5 CutMix, then removes both strong augmentation and mixed targets while annealing from `0.01` to `1e-4`. Its switch checkpoint was 89.73%, first weak checkpoint was 93.16%, and accuracy continued rising to a final-equals-best 94.15%. EXP-012 also recovered strongly and finished at its best. These trajectories make the initial weak transition productive but nonstationary; averaging from 80% would mix rapid domain-adaptation weights into the estimator.

Starting at 90% excludes approximately the first seven weak epochs and retains about 2,690 late updates, or 6.9 EMA half-lives. By the terminal point, the initial shadow contributes only about 0.8%, while recent epochs dominate. A one-epoch half-life smooths within- and across-epoch SGD displacement but tracks the still-annealing solution more closely than a long uniform average. Starting at 95% would leave only about three to four epochs, while a conventional fixed `0.999` decay would encode an arbitrary roughly 1.8-epoch half-life. Neither alternative may be substituted after timing or accuracy is observed.

This is online parameter EMA, not classic SWA training: there is no constant SWA LR, restart, extra optimizer step, prediction ensemble, checkpoint search, or averaging across seeds. It retains the learning-rate annealing that both EXP-015 paper summaries identify as complementary to averaging and stays within one initialization/trajectory, consistent with the BatchNorm intrinsic-LR evidence.

## Exact Parameter Update

Maintain two detached lists on the model's existing CUDA device:

```python
ema_params = [param.detach().clone() for param in model.parameters()]
online_backup = [torch.empty_like(param) for param in model.parameters()]
ema_updates = 1
```

On every later eligible step, after `optimizer.step()` and before the existing terminal `torch.cuda.synchronize()`:

```python
torch._foreach_lerp_(
    ema_params,
    list(model.parameters()),
    1.0 - EMA_DECAY,
)
ema_updates += 1
```

This implements `ema = decay * ema + (1-decay) * online` for all Conv weights, BatchNorm affine weights/biases, and classifier weight/bias. All tensors are FP32, same-device, detached, and shape-aligned. Use one cached tuple of online parameters rather than recreating the list each step. `torch._foreach_lerp_` is the exact required multi-tensor path; its availability and numerical behavior are preflight gates. Do not silently fall back to one Python kernel launch per tensor or move shadows to CPU.

The shadow tensors are not `nn.Parameter` objects, have `requires_grad=False`, and never enter the optimizer. Do not average gradients, momentum buffers, LR, decay state, or online parameter values. The original parameter objects and optimizer references remain unchanged for the entire run.

## BatchNorm Buffer Policy

EMA only `named_parameters()`. Do **not** maintain EMA copies of floating or integer buffers:

- BatchNorm `weight` and `bias` are trainable parameters and are included;
- `running_mean` and `running_var` stay at the current online weak-tail endpoint;
- `num_batches_tracked` remains the current online integer count;
- evaluator mode prevents buffer mutation while EMA parameters are installed.

The evaluated state is explicitly `EMA parameters + current online BN buffers`, logged as `ema_bn_policy=current_online_buffers`.

There are three reasons not to EMA floating buffers. First, BN running moments already use an activation EMA (momentum 0.1); applying the parameter decay again would double-smooth and substantially lag the final weak distribution. Second, arithmetic averages of moments collected under changing online weights are not the activation moments of the averaged weights. Third, the nearby EMA parameters have a roughly 1.44-epoch mean age, while current buffers rapidly track the same weak crop/flip distribution and are likely a better approximation than doubly lagged moments.

Do not average `running_mean`/`running_var`, copy integer counters into shadow state, or run BN recalibration. Recalibration would consume an extra data pass, alter loader/RNG progression, and add a distinct algorithmic mechanism. The online-buffer mismatch remains a declared causal risk rather than permission for a second buffer policy.

## Evaluation and Restoration

Retain the existing `checkpoint_due or dense_tail_due or training_done` branch, which reaches `evaluator.evaluate()` at most once per epoch. At an endpoint:

- if `ema_params is None`, evaluate online parameters exactly as EXP-010;
- otherwise, copy online parameters to the preallocated backup, copy EMA values in place into the same parameter objects, evaluate once, and restore online values in `finally`;
- update `best_acc` from that one result only;
- log `model=online` or `model=ema`, `ema_updates`, `ema_decay`, and the BN policy.

Use in-place `copy_` under `torch.no_grad()` with `zip(..., strict=True)`. Do not construct a second module, replace Parameters, load a state dict, or swap optimizer state. Restoration in `finally` is mandatory even if evaluation raises. The final summary's `final_test_acc/loss` describe the final EMA evaluation when the shadow exists, although online values are restored afterward. No second terminal online evaluation or hidden online accuracy diagnostic is allowed.

EXP-010's 80% switch evaluation remains online because EMA cannot exist during the strong phase. If 90% is crossed by a step whose pre-step progress was still below 90%, that epoch's evaluation also remains online; EMA begins on the next qualifying step. This deterministic one-step boundary rule avoids uncharged endpoint initialization and cannot be adjusted after inspecting the trajectory.

## Fixed-Time Accounting

All candidate-specific tensor work counts against the 300-second training budget:

- the first shadow clones and backup allocations occur after the first eligible optimizer step, inside the existing `t0` through synchronized `dt` region;
- every `_foreach_lerp_` update executes before the existing synchronization and is therefore included in that step's `dt`;
- backup, EMA copy-in, and online restoration around evaluation are separately synchronized, wall-timed, and added to `total_training_time`; only `Eval.evaluate()` itself remains excluded, matching the harness.

After charged swap or restore work, recompute `progress` and `training_done` before continuing. Log cumulative `ema_update_seconds` and `ema_swap_seconds`. No initialization, update, copy, or synchronization may occur for free outside the counted budget.

The shadow consumes approximately `1,073,962 * 4 = 4.10 MiB`; the online backup consumes another 4.10 MiB. There are no duplicate gradients, optimizer states, modules, or buffers. Expected persistent overhead is about 8.20 MiB plus tensor metadata, taking EXP-010's 598.7 MiB peak to roughly 607-615 MiB.

## RNG and Causal Alignment

No architecture or constructor changes occur, so all model parameters/buffers and post-construction CPU/CUDA RNG states begin bitwise identical to EXP-010. `clone`, `_foreach_lerp_`, and `copy_` consume no random numbers. Shadow initialization happens only after the stochastic stream has already reached the predeclared 90% boundary, and must leave CPU, CUDA, loader, worker, shuffle, transform, and CutMix RNG progression unchanged.

The online SGD path is identical except for the tiny reduction in completed steps caused by charged EMA work. Evaluation swaps cannot affect it because values are restored bitwise and optimizer momentum is untouched. The intervention changes which late trajectory is observed by the fixed evaluator, not training targets, feature geometry, regularization, or optimization state.

## Preserved EXP-010 Recipe

Keep unchanged:

- width-2 postactivation ResNet-20, Option-A shortcuts, global average pool, 128-to-10 classifier, and 1,073,962 trainable parameters;
- seed 42 and current initialization;
- batch 128, N1/M7, p=0.5 alpha-1 CutMix only through 80%, and hard weak crop/flip tail;
- ordinary SGD momentum 0.9 with coupled all-parameter decay `1e-4`;
- `lr=0.1` through 80%, then `0.01` cosine annealing to `1e-4`;
- persistent workers, strong-loader shutdown, weak-loader creation, target assertions, timer, evaluator, checkpoint cadence, maximum-step guard, and summary.

Do not combine EMA with EXP-014 pooling, BN recalibration, a different decay/start, endpoint uniform averaging, mixed precision, compilation, batch changes, LR changes, or extra evaluation.

## Correctness Gates

Before H20 timing, require all of the following in disposable tests:

1. Static scope: only reviewed EMA state/update/swap/timing/log code changes in `train.py`; architecture, optimizer, data, schedule, evaluator branch, and seed remain unchanged.
2. State identity: exactly 1,073,962 trainable parameters; original parameter object IDs, optimizer membership, momentum tensor IDs/values, and all BN buffers remain unchanged by shadow creation and swap/restore.
3. Arithmetic: on known FP32 tensor sequences, `_foreach_lerp_` matches the scalar recurrence within operation-order tolerance. Starting from `A` and repeatedly observing constant `B`, 390 updates leave the `A` displacement at exactly one half within FP32 tolerance.
4. Shadow integrity: every EMA tensor is detached, FP32, same-device, finite, shape-aligned, absent from optimizer state, and initialized bitwise to the first eligible online snapshot.
5. Restoration: after backup/copy/evaluation-mode smoke/restore, every online parameter is bitwise equal to its pre-swap value; the next optimizer step matches a control continuation from identical model/optimizer state.
6. Buffers: EMA state contains no buffers. Running means, variances, and counters remain bitwise unchanged across swap/evaluation/restore and continue updating only during online `model.train()` steps.
7. RNG: CPU and CUDA RNG states are bitwise equal before/after shadow initialization, update, swap, and restore; no module construction or loader iteration is hidden in helpers.
8. Phase semantics: a synthetic progress trace proves no shadow before a qualifying weak step at pre-progress >=90%, monotonic `ema_updates`, and online-only evaluation before initialization.
9. Evaluator semantics: exactly one static evaluator call remains in the per-epoch branch; a trace proves no epoch can evaluate both states.
10. Targets/lifecycle: normal training succeeds with hard and probability targets; the 80% worker switch and eight-worker shutdown are unaffected.

Any failure blocks the full run. Fix only implementation defects; do not change decay, start, buffers, cadence, or use a foreach fallback.

## H20 Timing Gates

On one idle 97,871 MiB H20, benchmark the exact cached-parameter `_foreach_lerp_` implementation against accepted control in five alternating fresh-process pairs. Use batch 128, identical hard/probability target alternation, 100 warmups, and at least 500 synchronized complete active-EMA steps per trial. Separately benchmark initialization and seven backup/swap/restore cycles.

Require:

- active-EMA candidate/control median-of-trial-means step ratio `<=1.03`;
- trial-mean CV `<=2%` for each and candidate p95 no more than 1.08x control;
- projected whole-run exposure, accounting for EMA on only the last 10%, `>=26,629` steps (99% of EXP-010);
- projected late-window update retention `>=97%`;
- cumulative charged shadow initialization plus all projected swaps/restores `<=0.50s`;
- peak allocation `<620 MiB` and no more than 16 MiB over paired control;
- conservative total runtime projection below 540 seconds.

Also run an integrated synthetic schedule with 90% ordinary and 10% EMA-active steps to validate the projection and timer accounting. A gate miss retires this exact policy; do not move shadows to CPU, update less frequently, use per-tensor loops, shorten the window, or make work uncounted.

## Full-Run Verification and Decision Rule

After every preflight gate passes, run once at seed 42 with all output redirected to `run.log`. No valid run may be retried.

Require:

- exit zero, ten unique finite summary fields, 300.0 counted seconds, total below 600 seconds, 1,073,962 trainable parameters, and peak VRAM below the measured gate;
- at least 26,629 optimizer steps, one augmentation/CutMix switch near 80%, eight workers stopped, approximately 50% realized strong mixing, and no soft weak target;
- unique evaluation epochs and no more than one evaluator call per epoch;
- no EMA before a qualifying >=90% weak step, monotonic per-step `ema_updates`, at least five EMA evaluations, fixed approximately 0.998224 decay, and `current_online_buffers` provenance;
- cumulative update/swap timing present and charged; online restoration and momentum integrity preserved.

Compare the 89.73% strong switch, 93.16% first weak checkpoint, late trajectory, final 0.1934 NLL, best/final gap, steps, epochs, and runtime with EXP-010. These are mechanism diagnostics only.

- **Improvement:** `best_test_acc >=94.25%` and every protocol/integrity gate passes.
- **Valid lower accuracy:** no-improvement; restore EXP-010 with no reroll, decay change, window change, or buffer rescue.
- **Fewer than five EMA evaluations:** score the metric normally but record insufficient averaged-model observation; do not start earlier in the same experiment.
- **Accuracy pass below exposure floor:** formally clears the metric but has timing-confounded attribution and cannot be described as compute-negligible.
- **Crash, timeout, duplicate evaluation, uncharged work, RNG/buffer/restoration/lifecycle fault:** invalid. Fix only the protocol fault and rerun the unchanged declared policy.

## Causal Risks and Failure Mechanisms

- **Trajectory lag:** EXP-010 was still improving and finished at its best; an EMA with mean age 1.44 epochs can trail the superior final online iterate.
- **Correlated samples:** consecutive per-step weights may be so similar that EMA changes little beyond introducing lag.
- **Nonlinear interpolation:** an EMA of parameters can leave the low-loss path even when each online endpoint is good.
- **BatchNorm mismatch:** current buffers are not exact activation moments for EMA parameters; this approximation can erase a 0.10-point gain.
- **Double-smoothing alternative rejected:** EMA floating buffers would lag weak-distribution adaptation and still would not produce correct moments for EMA weights.
- **Lost online selection opportunities:** after initialization, online tail accuracies are deliberately unobserved; an online peak may be forfeited, but evaluating both would violate the protocol's model-selection budget.
- **Counted overhead:** even fused foreach updates can remove terminal low-LR steps, which may be disproportionately valuable.
- **Private foreach dependency:** installed `torch._foreach_lerp_` behavior must pass preflight; no silent slower fallback is allowed.
- **Window arbitrariness:** one epoch is interpretable but not proven optimal; the protocol tests this point once rather than tuning it.
- **Single-seed resolution:** the 0.10-point threshold is ten CIFAR-10 examples. A bare pass is protocol-valid but weak causal evidence and cannot be confirmed by reroll.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`: only-`train.py`, one-H20, fixed-time, seed, evaluation, and primary-metric rules.
- `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`: accepted exposure, large shadow-memory headroom, evaluation cost, and late averaging as an open question.
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md` and `04-results.tsv`: validated high-LR exploration/weak tail and failures from stronger regularization or altered representation.
- `goals/maximize-cifar10-best-test-accuracy/experiments/014/04-analysis.md`: catastrophic max-readout scale failure and explicit recommendation to test endpoint averaging next.
- `goals/maximize-cifar10-best-test-accuracy/experiments/015/papers/when-where-why-average-weights.md`: trajectory averaging complements LR annealing and can mildly improve generalization at low memory cost.
- `goals/maximize-cifar10-best-test-accuracy/experiments/015/papers/intrinsic-learning-rate.md`: preserve high-LR BN-SGD exploration and average one coherent late trajectory.
- `train.py`: accepted online trajectory, 390-step epochs, BatchNorm buffer structure, weak-tail cadence, timer, and evaluator branch.
