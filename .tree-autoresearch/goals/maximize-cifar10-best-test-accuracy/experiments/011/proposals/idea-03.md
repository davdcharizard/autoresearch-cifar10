# Proposal: SAM-Pulse-Conditioned Weight EMA

## Summary

Add an evaluation-only exponential moving average (EMA) to EXP-004, updated only after successful optimizer steps that used the validated SAM perturbation. Preserve the live training trajectory exactly: the model still receives every independent batch, the full front-loaded CutMix dose, the same ordinary/SAM step sequence, the same Nesterov updates, and the same 300-second charged schedule. The shadow is never used for training.

Use one fixed policy:

- start: the first completed SAM optimizer step at pre-batch charged progress `>= 0.75`;
- cadence: every completed SAM step, and no ordinary step;
- averaging: EMA, not uniform averaging;
- half-life: 512 SAM pulses;
- decay: `EMA_BETA = 2 ** (-1 / 512) ~= 0.9986471129` per SAM pulse;
- horizon: from the first SAM pulse through budget exhaustion;
- evaluated source: live model before EMA initialization, then EMA only;
- one evaluator call per epoch, with exact live-state restoration afterward.

The shadow is initialized by exact copy from the live post-optimizer state on the first SAM pulse. It is therefore not zero-biased and needs no bias correction. Later pulse updates use `shadow = beta * shadow + (1-beta) * live`.

## Why EMA Rather Than Uniform SWA

SWA's strongest rationale assumes a constant or cyclic learning rate that maintains diverse late iterates. EXP-004 instead decays from roughly 0.034 at progress 0.75 to 0.002 at the end. A uniform average over all approximately 2,449 SAM samples would give the much higher-LR start of the clean tail the same weight as the final low-LR solution and could produce a stale center.

EMA retains trajectory smoothing while favoring the recent low-LR state. A 512-pulse half-life corresponds to about 1,024 optimizer steps because SAM is period two. EXP-004 produced about 32.7 SAM pulses per charged second and 97.5 per epoch, so the fixed half-life is approximately 15.7 charged seconds or 5.25 parent epochs. The first shadow state retains only about `2^(-2448/512) ~= 3.6%` of its original weight by the end, while several recent epochs remain represented.

The half-life is expressed in pulse count, not copied as a familiar EMA scalar. This follows the cadence/horizon principle in *How to Scale Your EMA* and makes the effective window auditable when wall-clock throughput changes.

Sources:

- `knowledge/papers/stochastic-weight-averaging.md`
- `knowledge/papers/how-to-scale-your-ema.md`
- `02-system-understanding.md`
- `03-experiment-learnings.md`
- `experiments/004/04-analysis.md`
- `experiments/006/04-analysis.md`
- `experiments/011/00-navigate.md`

## Why Condition on SAM Pulses

The proposed conditioning is defensible but not neutral. A post-SAM checkpoint is the restored live model immediately after the sole Nesterov update computed from the perturbed-loss gradient. Sampling at that point aligns the shadow with the mechanism that produced EXP-004's accepted +0.17-point gain and halves averaging overhead. Every sampled state also contains all earlier ordinary updates, including the ordinary step in the preceding two-step pair; the EMA does not discard those gradients from the cumulative trajectory.

However, fixed even-step sampling creates a stroboscopic subsequence. It can bias the average if ordinary and SAM updates form a systematic two-cycle, if even-step data differ by chance, or if post-SAM states are not more representative than the intervening post-ordinary states. The literature supports late trajectory averaging and cadence-aware EMA, not the claim that post-SAM points are intrinsically better averaging samples. A gain must therefore be attributed to the complete SAM-conditioned EMA policy, not to generic EMA or a proven flatness filter.

Preflight should compare this pulse EMA with a diagnostic all-tail EMA on a short fixed synthetic/real-data trajectory, using the same half-life in optimizer-step time (`beta_all = 2 ** (-1/1024)`). The all-tail shadow is diagnostic only, never evaluated in the metric run. Report pulse-vs-all shadow distance and update-direction cosine. A large difference confirms conditioning matters but does not select between them; the production policy remains pulse-conditioned regardless of the diagnostic.

## Exact Shadow State

Preallocate shadow tensors on GPU before charged training for:

- every trainable parameter;
- every floating BatchNorm `running_mean` and `running_var` buffer.

Do not average integer `num_batches_tracked`; keep the live value during evaluation. The model has no other meaningful floating buffers. Preserve state-dict order and assert one-to-one name/shape/dtype/device coverage.

At a successful scheduled SAM step:

1. Complete the existing first backward, SAM perturbation, separately autocast second forward/backward, BatchNorm-flag restoration, exact parameter restoration, and sole `optimizer.step()` unchanged.
2. If this is the first pulse, copy post-update parameters and floating BN buffers exactly into the shadow and set `ema_updates=1`.
3. Otherwise update all shadow floats with `beta=2^(-1/512)` and increment the counter.
4. Perform the existing CUDA synchronization after the EMA operation so its cost is included in charged `dt`.

The EMA never has gradients, never enters the optimizer, never changes model parameters or buffers during training, and never consumes RNG. Updating after `optimizer.step()` is essential: averaging perturbed temporary parameters, pre-update restored parameters, or first-pass weights would describe a different mechanism.

BatchNorm buffers are sampled from the same post-SAM states as the parameters and averaged with the same beta. This creates a self-contained shadow without an uncharged training-data recalibration pass. It is an approximation: BN running statistics already have their own momentum and may lag the EMA weights. Using live BN buffers would instead create an unreviewed parameter/statistic mismatch. No BN recalibration or second model evaluation is allowed.

## Single-Model Evaluation Swap

Before EMA activation, retain the parent behavior and evaluate the live model once per epoch. After activation, evaluate only the EMA state once per epoch; never evaluate both live and EMA in the same epoch.

Preallocate a separate exact restore buffer for all live floating parameters and BN running buffers. At each EMA evaluation:

1. Assert no SAM perturbation is active and synchronize CUDA.
2. Copy current live floats into the restore buffer.
3. Copy EMA parameter and floating BN state into the live model.
4. Call the unchanged `evaluator.evaluate(model, device)` exactly once inside `try`/`finally`.
5. In `finally`, restore every live float exactly before returning to training.
6. Leave integer BN counters live throughout; evaluation mode does not consume them.

The evaluator itself sets `model.eval()`. The next epoch's existing `model.train()` remains responsible for restoring training mode. `best_acc`, `test_acc`, and `test_loss` must be updated only from the single predetermined evaluated source. The final summary therefore reports the final EMA evaluation after activation, while early live evaluations remain eligible for `best_test_acc` under the fixed routing policy.

Any evaluator exception must restore live state before propagating. No averaged state may leak into optimizer training, and no live metric may be computed privately for model selection.

## Counters and Mechanism Audit

Add a final audit line containing:

- `ema_updates`, `ema_first_step`, `ema_first_progress`, `ema_last_step`;
- `ema_half_life_pulses=512`, exact beta, and expected oldest-state weight `beta ** (updates-1)`;
- `ema_eval_count`, `live_eval_count`, and total evaluator calls;
- parameter and floating-BN tensor/element coverage;
- `condition_mismatches`, `restore_failures`, and nonfinite counts;
- final and per-evaluation sampled `||ema-live||_2 / ||live||_2` for parameters;
- corresponding relative distance for floating BN buffers;
- final cosine between `(ema-live)` and the most recent post-SAM optimizer displacement, where defined.

Compute state distances immediately before the one scheduled evaluation, outside the charged training interval but without another data/model forward. Record min/mean/max and final values durably in `03-execute.md` before transient log deletion. These diagnostics prove the shadow is neither identical nor catastrophically far from live; they are not pass/fail accuracy substitutes.

Required arithmetic invariants:

- `ema_updates == sam_applied_batches`;
- every EMA update step satisfies the production SAM schedule and is even in one-based step numbering;
- first EMA step/progress equals first SAM step/progress;
- no update occurs before 0.75 or after a failed optimizer step;
- `ema_eval_count + live_eval_count == num_epochs` and each epoch appears once;
- live floating state is bitwise restored after every swap.

## Overhead and Memory

The 2,748,890-parameter model is about 11 MiB in FP32. The EMA shadow and evaluation restore buffer add about 22 MiB, plus negligible BN buffers, on top of the existing approximately 11 MiB SAM snapshot. Expected peak allocation is roughly 1,213-1,225 MiB, far below the H20 limit.

Each of roughly 2,449 pulse updates reads live/shadow floats and writes shadow once, about 33 MiB of parameter traffic per pulse or approximately 81 GiB over the final 75 seconds. H20 bandwidth makes arithmetic cost small, but foreach launch and synchronization placement can still affect step exposure. Use preallocated ordered lists and `torch._foreach_lerp_` if supported and numerically validated; otherwise use preallocated foreach multiply/add operations. Do not allocate model-sized tensors per pulse.

Expected charged overhead is below 1.5%, retaining approximately 25,200-25,560 steps versus EXP-004's 25,560. Evaluation still performs one model forward stream; state copies add only memory traffic outside charged training but inside the 600-second total limit.

## Parent-Relative Preflight

On physical GPU 0, compare unmodified EXP-004 and candidate in the same harness with alternating order and at least five trials:

1. Warm at least 50 steps, then time at least 500 production final-quarter two-step pairs (ordinary plus SAM) with synchronization.
2. Candidate includes the actual post-SAM EMA update; parent does not. Report paired median/p90/mean latency, dispersion, peak VRAM, and projected full-run steps from 25,560.
3. Time at least 20 full epoch-style evaluations for parent and candidate, including candidate swap/evaluate/restore, and verify identical evaluator call count.
4. Exercise enough pulses to initialize and update the shadow, then run the pulse-vs-all-step diagnostic without using accuracy to select a policy.

Proceed only if:

- paired median training latency ratio `<= 1.03` and p90 ratio `<= 1.05`;
- projected full-run optimizer steps `>= 24,800`;
- candidate evaluation-with-swap latency ratio `<= 1.05`;
- peak allocation `< 1.35 GiB` and projected total runtime `< 600s`;
- the same-harness parent passes these relative gates by construction and trial coefficient of variation is `<= 5%`;
- all coverage, finite, update-count, swap, restore, RNG, BN, and optimizer invariants pass.

If this fixed implementation fails, reject it before the metric run. Do not move EMA work outside the charged interval, reduce its cadence, or change the horizon to rescue throughput.

## Smokes

1. **Closed-form EMA:** On hand-specified tensors and three pulse states, verify exact recursive values, first-copy semantics, beta, oldest-state coefficient, and no gradient/optimizer ownership.
2. **Cadence:** Simulate progress/step pairs and require updates only after successful even SAM steps at/after 0.75, with exact counter equality.
3. **State coverage:** Enumerate names and require every trainable parameter and floating BN buffer exactly once, no integer buffer in foreach lists, and matching layout/device/dtype.
4. **BN policy:** Verify running mean/variance follow the same pulse EMA, counters remain live, evaluation uses EMA floats, and no recalibration/data pass occurs.
5. **Swap/restore:** Randomize live and shadow states, inject evaluator success and failure, and require the evaluated state equals shadow while post-finally live floats are bitwise identical to pre-swap values.
6. **Training parity:** With EMA update disabled, candidate must match EXP-004 step-for-step. With it enabled, live parameters, optimizer state, BN state, RNG state, losses, and outputs after each training pair must still match; only shadow/audit state may differ.
7. **SAM integration:** Require perturbation norm 0.05, RNG replay, one BN update, exact perturbation restore, one optimizer update, then exactly one EMA update from the post-optimizer state.
8. **Bias diagnosis:** Compare pulse-conditioned and time-matched all-step shadows on a short trajectory; report distance/cosine without evaluating accuracy or changing production policy.
9. **GPU integration:** Run BF16/channels-last production pairs and an EMA evaluation swap on GPU 0; require finite values, one evaluation, exact restore, and no unexpected synchronization or allocation churn.

## Risks

- **Conditioning bias:** Post-SAM states are a fixed parity subsequence, not proven superior checkpoints. A two-cycle or data-order correlation can make the EMA less representative than all-step averaging.
- **Cosine-tail diversity may be insufficient:** As LR approaches 0.002, sampled states can become nearly identical, leaving too little distance for averaging to help.
- **Stale early-tail contribution:** Even EMA retains some higher-LR initial state; the fixed 512-pulse horizon may over-smooth or under-smooth this short run.
- **BN approximation:** EMA of already momentum-averaged running statistics can lag parameter state. Recalibration is intentionally prohibited because it would add uncharged training-data work.
- **Primary objective mismatch:** EXP-004 final accuracy already equals its 95.40 best, so the observed tail variance may not imply headroom from smoothing.
- **Evaluation routing:** Once active, live checkpoints become invisible. EMA may smooth away a genuine live peak; evaluating both would violate the preregistered single-model policy and increase test-selection pressure.
- **Overhead:** Model-sized memory traffic every SAM pulse could reduce the validated update horizon despite no extra forward.
- **Noise floor:** Formal +0.10 gains are smaller than observed 0.14-0.29 variation. A threshold-only pass may remain weak evidence.

## Testable Hypothesis and Falsification

Run exactly once after passing preflight:

```bash
timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1
```

Require physical GPU 0 to be the 97,871 MiB H20, exit 0, 299.5-301.0 charged seconds, total time below 600 seconds, at least 24,800 steps, unchanged 2,748,890 trainable parameters, complete summary, one evaluation per epoch, unchanged CutMix and SAM phase/cadence, exact EMA/SAM counter equality, finite nonzero state distance after initialization, and zero restore/condition/nonfinite failures.

Formal success requires `best_test_acc >= 95.50%` versus EXP-004 at 95.40%. A meaningful mechanism-sized result requires `>=95.70%`. The hypothesis is that a 512-pulse post-SAM EMA will reduce late-checkpoint variance enough to reach 95.50-95.75 without altering training exposure; evidence for 95.70 or above would be substantially stronger.

The proposal is falsified by a score below 95.50, preflight failure, fewer than 24,800 steps, identical shadow/live state throughout the active window, a swap/restore or BN-coverage error, extra evaluation, or any scope/timing violation. Do not retry, evaluate live and EMA together, or tune beta/start/cadence from the result.
