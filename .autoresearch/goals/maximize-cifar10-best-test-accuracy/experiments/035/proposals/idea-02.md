# Proposal 02: Weak-Tail Full-State EMA

## Summary

Keep the accepted width-2 ResNet-20, optimizer, augmentation, and 80/20 LR schedule unchanged. At the existing 80% strong-to-weak boundary, make one detached FP32 shadow copy of the online model. After every subsequent weak-tail SGD step, update every floating-point item in the shadow model state with a fixed decay of `0.999` and copy integer BatchNorm counters. Use that shadow, and only that shadow, for every scheduled weak-tail evaluation. The online model continues SGD uninterrupted and is never evaluated beside the EMA model.

This is deliberately not a narrower retry of EXP-018. EXP-018 uniformly averaged eight epoch endpoints from 87.14% to 97.30%, stopped optimization at 98%, spent 5.99 counted seconds recalibrating BN, and produced 93.85% versus its 94.02% online best. Here the shadow is exponentially weighted per optimizer step, is fully mature by the end of the existing tail, tracks floating BatchNorm state at the same cadence, gives up no final SGD interval, and requires no BN-refresh phase. At the terminal look, states more than roughly seven seconds old have less than half the weight, and the first half of the weak tail contributes only about 6.7% in aggregate.

## Fixed EMA Definition

Set exactly:

```python
EMA_DECAY = 0.999
EMA_START_FRACTION = LR_HOLD_FRACTION  # 0.8
```

The shadow is initialized once from the online model immediately after the existing boundary evaluation and before the first weak-loader optimizer step. Initialization is a copy, not a zero-start average, so no bias correction is used. For weak-tail step `t`, after `optimizer.step()` and before the step's existing CUDA synchronization:

```python
ema_float_state = 0.999 * ema_float_state + 0.001 * online_float_state
ema_integer_state.copy_(online_integer_state)
```

The floating state is the ordered union of parameters and floating buffers, including every BatchNorm `running_mean` and `running_var`. The only integer state in this model is BatchNorm `num_batches_tracked`; it is copied rather than averaged. Shadow tensors are FP32 CUDA tensors, detached, `requires_grad=False`, absent from the optimizer, and updated under `torch.no_grad()`. Use one precomputed ordered list and `torch._foreach_lerp_` for the floating update; do not traverse or allocate a new `state_dict` on every step.

Decay `0.999` is fixed without a sweep. At the accepted approximately 90 weak updates/s, it has an effective window of about 1,000 updates (2.56 loader epochs, about 11 seconds) and a half-life of 693 updates (1.78 epochs, about 7.7 seconds). Across the roughly 5,300-step weak tail, the boundary copy retains only `0.999**5300 ~= 0.5%` terminal mass. One loader epoch retains `0.999**390 ~= 67.7%`, so the shadow smooths several nearby low-LR iterates but responds much faster than EXP-018's uniform 12-point-of-budget window. Starting at 80%, rather than selecting a later start, supplies enough burn-in for this pre-registered timescale and aligns the mechanism with the already accepted objective/data transition.

Do not change the decay, ramp it, use a warm-start correction, update only selected layers, average gradients, or derive the start from validation results. A failure retires this exact full-state EMA point rather than authorizing a decay/start/window sweep.

## BatchNorm Policy

Do not copy the final online BN buffers into the shadow and do not run an extra BN calibration pass. Copying current online buffers would pair a roughly ten-batch online-statistics timescale with parameters representing roughly 1,000 recent steps. Recalibration would consume optimizer exposure and repeat EXP-018's expensive terminal special phase. Instead, exponentially average the online floating BN buffers with the same `0.999` decay as the parameters. This gives the shadow one internally consistent historical weighting rule for both learned tensors and running statistics. `num_batches_tracked` is metadata rather than a real-valued moment, so it follows the current online counter exactly.

This is an approximation: the mean of running variances is not exactly the variance produced by the averaged network, and each online BN update precedes that step's parameter update. The tail LR is at most 0.01 and decreases, making adjacent functions much closer than the wide arithmetic window in EXP-018. Preflight therefore treats buffer alignment, positivity, and inference safety as load-bearing rather than assuming the approximation is harmless. No evaluation-time mutation of the shadow is allowed.

## Evaluator and Checkpoint Semantics

Retain the accepted `EVAL_CHECKPOINTS`, dense-tail condition, epoch boundaries, and `Eval.evaluate()` implementation verbatim. The selected model at each existing look is fixed in advance:

- Before and at the 80% boundary: evaluate the online model. The boundary copy is identical, so this does not hide an alternative candidate.
- After at least one weak-tail EMA update: evaluate only the EMA shadow.
- At training completion: evaluate only the terminal EMA shadow and report its loss/accuracy as `final_test_loss` and `final_test_acc`.

Never evaluate online and EMA models on the same epoch, add a diagnostic test pass, choose between their scores, or fall back to online weights. `best_test_acc` remains the maximum over the same single-model stream of exactly 19 pre-registered looks, with at most one evaluation per unique epoch. Log which fixed model supplied each look and require the production run to contain exactly 19 evaluations on 19 unique epochs. If timing would project a different count, the experiment does not launch; the schedule is not edited to manufacture parity.

The shadow is not installed into the online optimizer model and no optimizer checkpoint is rewound. An earlier EMA look may legitimately supply `best_test_acc`, just as an earlier accepted online look could; this is not checkpoint selection because every look and its evaluated model are fixed before the run. The summary must also record `ema_start_step`, `ema_updates`, terminal boundary-copy mass, EMA bookkeeping seconds, and whether the best look was an EMA look. A formal pass requires the best to come from the EMA portion; no claim is made from the uncompetitive pre-tail online looks.

## Counted Time, Exposure, and Resource Accounting

EMA initialization is performed at the boundary and explicitly charged: synchronize, copy the model, synchronize again, and add elapsed wall time to `total_training_time`. Each per-step EMA update is placed after `optimizer.step()` and before the existing `torch.cuda.synchronize()`, so its GPU work is already included in that step's `dt`; do not add the same duration twice. Optional bookkeeping may accumulate CUDA-event time for reporting, but those events must not introduce an extra per-step synchronization.

No averaging operation, copy, BN work, or finalization forward may be moved outside the 300-second counter. Evaluation remains excluded exactly as in `prepare.py`. Loader waits, loader rebuilding, and evaluation wall time retain accepted semantics. The online model performs SGD until the ordinary 300-second/`MAX_STEPS` stop; unlike EXP-018, no fraction of the tail is reserved for installation or BN refresh.

The extra model state is about 4.3 MB before allocator effects and should keep peak memory below 700 MiB. Launch requires a fresh-process conservative timing projection of at least 99% of EXP-010's 26,898 updates (`>=26,629`), exactly 19 evaluator looks, and less than 540 seconds total wall time. Production integrity uses the same `>=26,629` update floor; missing it invalidates the run rather than inviting an uncounted optimization.

## Mechanistic Hypothesis and Success Criteria

The accepted trajectory finishes at its best but remains a single noisy iterate after only about 60 seconds of weak low-LR refinement. The ICML 2025 weight-averaging review reports that averaging can complement annealing, and the NeurIPS 2023 EMA analysis treats decay as a real algorithmic timescale. A 0.999 per-step full-state EMA should suppress short-lived parameter noise while assigning exponentially less weight to the early rapidly adapting weak-tail states that biased EXP-018's uniform average backward.

**Testable hypothesis:** under seed 42 and the unchanged 300-second protocol, weak-tail full-state EMA with decay 0.999 will retain at least 99% of accepted optimizer exposure and achieve `best_test_acc >=94.25%` on one of the fixed EMA evaluations, exceeding the 94.15% frontier by the required 0.10 points. The final EMA model is expected to remain finite and reach at least 94.15%; final NLL below EXP-010's 0.1934 is supportive but not a substitute for the primary metric.

A mergeable result requires all protocol gates plus `best_test_acc >=94.25%`, with the best look identified as EMA. There is no requirement that final equal best because EMA is itself a continuously changing training-time state. A valid completed run below 94.25% is no-improvement and cannot be rerun, even if a different decay, online checkpoint, copied BN buffer, or narrower start looks attractive afterward.

## Preflight

### Structural and Arithmetic Gates

- Prove ordered name/shape/dtype/device equality between online and shadow state. Require exactly 1,073,962 trainable online parameters, no shadow tensor in the optimizer, no shadow gradients, and no aliasing.
- On known FP32 tensors, compare at least 1,000 consecutive production-helper updates with an explicit high-precision recurrence. Require FP32-rounding agreement, correct update count, copied integer state, and an analytically matching boundary-copy mass.
- Require initialization and each update to preserve CPU/CUDA RNG byte-for-byte and leave every online parameter, buffer, gradient, and optimizer tensor bitwise unchanged.
- Require finite shadow parameters/buffers, strictly positive BN running variances, aligned BN counters, and no per-update tensor allocation after setup.

### Immutable-Corpus Safety Gate

Use the already registered exact post-transform corpora rather than fresh forkserver augmentation: EXP-022's 200-batch strong hard/soft corpus (`e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946`) followed by EXP-028's 64-batch weak hard corpus (`ffefe980241d9719c8d7f2b44fe81c1b3f94e35003b0a645d3fea5999a745032`). Replay an online control and an EMA-enabled online candidate from identical state and backend settings.

The EMA path must leave the candidate online trajectory bitwise equal to control through all 264 updates: logits, losses, gradients, updates, parameters, buffers, optimizer state, and RNG. At registered weak checkpoints, run inference with the shadow on the same immutable inputs and require finite logits/loss, positive BN variances, the expected exact recurrence, and no candidate-only greater-than-95% one-class prediction concentration versus the matched online control. Record shadow/online logit RMS, parameter RMS distance, BN-buffer RMS distance, gradient/update ratios, and class shares; these are diagnostics except for nonfinite state, online divergence, recurrence failure, or candidate-only concentration.

This gate establishes non-interference and catches broken shadow state; it is not an accuracy proxy and its 64 weak steps do not claim terminal EMA maturity.

### Timing, Evaluation, and Lifecycle Gates

- Run one conditioning process and at least five fresh paired H20 timing processes using the exact production helper. Alternate control and candidate order; measure the full weak step and one charged shadow initialization.
- Conservatively project the 80% ordinary plus 20% EMA schedule. Require overall update retention `>=0.99`, projected steps `>=26,629`, total EMA setup/update work `<1.0s`, peak allocation `<700 MiB`, total wall `<540s`, and exactly 19 evaluations under an integrated epoch/progress simulation.
- Exercise the real strong-loader shutdown and weak-loader lifecycle with eight persistent workers. Require the unchanged one-time switch, hard one-dimensional weak targets, no shadow-held loader references, and clean shutdown.
- Verify the fixed evaluator routing with a counting stub: online through the boundary, EMA thereafter, exactly one call per scheduled epoch, 19 unique calls total, no online/EMA paired call, and terminal EMA only.

Any gate failure retires the proposal. Do not rescue it by moving work outside the timer, reducing update cadence, averaging parameters only, copying buffers, changing decay/start, recalibrating BN, or evaluating online as a fallback.

## One-Run Verification

Run once with `CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1` on one idle H20. Require only `train.py` tracked, seed 42, a complete finite ten-field summary, 300 counted seconds, total below 600 seconds, 1,073,962 online parameters, at least 26,629 optimizer steps, exactly one 80% augmentation switch with eight stopped workers, 45-55% realized plateau CutMix, hard weak targets, and exactly 19 evaluations on 19 unique epochs.

EMA provenance must show one boundary initialization, no pre-boundary updates, one update after every subsequent weak optimizer step, terminal boundary mass matching `0.999**ema_updates`, positive finite BN variance, counters aligned with online state, setup/update work charged once, and the final evaluator receiving the EMA shadow. Inspect the diff and logs to confirm no new RNG draw, no `Eval` modification, no additional test pass, no online-tail score, and no post-hoc model selection.

Report the moving-baseline delta, best/final EMA accuracy and NLL, best-model identity, strong boundary accuracy, number of EMA updates, effective age/boundary mass, shadow/online parameter and BN-buffer distances, counted EMA overhead, exposure, evaluation count, VRAM, startup, and total wall time. The only formal accuracy decision is whether the fixed EMA stream reaches at least 94.25%.

## Risks

- Even exponential weights may lag a monotonically improving cosine tail; 0.999 could be too slow for this short regime.
- Averaged BatchNorm variances are only an approximation to the moments of averaged parameters; full-state consistency can still be numerically insufficient.
- EMA may reduce useful terminal stochasticity without moving the solution into a better basin, leaving accuracy unchanged.
- Evaluating only EMA after the boundary deliberately gives up knowledge of the online tail. That is necessary for equal test-look parity but can make the net method worse.
- A small per-step foreach operation can still reduce the final partial epoch enough to violate 19-look parity; timing must reject that before production.
- A 0.10-point single-seed pass is formally valid but remains a narrow estimate and should not be generalized beyond this protocol.

## Distinction From EXP-018

EXP-018 tested a parameter-only, equal-weight average across widely separated endpoints, then stopped SGD and recalibrated BN. This proposal tests a continuously updated, exponentially recent, full-state model; it keeps the optimizer active for the entire budget and evaluates the predeclared shadow throughout the weak tail. The changed weighting kernel, update cadence, BN treatment, and exposure semantics are the mechanism. It is not a shifted SWA window or a post-result attempt to recover EXP-018.
