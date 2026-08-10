# Proposal 01: Late Arithmetic SWA with In-Budget BN Recalibration

## Exact Protocol

Preserve the complete accepted EXP-010 online trajectory through 86% of counted time. At every completed weak-tail epoch whose post-epoch progress is in `[0.86, 0.98)`, add the online trainable parameters to one equal-weight arithmetic average. At the first such endpoint, clone parameters as sample one; subsequently update with:

```python
torch._foreach_lerp_(swa_params, online_params, 1.0 / num_snapshots)
```

Snapshots are equally spaced by one 390-step epoch. Do not average per step, use EMA decay, change LR, or sample a different interval.

When optimizer work first reaches 98% of `TIME_BUDGET_S`, stop SGD immediately. Record `install_step`, install the averaged parameters into the online model, reset every BatchNorm `running_mean` to zero, `running_var` to one, and `num_batches_tracked` to zero, temporarily set BN momentum to `None`, then spend the remaining counted time running no-gradient forward passes from the existing weak crop/flip loader in `model.train()` mode. Explicitly recreate the iterator after each loader exhaustion. Ignore labels. Stop when `total_training_time >= 300s`, restore every original BN momentum, switch to normal evaluator mode, and perform the terminal evaluation once.

Evaluations before finalization remain online and follow the accepted cadence. The terminal SWA evaluation replaces, rather than supplements, the terminal online evaluation. No checkpoint search, online/SWA pair, prediction ensemble, extra test pass, reroll, or fallback window is allowed.

## Why This BN Policy

Arithmetic averaging is valid for parameters, not BatchNorm statistics. Averaging `running_mean`/`running_var` would combine moments produced by different online functions; copying the last online buffers would mismatch the averaged parameters; averaging integer counters is meaningless. This proposal solves the mismatch by cumulatively recomputing BN moments with the installed SWA parameters on the already-active weak training distribution, following the `torch.optim.swa_utils.update_bn` momentum-reset principle rather than retaining the default 0.1 EMA.

The final 2% reservation is exactly six counted accelerator seconds. Recalibration includes H2D transfer, forward, and synchronization inside the existing accounting style and must process at least 390 weak batches (one dataset pass), continuing across fresh iterators when time permits. Loader wait remains excluded exactly as in accepted training but remains inside total wall time. No augmentation or strong/CutMix batch is used for BN refresh, because the evaluated tail model is adapted to the weak distribution.

## Counted-Time Fairness

Every SWA operation is charged to `total_training_time`: first shadow allocation/copy, each arithmetic update, final parameter install, BN reset, and all recalibration transfers/forwards/synchronizations. Wrap each operation with wall timing and `torch.cuda.synchronize()` before accumulating its duration. Only `Eval.evaluate()` remains excluded, matching the harness.

The shadow contains detached same-device FP32 parameter tensors only. It never enters SGD and does not include gradients, momentum, or buffers. Online optimizer state is untouched until SGD stops. Parameter count remains 1,073,962; extra memory is informational and must remain below 700 MiB peak.

## Rationale and Hypothesis

ICML 2024 gives SWA generalization support under without-replacement sampling, matching the shuffled CIFAR loader. The local reason is narrower: EXP-017 improved switch/first-weak fit but worsened late NLL, while the accepted solution is already strong and the remaining limiter appears to be late generalization rather than more strong-phase fit. Equal averaging of nearby annealed endpoints may select a wider, better-calibrated solution without perturbing online SGD.

The evidence is not decisive. EXP-010 finished at its best and late checkpoints are highly correlated; arithmetic averaging can lag an improving path. The fixed hypothesis is: at least seven eligible snapshots plus correctly recalibrated BN retain at least 97% of accepted optimizer exposure and raise `best_test_acc` from 94.15% to at least 94.25%. Mechanism support additionally requires final SWA NLL below 0.1934, final SWA accuracy at least the recorded pre-install online best, and nontrivial finite endpoint parameter spread.

## Preflight Gates

Before a production run, require:

- exact parameter-only arithmetic: after known tensor snapshots, every shadow equals their explicit FP64-reference mean within FP32 rounding tolerance;
- shadow tensors are detached FP32 CUDA tensors, absent from optimizer state, finite, and shape/order aligned;
- snapshot updates do not mutate online parameters, buffers, gradients, momentum, CPU/CUDA RNG, or loader state;
- installing the shadow changes only parameters; BN reset is exact; one full weak pass produces finite non-default running statistics and positive counters;
- recalibration changes no parameter or optimizer tensor and consumes no backward/optimizer step;
- final evaluator uses the installed SWA parameters and refreshed buffers, with no second model evaluation.

Benchmark five fresh processes plus one integrated 86/12/2 schedule simulation. Require projected optimizer steps at least **26,200** under the joint conservative timing bound (with 26,091 remaining the production integrity floor), SWA bookkeeping below 0.5 seconds total, at least eight projected snapshots and seven actual snapshots, at least 390 calibration batches in six counted seconds, peak allocation below 700 MiB, and projected total wall time below 540 seconds. Verify exact eight-worker strong-loader shutdown and explicit iterator recreation across weak-loader exhaustion. Persist consecutive and first-to-last normalized parameter RMS distances; require median consecutive distance at least `1e-6` and first-to-last distance at least `1e-5` for mechanism validity.

A gate miss retires this protocol. Do not move averaging outside the timer, shorten calibration, copy online BN buffers, start earlier, switch to EMA, or add a CPU shadow.

## One-Run Verification

Run once on the sole idle H20 with seed 42 and output only to `run.log`. Only `train.py` may change; architecture, data, CutMix, optimizer, LR schedule, evaluator, and dependency files remain accepted.

Require exit zero, approximately 300 counted seconds, total below 600 seconds, a finite standard summary, 1,073,962 parameters, at least 26,091 optimizer steps, at least seven SWA snapshots, one finalization near 98%, at least 390 BN-refresh batches, and at most one evaluation on each unique epoch. The 80% switch must stop all eight strong workers, realized CutMix must remain near 50%, and no optimizer step may occur after SWA installation.

Accept as a mergeable improvement only if `best_test_acc >=94.25%`, final SWA accuracy is itself at least 94.25% and no lower than the recorded pre-install online best, `install_step == num_steps`, and both spread floors pass. Report final SWA NLL versus 0.1934, endpoint distances, snapshot count, averaging/calibration seconds, refresh batch count, best/final gap, exposure, VRAM, and wall time. A valid miss rejects this exact arithmetic window and BN policy; do not retry another start, reserve, sampling cadence, or averaging rule.

## Risks

- Equal late snapshots may average an improving trajectory backward and underperform the final online iterate.
- Six seconds of BN calibration sacrifices roughly 2% of optimizer time, plus bookkeeping overhead.
- Weak augmented recalibration gives empirical moments for the SWA model but finite-sample noise remains.
- Parameter interpolation can cross higher loss even when endpoints are good.
- Replacing the final online look can hide a superior endpoint, while evaluating both would unfairly add metric opportunities.
- A bare 94.25% pass is formal but low-confidence single-seed evidence and must not be overstated.
