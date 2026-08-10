# Proposal: Short-Timescale Late Parameter EMA

## Decision and falsifiable hypothesis

Maintain a detached FP32 exponential moving average of **parameters only** during the final part of ordinary weak-tail SGD. Start on the first step whose pre-step counted progress is at least `0.90`; initialize the shadow from that step's post-SGD parameters, then after each later optimizer update apply

```text
ema <- 0.999 * ema + 0.001 * online
```

Stop SGD at the first completed step reaching `0.99` counted progress, install the EMA once, recompute all BatchNorm running statistics on hard weak-training batches inside the remaining counted budget, and use the existing terminal evaluation for the installed EMA. Preserve every accepted model, optimizer, LR, augmentation, loader, seed, evaluator, and summary setting.

**Hypothesis:** EXP018 failed because a uniform 86-98% average gave early, inferior annealed states equal influence. Decay `0.999` has a 693-update half-life (about 7.6 seconds at the measured 10.9 ms step), so after the expected roughly 2,470 updates from 90-99%, the initial state retains only about 8.5% weight and half the final EMA mass comes from the last roughly 2.5% of the budget. This should smooth terminal SGD noise without EXP018's long backward bias. The falsifiable prediction is seed-42 `best_test_acc >=94.25%`, terminal EMA accuracy at least the pre-install online best, and terminal NLL below 0.1934. One valid sub-threshold run rejects this exact start/decay/finalization policy; no decay/window tuning or rerun is allowed.

## Evidence and distinction from EXP018

- Ajroldi, Orvieto, and Geiping, *When, Where and Why to Average Weights?* (ICML 2025), finds modest generalization gains from checkpoint averaging and supports combining averaging with LR annealing. This is directional evidence, not a local effect-size guarantee.
- The EXP038 TWA review argues that historical-state coefficients matter and that fixed uniform/exponential weights can be suboptimal. Full TWA is infeasible here because checkpoint storage, subspace construction, and coefficient optimization would replace useful training time; a one-shadow EMA is the smallest test of the coefficient-shape hypothesis.
- EXP018 is decisive against its **uniform** 86-98% endpoint average: eight nondegenerate snapshots plus 1,624 BN-refresh batches lowered 94.02% online to 93.85% and worsened NLL. It explicitly left EMA unexplored. The proposed EMA samples every optimizer step, strongly favors recent states, begins later, and has an effective timescale of about 2.5% rather than a uniform 12% window.
- EXP018 also establishes feasibility primitives: a parameter shadow costs only a few MiB, exact installation is cheap, and cumulative BN refresh processed 780 batches in 3.001 seconds and 1,624 in 5.991 seconds. The remaining uncertainty is per-step EMA overhead and whether the recent average improves the installed function.

Expected effect is modest, approximately `+0.10` to `+0.25` points if late checkpoint noise is limiting. The main contrary evidence is that EXP010's dense tail had little best-final gap and EXP018's averaged model degraded calibration; the true effect may therefore be null or negative.

## Exact algorithm and charged costs

1. Leave online training bitwise unchanged through 90% progress. The shadow does not exist before then.
2. For a step with pre-step `total_training_time / TIME_BUDGET_S >= 0.90`, perform ordinary forward/backward/SGD. If this is the first EMA step, clone the ordered detached FP32 parameters on device; otherwise update the shadow with one production `torch._foreach_lerp_(ema_tensors, online_parameters, 0.001)`. The existing `t0` begins before H2D and the existing synchronization occurs after the EMA transaction, so allocation, reads, writes, and launches are included once in that step's `dt`. The EMA never reads or writes gradients, optimizer state, RNG, or buffers.
3. After accounting for a completed step, if counted progress is at least `0.99`, break before any online evaluation. Record `install_step`, the existing `best_acc`, and online parameter/BN hashes. No more backward or optimizer steps occur.
4. Score the online model on the non-test diagnostic corpus, install EMA parameters in stable `named_parameters()` order, score the EMA with the still-identical online BN buffers, then reset every BatchNorm `running_mean` to zero, `running_var` to one, and `num_batches_tracked` to zero. Temporarily set BN momentum to `None` and refresh with the active hard-target weak loader in `model.train()` under `torch.no_grad()`. Iterator recreation is explicit after exhaustion. Each synchronized H2D+forward transaction, the install/reset, and diagnostic forwards are timed and added exactly once to `total_training_time`; refresh stops when the original 300-second budget is exhausted. Restore original BN momenta, score the post-refresh diagnostic, and enter `eval()` before test evaluation.
5. The existing dense-tail evaluation for this final epoch becomes the sole terminal EMA test look. There is no same-epoch online test evaluation and no evaluation after it. Earlier scheduled online results remain in `best_acc`; the terminal EMA result competes with them under the unchanged max metric. Evaluation remains excluded from the 300-second counter exactly as in the harness.

BN affine weight and bias are parameters and are averaged. Floating or integer BN buffers are **never averaged or copied from historical states**: all 19 modules are reset and cumulatively recomputed for the installed EMA. Require at least 390 refresh batches, equal `num_batches_tracked` in every BN, finite non-default running means/variances, exact parameter preservation during refresh, restored momenta, and hard one-dimensional targets. This avoids the parameter/buffer mismatch that would result from pairing EMA parameters with online or exponentially averaged BN statistics.

Print one bounded `ema_finalization` provenance line containing start/install progress and steps, decay, update count, normalized online-to-EMA RMS, pre-install online best, refresh batches, aligned BN counter, shadow/update/install/diagnostic/refresh seconds, and final EMA accuracy/loss. Keep the existing ten summary fields unchanged and require `install_step == num_steps`.

## Causal within-run comparisons without extra test looks

Before training, materialize a fixed deterministic corpus of the first 512 CIFAR-10 **training** examples using `ToTensor` plus the accepted normalization and record its tensor/label hash. It is diagnostic only and never enters the optimizer. At the 99% boundary:

1. in `eval()` mode, score the online parameters with the current online BN buffers;
2. install EMA parameters while retaining the identical online BN buffers and score the same tensors again, isolating the parameter-average effect;
3. after BN reset/refresh, score the same tensors a third time, isolating the buffer-recalibration effect.

All three GPU forwards are charged. Report hard CE, top-1, logit RMS, prediction agreement, and relative logit L2, plus normalized parameter RMS. These are training-corpus mechanism diagnostics, never eligibility metrics. Online parameters need not be restored after comparison because the EMA installation is the intended terminal state. The terminal EMA versus the already observed `pre_ema_best` supplies a same-run test comparison without increasing the accepted evaluation count; only the fixed diagnostic corpus supports the strict same-state causal attribution.

## Preflight and safety gates

- **Arithmetic/state gate:** production helpers must match an explicit FP64 EMA recurrence over at least 2,000 synthetic updates within FP32 tolerance; prove ordered install, nonaliasing, exact online parameter/buffer/optimizer/gradient/RNG preservation before install, and one-time charge accounting. Require the expected 693-update half-life and exact `0.999` recurrence rather than a bias-corrected or time-varying decay.
- **EMA geometry gate:** replay at least 2,000 cheap known parameter updates with distances calibrated to the registered accepted trajectory. Require finite shadow state, nonzero normalized online/EMA RMS in `[1e-5,0.05]`, and at least 70% of final normalized EMA mass arising after progress 0.94. The latter is checked analytically from actual update counts and recurrence coefficients; the expected value is about 75%, and failure means the run is too short for the declared timescale.
- **Install/BN gate:** on the registered real weak corpus, compare online and installed-EMA logits under identical online BN buffers, then refresh at least 390 batches. Require no greater-than-95% candidate-only class concentration, pre-refresh EMA/online logit RMS ratio in `[0.5,2.0]`, CE ratio in `[0.8,1.2]`, finite state, 19 aligned BN counters, parameter immutability during refresh, and restored momentum. Controls must qualify all ratio gates using denominator-safe absolute-plus-relative rules.
- **Lifecycle/protocol gate:** prove no online test look occurs at finalization, terminal evaluation happens once after momentum restoration in `eval()` mode, the best metric retains earlier online evaluations, refresh uses only hard weak targets, and no optimizer step occurs after `install_step`.

Because the shadow is observational until installation, class-trajectory replay is unnecessary for the pre-90% online model; exact state non-mutation is stronger. Installation geometry and BN semantics are the candidate-specific safety questions.

## Timing and fixed-budget feasibility

Use one conditioning process and seven counterbalanced fresh-process pairs on one idle H20. Measure exact production steps with and without the active foreach EMA transaction after warmup, plus shadow initialization, three diagnostic forwards, install/reset, and at least 390 real weak refresh batches. Persist raw trials before assertions. Require:

- active-step candidate/control median ratio `<=1.08`, aggregate CV below 3%, no pair above `1.12`, and no host synchronization outside the already timed boundary;
- a conservative whole-run projection of at least 98% of accepted optimizer exposure (at least 26,360 steps relative to EXP010's 26,898), after charging the 1% refresh reserve and EMA overhead;
- at least 390 refresh batches before 300 seconds, projected terminal evaluation count `<=19`, peak allocation `<650 MiB`, and total wall `<540 seconds`.

The 98% exposure floor is deliberate: reserving the final 1% already removes roughly 269 ordinary updates, while EMA is active for only 9% of the budget. EXP018 measured ample refresh throughput and about 611 MiB peak allocation, so the idea is technically feasible if foreach EMA adds no more than modest active-step overhead. A timing failure rejects this implementation; it does not authorize CPU shadows, sparse update intervals, a different start/decay, skipped BN refresh, or uncharged work.

## Production integrity, risks, and verdict

Before the sole scored run, require the 94.15 frontier at `7c1e7d8`, only tracked `train.py` modified, no stale log, compile/Ruff/format success, exactly one idle H20, fixed seed42, 1,073,962 parameters, accepted loaders/transforms/CutMix/LR/SGD/evaluator, and no more than 19 test looks. Run once under the 600-second supervisor. A valid result requires 300 seconds counted, total below 600, one 80% loader switch with eight workers stopped, 45-55% CutMix, at least the timing-qualified exposure floor, exact EMA/BN provenance, and a complete finite summary.

Key risks are (1) no meaningful checkpoint noise to smooth, (2) lag even at this short decay during rapid terminal refinement, (3) EMA overhead displacing useful SGD updates, and (4) recalibrated BN buffers hurting calibration as in EXP018. Improvement requires `best_test_acc >=94.25%`, while the stronger mechanism confirmation requires terminal EMA accuracy at least `pre_ema_best` and NLL below 0.1934. A valid run below 94.25 is no-improvement regardless of diagnostics. Overall verdict: **feasible enough to preflight, but not high confidence**; EXP018 supports the mechanics and exposes the central accuracy risk.
