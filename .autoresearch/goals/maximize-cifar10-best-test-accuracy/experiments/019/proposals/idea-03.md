# Proposal 03: One-Epoch-Half-Life Weak-Tail EMA

## Exact Protocol

Preserve accepted EXP-010 training unchanged until the first optimizer step whose pre-step counted progress is at least 90%. Initialize a detached FP32 CUDA shadow from the post-step online parameters, then update after every later optimizer step with:

```python
EMA_DECAY = 0.5 ** (1.0 / 390.0)  # 0.9982242780002207
torch._foreach_lerp_(ema_params, online_params, 1.0 - EMA_DECAY)
```

The fixed half-life is exactly one accepted 390-step epoch. It is derived from loader cadence, not tuned to accuracy. Start at 90% so the shadow excludes strong data and early weak-distribution adaptation while receiving roughly six to seven half-lives before termination. Do not change the start, decay, update interval, or LR schedule.

This is materially distinct from EXP-018: it is a recency-weighted per-step EMA, not a uniform average of separated epoch endpoints; SGD continues through 100%; there is no 98% install cutoff and no reserved BN-recalibration phase.

## BN and Evaluation Semantics

EMA only `named_parameters()`, including BN affine weights/biases. Do not average any buffers. At each existing evaluation after EMA starts, temporarily install EMA parameters into the same parameter objects while retaining the **current online** BN `running_mean`, `running_var`, and `num_batches_tracked`; evaluate once; restore online parameters bitwise in `finally`. The evaluator runs in eval mode, so buffers cannot mutate during the swap.

Current buffers are an approximation, not exact moments for EMA weights. They are selected because the shadow's one-epoch half-life keeps it near the online weak-tail function, while BN's momentum 0.1 tracks the same weak distribution over a much shorter horizon. EMA-averaging buffers would double-lag and still not produce moments for averaged parameters; production recalibration would add data exposure and repeat EXP-018's separate mechanism. Neither is an allowed fallback.

Before EMA exists, evaluations remain online. Afterward, each scheduled evaluation is EMA-only, including terminal evaluation. Never evaluate online and EMA in the same epoch. This preserves the accepted number of metric opportunities but knowingly forfeits late online checkpoints.

## Counted-Time Fairness

Cache online/shadow tensor tuples once. Initialize EMA inside the first eligible timed step. Run every EMA update before the step's existing synchronize, so its GPU cost is included in `dt`. Around each evaluation, synchronize and add online-to-backup copy plus EMA install time to `total_training_time`; after evaluation, restore in `finally`, synchronize, and charge restoration time as well. `Eval.evaluate()` alone remains excluded, matching the harness.

The shadow and backup are detached same-device FP32 tensors, never optimizer parameters. Momentum, gradients, online parameters, RNG, and loader state remain untouched. Architecture and reported trainable parameter count stay 1,073,962.

## Feasibility Gates

Functional preflight must prove exact initialization, the scalar EMA recurrence against an FP64 reference, finite/order-aligned shadows, absence from optimizer state, and bitwise online restoration after forced evaluation failure. Swaps must preserve BN buffers/counters, momentum, gradients, CPU/CUDA RNG, and parameter object identities.

The current-buffer approximation is a hard gate. Across a three-epoch (1,170-step) weak diagnostic, compare at three predeclared points representing early, mature, and terminal-like EMA ages:

1. EMA parameters plus current online buffers; and
2. an analysis-only clone of the same EMA parameters whose BN moments are reset and recalibrated over one full weak-loader pass.

On 20 held-out weak training batches, require FP32 logit cosine at least 0.999 at every point, top-1 agreement at least 99%, and mean cross-entropy difference at most 0.02. This uses no test-set evaluation. If it fails, the declared production BN state is not credible and the candidate is infeasible; do not recalibrate production.

On one idle H20, run five alternating fresh-process accepted/candidate timing pairs with 100 warmups and at least 500 active-EMA synchronized steps, plus seven swap/evaluate-restore simulations and an integrated 90/10 schedule. Require:

- active-EMA step ratio at most 1.03;
- whole-run projected optimizer exposure at least **26,629 steps** (99% of EXP-010's 26,898);
- shadow plus backup peak allocation below 650 MiB;
- at least five EMA evaluations, timer-accounting error below 1%, and projected total wall time below 540 seconds;
- unchanged strong-loader shutdown, weak-loader provenance, and at most one evaluation per unique epoch.

A gate miss retires this policy. Do not move EMA to CPU, update less often, omit charged swaps, change buffers, or shorten the half-life.

## Hypothesis and One Run

EXP-018 showed uniform 86-98% SWA pulled a monotonically improving solution backward: final SWA was 0.17 points below its own online best and worsened NLL. A one-epoch-half-life EMA gives exponentially little weight to those older states while still damping sub-epoch SGD noise. The fixed hypothesis is that this recent shadow retains 99% exposure and raises `best_test_acc` from 94.15% to at least 94.25%.

If all gates pass, modify only `train.py`, run seed 42 once on the sole idle H20 as `uv run train.py > run.log 2>&1`, and forbid reruns or alternate EMA settings. Require exit zero, about 300 counted seconds, total below 600, finite standard summary, one 80% switch with eight workers stopped, approximately 50% CutMix, hard weak targets, at least 26,629 steps, EMA start at/after 90%, at least five EMA-only late evaluations, and exact restoration after every nonterminal swap.

Accept only if `best_test_acc >=94.25%`. Report EMA updates/evaluations, effective age, charged overhead, exposure, BN policy, best/final/NLL, VRAM, and wall time. A valid miss rejects this exact start/half-life/buffer policy; do not tune decay, start, or evaluation semantics.

## Risks

- EXP-010 and EXP-018 both indicate a still-improving tail; even short EMA lag may lose to online.
- Current BN buffers are approximate despite the diagnostic agreement gate.
- Consecutive parameters may be too correlated for useful smoothing.
- Per-step foreach and charged swaps may cost more exposure than expected.
- Replacing online late evaluations can hide a superior endpoint, while evaluating both would bias the max metric.
- A bare 94.25% pass remains weak single-seed evidence.
