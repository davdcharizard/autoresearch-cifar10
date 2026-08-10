# Proposal 02: Whole-Trajectory Lookahead Momentum SGD

## Exact Optimizer

Wrap the accepted PyTorch SGD trajectory with Lookahead's canonical fixed point:

- inner/fast optimizer: unchanged `SGD(lr=time_schedule, momentum=0.9, weight_decay=1e-4, nesterov=False)`;
- synchronization period: `k=5` completed inner updates;
- slow interpolation: `alpha=0.5`;
- slow initialization: detached FP32 CUDA copies of all trainable parameters before `t_start_training`;
- momentum policy: leave every SGD momentum buffer unchanged at synchronization.

After each ordinary `optimizer.step()`, and before the existing synchronize, execute on steps divisible by five:

```python
with torch.no_grad():
    torch._foreach_lerp_(slow_params, fast_params, 0.5)
    torch._foreach_copy_(fast_params, slow_params)
```

Thus `slow <- slow + 0.5 * (fast - slow)` and `fast <- slow`. The first synchronization follows inner updates 1-5. Do not reset, interpolate, or pull back momentum; change LR/decay; synchronize at phase/evaluation boundaries; or substitute another `k`/`alpha` after any gate.

`k=5, alpha=0.5` is the paper's default-like, broadly evaluated operating point and is the only externally anchored pair in this proposal. The NeurIPS 2019 evidence includes SGD on CIFAR/ImageNet, but transfer to this short CutMix trajectory remains uncertain (`papers/lookahead-optimizer.md`).

## State, BN, Timer, and Evaluator

Slow state contains only parameters. BatchNorm affine parameters participate; `running_mean`, `running_var`, and `num_batches_tracked` remain the current fast model's buffers and update on every real batch. Do not interpolate buffers: they are distribution moments, not optimizer coordinates. Do not recalibrate BN: unlike post-hoc averaging, Lookahead's pulled fast model generates subsequent training activations, so current buffers track the actual sequence of fast/slow-synchronized functions.

The current fast model is always the sole evaluated model. Between synchronization points it contains ordinary inner-SGD weights; immediately after a period it equals slow weights. Do not force an extra pull before evaluation or evaluate slow separately. This preserves the exact once-per-epoch evaluator and lets the fixed step count determine whether a partial phase/final epoch is synchronized.

Slow allocation is optimizer setup and is included in reported startup wall time, just like momentum-state setup; no real example or update occurs. Every periodic interpolation/copy is inside the accepted `t0`/synchronize interval and therefore counted in `total_training_time`. Use cached parameter tuples and foreach operations only. Add one final provenance summary, not timed per-step logging.

Architecture, 1,073,962 trainable parameters, width 2, batch 128, seed 42, N1/M7 plus p=0.5 alpha-1 CutMix through 80%, hard weak tail, LR schedule, evaluator, workers, and all dependency files remain accepted.

## Materialized Functional and Safety Gates

EXP-019 showed that fresh forkserver processes do not replay post-transform batches from seed alone. Before assertions, materialize and serialize exact post-transform tensors/targets/provenance from one strong calibration epoch plus 20 held-out strong batches, and the same for weak data. Replay identical tensors for accepted and Lookahead arms; serialize diagnostics before any veto assertion.

Require exact recurrence through the first synchronization:

- steps 1-4 fast parameters, gradients, BN buffers, loss, and momentum are bitwise accepted;
- after inner step 5, candidate pre-pull fast and momentum are bitwise accepted;
- slow and installed fast equal `0.5 * slow_0 + 0.5 * fast_5` within FP32 foreach arithmetic tolerance;
- momentum buffers remain bitwise unchanged by the pull; slow tensors are detached, finite, absent from optimizer groups/state, and parameter objects are never replaced;
- normalized pull distance is exactly 0.5 of the pre-pull fast-to-slow distance, subject only to FP32 tolerance.

Replay at least 200 distinct strong batches with the accepted hard/CutMix pattern, then apply the exact `0.1 -> 0.01` LR and strong-to-weak data transition and replay 200 distinct weak batches without resetting optimizer, slow weights, momentum, or BN. Require finite losses/gradients/parameters/slow weights/momentum/buffers, candidate loss EMA no more than 1.5x paired control, no candidate-only one-class concentration above 95%, exactly 80 synchronizations, and finite nonzero fast-slow displacement at every period. Do not insert an extra synchronization at the transition.

The BN policy is also fail-closed. At the end of each materialized phase, compare the production current-buffer candidate against an analysis-only clone with identical current fast parameters and BN moments recalibrated over one full persisted epoch of that phase. On 20 held-out same-distribution batches require logit cosine at least 0.999, top-1 agreement at least 99%, and mean cross-entropy difference at most 0.02. If current buffers fail, do not average buffers or recalibrate production; retire Lookahead.

Catastrophic thresholds do not predict full-phase accuracy. Pre-register 87.08% as the recurring switch-underfit marker and 89.0% as healthier expectation; neither permits tuning or rerun.

## Timing and Exposure Gates

On the sole idle H20, run five alternating fresh-process accepted/Lookahead pairs over identical persisted hard/probability-target batches. Use 100 warm steps and at least 1,000 measured steps so each candidate trial contains 200 pulls. Report full-step mean/median/p95, CV, pull-step versus non-pull-step distributions, CUDA optimizer-stage time, and peak memory.

Advance only if:

- candidate/control median full-step ratio is at most `1.02`, no pair exceeds `1.04`, and paired-ratio CV is below 2%;
- projected whole-run exposure is at least **26,629 steps** (99% of EXP-010's 26,898);
- pull-step p95 is below 1.10x accepted p95 and non-pull steps are within 1%;
- extra CUDA time is localized to the two foreach parameter operations and timer accounting error is below 1%;
- slow-state peak allocation stays below 620 MiB with no growth;
- the exact eight-worker switch passes and total runtime projects below 540 seconds under unchanged evaluations.

A timing/safety miss makes the canonical pair infeasible. Do not update slow weights on CPU, reduce pull frequency, change alpha, fuse with custom CUDA, or exempt copies from counted time.

## Why This Is Not Failed SWA

EXP-018 left SGD untouched, uniformly averaged eight old annealed endpoints once, then installed the mean for terminal evaluation; that backward-biased model trailed its own online best by 0.17 points after separate BN recalibration. Lookahead instead interpolates every five updates from step 1, immediately copies slow into fast, and lets the pulled weights determine every subsequent gradient and BN observation. Its slow state is an exponential history of short fast excursions, not a post-hoc uniform mean of late checkpoints. Success would validate a different optimization trajectory; failure cannot be rescued by calling it a narrower SWA window.

## Hypothesis, Risks, and One Run

**Hypothesis:** canonical Lookahead reduces stochastic fast-weight variance throughout the N1/M7+CutMix and weak phases, retains at least 99% exposure and healthy strong fit, and raises `best_test_acc` from 94.15% to at least 94.25%.

The main risk is systematic under-progress: pulling halfway back every five steps can waste useful motion in a fixed 300-second trajectory, especially during the high-LR strong phase. Unchanged momentum applied from pulled weights can overshoot; current BN buffers may lag pulls; and the accepted final-at-best path gives variance reduction limited headroom. The paper's robust defaults do not establish a gain for shallow ResNet-20.

If all gates pass, change only `train.py`, run seed 42 exactly once on the pinned idle H20 as `uv run train.py > run.log 2>&1`, and forbid a valid-run retry. Require exit zero, approximately 300 counted seconds, total below 600, finite summary, 1,073,962 parameters, at least 26,629 steps, `floor(num_steps/5)` pulls, one 80% switch, eight workers stopped, about 50% CutMix, hard weak targets, and unique at-most-once-per-epoch evaluations.

Accept only if `best_test_acc >=94.25%`. Report pull count/cost, fast-slow distances, switch/first-weak/best/final/NLL, exposure, VRAM, and wall time. A valid miss rejects `k=5, alpha=0.5`, no-momentum-pullback Lookahead; do not try another pair or combine Nesterov/PNM.
