# Proposal: Signal-Scale-Matched Positive-Negative Momentum

## Decision and hypothesis

Replace accepted PyTorch momentum SGD with beta0=1 Positive-Negative Momentum (PNM), but correct the deterministic-scale defect that caused EXP020's reviewer to reject paper-default PNM. Preserve the complete accepted CIFAR recipe: width-2 postactivation ResNet-20, N1/M7 plus 50% alpha-1 CutMix through 80%, hard crop/flip tail, batch 128, seed 42, all-parameter coupled weight decay `1e-4`, elapsed-time LR schedule, evaluator, workers, and 300-second counter.

PNM will keep zero-initialized odd/even EMAs and the paper's `+2/-1` geometry. A deterministic per-step scalar will make its response to a constant decay-augmented gradient equal to accepted PyTorch momentum at **every** step, not merely asymptotically. This isolates negative-history/noise geometry far better than unchanged-LR paper PNM, whose steady direction is 22.36x too small.

**Falsifiable hypothesis:** scale-matched PNM improves generalization through alternating negative gradient history without suppressing strong-phase fit, producing `best_test_acc >=94.25%` from the 94.15% baseline. Point prediction: **94.30%**, at least **26,091 steps** (97% of EXP010), switch accuracy at least 88.73%, and normal completion. A valid lower result is no-improvement; diagnostics cannot rescue it.

## Evidence, novelty, and counterevidence

Xie et al. report CIFAR-10 ResNet-18 error `4.48 +/-0.09` for PNM versus `5.01 +/-0.03` for momentum SGD. PNM alternates two momentum streams and subtracts stale history to amplify stochastic-gradient noise without a second gradient evaluation. This is unusually close external evidence, but the paper used a longer horizon and its official optimizer defaults to decoupled decay; it does not establish this short CutMix recipe. Source: `knowledge/papers/positive-negative-momentum.md`; Xie et al., ICML 2021.

EXP020 found no scored result: isolated Nesterov hit 96.875% one-class concentration at step 11 despite finite state and lower loss. Its report explicitly recommends a scale-matched PNM revisit. EXP022's persistent-momentum Lookahead similarly concentrated at steps 7 and 13 because parameter pullback left velocity attached to a mismatched location. These failures make immutable production-batch safety mandatory, but they do not reject PNM: this candidate neither adds Nesterov's 1.9x first direction nor moves parameters independently of its state.

The main counterevidence is that PNM's negative coefficient can itself create non-descent directions and class transients. Scale matching fixes coherent signal magnitude, not instantaneous update norm. Coupled decay is also filtered through two streams, unlike the official default. The paper's gain is therefore directional evidence only.

## Exact scale derivation

Let accepted momentum be `mu=0.9`; PNM uses `rho=mu**2=0.81`, injection `a=1-rho=0.19`, beta0=1, and `z=sqrt(5)`. For global step `t>=1`, update only the parity stream:

```text
d_t = grad_t + 1e-4 * theta_t
current_t = rho * current + a * d_t
raw_t = (2 * current_t - previous) / z
```

For a constant `d`, after updating step `t`, the current stream has `n_c=ceil(t/2)` updates and the previous stream `n_p=floor(t/2)`. Their coherent coefficients are `c_t=1-rho**n_c` and `p_t=1-rho**n_p`, so paper PNM's signal coefficient is:

```text
q_pnm(t) = (2*c_t - p_t) / sqrt(5)
```

PyTorch SGD initializes its buffer to `d_1`, then uses `b_t=mu*b_(t-1)+d_t`; its constant-gradient coefficient is:

```text
q_sgd(t) = (1 - mu**t) / (1 - mu)
```

Use the preregistered scale `s_t=q_sgd(t)/q_pnm(t)` and update `theta -= lr_t*s_t*raw_t`. Values begin `5.884389, 22.360680, 12.173050, 22.360680, 15.436015, 22.360680` and converge to `sqrt(5)/(1-mu)=22.360680`. Thus the first constant-gradient direction is exactly `d_1`, every later constant-gradient direction matches accepted momentum, and the scheduled LR retains its accepted meaning. The odd/even negative-history coefficients on changing gradients remain PNM's intervention.

No empirical update-norm matching, clipping, warmup, bias correction, buffer copying, beta0 sweep, or post-observation scalar change is allowed.

## Coupled decay semantics

Apply `d_t=grad_t+WEIGHT_DECAY*theta_t` before the PNM recurrence for every Conv/Linear weight, BN affine parameter, and bias. Do not mutate `.grad`. This preserves accepted all-parameter coupled-L2 placement and gives exact accepted shrinkage in the constant-parameter/zero-data-gradient oracle under the scale proof. On a changing trajectory the two-stream filter changes decay history; that is an unavoidable part of the optimizer hypothesis and must be reported. Do not use official-code `decoupled=True`, AdamW-style parameter multiplication, exclusions, or a retuned scalar: EXP008/009 show local decay semantics are accuracy-sensitive.

## `train.py` implementation

Implement a local `ScaleMatchedPNM(optim.Optimizer)` and replace only the accepted optimizer constructor. Pin `PNM_BETA0=1.0`, `PNM_RHO=MOMENTUM**2`, `PNM_INJECTION=1-PNM_RHO`, and `PNM_NOISE_NORM=sqrt(5)`.

Use one parameter group and one global `pnm_step` stored in the group for complete `state_dict()` semantics. Allocate two detached same-device FP32 zero buffers per parameter without RNG. On each `@torch.no_grad()` step:

1. require every gradient dense, finite, FP32, and present;
2. form out-of-place decay-augmented directions with `torch._foreach_add(grads, params, alpha=weight_decay)`;
3. choose odd on odd global steps and even on even steps; multiply only current buffers by `rho` and add directions with `alpha=0.19`;
4. form out-of-place `2*current-previous`, compute `s_t` in Python float64 from the closed form, and apply one `torch._foreach_add_` parameter update with alpha `-lr*s_t/sqrt(5)`;
5. increment parity exactly once after a successful update.

The inactive stream must remain byte-identical. `zero_grad()` remains inherited; the existing loop continues mutating `group["lr"]`. No Python per-parameter update fallback, fused extension, closure, Nesterov, momentum reset at epochs/evaluation/80%, EMA/SWA, extra synchronization, or uncounted update is permitted. PNM consumes no RNG and does not change model parameter count.

## Algebraic and immutable safety gates

Before GPU timing, serialize/fsync all evidence before assertions.

- Prove with scalar and multi-tensor constant-gradient oracles for steps 1-64 that PNM directions match PyTorch momentum within max absolute `1e-6` and relative norm `1e-6`, including coupled decay, an LR change, odd/even parity, and state save/load at steps 1, 2, 63, and 64.
- Prove a manual FP32 recurrence matches the installed foreach implementation for changing gradients through at least 16 steps and across steps 200-201. Verify exact inactive-buffer preservation, one parity increment, no `.grad` mutation, RNG neutrality, finite state, and exact scale values.
- Verify first-step parameters agree with accepted SGD within max absolute `1e-7` and relative norm `1e-6`; unlike EXP020 Nesterov, there is no intentional first-step amplification.
- Validate the existing EXP022 immutable 200-batch accepted corpus before reuse at SHA-256 `e04dc2fe9d3994cef8bf192401bc36c63f306946fd3b9a2339b9f64040318946` (94 hard, 106 CutMix). Never regenerate it to clear a veto. Materialize once and hash a separate 64-batch weak corpus before either arm.
- From bitwise-identical seed-42 model states, run explicit accepted SGD and PNM arms over all 200 shared strong batches at LR 0.1, then the same 64 weak batches with the registered abrupt LR 0.01/cosine-tail semantics. Record every loss, class histogram, update norm, gradient norm, state norm, and scale.
- Require finite logits/loss/gradients/parameters/BN buffers/optimizer state, exact BN counters, complete two-buffer state, no candidate-only maximum class share `>95%` while control is `<=95%`, terminal loss-EMA ratio `<=1.5` in each phase, and no candidate update above 5x control or 10x its preceding 16-step median.

The immutable gate directly addresses EXP020/022. Lower candidate loss cannot override concentration. Failure cannot be rescued with a new corpus, threshold, warmup, scale, beta0, decay mode, or LR.

## Timing and exposure

After safety passes, confirm one idle 97,871-MiB H20. Run one unscored conditioner and five alternating fresh-process control/candidate pairs. Each uses the real eight-worker accepted loader, identical backend flags, at least 100 warmups and 1,000 synchronized complete steps spanning the 80/20 strong/weak mix, then clean worker shutdown. Record forward/backward and optimizer CUDA-event time separately, full counted time, iterator wait, wall time, allocation, state bytes, and raw trials.

Require median-of-pair candidate/control counted-step ratio `<=1.03`, every pair `<=1.06`, per-arm CV `<3%`, projected exposure `floor(26898*control_mean/candidate_mean)>=26091`, median/p95 loader wait below 10%/20% of step time, peak allocation `<650 MiB`, stable allocator/state identities, no worker growth/leak, weak rebuild `<5s`, integrated wall/count `<=1.10`, and projected total `<540s`. Two momentum buffers should add about 8.2 MiB; repeated temporary allocations or Python-loop fallback are no-go conditions.

## Production and verdict

Confirm baseline 94.15 at `7c1e7d8`, only tracked `train.py` differs, compile/Ruff/format/diff checks pass, no stale log exists, and one H20 is idle. Run seed 42 exactly once:

```text
timeout --kill-after=5s 595s uv run train.py > run.log 2>&1
```

Require exit zero, one complete finite ten-field summary, `299.9<=training_seconds<=300.2`, `total_seconds<600`, 1,073,962 parameters, at least 26,091 steps, one valid 80% loader transition with eight workers stopped, hard weak targets, unique at-most-once-per-epoch evaluations, finite PNM state, and no retry. Improvement requires `best_test_acc>=94.25%`; a finite lower value is no-improvement.

Report switch/first-weak/best/final accuracy, final NLL, steps, optimizer-stage ratio, scale range, odd/even/update norms, VRAM, corpus/report hashes, and best/final gap. A switch below 88.73% supports harmful noise geometry despite signal matching; a healthy switch with a miss means PNM did not improve the relevant basin. Neither changes the formal verdict.

## Risks and abort criteria

- **Scientific risk — high:** close paper evidence does not transfer its long horizon or decay default; local optimizer-path failures recur twice.
- **Optimization risk — high:** negative stale history can amplify class-specific gradients even when coherent signal scale is exact.
- **Implementation risk — medium:** parity, scale indexing, foreach ordering, and coupled decay must match the derivation exactly.
- **Runtime risk — medium:** extra buffers, temporaries, and launches can lose fixed-budget exposure, although backward remains dominant.

Abort for any scope/recipe/evaluator drift; wrong recurrence, scale, parity, first-step, decay, state, RNG, or corpus digest; non-finite state; candidate-only concentration; update/loss veto; timing/exposure/memory/lifecycle miss; GPU contention; or stale logs. Do not rescue with constant 22.36 scaling, paper-default scaling, decoupled decay, buffer initialization tricks, clipping, warmup, ordinary-momentum tail, or another seed. Those are distinct experiments.
