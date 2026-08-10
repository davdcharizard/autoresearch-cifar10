# Proposal: Default-beta0 Positive-Negative Momentum

## Decision

Replace accepted PyTorch momentum SGD with the paper's stochastic Positive-Negative Momentum (PNM) recurrence for the entire run. Pin:

```python
PNM_BETA0 = 1.0
PNM_NOISE_NORM = math.sqrt((1.0 + PNM_BETA0) ** 2 + PNM_BETA0**2)
```

Use accepted `MOMENTUM=0.9` as paper beta1, accepted per-step elapsed LR as paper eta, and accepted all-parameter coupled `WEIGHT_DECAY=1e-4`. Maintain two alternating FP32 momentum streams. Do not combine PNM with Nesterov, momentum reset, warmup, clipping, a different LR, decoupled decay, EMA/SWA, or a beta0 sweep.

The hypothesis is that default beta0=1 amplifies anisotropic stochastic-gradient noise enough to improve generalization on the already-capable width-2 model, reaching at least 94.25% from the 94.15% frontier with a point prediction of **94.35%**, while retaining at least 97% of EXP-010's 26,898 updates.

## Evidence and beta0 Choice

Xie et al. report CIFAR-10 ResNet-18 error `4.48 +/- 0.09` for PNM versus `5.01 +/- 0.03` for momentum SGD. This is unusually close external evidence in dataset and architecture family, but it is not the local operating point: ResNet depth, augmentation, CutMix, LR schedule, decay implementation, optimizer convention, horizon, and evaluation all differ.

Use beta0=1 because it is the paper and official implementation default, not because it has been tuned locally. The unnormalized positive-negative combination has coefficients `+2` and `-1`. Under the paper's approximate independent-stream analysis, its noise variance is multiplied by:

```text
(1 + beta0)^2 + beta0^2 = 2^2 + 1^2 = 5
```

The update divides by `sqrt(5)`, pinning the paper's normalization. This does not remove the intended relative noise increase: the positive and negative stream combination reduces coherent signal relative to its stochastic difference. It also creates a deterministic scale mismatch against accepted PyTorch momentum, addressed explicitly below.

No beta0=0, 0.5, or adaptive policy may replace beta0=1 after preflight. A failure retires this exact literature point.

## Exact Paper-Faithful Recurrence

Let `mu=0.9`, `rho=mu^2=0.81`, `injection=1-rho=0.19`, `beta0=1`, and `z=sqrt(5)`. For every parameter `theta_t`, form the accepted coupled-decay gradient before momentum:

```text
d_t = grad_t + 1e-4 * theta_t
```

Maintain odd and even streams, both initialized to zero. On global step `t` starting at one:

```text
if t is odd:
    odd  = rho * odd  + injection * d_t
    current, previous = odd, even
else:
    even = rho * even + injection * d_t
    current, previous = even, odd

direction_t = ((1 + beta0) * current - beta0 * previous) / z
theta_(t+1) = theta_t - lr_t * direction_t
```

Only the current parity stream updates; the opposite stream remains unchanged. The physical odd buffer is positive on odd steps and negative on even steps, while the even buffer does the reverse. The sign attaches to current-versus-previous role, not permanently to one named tensor. Global parity continues monotonically across epochs, evaluation, the 80% loader switch, and LR changes.

The first two exact directions are:

```text
direction_1 = 0.38 * d_1 / sqrt(5) = 0.169941... * d_1
direction_2 = (0.38 * d_2 - 0.19 * d_1) / sqrt(5)
```

There is no bias correction or special first-step warm start in Algorithm 2 or the official PNM implementation. Both buffers start at exact zero. Adding PyTorch-style first-buffer initialization, copying one stream into the other, or delaying the negative term would define a different optimizer.

## LR Normalization and the Ordinary-Momentum Confound

The scheduled group LR is paper eta. Apply it as `lr_t / sqrt(5)` to the raw `2*current - previous` combination. Do not multiply by `sqrt(5)`, `1/(1-mu)`, or an empirically chosen adapter to match accepted update norms. Such a factor may create a defensible alternative, but it is not this paper-faithful default-beta0 experiment.

This makes the comparison to accepted `torch.optim.SGD(momentum=0.9)` nontrivial. PyTorch uses an unnormalized single buffer:

```text
b_t = 0.9 * b_(t-1) + d_t       (b_1 = d_1)
theta_(t+1) = theta_t - lr_t * b_t
```

PNM uses normalized two-step EMAs with `0.19` gradient injection and divides their combination by `sqrt(5)`. Its first direction is only `0.169941*d_1` versus PyTorch's `d_1`. For a constant gradient after transients, each PNM stream approaches `d`, giving direction `d/sqrt(5)=0.447214*d`, while PyTorch's buffer approaches `d/(1-0.9)=10*d`. At the same numeric LR, steady deterministic PNM drift is therefore about **22.36 times smaller** than accepted PyTorch momentum.

Consequently, a production result is the net effect of the paper recurrence, including altered deterministic scale, temporal filtering, and noise geometry. It cannot be described as a pure noise-variance intervention against EXP-010. The close paper comparison may have used its own LR tuning/convention and does not resolve this local mismatch. This is the candidate's main adversarial weakness and a reason external review may reject it before execution.

## Implementation Scope in `train.py`

Implement a local `PositiveNegativeMomentum(optim.Optimizer)` because dependencies cannot change. Replace only the optimizer constructor plus add the class and named constants. Preserve the existing scheduler's `group["lr"]` mutation and inherited `zero_grad()` behavior.

Use one parameter group and one global integer step counter. Lazily allocate exactly two detached FP32 zero buffers per parameter on the first step. All accepted model parameters have dense gradients every step; assert this in preflight. Store buffers and step in optimizer state so `state_dict()` is complete even though production does not checkpoint.

Use the installed CUDA foreach API for the exact batched recurrence:

1. construct decay-augmented directions from cached gradients and parameters;
2. multiply the current parity buffers by `0.81` and add directions with alpha `0.19`;
3. form out-of-place `2*current - previous` temporaries;
4. update parameters with alpha `-group_lr/sqrt(5)`.

Do not mutate `.grad`, parameter identities, the inactive buffers, or RNG state. No per-tensor Python-loop fallback is allowed if foreach semantics fail; that changes timing and floating-point ordering and must be treated as a no-go.

## Coupled Weight Decay

Apply `d_t = grad_t + weight_decay*theta_t` before the parity-stream update for every parameter, including Conv/Linear weights, BN affine parameters, and biases. This matches the accepted all-parameter L2-style semantics and the paper algorithm with official optimizer `decoupled=False`.

Do not use the official repository's default `decoupled=True`, which multiplies parameters outside momentum. Do not mutate `p.grad` in place as the reference implementation does; an out-of-place/foreach decay-augmented direction is algebraically equivalent and avoids contaminating gradient diagnostics. Manual recurrence tests pin the exact operation order used in production.

Because decay enters both alternating streams and later receives positive or negative coefficients, PNM changes effective regularization even at the same `1e-4` scalar. This is part of the net optimizer replacement, not evidence that decay has been isolated.

## State, Alternation, and RNG Gates

Before real-batch work:

1. Static checks require exactly one model parameter group, 1,073,962 model parameters, beta1 0.9, beta0 1, coupled decay `1e-4`, and no Nesterov/fallback path.
2. A scalar and multi-tensor FP64 reference over at least six alternating steps, including changing gradients, coupled decay, and an LR change, must match FP32 production buffers/directions/parameters within operation-order tolerance.
3. Explicitly verify the first two formulas above, zero initialization, odd/even update exclusivity, frozen inactive buffer, continuous parity across a simulated epoch/evaluation/switch boundary, and no bias correction.
4. Construct seed-42 accepted and PNM models with bitwise-identical initial parameters, buffers, hard/soft logits, losses, raw gradients, and CPU/CUDA RNG states. Optimizer construction must consume no RNG.
5. PNM state must contain two same-shape FP32 buffers per parameter, no gradient-bearing state, finite values, stable identities, and exactly one increment of global parity per successful optimizer step.
6. Hard `[128]` and probability `[128,10]` targets, state serialization/restore, inherited zero-grad, scheduler LR mutation, and model evaluation must all work.

Any failure blocks execution. Do not repair it by changing initialization, beta0, normalization, decay, or backend.

## Persisted Production-Batch Safety Gate

EXP-019 proved that fresh forkserver processes do not reproduce post-transform batches from seed alone. Materialize the exact safety corpus **once before either optimizer arm**, persist it to an untracked experiment artifact, and record a SHA-256 digest before any assertion:

- 200 distinct N1/M7 batches from the production p=0.5 CutMix collator, with 45-55% mixed targets;
- production shutdown of all eight strong workers;
- 64 weak crop/flip hard-label batches from the rebuilt loader.

Persist the actual post-transform FP32 inputs and targets, not dataset indices or RNG states. Both accepted PyTorch momentum and PNM arms must load the same immutable artifact, start from cloned bitwise-identical model state, and run in the same process/device environment. If a controller bug requires retry, replay the same digest; generating another corpus is not an allowed retry.

Train 200 strong steps at LR 0.1, then preserve optimizer state/parity, set LR to 0.01, and train 64 weak steps. Serialize all threshold evidence before raising. Require:

- manual PNM direction/buffer equality for the first six steps and at the 200-to-201 transition;
- finite logits, losses, gradients, parameters, directions, and both buffers for all 264 steps;
- no candidate-only step above 95% one-class predictions when control is at or below 95%; persist paired histograms;
- no PNM update-norm spike above 10x its preceding 16-step median after initialization;
- strong steps 101-200 loss EMA no more than 2.0x control and no terminal-eight-batch collapse to fewer than two classes;
- first eight weak loss EMA no more than 2.5x control and no candidate-only finite/concentration failure across the weak segment.

These broad bounds accommodate PNM's much smaller deterministic drift and detect gross failure only. Passing cannot show that noise improves generalization. Failure cannot be rescued with a larger LR, copied buffer, momentum reset, beta0 change, or a different corpus.

## Timing, Exposure, and Memory Gates

On one idle 97,871 MiB H20, run five alternating fresh-process accepted/PNM pairs using the same persisted hard/soft batch corpus, reset model/optimizer state, 100 warmups, and at least 1,000 synchronized complete production-timed steps per trial.

Require:

- PNM/control median-of-trial-means step ratio `<=1.03`;
- trial-mean CV `<=2%` each and PNM p95 no more than 1.08x control;
- projected exposure `floor(26,898 * control_mean / pnm_mean) >=26,091`, retaining 97%;
- peak allocation `<625 MiB` and no more than 24 MiB above paired control;
- finite state and conservative total-runtime projection below 540 seconds.

PNM doubles persistent momentum storage from about 4.10 to 8.20 MiB and needs temporary decay/direction tensor lists. Model forward/backward remain identical, but several foreach operations replace PyTorch's optimized SGD step. The measured accepted optimizer stage is small, yet launch and allocation cost must be observed. No Python-loop, fused, CPU-state, or uncounted-update fallback is allowed.

## Evaluator and Protocol Fairness

Keep seed 42 with one production run and no reroll. PNM consumes no randomness, so initial model and data-policy RNG are unchanged; numerical divergence begins only at its first update. The persisted safety artifact is never used for production training.

The fixed evaluator, early checkpoints, dense weak-tail cadence, terminal evaluation, and at-most-one-call-per-epoch structure remain unchanged. No control evaluation or extra model-selection opportunity is added. Parameter count remains 1,073,962; optimizer-state tensors do not count as model parameters.

## One-Run Hypothesis and Decision

**Hypothesis:** paper-default beta0=1 PNM will use alternating positive-negative stochastic-gradient history to improve generalization enough to reach **94.35%** point-estimate best test accuracy, while retaining at least 26,091 updates and completing normally under the accepted recipe.

Require exit zero, ten unique finite summary fields, 300.0 counted seconds, total below 600 seconds, exactly 1,073,962 model parameters, memory within preflight, at least 26,091 steps, one 80% switch, eight stopped workers, 45-55% strong CutMix, hard weak targets, and unique evaluation epochs.

Compare EXP-010's 89.73% switch, 93.16% first weak, 94.15% final/best, 0.1934 NLL, 26,898 steps, and 330.7-second total. Log no new production diagnostics; infer state validity from normal completion and preflight. A switch below 87.08 is expected evidence of insufficient deterministic drift but cannot trigger LR or state rescue.

- **Improvement:** `best_test_acc >=94.25%` with all protocol and recurrence gates passing.
- **Valid lower result:** no-improvement; revert to accepted SGD with no reroll or PNM tuning.
- **Accuracy pass below 26,091 steps:** formally above the metric gate but exposure-confounded; do not describe PNM as low-cost.
- **Crash, timeout, state/RNG/target/timer/evaluator/lifecycle fault:** invalid; fix only protocol defects with recurrence/beta0/LR semantics unchanged.

## Failure Modes

- **Deterministic-scale mismatch:** at accepted numeric LR, paper PNM's steady constant-gradient direction is about 22.36x smaller than PyTorch momentum's; severe underfit is plausible and external CIFAR results may rely on different LR conventions/tuning.
- **Cold alternating initialization:** zero streams yield only 0.1699x the accepted first direction and oscillatory early signal without bias correction.
- **Excess noise under CutMix:** beta0=1 targets stronger stochasticity, but N1/M7 plus CutMix already regularizes the strong phase and has shown underfit sensitivity.
- **Negative cancellation:** the previous stream may contain useful shared-feature history that the `-1` coefficient removes, producing non-descent directions.
- **80% state shock:** the first weak gradient enters one parity stream with coefficient `+2` while the opposite strong-view stream enters with `-1`; LR falls but state is not reset.
- **Coupled-decay distortion:** positive-negative filtering applies to decay-augmented directions and may differ materially from the paper's best reported setup.
- **Optimizer overhead:** two buffers, temporaries, and multiple foreach launches can remove valuable fixed-budget updates.
- **Approximate-independence failure:** odd/even gradients are correlated along a changing parameter trajectory; the fivefold variance argument is asymptotic and idealized.
- **Paper-transfer gap:** ResNet-18 results do not fix the local shallow model, schedule, augmentation, or PyTorch momentum-convention mismatch.
- **Single-seed resolution:** a 0.10-point pass is ten test images and cannot be confirmed by reroll.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`, `02-system-understanding.md`, `03-experiment-learnings.md`, and `04-results.tsv`.
- `goals/maximize-cifar10-best-test-accuracy/experiments/020/01-brainstorm.md` and `papers/positive-negative-momentum.md`.
- [Xie et al., ICML 2021 paper](https://proceedings.mlr.press/v139/xie21h.html) and [official PNM implementation](https://github.com/zeke-xie/Positive-Negative-Momentum/blob/main/pnm_optim/pnm.py): recurrence, defaults, normalization, and CIFAR results.
- `goals/maximize-cifar10-best-test-accuracy/experiments/019/04-analysis.md`: exact post-transform persistence requirement after non-replayable forkserver safety paths.
- `goals/maximize-cifar10-best-test-accuracy/experiments/019/proposals/idea-01.md`: adversarial ordinary/Nesterov semantics used as comparison discipline.
- Current `train.py`: accepted optimizer, model/data/schedule, timer, evaluator, and worker lifecycle.
