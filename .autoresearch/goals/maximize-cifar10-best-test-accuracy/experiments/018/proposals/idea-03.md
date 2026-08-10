# Proposal: Isolated PyTorch Nesterov Momentum

## Decision

Test one literal optimizer change on the exact accepted EXP-010 recipe:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)
```

The production diff may add only `nesterov=True`. Pin installed PyTorch 2.9.1 defaults: `momentum=0.9`, `dampening=0`, `maximize=False`, `foreach=None`, `fused=None`, and `differentiable=False`. No schedule, LR, decay, batch, data, target, architecture, graph, precision, seed, timing, evaluation, or logging change is part of the candidate.

The one-run hypothesis is that Nesterov's current-gradient correction will improve optimization/generalization under the validated noisy long-plateau trajectory enough to raise `best_test_acc` from 94.15% to at least 94.25%, with a point prediction of **94.30%**, while retaining at least 99% of EXP-010's 26,898 updates.

## Deconfounding EXP-001

EXP-001 did not isolate Nesterov. It simultaneously enabled Nesterov, replaced the scheduler, reduced the `lr=0.1` hold to 15%, introduced an 85%-budget cosine, enabled persistent workers, and changed evaluation cadence. It reached 91.57% with a 0.0215 final train-loss EMA, below the then-91.67% baseline. Its own analysis identifies early annealing and loss of high-LR implicit regularization as the dominant mechanism, while explicitly retaining Nesterov as a confound.

EXP-002 simultaneously removed Nesterov **and** restored the productive 80% `lr=0.1` hold followed by a `0.01` cosine tail, reaching 91.83%. The 0.26-point cross-run difference cannot assign a sign to Nesterov. The accepted lineage then added N1/M7, width 2, and p=0.5 CutMix without revisiting the optimizer flag.

EXP-010 remains the 94.15% frontier after EXP-017's learned transition shortcuts reached only 94.09%. The optimizer question is therefore still open. This proposal preserves every accepted schedule/data/graph decision and changes the flag alone, resolving the original confound cleanly.

## Exact PyTorch Semantics

For parameter `theta`, raw minibatch gradient `grad`, coupled weight decay `lambda=1e-4`, momentum `mu=0.9`, and scheduled LR `gamma`, installed PyTorch computes:

```text
d_t = grad_t + lambda * theta_(t-1)
b_t = d_t                              if t = 1
b_t = mu * b_(t-1) + d_t               if t > 1 (dampening = 0)
u_t = d_t + mu * b_t                    if nesterov = True
theta_t = theta_(t-1) - gamma * u_t
```

Accepted ordinary momentum uses `u_t=b_t`. PyTorch does not run an extra forward at lookahead weights; it evaluates the model once at current parameters and changes only the update direction. This is PyTorch's documented variant, not a hand-written or framework-generic NAG interpretation.

Coupled decay enters `d_t` before momentum. Nesterov therefore applies its correction to the decay-augmented direction as well as the loss gradient. The configured scalar stays `1e-4`, but the effective trajectory-level regularization changes. Decoupling or rescaling decay would be a second lever and is forbidden.

## Initial Equivalence Boundary

Initial **model behavior** must be equivalent. With reset seed 42, control and candidate must have bitwise-identical parameters, buffers, logits, losses, raw gradients, CPU/CUDA RNG states, loader stream, and empty optimizer state before the first update. Optimizer construction consumes no RNG and adds no model state.

Initial **parameter updates must not be equivalent**. Both optimizers initialize the first momentum buffer to the same `b_1=d_1`, but:

```text
ordinary direction  = d_1
Nesterov direction  = d_1 + 0.9*d_1 = 1.9*d_1
```

Thus Nesterov's first update at accepted LR 0.1 is 1.9 times the ordinary direction before FP32 parameter-storage rounding. That difference is the intervention. Requiring equal first deltas would force LR scaling, warmup, or a custom buffer initialization and would no longer test `nesterov=True` in isolation.

What matters is exact attribution: equal pre-step state and gradients, equal first momentum buffers, and stored candidate/control parameters matching their respective manual PyTorch formulas. Because `after-before` can lose precision when a small delta is subtracted from a large FP32 value, compare full manually updated tensors and direction norms rather than demanding an elementwise 1.9 ratio for rounded zero deltas.

## Optimization and Generalization Mechanism

Nesterov combines the low-pass momentum history with an extra current minibatch direction. Along gradients that remain coherent across batches, this can respond faster than ordinary momentum and alter which basin the long `lr=0.1` exploration reaches. At the 80% strong-to-weak transition, it may also react faster to the first hard weak gradients while retaining the plateau momentum buffer.

The same mechanism can amplify high-frequency stochastic directions. The EXP-018 SGD-noise paper does not establish a Nesterov gain; it shows that stochastic-gradient noise and LR schedule affect generalization and that lower train loss does not guarantee better test accuracy. This proposal therefore preserves batch 128, the entire LR schedule, and the N1/M7/CutMix noise source. Nesterov changes how that same noisy sequence is filtered rather than reducing noise through a larger batch.

A useful outcome would be a better generalizing basin, not merely faster loss reduction. EXP-001's very low train loss and EXP-017's higher switch fit both failed to improve final accuracy. Train loss, short-horizon fit, and switch accuracy are diagnostics only; the fixed evaluator's test top-1 remains decisive.

## Accepted Recipe Held Fixed

Preserve:

- width-2 postactivation ResNet-20, active residual branches, Option-A shortcuts, global average pooling, and 1,073,962 parameters;
- seed 42, current Kaiming/BN initialization, FP32 model/training/evaluation, and batch 128;
- N1/M7 plus alpha-1 CutMix probability 0.5 through 80% counted progress;
- hard-label weak crop/flip loader after the same eight-worker shutdown/rebuild;
- one all-parameter optimizer group, momentum 0.9, and coupled decay `1e-4`;
- `lr=0.1` through 80%, abrupt step to `0.01`, cosine to `1e-4`, and 300-second counted timer;
- loader/collator RNG handling, hard/probability-target CE, evaluator branch/cadence, maximum-step guard, and summary.

Do not add warmup, clipping, gradient scaling, lower LR, alternate momentum, decay compensation, fused or forced-foreach SGD, EMA, schedule smoothing, extra evaluation, or any fallback. A failed gate or valid no-improvement retires this exact operating point.

## Semantic and State Gates

Before GPU trajectory work:

1. Static diff/AST checks prove the only tracked semantic change is literal `nesterov=True`; the optimizer still has one group and exact accepted options.
2. A hand-computed FP32 test over at least three steps, including nonzero coupled decay and an LR change, matches installed ordinary and Nesterov parameter/momentum recurrence.
3. Paired seed-42 models prove bitwise-identical initial state, hard/soft logits, losses, raw gradients, BN buffers, and RNG states.
4. After step one, corresponding momentum buffers are bitwise equal and match `grad + 1e-4*theta_0`; candidate/control parameters match their separate manual formulas; resolvable direction norms have the required 1.9 ratio.
5. Later states follow the divergent manual recurrence, remain finite, and contain the same optimizer keys/shapes/dtypes/devices. Parameter identity/group membership never changes.
6. Optimizer construction and stepping consume no CPU/CUDA RNG. Hard `[128]` and CutMix probability `[128,10]` targets both execute normally.
7. Model graph, parameter count, timer, evaluator reachability, loader switch, and worker lifecycle remain unchanged.

Failure blocks execution. Correct only verification or implementation defects; do not normalize away the intended update difference.

## Production-Batch Trajectory Gate

Use real production inputs because prior work showed synthetic Gaussian probes can misclassify stability. In one disposable fresh process, materialize **200 distinct seed-42 N1/M7 batches** through the exact p=0.5 CutMix collator, require 45-55% mixed targets, and feed identical stored batches in identical order to paired accepted/Nesterov models from the same state.

Record per-step loss, finite status, global raw-gradient/update/momentum norms, first-step manual checks, and paired prediction histograms. Require:

- exact pre-step equality and the required first buffer/manual 1.9 direction result;
- finite logits, losses, gradients, parameters, and optimizer state for all 200 steps;
- first-batch replay loss after Nesterov step no greater than 2x its pre-update loss;
- no candidate-only step above 95% one-class predictions when control is at or below 95% at that same step;
- over steps 101-200, candidate loss EMA no more than 1.5x control and no final-eight-batch candidate-only collapse to fewer than two predicted classes;
- no post-step-one candidate global-update spike above 10x its preceding 16-step median.

These broad bounds detect an EXP-014/016-style collapse; they do not select for short loss superiority. EXP-015 passed favorable real-batch short checks and still underfit the full strong phase. If Nesterov fails, do not relax thresholds, reroll batches, warm up, clip, or lower LR.

## Timing, Exposure, and Wall Gates

On one idle 97,871 MiB H20, run five alternating fresh-process control/Nesterov pairs with reset identical model/optimizer states, batch 128, deterministic alternating hard/probability targets, 100 warmups, and at least 1,000 synchronized complete production-timed steps per trial.

Require:

- candidate/control median-of-trial-means step ratio `<=1.01`;
- trial-mean CV `<=2%` for each and candidate p95 no more than 1.04x control;
- projected exposure `floor(26,898 * control_mean / candidate_mean) >=26,629`, or 99% of EXP-010;
- candidate peak allocation `<610 MiB` and no more than 8 MiB above paired control;
- finite losses/state and conservative end-to-end runtime projection below 540 seconds.

Forward/backward account for 97.6% of measured GPU-stage time, while reset plus SGD is only 1.67%; Nesterov should be near-neutral but still adds update arithmetic. Inference is graph-identical and initial logits must be bitwise equal. A timing miss is a no-go; do not force another optimizer backend or exclude work from the timer.

## Fixed-Seed Evaluator Fairness

Run one production experiment only after every gate passes. Seed remains 42 with no reroll. Since the optimizer consumes no randomness, candidate and accepted control use the same initialization and stochastic data-generation policy; trajectory divergence begins only at the declared first parameter update.

The existing evaluator call remains in the same single per-epoch branch. Early checkpoints `(0.2,0.4,0.6,0.7)`, dense weak-tail evaluations, terminal evaluation, and at-most-one-call-per-epoch behavior are unchanged. Do not add paired control evaluation, online diagnostic evaluation, or checkpoint selection. `best_test_acc` therefore receives the same number and schedule of opportunities as EXP-010, modulo ordinary epoch-count variation from measured step cost.

## One-Run Hypothesis and Decision Rule

**Hypothesis:** isolated PyTorch Nesterov at momentum 0.9 will preserve at least 26,629 updates, keep the 80% strong checkpoint above the 87.08 underfit marker, and reach **94.30%** point-estimate best test accuracy, clearing the formal **94.25%** threshold.

Require exit zero, ten unique finite summary values, 300.0 counted seconds, total below 600 seconds, exactly 1,073,962 parameters, memory within gate, at least 26,629 steps, one 80% augmentation/CutMix switch, eight stopped workers, 45-55% strong CutMix, hard weak targets, and unique evaluation epochs.

Compare EXP-010's 89.73% switch, 93.16% first weak, 94.15% final/best, 0.1934 NLL, 26,898 steps, and 330.7-second total. Also inspect behavior immediately after the tenfold LR/augmentation transition because Nesterov carries the plateau buffer into a new target distribution. These diagnostics explain but never override the primary metric.

- **Improvement:** `best_test_acc >=94.25%` and every integrity gate passes.
- **Valid lower result:** no-improvement; no reroll, LR/momentum/decay tuning, or fallback.
- **Accuracy pass below exposure floor:** formally above the metric gate but timing-confounded; do not claim near-zero optimizer cost.
- **Crash, timeout, RNG/state/target/timer/evaluator/lifecycle fault:** invalid; fix only protocol faults with the exact one-keyword candidate unchanged.

## Predicted Benefit and Failure Mechanisms

The plausible successful range is **94.25-94.40%**. Nesterov adds no data, capacity, or invariance, so a multi-point gain has no local or paper support. The expected benefit is a small basin-selection/generalization effect, not higher exposure.

- **First-step overshoot:** the required 1.9x update at LR 0.1 can destabilize class geometry before momentum history forms.
- **Noise amplification:** the extra current-gradient term can emphasize CutMix/RandAugment stochasticity instead of beneficially filtering it.
- **Implicit-regularization change:** coupled decay and gradient noise pass through a different update filter, possibly finding a sharper solution despite lower train loss.
- **Transition shock:** accumulated strong-phase momentum plus current weak hard-label gradient can overshoot after the 80% LR step.
- **No generalization headroom:** ordinary momentum already supports the 94.15% final-equals-best frontier; optimizer geometry may move fit but not test accuracy, as EXP-017 warns more generally.
- **Confounded historical evidence:** EXP-001 cannot validate or reject isolated Nesterov.
- **Short-probe limitation:** 200 stable batches exclude gross collapse but cannot predict 240 seconds of strong-view representation learning.
- **Single-seed resolution:** 0.10 points equals ten CIFAR-10 images; a bare pass is protocol-valid but weak causal evidence and cannot be confirmed by reroll.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`: only-`train.py`, one-H20, fixed-time/seed/evaluator, and primary metric rules.
- `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`: step decomposition, accepted exposure, and optimizer-cost ceiling.
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md` and `04-results.tsv`: production-batch safety, generalization failures despite fit, and unchanged 94.15% frontier.
- `goals/maximize-cifar10-best-test-accuracy/experiments/001/02-plan.md` and `04-analysis.md`: exact bundled changes and explicit Nesterov/schedule confound.
- `goals/maximize-cifar10-best-test-accuracy/experiments/010/04-analysis.md`: accepted recipe, trajectory, exposure, NLL, runtime, and evaluation reference.
- `goals/maximize-cifar10-best-test-accuracy/experiments/018/papers/sgd-noise-generalization.md`: noise-scale/schedule interaction and warning that lower train loss need not improve test accuracy.
- Installed PyTorch 2.9.1 `torch.optim.SGD` documentation and `train.py`: exact update semantics, defaults, and accepted implementation.
