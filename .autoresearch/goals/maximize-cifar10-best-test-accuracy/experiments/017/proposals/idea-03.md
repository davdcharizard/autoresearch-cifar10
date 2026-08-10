# Proposal: Isolated PyTorch Nesterov on the Accepted EXP-010 Recipe

## Decision and Exact Scope

Change one optimizer keyword and nothing else:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)
```

Pin the installed PyTorch 2.9.1 defaults: `momentum=0.9`, `dampening=0`, `maximize=False`, `foreach=None`, `fused=None`, and `differentiable=False`. The production diff must contain only `nesterov=True`; it may not add safety logging or change schedule, LR, momentum, decay, optimizer grouping, data, targets, architecture, forward graph, timer, evaluator, or lifecycle.

The single-run hypothesis is that Nesterov's current-gradient correction will select a slightly better solution under the already-validated long high-LR exploration and hard weak tail, raising `best_test_acc` from 94.15% to at least 94.25%, with a point prediction of **94.30%**, while retaining at least 99% of EXP-010's 26,898 updates.

## Updated Historical Case

EXP-001 is not an isolated Nesterov result. It coupled Nesterov to a 15% `lr=0.1` hold and an 85%-budget cosine, reaching 91.57% with an extremely low 0.0215 train-loss EMA. EXP-002 changed both factors: it restored ordinary momentum and introduced the productive 80% high-LR hold plus `0.01` cosine tail, reaching 91.83%. The 0.26-point difference cannot separate optimizer geometry from the dominant schedule change.

The full accepted lineage since EXP-002 uses ordinary momentum. EXP-010 reached 94.15% with width 2, active postactivation branches, p=0.5 CutMix/N1-M7 through 80%, and a hard weak tail. EXP-012 and EXP-015 show that identity-oriented graph/initialization changes suppress strong-phase representation fit even at equal exposure. EXP-016's selected BF16-funded width-3 candidate never reached timing or accuracy: its paired real-batch trajectory triggered candidate-only greater-than-95% one-class concentration. The baseline therefore remains EXP-010, and isolated Nesterov remains untested.

EXP-016 strengthens two design choices here. First, preserve FP32 accepted numerics and the complete graph. Second, fail closed on candidate-only concentration using paired production-distribution batches before consuming a full run. It does not supply evidence that Nesterov will improve accuracy; the candidate's ceiling remains modest and close to the ten-image gate.

## Exact Installed SGD Semantics

For online parameter `theta`, raw loss gradient `grad`, coupled decay `lambda=1e-4`, momentum `mu=0.9`, and current LR `gamma`, installed PyTorch performs:

```text
d_t = grad_t + lambda * theta_(t-1)
b_t = d_t                              on the first step
b_t = mu * b_(t-1) + d_t               later (dampening = 0)
u_t = d_t + mu * b_t                    when nesterov = True
theta_t = theta_(t-1) - gamma * u_t
```

The accepted optimizer instead uses `u_t=b_t`. This is PyTorch's documented Nesterov implementation, not an extra forward at lookahead parameters and not another framework's variant. Forward, loss, and backward still execute once at current online weights.

On step one, both optimizers create `b_1=d_1`. The Nesterov direction is therefore `(1+0.9)d_1=1.9d_1`, while ordinary momentum uses `d_1`. The first Nesterov update at `lr=0.1` is exactly 1.9 times the ordinary direction before FP32 storage rounding. This is required behavior, not a reason to add warmup or lower LR.

Coupled decay enters `d_t` before momentum and appears in both terms of the Nesterov direction. Although the configured `weight_decay` remains exactly `1e-4`, its trajectory-level effect changes. Rescaling it, excluding BN/bias, or converting to decoupled decay would violate the isolated scope and repeat already failed decay exploration.

## Everything Else Preserved

Keep the accepted `train.py` byte-identical outside that keyword:

- width-2 postactivation ResNet-20, raw/Option-A shortcuts, global average readout, and 1,073,962 parameters;
- seed 42, current Kaiming/BN initialization, FP32 forward/backward/SGD/evaluation, and batch 128;
- N1/M7 plus alpha-1 CutMix probability 0.5 through exactly 80% counted progress;
- hard-label weak crop/flip loader after the same eight-worker shutdown and rebuild;
- all-parameter coupled decay `1e-4`, momentum coefficient 0.9, one optimizer group;
- `lr=0.1` through 80%, abrupt step to `0.01`, cosine to `1e-4`, and 300-second timer;
- hard/probability-target cross entropy, loader/collator RNG, checkpoints, one evaluator call per epoch, summary, and ten-minute supervisor.

No warmup, clipping, gradient scaling, LR reduction, momentum tuning, decay compensation, fused/forced-foreach optimizer, EMA, schedule smoothing, extra evaluation, or fallback is allowed. If exact Nesterov fails a gate or a valid run, retire it at this operating point.

## State and RNG Semantics

The keyword adds no module, parameter, buffer, model operation, constructor draw, or optimizer tensor before the first step. Paired control/candidate must begin with bitwise-identical model state and CPU/CUDA RNG state. Optimizer construction must consume no RNG and have empty per-parameter state.

After one identical backward but before either update, losses, logits, raw gradients, BN buffers, and RNG states must be bitwise equal. After step one:

- both optimizers contain one same-shaped FP32 momentum buffer per gradient-bearing parameter;
- each buffer equals the manually computed `grad + 1e-4*pre_step_parameter` within installed FP32 operation tolerance;
- corresponding control/Nesterov momentum buffers are bitwise equal;
- control and candidate parameters match their respective manual update formulas;
- parameter identities, group order/membership, optimizer-state keys, and model buffer values remain aligned;
- optimizer stepping does not advance CPU or CUDA RNG.

From step two onward parameter/gradient trajectories differ by design, but random draws do not: the optimizer uses no randomness, and unchanged loaders/workers/collator consume the same stochastic stream. No production diagnostic may perturb that stream; every preflight runs in a disposable process.

## Functional and Update Gates

Before any production run:

1. Static diff and AST checks prove the only tracked semantic change is literal `nesterov=True`; the single optimizer group retains all parameters and exact accepted options.
2. A hand-computed FP32 recurrence on multiple tensors for at least three steps, including nonzero coupled decay and LR change, matches installed ordinary and Nesterov SGD parameters and momentum buffers.
3. Paired seed-42 accepted models prove exact initial state/logit/loss/gradient/RNG equality for hard `[128]` and probability `[128,10]` targets.
4. Step-one Nesterov direction norm is 1.9 times the ordinary direction for every resolvable nonzero tensor. Because subtracting small FP32 deltas from large parameters can round, assert stored parameters directly against manual FP32 updates rather than relying only on `after-before` ratios.
5. Momentum buffers remain equal after step one and match manual divergent recurrences later; every loss, gradient, parameter, and optimizer-state tensor remains finite.
6. Parameter count, model graph, evaluator reachability, timer placement, CutMix target behavior, and worker switch code remain unchanged.

Any mismatch blocks execution. Fix only a verification or literal implementation error; never compensate for the required 1.9 first update.

## Paired Real-Batch Stability Gate

Use the production distribution, not Gaussian images. In one fresh disposable process, materialize **200 distinct seed-42 N1/M7 batches** through the exact p=0.5 CutMix collator. Require a 45-55% realized mixed-batch fraction. Feed identical stored batches in identical order to paired accepted/Nesterov models cloned from the same initial state, with their separate exact optimizers.

Record first-step manual-update checks, per-step loss, gradient/update norms, finite status, predicted-class histogram, and momentum-buffer norms. Require:

- exact first-step gradient equality, buffer equality, and the manual 1.9 Nesterov direction;
- no non-finite loss, logit, gradient, parameter, or momentum value across 200 steps;
- first-batch replay loss after the Nesterov update no greater than twice its pre-update loss;
- no candidate-only step with more than 95% of the 128 predictions in one class when paired FP32 control is at or below 95% at that same step;
- over steps 101-200, Nesterov loss EMA no more than 1.5x the paired control and no terminal eight-batch candidate-only collapse to fewer than two predicted classes;
- finite global update norms, with the declared step-one 1.9 ratio and no later candidate spike above 10x its preceding 16-step median.

The candidate-only concentration rule carries EXP-016's fail-closed safety standard into this FP32 optimizer test. Persist the failing step and both paired histograms before raising so a veto is auditable. Passing only excludes immediate numerical/optimization collapse. It cannot establish full-phase generalization: EXP-015 passed favorable 64-batch real-data checks and still lost 0.35 points after the complete strong phase.

Any safety failure retires `nesterov=True` at LR 0.1. Do not relax the threshold, rerun for a favorable trajectory, add warmup/clipping, or fall back to a modified optimizer.

## Paired Timing and Exposure Gates

On one idle 97,871 MiB H20, run five alternating fresh-process accepted/Nesterov timing pairs. Reset identical model/optimizer state per trial, alternate deterministic hard/probability targets, perform 100 warmups, then time at least 1,000 complete synchronized steps using the exact production H2D, zero-grad, forward, loss, backward, optimizer, and final synchronization region.

Require:

- candidate/control median-of-trial-means step ratio `<=1.01`;
- CV of trial means `<=2%` for each and candidate p95 no more than 1.04x control;
- `floor(26,898 * control_mean / candidate_mean) >=26,629`, retaining 99% of accepted exposure;
- candidate peak allocation `<610 MiB` and no more than 8 MiB over paired control;
- finite losses/state throughout and conservative end-to-end projection below 540 seconds.

The system decomposition bounds optimizer-only work at a small fraction of the 10.927 ms accepted step, so Nesterov should be near-neutral. Measure rather than infer: its extra `d_t + mu*b_t` foreach operation could still add a launch or temporary tensor list. Inference and evaluator cost are graph-identical; paired initial logits must be bitwise equal and no additional evaluator timing opportunity is introduced.

If a timing or exposure gate fails, do not force `foreach`, enable fused SGD, exclude optimizer time, or combine a speed mechanism. This exact candidate is a no-go.

## Full-Run Hypothesis and Decision

After all semantic, real-batch, timing, exposure, and wall gates pass, run once at seed 42 with required redirection to `run.log`.

**Hypothesis:** isolated PyTorch Nesterov at momentum 0.9 will preserve at least 26,629 updates, keep strong-phase fit above the established underfit region, and reach **94.30%** point-estimate best test accuracy, clearing the formal **94.25%** threshold without any schedule or representation change.

Require exit zero, ten unique finite summary values, 300.0 counted seconds, total under 600 seconds, exactly 1,073,962 parameters, peak memory within preflight, at least 26,629 steps, one switch near 80%, all eight old workers stopped, 45-55% realized strong CutMix, hard weak targets, and unique evaluation epochs.

Compare against EXP-010's 89.73% switch checkpoint, 93.16% first weak checkpoint, 94.15% final/best, 0.1934 NLL, 26,898 steps, and 330.7-second total. Inspect the first weak epoch and loss immediately after the `0.1 -> 0.01` transition because the high-LR momentum buffer persists while Nesterov adds the new current weak gradient. A switch below 87.08 or transition instability is diagnostic only; it cannot trigger early stop or adjustment.

- **Improvement:** `best_test_acc >=94.25%` and all integrity gates pass.
- **Valid lower result:** no-improvement, no reroll, no momentum/LR retuning, and return to EXP-010.
- **Accuracy pass below exposure floor:** formally clears the metric but has timing-confounded attribution; do not call it compute-neutral.
- **Crash, timeout, RNG/state/target/timer/evaluator/lifecycle fault:** invalid; repair only protocol defects while preserving the one-keyword candidate.

## Ceiling and Risks

- **Modest ceiling:** Nesterov adds no data, capacity, invariance, or extra budget. A plausible successful range is only 94.25-94.40%; a multi-point gain has no local basis.
- **First-step overshoot:** the exact first update is 1.9x at the accepted `lr=0.1`, which can destabilize initial class geometry.
- **Noisy lookahead:** N1/M7 and CutMix gradients are intentionally noisy; adding the current direction to accumulated momentum can amplify oscillation rather than accelerate.
- **Coupled-decay amplification:** decay enters current and momentum terms, altering effective regularization despite the same scalar.
- **Tail transition shock:** Nesterov carries the plateau buffer through the abrupt tenfold LR step and simultaneously emphasizes the first weak hard-label gradients.
- **Accepted optimizer strength:** ordinary momentum has survived every accepted improvement and EXP-010 ends at its best, leaving little obvious optimizer headroom.
- **Historical ambiguity:** EXP-001 is weak negative evidence only; its early anneal, smaller model, and simpler data prevent attribution.
- **Safety is not accuracy:** 200 stable real batches cannot predict the 240-second strong-view representation, as EXP-015 demonstrates.
- **Single-seed resolution:** the required 0.10 point is ten test images. A bare pass is valid for this protocol but weak causal evidence and cannot be confirmed by reroll.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`: scope, one-H20, fixed-time, fixed-seed, evaluation, and primary threshold rules.
- `goals/maximize-cifar10-best-test-accuracy/02-system-understanding.md`: measured 10.927 ms step decomposition and low optimizer-only cost ceiling.
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md` and `04-results.tsv`: accepted recipe, real-batch safety requirement, recurring strong-phase suppression, and unchanged 94.15% frontier.
- `goals/maximize-cifar10-best-test-accuracy/experiments/001/04-analysis.md` and `002/04-analysis.md`: confounded Nesterov/hold comparison and validated long-hold ordinary momentum.
- `goals/maximize-cifar10-best-test-accuracy/experiments/010/04-analysis.md`: accepted CutMix recipe, switch/tail trajectory, exposure, NLL, and runtime reference.
- `goals/maximize-cifar10-best-test-accuracy/experiments/015/04-analysis.md`: favorable short real-batch behavior did not predict full strong-phase accuracy.
- `goals/maximize-cifar10-best-test-accuracy/experiments/016/04-analysis.md`: candidate-only greater-than-95% real-batch concentration vetoed BF16 width 3 before timing/production.
- `goals/maximize-cifar10-best-test-accuracy/experiments/016/proposals/idea-02.md`: prior unexecuted Nesterov specification carried forward and tightened here.
- `goals/maximize-cifar10-best-test-accuracy/experiments/017/01-brainstorm.md`: isolated Nesterov candidate framing and modest expected ceiling.
- Installed PyTorch 2.9.1 `torch.optim.SGD` semantics and `train.py`: exact optimizer recurrence, defaults, accepted graph, data, RNG, schedule, timer, and evaluator.
