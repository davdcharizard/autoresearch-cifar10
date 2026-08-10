# Proposal: Isolated PyTorch Nesterov Momentum

## Decision

Change exactly one optimizer keyword on the accepted EXP-010 source:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)
```

Pin installed PyTorch 2.9.1 behavior and leave `momentum=0.9`, `dampening=0`, `maximize=False`, `foreach=None`, `fused=None`, and `differentiable=False`. The production `train.py` diff may contain only the literal `nesterov=True`; no diagnostic code, warmup, clipping, LR/decay adjustment, optimizer backend selection, schedule, data, graph, precision, timer, or evaluator change is allowed.

The one-run hypothesis is that the current-gradient correction changes the noisy long-plateau optimization path enough to find a slightly better generalizing solution, raising `best_test_acc` from 94.15% to at least 94.25%, with a point prediction of **94.30%**, while retaining at least 99% of EXP-010's 26,898 updates.

## Local History and Isolation

EXP-001 bundled Nesterov with a 15% `lr=0.1` hold, an 85%-budget cosine, persistent workers, and a new evaluation cadence. It reached 91.57% with very low train loss. EXP-002 simultaneously removed Nesterov and restored the successful 80% plateau plus `0.01` cosine tail, reaching 91.83%. Their 0.26-point difference cannot separate Nesterov from the dominant schedule change; EXP-001's own report names Nesterov as a confound.

EXP-010 remains the 94.15% frontier. Subsequent graph, initialization, precision/capacity, shortcut, and averaging candidates did not improve it. Most recently, EXP-018's uniform late SWA had eight nondegenerate snapshots and extensive BN recalibration but pulled the final solution backward to 93.85%; its online best was only 94.02%. This rejects that averaging window, not online optimizer geometry, and explicitly leaves isolated Nesterov as the clean remaining optimizer confound.

This experiment retains the complete accepted width/data/schedule path and changes one update rule. It does not retry EXP-001 and does not combine Nesterov with EMA/SWA.

## Exact Installed Recurrence

For parameter `theta`, raw minibatch gradient `grad`, coupled decay `lambda=1e-4`, momentum `mu=0.9`, and scheduled LR `gamma`, PyTorch computes:

```text
d_t = grad_t + lambda * theta_(t-1)
b_t = d_t                              when no buffer exists
b_t = mu * b_(t-1) + d_t               later (dampening = 0)
u_t = d_t + mu * b_t                    with nesterov = True
theta_t = theta_(t-1) - gamma * u_t
```

Accepted ordinary momentum uses `u_t=b_t`. There is no second forward or gradient evaluation at lookahead parameters. This is PyTorch's recurrence, not a generic textbook NAG substitution.

On the first step both optimizers store the same `b_1=d_1`. Nesterov then uses `(1+mu)d_1=1.9d_1`, so its first direction is exactly 1.9 times ordinary momentum's before FP32 parameter-storage rounding. Initial logits, loss, raw gradients, and momentum buffers must be equal; initial parameter updates must **not** be equal. Matching them would require an unreviewed LR/warmup/buffer change and erase the intervention.

Weight decay is coupled before momentum. It contributes to both the current and historical terms, so `nesterov=True` changes effective regularization dynamics even though the configured scalar remains exactly `1e-4`. Do not compensate decay or create parameter groups.

## Mechanism and Evidence Limits

The momentum-generalization paper gives a mechanism by which historical gradients can preserve shared feature signal instead of memorizing low-margin examples. The accepted optimizer already has ordinary momentum 0.9, so this is not evidence that Nesterov is better. Nesterov gives additional weight to the current minibatch direction; it could react faster along consistent features, but it could also weaken the historical-gradient filtering that the paper associates with generalization under heterogeneous hard and CutMix margins.

The nonconvex acceleration paper proves faster saddle escape for a specific Nesterov-style deterministic method. It is not installed PyTorch stochastic SGD, and second-order stationarity is not CIFAR-10 test accuracy. It supports only the narrow plausibility that a current-gradient correction can change nonconvex exploration, not the predicted effect size or sign here.

Locally, the candidate's merit is attribution and low compute, not strong positive evidence. Lower short-run loss or higher switch accuracy is insufficient: EXP-001 overfit, EXP-017 fit the strong/first-weak checkpoints better while losing final accuracy, and EXP-018's averaged model worsened NLL. Only fixed-evaluator top-1 can validate the mechanism.

## Accepted Recipe Preserved

Keep byte-identical outside the keyword:

- width-2 postactivation ResNet-20, active residual branches, Option-A shortcuts, global average readout, and 1,073,962 parameters;
- seed 42, current initialization, FP32 model/training/evaluation, and batch 128;
- N1/M7 plus alpha-1 CutMix probability 0.5 through exactly 80% counted progress;
- hard-label weak crop/flip data after the same eight-worker shutdown/rebuild;
- one all-parameter optimizer group, momentum 0.9, and coupled decay `1e-4`;
- `lr=0.1` through 80%, abrupt step to `0.01`, cosine to `1e-4`, and 300 counted seconds;
- loader/collator RNG handling, hard/probability CE, checkpoints, one evaluator call per epoch, terminal evaluation, and summary.

No warmup, clipping, gradient scaling, LR reduction, momentum tuning, decay compensation, forced foreach/fused path, extra evaluation, or fallback is permitted. A gate failure or valid no-improvement retires this exact point.

## Semantic, State, and RNG Gates

Before trajectory work:

1. Static diff/AST checks prove the only tracked semantic change is `nesterov=True`; all optimizer defaults and the single group are exact.
2. A manual FP32 recurrence over at least three steps, including nonzero coupled decay and an LR change, matches installed control/Nesterov parameters and buffers.
3. Reset seed-42 models have bitwise-identical initial parameters, buffers, hard/soft logits, losses, raw gradients, and CPU/CUDA RNG states. Optimizer state is initially empty.
4. After step one, corresponding momentum buffers are bitwise equal and match `grad + 1e-4*theta_0`; stored parameters match their respective manual FP32 formulas; resolvable direction norms have the required 1.9 ratio.
5. Later divergent states continue matching manual recurrence and remain finite with identical state keys/shapes/dtypes/devices and unchanged parameter identities/group membership.
6. Optimizer construction/stepping consumes no RNG. Model graph/count, target handling, timer, evaluator reachability, and worker code remain unchanged.

Correct only controller or literal implementation defects. Do not normalize away the first-step difference.

## Real-Batch Strong and Transition Safety

Use real production data, since Gaussian probes have produced misleading collapse signals. In one disposable process, materialize **200 distinct seed-42 N1/M7 batches** through the production p=0.5 CutMix collator, requiring 45-55% mixed targets. Then shut down that loader exactly as production does, materialize **64 weak crop/flip hard-label batches**, and feed identical stored batches in identical order to paired control/Nesterov models.

Train both pairs for the 200 strong batches at `lr=0.1`. Without clearing momentum, set both to `lr=0.01` and continue for the 64 weak batches. This is not a miniature accuracy proxy; it deliberately exercises hard/soft loss paths and the tenfold LR plus target/augmentation transition with live momentum state.

Require:

- exact first-step gradient/buffer/manual 1.9-direction checks;
- finite logits, losses, gradients, parameters, and momentum buffers for all 264 steps;
- first-batch Nesterov replay loss no greater than 2x its pre-update loss;
- no candidate-only step above 95% one-class predictions when paired control is at or below 95%; persist both histograms before vetoing;
- no post-step-one Nesterov update-norm spike above 10x its preceding 16-step median;
- across strong steps 101-200, Nesterov loss EMA no more than 1.5x control;
- across all 64 weak steps, no candidate-only non-finite/concentration event and no first-eight-weak loss EMA above 2x paired control.

The transition check addresses a specific risk: the momentum buffer accumulated under noisy strong views survives the `0.1 -> 0.01` LR step, while Nesterov additionally injects the first current weak hard-label gradient. It may accelerate adaptation or create a directional shock. Passing only rules out gross instability; it does not predict the full 240-second strong representation or terminal generalization.

Any failure is a no-go. Do not relax thresholds, reroll materialized batches, reset momentum at 80%, warm up, clip, or lower LR.

## Paired Timing and Exposure Gates

On one idle 97,871 MiB H20, run five alternating fresh-process accepted/Nesterov pairs from reset identical model/optimizer state. Use batch 128, deterministic alternating hard/probability targets, 100 warmups, and at least 1,000 synchronized complete production-timed steps per trial.

Require:

- candidate/control median-of-trial-means ratio `<=1.01`;
- trial-mean CV `<=2%` each and candidate p95 no more than 1.04x control;
- `floor(26,898 * control_mean / candidate_mean) >=26,629`, retaining 99% exposure;
- peak allocation `<610 MiB` and no more than 8 MiB above paired control;
- finite loss/state and conservative total-runtime projection below 540 seconds.

The measured optimizer/reset stage is only 1.67% of the 10.927 ms step, so the extra Nesterov arithmetic should be near-neutral, but it may add foreach work or temporaries. Measure it. A miss cannot be rescued by forcing an optimizer backend or excluding update time.

## Evaluator and Seed Fairness

The production run uses seed 42 once, with no reroll. The optimizer consumes no randomness, so the initial model and stochastic data policy are unchanged; numerical divergence begins at the declared first update.

The existing evaluator call, early checkpoint tuple, dense weak-tail cadence, and terminal evaluation remain untouched. No paired control evaluation, extra checkpoint, hidden online metric, or changed selection is allowed. Evaluation stays at most once per epoch. `best_test_acc` receives the same scheduled opportunities as EXP-010, apart from ordinary epoch-count variation caused by measured optimizer cost.

## One-Run Hypothesis and Gates

**Hypothesis:** isolated PyTorch Nesterov at momentum 0.9 will preserve at least 26,629 updates, maintain strong-view fit outside the established underfit region, and reach **94.30%** point-estimate best test accuracy, clearing **94.25%**.

Require exit zero, ten unique finite summary fields, 300.0 counted seconds, total below 600 seconds, exactly 1,073,962 parameters, memory within preflight, at least 26,629 steps, one 80% switch, eight stopped workers, 45-55% strong CutMix, hard weak targets, and unique evaluation epochs.

Compare with EXP-010's 89.73% switch, 93.16% first weak, 94.15% final/best, 0.1934 NLL, 26,898 steps, and 330.7-second total. Inspect the first weak epoch and loss transition for momentum shock. A switch below 87.08 or better train loss is diagnostic only and cannot alter the run or verdict.

- **Improvement:** `best_test_acc >=94.25%` with every integrity gate passing.
- **Valid lower result:** no-improvement; no reroll, retuning, rescue, or fallback.
- **Accuracy pass below exposure floor:** formally above the metric gate but timing-confounded; do not call Nesterov compute-neutral.
- **Crash, timeout, RNG/state/target/timer/evaluator/lifecycle fault:** invalid; fix only protocol faults with `nesterov=True` unchanged.

## Predicted Benefit and Failure Modes

The plausible successful range is **94.25-94.40%**. Nesterov adds no data, capacity, or invariance, and the papers do not establish a CIFAR gain over ordinary momentum. Its ceiling is a small online basin/generalization shift.

- **First-step overshoot:** the required 1.9x direction at LR 0.1 may destabilize initial logits.
- **Noise/generalization tradeoff:** extra current-gradient weight can amplify CutMix/RandAugment noise or weaken useful historical shared-feature filtering.
- **Coupled-decay shift:** decay appears in current and momentum terms, altering effective regularization.
- **80% transition shock:** plateau momentum and first weak current gradient are combined immediately despite the tenfold LR drop.
- **Faster fit without better test accuracy:** local history repeatedly shows optimization diagnostics diverging from final generalization.
- **SWA lesson:** EXP-018 shows the online tail is already delicate; changing its approach path may not improve calibration even if convergence is faster.
- **Evidence mismatch:** theoretical momentum generalization and accelerated saddle escape do not analyze this exact stochastic recurrence or metric.
- **Single-seed resolution:** 0.10 points is ten test images; a bare pass is valid but weak causal evidence and cannot be confirmed by reroll.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`, `02-system-understanding.md`, `03-experiment-learnings.md`, and `04-results.tsv`.
- `goals/maximize-cifar10-best-test-accuracy/experiments/001/04-analysis.md` and `002/04-analysis.md`: unresolved schedule/Nesterov confound.
- `goals/maximize-cifar10-best-test-accuracy/experiments/010/04-analysis.md`: accepted model/data/schedule, trajectory, exposure, NLL, and runtime.
- `goals/maximize-cifar10-best-test-accuracy/experiments/018/04-analysis.md`: uniform annealed-tail SWA worsened its own online accuracy and NLL.
- `goals/maximize-cifar10-best-test-accuracy/experiments/019/papers/momentum-generalization.md`: historical-gradient shared-feature mechanism, with no direct Nesterov-over-momentum result.
- `goals/maximize-cifar10-best-test-accuracy/experiments/019/papers/nesterov-nonconvex.md`: limited plausibility for altered nonconvex exploration, not PyTorch SGD generalization.
- Installed PyTorch 2.9.1 `torch.optim.SGD` documentation, EXP-018 prior proposal, and current `train.py`: exact recurrence/defaults and one-keyword implementation.
