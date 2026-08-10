# Proposal: Isolated PyTorch Nesterov Momentum on EXP-010

## Decision

Change exactly one optimizer argument on the complete accepted EXP-010 recipe:

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)
```

Pin installed PyTorch 2.9.1 semantics and leave `momentum=0.9`, `dampening=0`, `maximize=False`, `foreach=None`, `fused=None`, and `differentiable=False`. Do not change the 80% LR hold, `0.1 -> 0.01` transition, terminal cosine, augmentation phases, decay scalar, or any other lever.

The one-run hypothesis is that Nesterov's gradient-informed momentum direction will reach a slightly better basin under the validated long exploration/tail schedule and improve `best_test_acc` from 94.15% to at least 94.25%, with a point prediction of **94.30%**, while retaining at least 99% of EXP-010's 26,898 updates.

## Why EXP-001 Did Not Test This Question

EXP-001 bundled Nesterov with a radically different schedule: only 15% at `lr=0.1`, then an 85%-budget cosine decay. It reached 91.57%, 0.10 below the original 91.67% baseline, despite monotonically fitting to a very low 0.0215 train-loss EMA. EXP-002 simultaneously restored ordinary momentum **and** changed to the successful 80% `lr=0.1` plateau followed by a `0.01` cosine tail; it reached 91.83%.

The 0.26-point EXP-001-to-002 difference therefore cannot identify Nesterov. The dominant local inference is that EXP-001 annealed too early and lost useful high-LR exploration/implicit regularization. The accepted lineage since EXP-002 has never isolated the optimizer flag. EXP-015 now recommends precisely this deconfounded test after another identity-oriented representation change preserved compute but suppressed strong-phase fit.

This proposal retains every validated schedule, architecture, data, regularization, and evaluation decision. It tests the unresolved optimizer component rather than retrying the failed EXP-001 bundle.

## Exact PyTorch Update Semantics

For parameter `theta`, raw loss gradient `grad`, coupled weight decay `lambda=1e-4`, momentum `mu=0.9`, and current LR `gamma`, PyTorch performs:

```text
d_t = grad_t + lambda * theta_(t-1)
b_t = d_t                              if no momentum buffer exists
b_t = mu * b_(t-1) + d_t               otherwise (dampening = 0)
u_t = d_t + mu * b_t                    Nesterov direction
theta_t = theta_(t-1) - gamma * u_t
```

Accepted ordinary momentum instead uses `u_t = b_t`. This is PyTorch's documented formulation, not a hand-written lookahead-gradient evaluation and not necessarily another framework's NAG convention. The model forward/backward still occurs at current online parameters exactly once.

The first-step consequence is material: both optimizers initialize `b_1=d_1`, but Nesterov uses `u_1=(1+mu)d_1=1.9*d_1`. With identical gradients, every nonzero first parameter delta must therefore be exactly 1.9 times the control delta within FP32 tolerance. On later steps, Nesterov adds the current decay-augmented gradient to the momentum lookahead direction.

Coupled decay is not held out of this transformation. `lambda*theta` enters `d_t` before momentum, so Nesterov changes the effective decay trajectory even though the configured `weight_decay=1e-4` is identical. Creating a decoupled decay group or rescaling lambda would be a second optimizer experiment and is forbidden.

## Preserved EXP-010 Recipe

Keep byte-identical outside the single optimizer flag:

- width-2 postactivation ResNet-20, Option-A shortcuts, global average pool, 1,073,962 parameters, and current initialization;
- seed 42, FP32 model/training/evaluation, batch 128, hard/probability-target cross entropy;
- N1/M7 and p=0.5 alpha-1 CutMix through exactly 80%, then hard-label weak crop/flip data;
- all-parameter coupled decay `1e-4` and momentum coefficient 0.9;
- `lr=0.1` through 80%, the step to `0.01`, and elapsed-time cosine to `1e-4`;
- loaders, CutMix RNG fork, persistent workers, switch/shutdown lifecycle, counted timer, synchronization, evaluator cadence, and summary.

Do not add warmup, lower the initial LR, alter the LR drop, change momentum, clip gradients, change decay semantics, enable fused SGD, combine EMA, or add an adaptive safety intervention. A valid but poor trajectory must finish unchanged.

## State and RNG Identity

The optimizer flag adds no model parameters, tensors at construction, RNG draws, model operations, gradients, or evaluator work. Candidate and control must begin with bitwise-identical model parameters/buffers and CPU/CUDA RNG states. Optimizer construction must consume no RNG and create no momentum buffers before the first step.

After the first identical backward:

- raw parameter gradients and loss must be bitwise equal;
- each optimizer must create one momentum buffer for every parameter with a gradient;
- corresponding control/Nesterov first momentum buffers must be bitwise equal to the same coupled-decay direction;
- parameter objects, optimizer membership, buffer shapes/dtypes/devices, and parameter count remain identical;
- only updated parameter values and the param-group `nesterov` flag may differ.

Optimizer steps are deterministic tensor arithmetic and must not alter CPU/CUDA RNG states. Because the data, model constructors, and loader are unchanged, shuffle, worker, transform, and CutMix streams remain seed-aligned until the differing parameter trajectory affects only numerical loss/gradients, not random draws.

## Expected Cost and Ceiling

Forward, loss, backward, H2D, loader, and evaluation are identical. Nesterov adds the `d_t + mu*b_t` combination during SGD update. The system decomposition places reset plus SGD at only 0.182 ms of a 10.927 ms accepted step, with the optimizer itself below roughly 0.1 ms. Even doubling optimizer arithmetic would leave a low single-percent ceiling, and PyTorch's default CUDA foreach path should batch the extra operation across tensors.

Expected parameter, persistent optimizer-state, and model-activation memory are unchanged. Temporary foreach intermediates may differ slightly; the H20 has enormous headroom, but measured peak must remain near EXP-010's 598.7 MiB.

Accuracy upside is likely modest. Nesterov adds no data, capacity, invariance, or training time. It can alter transient response and basin selection, especially during the long high-LR phase and abrupt 80% LR step, but the accepted standard-momentum recipe is already well tuned and ends at its best. A plausible successful range is 94.25-94.40%; a multi-point gain has no local support. The 0.10 gate is still meaningful but close to single-seed resolution.

## Functional Semantics Gates

Before H20 work, use disposable CPU/CUDA tests to require:

1. Static scope shows only `nesterov=True` in the optimizer call; one parameter group still contains all 1,073,962 parameters with exact LR, momentum, decay, dampening, foreach, fused, maximize, and differentiable settings.
2. A hand-computed two-parameter, three-step recurrence including nonzero coupled decay matches installed `torch.optim.SGD` control and Nesterov updates and momentum buffers within FP32 operation-order tolerance.
3. Paired seed-42 accepted models have bitwise-identical initial state, logits, hard/soft losses, raw gradients, and pre-step CPU/CUDA RNG states.
4. On step one, corresponding momentum buffers are bitwise equal and nonzero parameter-delta ratios are 1.9 within tight FP32 tolerance. Zero-gradient/zero-parameter edge cases are excluded from the ratio denominator rather than treated as failures.
5. After multiple steps, manual recurrence and optimizer state still agree; every tensor remains finite, and optimizer construction/steps consume no RNG.
6. Hard `[128]` and CutMix probability `[128,10]` targets both execute correctly. Evaluator reachability, worker lifecycle, timer, and summary remain unchanged.

Any semantic, state, RNG, target, or scope failure blocks execution. Fix only the test/implementation defect; do not alter Nesterov's operating point.

## Production-Distribution Safety Gate

EXP-015 established that Gaussian-image safety probes can falsely predict class collapse, so all optimization-collapse checks must use materialized CIFAR-10 batches from the actual strong pipeline. In a disposable fresh process, create 64 seed-42 N1/M7 batches through the production collator, preserving its realized hard/soft target mix. Materialize each CPU batch once, then feed identical copies in identical order to paired accepted and Nesterov models initialized from the same state.

Require:

- first-step coupled gradient equality and the predicted 1.9 delta ratio;
- finite logits, losses, gradients, parameters, and momentum buffers for all 64 updates;
- after step one, same-batch replay loss no greater than 2x the pre-update loss and no class receiving more than 80% of predictions;
- over updates 33-64, Nesterov loss EMA no greater than 1.25x control, terminal top-class concentration no greater than 60%, and all ten classes represented across the final eight batches;
- maximum parameter/update norm finite and no single step exceeding 10x the paired control global update norm after step one.

These deliberately broad bounds detect EXP-014-style immediate optimizer collapse without selecting for short-run superiority. Passing does **not** predict final generalization: EXP-015's candidate looked better over 64 real batches yet underfit the full strong phase. Do not use short loss, concentration, or update norms to tune LR/momentum or choose a rescue. A gate failure retires exact Nesterov at this LR; it does not authorize warmup or clipping.

## Paired H20 Timing Gate

On one idle 97,871 MiB H20, run five alternating fresh-process accepted/Nesterov pairs with cloned model state, identical fresh optimizer state, batch 128, and deterministic alternating hard/probability targets. Use 100 warmups and at least 1,000 synchronized complete training steps per trial, measuring the exact production `t0` through optimizer step and final synchronization.

Require:

- candidate/control median-of-trial-means step ratio `<=1.01`;
- trial-mean CV `<=2%` for each and candidate p95 no more than 1.04x control;
- projected exposure `floor(26,898 * control_time / candidate_time) >=26,629`, retaining 99% of EXP-010;
- peak allocated memory `<610 MiB` and no more than 8 MiB above paired control;
- finite hard/soft losses and optimizer state;
- conservative total-runtime projection below 540 seconds.

Inference is architecture-identical and must produce bitwise-equal pre-training logits; no candidate inference path is added. A timing miss rejects this exact optimizer flag. Do not force `foreach`, enable fused SGD, or make optimizer time uncounted as a fallback.

## Full-Run Hypothesis and Verification

After all gates pass, run exactly once at seed 42 with all output redirected to `run.log`.

**Hypothesis:** PyTorch Nesterov at momentum 0.9 will preserve at least 26,629 updates, maintain a healthy strong-view representation, and reach **94.30%** point-estimate best test accuracy, clearing the formal **94.25%** threshold on the otherwise unchanged EXP-010 protocol.

Require exit zero, ten unique finite summary fields, 300.0 counted seconds, total below 600 seconds, 1,073,962 parameters, peak VRAM within the timing gate, at least 26,629 steps, one 80% strong-to-weak switch, eight stopped workers, approximately 50% strong CutMix, no soft weak target, and unique evaluation epochs.

Compare with EXP-010's 89.73% switch accuracy, 93.16% first weak accuracy, 94.15% final/best, 0.1934 NLL, 26,898 steps, and 330.7-second total. Record loss trajectory and behavior immediately after the `0.1 -> 0.01` transition because Nesterov carries a high-LR momentum buffer into the tail. A switch below the established 87.08 underfit marker or unstable transition is diagnostic only and cannot trigger early stopping or rescue tuning.

- **Improvement:** `best_test_acc >=94.25%` with every integrity gate passing.
- **Valid lower result:** no-improvement; retain EXP-010 with no reroll or optimizer retuning.
- **Accuracy pass below 26,629 steps:** formally above the metric gate but timing-confounded; do not claim compute identity.
- **Crash, timeout, state/RNG/target/timer/evaluator/lifecycle fault:** invalid. Fix only protocol defects while preserving the single `nesterov=True` change.

## Risks and Failure Mechanisms

- **First-step overshoot:** PyTorch's first Nesterov delta is 1.9x ordinary momentum at `lr=0.1`; the validated LR may be too aggressive without warmup.
- **High-LR oscillation:** current-gradient lookahead can amplify noisy N1/M7/CutMix directions during the long plateau rather than improve basin exploration.
- **Effective decay shift:** coupled `1e-4` decay is included in both current gradient and momentum direction, so identical configuration does not mean identical regularization dynamics.
- **80% transition shock:** the momentum buffer accumulated at `lr=0.1` persists across the abrupt step to `0.01`; Nesterov also adds the current weak hard-label gradient and may adapt faster or overshoot differently.
- **Already-strong baseline:** standard momentum plus long exploration is validated through many experiments; optimizer geometry alone may have less than 0.10-point headroom.
- **Confounded historical evidence:** EXP-001 is weak negative evidence only because its schedule, evaluation cadence, and width/data recipe differ substantially.
- **Short-probe non-predictiveness:** safety over 64 batches can exclude collapse but cannot guarantee full-phase representation quality, as EXP-015 demonstrated.
- **Single-seed resolution:** the threshold is ten test images. A bare pass is protocol-valid but weak causal evidence and cannot be confirmed by reroll.

## Evidence

- `goals/maximize-cifar10-best-test-accuracy/01-definition.md`: only-`train.py`, one-H20, fixed-time, fixed-seed, evaluation, and metric rules.
- `goals/maximize-cifar10-best-test-accuracy/experiments/001/04-analysis.md`: Nesterov confounded with a 15% hold and early cosine, producing 91.57% despite low train loss.
- `goals/maximize-cifar10-best-test-accuracy/experiments/002/04-analysis.md`: ordinary momentum plus the accepted 80% hold/tail reached 91.83%, without isolating the optimizer flag.
- `goals/maximize-cifar10-best-test-accuracy/experiments/015/04-analysis.md`: compute-neutral identity initialization failed on full strong-phase fit and motivates isolated Nesterov next.
- `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md` and `04-results.tsv`: accepted recipe, underfit markers, and production-distribution safety-probe requirement.
- `goals/maximize-cifar10-best-test-accuracy/knowledge/papers/sgdr.md`: CIFAR evidence for retaining the validated elapsed-time cosine schedule while changing no schedule component.
- Installed PyTorch 2.9.1 `torch.optim.SGD` documentation: exact momentum initialization, coupled decay order, Nesterov direction, and default foreach/fused semantics.
- `train.py`: accepted architecture, optimizer, LR/phase schedule, RNG, timing, loader lifecycle, and evaluator.
