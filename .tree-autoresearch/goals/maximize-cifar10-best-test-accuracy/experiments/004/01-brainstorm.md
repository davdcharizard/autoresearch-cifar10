# Brainstorm EXP-004
**Created**: 2026-08-05

## Web Search & Literature Review

- **Averaging Weights Leads to Wider Optima and Better Generalization** (`experiments/004/papers/stochastic-weight-averaging.md`)
  SWA improves residual-network generalization on CIFAR by averaging later SGD trajectory points, with little extra computation; learning-rate behavior and BatchNorm state are coupled implementation concerns.
- **When Does Label Smoothing Help?** (`experiments/004/papers/when-label-smoothing-helps.md`)
  Mild soft targets can improve generalization and calibration through tighter class representations, but their interaction with existing soft-label methods should not be assumed additive.
- **Sharpness-Aware Minimization for Efficiently Improving Generalization** (`experiments/004/papers/sharpness-aware-minimization.md`)
  SAM improves CIFAR generalization by optimizing neighborhood sharpness, but its second sequential gradient calculation makes full-run SAM a poor fit for a fixed wall-clock budget.
- **Goal knowledge: phase-dependent and mixed-sample regularization** (`knowledge/papers/time-matters-regularization.md`, `knowledge/papers/mixed-sample-analysis.md`)
  Early regularization can be removed late without losing its benefit, and Mixup/CutMix exert distinct input-gradient effects that leave hybrid mechanisms plausible.

## Experimental History Review

- BASE reached 91.51%. EXP-001 replaced the baseline with a time-scheduled BF16 PreAct WRN-16-4 and improved to 94.62%, demonstrating that architecture and charged-time scheduling were the dominant first-order levers.
- EXP-002 added front-loaded CutMix on isolated RNG streams and improved to the accepted global best of 95.23%, with 0.2044 final loss, 27,950 steps, 2.75M parameters, and only 1,178.9 MiB peak allocation.
- EXP-003 swept CutMix probability and drop path. Search leaders at 95.48% and 95.42% confirmed at only 95.19% and 95.28%, so narrow scalar tuning is exhausted locally and selected gains below roughly 0.15-0.30 points are vulnerable to measurement variance.
- The current model's late training loss approaches zero while memory and timeout headroom remain large. The limiting gap is stable generalization from the learned trajectory, not inability to fit, accelerator memory, or outer runtime.
- Untried gaps include parameter-trajectory averaging, output-space regularization on clean examples, optimization-space flatness methods, alternative mixed-sample mechanisms, and low-cost representation changes.

## Collected Ideas

- **Sparse late state EMA** — Maintain a sparsely updated EMA model only after the 75% clean-tail transition, averaging parameters and floating BatchNorm buffers without a recalibration pass. This attacks trajectory variance with negligible compute and uses the large memory headroom.
- **Clean-only two-stage label smoothing** — Apply epsilon 0.05 smoothing only to non-CutMix batches before 75% progress, then restore hard targets for all late batches. This fills the early clean-batch regularization gap without double-softening CutMix targets.
- **Clean-finish periodic SAM** — Use plain SAM on one of every four steps only after 75% progress. It targets local sharpness through a qualitatively different optimizer mechanism while bounding the extra forward/backward cost.
- **CutMix/Mixup dual policy** — Split early mixed batches between CutMix and Mixup while preserving the same overall mixing probability and clean tail. The mixed-sample analysis predicts distinct input-gradient regularization, so diversity may beat more CutMix strength.
- **Late SWA with a shallow cyclic tail** — Replace the final cosine segment with a small constant or one-cycle LR and uniformly average tail checkpoints. This follows classical SWA more faithfully than EMA but risks disrupting the validated time schedule and requires careful BatchNorm handling.
- **Squeeze-and-excitation in late stages** — Add low-cost channel attention to the 128- and 256-channel residual stages. This uses abundant memory to improve representation selectivity with far less compute than simply widening the whole network.
- **Anti-aliased stage transitions** — Replace stride-2 convolutions with stride-1 convolution plus fixed low-pass downsampling. This targets shift sensitivity and aliasing rather than confidence or stochastic regularization, but adds bandwidth and may reduce fixed-time step exposure.
- **ShakeDrop replacement** — Replace binary drop path with expectation-controlled residual-branch scaling. The stored ShakeDrop evidence supports richer residual disturbances, but the shallow six-block WRN and existing CutMix make stability uncertain.
- **Lookahead slow weights** — Add a low-frequency interpolation from fast SGD weights to a slow copy. This simplifies averaging to periodic synchronization and avoids a second evaluation model, though resetting the online trajectory could interfere with momentum and late convergence.
- **Efficient block redesign moonshot** — Reallocate the WRN toward bottleneck or depthwise-separable blocks to spend H20 headroom on more depth or width per charged second. Upside is large, but it abandons the validated architecture and creates substantial optimization risk in one loop.

## Combinations

- **Sparse EMA + clean-only label smoothing**: label smoothing could stabilize early clean gradients while EMA reduces late trajectory variance, attacking different stages. The combination may exceed either alone, but bundling two unvalidated effects would weaken attribution and should follow a successful single-mechanism result.
- **Sparse EMA + late channel attention**: attention may learn richer representations while EMA stabilizes their final trajectory. This could combine capacity and generalization benefits, but should be staged because EMA's inference-state handling is already nontrivial.
- **Periodic SAM + sparse EMA**: SAM could generate flatter late iterates for EMA to average, closer to a flat-basin ensemble than either method alone. The combined bookkeeping and charged overhead make it unsuitable before each mechanism is independently validated.
- **CutMix/Mixup dual policy + clean tail**: preserving the existing 75% cutoff keeps late fitting identical while diversifying early input-gradient regularization. This is stronger than merely raising CutMix probability, but interaction parameters would add selection pressure.

## Candidate Ideas

### Sparse Late State EMA
**Summary**: Start a complete-state EMA at 75% charged progress, when CutMix stops and drop path begins decaying. Update once every 32 optimizer steps with an arithmetic-average ramp capped at decay 0.99; average model parameters and floating BatchNorm buffers, copy integer buffers, and evaluate only the EMA model after activation. Full design: `proposals/idea-01.md`.

**What it targets**: The parent fits the training data and has stable late checkpoints, while EXP-003 exposed 0.14-0.29-point selected-run instability. EMA directly reduces late parameter-trajectory variance without changing training examples or adding another network pass.

**Reasoning**: SWA improves CIFAR residual networks through weight-space averaging with low compute. A sparse, clean-tail EMA adapts that mechanism to the existing cosine schedule and avoids an uncharged BatchNorm recalibration pass. The extra FP32 state is only about 11-15 MiB and expected charged overhead is below 0.5%.

**Sources**: `papers/stochastic-weight-averaging.md`; `proposals/idea-01.md`; EXP-002 and EXP-003 analyses.

**Estimated Effort**: medium

**Risk Assessment**: Averaged BatchNorm moments do not exactly match the averaged weights, the low-LR tail may lack trajectory diversity, and EMA lag may erase gains. Evaluating EMA rather than online weights after activation also makes model-source logging essential.

### Clean-Finish Periodic SAM
**Summary**: Apply plain SAM with rho 0.05 on every second optimizer step only during the final clean quarter. Cadence uses the upcoming one-based step, `(step + 1) % 2 == 0`. Replay stochastic-depth RNG between the two passes, suppress the second BatchNorm buffer update, restore parameters exactly, and perform one existing Nesterov update. The initial period-four design is in `proposals/idea-03.md`; the period-two refinement comes from adversarial review.

**What it targets**: The accuracy limiter is generalization from a well-fitted solution. SAM directly optimizes local neighborhood flatness after the successful CutMix representation-learning phase rather than increasing data or residual regularization.

**Reasoning**: ICLR evidence shows strong CIFAR generalization gains from SAM. Restricting it to period-two pulses in the last quarter preserves all early parent steps and should retain about 25,000-25,800 total updates after fixed SAM overhead, rather than full SAM's near-halving. The stronger dose is justified because three intervening Nesterov updates could dilute period-four SAM in the momentum buffer.

**Sources**: `papers/sharpness-aware-minimization.md`; `proposals/idea-03.md`; EXP-002 and EXP-003 analyses.

**Estimated Effort**: high

**Risk Assessment**: The late periodic variant is not directly validated by the source and sacrifices optimizer exposure. Correct RNG replay, BatchNorm handling, parameter restoration, one-based cadence, and BF16 gradient perturbation create a materially larger correctness surface.

### Clean-Only Two-Stage Label Smoothing
**Summary**: Add epsilon 0.05 label smoothing only to clean, non-CutMix batches while progress is below 0.75. Leave CutMix's existing two-label objective unchanged and return to hard-label cross-entropy for the final quarter. Full design: `proposals/idea-02.md`.

**What it targets**: Roughly half of early EXP-002 batches retain hard targets even though the model ultimately overfits. Clean-only smoothing adds an output-space generalization mechanism to those batches without further softening mixed targets or weakening late fitting.

**Reasoning**: NeurIPS evidence links mild label smoothing to improved generalization and class representations; phase-dependent regularization evidence supports removing it late. The implementation changes only the clean loss branch, consumes no RNG, adds negligible compute, and preserves every accepted parent mechanism.

**Sources**: `papers/when-label-smoothing-helps.md`; `knowledge/papers/time-matters-regularization.md`; `proposals/idea-02.md`; EXP-002 analysis.

**Estimated Effort**: low

**Risk Assessment**: CutMix and drop path may already provide enough early regularization, so smoothing every remaining early batch could underfit or merely shift confidence without improving accuracy. The predicted 0.10-0.30-point effect is close to the variance observed in EXP-003.

## Review

Claude's cross-model review found no hard-constraint or integrity violation, but judged the candidates against EXP-003's 0.14-0.29-point selection variance. It rejected sparse late EMA because the parent shows only a 0.04-point best/final gap and lacks SWA's LR-diverse tail. It judged clean-only label smoothing well designed but too likely to be redundant with the already saturated CutMix/drop-path regime. It selected periodic SAM for its larger effect ceiling and direct attack on generalization, while warning that period four could be diluted by intervening momentum updates. The chosen design adopts period two and explicitly defines cadence as `(step + 1) % 2 == 0`. Full review: `01-idea-review.md`.

## Idea Evaluation

The Claude verdict is adopted. Periodic SAM scored highest on potential impact and tied label smoothing on evidence/quality, while EMA's mechanism did not match the observed variance. The period-two refinement increases expected charged overhead from roughly 4-5% to 7-8%, but doubles the sharpness-aware dose and still preserves the entire early EXP-002 representation-learning phase.

## Chosen Idea
**Selected**: Clean-Finish Periodic SAM

**Why this idea**:
The parent already fits strongly, scalar regularization tuning did not confirm, and the measured selection noise makes small-effect interventions low-information. SAM is the only finalist with a demonstrated effect ceiling comfortably larger than the 0.10-point gate. Limiting it to every second step after the 75% clean transition preserves the successful CutMix phase while delivering a sustained optimizer-level flatness bias at bounded cost.

**Hypothesis**:
Plain SAM with rho 0.05 on every second step of the final clean quarter will retain at least 24,000 optimizer steps (25,000-25,800 expected), complete within the fixed 300-second charged budget and 600-second outer timeout, and achieve `best_test_acc >= 95.33%` versus the 95.23% EXP-002 parent. The gain should come from flatter late optimization rather than additional data regularization; any threshold miss or protocol violation is a no-improvement without retry.
