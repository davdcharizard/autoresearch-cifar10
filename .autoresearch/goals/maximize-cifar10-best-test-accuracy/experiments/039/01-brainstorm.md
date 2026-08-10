# Brainstorm EXP-039
**Created**: 2026-08-06

## Web Search & Literature Review

- **PyTorch Channels-Last Memory Format** (`knowledge/references/pytorch-channels-last.md`; official PyTorch tutorial)
  CUDA convolution and BatchNorm support channels-last physical layout while preserving logical NCHW shapes, but FP32 32x32 performance must be measured locally and hidden conversions can erase gains.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`; ICML 2025)
  Recency-weighted averages can complement annealing, but EXP018's local uniform-SWA loss and the accepted tiny best/final gap weaken the case that terminal noise is limiting.
- **Mixed Pooling for Convolutional Neural Networks** (`knowledge/papers/mixed-pooling.md`)
  Combining spatial statistics can preserve localized evidence, but local max-path failures require an intrinsically dense, lifetime-bounded alternative.

## Experimental History Review

- EXP010 remains 94.15% with width2, N1/M7 plus 50% CutMix to80%, and a hard weak tail;39 experiments have not moved the frontier.
- EXP038's thorough six-proposal pass already developed channels-last, batch192, EMA, bounded RMS pooling, stochastic depth, and cosine classification. Independent review scored channels-last evidence/reasoning highest (7/10) but selected cosine for impact; cosine then failed row geometry. Batch192 was rejected for losing decisions and raising global LR.
- Model forward/backward is97.57% of counted GPU work, backward alone75.46%; memory is only0.61% used. Channels-last is the only untried finalist that directly targets measured work without changing batch noise, model, data, or SGD.
- EXP018 rejects uniform late SWA, not short EMA, but the accepted trajectory has almost no best-final gap. EXP014/031 reject sparse max readouts, while EXP038's `1/64` average-plus-RMS proposal remains bounded but has weak direct accuracy evidence.
- Any speed change can add epoch-based test looks. Rather than rewrite cadence, a candidate may preserve the accepted conditions and cap evaluations at the baseline's19 opportunities; adversarial plan review must decide whether that is comparable or requires a same-cadence baseline.

## Objective Limiter Diagnosis

Accuracy remains limited by generalization under the short strong phase, but the reviewed representation candidates have repeatedly failed safety or regressed fit. The clearest remaining measurable lever is backward kernel cost. A channels-last speedup cannot guarantee accuracy, yet it can preserve every accepted learning choice while adding strong and weak optimizer decisions. Its mechanism is directly falsifiable with layout propagation and fresh paired timing, so it is now preferable to another weakly grounded representation perturbation.

## Collected Ideas

## Combinations

## Candidate Ideas

### Short-Timescale Late Parameter EMA
**Summary**: Maintain a parameter-only EMA with decay0.999 from90-99%, install it once, and spend the last1% recomputing BN on hard weak batches before the existing terminal evaluation. Full development: `../038/proposals/idea-03.md`.

**What it targets**: Terminal weak-tail parameter noise with recency weighting, distinct from EXP018's uniform86-98% average that pulled toward inferior earlier states.

**Reasoning**: Literature supports averaging alongside annealing, and EXP018 proves the local mechanics and BN refresh are feasible. However, the accepted best/final gap is only0.01 and EXP018's within-run average worsened both accuracy and NLL, so the remaining effect has a weak prior.

**Sources**: `knowledge/papers/weight-averaging.md`; EXP018; `../038/proposals/idea-03.md`.

**Estimated Effort**: high.

**Risk Assessment**: EMA lag, per-step overhead, reserved refresh time, or BN recalibration can displace useful updates and worsen the already monotonic tail.

### FP32 Channels-Last Training
**Summary**: Convert the already-initialized model's 4-D parameters to `torch.channels_last`, transfer each 4-D training input in that format inside counted time, and add a forward-boundary conversion for immutable evaluator inputs. Preserve FP32/default TF32, batch128, model, optimizer, and curriculum. Cap the unchanged evaluation conditions at19 calls only if measured speed would otherwise add opportunities. Full development: `../038/proposals/idea-01.md`.

**What it targets**: The measured convolution/BN workload—75.46% backward and22.11% forward—by selecting NHWC-compatible CUDA kernels, potentially adding fixed-budget decisions without changing statistical batch geometry.

**Reasoning**: Official PyTorch support and the local profile give a direct systems mechanism. EXP016's BF16 failure is irrelevant because this stays FP32/width2; EXP029 never tested layout. Exact stride propagation, profiler conversion counts, numerical replay, and seven fresh timing pairs can establish or veto the premise before production.

**Sources**: `knowledge/references/pytorch-channels-last.md`; `02-system-understanding.md`; EXP013 timing/evaluation lessons; EXP029; `../038/proposals/idea-01.md`.

**Estimated Effort**: high.

**Risk Assessment**: Tiny FP32 CIFAR tensors and Option-A slice/pad may be slower or trigger repairs; a real 3% speedup may still be too small for +0.10 accuracy. Changed cuDNN reductions make the result the net implementation effect, not a bitwise counterfactual.

### Intrinsically Bounded Average-plus-RMS Readout
**Summary**: Replace global average descriptor `mu` by `mu + (rms-mu)/64` on nonnegative final maps. This has no parameters, keeps every spatial gradient dense, and bounds descriptor/Jacobian perturbation to10.94% for every example. Full development: `../038/proposals/idea-04.md`.

**What it targets**: Localized CutMix feature dilution at the final8x8 aggregation while preserving the accepted backbone, optimizer, and area-dominant representation.

**Reasoning**: It is materially different from EXP014/031 hard-max failures because it has no argmax, new head, or initialization-only calibration and carries lifetime algebraic bounds. The potential direct accuracy effect is larger than a small layout speedup, but external/local evidence that RMS energy helps is weak.

**Sources**: `knowledge/papers/mixed-pooling.md`; `knowledge/papers/cutmix.md`; EXP014/031; `../038/proposals/idea-04.md`.

**Estimated Effort**: medium-high.

**Risk Assessment**: The safe coefficient may be sub-threshold, RMS can emphasize augmentation artifacts, and the extra norm reduction can reduce fixed-budget exposure.

## Review

Claude's independent review (`01-idea-review.md`) selected **Intrinsically Bounded Average-plus-RMS Readout**, scoring it6/10 for evidence/reasoning and6/10 for impact. It judged the lifetime descriptor/Jacobian bounds a genuine cure for EXP014/031's sparse max geometry and valued that the idea directly targets localized CutMix representation. It scored channels-last6/10 and4/10 because the exposure-to-accuracy link is unproven and a timing veto is likely, and EMA4/10 and3/10 because the accepted best/final gap is only0.01 and EXP018's harmful BN refresh is inherited.

I accept the review's main RMS caveat: `1/64` may be too weak. Planning will add a fixed training-corpus descriptor/logit activity diagnostic that proves the perturbation is nonzero and distributed, without using labels, accuracy, or the diagnostic to select a coefficient. It remains a mechanism check, not a promise that the effect is large enough for +0.10.

## Idea Evaluation

- **Average-plus-RMS readout** — Advance. Best alignment with the accuracy limiter and strongest lifetime safety argument; add a non-scoring activity diagnostic and preserve the coefficient exactly.
- **FP32 channels-last** — Defer. Strong systems evidence but low accuracy ceiling and likely pre-production timing veto; retain as an infrastructure candidate.
- **Short late EMA** — Reject. The baseline has negligible terminal noise and full BN refresh repeats an EXP018 failure channel.

## Chosen Idea
**Selected**: Intrinsically Bounded Average-plus-RMS Readout

**Why this idea**:
It makes one dense, parameter-free final-representation change with an intrinsic lifetime bound: every descriptor and pooling-Jacobian coefficient stays within10.94% of ordinary average pooling for every nonnegative example. That directly addresses localized CutMix dilution while avoiding the argmax sparsity, new-head recruitment, and calibration-only bounds that failed in EXP014/031.

**Hypothesis**:
Replacing global average by `mu + (rms-mu)/64` will create a measurable but distributed descriptor/logit change, retain control-qualified safe geometry and near-full exposure, preserve strong-phase fit, and raise seed-42 `best_test_acc` from94.15% to at least94.25%. A safety/timing veto or valid miss retires this exact `1/64` point without coefficient, GeM, learned-gate, phase, or pooling rescue.
