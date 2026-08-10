# Brainstorm EXP-022
**Created**: 2026-08-06

## Web Search & Literature Review

- **Lookahead Optimizer: k steps forward, 1 step back** (`experiments/022/papers/lookahead-optimizer.md`; NeurIPS 2019)
  Fast/slow interpolation reduces optimizer variance with small amortized overhead; the paper reports CIFAR gains and robust `k=5, alpha=0.5` behavior with retained inner state.
- **ShakeDrop Regularization for Deep Residual Learning** (`experiments/022/papers/shakedrop.md`; ICLR workshop / IEEE Access)
  Stochastic residual-branch perturbation improves CIFAR ResNets, but its deliberately strong disturbance warns against paper-default strength in this already underfit-prone short horizon.
- **DropBlock** (`experiments/022/papers/dropblock.md`; NeurIPS 2018)
  Structured feature masking can outperform independent dropout, especially when scheduled gradually, but it adds timed masking and likely compounds the current strong augmentation.
- **Swapout** (`experiments/022/papers/swapout.md`; NeurIPS 2016)
  Sampling identity/residual path combinations can improve structurally identical CIFAR ResNets, providing outside support for residual-path stochasticity while leaving depth and fit transfer uncertain.

## Experimental History Review

- The moving baseline is 94.15% at `7c1e7d8` from EXP-010: width-2 postactivation ResNet-20, N1/M7 and p=0.5 alpha-1 CutMix through 80%, then a hard weak tail. This recipe retained 26,898 updates and finished with its best checkpoint.
- The durable gains were long high-LR exploration (EXP-002), plateau-only RandAugment (EXP-004), width 2 (EXP-007), and conservative CutMix (EXP-010). These establish capacity plus strong-view exploration followed by weak hard-label refinement as the recipe to preserve.
- Stronger CutMix, preactivation, and selective zero-gamma (EXP-011/012/015) suppressed strong-phase fit. Width-2 decay variants (EXP-008/009), transition shortcuts (EXP-017/021), uniform late SWA (EXP-018), raw max readout (EXP-014), and no-warmup Nesterov (EXP-020) did not advance or failed safety gates.
- EXP-013 measured backward at 75.46% of GPU-stage time and rejected batch 256 because image throughput rose only 18.91%. FP32 channels-last remains an untried direct kernel-layout attack, but prior review judged its FP32 speed premise weak and exposure-to-accuracy unproven.
- EXP-019 did not establish Mixup quality because fresh forkserver processes failed to replay post-transform data; future paired gates must materialize exact augmented tensors. EXP-021 then showed exact-corpus gates can detect candidate-only transients.
- Remaining gaps include optimizer trajectory smoothing that feeds back into training, lighter CutMix geometry at the accepted event rate, exact FP32 memory-layout timing, weak-tail EMA rather than uniform SWA, and depth/width allocation changes. The current accuracy limiter is generalization under a short strong-view phase while the system limiter is convolution/BN backward.

## Collected Ideas

- **Lookahead-wrapped momentum SGD** — retain accepted inner SGD but synchronize fast parameters toward slow parameters every five updates. It targets noisy high-LR optimization and short-horizon basin quality; NeurIPS CIFAR evidence and the distinction from post-hoc SWA make it a plausible untried mechanism.
- **FP32 full-model channels-last** — convert the width-2 model, training inputs, and evaluation boundary to NHWC physical storage while preserving logical NCHW and FP32. It attacks the measured 97.57% model forward/backward share directly; any value depends on a paired H20 speedup and then on extra exposure helping accuracy.
- **CutMix alpha 0.5 at p=0.5** — keep the successful event count but move the beta distribution toward endpoint-heavy rectangles, reducing expected target ambiguity by 25%. It targets the small EXP-010 strong-fit deficit without repeating EXP-011's stronger probability, though it may hollow out useful moderate occlusions.
- **One-epoch-half-life weak-tail EMA** — maintain a recent-weight EMA only after the 80% switch and evaluate the live endpoint and EMA with correctly handled BN state under a parity-capped schedule. It targets weak-tail checkpoint noise and differs from EXP-018's uniform long-window average, but still risks backward bias and extra parameter-update cost.
- **Low-rate strong-phase stochastic depth** — randomly bypass only same-width residual branches at a small depth-scaled rate during the high-LR phase, restoring the full graph for refinement. It imports ShakeDrop/Swapout's ensemble regularization, but shallow depth and the recurring identity-oriented underfit pattern make it a secondary idea.
- **Scheduled final-stage DropBlock** — gradually introduce a very low block-drop rate only in stage 3 during the latter half of the strong phase, then disable it in the weak tail. This targets localized feature co-adaptation with literature support, but the model already receives RandAugment and CutMix and the extra mask kernel consumes counted time.
- **FP32 width-3 with fewer residual blocks** — trade depth for width (for example two blocks per stage at width 3) to increase channel capacity without the full cost of width-3 ResNet-20. It attacks representation allocation and the backward bottleneck together, but requires careful timing and has no local evidence that losing depth preserves CIFAR hierarchy.
- **Strong-phase residual noise plus Lookahead** — use very mild path perturbation while Lookahead damps the resulting update variance. This moonshot combines representation-ensemble regularization with optimizer stabilization, but causal attribution is poor and either component could suppress already scarce fit.

## Combinations

- **Channels-last + depth/width rebalance**: layout speed could fund a wider, shallower model that neither change can afford alone; the cross is stronger only if NHWC accelerates the actual FP32 kernels and the new architecture converts that budget into useful channel capacity.
- **Lookahead + weak-tail EMA**: Lookahead would stabilize training throughout, while a short EMA would select a smoother terminal predictor. The cross could reduce both optimizer and endpoint variance, but the mechanisms overlap and would obscure whether EXP-018's averaging failure was truly fixed.
- **Endpoint-heavy CutMix + low-rate residual noise**: less ambiguous input mixing could create regularization headroom for feature-path stochasticity. The cross may preserve total disturbance while diversifying its source, but it is too underfit-prone and multi-variable for the next isolated run.

## Candidate Ideas

### Lookahead-Wrapped Momentum SGD
**Summary**: Add Lookahead around the accepted SGD using `k=5`, `alpha=0.5`, and persistent momentum state. Every fifth normal update, interpolate a slow parameter copy halfway toward fast weights and copy it into the live model; preserve every architecture, schedule, augmentation, and evaluation setting. Full specification: `proposals/idea-01.md`.

**What it targets**: The accuracy-side limiter in `02-system-understanding.md`: generalization from a short, noisy high-LR strong-view trajectory. Unlike post-hoc averaging, it changes the trajectory throughout training.

**Reasoning**: The NeurIPS paper reports lower variance, faster convergence, and small CIFAR gains with momentum SGD at batch 128. The accepted recipe has a long LR-0.1 RandAugment/CutMix phase where such damping can matter. EXP-018's failed uniform SWA is not the same operation because Lookahead exponentially weights recent proposals and feeds each synchronized slow point back into later optimization.

**Sources**: `papers/lookahead-optimizer.md`; `proposals/idea-01.md`; EXP-010 and EXP-018 analyses; `02-system-understanding.md`.

**Estimated Effort**: Medium — small production code, but optimizer-state semantics, counted overhead, and aligned safety checks need verification.

**Risk Assessment**: Synchronization may shrink effective progress, mismatch retained momentum with the copied location, or improve NLL without top-1. The paper used longer training and its small gain may not transfer. Require at least 99% exposure and diagnose strong-phase suppression against 89.73%/87.08% markers.

### FP32 Full-Model Channels-Last
**Summary**: Convert the accepted width-2 model and all 4D inputs to channels-last physical storage while retaining FP32 and logical NCHW. Add only a 19-evaluation parity cap if speed adds epochs. Full specification: `proposals/idea-02.md`.

**What it targets**: The measured systems limiter in `02-system-understanding.md`: convolution/BN backward dominates 75.46% of step time. A genuine kernel speedup could buy more strong and weak updates in the same 300 seconds.

**Reasoning**: The graph is almost entirely Conv/BN/ReLU/add and memory headroom is enormous. PyTorch/cuDNN support NHWC propagation, so a pure-layout experiment is feasible and semantically narrow. However, existing guidance emphasizes reduced precision and prior EXP-021 review correctly noted there is no direct evidence for a speedup on this tiny FP32 workload.

**Sources**: `proposals/idea-02.md`; EXP-013 systems profile; EXP-021 brainstorm/review; official PyTorch and NVIDIA sources cited by the proposal.

**Estimated Effort**: Medium — small production diff, substantial layout, numerical, profiler, timing, and fairness gates.

**Risk Assessment**: Tiny CIFAR kernels, the RGB stem, Option-A slice/pad, or pooling may force conversions or select slower kernels. Even a 3% speedup may not improve a generalization-limited metric, and different FP32 reductions alter the fixed-seed trajectory.

### CutMix Alpha 0.5 at Fixed Probability
**Summary**: Change only `CUTMIX_ALPHA` from 1.0 to 0.5 while retaining p=0.5, the 80% strong phase, width 2, optimizer, and hard weak tail. Full specification: `proposals/idea-03.md`.

**What it targets**: The small strong-phase fit deficit at the accepted frontier: EXP-010 switched at 89.73% versus EXP-007's 90.08%, while stronger CutMix frequency caused severe underfit in EXP-011.

**Reasoning**: Alpha 0.5 preserves mean lambda and event count but lowers expected two-class ambiguity from 1/3 to 1/4 by favoring near-endpoint rectangles. It is a clean, compute-neutral test of whether the accepted regional regularizer can be made easier to fit without removing it.

**Sources**: `proposals/idea-03.md`; `knowledge/papers/cutmix.md`; EXP-010, EXP-011, and EXP-021 review.

**Estimated Effort**: Low production effort, medium verification effort due exact augmented-corpus and empirical geometry gates.

**Risk Assessment**: Moderate rectangles may be the source of CutMix's useful localization benefit; endpoint-heavy draws may act like no-ops/full donors, yielding a low-ceiling scalar retry contrary to prior guidance favoring an orthogonal representation lever.

## Review

The external Claude harness returned an empty file, so the required independent idea-critic fallback produced `01-idea-review.md`. It selected Lookahead as the only finalist with a literature-backed mechanism acting throughout the diagnosed noisy strong-view trajectory and more ceiling than a scalar refinement.

The review's main concern is significant: `k=5, alpha=0.5` filters committed displacement and coupled decay as well as variance, while retaining momentum after copying weights creates a transient location/velocity mismatch. The plan must therefore compare committed five-step displacement and decay-only displacement with aligned SGD, and inspect every synchronization plus following step through step 50 for class concentration, update/momentum norms, and fast/slow distance. These are defect and mechanism gates, not permission to compensate LR or decay.

The channels-last critique is accepted: direct FP32 speed evidence and exposure-to-accuracy evidence are both weak, and different cuDNN reduction order would confound attribution. The CutMix critique is also accepted: alpha 0.5 reduces meaningful moderate mixtures rather than merely reducing ambiguity, opposing an ingredient that supplied a 0.60-point gain. Both remain valid future measurements but have lower expected value than Lookahead in this slot.

## Idea Evaluation

| Candidate | Evidence / reasoning | Potential impact | Decision |
|---|---:|---:|---|
| Lookahead-wrapped momentum SGD | 3.5/5 | 3.5/5 | Select, with synchronization-local safety and committed-displacement diagnostics. |
| CutMix alpha 0.5 | 3/5 | 2.5/5 | Defer; clean but low-ceiling and likely weakens useful moderate regional events. |
| FP32 channels-last | 2.5/5 | 2/5 | Defer; two speculative links and likely timing veto. |

## Chosen Idea
**Selected**: Lookahead-Wrapped Momentum SGD

**Why this idea**:
Lookahead is distinct from the failed uniform late SWA because it feeds a recent-weight slow point back into every subsequent optimization segment. Its variance-reduction and faster-convergence rationale has direct NeurIPS CIFAR evidence, it preserves the accepted architecture/data recipe, and it targets the short noisy high-LR trajectory rather than relying on an unproven exposure dividend or weakening accepted CutMix. The material risks—effective-progress filtering and location/momentum mismatch—are observable with aligned early-step diagnostics and do not justify modifying the registered operating point.

**Hypothesis**:
Wrapping accepted momentum SGD with Lookahead at `k=5`, `alpha=0.5`, while retaining inner momentum state, will reduce high-LR trajectory variance enough to improve terminal generalization and raise `best_test_acc` from 94.15% to at least 94.25%, while retaining at least 26,629 optimizer steps. The hypothesis is falsified if synchronization causes candidate-only concentration or severe committed-displacement suppression, if exposure falls below the gate, or if a valid seed-42 run misses 94.25%.
