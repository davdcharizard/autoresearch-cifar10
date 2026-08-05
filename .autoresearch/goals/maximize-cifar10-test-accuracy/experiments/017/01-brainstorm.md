# Brainstorm EXP-017
**Created**: 2026-07-26

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): retain the accepted moderate-depth wide residual backbone; local results now show both early and late dense transforms matter.
- **RandAugment** (`knowledge/papers/randaugment.md`): one low-magnitude operation is an available label-preserving invariance lever, though always-on stacking is locally risky.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): supports confining expensive or regularizing interventions early and preserving the accepted hard-label tail. Offline project knowledge only; no network access.

## Experimental History Review

- The accepted 94.07% `[2,2,2]` WRN with early batch-shared mixup remains unbeaten after sixteen experiments.
- Dense late width/depth gave the only near-positive architecture scores, but replacing an early block with late depth produced 171.70 passes and only 93.82%. Both high-resolution refinement and dense late semantics matter; further raw block reallocation is closed.
- Additional target/feature regularization, altered schedules/decay, precision, averaging, initialization, and coefficient decorrelation all regressed. The model fits fully, so useful per-example feature selection or a qualitatively distinct geometry objective remains more plausible than exposure or stronger softness.
- Stage-3 attention, mild input invariance, and a short sharpness-aware window were developed but not scored in EXP-016/015. The solution space for this loop is narrow enough for a quick comparison.

## Collected Ideas

## Combinations

## Candidate Ideas

### Neutral Stage-3 Squeeze-and-Excitation
**Summary**: Attach ratio-16 SE gates to the two existing stage-3 residual branches after `conv2` and before addition. Use zero second projections and `2*sigmoid`, so every scale is exactly one and the initial candidate function equals accepted; add 4,368 parameters and preserve `[2,2,2]` (`experiments/016/proposals/idea-03.md`).

**What it targets**: Input-conditioned selection of the dense 8x8 features whose width/depth showed positive local signal, without deleting early depth or paying for another transform.

**Reasoning**: EXP-010/011 indicate late semantic capacity helps, while EXP-012/016 show compressed late paths or early-to-late exchange fail. Identity-initialized channel selection is a new mechanism that leaves accepted representation and optimization intact at step zero and should retain nearly all exposure.

**Sources**: EXP-010/011/012/016 reports; `knowledge/papers/wide-residual-networks.md`; `experiments/016/proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: The capacity signal does not directly prove attention demand. Gates may remain inert, amplify confident errors, or learn slowly because the first projection opens only after the zero second projection updates; training-only gate displacement/saturation must be recorded.

### Ten-Percent Early-Window SAM
**Summary**: Use non-adaptive SAM at rho 0.05 only before 10% counted progress, then exact accepted SGD for 90%, reusing each mixed batch twice and restoring BatchNorm buffers so one persistent update occurs (`experiments/015/proposals/idea-03.md`).

**What it targets**: Solution sharpness rather than capacity, exposure, or target softness.

**Reasoning**: It has the highest qualitatively distinct upside and confines double-pass cost to about 30 seconds. However, no local SAM evidence calibrates rho/window and the treatment projects about 5% fewer updates, including near peak LR.

**Sources**: `knowledge/papers/time-matters-regularization.md`; EXP-002/009/013/016; `experiments/015/proposals/idea-03.md`.

**Estimated Effort**: high

**Risk Assessment**: Uncalibrated radius, BatchNorm interaction, restoration complexity, and lost early update density make the result a compound fixed-budget trade.

### One-Operation Mild RandAugment
**Summary**: Add `RandAugment(num_ops=1, magnitude=5)` after crop/flip and before tensor conversion for all training samples, keeping accepted mixup and model behavior (`experiments/016/proposals/idea-02.md`).

**What it targets**: Label-preserving invariances absent from crop/flip, with no model or GPU compute change.

**Reasoning**: RandAugment has direct CIFAR evidence and is locally available. A deliberately weak operation may move generalization by several tenths, but the full-run worker-safe design conflicts with the successful clean tail and repeated additive-regularization failures.

**Sources**: `knowledge/papers/randaugment.md`; `knowledge/papers/time-matters-regularization.md`; EXP-002/003/005/006/015; `experiments/016/proposals/idea-02.md`.

**Estimated Effort**: low

**Risk Assessment**: Always-on augmentation can obstruct late clean margin refinement; worker RNG and PIL wall cost change intrinsically, and even magnitude-independent operations may be strong.

## Review

The blind review selected **Neutral Stage-3 Squeeze-and-Excitation**. I adopt the attribution warning that early gate learning may initially be bias-driven, making it channel scaling before it becomes input-conditioned attention. The plan must preregister evaluator-free observational diagnostics for gate mean displacement, saturation, per-example variance, and feature-versus-bias logit contribution; none may alter the treatment or verdict. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Neutral SE best preserves every representation component that recent failures showed necessary while targeting the repeatedly positive late-feature neighborhood. SAM has greater speculative upside but no measured sharpness premise; always-on RandAugment conflicts with the accepted clean tail.

## Chosen Idea
**Selected**: Neutral Stage-3 Squeeze-and-Excitation

**Why this idea**:
It adds a low-cost selector to both existing dense 8x8 branches without removing early depth, changing accepted convolutions, or perturbing the initial function. The placement follows EXP-010/011, while identity initialization and RNG isolation keep attribution cleaner than the alternatives.

**Hypothesis**:
Exactly neutral ratio-16 SE gates on both stage-3 residual branches will retain at least 95% matched throughput and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% by learning useful late-channel modulation without sacrificing accepted representation or exposure.
