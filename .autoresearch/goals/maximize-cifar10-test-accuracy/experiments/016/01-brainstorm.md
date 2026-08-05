# Brainstorm EXP-016
**Created**: 2026-07-26

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): moderate depth and width can spend CIFAR compute more effectively than extreme depth; the local low-resolution capacity near-misses sharpen this into an allocation question.
- **RandAugment** (`knowledge/papers/randaugment.md`): a small operation-count/magnitude space improves CIFAR augmentation without a separate policy search and is available in torchvision.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early regularization can persist after removal, but worker-side temporal augmentation is operationally harder than the accepted GPU-side mixup cutoff. No network access was used.

## Experimental History Review

- The accepted 94.07% recipe remains WRN-16-2 plus batch-shared alpha-0.2 mixup through 65% and a hard-label tail. Exposure, precision, schedule endpoints, averaging, initialization, and coefficient decorrelation have not improved it.
- The clearest positive residual signal is capacity at 8x8: width 160 reached 94.11% and an added full 128-channel block reached 94.15%, although both sacrificed about 7% exposure and missed 94.17%.
- The rank-64 substitute failed, indicating that the positive signal depends on dense low-resolution transformations rather than merely another residual path. A fixed-block-count reallocation remains untested.
- Stronger or additive regularization has usually regressed, but the accepted augmentation is only crop/flip plus mixup. Mild learned-policy-style input invariance and channel attention remain untested and mechanistically distinct.
- The limiting gap remains top-1 generalization under a fully fit model, not raw exposure. The next experiment should either allocate existing transformation budget toward the late semantic stage or add a high-value invariant/representation mechanism with tightly controlled regularization.

## Collected Ideas

- **Stage-depth redistribution `[1,2,3]`** — remove the second 32-channel 32x32 block and add a third dense 128-channel 8x8 block. This keeps six residual blocks and near-constant convolutional MACs while reallocating parameters to the stage that produced EXP-011's +0.08 signal, potentially retaining exposure better than simply adding depth.
- **Alternative redistribution `[2,1,3]`** — move one 64-channel 16x16 block to 8x8, preserving early high-resolution extraction while shortening the middle stage. It is less aggressive than removing stage-1 depth but may disrupt the transition hierarchy.
- **Mild RandAugment** — insert `transforms.RandAugment(num_ops=1, magnitude=5)` before tensor conversion while keeping crop/flip and accepted mixup. One low-magnitude operation could add invariance beyond spatial crop/flip, but full-run stacking may over-regularize and CPU transform semantics must be verified.
- **Stage-3 squeeze-and-excitation** — add a small channel gate to each existing 8x8 residual branch using global pooling and a narrow MLP, initialized to a near-neutral scale. It targets feature selection at the empirically promising stage with low spatial compute, but adds nonlinear state and a gating calibration choice.
- **Stage-2 manifold mixup** — apply the accepted batch-shared interpolation to intermediate activations after stage 1 or 2 rather than pixels, retaining paired-label loss and cutoff. This representation-level regularizer may avoid pixel-space artifacts, but it replaces a proven mechanism and complicates forward semantics.
- **Late BatchNorm-statistics freeze** — stop updating running statistics near the end while continuing affine/weight optimization, simplifying the final tail's evaluation state. It could reduce late running-stat drift, but no existing evidence identifies BN drift and the accepted final evaluation is already stable.
- **Cosine classifier** — normalize pooled features and classifier rows with a fixed or learned scale, turning decisions into angular margins. This directly changes representation geometry but requires a scale operating point and risks disrupting the successful late hard-label convergence.
- **Shake-style dual-branch moonshot** — split selected late residual transformations into two branches with stochastic convex forward/backward combination. It has high ensemble-like upside but adds compute and strong stochastic regularization to a recipe already sensitive to both.

## Combinations

- **`[1,2,3]` redistribution + neutral stage-3 channel gates**: spend the removed early block's representational budget on both dense late transformation and channel selection. The combination could amplify the positive late-capacity signal, but it confounds allocation and attention and should follow standalone evidence.
- **Mild RandAugment + unchanged temporal mixup**: use one weak input operation while preserving the validated batch-shared mixup schedule. The two act on invariance and interpolation respectively, but prior additive-regularization failures make standalone mild augmentation the cleaner first test.
- **`[1,2,3]` redistribution + accepted mixup**: retain the full accepted training recipe while moving one block late. This is stronger than added depth alone if it recovers update exposure without losing early feature quality.

## Candidate Ideas

### One-Operation Mild RandAugment
**Summary**: Insert `RandAugment(num_ops=1, magnitude=5)` after crop/flip and before tensor conversion, active for the full run, while preserving accepted batch-shared mixup and every model/optimizer setting (`proposals/idea-02.md`).

**What it targets**: Missing label-preserving input invariances beyond crop and flip. It expands color and mild geometric variation without changing model compute or soft-target strength.

**Reasoning**: RandAugment has direct CIFAR evidence and exists in the installed torchvision. The operating point is deliberately weaker than standard policies because stronger mixup, CutMix, and dropout all regressed locally. Worker-side cost is outside counted GPU time but must stay under the wall limit.

**Sources**: `knowledge/papers/randaugment.md`; `knowledge/papers/time-matters-regularization.md`; EXP-002/003/005/006/015; `proposals/idea-02.md`.

**Estimated Effort**: low

**Risk Assessment**: Always-on augmentation stacks with mixup and can interfere with the validated hard-label tail. Worker RNG changes are intrinsic, magnitude-independent operations may still be strong, and PIL overhead can lengthen wall time.

### Neutral Stage-3 Squeeze-and-Excitation
**Summary**: Add ratio-16 channel gates to both existing stage-3 residual branches, using `2*sigmoid` and zero second projections so every initial scale is exactly one. Attach gates after `conv2` and before shortcut addition, adding 4,368 parameters and negligible arithmetic (`proposals/idea-03.md`).

**What it targets**: Per-example selection of dense low-resolution features rather than more raw capacity. It follows the same stage-3 signal while avoiding another full transform.

**Reasoning**: The two late-capacity probes were directionally positive, but the extra block worsened loss and the compressed bottleneck failed. A neutral channel selector is a distinct mechanism with near-baseline exposure and an accepted initial forward function.

**Sources**: EXP-010/011/012 reports; `knowledge/papers/wide-residual-networks.md`; accepted `train.py`; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: Attention may be redundant or learn too slowly from a neutral start, and input-conditioned amplification can worsen confidence. Small pooling/MLP kernels may cost more than static MACs suggest.

### Stage-Depth Redistribution `[1,2,3]`
**Summary**: Move the second 32-channel stage-1 block to the end of the 128-channel stage 3, keeping six residual blocks and exactly 101,106,944 convolution/linear MACs per image. Preserve all surviving accepted initialization and RNG state by constructing the accepted graph first, removing `layer1[1]`, and attaching one locally initialized full-width late block (`proposals/idea-01.md`).

**What it targets**: Misallocation of the fixed transformation budget. Dense 8x8 capacity is the only repeated positive neighborhood, while this reallocation aims to retain accepted throughput rather than paying EXP-011's seventh-block cost.

**Reasoning**: EXP-010 and EXP-011 reached 94.11% and 94.15% despite only about 132 passes; EXP-012 showed a cheap rank-64 substitute does not preserve that signal. A stage-1 and stage-3 identity block have equal convolutional MACs because width doubles as spatial resolution halves twice, so `[1,2,3]` tests whether semantic depth is more valuable than local refinement without adding arithmetic.

**Sources**: `knowledge/papers/wide-residual-networks.md`; EXP-010/011/012 reports and `04-results.tsv`; `proposals/idea-01.md`.

**Estimated Effort**: medium

**Risk Assessment**: One high-resolution block may underdevelop local features, while parameters increase 40% despite equal MACs and could worsen confidence. Shape-dependent kernels and careful RNG-preserving construction require strict preflight.

## Review

The blind reviewer selected **Stage-Depth Redistribution `[1,2,3]`**. I adopt the warning that EXP-010/011 support dense late capacity but do not prove the removed early block is expendable. The experiment will therefore be interpreted as a verdict on the coupled allocation, with final loss reported alongside top-1 and no rescue via `[2,1,3]`, local initialization seed, or added attention. Full feedback is in `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Redistribution has the strongest repeated local evidence and turns EXP-011's near miss into a fixed-MAC hypothesis. Always-on RandAugment conflicts with the successful clean tail, while SE infers attention demand from capacity evidence without a direct source.

## Chosen Idea
**Selected**: Stage-Depth Redistribution `[1,2,3]`

**Why this idea**:
It directly composes the only two directionally positive post-baseline signals: dense 8x8 width and depth. Exchanging equal-MAC blocks may retain the useful late transformation while recovering EXP-011's lost exposure, and the failure question is explicit: whether one 32x32 block is necessary.

**Hypothesis**:
Redistributing the six residual blocks from `[2,2,2]` to `[1,2,3]`, while preserving 101,106,944 convolution/linear MACs per image, at least 97% measured throughput, common accepted initialization/RNG state, and every training hyperparameter, will raise fixed-seed `best_test_acc` from 94.07% to at least 94.17%.
