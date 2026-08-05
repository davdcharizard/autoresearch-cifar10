# Brainstorm EXP-012
**Created**: 2026-07-24

## Web Search & Literature Review

- **Weight averaging** (`knowledge/papers/weight-averaging.md`): a carefully selected late averaging window can mildly improve generalization with low implementation and memory overhead.
- **Label smoothing** (`knowledge/papers/label-smoothing.md`): mild soft targets can reduce overconfidence, but stacking soft-target methods requires calibration.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): shallow/wide residual allocation is generally more compute-efficient than raw depth; this tempers interpretation of EXP-011.
- **Time-local regularization** (`knowledge/papers/time-matters-regularization.md`): early regularization can preserve its benefit after removal, consistent with accepted mixup plus a hard-label tail.
- **RandAugment** (`knowledge/papers/randaugment.md`): a small policy space can add invariance, but CPU overhead and excessive strength remain risks.

No network search was performed because this autoresearch session is explicitly offline. Existing distilled papers and eleven completed experiments provide the evidence base.

## Experimental History Review

- The accepted WRN-16-2 plus alpha-0.2 mixup through 65% remains 94.07% at 141.9 passes. Its near-zero late training loss and accepted 0.2432 test loss show that generalization, not optimization throughput alone, is the current limiter.
- Additive regularization has repeatedly regressed: CutMix 93.72%, stronger mixup 93.57%, and residual dropout 93.52%. The accepted long hard-label tail and continuous weight decay/floor should remain intact unless a proposal directly targets their mechanism.
- Faster or denser training is not sufficient: BF16 reached 159.1 passes at 93.81%, cosine-to-zero reached 142.5 at 93.80%, and a local fused-SGD check failed its materiality gate.
- Low-resolution capacity is directionally positive but below the margin: `[32,64,160]` reached 94.11% at 132.2 passes and `[2,2,3]` reached 94.15% at 132.9. Exact settings are closed. The depth result's 0.2782 test loss despite tiny training loss argues against simply adding more raw capacity.
- The main untried gap is a low-overhead generalization mechanism acting on late iterates, with a secondary opportunity for a compute-efficient residual transform that changes the capacity/generalization tradeoff.

## Diagnosis

Accuracy is not limited by GPU memory, catastrophic under-training, or evaluation frequency. Accepted training already drives hard-label loss near zero within 300 seconds, while more exposure alone and several regularizers failed. The two capacity probes yielded diminishing positive deltas but no accepted gain; the deeper probe's worse test loss makes overconfident specialization a stronger explanation than insufficient convergence. EXP-012 should therefore prioritize variance/flatness control at evaluation or a genuinely more efficient representation, while preserving the accepted early-mixup and late-refinement recipe.

## Collected Ideas

- **Late whole-state EMA**: maintain a detached FP32 exponential moving average of parameters and floating BatchNorm buffers only after mixup ends at 65%, then evaluate the EMA state at each accepted cadence while training continues on live weights. This targets late SGD variance and the observed confidence gap with one cheap state update per optimizer step; the exact decay/start and swap semantics must be fixed before scoring.
- **Uniform last-window averaging**: accumulate a simple arithmetic mean of model parameters during the last 15% of counted time and evaluate the averaged state terminally. It maps closely to checkpoint averaging literature but makes intermediate `best_test_acc` incomparable unless averaging begins earlier, and BatchNorm statistics require an explicit policy.
- **Exact 8x8 bottleneck residual block**: append a pre-activation identity branch `BN-ReLU-1x1 128->64, BN-ReLU-3x3 64->64, BN-ReLU-1x1 64->128` after accepted stage 3. It adds about 53.8k parameters and 3.41M MACs/image, far less than EXP-011's full block, testing whether nonlinear refinement without large overcapacity preserves exposure and generalization.
- **Zero-initialize residual endpoints**: initialize every basic block's second BatchNorm scale to zero so the network starts close to its skip-path function and learns residual corrections gradually. This changes optimization geometry without adding compute or regularization, but the fixed 300-second schedule may spend too long opening branches.
- **Mild transition-only label smoothing**: use epsilon 0.05 cross-entropy from 65% to 85%, then restore exact hard labels for the final 15%. This directly targets overconfidence while preserving a clean terminal tail, but it adds another soft-target phase after mixup and conflicts with repeated over-regularization evidence.
- **Low-magnitude RandAugment**: apply torchvision `RandAugment(num_ops=1,magnitude=5)` before crop/flip only during training. It targets missing image invariances rather than optimization, but remains active in the useful hard-label tail and may reduce exposure through CPU transform cost.
- **True per-channel standardization**: replace unit standard deviations with canonical CIFAR-10 channel standard deviations while keeping mean subtraction. Better-conditioned inputs could help optimization, but the accepted LR was calibrated to the current scale, so this single edit implicitly changes effective step sizes.
- **SAM moonshot**: use sharpness-aware updates only during an early window to seek flatter minima, then return to accepted SGD. It directly targets generalization but roughly doubles forward/backward work during the window and introduces a perturbation radius; feasibility and attribution risk are high under 300 seconds.

## Combinations

- **Exact bottleneck + zero endpoint initialization**: make the new 8x8 branch start as an identity correction, plausibly retaining accepted optimization while adding efficient nonlinear capacity. The combination is stronger than an immediately active bottleneck if branch disruption caused EXP-011's higher test loss, but it mixes architecture and initialization and should be considered one fully specified block design only if isolated evidence supports it.
- **Late EMA + transition-only smoothing**: smoothing could reduce overconfident updates while EMA suppresses late iterate variance. The pair attacks complementary aspects of the generalization gap, but local history makes stacked target regularization risky and it would obscure whether averaging alone works.
- **Late EMA + exact bottleneck**: average the efficient capacity model to control its generalization. This could combine the positive capacity signal with flatness control, but it is an adaptive rescue of two unvalidated changes and is not appropriate before either component succeeds alone.

## Candidate Ideas

### Exact 8x8 Bottleneck Residual Refinement
**Summary**: Append one fixed pre-activation identity residual unit after accepted `layer3`: `BN-ReLU-1x1 128->64`, `BN-ReLU-3x3 64->64`, then `BN-ReLU-1x1 64->128`. Use accepted initialization and no endpoint zeroing. This adds exactly 53,760 parameters and 3,407,872 MACs/image while leaving all accepted stages, final BN, classifier, and training choices unchanged.

**What it targets**: The diagnosis shows directionally useful low-resolution capacity but worsening generalization from dense depth. A rank-64 residual correction tests whether constrained nonlinear refinement yields a better generalization-per-MAC tradeoff.

**Reasoning**: It uses only about 18% of EXP-011's added parameters/MACs and differs structurally from both closed width and full-block treatments. Its fixed reduce-transform-expand representation should preserve more accepted exposure; scoring requires at least 92% measured retention and 130.5 projected passes.

**Sources**: `proposals/idea-02.md`; `knowledge/papers/wide-residual-networks.md`; EXP-010 and EXP-011 analyses.

**Estimated Effort**: medium.

**Risk Assessment**: Three small sequential convolutions may cost more than static MACs predict, the half-width bottleneck may constrain useful features, and the immediately active random branch can still perturb a calibrated trajectory. No ratio, placement, or initialization rescue is allowed.

### Zero-Initialized Residual Endpoints
**Summary**: Keep the accepted WRN topology and Kaiming construction RNG consumption, then overwrite all six basic blocks' final `conv2.weight` tensors with zero. Do not zero `bn2`: in this pre-activation block, BN is followed by ReLU and literal zero scale/bias would make the branch permanently gradient-dead. The zero convolutions receive gradients on the first update and open upstream branch learning on the second.

**What it targets**: The diagnosis favors improved optimization geometry/generalization over added capacity. Starting from skip/projection paths makes residual corrections enter gradually without extra kernels, parameters, or explicit regularization.

**Reasoning**: The treatment preserves the exact accepted graph, RNG stream after construction, parameter count, and expected 141.9-pass exposure. Semantic tests can verify zero initial residuals, nonzero first-step endpoint gradients, and second-step upstream gradients. Scoring requires at least 97% retention and 135 projected passes.

**Sources**: `proposals/idea-03.md`; accepted `PreActBlock.forward`; EXP-010/011 analyses; `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: low.

**Risk Assessment**: The accepted schedule may rely on random residual features, and identity-biased initialization may change the early basin without improving generalization. Zeroing the wrong tensor is fatal, so detailed gradient tests are mandatory.

### Late Whole-State EMA
**Summary**: Preserve the accepted 691,674-parameter training path and initialize a detached FP32 EMA of parameters plus floating BatchNorm buffers on the first hard-label update at 65% counted time. Update it after every subsequent SGD step with decay 0.999, copy integral BN counters, and evaluate only the EMA state at existing post-initialization cadences through a fail-safe swap/restore. No live/EMA double evaluation or decay/window fallback is allowed.

**What it targets**: The diagnosis identifies late specialization and checkpoint variance as a remaining gap: accepted training fits nearly perfectly, while EXP-011 gained accuracy but worsened test loss. EMA suppresses short-horizon iterate and calibration noise without new target corruption or capacity.

**Reasoning**: The 65% boundary excludes mixed-target/high-LR training and leaves roughly 9,700 hard-label updates. Decay 0.999 gives an approximately 1,000-update e-folding horizon, short enough to track the improving tail while averaging meaningful SGD noise. A strict matched gate requires at least 95% throughput retention and 134.8 projected passes.

**Sources**: `proposals/idea-01.md`; `knowledge/papers/weight-averaging.md`; EXP-002 and EXP-011 analyses.

**Estimated Effort**: medium.

**Risk Assessment**: EMA may lag the improving live trajectory, and averaging BN running statistics is only an approximation for averaged weights. Swap/restore complexity must preserve parameter identities, optimizer references, live state, and one evaluation per epoch even on evaluator exceptions.

## Review

The blind offline critic selected the exact 8x8 bottleneck at 7.5/10 for both evidence and impact. I accept its central caveat: the two positive capacity probes support low-resolution transformation, but using only about 18% of EXP-011's added transform budget may erase rather than improve that signal. The experiment therefore tests one fixed rank-64 correction without claiming constrained capacity inherently generalizes better. Accepted initialization is mandatory; endpoint zeroing, EMA, ratio changes, placement changes, and adaptive fallbacks are excluded. Full feedback is in `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. The bottleneck has the strongest direct empirical bridge and a precise, materially distinct mechanism. EMA remains a rigorous future generalization candidate but its assumed late checkpoint variance is not locally demonstrated; zero initialization remains technically valid but weakly evidenced for this shallow network.

## Chosen Idea
**Selected**: Exact 8x8 Bottleneck Residual Refinement

**Why this idea**:
EXP-010 and EXP-011 independently produced positive deltas from low-resolution capacity, with the latter missing acceptance by only 0.02 points. A fixed post-stage-3 `128->64->64->128` residual correction is structurally distinct from both closed treatments and adds only 53,760 parameters / 3,407,872 MACs per image, preserving substantially more exposure while testing constrained nonlinear refinement.

**Hypothesis**:
The exact accepted-initialized rank-64 bottleneck will retain at least 92% matched production throughput, project at least 130.5 passes, realize at least 120 passes in one fixed-seed run, and improve `best_test_acc` from 94.07% to at least 94.17%. Any valid lower score is no-improvement and closes this exact ratio/placement/initialization only.
