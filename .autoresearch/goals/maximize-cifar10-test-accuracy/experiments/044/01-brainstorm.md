# Brainstorm EXP-044
**Created**: 2026-07-27

## Web Search & Literature Review

- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): mild uniform target mass can reduce overconfidence, but stacking it with another soft-target method requires calibration.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): regularization is most defensible early, followed by clean convergence; preserve the accepted hard-label tail.
- **mixup** (`knowledge/papers/mixup.md`): the accepted early target interpolation already supplies example-aware softness, weakening the incremental case for uniform smoothing.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): preserve the compute-effective spatial backbone and place any new representation work after fixed spatial reduction.

No network, remote source, or new retrieval was used. This thorough pass is offline and uses only the persistent knowledge base, accepted source, measured system understanding, and 43 completed experiment records.

## Experimental History Review

- EXP036 remains the 94.48% frontier with 94.45% final accuracy, 0.2456 loss, and 130.304 passes. Its scale-0.1 post-GAP `128->64->128` residual MLP is the sole recent improvement and must remain the base.
- The accepted hard tail nearly interpolates training while test loss remains 0.2456, so generalization and boundary quality still limit the metric. Backward consumes about 74% of a step and only 2.536% throughput can be lost before projected exposure falls below 127 passes; fixed reductions and pooled-scale work are preferable to learned spatial compute.
- EXP041-043 form a consistent warning: auxiliary CE, adaptive pooling, and convolution-gradient projection each retained at least 127 passes but worsened both top-1 and loss. Preserve sole refined-path CE, uniform GAP as the direct statistic, ordinary classifier/optimizer geometry, and all raw learned directions.
- EXP042 rejected one learned content-weighted pooling rule, not every statistic that GAP discards. A fixed permutation-invariant dispersion statistic supplied only to the subordinate accepted MLP is distinct: it leaves the mean direct path and startup function exact and has no learned spatial weighting.
- Label smoothing is untested but weakly diagnosed. Accepted early mixup already creates soft targets, while the 35% one-hot tail is validated by EXP002/004/020. If scored, only mild early-only smoothing preserves that temporal contract, yet it still compounds regularization.
- Canonical training standardization is structurally mismatched because the frozen evaluator remains mean-only/std-one. A neighboring local run of the exact tuple lost 1.79 points, and preactivation BN absorbs much of the intended scale conditioning; it should not lead.
- The goal is not saturated: 94.48% is materially below known CIFAR model capacity and the fixed final 8x8 map still contains distributional information beyond its mean. The challenge is adding usable information without disrupting the accepted invariant path or exposure.

## Collected Ideas

- **Exact-neutral spatial-dispersion input** - compute per-channel population standard deviation of the final post-BN/ReLU 8x8 map and feed it through one zero-initialized bias-free `128->64` adapter into the accepted pooled MLP hidden preactivation. GAP remains exact as the direct path; this targets fixed second-order information discarded by the mean with 8,192 parameters and no learned spatial kernel.
- **Early-only epsilon-0.05 label smoothing** - add PyTorch-uniform smoothing to both accepted mixup cross-entropies through exactly 65%, then preserve the one-hot hard tail. It targets overconfidence at negligible cost, but uniform softness is redundant with mixup and no calibration diagnostic identifies it as the remaining error.
- **Anti-aliased transition downsampling** - replace each stride-2 residual/projection sampling operation with fixed low-pass pooling followed by stride-1 convolution. It could compose with accepted image invariance by reducing shift aliasing at similar convolution output shapes, but changes two high-value transitions, adds pooling backward, and lacks local evidence.
- **Fixed training-channel standardization** - replace training std ones with `(0.2470,0.2435,0.2616)` while leaving means and all other transforms exact. It is nearly free and changes stem conditioning, but the immutable evaluator uses std ones, creating a train/eval scale mismatch that strongly dominates its prior.
- **Second independently initialized pooled residual expert** - add another cheap `128->64->128` residual MLP and average its correction with the accepted head. It could reduce representation variance at pooled cost, but is adjacent width/capacity tuning around EXP036 and arbitrary scaling/initialization would confound the mechanism.
- **One-time hard-boundary momentum reset** - delete all Nesterov buffers immediately before the first hard-label update. It cleanly targets objective-transition state without inference or sustained compute, but inherited velocity decays below 1% within 44 updates and no transition instability is observed.
- **SiLU final-head activation** - replace only the accepted pooled MLP ReLU with SiLU, preserving the spatial backbone and parameter count while allowing smooth signed hidden responses. It is a cheap representation change, but replaces a component of the successful head with no local diagnosis and changes startup globally.
- **Decoupled matrix decay** - move accepted `5e-4` shrink outside momentum accumulation while retaining SGD/Nesterov. It can separate norm control from velocity, but matching effective regularization is ambiguous and adjacent classifier-decay plus optimizer-geometry results provide no positive evidence.
- **Low-rank covariance readout moonshot** - compute a compact cross-channel covariance sketch from the final map and project it into the accepted hidden layer. It could capture co-occurrence absent from mean/std, but stabilization and rank choices are arbitrary and covariance backward threatens the protected exposure regime.

## Combinations

- **Dispersion statistic + accepted pooled MLP**: inject zero-started standard-deviation features into the existing 64-unit hidden preactivation while retaining GAP as the residual direct path. This is stronger than replacing GAP or adding a separate classifier because it lets the already validated nonlinear remap combine mean and spread without perturbing startup.
- **Early label smoothing + accepted temporal mixup removal**: apply the mild uniform prior only while mixup is active and return to exact hard CE at 65%. This is stronger than whole-run or tail-only smoothing because it preserves the validated clean refinement phase, though redundancy remains its main weakness.
- **Anti-aliased transitions + early RandAugment**: fixed low-pass downsampling may turn learned augmentation invariance into more stable intermediate sampling. The pair could beat either alone by aligning data and architectural shift robustness, but it also risks compounding information loss like prior masking treatments.

## Candidate Ideas

### Early-Only Epsilon-0.05 Label Smoothing on Accepted Mixup
**Summary**: Add `label_smoothing=0.05` to both cross-entropies in the accepted early mixup branch through exactly 65%, then return to exact accepted hard-label CE for the final 35%. The resulting target is `(1-0.05)*y_mix + 0.05/10`, with no model, RNG, data, optimizer, or hard-tail change.

**What it targets**: The generalization gap and potential overconfidence created while learning interpolated examples, without sacrificing the validated one-hot boundary-refinement tail.

**Reasoning**: Offline label-smoothing literature supports mild uniform target entropy, and time-local regularization supports early removal. This exact timing is the only version compatible with local mixup-duration evidence, but it still stacks uniform softness on already soft example-aware targets and epsilon 0.05 lacks a local bracket. Full contract: `proposals/idea-01.md`.

**Sources**: `knowledge/papers/label-smoothing.md`; `knowledge/papers/time-matters-regularization.md`; `knowledge/papers/mixup.md`; EXP002/004/020/041; `proposals/idea-01.md`.

**Estimated Effort**: low

**Risk Assessment**: Redundant regularization may blunt useful class motion, near-zero train loss does not diagnose overconfidence, and the fixed epsilon is convention-shaped rather than locally measured. A normal-exposure miss closes immediate epsilon/timing/convention variants as anti-tuning policy.

### Exact-Neutral Spatial-Dispersion Input to the Accepted Pooled MLP
**Summary**: Preserve accepted GAP and the complete scale-0.1 pooled residual head, but compute per-channel population standard deviation of the final post-BN/ReLU map and add a zero-initialized bias-free `Linear(128,64)` adapter to the existing hidden preactivation. The adapter adds 8,192 decayed parameters for a total of 1,011,674; at zero it preserves accepted pooled features, logits, common gradients, and RNG while receiving a generally nonzero first data gradient.

**What it targets**: The fixed final mean discards whether each channel's evidence is spatially diffuse or concentrated. The candidate exposes that invariant distributional statistic without learned spatial work or replacing the accepted mean direct path.

**Reasoning**: EXP036 validates exactly this post-reduction nonlinear placement, while EXP042 changed uniform pooling itself and lost 0.68 points. Standard deviation is permutation-invariant, can distinguish equal-mean feature maps, and enters only the subordinate MLP. The zero start corrects the earlier unscored active identity-start proposal's main confound. Full contract: `proposals/idea-03.md`.

**Sources**: `02-system-understanding.md`; `experiments/036/04-analysis.md`; `experiments/041/04-analysis.md`; `experiments/042/04-analysis.md`; `experiments/042/01-idea-review.md`; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: After BatchNorm/ReLU, channel mean and standard deviation may be strongly correlated, making the branch redundant. Variance/sqrt backward adds small kernels near the exposure limit, and epsilon/statistic/placement are fixed one-shot choices. A miss closes this exact branch and immediate variance/RMS/max/epsilon/width/scale/startup rescues.

### Fixed CIFAR-10 Per-Channel Training Standardization
**Summary**: Change only `make_train_transform` standard deviations from `(1,1,1)` to fixed `(0.2470,0.2435,0.2616)`, retaining transform order, means, RNG, model, optimizer, and training. The frozen evaluator necessarily remains std-one.

**What it targets**: Unequal RGB scale and conditioning in the unnormalized stem at essentially zero counted GPU cost.

**Reasoning**: Canonical channel scaling is simple and does not delete raw gradients, but the evaluator mismatch means training BN statistics see roughly four-times-scaled inputs while test inputs do not. Preactivation BN absorbs much of the prospective conditioning benefit, and a neighboring exact-tuple local run regressed 1.79 points. Full contract: `proposals/idea-02.md`.

**Sources**: accepted `train.py`/`prepare.py`; EXP043; neighboring local `../v2.9.0-opus-4-6/.autoresearch-dep-v2.9.0/reports/exp-report-030.md`; `proposals/idea-02.md`.

**Estimated Effort**: low

**Risk Assessment**: Immutable train/eval scale mismatch is structural, the statistic convention is arbitrary, and direct neighboring evidence is strongly adverse. This should rank last and be rejected if either alternative remains executable.

## Review

The offline adversarial reviewer selected the exact-neutral spatial-dispersion input at 3/5 evidence and 3/5 impact. I adopt its cautions: EXP036 validates placement but does not diagnose dispersion; post-BN/ReLU mean and std may be redundant; `sqrt(1e-5)` gives constant channels a small nonzero adapter feature; and variance backward must pass measured H20 timing. Preflight will prove zero adapter bytes, accepted startup/logits/common gradients, nonzero adapter gradients in both regimes, and fresh/preseeded updates while reporting mean/std correlation, epsilon-floor scale, and branch contribution without gates or tuning. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. Dispersion is the only finalist that preserves the accepted function and all common first-step learning while opening a genuinely new invariant representation signal through a real data gradient. Early label smoothing adds no information and redundantly softens mixup; training-only standardization is structurally mismatched to the frozen evaluator and directly opposed by neighboring evidence. Selection remains conditional on semantic and timing qualification.

## Chosen Idea
**Selected**: Exact-Neutral Spatial-Dispersion Input to the Accepted Pooled MLP

**Why this idea**:
Preserve the accepted mean vector as the complete direct path, compute per-channel population standard deviation of the same final 8x8 map, and feed it through one zero-initialized bias-free `128->64` adapter into the accepted pooled MLP hidden preactivation. At startup, the adapter contribution and backbone derivative are zero while the adapter receives `dL/dD=(dL/da)^T sigma`; this isolates one fixed missing statistic with 8,192 parameters and no learned spatial weighting, objective change, classifier constraint, or raw-gradient deletion.

**Hypothesis**:
If per-channel spatial dispersion contains useful boundary information not recoverable from the accepted global mean, the exact zero-start population-standard-deviation adapter will retain at least 127 projected and realized passes and raise fixed-seed `best_test_acc` from 94.48% to at least 94.58%. Final accuracy/loss are descriptive only. A valid normal-exposure miss closes this exact branch and immediate variance/RMS/max/epsilon/width/scale/initialization/placement rescues as experiment policy.
