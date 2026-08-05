# Brainstorm EXP-047
**Created**: 2026-07-27

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): convex example/label interpolation is the strongest validated regularization mechanism in this learner; moving its representation site is a replacement test, not an additive tweak.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): class-uniform target mass can reduce overconfidence, but stacking it with soft pairwise targets risks redundant entropy.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): any new target regularizer should preserve the accepted exact-label final35% rather than perturb boundary refinement.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): retain the accepted capacity allocation and focus on cheap pooled representation or training-law changes after high-resolution simplifications failed.

No network or new retrieval was used. This thorough pass uses the offline knowledge base, accepted source, measured system understanding, and46 completed experiments.

## Experimental History Review

- The accepted94.48% learner nearly interpolates its hard tail at130.304 passes; generalization and boundary quality remain limiting, while backpropagation consumes74% of counted steps. Favor state-free pooled or objective changes with negligible spatial cost.
- EXP041-045 close immediate auxiliary supervision, content/statistical pooling, gradient projection, classifier geometry, and shortcut anti-aliasing variants. Preserve exact GAP, sole refined-path CE, raw gradients, ordinary classifier, and learned downsampling.
- EXP046 semantically qualified mean crop fill but did not score because its prospective delay-free service gate was unstable even for accepted control. CPU transform accuracy remains unresolved, but that exact treatment and protocol rescue are closed; return to GPU-local work.
- EXP036 is the only recent improvement: a narrow bias-free ReLU residual MLP after GAP. Its activation is untested, but adjacent width/scale/init/readout search is disallowed absent a distinct activation mechanism.
- Accepted input mixup gained0.69 points; alpha0.2,65% duration, and batch-shared coherence are locally bracketed. Pooled feature interpolation and uniform smoothing remain unscored, but the former replaces a strong mechanism and the latter compounds it.
- The goal is not saturated: the frontier is below known CIFAR capacity and memory/wall headroom is large. The constraint is useful accuracy per backward pass, not an established statistical ceiling.

## Collected Ideas

- **Pooled-head SiLU** - replace only the state-free ReLU inside the accepted64-unit pooled residual bottleneck with default SiLU. It retains signed moderate negative contrasts and smooth gradients at almost zero cost, but also attenuates positive evidence and changes the sole successful recent component from step one.
- **Post-GAP feature mixup replacement** - stop mixing pixels early and instead mix128-dimensional pooled features immediately before the accepted nonlinear MLP with the same coefficient/permutation/targets. It targets representation linearity and clean spatial BN statistics without more convolution, but deletes the validated low-level input-mix mechanism.
- **Early epsilon0.05 label smoothing** - apply PyTorch-uniform smoothing only to both early mixup component CEs, retaining the exact hard tail. It adds class-wide entropy beyond pairwise targets nearly for free, though no calibration diagnosis supports the conventional epsilon.
- **One-time boundary momentum reset** - clear SGD momentum exactly on the first hard-label step so clean refinement is not pulled by soft-target velocity. It is free and causal, but the transition is not logged as unstable and only a short gap separates main-process mixup and exhausted-worker RandAugment changes.
- **Ghost BatchNorm** - compute training statistics over fixed virtual groups while preserving affine tensors and inference buffers. It may regularize batch256 features without convolution cost, but spatial samples already make estimates large and virtual size is arbitrary.
- **Pooled hidden centering** - subtract the batch mean from only the64-unit pooled-head preactivation during early training, then use exact inference/hard behavior. It could decorrelate corrections cheaply, but introduces batch dependence absent at evaluation and lacks evidence.
- **Early logit Z-penalty** - add a small fixed `logsumexp(logits)^2` term through65% to control log-partition drift while leaving class differences supervised by mixup. It is cheap but scale is arbitrary and adjacent to failed extra supervision.
- **True CIFAR channel scaling** - replace unit standard deviations with canonical per-channel standard deviations. This changes stem optimizer geometry while downstream preactivation BN largely absorbs scale, making benefit uncertain and re-entering CPU transform space after EXP046.
- **Low-rank cross-channel covariance moonshot** - expose a compact second-order pooled representation to an exact-neutral expert. It may add information beyond failed marginal dispersion, but rank/normalization and backward cost remain underdetermined.

## Combinations

- **SiLU + accepted residual scale0.1**: signed smooth corrections remain subordinate to exact GAP, which is safer than a backbone activation change. The direct path limits damage while retaining a distinct negative-evidence test.
- **Feature mixup + exact hard tail**: representation-linearity pressure acts only in the established early critical period, followed by ordinary examples for calibration. This is stronger than full-run manifold mixing but still replaces successful input invariance.
- **Label smoothing + exact hard tail**: class-uniform entropy disappears at the protected boundary. This temporal cross is safer than whole-run smoothing, though redundancy with early mixup remains.
- **Momentum reset + exact boundary**: optimizer state and target law change together at one causal event, potentially shortening adaptation. The cross is stronger than an arbitrary epoch reset but unsupported by observed loss transients.

## Candidate Ideas

### Early-Only Epsilon-0.05 Uniform Label Smoothing
**Summary**: Add PyTorch-uniform `label_smoothing=0.05` to both accepted component CEs only while input mixup is active, then retain literal accepted hard CE. Inputs/logits/state/RNG remain step-aligned with accepted before loss.

**What it targets**: Possible early overconfidence across classes beyond the one/two-class support of pairwise mixup, while protecting the exact-label final boundary-refinement phase.

**Reasoning**: Saved literature supports mild smoothing and time-limited regularization, and cost/state are negligible. The causal case is weakest: mixup already softens targets, alpha/duration are bracketed, epsilon is conventional, no calibration diagnostic exists, and EXP041 warns that added CE-derived pressure can hurt.

**Sources**: `knowledge/papers/label-smoothing.md`; `knowledge/papers/mixup.md`; `knowledge/papers/time-matters-regularization.md`; EXP004/005/020/035/041-046; `proposals/idea-03.md`.

**Estimated Effort**: low

**Risk Assessment**: Uniform mass may blunt useful early class motion. A valid miss closes immediate epsilon/convention/ramp/placement/confidence-penalty variants without a new calibration diagnosis.

### Early Post-GAP Pooled-Feature Mixup Replacement
**Summary**: Through exactly65%, replace input interpolation with one interpolation of the128-dimensional post-GAP vector immediately before the accepted nonlinear MLP, using the same batch-shared Beta draw, permutation, paired CE, and targets. The exact ordinary path remains for the final35% and evaluation.

**What it targets**: Linear behavior in the learned decision representation at negligible spatial cost while keeping every spatial BN on independently augmented real inputs.

**Reasoning**: EXP002 validates the coefficient/target/time prior, EXP027 validates early invariance plus added capacity, and EXP036 validates nonlinear pooled remapping. The treatment has substantial upside but bundles clean spatial BN with feature-line interpolation and removes the strongest validated input regularizer without a saved manifold-mixup source.

**Sources**: `knowledge/papers/mixup.md`; `knowledge/papers/time-matters-regularization.md`; EXP002/004/005/015/020/027/035/036/041/046; `proposals/idea-02.md`.

**Estimated Effort**: high

**Risk Assessment**: A late mixture may be too easy/off-manifold and cannot replace low-level mixed-pixel invariance. One score cannot deconfound BN statistics from placement; a miss closes earlier/later/random/compound feature-mix rescues.

### Default SiLU in the Accepted Pooled Residual MLP
**Summary**: Replace only the state-free `nn.ReLU()` between accepted pooled-head matrices with default non-inplace `nn.SiLU()`. Preserve every parameter byte, seed36036 initialization, GAP/direct path, residual scale0.1, classifier, objective, optimizer, data, and schedule.

**What it targets**: Representation efficiency in the cheap64-unit post-GAP bottleneck. Signed `W1` contrasts can be negative even though pooled inputs are nonnegative; SiLU retains a bounded negative lobe and smooth near-zero gradients that ReLU deletes.

**Reasoning**: EXP036 directly validates this narrow nonlinear placement and system profiling puts the head at only1.4% of forward cost. The treatment adds no state/spatial work and avoids worker risk. Evidence remains weak because it is adjacent tuning of the sole successful head, attenuates every finite positive activation, and signed `W2` plus direct GAP may already express inhibitory corrections.

**Sources**: EXP036/041-046; accepted `train.py`; `proposals/idea-01.md`.

**Estimated Effort**: low

**Risk Assessment**: Startup branch amplitude and every gradient change immediately under ReLU-calibrated weights. A miss closes SiLU and immediate signed/smooth activation, gain, init, scale, width, and activation-comparison rescues.

## Review

The offline critic selected pooled-feature mixup at3/5 evidence and4/5 impact. I adopt its corrections: the score tests the complete bundled replacement, not manifold linearity or clean BN independently; most interpolation passes through an affine direct path and only the scale0.1 MLP supplies downstream nonlinearity; training-only Jensen/gradient/BN diagnostics remain descriptive; and a miss falsifies only fixed post-GAP/pre-MLP replacement. Earlier/later placements are declined as search policy, not empirically disproven. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's high-upside pick. Feature mixup changes the complete early training geometry while preserving coefficient/target/time laws and near-normal exposure. Default SiLU reopens an unchanged candidate previously rejected without the missing hidden-region diagnosis. Label smoothing remains redundant and twice deprioritized without calibration evidence.

## Chosen Idea
**Selected**: Early Post-GAP Pooled-Feature Mixup Replacement

**Why this idea**:
During exactly the accepted first65%, feed individually augmented inputs through the exact spatial backbone, interpolate only the128-dimensional post-GAP vector using the accepted batch-shared Beta draw and permutation, then apply the accepted nonlinear pooled MLP/classifier and paired CE. Preserve the exact ordinary hard/evaluation path, all state, optimizer, schedule, and worker behavior. This is a fixed bundled test of clean spatial BN plus pre-MLP feature interpolation replacing mixed pixels.

**Hypothesis**:
If the accepted convex-label prior is more useful at the learned decision representation than at raw pixels, the complete fixed post-GAP/pre-MLP replacement will retain at least127 realized passes and raise fixed-seed `best_test_acc` from94.48% to at least94.58%. A normal-exposure miss falsifies only this exact bundled replacement; other placements are declined without claiming they were tested.
