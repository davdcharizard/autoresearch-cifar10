# Brainstorm EXP-013
**Created**: 2026-08-06

## Web Search & Literature Review

- **PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions** (`papers/poly-loss.md`; `knowledge/papers/polyloss.md`; https://openreview.net/forum?id=gSdSJoenupI)
  Poly-1 adds `epsilon*(1-p_t)` to cross-entropy and explicitly supports soft targets. For one-hot softmax CE its per-example gradient is rescaled by `1+epsilon*p_t`; with CutMix's two-label soft target the polynomial gradient is coherent but not generally collinear with mixed CE, so the proposal must audit both hard and mixed behavior.
- **Temperature in Cosine-based Softmax Loss** (`papers/temperature-cosine-softmax.md`; ICCV 2025)
  Cosine classifiers are highly scale-sensitive. A fixed scale of 40 improved ResNet-34/CIFAR-10 accuracy from 95.56 to 95.85 in the reported setup, but the paper's learned-scale method reached 95.49; any transfer therefore needs one preregistered scale and no validation search.
- **Dual Focal Loss for Calibration** (`papers/dual-focal-loss.md`; ICML 2023)
  Calibration improvement is not sufficient evidence for top-1 accuracy: WideResNet-26-10/CIFAR-10 error was 3.96% with Dual Focal versus 3.86% with plain cross-entropy. This weakens calibration-only auxiliaries as the next bet.
- **How to Scale Your EMA** (`knowledge/papers/how-to-scale-your-ema.md`; NeurIPS 2023) and **Averaging Weights Leads to Wider Optima and Better Generalization** (`knowledge/papers/stochastic-weight-averaging.md`; UAI 2018)
  EMA behavior should be parameterized by exposure horizon, while useful weight averaging requires a trajectory diverse enough to offset old-iterate lag. EXP-011's time-derived full-state EMA is sound, but its 95.49 plateau leaves the horizon/lag operating point unresolved.

## Experimental History Review

- The successful lineage is BASE 91.51 -> WRN EXP-001 94.62 -> CutMix EXP-002 95.23 -> clean period-two SAM EXP-004 95.40 -> full-state charged-time EMA EXP-011 95.61. Gains have narrowed, and EXP-011's final-16 EMA mean is 95.493125 rather than a stable 95.61 level.
- EXP-011 retains 25,798 steps at only 1,222 MiB, so memory is abundant; ordinary/SAM steps are about 10/20 ms, making extra forwards expensive. Sparse state/loss/classifier changes remain feasible (`02-system-understanding.md`).
- EXP-012 is the only child of EXP-011. Full-probability size-16 Cutout on the early non-CutMix complement reached 95.52 with a 95.418 tail and missed its dose floor at 25,376 steps. Do not repeat this spatial-erasure package or assume paired latency guarantees realized dose.
- Earlier failures reject identity-halving overlap batches, replacing validated CutMix with low-dose manifold mixup, uncalibrated literature-scale ASAM, dual-view CPU RandAugment, multi-launch SE gates, and back-loaded equal-MAC depth. They do not reject a one-line soft-target loss, time-horizon EMA refinement, or single-matmul classifier geometry.
- The primary limiter is stable generalization above the EMA plateau, not memory or raw optimization throughput. A child must clear 95.71 and should plausibly lift the stable tail by about 0.25 points; sub-0.30 selected maxima require careful plateau context.

## Collected Ideas

- **Bounded soft-target Poly-1** - Replace both hard-label and CutMix cross-entropy terms with one coherent Poly-1 objective using target-weighted `p_t`. Choose one epsilon from a preregistered hard-target inflation budget, separately characterize the non-collinear two-label CutMix gradient, and use the identical loss on both SAM passes. This attacks confidence-weighted optimization with negligible compute and no new data path; coefficient transfer and overlap with CutMix soft labels are the main risks.
- **Shorter-lag charged-time EMA** - Keep cadence 31 and full-state ownership but shorten the time half-life so the averaged model tracks the rapidly improving final tail more closely. This directly targets old-state bias in EXP-011's 18.75-second horizon without changing online training or adding a forward. Too little smoothing may collapse back toward noisy live weights, and live-tail accuracy is unobserved.
- **Fixed-temperature cosine classifier** - Normalize pooled features and classifier rows, then apply one fixed scale chosen from direct CIFAR evidence rather than learned or validation-tuned temperature. This changes margin geometry with one normalized classifier operation and negligible parameter growth. Scale sensitivity, optimizer dynamics, and incompatibility with the parent's Kaiming-linear geometry are substantial risks.
- **Dual shared-backbone classifier heads** - Add a second independently initialized linear head, train both with the same targets, and average logits during the ordinary training/evaluation forward. The extra 2,570 parameters and classifier matmul are cheap, while head diversity may reduce class-boundary variance. Shared features may make the heads nearly identical, and changing initialization consumption must be isolated.
- **Multi-rate full-state EMA** - Use a shorter horizon for the classifier and a longer/current horizon for the backbone and BatchNorm state. This targets faster classifier adaptation while preserving representation smoothing and costs only different lerp coefficients. The split is weakly evidenced, creates state-group complexity, and risks incoherent feature/classifier averaging.
- **Earlier drop-path simplification** - Begin decaying stochastic depth before 75% while preserving CutMix and the SAM/EMA tail, testing whether the mature WRN is over-regularized before its clean finish. This removes work conceptually rather than adding another regularizer, but prior success supports strong early regularization and there is no direct evidence for the new boundary.
- **Gradient centralization before SGD/SAM** - Subtract per-output-channel mean from convolutional/linear gradients before optimizer and SAM norm construction. This is a fused representation-geometry intervention with no forward, but it changes SAM's perturbation geometry and adds many reduction kernels, making causal isolation and throughput uncertain.
- **Switch-EMA moonshot** - At a sparse, preregistered late cadence, copy the EMA state back into the online model and continue optimizing so the trajectory repeatedly leaves sharp endpoints. It uses existing shadow state and no extra forward, but it disrupts momentum/BN consistency and has weak evidence under a short 75-second tail.

## Combinations

- **Poly-1 + shorter-lag EMA**: Poly-1 could raise the underlying classifier trajectory while a shorter EMA reduces lag, plausibly improving both optimization and checkpoint selection. The cross is stronger only if each effect is small and orthogonal; as a first test it is too confounded and should follow single-lever evidence.
- **Cosine classifier + Poly-1**: normalized margin geometry could make the confidence-dependent Poly-1 rescale better behaved, but both alter logits and the effective optimum scale. The interaction creates two sensitive coefficients and is less defensible than either alone.
- **Dual heads + multi-rate EMA**: independently averaged heads could preserve classifier diversity atop one smoothed representation at very low MAC cost. It is plausible as a future ensemble-like branch, but state ownership and evaluation semantics are much more complex than the current single-head package.

## Candidate Ideas

### Six-half-life responsive full-state EMA
**Summary**: Change only `EMA_TAIL_HALF_LIVES` from 4 to 6, shortening the charged-time half-life from 18.75 to 12.5 seconds while keeping start 0.75, cadence 31, full parameter/buffer inventory, online training, and swap/restore routing exact (`proposals/idea-02.md`).

**What it targets**: Potential old-state lag in the successful EMA summary. The candidate puts 64.6% rather than 50% of terminal weight on the most recent 18.75 seconds while still averaging an effective roughly 77 samples.

**Reasoning**: This is the cleanest test of EXP-011's own unexplored horizon question and should be effectively compute-neutral. Charged-time parameterization is robust to throughput, and the parent's final-16 mean/final below its maximum are compatible with, though do not prove, lag.

**Sources**: `proposals/idea-02.md`; `knowledge/papers/how-to-scale-your-ema.md`; `knowledge/papers/stochastic-weight-averaging.md`; `experiments/011/04-analysis.md`.

**Estimated Effort**: low.

**Risk Assessment**: No live-tail control exists, so lag is unidentified. A shorter horizon may simply remove useful smoothing, and the expected accuracy effect may be smaller than known max/tail noise. This primarily changes the reported EMA model rather than the online trajectory, so causal claims must stay narrow.

### Fixed-scale-40 cosine-normalized classifier
**Summary**: Preserve the existing initialized `nn.Linear` state shape but compute logits as `40 * cosine(normalized pooled feature, normalized classifier row)`, freeze and ignore the zero bias, and run the final normalization/matmul in FP32. Keep CutMix, SAM, EMA, and all backbone behavior unchanged (`proposals/idea-03.md`).

**What it targets**: Class-boundary geometry rather than another input regularizer. Removing radial logit degrees of freedom and using an evidence-anchored scale could lift the stable plateau at negligible model MAC cost.

**Reasoning**: ICCV 2025 reports 95.85% for fixed scale 40 versus 95.56% standard softmax on ResNet-34/CIFAR-10, an effect sized to clear this goal's threshold. Construction parity can preserve all parent's raw initial tensors and RNG; only the forward geometry changes.

**Sources**: `proposals/idea-03.md`; `papers/temperature-cosine-softmax.md`; ICCV 2025 Temperature in Cosine-based Softmax Loss.

**Estimated Effort**: medium.

**Risk Assessment**: Scale transfer is highly sensitive; normalization removes useful feature norms, makes classifier weight decay radial, and changes raw-weight SAM/EMA geometry. FP32 reduction launches may lower dose, and sharp logits may conflict with early CutMix soft targets.

### Bounded soft-target-aware Poly-1
**Summary**: Add `CE + 0.25*(1-p_t)` to every primary and SAM loss, using the existing area-corrected CutMix lambda to form target-weighted probability without a dense target. The fixed `epsilon=0.25` comes from a maximum 1.25 hard-label/constituent gradient multiplier, not a metric sweep. Use identical loss semantics on both SAM passes and preserve the full EXP-011 EMA/evaluation path (`proposals/idea-01.md`).

**What it targets**: Confidence-weighted decision-boundary refinement on every training example, addressing the stable 95.493 EMA plateau with no additional model pass or data transport (`02-system-understanding.md`).

**Reasoning**: Poly-1 has direct image-classification evidence and explicit soft-target support. For hard labels it is a bounded confidence-dependent CE rescale; for two-label CutMix its exact gradient is separately auditable and coherent, though not generally collinear with mixed CE. Its tiny 10-logit FP32 softmax should preserve realized dose better than EXP-012's augmentation kernels.

**Sources**: `proposals/idea-01.md`; `papers/poly-loss.md`; `knowledge/papers/polyloss.md`; ICLR 2022 PolyLoss.

**Estimated Effort**: low code, medium verification.

**Risk Assessment**: The stability-derived coefficient may be too weak for 95.71, while confident-example emphasis may be counterproductive. CutMix cancellation means the 1.25 constituent bound does not bound the combined soft-gradient ratio, and Poly-1 changes both SAM direction and update.

## Review

Claude's blind adversarial review selected the fixed-scale-40 cosine classifier (`01-idea-review.md`). It judged Poly-1's accuracy-blind `epsilon=0.25` underpowered and likely to resemble a near-uniform tail LR increase, and found the six-half-life EMA premise unidentified and its expected effect inside checkpoint noise. Cosine classification was the only candidate with direct effect-size evidence matching the diagnosed plateau gap.

Three concerns materially refine the choice. First, scale 40 is plausibly the selected maximum of the paper's scale sweep, so the expected transfer is reduced from +0.19-0.39 to roughly +0.10-0.25 and the result will not be described as an unbiased literature expectation. Second, this WRN's 256-dimensional feature space has wider random cosine spread than the paper's 512-dimensional ResNet-34 space, making scale 40 effectively sharper; planning must include an accuracy-blind paired early-optimization trace on identical batches to expose slow-start or CutMix conflict. Third, radial weight decay can shrink raw classifier norms while normalized logits stay invariant, so online row norms must be recorded per epoch and EMA/online norm/direction diagnostics at terminal time. These are observability requirements only: no norm mitigation, alternate scale, or retry is allowed after measurement.

## Idea Evaluation

Adopt Claude's pick. The cosine classifier scored 7/10 for evidence/reasoning and 8/10 for potential impact, versus 6/3 for bounded Poly-1 and 5/3 for the EMA horizon. Its transfer risk is higher, but a clean result is more informative and its same-dataset fixed-scale evidence is the only finalist evidence plausibly large enough to clear both the formal 95.71 threshold and the stable-tail limiter.

## Chosen Idea
**Selected**: Fixed-scale-40 cosine-normalized classifier

**Why this idea**:
It targets class-boundary geometry rather than adding another spatial regularizer or changing checkpoint smoothing. The implementation can preserve parent initialization/state names and all data/SAM/EMA RNG streams while adding only small FP32 vector normalizations and a 10-way matmul. Scale 40 is a single preregistered, same-dataset literature point, accepted with explicit selection-bias and dimension-transfer caveats and no local scale search.

**Hypothesis**:
On the single fixed-seed GPU-0 run, scale-40 cosine softmax will reach `best_test_acc >=95.71%` versus EXP-011 at 95.61 while retaining at least 25,300 optimizer steps and intact CutMix/SAM/EMA audits. A mechanism-supporting result additionally raises the final-16 EMA mean from 95.493125 to at least 95.64; the realistic preregistered best range is 95.71-95.86 after discounting the paper's +0.29 scale-selected result. Below 95.71 is one no-improvement with no alternate scale or retry.
