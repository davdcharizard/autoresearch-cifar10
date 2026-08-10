# Brainstorm EXP-015
**Created**: 2026-08-06

## Web Search & Literature Review

- **PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions** (`knowledge/papers/polyloss.md`; https://openreview.net/forum?id=gSdSJoenupI)
  Poly-1 adds a target-confidence-dependent term with negligible forward cost. Its coefficient is task-dependent, and CutMix requires a coherent soft-target definition rather than treating two constituent CEs inconsistently.
- **When, Where and Why to Average Weights?** (`knowledge/papers/when-where-why-average.md`; https://proceedings.mlr.press/v267/ajroldi25a.html)
  Modern cross-workload evidence supports averaging with annealed learning rates, but EXP-011 already has a successful full-state EMA. Any variant must distinguish actual kernel mass, state age, and BatchNorm semantics.
- **Lookahead Optimizer: k steps forward, 1 step back** (`papers/lookahead-optimizer.md`; https://papers.nips.cc/paper_files/paper/2019/hash/90fd4f88f588ae64038134f1eeaa023f-Abstract.html)
  NeurIPS evidence shows a standard-optimizer wrapper can reduce trajectory variance with one parameter copy and amortized interpolation, including CIFAR gains at common `k=5`, `alpha=0.5`. Its slow path may overlap EXP-011's evaluation EMA.
- **Existing goal knowledge** (`knowledge/README.md`)
  Label smoothing can overlap mixed targets; SAM is already compute-aware and validated; stochastic residual regularization, output calibration, and low-cost averaging remain plausible only with exact dose and state audits.

## Experimental History Review

- The successful chain is BASE `91.51` -> time-aware WRN EXP-001 `94.62` -> CutMix EXP-002 `95.23` -> clean period-two SAM EXP-004 `95.40` -> charged-time full-state EMA EXP-011 `95.61`. EXP-011's final-16 EMA mean is `95.493125`, so the unresolved target is a stable roughly 0.2-0.3-point generalization lift, not memory or gross optimization failure.
- Three children of EXP-011 failed through distinct mechanisms. EXP-012 complementary Cutout reached `95.52` with a `95.418` tail and slight dose shortfall; EXP-013 fixed-scale cosine geometry completed full dose but fell to `95.11`; EXP-014's fixed width-320 taper was rejected before accuracy at `1.160975x` weighted step latency. Do not retry erasure, fixed-scale cosine normalization, or conditional width fallback.
- The current recipe already uses two trajectory mechanisms: Nesterov/SAM online optimization and a sparse evaluation EMA. A third averaging mechanism must show why it changes useful bias/variance rather than simply over-smoothing the same states.
- Fixed-time headroom favors scalar loss work or sparse fused state updates. Extra forwards reduce exposure, CPU multi-view augmentation exhausted loader headroom, and multi-launch channel modules failed latency gates (`02-system-understanding.md`; goal learnings).
- Untried gaps include coherent soft-target loss shaping, online variance reduction with exact optimizer-state semantics, bias-corrected or blended EMA evaluation, classifier-only diversity regularization without normalization, and symmetry-aware representations.

## Collected Ideas

- **Bounded soft-target Poly-1** - Replace every hard/CutMix CE with one coherent `CE(q,p) + epsilon*(1-q dot p)` objective using a single preregistered epsilon. This directly targets decision-loss weighting at negligible compute and preserves every image/update; exact soft-target calculus and a gradient-inflation audit keep it from becoming an implicit uncontrolled LR change.
- **Default Lookahead around Nesterov** - Maintain one slow parameter copy and every five optimizer updates interpolate it halfway toward online weights, then copy back while retaining momentum. This imports a CIFAR-tested variance-reduction method with no extra forward, but it changes every online trajectory and may double-average with EMA or interfere with late SAM.
- **Bias-corrected EMA/live endpoint blend** - At evaluation, use a fixed convex blend of the existing charged-time EMA and restored live state, including floating BN buffers, while copying integer counters by a preregistered policy. It targets the 95.49 stable plateau if EMA lag is over-smoothing, but a blend coefficient lacks direct evidence and mixing BN statistics may be invalid.
- **Shorter charged-time EMA half-life** - Keep cadence, support, and full-state mechanics but halve the 18.75-second half-life to emphasize the lower-LR endpoint. This is operationally clean and almost free, yet it is exposed parameter tuning with only one parent trajectory and may reduce the variance benefit that produced EXP-011's gain.
- **Classifier-row decorrelation penalty** - Add a small Gram off-diagonal penalty on the ten affine classifier rows while preserving feature/logit scale. EXP-013 observed severe directional correlation under cosine normalization, so a cheap affine-only diversity term could improve margins without changing the backbone; no parent row-correlation baseline or coefficient effect-size evidence currently exists.
- **Late-only confidence penalty** - Add entropy-based output regularization only after CutMix ends, avoiding overlap with mixed targets and focusing the clean decision-boundary phase. It is cheap and separable from early augmentation, but touches only one quarter of time and may conflict with SAM or suppress useful late confidence.
- **ShakeDrop-style residual noise** - Replace binary drop path with signed/scaled residual perturbations using the existing six block gates and replay the exact CUDA state for SAM. Literature supports deep residual regularization, but this compact six-block model and already strong CutMix make published transfer uncertain; backward-rule fidelity and BF16 stability raise implementation risk.
- **Reflection-equivariant stem moonshot** - Duplicate early features under horizontal reflection with shared kernels and fuse before the existing middle stage, targeting sample efficiency through a true CIFAR symmetry. It has high representational upside but changes activation traffic, initialization, and augmentation interactions and needs a strict latency gate after EXP-014.

## Combinations

- **Poly-1 + shorter EMA**: confidence-dependent training geometry could lift the online basin while a shorter EMA follows its endpoint more closely. The combination plausibly addresses both decision boundaries and lag, but two uncalibrated coefficients destroy attribution and should not be the first test of either.
- **Lookahead + existing EMA**: online slow-weight interpolation can reduce optimizer variance while charged-time EMA ensembles the resulting lower-variance trajectory. This might stabilize the selected tail more than either alone, but the nested exponential kernels could over-smooth and attenuate SAM, so the interaction needs more evidence before selection.
- **Affine classifier decorrelation + Poly-1**: Poly-1 reweights examples by confidence while a row-Gram penalty directly separates class directions without cosine normalization. The levers are complementary, but simultaneous loss terms add coefficient ambiguity and EXP-013 does not establish that parent affine rows are correlated.

## Candidate Ideas

### Canonical Lookahead Around Nesterov
**Summary**: Wrap every Nesterov update in canonical parameter-only Lookahead with fixed `k=5`, `alpha=0.5`, retaining momentum. Interpolate and copy back after optimizer/SAM restoration and before cadence-31 EMA sampling; never interpolate BN buffers or add a Lookahead evaluation source.

**What it targets**: Online optimizer variance and convergence stability without another forward, while retaining the successful downstream EMA (`02-system-understanding.md`).

**Reasoning**: NeurIPS evidence shows CIFAR gains and robustness for a cheap wrapper. Cadences 5 and 31 distribute EMA samples across fast/slow phases, and one extra parameter copy fits memory. The proposal gives exact parameter, momentum, BN, SAM, EMA, and RNG semantics.

**Sources**: `proposals/idea-02.md`; `papers/lookahead-optimizer.md`; `knowledge/papers/lookahead-optimizer.md`; EXP-011.

**Estimated Effort**: medium.

**Risk Assessment**: EXP-011 shows no diagnosis of excessive online variance and already combines SAM with evaluation EMA. Nested smoothing may attenuate useful exploration; parameter pullback with retained momentum and unrecalibrated BN can hurt, and expected upside is only roughly 0.00-0.15 points.

### Activation-Anchored Bias-Corrected EMA
**Summary**: Preserve EXP-011's 18.75-second half-life, cadence 31, full-state support, and one-source evaluation, but normalize exponential mass from the exact 225-second activation boundary. This removes the parent copy-in first-state point mass while keeping parameters and floating BN buffers on one mathematically forced kernel and copying integer buffers latest.

**What it targets**: Stale-state bias in the stable 95.49 EMA tail without changing online optimization, evaluation count, half-life, or exposure (`02-system-understanding.md`).

**Reasoning**: The parent kernel gives its first clean-tail state about 6.30% terminal mass rather than the roughly 0.12% previously assumed. Boundary-anchored normalization reduces that oldest weight to about 0.033%, increases effective sample size from roughly 79 to 101, and lowers mean state age from roughly 25.1 to 21.8 seconds. No new fitted coefficient is introduced.

**Sources**: `proposals/idea-03.md`; `knowledge/papers/when-where-why-average.md`; `knowledge/papers/how-to-scale-your-ema.md`; EXP-011 and EXP-014 idea review.

**Estimated Effort**: medium.

**Risk Assessment**: The stale anchor may be beneficial regularization; normalized EMA can follow noisy late SAM states more closely, and averaged BN buffers remain a package-level approximation. The correction is well isolated but its expected effect may sit below the 0.10-point formal threshold.

### Bounded Soft-Target Poly-1
**Summary**: Use fixed `CE(q,p) - 0.25*(1-q dot p)` for every optimizer-driving ordinary/CutMix loss, attenuating confident hard-example gradients by at most 25% while preserving uncertain examples. On SAM steps retain the parent's plain-CE first pass and rho-0.05 ascent direction, then use negative Poly-1 only for the perturbed optimizer-driving pass. Model, data, optimizer, EMA, and evaluation remain unchanged.

**What it targets**: Decision-loss weighting on every update while preserving the optimizer/data exposure identified as critical in `02-system-understanding.md`.

**Reasoning**: PolyLoss reports image-classification gains from a tiny objective change and permits coefficient sign to change example weighting. Exact hard gradients retain direction with multiplier `1-0.25*p_y` in `[0.75,1]`: `0.975` at uniform initialization, `0.875` at `p_y=0.5`, and `0.775` at `p_y=0.9`. Negative epsilon relatively prioritizes residual errors and shifts unequal CutMix targets toward rather than away from softening. One 256x10 FP32 softmax should cost below 1% latency.

**Sources**: `proposals/idea-01.md`; `knowledge/papers/polyloss.md`; EXP-011 through EXP-014 reports.

**Estimated Effort**: medium.

**Risk Assessment**: The fixed magnitude is anchored to bounded confident-gradient attenuation rather than a local accuracy effect, can under-train class prototypes or over-soften CutMix, and creates a deliberate CE-ascent/Poly-descent SAM hybrid. A null rejects only epsilon -0.25 under this complete package.

## Review

Claude Opus verified all three finalists' central arithmetic. It scored Lookahead `4/10` evidence and `3/10` impact because alpha 0.5 halves five-step displacement absent LR retuning, duplicates EMA variance reduction, and may breach latency through frequent foreach launches. It scored activation-anchored EMA `8/10` evidence but `3/10` impact: the 6.30% anchor, 0.9373 mass, and ESS 79-to-101 correction are exact, yet the resulting displacement is likely too small for the formal bar and higher ESS reduces max-selection bias. It scored Poly-1 `5/10` evidence and `6/10` impact, the only ceiling plausibly reaching the goal, while requiring a sign decision, effect-size rationale, and narrower SAM interaction.

The significant feedback is adopted as follows: fix epsilon at `-0.25` so confident gradients are attenuated and unequal CutMix targets are softened rather than sharpened; preserve plain CE only for the SAM ascent pass and use Poly-1 for every optimizer-driving gradient; retain the parent CE implementation plus a single FP32 softmax; preregister the exact per-path audits and `95.69` scientific tail bar. The suggestion to choose magnitude from a training proxy is not adopted because even accuracy-blind local selection would turn the experiment into a data-conditioned coefficient search. The fixed 25% maximum attenuation produces a stated, material gradient effect while remaining bounded.

Claude's focused follow-up returned `PASS` with no blockers. It independently solved an unequal CutMix fixed point (`lambda=0.7` moves to about `0.678`), confirmed the CE-defined SAM adversary is a controlled fixed parent component rather than a fatal objective mismatch, and approved the retained CE plus one FP32 softmax. The audit now scopes scalar multipliers to hard calls, checks CutMix by its full vector formula, asserts empty CutMix/SAM intersection and gradient clearing, and uses dtype-aware tolerances. Full initial and follow-up reviews: `01-idea-review.md`.

## Idea Evaluation

The Claude pick is adopted after the prerequisite refinement. Bias-corrected EMA is the cleanest estimator correction but its own verified kernel arithmetic caps likely movement below the scarce one-run threshold. Lookahead has stronger literature breadth but weaker local mechanism fit and introduces effective-step, BN, and nested-smoothing risks. Negative Poly-1 is the only finalist that changes boundary learning on essentially every update while preserving exposure; its risks are explicit and its fixed operating point is falsifiable. Full review: `01-idea-review.md`.

## Chosen Idea
**Selected**: Bounded Soft-Target Poly-1 with confidence attenuation

**Why this idea**:
It is the sole finalist with a plausible effect ceiling above the `+0.10` formal and roughly `+0.20` stable-tail needs at negligible compute. The negative sign aligns example weighting and CutMix equilibrium movement with the diagnosed residual-boundary limiter, while keeping CE-defined SAM ascent preserves the validated neighborhood geometry.

**Hypothesis**:
On one fixed-seed physical-GPU-0 run, `epsilon=-0.25` soft-target Poly-1 applied to every optimizer-driving gradient under parent CE-based SAM ascent will pass the first paired preflight at median latency ratio `<=1.01`, retain at least 25,300 updates and 155 balanced EMA samples, reach `best_test_acc>=95.71%`, and lift the final-16 EMA mean to at least `95.69%`. Any valid preflight failure or sub-threshold run is recorded without changing sign, magnitude, phase scope, or SAM policy.
