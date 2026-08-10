# Brainstorm EXP-021
**Created**: 2026-08-06

## Web Search & Literature Review

- **Deeply-Supervised Nets** (`knowledge/papers/deeply-supervised-nets.md`): training-only intermediate companion classifiers can make hidden representations directly discriminative and reported CIFAR gains without inference-time cost.
- **Auxiliary Training: Towards Accurate and Robust Models** (`experiments/021/papers/auxiliary-training.md`): a later CVPR line independently supports disposable auxiliary classifiers, but its strongest results use corruption, selective BN, distillation, and alignment beyond a plain companion head.
- **Supervised Contrastive Learning** (`experiments/021/papers/supervised-contrastive-learning.md`): same-class batch positives can structure representations and outperform CE, but the primary method uses two views, a projection head, long training, and a separate linear-probe stage.
- **Dissecting Supervised Contrastive Learning** (`experiments/021/papers/dissecting-supervised-contrastive-learning.md`): CE and SupCon share a simplex-like optimum, so a joint single-view SupCon term offers an optimization bias rather than guaranteed new endpoint geometry.
- Existing notes on manifold mixup, ShakeDrop, SE, and late-stage width (`knowledge/papers/`) supplied alternatives, while prior in-repo attempts give stronger evidence about their local cost/effect.

## Experimental History Review

- The lineage BASE -> EXP001 -> EXP002 -> EXP004 reached 95.40 through WRN-16-4, front-loaded CutMix, and period-two clean-tail SAM. EXP004's final equaled its best and loss was 0.1654 despite 8.6% fewer steps than EXP002, making representation/generalization—not raw exposure or memory—the limiter.
- EXP011 successfully added clean-tail EMA and reached the 95.61 global best, but its 95.493 tail plateau has four failed children. EXP021 uses pre-EMA EXP004 so a representation mechanism can be tested cleanly at the 95.50 threshold and composed with EMA only after earning a gain.
- Failed children of EXP004 close half-overlap temporal distillation, substitution of validated CutMix with low-dose manifold mixup, uncalibrated ASAM, loader-bound paired RandAugment, and a procedurally rejected uniform-SWA harness. They do not test training-only intermediate supervision or a same-batch feature objective.
- Other branches reject multi-launch SE on H20, fixed-scale cosine geometry, negative Poly-1, frequent Lookahead, and full/conv-only official-order GC. Width-320 was only 0.95% beyond its preregistered latency ceiling and never reached accuracy, leaving cheaper late capacity open.
- The measured bottleneck is a stable generalization lift, with abundant memory but a tight per-step compute budget. A credible EXP004 child should preserve roughly 24k+ steps and plausibly move a stable endpoint by at least 0.2-0.3 points; extra backbone forwards and CPU dual views are poor fits.

## Collected Ideas

- **Stage-2 training-only companion classifier** — return the pooled 128-channel representation after block 4 during training, attach a zero-evaluation-cost linear auxiliary head, and add a fixed weighted CE using the exact hard/CutMix target semantics. This directly attacks hidden-feature discriminability with no extra backbone forward and has two independent auxiliary-supervision literature priors.
- **Single-view same-batch supervised contrastive auxiliary loss** — normalize final pooled features, optionally pass them through a small disposable projection, and add an `O(B^2)` many-positive SupCon term only on hard-label batches. It targets class compactness/margins without a second image view, but transfer from the two-view, two-stage paper is uncertain and SAM doubles its late cost.
- **Training-only class-center regularization** — maintain ten feature prototypes or learnable centers and pull clean-example pooled features toward their class center while separating centers. It is cheaper than pairwise contrastive loss and directly shapes representation clusters, but introduces state/update semantics and a coefficient with weaker evidence in this exact clean-label regime.
- **Stage-3 width 288** — widen only the final 8x8 stage from 256 to 288 channels, preserving block count and using unused H20 memory. EXP014's width-320 path was operationally sound but 1.161x slower; a 288 taper should retain more exposure while testing the same late-semantic-capacity hypothesis.
- **Low-rank stage-3 residual adapter** — add a training-and-inference 1x1 bottleneck adapter around the final pooled or 8x8 representation, initialized near identity. This moonshot spends memory/parameters more efficiently than dense width, but extra compact kernels may be launch-bound and external evidence is indirect.
- **Classifier-row local decorrelation penalty** — retain the successful affine classifier and add a small weight-space penalty that discourages positive correlations between class rows. It responds to EXP013's observed cosine collapse without imposing fixed-scale cosine logits, though ten-row geometry is tiny and likely below the required effect size.
- **Conservative ShakeDrop replacement** — replace existing binary drop path with expectation-controlled residual scaling, retaining the same depth schedule. CIFAR literature gives a high ceiling, but the current CutMix/SAM recipe may already be strongly regularized and negative residual scaling raises optimization risk.
- **Simplify early stochastic depth** — reduce or remove the inherited 0.08 drop path now that CutMix provides early regularization and SAM shapes the tail. This tests redundancy with zero added compute, but it is a scalar operating-point change with weak direct evidence and risks overfitting.
- **Auxiliary-to-primary logit distillation** — train a pooled intermediate classifier and transfer only its complementary confidence into the main logits, discarding the head at evaluation. This could make the auxiliary signal affect the deployed decision layer more directly than companion CE alone, but teacher/student direction and stop-gradient semantics materially expand the hypothesis.

## Combinations

- **Companion CE + later EMA composition**: first prove stage-2 deep supervision on EXP004; if it lifts the live stable endpoint, compose it with EXP011's validated EMA in a separate child. This staged A+B is stronger scientifically than changing both now because auxiliary representation gain and trajectory smoothing remain independently attributable.
- **Stage-3 adapter + companion head**: use a cheap late adapter for capacity and supervise it directly with a disposable head. Direct supervision could ensure the new capacity becomes class-discriminative, but two simultaneous architecture/objective changes and multiple new kernels make it too risky for the first fork.
- **Single-view SupCon + CE companion logits**: one pooled feature can feed both a cheap classifier head and batch geometry loss, potentially combining pointwise and relational supervision. It may outperform either signal alone, but adds two coefficients and obscures which representation pressure helps.

## Candidate Ideas

### Fixed 288-channel final-stage taper
**Summary**: Preserve EXP004's six blocks and widen only the final 8x8 stage from 256 to 288 channels, including its two residual blocks, final BN, and affine classifier. The fixed architecture has 3,260,442 parameters (+18.61%) and 425,315,136 MACs/image (+8.33%), with no extra module type, loss, stochastic decision, or kernel launch. All CutMix/SAM/training settings remain unchanged.

**What it targets**: Unused H20 memory and potentially insufficient late semantic capacity, while retaining the early/middle processing whose removal failed in EXP010. It attacks representation capacity directly rather than optimization or calibration.

**Reasoning**: EXP014's width-320 implementation was sound but rejected at 1.160975x latency without measuring accuracy; MAC-scaled interpolation predicts about 1.076x for width 288. A multiple-of-32 width should retain roughly 23.2k-23.9k steps and tests an explicitly open avenue. Unlike an adapter, it changes existing dense shapes without adding launch-bound paths.

**Sources**: `proposals/idea-03.md`; EXP014; EXP010; `knowledge/papers/deep-pyramidal-residual-networks.md`; `02-system-understanding.md`.

**Estimated Effort**: low

**Risk Assessment**: Lost exposure may exceed any capacity gain; the task may already be objective/data-limited; global SAM rho and per-weight decay change package meaning with parameter count. Shape-dependent initialization also shifts later RNG streams, so a narrow gain is package-level evidence. A fixed preflight must reject latency above 1.10x with no fallback width.

### Stage-2 training-only companion classifier
**Summary**: Tap the `[B,128,16,16]` tensor after residual block 3, apply ReLU plus global pooling and a disposable `Linear(128,10)` head, and optimize `L_main + 0.15*L_companion` on every step. The companion shares exact hard/CutMix targets and participates coherently in both SAM passes; the default/evaluator forward remains main-logits-only. Construct the 1,290-parameter head after all inherited initialization with an isolated seed so parent weights and RNG streams remain unchanged.

**What it targets**: The measured stable-generalization limiter by directly making the middle representation class-discriminative while retaining the validated final two blocks, CutMix, and SAM. It adds no backbone forward and no deployed inference path.

**Reasoning**: Deeply-Supervised Nets reports CIFAR gains from companion objectives, and CVPR auxiliary training independently supports disposable classifiers. Stage 2 is neither low-level nor redundant with the final head. The fixed 0.15 weight is subordinate but persistent, and identical joint objectives on the SAM ascent/descent passes avoid a geometry mismatch. The compact WRN may not be gradient-starved, so the real hypothesis is useful representation bias rather than optimization rescue.

**Sources**: `proposals/idea-01.md`; `knowledge/papers/deeply-supervised-nets.md`; `papers/auxiliary-training.md`; EXP004.

**Estimated Effort**: medium

**Risk Assessment**: Intermediate linear separability may discard information useful to stage 3; CutMix area targets are only approximate for receptive-field features; the head changes the global SAM perturbation; small pooling/loss kernels may be launch-bound. A decisive parent-relative latency preflight must retain at least 24,000 projected steps. A null result closes only this block-3, linear, weight-0.15 package.

### Hard-batch single-view supervised contrastive auxiliary loss
**Summary**: Return the existing 256-dimensional pooled feature on training calls and add `0.05*SupCon(T=0.1)` directly to main CE on hard-label batches only. Use all same-class non-self samples as positives in the `256x256` batch matrix, handle zero-positive anchors exactly, exclude every applied CutMix batch, and recompute the joint objective on both SAM passes. Add no projection head, second view, memory bank, parameter, or evaluation path.

**What it targets**: The same stable-generalization gap through within-class compactness and inter-class separation at the exact deployed representation, using relational batch supervision rather than another scalar logit calibration.

**Reasoning**: NeurIPS SupCon supplies a strong class-geometry prior and temperature 0.1, while the ICML analysis shows CE and SupCon can share a simplex optimum but follow different optimization paths. Direct single-view use is compute-conscious and attribution-clean, but it is a substantial adaptation from canonical two-view, projection-head, long-training, linear-probe SupCon. The 0.05 weight keeps its initial scalar contribution near one tenth of CE.

**Sources**: `proposals/idea-02.md`; `papers/supervised-contrastive-learning.md`; `papers/dissecting-supervised-contrastive-learning.md`; EXP004.

**Estimated Effort**: medium

**Risk Assessment**: The pairwise backward adds about 16.8M MAC terms per eligible pass and is doubled on SAM pulses; direct feature compactness may erase useful intra-class variation; excluding CutMix yields only about 62.5% primary-step dose. The evidence transfer omits canonical paired views/projection/two-stage training, so a null result has narrow scope.

## Review

Claude selected the stage-2 training-only companion classifier, scoring it 7/10 for evidence and 7/10 for impact versus 4/6 for single-view SupCon and 6/5 for width 288 (`01-idea-review.md`). I adopted its conditioning concern: a BN-free head is still the cleanest intervention, but sparse pooled stage-2 feature norms must be reported so drift in the raw pre-activation residual stream is visible. I also hardened composition: a formal 95.50-95.59 gain or a best-only spike cannot justify adding EMA. Composition requires `best_test_acc >=95.60%`, final-16 mean `>=95.50%`, and final accuracy within 0.15 points of best. The literature is framed as a representation-bias prior, not transferable evidence that this shallow BN residual net is gradient-starved.

## Idea Evaluation

Adopt the verdict. Width 288 has the strongest systems case but pays a 7-10% exposure tax for an unproven capacity limiter and shifts the fixed-seed initialization/data package. Single-view direct-feature SupCon is more orthogonal, but the compute-mandated removal of paired views and a projection head strips away important source-protocol components while risking growing interference in the clean SAM tail. The companion head best aligns mechanism, near-parent exposure, causal isolation, and later composability.

## Chosen Idea
**Selected**: Stage-2 training-only companion classifier

**Why this idea**:
It directly pressures the middle representation to become class-discriminative, keeps the deployed/evaluator path unchanged, adds no backbone forward, preserves inherited initialization and stochastic streams through isolated head construction, and has independent CIFAR literature priors. Its exact hard/CutMix targets and joint-objective SAM semantics make the result attributable to one coherent training-only mechanism.

**Hypothesis**:
Full-run block-3 companion CE at fixed weight 0.15 will preserve at least 24,000 optimizer steps and raise EXP004's `best_test_acc` from 95.40% to at least 95.50% by improving stage-2 discriminability. A composition-worthy result must additionally reach at least 95.60%, a final-16 mean of at least 95.50%, and final accuracy within 0.15 points of best; otherwise the mechanism will not be carried onto EMA on the strength of selection noise.
