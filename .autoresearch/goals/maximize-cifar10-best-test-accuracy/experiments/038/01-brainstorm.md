# Brainstorm EXP-038
**Created**: 2026-08-06

## Web Search & Literature Review

- **Control Batch Size and Learning Rate to Generalize Well** (`papers/control-batch-size-and-learning-rate.md`; NeurIPS 2019)
  Across 1,600 CIFAR models, generalization correlated with batch-size/LR ratio; a larger-batch candidate should scale LR coherently rather than double the ratio.
- **Demystifying Loss Functions for Classification** (`papers/demystifying-loss-functions.md`; ICLR 2021 submission)
  Cosine/logit-normalized heads can regularize classification, but temperature, LR, and decay jointly determine their optimizer geometry; a drop-in arbitrary scale is not supported.
- **Trainable Weight Averaging** (`papers/trainable-weight-averaging.md`; ICLR 2023)
  Historical checkpoint coefficients matter and a learned low-dimensional combination can beat uniform averages, but its projected optimization and BN handling are too costly for a clean 300-second local test.
- **When, Where and Why to Average Weights?** (`knowledge/papers/weight-averaging.md`; ICML 2025)
  Recency-aware averaging alongside LR annealing can give modest gains, motivating EMA as distinct from EXP018's uniform window while leaving a strong local null-risk.
- **PyTorch Channels-Last Memory Format** (`knowledge/references/pytorch-channels-last.md`; official PyTorch tutorial)
  CUDA convolution and BatchNorm can propagate NHWC physical layout without changing logical NCHW semantics, but tiny FP32 CIFAR shapes require direct end-to-end timing.
- **Deep Networks with Stochastic Depth** (`knowledge/papers/stochastic-depth.md`; ECCV 2016)
  Residual bypass can regularize and shorten training graphs, though published gains concern networks far deeper than this nine-block model.

## Experimental History Review

- The current frontier is EXP010 at 94.15%: width-2 postactivation ResNet-20, N1/M7 plus 50% alpha-1 CutMix through 80%, then a hard weak cosine tail. Its 89.73% switch fit and ordinary SGD path are valuable invariants.
- EXP007 and EXP010 show that capacity and class-bearing mixed views can overcome moderate update loss. EXP011/026/027 show that stronger or earlier-changed mixing deepens strong-phase underfit.
- The measured system is 97.57% forward/backward GPU work, with 75.46% in backward and only 0.61% VRAM used. Python, loss, transfer, and optimizer micro-optimizations have little ceiling.
- EXP013 rejected batch256/LR0.2 on a stable 1.189x image-throughput result below its 1.20x premise; it produced no accuracy result and explicitly left batch192 as a new knee. Evaluation count must be fixed when epoch length or speed changes.
- EXP018's uniform 86-98% SWA trailed its own online checkpoint despite ample spread and BN refresh. A short EMA is distinct but the accepted baseline's tiny best-final gap argues that terminal noise is not the main limiter.
- EXP012/015 show that suppressing residual activity damages the short strong phase; this is direct negative evidence against shallow stochastic depth despite possible compute savings.
- EXP014/031 show that sparse max readouts can collapse through uncontrolled gradient geometry. A tiny average-plus-RMS readout is intrinsically bounded, but its maximum 10.94% descriptor effect has weak direct evidence.
- EXP020/022/028 make global optimizer-path changes a recurring collapse risk. EXP034-037 further require control-qualified, denominator-safe gates and effect-specific—not generic chaotic—mechanism checks.
- Untried gaps are an FP32 layout speedup that preserves batch noise, an intermediate batch/update-exposure knee, and a bounded angular classifier whose initial output scale is matched to the accepted head.

## Objective Limiter Diagnosis

The direct systems limiter is convolution/BN backward, so a layout change has a clean route to more decisions in the same 300 counted seconds. The accuracy limiter is useful generalization after a short strongly augmented high-LR phase: candidates must preserve the accepted model's switch fit and ordinary SGD geometry. Memory has vast headroom, but EXP013 shows raw image throughput is insufficient if optimizer decisions fall too sharply. Representation changes need lifetime bounds and production-distribution safety because locally benign initial functions have repeatedly destabilized within a few updates.

## Collected Ideas

- **FP32 channels-last training** — convert model Conv weights and every 4-D input to channels-last while preserving logical shapes, batch128, FP32, and the accepted algorithm. It directly targets the 97.57% forward/backward workload and could turn a kernel-layout gain into more fixed-budget updates.
- **Batch-192 with a linearly scaled LR curve** — use batch192 and `0.15/0.015/0.00015`, preserving the batch/LR ratio and per-pass LR mass. It tests the unmeasured Pareto knee between accepted 128 and rejected 256, with roughly 75% of optimizer decisions and a prior near 12% extra image exposure.
- **Short-timescale late parameter EMA** — maintain a `0.999` parameter shadow from 90-99%, then install it and recompute BN inside the last 1%. This is recency-weighted and step-dense, unlike EXP018's uniform 86-98% endpoints, but accepted best/final proximity makes the gain uncertain.
- **Bounded average-plus-RMS readout** — mix global average with RMS at fixed coefficient `1/64`. The descriptor and every spatial Jacobian coefficient remain within 10.94% of average pooling, providing a dense localized-feature bias without prior max-path sparsity.
- **Very mild strong-phase stochastic depth** — batchwise bypass only six same-width blocks at 1-6% rates, restoring full depth for weak training and evaluation. It attacks backward cost while regularizing, but omitting even one of nine blocks is coarse and local residual-suppression evidence is negative.
- **Output-RMS-matched cosine classifier** — normalize pooled features and classifier rows, omit the bias in-function, and apply frozen scale `22.7869167` derived from an immutable accepted batch. It bounds logits and removes radial shortcuts while matching initial affine-logit RMS, but tangent/inverse-norm dynamics still face global LR0.1.
- **Gentle stage-width growth** — replace abrupt stage doubling with smaller per-block channel increments to spend unused H20 memory while avoiding EXP024's 32/64/160 jump. This requires many Option-A shapes and lacks a clean initialization/optimizer invariant, so it is not yet proposal-ready.
- **Online self-distillation moonshot** — train a student against a lagged same-run teacher on strong views to stabilize class geometry. A second model forward would consume too much of the backward-bound fixed budget and introduces target/BN ambiguity, making it infeasible without a new efficiency mechanism.

## Combinations

- **Channels-last + batch192** could make larger batches more efficient and move the hardware Pareto knee beyond either alone. It also changes layout, kernel numerics, batch noise, LR, and evaluation cadence simultaneously, so attribution and safety are too weak for a first test.
- **Channels-last + late EMA** could subsidize EMA's foreach cost while smoothing the extra online decisions. The accepted final gap and unknown layout speed make the combination less informative than measuring the layout alone.
- **Average-plus-RMS + cosine head** could preserve localized CutMix energy before angular classification, potentially complementing two representation biases. It compounds pooling and optimizer geometry and offers no way to identify which component helped.
- **Batch192 + stochastic depth** could regain some lost decisions through skipped branches while adding generalization noise to the larger batch. The resulting variable-depth LR0.15 trajectory combines two independent instability risks and is rejected.

## Candidate Ideas

### Batch-192 Linear-Scale Pareto Knee
**Summary**: Change batch128 to192 and scale the entire LR curve by1.5 to `0.15/0.015/0.00015`, while fixing 19 elapsed-progress evaluation looks. The method preserves batch/LR ratio and first-order LR/decay mass per 49,920-image pass. Full proposal: `proposals/idea-02.md`.

**What it targets**: Unused H20 parallelism and the open update-versus-image-exposure tradeoff. Interpolation from EXP013 predicts about 1.119x image throughput while retaining about 74.6% of accepted optimizer decisions, materially more decisions than batch256.

**Reasoning**: EXP013 produced a stable hardware endpoint rather than an accuracy result and explicitly identified batch192 as unexplored. NeurIPS evidence supports keeping batch/LR ratio fixed. This candidate has greater plausible accuracy leverage than a few-percent layout gain, provided its new knee and LR0.15 trajectory qualify prospectively.

**Sources**: `papers/control-batch-size-and-learning-rate.md`; EXP010; EXP013; control-qualified trajectory lessons from EXP020/022/028; `proposals/idea-02.md`.

**Estimated Effort**: high — image-aligned safety replay, new seven-pair knee timing, integrated loader/evaluation simulation, then conditional production.

**Risk Assessment**: It may lose too many optimizer decisions, reduce useful SGD/BN noise, or become transiently unstable at LR0.15. The expected gain rests on a nonlinear hardware interpolation and untested short-horizon large-batch generalization.

### Output-RMS-Matched Cosine Classifier
**Summary**: Preserve the existing Linear state and global average features but compute `s * cosine(h,w)` with epsilon `1e-6`, no functional bias, and frozen `s=22.786916732788086`. The scale uniquely matches the accepted initial affine-logit RMS on an immutable pre-existing CutMix batch and also nearly matches the next hard batch. Full proposal: `proposals/idea-06.md`.

**What it targets**: Confidence/generalization under strong views and area-soft CutMix by removing feature/classifier norm shortcuts and bounding logits, while preserving the backbone and measured convolutional workload.

**Reasoning**: Classification-loss literature supports angular/logit-normalized objectives but warns that scale and optimizer settings matter. Output-RMS calibration removes EXP037's arbitrary scale8 mismatch without labels or accuracy selection. It offers the largest representation upside among bounded finalists and is materially distinct from sparse max pooling failures.

**Sources**: `papers/demystifying-loss-functions.md`; EXP014/031/034/035; EXP037 deferred candidate; immutable-corpus protocol; `proposals/idea-06.md`.

**Estimated Effort**: high — formula/gradient oracles, calibration reproduction, full control-qualified trajectory, paired timing, then conditional production.

**Risk Assessment**: Matching output RMS does not match Jacobians. Inverse feature/row norms, radially ineffective decay, global LR0.1, and lost magnitude cues for CutMix can destabilize or underfit; the cited literature usually tunes temperature, LR, and decay jointly.

### FP32 Channels-Last Training
**Summary**: Initialize the accepted model identically, convert its four-dimensional parameters to `torch.channels_last`, transfer training inputs in the same format inside counted time, and add a forward-boundary restride for immutable evaluator inputs. Fix the experiment to 19 elapsed-progress evaluation opportunities so any speedup cannot gain extra test-set looks. Full proposal: `proposals/idea-01.md`.

**What it targets**: The measured 97.57% forward/backward GPU stage, especially 75.46% convolution/BN backward. A >=3% complete-step speedup would add at least 807 batch128 optimizer decisions without changing batch noise, capacity, curriculum, or SGD logic.

**Reasoning**: Official PyTorch guidance supports CUDA convolution/BN channels-last propagation, and the local H20 has no memory pressure. Unlike AMP, width, optimizer, or representation interventions, logical computation remains the accepted algorithm; only physical strides and legal kernel reductions differ. Propagation/profiler checks can prove the mechanism before paired timing.

**Sources**: `knowledge/references/pytorch-channels-last.md`; `02-system-understanding.md`; EXP013 timing/evaluation lessons; EXP016 and EXP029 distinctions; `proposals/idea-01.md`.

**Estimated Effort**: high — layout propagation, conversion profiling, numerical replay, seven timing pairs, then conditional production.

**Risk Assessment**: Tiny 32x32 FP32 tensors and Option-A slice/pad may erase NHWC gains or force hidden repairs; even a real 3% exposure gain may be too small to move accuracy by 0.10.

## Review

Claude's independent review (`01-idea-review.md`) selected **Output-RMS-Matched Cosine Classifier**, scoring evidence/reasoning 6/10 and impact 7/10. It scored channels-last 7/10 and 3/10: the speed mechanism is strongest, but its accuracy ceiling depends entirely on an unproven exposure effect and the predicted metric sits exactly at the gate. It scored batch192 4/10 and 4/10 because raw exposure has weak local support, roughly one quarter of optimizer decisions disappear, and LR0.15 approaches the recurring optimizer-instability family.

The review's decisive cosine concern is valid: 264 replay steps can detect an immediate spike but not slow classifier-row shrinkage and inverse-norm Jacobian amplification over a ~26.9k-step run. Planning must therefore add a materially longer repeated-corpus survival stage with fixed phase/LR semantics, row/feature norm floors, tangent update tracking, and a prospective monotonic-drift bound qualified by accepted controls. The 89.73% accepted switch fit is also load-bearing; the plan must pre-register a strong-phase fit floor and never reinterpret a primary-metric pass as support for the intended mechanism if angular normalization materially suppresses it. These changes refine safety and attribution without tuning scale, LR, epsilon, or decay.

## Idea Evaluation

- **Output-RMS-matched cosine classifier** — Advance. Highest plausible accuracy impact, a genuinely new angular mechanism, and the immutable RMS calibration fixes the earlier arbitrary-temperature objection. Long-horizon norm/update survival and strong-fit preservation are mandatory additions.
- **FP32 channels-last** — Defer. Best engineering evidence and clean bottleneck attack, but likely only a few percent more tail-heavy decisions and no established exposure-to-accuracy mechanism.
- **Batch192 linear scaling** — Reject for this loop. It is coherent and distinct from batch256, but loses too many decisions while increasing global LR; local evidence does not justify that joint risk.

## Chosen Idea
**Selected**: Output-RMS-Matched Cosine Classifier

**Why this idea**:
It replaces radial affine classification with a bounded angular readout while preserving the accepted backbone, pooling, parameter construction, augmentation, and ordinary SGD. The frozen scale is derived without labels or accuracy from an immutable accepted batch and matches the accepted initial logit RMS, removing the arbitrary scale8 flaw. Among the finalists it is the only one with a direct generalization mechanism large enough to plausibly clear +0.10 rather than relying solely on extra exposure.

**Hypothesis**:
Using fixed scale `22.786916732788086` cosine logits will preserve accepted initial output scale, hard/CutMix loss magnitude, and strong-phase fit while preventing feature- and classifier-norm shortcuts. It will remain free of candidate-specific class concentration and slow inverse-norm update amplification through a long control-qualified replay, cost no catastrophic fixed-budget time, and raise seed-42 `best_test_acc` from 94.15% to at least 94.25%. A safety/timing veto or valid miss retires this exact scale/epsilon/global-LR operating point without temperature, head-LR, decay, phase, or bias rescue.
