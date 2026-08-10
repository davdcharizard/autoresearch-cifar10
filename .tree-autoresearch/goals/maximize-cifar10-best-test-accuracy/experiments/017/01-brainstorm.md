# Brainstorm EXP-017
**Created**: 2026-08-06

## Web Search & Literature Review

- **Gradient Centralization** (`knowledge/papers/gradient-centralization.md`)
  ECCV 2020 frames zero-mean eligible weight gradients as projected optimization and reports broad vision gains without another forward pass; this fits the fixed compute bottleneck but needs charged overhead and removed-component audits.
- **Deeply-Supervised Nets** (`knowledge/papers/deeply-supervised-nets.md`)
  AISTATS 2015 shows companion hidden-layer objectives can improve intermediate discriminativeness and CIFAR accuracy; a pooled middle-stage head is a low-cost adaptation, though this shallow pre-activation WRN may not be gradient-limited.
- **Unified Mixed-Sample Analysis** (`knowledge/papers/mixed-sample-analysis.md`)
  Mixup and CutMix impose distinct input-gradient regularization, supporting a hybrid only if it retains the parent's validated CutMix dose and does not erase clean identities.
- **ShakeDrop** (`knowledge/papers/shakedrop.md`)
  Residual-branch disturbance has direct CIFAR/ResNet evidence, but stability depends on architecture and strength and may overlap the parent's existing stochastic depth.
- **When Does Label Smoothing Help?** (`knowledge/papers/when-label-smoothing-helps.md`)
  Mild soft targets can improve calibration cheaply, but half of EXP-002's early batches already receive CutMix soft labels.

## Experimental History Review

- The lineage rose from 91.51 to 94.62 through WRN-16-4, then to EXP-002's 95.23 through front-loaded probability-0.5 CutMix. EXP-002 retained 27,950 steps, ended at 95.19, and improved final CE loss to 0.2044 (`experiments/002/04-analysis.md`).
- EXP-002's successful child EXP-004 added late periodic SAM for 95.40 and ultimately enabled EXP-011's 95.61 full-state EMA tip. EXP017 deliberately explores before those mechanisms rather than adding a fifth child to the saturated tip.
- Direct failures from EXP-002 reject a narrow CutMix/drop-path scalar sweep (EXP-003), four active FP32 SE gates with 20.7% overhead (EXP-009), and one equal-MAC back-loaded stage-depth allocation (EXP-010). Do not repeat these packages.
- The dominant limiter is stable generalization, not VRAM. Extra forwards and loader work reduce useful exposure; one-pass auxiliary heads, gradient transformations, and sparse state arithmetic are feasible (`02-system-understanding.md`).
- The local threshold is 95.33. This is easier than the global 95.71 threshold but should be interpreted as an exploratory branch step, not a new goal-wide best unless it exceeds 95.61.

## Collected Ideas

- **Full-run eligible-weight gradient centralization** — After each backward, materialize coupled L2 on every gradient, then subtract each eligible convolutional/linear direction's mean over non-output dimensions before Nesterov consumes it. This matches the official GC optimizer ordering, directly changes optimizer geometry at linear parameter cost, preserves data exposure and model structure, and is supported by ECCV vision evidence. The main uncertainty is whether BatchNorm and Nesterov already supply most of its conditioning benefit.
- **Middle-stage companion classifier** — Expose the representation after block four, global-average-pool it, and train a small 128-to-10 head with a fixed companion CE using exactly the same hard or CutMix-weighted targets as the main head. It adds direct representation supervision without a second backbone pass and is discarded from evaluation. The shallow WRN may not need extra gradient signal, and the coefficient is an unvalidated scale.
- **Direct clean-tail full-state EMA from EXP-002** — Add the validated cadence-31, 18.75-second full-state EMA estimator to EXP-002 without SAM. This tests whether averaging alone supplies a stable local gain and is operationally low risk, but it borrows a later branch mechanism and likely cannot match the global best without SAM.
- **Clean-batch-only label smoothing** — Preserve every CutMix batch exactly and apply a small fixed uniform target smoothing only to otherwise-clean early batches, returning to hard CE in the final quarter. This fills an output-regularization gap cheaply but may over-regularize a recipe whose existing CutMix/drop-path balance already failed scalar tuning.
- **Additive Mixup on a subset of non-CutMix batches** — Keep the parent's probability-0.5 CutMix gate and use a dedicated second gate to turn half of remaining early clean batches into Mixup, preserving CutMix dose while diversifying mixed-sample geometry. It reduces clean identity exposure and may repeat the substitution/data-diversity lesson in a softer form.
- **Conservative ShakeDrop replacement** — Replace the current per-sample residual drop-path multiplier with a weak, expectation-controlled ShakeDrop rule during the same early window and anneal it away at 75%. It changes representation noise without extra forwards, but literature settings target much deeper networks and stronger disturbance could destabilize this six-block WRN.
- **Classifier-row affine decorrelation** — Add a small unnormalized off-diagonal Gram penalty on the ten classifier rows. It targets class-boundary diversity without cosine normalization, remaining distinct from EXP-013's failed fixed-scale cosine classifier, but no measured row-collapse diagnosis or coefficient evidence exists.
- **Input-channel standardization moonshot** — Replace mean-only input normalization with standard CIFAR per-channel standard deviation. This is nearly free and conventional, but downstream BatchNorm makes much of the scale change redundant and the effective first-layer/weight-decay geometry shifts in an uncontrolled way.

## Combinations

- **Gradient centralization + direct EMA**: GC changes the path while EMA summarizes its late trajectory; the combination could produce smoother, more generalizable iterates than either alone. It is plausibly stronger but confounds two untested mechanisms on this base, so isolate GC first.
- **Companion classifier + gradient centralization**: direct intermediate supervision creates a new gradient signal and GC prevents dominant mean directions from monopolizing it. The cross could improve representation diversity, but attribution and charged overhead become harder than either standalone test.
- **ShakeDrop + clean-tail EMA**: strong early representation perturbation followed by clean late averaging mirrors the successful early-regularize/late-stabilize pattern. It may outperform either, but first establish a stable ShakeDrop operating point rather than hiding instability with averaging.

## Candidate Ideas

### Full-Run Eligible-Weight Gradient Centralization

**Summary**: After every backward, reproduce coupled L2 on every gradient, then centralize each eligible `Conv2d.weight` direction independently over its input/spatial axes and each `Linear.weight` row over input features, immediately before PyTorch Nesterov momentum. Leave BN affine parameters and biases uncentralized. The fixed inventory is 17 tensors, 2,745,264 elements, and 2,266 rows; internal optimizer decay is disabled to avoid double application (`proposals/idea-01.md`).

**What it targets**: Stable generalization under a strict forward budget. It changes optimizer geometry across every hard-CE and CutMix step without another model forward, data change, parameter, or persistent state (`02-system-understanding.md`).

**Reasoning**: ECCV 2020 reports smoother optimization and vision generalization from this coefficient-free projection. Unlike EXP-004 SAM, it pays only linear gradient reductions; unlike EXP-003 it is not a scalar retune. Sparse removed-component audits can prove the mechanism. The main risks are redundancy with pre-activation BN/Nesterov and 17 reduction/subtraction launch pairs on a small kernel-bound WRN.

**Sources**: `knowledge/papers/gradient-centralization.md`; `experiments/002/04-analysis.md`; `experiments/004/04-analysis.md`; `experiments/017/proposals/idea-01.md`.

**Estimated Effort**: Medium.

**Risk Assessment**: Medium implementation and scientific risk, medium potential upside. Full-run overhead or removal of useful coordinated classifier/CutMix gradient components could negate the gain.

### Training-Only Fourth-Block Companion Classifier

**Summary**: Tap block index 3 (`[B,128,16,16]`), apply ReLU, global average pooling, and a `Linear(128,10)` companion head. Train it with the exact same hard or area-weighted CutMix targets as the main head at weight 0.15 through 50% charged progress, tapering linearly to zero at 75%. The 1,290-parameter head is never executed or consulted by evaluation (`proposals/idea-02.md`).

**What it targets**: Intermediate representation quality without another backbone forward. It supplies a direct discriminative objective to the stem and first four blocks while preserving the clean final quarter and frozen evaluator (`02-system-understanding.md`).

**Reasoning**: AISTATS Deeply-Supervised Nets reports CIFAR gains from companion objectives, and the attachment adds only a pooled linear path rather than EXP-009's four active SE gates. It retains CutMix and the 2-2-2 backbone. The counter-case is strong: a six-block pre-activation WRN is not demonstrably gradient-starved, and forcing middle-stage linear separability may overconstrain useful features.

**Sources**: `knowledge/papers/deeply-supervised-nets.md`; `experiments/002/04-analysis.md`; `experiments/009/04-analysis.md`; `experiments/010/04-analysis.md`; `experiments/017/proposals/idea-02.md`.

**Estimated Effort**: Medium.

**Risk Assessment**: Medium implementation risk and high scientific risk, with medium-high upside. The fixed coefficient/attachment are weakly calibrated and small auxiliary kernels may still be launch-bound.

### Readiness-Gated Clean-Tail Full-State EMA Without SAM

**Summary**: Add an activation-anchored, bias-corrected cadence-31 full-state EMA with an 18.75-second half-life directly to EXP-002, explicitly without SAM. Average parameters and persistent floating buffers, latest-copy integer buffers, and evaluate live until both normalized mass reaches 0.75 and ESS reaches 90; thereafter evaluate EMA only (`proposals/idea-03.md`).

**What it targets**: Late-iterate variance and stable generalization using cheap sparse state arithmetic, while isolating EMA's contribution from the successful SAM child (`02-system-understanding.md`).

**Reasoning**: EXP-011 validates the operational class and reached 95.61 on EXP-004, while EXP-016 showed why estimator readiness and source-at-best must be explicit. Direct EMA from EXP-002 is the strongest causal isolation test and likely low overhead. However, EXP-002's best-to-final gap is only 0.04, EMA can lag a smoothly improving cosine tail, and a local pass may remain far below the global best.

**Sources**: `knowledge/papers/how-to-scale-your-ema.md`; `knowledge/papers/when-where-why-average.md`; `experiments/002/04-analysis.md`; `experiments/011/04-analysis.md`; `experiments/016/04-analysis.md`; `experiments/017/proposals/idea-03.md`.

**Estimated Effort**: Medium-high.

**Risk Assessment**: Low-to-medium implementation risk and medium scientific risk, with modest upside. Readiness reduces attribution ambiguity but leaves fewer EMA evaluations and may surrender max-selection premium.

## Review

Claude Opus selected Full-Run Eligible-Weight Gradient Centralization (`01-idea-review.md`). The decisive advantages were same-regime ECCV evidence on BN ResNets, no extra forward or tunable coefficient, and composability with the global-best SAM+EMA frontier if the local result is strong. The review rejected direct EMA as a low-impact ablation on the wrong node and found the companion head's evidence poorly matched to a six-block pre-activation WRN.

I adopted the significant GC refinements. Coupled L2 is materialized before GC so the eligible regularized direction entering momentum, rather than only the data-loss gradient, is zero-row-mean; this matches the official CIFAR `SGD_GC` ordering. The 17 row means remain exact per-tensor reductions, but their broadcast subtractions use one `torch._foreach_sub_`; a local smoke confirmed heterogeneous broadcast shapes work, and preflight must prove bitwise parity with the loop reference. Audits now mandate FP64 decomposition accumulation, split convolution/classifier removed energy, and preregister at most 1% removed/regularized as evidence for BN redundancy, 1-5% as ambiguous, and at least 5% as substantial removed signal. The formal threshold remains 95.33, while 95.53 is only a noise-limited context bar; only a strong local result motivates considering GC on EXP-011.

## Idea Evaluation

The review scored GC 7/10 for evidence and 7/10 for impact, the companion classifier 4/10 and 5/10, and direct EMA 6/10 and 3/10. I adopt the GC verdict. Its main downside, full-run reduction overhead, is measurable without accuracy and is reduced by exact foreach subtraction rather than by weakening eligibility. Its removed-energy audit also makes either sign scientifically useful.

## Chosen Idea
**Selected**: Full-Run Eligible-Weight Gradient Centralization

**Why this idea**:
GC targets stable generalization through optimizer geometry while preserving EXP-002's architecture, CutMix dose, data exposure, and one-forward training. It is coefficient-free, supported in the relevant BN-ResNet vision regime, and a sufficiently strong local result can compose with SAM+EMA at the frontier. Exact parent parity through backward, excluded-tensor parity under relocated coupled decay, and explicit decay-before-centralization Nesterov checks make the intervention unusually auditable.

**Hypothesis**:
Centralizing all 16 convolutional and one classifier regularized directions per output row before every Nesterov step will suppress correlated mean-direction drift at no meaningful exposure cost, producing `best_test_acc >=95.33%` from EXP-002. A result of at least 95.53 with stable final/tail context is a weak, noise-limited reason to consider stacking it on EXP-011; removed/regularized energy and classifier-vs-convolution splits will distinguish BN redundancy from discarded useful signal if accuracy does not improve.
