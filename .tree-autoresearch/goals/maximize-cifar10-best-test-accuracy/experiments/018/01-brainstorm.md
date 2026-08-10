# Brainstorm EXP-018
**Created**: 2026-08-06

## Web Search & Literature Review

- **Gradient Centralization** (`knowledge/papers/gradient-centralization.md`)
  ECCV 2020 and the official CIFAR optimizer support zero-row-mean eligible regularized directions in the exact order `data gradient + coupled L2 -> GC -> momentum`; it costs no extra forward and remains scientifically untested here because EXP017 crashed in its temporary preflight.
- **Deeply-Supervised Nets** (`knowledge/papers/deeply-supervised-nets.md`)
  AISTATS 2015 reports CIFAR gains from training-only companion objectives that shape intermediate representations, but a six-block pre-activation WRN may not be gradient-limited and the attachment/coefficient need strict preregistration.
- **Lookahead Optimizer** (`knowledge/papers/lookahead-optimizer.md`)
  NeurIPS 2019 reports variance reduction and CIFAR gains from canonical `k=5`, `alpha=0.5` slow/fast interpolation with no extra forward; direct use on EXP002 avoids nested smoothing with EXP011's EMA.
- **ShakeDrop** (`knowledge/papers/shakedrop.md`)
  Residual-branch stochastic scaling has direct CIFAR residual-network evidence, but the compact WRN already uses drop path and strong disturbance can destabilize shallow networks.
- **RandAugment** (`knowledge/papers/randaugment.md`)
  CIFAR WRN evidence supports reduced-search image policies, though loader cost and overlap with CutMix make a clean fixed-budget test harder than a gradient/state intervention.
- **Unified Mixed-Sample Analysis** (`knowledge/papers/mixed-sample-analysis.md`)
  Mixup and CutMix regularize input gradients differently; an additive hybrid remains plausible only if it preserves the validated CutMix dose and enough clean identities.

## Experimental History Review

- The lineage improved 91.51 -> 94.62 with the time-aware WRN and then 94.62 -> 95.23 with front-loaded probability-0.5 CutMix. EXP002 retained 27,950 steps, ended at 95.19, and reduced final CE loss to 0.2044 (`experiments/002/04-analysis.md`).
- EXP004's clean-tail periodic SAM was the only successful child of EXP002, reaching 95.40 and later enabling EXP011's 95.61 EMA tip. EXP018 deliberately remains below SAM/EMA to isolate a new mechanism before composition.
- Direct EXP002 failures reject a small CutMix/drop-path scalar sweep (EXP003), four active late-stage SE gates with 20.7% latency (EXP009), and one equal-MAC back-loaded depth layout despite higher throughput (EXP010). Do not repeat those packages.
- EXP017 implemented official-order GC and passed deterministic math/RNG/update checks, but its temporary preflight retained device loss scalars and exhausted its single repair before timing or metric access. Claude audited `crash/NaN` and explicitly judged GC untested (`experiments/017/04-analysis.md`).
- The measured limiter is a stable generalization lift, not memory. WRN/CutMix/SAM/EMA gains diminished to +0.17 and +0.21 points, while final-16 selection noise spans roughly 0.17 points. A credible mechanism should plausibly move 0.25-0.30 points and preserve data exposure (`02-system-understanding.md`; `03-experiment-learnings.md`).
- Future feasibility harnesses must use fixed device-scalar diagnostic accumulation and release transient tensors before comparing allocations. Preflight timing should use parent-relative gates calibrated above observed round noise.

## Collected Ideas

- **Clean reference-ordered GC retry** — Re-run the exact all-eligible, full-run official GC mechanism from EXP017 under a newly preregistered harness that fixes audit counting and never retains per-step CUDA tensors. It directly attacks optimizer conditioning and stable generalization without another forward, coefficient, parameter, or stochastic draw; the prior leaf supplies implementation evidence but no metric evidence.
- **Training-only middle-stage companion classifier** — Pool the representation after the fourth residual block, attach a small 128-to-10 head, and add a fixed tapered companion CE using the same hard/CutMix targets. It targets intermediate discriminativeness using one backbone forward, but adds a new coefficient and could overconstrain a shallow network.
- **Direct canonical Lookahead** — Wrap EXP002's Nesterov path with parameter-only slow weights, interpolating every five optimizer steps at alpha 0.5 while retaining fast momentum. It uses abundant memory and sparse fused arithmetic to reduce trajectory variance, but may lag the time-cosine endpoint or duplicate the benefit later supplied by EMA.
- **Conservative ShakeDrop replacement** — Replace the existing residual drop-path multiplier with expectation-controlled ShakeDrop during the same early 75% window and anneal it away for the clean tail. This attacks representation diversity with direct ResNet evidence, yet the literature's stronger/deeper regimes may not transfer and the intervention risks destabilizing the six-block WRN.
- **CutMix-complementary Mixup** — Preserve every current CutMix draw and turn a fixed subset of otherwise-clean early batches into Mixup with a separate RNG stream, leaving the last quarter clean. The hybrid targets distinct input-gradient regularization, but it reduces clean identity exposure and could repeat the over-regularization/selection-noise lesson of EXP003.
- **Early capacity-matched RandAugment** — Apply a published CIFAR WRN policy only before the 75% clean-tail boundary, keeping CutMix and all training/evaluation semantics fixed. It offers an orthogonal data-space lever without GPU forwards, but PIL worker cost and augmentation overlap could lower useful exposure before any accuracy gain.
- **Drop-path simplification with optimizer geometry** — Remove stochastic depth while adding GC, testing whether CutMix plus projected optimization makes residual dropping redundant. This simplification could recover a cleaner gradient signal, but it confounds two mechanisms and EXP003 already showed that small regularization-balance changes are hard to resolve.

## Combinations

- **GC + direct Lookahead**: GC removes mean directions every step while Lookahead periodically reduces path variance. The cross could be stronger than either if conditioning and trajectory noise are distinct, but attribution and overhead are worse; isolate GC first because it remains untested.
- **Companion classifier + GC**: the companion adds discriminative intermediate gradients and GC prevents their shared mean component from dominating incoming weights. This may shape useful features more strongly than either alone, but it obscures whether the shallow WRN needed auxiliary supervision.
- **ShakeDrop + Lookahead**: stronger early residual perturbation explores a wider basin while slow weights damp the induced optimizer noise. This is plausibly synergistic, but it combines the least calibrated stochastic and state mechanisms and is too risky for one fixed-seed confirmation.

## Candidate Ideas

### Direct Canonical Lookahead on EXP002
**Summary**: Wrap EXP002's Nesterov optimizer with canonical parameter-only Lookahead at fixed `k=5`, `alpha=0.5`. Keep one FP32 slow copy, interpolate it toward fast parameters after every fifth inner step, copy it back, retain momentum buffers, and exclude BN running buffers. Evaluate slow parameters: every complete epoch is already synchronized because `195 % 5 == 0`, while a budget-truncated final epoch uses an exact temporary slow swap/fast restore. Audit sync ownership and distance without per-step device retention (`proposals/idea-03.md`).

**What it targets**: Optimizer trajectory variance using abundant memory and sparse fused state arithmetic, without another forward or a change to image exposure. Direct use below SAM/EMA isolates feedback averaging from EXP011's evaluation-time trajectory estimator (`02-system-understanding.md`).

**Reasoning**: NeurIPS 2019 reports CIFAR gains with the same canonical settings and no extra backward. Later lineage gains from SAM and EMA show that optimization path and variance matter. Unlike EMA, Lookahead feeds the slow trajectory back into subsequent training. However, the smoothly annealed SGD path may already be stable, retaining momentum after interpolation may produce mismatch, and frequent feedback can lag the cosine endpoint.

**Sources**: `knowledge/papers/lookahead-optimizer.md`; `experiments/002/04-analysis.md`; `experiments/011/04-analysis.md`; `proposals/idea-03.md`.

**Estimated Effort**: Medium.

**Risk Assessment**: Medium-high scientific risk and low-to-medium implementation/throughput risk. It may duplicate later EMA benefits or damp useful progress; canonical momentum retention is deliberate but can create state/parameter mismatch.

### Training-Only Middle-Stage Companion Classifier
**Summary**: Attach exactly one Kaiming-initialized `Linear(128,10)` head to the pooled post-block-3 representation. Train it with the exact same hard or area-weighted CutMix targets as the main head at weight 0.15 through 50% charged progress, taper to zero at 75%, and skip the head entirely in the clean tail and all evaluation. Isolate its initialization RNG and add only 1,290 disposable parameters (`proposals/idea-02.md`).

**What it targets**: Intermediate representation discriminativeness in the first four blocks, using one backbone forward and direct supervised gradient while preserving the parent clean final quarter. This is an orthogonal representation lever for the stable-generalization bottleneck (`02-system-understanding.md`).

**Reasoning**: Deeply-Supervised Nets reports CIFAR gains from companion objectives, and this attachment is much cheaper than EXP009's four active SE gates. Sharing exact CutMix semantics isolates supervision depth from target policy. The counter-case is substantial: a six-block pre-activation WRN is not demonstrably gradient-starved, and early linear separability plus CutMix/drop path may over-regularize useful nonlinear features.

**Sources**: `knowledge/papers/deeply-supervised-nets.md`; `experiments/002/04-analysis.md`; `experiments/009/04-analysis.md`; `proposals/idea-02.md`.

**Estimated Effort**: Medium.

**Risk Assessment**: Medium implementation/throughput risk and high scientific risk. The fixed attachment and coefficient are weakly calibrated, the head may constrain features, and small pooling/GEMM kernels can cost enough steps to erase a modest gain.

### Clean Reference-Ordered Full-Run GC Retry
**Summary**: Retry EXP017's unchanged scientific mechanism from EXP002: after every backward, materialize coupled L2 on all 44 gradients, centralize exactly the 16 convolution and one classifier regularized directions per output row, then apply unchanged PyTorch momentum/Nesterov with internal decay disabled to prevent duplication. The new experiment changes only its accuracy-blind measuring instrument: correct cadence arithmetic, fixed device-scalar finiteness, matched post-audit allocation snapshots, and explicit transient release (`proposals/idea-01.md`).

**What it targets**: Optimizer conditioning and stable generalization without another forward, model parameter, tunable coefficient, data change, or stochastic draw. This is aligned with the system's stable-generalization bottleneck while protecting EXP002's 27,950-step data exposure (`02-system-understanding.md`).

**Reasoning**: ECCV 2020 and the official CIFAR implementation provide relevant vision evidence and exact ordering. EXP017 passed mechanism, RNG, excluded-update, and numerical checks but emitted no latency or accuracy result; Claude classified it as a protocol crash and explicitly left GC untested. A retry therefore resolves an unanswered scientific question rather than repeating a failed method. The primary risk is functional redundancy with BatchNorm or removal of useful scale directions.

**Sources**: `knowledge/papers/gradient-centralization.md`; `experiments/017/04-analysis.md`; `03-experiment-learnings.md`; `proposals/idea-01.md`.

**Estimated Effort**: Low-to-medium.

**Risk Assessment**: Medium scientific risk, low-to-medium implementation/throughput risk. Seventeen reductions may be launch-bound, and a 0.10-point pass remains below single-seed resolution; a complete feasibility or accuracy failure would finally supply negative evidence for this exact composition.

## Review

Claude Opus selected Direct Canonical Lookahead (`01-idea-review.md`), scoring it 7/10 for both evidence and impact versus GC retry at 5/4 and the companion classifier at 3/3. The decisive evidence is same-lineage: SAM and EMA both improved optimizer geometry/trajectory behavior, while canonical Lookahead adds no forward and should preserve nearly all EXP002 exposure.

I adopted slow-parameter evaluation, achieved-step prominence, tighter 1% median/3% maximum latency gates, final-16 context, and normalized slow-fast distance auditing. With `k=5` and 195 batches per full epoch, all complete-epoch evaluations naturally occur just after synchronization; only the final partial epoch requires a swap, which must restore live fast parameters exactly. I did not adopt Claude's proposal to replace the formal verdict with final-16 mean because the goal's frozen necessary condition is explicitly parent-relative `best_test_acc`. Tail mean/range/premium remain mandatory context, and 95.61/95.71 remain global-frontier context rather than local gates.

## Idea Evaluation

Adopt the Claude verdict. Lookahead offers stronger in-repo evidence and lower exposure confounding than the other finalists. GC remains a legitimate untested idea after EXP017, but ubiquitous BatchNorm lowers its expected effect and its 17 reductions carry more timing risk. The companion head has the weakest architecture-matched evidence and the largest plausible representation harm.

## Chosen Idea
**Selected**: Direct Canonical Lookahead on EXP002

**Why this idea**:
Canonical parameter feedback directly targets optimizer-path variance, a mechanism class supported by the successful SAM and EMA lineage, while adding only one parameter copy and sparse fused interpolation. Fixed literature settings remove sweep degrees of freedom; exact sync/evaluation ownership and distance audits make either result interpretable. Starting below SAM/EMA isolates its causal effect and preserves the option to test composability only if the local result is strong.

**Hypothesis**:
Full-run `k=5`, `alpha=0.5` parameter-only Lookahead with retained Nesterov momentum will reduce trajectory variance without meaningful exposure loss, producing a valid `best_test_acc >=95.33%` from EXP002. A result of at least 95.53 with a stable final-16 slow-weight plateau is stronger but still noise-limited composability evidence; `>=95.61%` matches the global best and `>=95.71%` clears it. Slow-fast distance and achieved dose will distinguish active feedback from an effectively collapsed or underexposed mechanism.
