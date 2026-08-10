# Brainstorm EXP-012
**Created**: 2026-08-06

## Web Search & Literature Review

- **RICAP: Random Image Cropping and Patching Data Augmentation for Deep CNNs** (`papers/ricap.md`; https://proceedings.mlr.press/v95/takahashi18a.html)
  Controlled WideResNet CIFAR results show spatial feature removal is a strong regularizer: cutout reduced error from 3.89% to 3.08%, while RICAP improved further by combining cropping, patching, and area-soft labels. RICAP overlaps CutMix, but Cutout on non-CutMix batches is differentiated and cheap.
- **PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions** (`papers/polyloss.md`; https://openreview.net/forum?id=gSdSJoenupI)
  Poly-1 changes cross-entropy with one first-order probability term and essentially no model overhead, producing gains on image classification while requiring a task-dependent coefficient.
- **Four Things Everyone Should Know to Improve Batch Normalization** (`papers/four-things-batch-norm.md`; https://arxiv.org/abs/1906.03548)
  Ghost BatchNorm can regularize medium batches, while weight decay on BN gamma/beta may help residual networks. The current optimizer already decays BN parameters, so excluding them is poorly supported; GhostBN remains an untried but state-sensitive option.
- **Averaging Weights Leads to Wider Optima and Better Generalization** (`knowledge/papers/stochastic-weight-averaging.md`)
  Weight averaging benefits from late trajectory diversity, suggesting that the current EMA might combine with a non-collapsing low learning-rate tail, but any reheat risks losing the proven clean convergence.
- **ShakeDrop Regularization for Deep Residual Learning** (`knowledge/papers/shakedrop.md`)
  Residual-branch scaling can improve Wide ResNets when stabilized, offering a cheap representation regularizer distinct from output loss changes.

## Experimental History Review

- The successful lineage is BASE 91.51 -> WRN EXP-001 94.62 -> CutMix EXP-002 95.23 -> clean period-two SAM EXP-004 95.40 -> full-state clean-tail EMA EXP-011 95.61. EXP-011 is the fresh global best with no children and a stable late EMA mean of 95.493 rather than a stable 95.61 level.
- EXP-011 retained 25,798 steps, used only 1,222 MiB, and added EMA at 0.9996x paired parent latency. Compute permits sparse/fused additions and memory is abundant, but extra full forwards compete directly with validated training dose.
- Failed children from EXP-004 reject half-overlap identity reuse, replacing validated CutMix with low-dose manifold mixup, uncalibrated literature-scale ASAM, and dual-view CPU RandAugment. They do not reject additive single-view GPU augmentation, calibrated losses, or cheap classifier/normalization changes.
- EXP-009 rejects multi-launch SE gates at +20.7% latency. EXP-010 shows architecture shape affects H20 latency and fixed-seed initialization; architecture changes require paired measurement and package-level interpretation.
- The current limiter is stable generalization above the 95.493 tail plateau. Since the formal child threshold is 95.71 and sub-0.30-point changes sit near observed selection noise, prioritize mechanisms with a plausible 0.25-0.30-point stable effect while preserving CutMix/SAM/EMA exposure.

## Collected Ideas

- **CutMix-complementary GPU Cutout** - On early batches where the existing 0.5 CutMix gate is not selected, mask a dedicated-RNG 16x16 square per image or compact group directly on the GPU, ending at the same 75% boundary. This retains every validated CutMix batch, adds no second view or target ambiguity, attacks salient-feature overreliance, and has direct WideResNet/CIFAR evidence. The main risks are overlap with CutMix's occlusion mechanism and mask-kernel overhead.
- **Soft-target-aware Poly-1 loss** - Replace each cross-entropy term with `CE + epsilon*(1-p_target)`, using area-weighted target probability for CutMix and the same loss on both SAM passes. This targets decision-boundary generalization with negligible compute and preserves architecture/data exposure. The coefficient is task-sensitive, and paper evidence is not specific to CIFAR or mixed targets.
- **EMA-diversifying late LR floor** - Preserve the parent schedule through 75%, then keep the final cosine tail above a preregistered small floor or use a bounded single cycle while EMA averages the trajectory. This directly targets the 95.49 plateau using SWA's diversity mechanism, with no extra forward. It risks sacrificing the parent's validated low-loss convergence and conflates averaging with a schedule change.
- **Single-view GPU RandAugment** - Revisit EXP-008's augmentation family with exactly one image view and GPU-side tensor operations, applied only in the CutMix prefix with a dedicated generator. This addresses the prior loader bottleneck rather than repeating it and could add color/geometric invariance beyond crop/flip/CutMix. Faithful policy implementation without new dependencies is complex, and geometric transforms can be costly or distort CIFAR features.
- **Ghost BatchNorm groups** - Compute training statistics over fixed sub-batches within batch 256 while maintaining one online forward and persistent buffers compatible with EMA. It injects normalization noise with literature support and near-zero arithmetic growth. Vectorization, SAM's second-pass suppression, and running-stat interpretation are difficult; kernel fragmentation may erase throughput.
- **Stabilized ShakeDrop prefix** - Replace binary residual drop-path masks with depth- and time-scaled ShakeDrop coefficients during the first 75%, decaying to the exact clean tail. It uses a literature-supported residual regularizer while preserving CutMix/SAM/EMA phases. This replaces a validated mechanism, alters RNG/gradient semantics, and may need coefficient tuning larger than the experiment can justify.
- **Classifier geometry regularizer** - Add a tiny penalty on normalized classifier-row correlations or use squentropy to suppress confident incorrect classes, shaping the ten-class boundary without another forward. This exploits a different error surface and costs almost nothing. It introduces an underdetermined coefficient, and calibration improvement need not translate to top-1 accuracy.
- **Moonshot: batch-centroid feature compactness** - On non-CutMix batches, use the existing penultimate features to pull examples toward detached per-class batch centroids and repel active class centroids. This adds no second model evaluation and directly reshapes representation geometry. Small per-batch class counts, CutMix compatibility, and arbitrary loss scale make it a high-risk exploratory bet.

## Combinations

- **CutMix-complementary Cutout + dedicated RNG isolation**: use a separate CUDA generator and only the already-observed non-CutMix branch, so added occlusion cannot perturb shuffle, drop-path, CutMix, SAM replay, or EMA cadence. The combination is stronger than generic Cutout because it preserves validated mixed-label exposure and yields exact mechanism counters.
- **Poly-1 + clean-tail EMA**: optimize a probability-sensitive objective throughout the online trajectory while retaining the successful EMA evaluator. Poly-1 may raise the stable boundary quality and EMA may smooth its iterates; either alone addresses only loss geometry or trajectory variance.
- **Bounded late LR floor + horizon EMA**: maintain modest late iterate diversity specifically inside the existing 18.75-second averaging horizon. This is more coherent than changing LR alone or averaging a fully collapsed trajectory, but it has the largest risk of degrading final convergence.
- **Single-view GPU augmentation + CutMix complement gating**: apply a cheap color/translation operation only when CutMix is absent rather than transporting a second view. This directly resolves EXP-008's bottleneck and avoids substituting away proven CutMix, though policy breadth must be constrained for feasibility.

## Candidate Ideas

### EMA-Aligned Bounded Late LR Floor-Cycle
**Summary**: Preserve parent LR until approximately 88.75% progress, then take an upper envelope with a fixed 0.006-0.014 sinusoidal floor over four nominal 18.75-second cycles, aligned to the EMA half-life. No model/data/optimizer/EMA implementation changes. See `proposals/idea-03.md`.

**What it targets**: Insufficient late trajectory diversity for EMA, whose final live/EMA parameter distance was only 1.51%, while avoiding an extra forward (`02-system-understanding.md`).

**Reasoning**: SWA literature connects useful averaging with constant/cyclic late LR. The proposed envelope is continuous at the clean-tail boundary, changes only the final 33.7 charged seconds, and increases full-run integrated LR dose by only about 0.67%.

**Sources**: `knowledge/papers/stochastic-weight-averaging.md`; `knowledge/papers/how-to-scale-your-ema.md`; `proposals/idea-03.md`; EXP-011.

**Estimated Effort**: low implementation, high schedule/integrity verification

**Risk Assessment**: Values and phase coupling are not literature-validated under SAM, only 1.8 cycles are active, and greater trajectory distance is not greater accuracy. Reheating can degrade the parent's low-loss clean convergence and BatchNorm compatibility.

### CutMix-Complementary GPU Cutout
**Summary**: Apply center-sampled nominal 16x16 per-image mean-fill Cutout with edge clipping to every early batch where the existing CutMix gate is not selected, ending at progress 0.75. A dedicated seed-43 CUDA generator and a precomputed 1,024-mask bank keep parent RNG streams isolated; every validated CutMix batch and the entire SAM/EMA tail remain unchanged. See `proposals/idea-01.md`.

**What it targets**: Salient-feature overreliance and the stable 95.493 EMA-tail plateau, while respecting the measured constraint that extra model views and CPU augmentation lose too much exposure (`02-system-understanding.md`).

**Reasoning**: RICAP's controlled WideResNet/CIFAR table reports 16x16 Cutout reducing error from 3.89% to 3.08%. Complement gating differentiates the mechanism from CutMix rather than substituting it, and GPU single-view masking directly follows the reusable lesson from EXP-008's loader failure.

**Sources**: `papers/ricap.md`; `proposals/idea-01.md`; EXP-008; `03-experiment-learnings.md` loader and RNG findings.

**Estimated Effort**: medium

**Risk Assessment**: CutMix already supplies occlusion, so the effect may be redundant; masking every non-CutMix early batch removes unmodified early supervision and may over-regularize. Several small GPU kernels on about 40.7% of steps could reduce dose, requiring a <=1.01x parent-relative preflight.

### Soft-Target-Aware Poly-1
**Summary**: Replace every training CE term with `CE(q,p) + epsilon*(1-sum(q*p))` at fixed `epsilon=1.0`, using exact area-weighted CutMix targets and the same hard-label loss on both SAM passes. Architecture, data, optimizer, schedules, EMA, and frozen evaluation remain unchanged. See `proposals/idea-02.md`.

**What it targets**: Decision-boundary quality above the 95.493 plateau through loss geometry, without consuming model, sample, SAM, or EMA exposure (`02-system-understanding.md`).

**Reasoning**: PolyLoss reports image-classification gains from changing only the leading target-probability basis. The algebra extends coherently to CutMix soft targets and costs only a ten-class softmax/gather, making it a clean additive test rather than another mechanism substitution.

**Sources**: `papers/polyloss.md`; `proposals/idea-02.md`; EXP-006 substitution lesson; EXP-011 package.

**Estimated Effort**: low implementation, medium verification

**Risk Assessment**: The best coefficient is task-dependent, epsilon one is not locally validated, and the paper does not establish mixed-target or SAM behavior. The extra term may duplicate CutMix's soft-target regularization or change SAM perturbation geometry unfavorably.

## Review

Claude selected CutMix-complementary GPU Cutout and found no hard-constraint violation. The significant concerns were that the proposal's valid-top-left geometry was more center-biased/severe than reference Cutout; its `randint(out=..., device=...)` hot path was defective; the latency gate was too loose; mean-fill depended silently on `std=(1,1,1)`; and the cited 0.81-point effect must be discounted for CutMix redundancy and judged against the 95.493 stable tail.

The selected spec now samples centers uniformly over 32x32 and clips a nominal 16x16 window, uses `Tensor.random_` with a 1,024-mask bank, asserts normalization semantics, tightens median latency to <=1.01x with >=25,500 projected steps, and predicts only a 0.20-0.40 stable lift. The review suggested 47 top-left values, but the reference operator samples 32 image centers; the refinement adopts the underlying geometry concern with the correct center parameterization. Full complement dose is retained deliberately because it covers about 40.7% of all steps, below literature all-image Cutout, while preserving every CutMix batch and the entire tail. Poly-1 is deferred until its effective per-example gradient scaling and epsilon can be mechanistically normalized; the LR cycle is rejected because it primarily raises tail variance and worsens EMA/BN mismatch.

## Idea Evaluation

Adopted Claude's scored pick from `01-idea-review.md`. Cutout scored 7/10 for both evidence/reasoning and impact, versus Poly-1 at 4/10 and 5/10 and the LR floor-cycle at 3/10 and 4/10. After the geometry/API/gate refinements above, it is the only finalist with matched WideResNet/CIFAR evidence at an effect scale plausibly above the stable +0.22 requirement.

## Chosen Idea
**Selected**: CutMix-Complementary GPU Cutout

**Why this idea**:
It directly follows the recorded remedy for EXP-008 by using a single GPU-resident view, remains additive rather than replacing validated CutMix, and has the strongest architecture/dataset-matched evidence. Dedicated RNG and complementary gating preserve the parent's stochastic streams and full SAM/EMA tail. The refined reference geometry and tight preflight make its dose and throughput risk measurable before the sole metric run.

**Hypothesis**:
Center-sampled, edge-clipped Cutout on every early non-CutMix batch will preserve at least 25,500 projected optimizer steps, keep all CutMix/SAM/EMA integrity audits exact, reach `best_test_acc >=95.71%`, and raise the final-16 EMA mean from 95.493125 to at least 95.69 by forcing the WRN to use secondary object features without removing validated mixed-label exposure.
