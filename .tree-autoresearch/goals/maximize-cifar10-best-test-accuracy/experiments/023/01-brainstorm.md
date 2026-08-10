# Brainstorm EXP-023
**Created**: 2026-08-06

## Web Search & Literature Review

- **A Unified Analysis of Mixed Sample Data Augmentation** (`knowledge/papers/mixed-sample-analysis.md`): Mixup and CutMix impose distinct input-gradient regularization, so CutMix's success does not make whole-image interpolation redundant.
- **RICAP: Random Image Cropping and Patching Data Augmentation** (`knowledge/papers/ricap.md`): four-image spatial composition and area-weighted labels improve CIFAR residual families, but excessive target softening and patch mechanics add risk.
- **When Does Label Smoothing Help?** (`knowledge/papers/when-label-smoothing-helps.md`): mild uniform target smoothing can improve generalization at negligible compute, though its benefit may overlap mixed-sample soft targets.
- **Time Matters in Regularization** (`knowledge/papers/time-matters-regularization.md`): strong early regularization can be removed late while retaining generalization benefit, supporting a clean final quarter for all three candidates.

## Experimental History Review

- EXP001's time-aware 2.75M-parameter pre-activation WRN reached 94.62% in 28,790 steps, with nearly zero late train loss, final 94.52%, and only 1,178.9 MiB peak. Its remaining issue is a generalization gap, not optimization, memory, or gross capacity.
- EXP001 has no failed direct children. Its only child, EXP002, used probability-0.5 CutMix during the first 75% and improved by 0.61 points to 95.23% without reducing throughput. This validates early mixed-example regularization but occupies only the rectangular patch-replacement branch.
- Later descendants show that replacing validated CutMix with low-dose manifold mixup did not help EXP004, but that substitution does not test direct whole-image Mixup on EXP001. CutMix-complementary Cutout on EXP011 also underdosed the run and lowered the tail.
- Dense widths 288/320, multi-launch FP32 attention, and dual-view CPU augmentation are closed systems directions. New candidates must keep core tensor shapes, use one backbone forward, and operate on the existing GPU batch or loss.
- The formal local threshold is 94.72%, substantially below the global best. A successful alternative branch need not beat 95.61 immediately; it must establish local progress with a mechanism distinct from CutMix.

## Collected Ideas

## Combinations

## Candidate Ideas

### Front-Loaded Probabilistic Mixup
**Summary**: During the first 75% of charged training, apply whole-image Mixup with probability 0.5 and `Beta(1,1)` interpolation, using dedicated seed-42 CPU/CUDA generators and soft-target cross-entropy; use hard clean examples for the final quarter. Preserve every EXP001 model, optimizer, drop-path, time schedule, batch, and evaluation setting.

**What it targets**: EXP001's near-zero-train-loss generalization gap by enforcing approximately linear predictions between examples and a different first-layer gradient geometry than rectangular CutMix.

**Reasoning**: The exact temporal envelope and probability are already validated by EXP002, isolating mixed-sample geometry rather than dose. Mixup adds only one elementwise interpolation and one extra cross-entropy term in the existing forward, with no extra backbone pass or CPU transport. The unified analysis explicitly distinguishes Mixup and CutMix regularization, so this can create a genuinely alternative branch even though both use soft labels.

**Sources**: `knowledge/papers/mixed-sample-analysis.md`; `knowledge/papers/time-matters-regularization.md`; EXP001 and EXP002 `04-analysis.md`; EXP006 `04-analysis.md` for the distinction from manifold-mixup substitution.

**Estimated Effort**: medium

**Risk Assessment**: Whole-image interpolation can create more globally ambiguous examples than CutMix and may underperform its spatially localized labels. Shape-preserving RNG isolation is feasible, but Mixup's dense elementwise work must still pass a paired latency/dose check. One seed and one fixed alpha are mandatory.

### Time-Decayed Label Smoothing
**Summary**: Use uniform label smoothing `epsilon=0.1` for clean examples through 75% charged progress, then linearly decay epsilon to zero over the final quarter so the endpoint is exact hard-label cross-entropy. Keep inputs, batching, RNG streams, and model forward unchanged.

**What it targets**: Overconfident fitting in EXP001 by preventing early logit saturation while guaranteeing clean low-LR refinement at the end.

**Reasoning**: Label smoothing is nearly compute-free, carries no data-pipeline or extra-forward risk, and has direct generalization evidence. Unlike the mixed-example candidates, it does not change inputs or consume RNG. The decay follows the validated phase-dependent regularization principle and avoids soft targets at the final model-selection plateau.

**Sources**: `knowledge/papers/when-label-smoothing-helps.md`; `knowledge/papers/time-matters-regularization.md`; EXP001 `04-analysis.md`.

**Estimated Effort**: low

**Risk Assessment**: Epsilon 0.1 may be too strong for the short schedule or reduce useful class separation, and its likely effect may be below the 0.10-point gate. Linear decay adds a scalar schedule but no empirical calibration is allowed after the result.

### Early Four-Patch RICAP
**Summary**: With fixed probability 0.5 during the first 75%, construct each training sample from four batch-permuted crops using `Beta(0.3,0.3)` split proportions and area-weighted cross-entropy, then return to clean hard-label training. All crop coordinates and permutations use dedicated generators; the backbone still runs once per step.

**What it targets**: The generalization gap through stronger spatial occlusion and multi-object composition than CutMix, encouraging secondary-feature use while retaining exact image size and one forward.

**Reasoning**: RICAP reports improvements on CIFAR WideResNet/PyramidNet families and combines feature removal with mixed targets. EXP002 validates early patch mixing, but RICAP's four-source tiling is a materially different geometry with potentially broader coverage. It stays GPU-side and avoids the dual-view loader failure.

**Sources**: `knowledge/papers/ricap.md`; `knowledge/papers/time-matters-regularization.md`; EXP002 `04-analysis.md`; EXP008 `04-analysis.md`.

**Estimated Effort**: high

**Risk Assessment**: Four crops/permutations and four loss terms add indexing, allocation, and launch overhead. Extreme beta splits can produce tiny patches and noisy labels; clipping and area accounting need exact audits. It has more stochastic and correctness degrees of freedom than Mixup.

## Review

Pending adversarial review.

## Idea Evaluation

Pending adversarial review.

## Chosen Idea
**Selected**: Pending adversarial review

**Why this idea**:
Pending adversarial review.

**Hypothesis**:
Pending adversarial review.
