# Brainstorm EXP-010
**Created**: 2026-07-24

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): allocating CIFAR capacity to shallow width can improve accuracy and compute efficiency, but the prior uniform-width preflight showed that high-resolution widening is too expensive here.
- **EXP-009 thorough proposal set** (`experiments/009/proposals/idea-02.md`; `experiments/009/01-idea-review.md`): selective stage-3 width adds 39% parameters for 17% MACs and was the highest-impact remaining finalist, contingent on strict throughput gates.

No network search was performed because this session is explicitly offline; EXP-009 already completed the relevant broad local review.

## Experimental History Review

- EXP-001/002 established the 94.07% accepted WRN-16-2 plus alpha-0.2 mixup through 65%. Seven follow-ups have now regressed.
- Regularization, cutoff/strength, late decay removal, and zero-floor schedule changes all lost accuracy at normal exposure; those accepted choices remain fixed.
- BF16 produced 159.07 passes, 12.1% above accepted, but scored 93.81% with worse loss. More updates through a changed numerical path are not sufficient; return to FP32.
- Uniform WRN-16-3 was infeasible at 56.8% throughput / 80.6 projected passes. The remaining architecture gap is compute-aware capacity allocation: add abstract-stage channels without paying high-resolution width cost.

## Collected Ideas

Quick pass; candidates are recorded directly below.

## Combinations

Quick pass; candidates remain isolated.

## Candidate Ideas

### FP32 Fused SGD
**Summary**: Keep the accepted model and all arithmetic precision unchanged while enabling PyTorch's native fused SGD implementation on the two existing parameter groups. Score only if a matched production-path preflight shows meaningful throughput gain.

**What it targets**: Extra FP32 updates without BF16 rounding, directly following EXP-009's unresolved question of whether exposure can help when model numerics remain accepted.

**Reasoning**: Optimizer fusion can reduce kernel-launch overhead while preserving the SGD update definition and requires no dependency. It is a narrower numerical intervention than autocast and retains architecture/capacity.

**Sources**: `train.py`; EXP-009 analysis unexplored avenues.

**Estimated Effort**: low.

**Risk Assessment**: This tiny WRN is convolution-bound, so optimizer fusion may deliver negligible gain; fused kernels can also differ in floating-point order. A threshold-setting preflight would need a preregistered gain large enough to matter, and no local timing evidence currently supports it.

### Low-Magnitude RandAugment
**Summary**: Apply one torchvision RandAugment operation at magnitude 5/30 after crop/flip and before tensor conversion, preserving the accepted mixup cutoff and optimizer recipe.

**What it targets**: Geometric and photometric invariances absent from crop, flip, and mixup, using an established CIFAR augmentation family.

**Reasoning**: RandAugment is locally available and literature-backed, while `N=1, M=5` limits but does not eliminate strength. It is materially different from CutMix and feature dropout.

**Sources**: `experiments/009/proposals/idea-03.md`; `knowledge/papers/randaugment.md`; EXP-003/005/006 analyses.

**Estimated Effort**: low.

**Risk Assessment**: It stacks with a recipe that has repeatedly regressed under added regularization, and magnitude-independent operations are not uniformly mild. CPU transform cost and full-tail augmentation are additional risks; it has the weakest local fit.

### Selective 160-Channel Final Stage
**Summary**: Replace accepted stage widths `[32,64,128]` with `[32,64,160]`, leaving both high-resolution stages and the full FP32 optimization/data recipe unchanged. This adds 269,888 parameters (39.0%) but only 17.0% convolution/linear MACs.

**What it targets**: Abstract 8x8 class-separating capacity under a fixed compute budget, allocating parameters where spatial reuse cost is lowest rather than repeating rejected uniform widening.

**Reasoning**: The topology has 961,562 parameters and a prior of roughly 90.3% throughput / 128.1 passes, far above uniform width 3's measured 56.8% / 80.6 passes. Wide-residual evidence supports width, and EXP-009's critic gave this candidate the highest impact score (8/10) despite weaker direct bottleneck evidence.

**Sources**: `experiments/009/proposals/idea-02.md`; `experiments/009/01-idea-review.md`; `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: medium.

**Risk Assessment**: The unchanged 64-channel stage may bottleneck information, the abrupt 64-to-160 transition may optimize poorly, and exposure loss can dominate. Proceed only if a matched FP32 preflight retains at least 85% throughput and projects at least 120 passes; a negative result rejects this allocation, not capacity generally.

## Review

The offline blind critic selected selective stage-3 width. I accept its main refinements: the 90.3% throughput estimate is only a prior from structurally different endpoints, so a fresh matched H20 preflight must time pinned copies and the full production step in balanced order; the existing 85% retention and 120-pass gates remain fixed. A negative stable run will reject only `[32,64,160]`, not capacity generally, and no LR, width, or regularization rescue is allowed. Full concerns and scores are in `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Selective width scored 7.5/10 for evidence and 8/10 for impact: its exact cost allocation and strict gate offer meaningful headroom after uniform widening failed. Fused SGD lacks any measured opportunity and is likely a small fraction of the convolution-bound step; RandAugment conflicts with the strongest repeated local regularization pattern.

## Chosen Idea
**Selected**: Selective 160-Channel Final Stage

**Why this idea**:
It adds substantial FP32 representational capacity only at 8x8 resolution, where 39% more parameters cost 17% more MACs, while preserving the validated high-resolution stages and complete accepted training recipe. The architecture will be scored only if a production-path preflight retains at least 85% throughput and projects at least 120 passes.

**Hypothesis**:
Stage widths `[32,64,160]` will provide better abstract class separation than `[32,64,128]` while retaining enough fixed-budget optimization to reach `best_test_acc >=94.17%`. A valid run with at least 120 passes but below 94.17% falsifies this allocation regardless of loss or parameter count.
