# Brainstorm EXP-011
**Created**: 2026-07-24

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): residual capacity can be allocated through width or depth; local uniform widening was compute-poor, while selective 8x8 width produced the only post-EXP-002 positive delta.
- **Local FP32 fused-SGD opportunity check** (this brainstorm): a fail-closed 65/35 production-path benchmark measured only 1.033825x speedup and 146.70 projected passes, failing the preregistered 1.05x materiality gate. Fused SGD is discarded before candidate review and will not consume a scored run.

No network search was performed because this session is offline; accumulated architecture evidence and the new local timing check define the narrow space.

## Experimental History Review

- The accepted WRN-16-2 plus early mixup remains 94.07%. Seven experiments through EXP-009 regressed; EXP-010's `[32,64,160]` stage-3 widening reached 94.11% at 132.16 passes, the first positive delta but below the required 94.17%.
- EXP-010 shows low-resolution capacity can nearly offset a 6.9% exposure loss while preserving accepted loss. Its exact width is closed; a new allocation must differ structurally rather than tune a neighboring channel count.
- FP32 fused SGD is operationally insufficient at 3.38% measured speedup. Additive regularization remains a weak direction after CutMix, stronger mixup, and block dropout failures.
- The remaining gap is how to spend a similar low-resolution compute budget on feature transformation rather than channel expansion.

## Collected Ideas

Quick pass; candidates are recorded directly below.

## Combinations

Quick pass; candidates remain isolated.

## Candidate Ideas

### Low-Magnitude RandAugment
**Summary**: Add one torchvision RandAugment operation at magnitude 5/30 to training images while retaining accepted mixup and optimization.

**What it targets**: Image invariances absent from crop/flip/mixup rather than model capacity.

**Reasoning**: The method is literature-backed, locally available, and distinct from CutMix; it remains the developed non-architecture fallback.

**Sources**: `experiments/009/proposals/idea-03.md`; `knowledge/papers/randaugment.md`.

**Estimated Effort**: low.

**Risk Assessment**: It conflicts with repeated local additive-regularization failures, remains active in the hard-label tail, and `N=1,M=5` is not uniformly mild. Expected value is lower than the capacity signal.

### One Extra 8x8 Residual Block
**Summary**: Keep accepted stage widths `[32,64,128]` but change stage block counts from `[2,2,2]` to `[2,2,3]`, adding one pre-activation 128-to-128 residual block only at 8x8. All FP32 training choices remain accepted.

**What it targets**: Additional nonlinear transformation of abstract features rather than additional channel basis, using roughly the same low-resolution parameter/compute budget that produced EXP-010's positive delta.

**Reasoning**: The extra block adds about 295,424 parameters and 18.87M convolution MACs, closely matching selective width's +269,888 parameters / +17.24M MACs but spends them on depth. This is a materially distinct capacity allocation, not a neighboring-width retry, and should retain similar roughly 130-pass exposure.

**Sources**: EXP-010 analysis and preflight; `knowledge/papers/wide-residual-networks.md`; accepted `train.py` block structure.

**Estimated Effort**: medium.

**Risk Assessment**: The accepted optimizer may under-train a deeper tail; extra depth may add redundant transformations or worsen gradient flow. Wide-ResNet literature generally favors width over depth. A strict matched preflight must retain >=85% throughput / >=120 projected passes, and a negative run rejects only `[2,2,3]`.

### Bottlenecked 8x8 Expansion Block
**Summary**: Add a single pre-activation 1x1-3x3-1x1 residual bottleneck at 8x8 that expands internal features and returns to 128 channels, preserving the external stage/classifier interface.

**What it targets**: Richer low-resolution feature transformation with fewer MACs than a full 128-to-128 basic block, potentially retaining more exposure while adding architectural capacity.

**Reasoning**: EXP-010's positive delta supports low-resolution capacity, and a bottleneck can decouple internal capacity from interface width. Returning to 128 avoids changing final BN/classifier and may be more compute-efficient.

**Sources**: EXP-010 analysis; standard residual bottleneck concept; `train.py` architecture.

**Estimated Effort**: high.

**Risk Assessment**: This introduces a new block family, expansion ratio, additional BN placement, and initialization behavior with no local validation. The larger implementation/search surface weakens attribution and risks spending compute on an under-tuned architecture.

## Review

The blind offline critic selected one extra 8x8 basic block and verified exact totals: +295,424 parameters, +18,874,368 MACs/image, for 987,098 parameters and 119,981,312 MACs/image overall. I accept its caveat that similar static cost to EXP-010 does not imply similar wall time because the block adds sequential launches/dependencies. A matched 65/35 production-path preflight must retain >=85% throughput and >=120 projected passes. A negative run closes only `[2,2,3]`; no fused, bottleneck, RandAugment, LR, or architecture fallback is allowed. Full review is in `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. `[2,2,3]` scored 7.5/10 for both evidence and impact because it tests nonlinear depth at nearly EXP-010's capacity/compute budget while retaining the accepted 128-channel interface. The bottleneck is not experiment-ready due unspecified internal width/order/costs, and RandAugment remains contradicted by repeated regularization failures.

## Chosen Idea
**Selected**: One Extra 8x8 Residual Block

**Why this idea**:
It is a precise, materially distinct depth-versus-width test of the only positive post-baseline signal. Adding an identity residual branch at 8x8 preserves stage interfaces and spends nearly the same compute as EXP-010 on additional transformations rather than channels. Scoring is conditional on the strict throughput/pass gate.

**Hypothesis**:
Stage block counts `[2,2,3]` at accepted widths `[32,64,128]` will improve abstract feature refinement enough to reach `best_test_acc >=94.17%` while retaining at least 120 passes. Any valid lower score is no-improvement and a stable negative at >=120 passes rejects this exact depth allocation.
