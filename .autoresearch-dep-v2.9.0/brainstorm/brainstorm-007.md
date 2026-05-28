# Brainstorm EXP-007
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

WRN paper (https://arxiv.org/abs/1605.07146) Table 5: WRN-16-4 ≈ 94.98%, WRN-28-4 ≈ 95.7% on CIFAR-10 at 200 epochs. The n=3 (20-layer), k=4 design point interpolates to ~95.3%. With our AMP-enabled recipe (106 epochs vs 200), a discount to ~94.5-95.0% is expected.

## Experimental History Review

Current baseline: **94.44%** (EXP-005, width-2x + aug + WD=5e-4 + AMP, 106 epochs).

Key learnings:
- Width-2x (EXP-001): +0.57pp from capacity alone. Width-4x is the natural next step.
- AMP (EXP-005): 1.54x throughput, VRAM dropped from 598 MB to 266 MB. Width-4x with AMP should still fit comfortably in the H20's 98 GB.
- Schedule (EXP-006): (0.5, 0.75) is near-optimal. Don't change it.
- EXP-004: per-step overhead matters. Width-4x will increase per-step time but AMP compensates.

Width-4x parameters: ~4.3M (16x baseline's 270K). Per-step time will increase roughly 2-3x over width-2x (~15-22ms), yielding ~35-50 epochs with AMP. The wall-clock-fractional schedule handles this.

## Candidate Ideas

### 1. Width-4x (WIDTH_MULT=4) with AMP

**Summary**: Change `WIDTH_MULT = 2` to `WIDTH_MULT = 4`, quadrupling channel widths to {64, 128, 256}. ~4.3M params. Everything else unchanged from EXP-005 (AMP, aug, WD=5e-4, schedule (0.5, 0.75)).

**Reasoning**: The WRN paper places the n=3, k=4 point at ~95.3% on CIFAR-10. With AMP, we should get 35-50 epochs in 300s. The wall-clock-fractional schedule adapts automatically. VRAM with AMP at width-2x was 266 MB — width-4x should be ~1 GB, still well within the H20's 98 GB.

**Sources**: WRN paper Table 5, EXP-001 (width-2x success), EXP-005 (AMP VRAM headroom).

**Estimated Effort**: Very low — one constant change.

**Risk Assessment**: Medium. The concern is epoch count: width-4x with AMP may only get ~35-50 epochs, which is in the regime where EXP-006 showed reduced high-LR exploration hurts. However, the wider model's higher capacity may compensate — each epoch sees more features. The FP16 instability at LR=0.01 may be worse with wider layers. Worst case: a no-improvement if the epoch count is too low for convergence.

### 2. Batch size 256 + LR 0.2 with AMP (on width-2x)

**Summary**: Double BATCH_SIZE to 256 and LR to 0.2 (linear scaling). Keep width-2x and AMP. The larger batch increases GPU utilization, potentially adding 20-30% more epochs.

**Reasoning**: With AMP at width-2x, the GPU is underutilized (266 MB VRAM, ~7ms step time). Larger batches improve throughput. The linear scaling rule preserves optimization dynamics.

**Sources**: Goyal et al. 2017, EXP-005 VRAM data.

**Estimated Effort**: Very low — two constants.

**Risk Assessment**: Low-medium. Generalization gap from larger batches. LR=0.2 with FP16 may be less stable.

### 3. Width-3x (WIDTH_MULT=3) with AMP — intermediate capacity

**Summary**: Set WIDTH_MULT=3, channels {48, 96, 192}. ~2.4M params. A conservative capacity increase that should get ~60-80 epochs with AMP.

**Reasoning**: Width-3x is an intermediate point between the validated width-2x and the riskier width-4x. It preserves a higher epoch count (closer to width-2x's 106) while still increasing capacity significantly.

**Sources**: WRN paper interpolation between k=2 and k=4 points.

**Estimated Effort**: Very low — one constant.

**Risk Assessment**: Low. More conservative than width-4x, preserving epoch count while still gaining capacity.

## Idea Evaluation

**Evidence**: Candidate 1 (width-4x) has the strongest literature anchor (WRN-paper ~95.3%). Candidate 3 (width-3x) has no direct literature anchor but interpolates. Candidate 2 (batch-size) is orthogonal to capacity.

**Expected impact**: Candidate 1 targets ~95%, the highest ceiling. Candidate 3 targets ~94.5-95.0%. Candidate 2's impact is uncertain.

**Risk**: Candidate 3 is safest (higher epoch count). Candidate 1 is riskier (low epoch count) but highest ceiling. Candidate 2 is orthogonal.

**Decision**: The autoresearch trajectory has been following the WRN paper's recipe alignment — width-2x, augmentation, WD=5e-4, and now AMP. The natural culmination is width-4x, which targets the WRN-paper's ~95% ceiling. The epoch count risk is real but: (a) AMP compensates by 1.54x, (b) the model's higher capacity per-epoch may compensate for fewer epochs, and (c) EXP-005 showed that the extended LR=0.001 phase is where gains accumulate — even with 35-50 epochs, the 0.001 phase gets ~12-15 epochs which should still deliver convergence.

## Chosen Idea

**Selected**: Candidate 1 — **Width-4x (WIDTH_MULT=4) with AMP**

**Why this idea**: Targets the WRN paper's highest single-GPU CIFAR-10 anchor (~95.3% for n=3, k=4). With AMP's VRAM efficiency (266 MB at width-2x → ~1 GB at width-4x), the H20 has massive headroom. This is the highest-ceiling single-axis change available.

**Hypothesis**: Changing WIDTH_MULT from 2 to 4 will raise best_test_acc from 94.44% to **94.8-95.3%** by approximately doubling the model capacity (4.3M vs 1.07M params). The wall-clock-fractional schedule handles the reduced epoch count. The improvement bar is 94.54%.
