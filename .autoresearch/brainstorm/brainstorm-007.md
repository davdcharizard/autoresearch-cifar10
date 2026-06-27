# Brainstorm EXP-007
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new search this loop — the decision rests on in-project measurements plus already-distilled sources:
- **WRN (arXiv 1605.07146)** (knowledge/README.md): width gains at 16–22-layer depth continue to 8–12x given sufficient training — the external prior that capacity headroom exists above 4x.
- **project-insights.md § High (EXP-005)**: channel counts must be multiples of 32/64 on H20 — 6x widths (96/192/384) are all 32-aligned, avoiding the 5x penalty.
- **In-project throughput model**: aligned-width time-per-epoch scales sublinearly in FLOPs (EXP-002: 2.85x time for 4x FLOPs, exponent ~0.76). At 6x (2.25x FLOPs) compiled: 139 / 2.25^0.76 ≈ **75–80 epochs projected** — above the ~70-epoch starvation floor estimated from EXP-002/005.

## Experimental History Review

- **Trajectory**: 91.97 → 93.16 → 95.23 (4x) → 94.41 (8x fail) → 96.06 (RE) → 96.23 (TA) → 95.12 (5x fail) → **96.71 (compile)**. Baseline: 96.71 @ 1990397.
- **Patterns**: compile validated (1.22x img/s, pure profit, keep always); EXP-006's headline insight — final=best runs at 4x were STILL epoch-starved, so capacity gains were throughput-throttled, not exhausted; regularizers validated and kept.
- **Failed Approaches (count 2, Medium)**: capacity without throughput — 8x@40ep, 5x@52ep. BOTH predate the compile unlock, and 5x carried the alignment penalty. This loop differs on both counts: throughput is 1.22x higher AND 6x channels are aligned — the plan must (and does) justify the retry via these two changed factors.
- **Protocol Findings**: eval overhead ~1s/epoch toward the 600s cap; a 6x run has FEWER epochs → comfortable wall clock (~400s projected).
- **Untried gaps**: aligned 6x on the compiled recipe (the explicit next step in exp-report-006), Mixup/CutMix at 139 epochs, compile mode=max-autotune, wider-but-shallower topology.

## Candidate Ideas

### 1. Aligned width 6x (96/192/384, ~9.6M params) on the compiled recipe
**Summary**: WIDTH_MULT 4 → 6; everything else (compile, TA, RE, one-cycle) unchanged. Projected ~75–80 epochs.

**Reasoning**: EXP-006 proved capacity gains were being throughput-throttled (extra epochs converted at +0.48pp even at "converged" 4x), and the two prior width failures are both explained by since-removed factors: no compile (EXP-002/005) and channel misalignment (EXP-005). 6x is the canonical aligned step: 96/192/384 are multiples of 32, the projected epoch count sits just above the starvation floor, and WRN says the capacity headroom is real. If it works, expect a meaningful step (+0.2–0.6pp); if it fails with final=best at ~75 epochs, the width direction is closed at this hardware/budget for good — also valuable.

**Sources**: reports/exp-report-006.md § Next Steps (medium-high); project-insights § High (alignment rule); experiment-indices rows 001/002/005/006 (scaling datapoints); arXiv 1605.07146.

**Estimated Effort**: trivial (one constant)

**Risk Assessment**: Genuine uncertainty — 75–80 epochs is just above the measured starvation region, and wider nets fit augmented data slower. Clean failure mode (no-improvement). VRAM ~3.5GB fine; wall clock DROPS (fewer evals). Compile time may grow slightly with model size (still tens of seconds).

### 2. Mixup at the 139-epoch schedule
**Summary**: Batch-level Mixup (Beta(0.2) or 1.0) in the training step; loss = lam*CE(y_a) + (1-lam)*CE(y_b).

**Reasoning**: EXP-006 re-promoted this from sub-noise (epochs bind less now), and Mixup's published CIFAR-10 gains on WRN-28-10 are +0.5–1pp at full schedules. But it interacts with label smoothing and the TA+RE stack — three regularizers may over-constrain a 4.29M-param net; the measured in-project trend (RE +0.83 → TA +0.17) still points to capacity, not more regularization.

**Sources**: arXiv 1710.09412 (standard knowledge); exp-report-006 § Unexplored Avenues.

**Estimated Effort**: low-medium (loss-path change — more bug surface than a constant)

**Risk Assessment**: Moderate chance of landing in noise at current capacity; loss-path edits risk subtle bugs; failure semi-informative.

### 3. compile mode="max-autotune"
**Summary**: `torch.compile(model, mode="max-autotune")` — extra kernel search for a few more % img/s at 1–3 min extra startup.

**Reasoning**: Incremental on a lever just banked; expected gain (~3–7% img/s → ~4–9 extra epochs → likely <0.1pp) sits below the bar on its own. Better bundled mentally as "headroom if width needs it" than as a standalone loop.

**Sources**: PyTorch compile docs; exp-report-006 § Unexplored Avenues.

**Estimated Effort**: trivial

**Risk Assessment**: Autotune startup eats wall-clock margin; standalone gain likely sub-bar — low information, low expected value.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest case ever assembled for a width retry: a fresh in-project causal demonstration that epochs convert (EXP-006), both prior width failures explained by now-removed factors, the alignment rule satisfied by construction, and the WRN prior. Idea 2 has good external evidence but contradicts the in-project diminishing-regularization trend. Idea 3's expected effect is below measurement resolution.

**Mechanism clarity**: Idea 1 — capacity step at an epoch count above the starvation floor, enabled by measured throughput. Idea 2 — regularization where the local trend says capacity binds. Idea 3 — marginal throughput.

**Expected impact**: Idea 1: +0.2–0.6pp if the floor estimate holds; −0.3 to −0.8 if not. Idea 2: 0–0.3pp. Idea 3: <0.1pp.

**Risk profile**: all fail cleanly; Idea 1's failure is also maximally informative (closes the width direction permanently). Idea 2's failure is ambiguous (bug vs over-regularization vs capacity).

**Feasibility**: Idea 1 and 3 one-line; Idea 2 touches the loss path.

## Chosen Idea
**Selected**: Idea 1 — Aligned width 6x on the compiled recipe

**Why this idea**:
EXP-006 specifically demonstrated the mechanism this retry depends on (epochs convert into accuracy when throughput rises), both prior width failures are explained by factors that no longer apply (no compile; unaligned channels), and the experiment is decisive in both directions — success banks the largest available gain, failure at aligned ~75 epochs closes the capacity direction conclusively.

**Hypothesis**:
Raising WIDTH_MULT from 4 to 6 (96/192/384, ~9.6M params) on the compiled TA+RE recipe will raise best_test_acc from 96.71% to ≥96.85%, because EXP-006 showed accuracy at 4x was throughput-throttled rather than capacity-saturated, and the projected 75–80 aligned-channel epochs sit above the ~70-epoch starvation floor; epoch count in [70, 85] and total wall clock ≤ ~430s are the predicted execution signatures, with final≈best plus depressed accuracy as the failure signature that would close the width direction.
