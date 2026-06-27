# Brainstorm EXP-022
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only. -->

## Web Search & Literature Review

No new searches — the relevant scaling-rule literature is already characterized in prior loop documents; re-grounded for this ideation:

- **Hoffer et al., "Train longer, generalize better" (arXiv 1705.08741)**: √-scaling of the learning rate with batch size preserves the gradient-noise covariance structure, where linear scaling (Goyal et al. 1706.02677) systematically overshoots for small datasets/models — the canonical diagnosis-and-fix pair for exactly the failure EXP-012 exhibited (bouncy hot phase, mid-run deficit at linearly-scaled peak 0.8).
- **knowledge/README References (cifar10-fast)**: batch-512 one-cycle lineage; notes large-batch mixed precision dominates in short budgets. EXP-012 already validated the throughput half at 1024 (+8% img/s, 151 epochs, clean signatures) in the default-compile regime — the SAME execution arithmetic the recipe is tuned to (the EXP-021 numerics-equivalence requirement).
- **Bag of Tricks / RegNet / WRN / airbench caveat rows** (knowledge/README): the fixed-epoch evidence class has failed to transfer 4×; this loop's candidates are deliberately restricted to in-project-measured mechanisms.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006). **Sixteen consecutive misses (EXP-007…021).**
- **Axis closure is now total** (goal-learnings § Patterns High + EXP-021): constants bracketed both directions; four structural probes below baseline; schedule closed in every dimension; init closed both directions; capacity closed both directions; smoothing closed; shortcut topology closed; throughput closed at its last tier with the new law — epochs convert only under hyperparameter constancy AND numerics equivalence (EXP-021: +10 epochs, −0.20pp).
- **The closest miss on record**: EXP-012 — batch 1024 + linearly-scaled peak 0.8 → 96.66 (−0.05pp), with the throughput fully delivered (+12 epochs) and the deficit diagnosed: a bouncy ~18pp mid-schedule trajectory deficit from the too-hot linear rule that the tail almost (but not quite) repaid. Goal-learnings left the axis ajar: "remaining batch variants (sqrt-scaled LR, 768) have ≤ ~0.2pp headroom."
- **Untried gaps**: (a) batch 1024 with √-scaled peak (0.4×√2 ≈ 0.566) — the canonical fix for EXP-012's diagnosed defect; (b) heat-constant momentum trade (0.95 + peak 0.2) — last untouched constant; (c) progressive resizing (low-res early phase) — radical, but its budget-race evidence is ImageNet-only (cifar10-fast/airbench do NOT use it at 32×32; downsampling 32→24 destroys proportionally far more content than 224→128).

**Synthesis check (per exp-report-021 § Next Steps #2)**: sixteen misses; every axis closed; the honest posterior is that 96.71 is at-or-near the family optimum and all remaining candidates are low-prior. The loop continues per standing directive, so EV ranking among survivors is what matters: (a) is the only candidate built on an in-project near-miss (−0.05, the closest ever) with a measured throughput gain in the validated execution regime AND a literature-canonical fix for its specific diagnosed failure; (b) has no comparable-regime evidence at all; (c) has evidence from the wrong resolution regime and multiple confounds (BN-stat transients, two compile graphs, augmentation-distribution shift). Ranking: (a) > (b) > (c).

## Candidate Ideas

### 1. Batch 1024 with √-scaled peak LR (PEAK_LR 0.566) — fix EXP-012's diagnosed defect
**Summary**: `BATCH_SIZE 512→1024` and `PEAK_LR 0.4→0.566` (= 0.4×√2, √-scaling), everything else byte-identical — default compile mode, foreach SGD, same schedule shape/warmup/augmentation. EXP-012 measured the throughput half cleanly: ~+8% img/s, 151 vs 139 epochs, with the deficit isolated to the linearly-scaled peak 0.8 trajectory (bouncy hot phase, ~18pp mid-run deficit, converged −0.05). √-scaling is the canonical correction: it preserves the gradient-noise scale that linear scaling overshoots on small datasets.

**Reasoning**: This is the only remaining candidate where BOTH halves of the mechanism have direct support: throughput measured in-project at the exact configuration (EXP-012, same default-compile numerics — passing the EXP-021 equivalence requirement), and the trajectory fix targeting the exact diagnosed failure with the standard literature rule. The arithmetic: if √-scaling holds the trajectory within noise of baseline-family, +12 epochs ≈ +0.24pp by the (numerics-equivalent) EXP-006 conversion — clearing the +0.1 bar with margin. The closest miss (−0.05) needs only +0.15 of recovery. Re-walks a once-probed axis with a DIFFERENT approach (allowed; goal-learnings explicitly left √-scaling as the remaining variant).

**Sources**: exp-report-012 via TSV row 012 + goal-learnings § Failed Approaches (batch entry); arXiv 1705.08741 (√-scaling), 1706.02677 (linear-scaling baseline); project-insights (EXP-021 numerics equivalence — this candidate stays in the validated regime).

**Estimated Effort**: low — two constants; standard launcher; signatures predictable from EXP-012 (dt ~41–42ms at 1024, ~150 epochs, VRAM ~2.5–3GB).

**Risk Assessment**: Heat ambiguity is the main risk: per-example heat at 0.566/1024 is 0.71× baseline's 0.4/512 — if 1024's halved gradient noise does not compensate, the run lands mid-cold (EXP-015 showed cooler trains worse at 512). Two-constant change weakens single-variable attribution (mitigated: EXP-012 already isolates the batch=1024+linear point; this adds the √ point on the same axis). Failure graceful: converged no-improvement, closing the batch axis permanently.

### 2. Heat-constant momentum trade: MOMENTUM 0.95 + PEAK_LR 0.2
**Summary**: Hold lr/(1−β) = 4 while doubling the averaging horizon; the only never-touched constant, admissible only as a compensated trade.

**Reasoning**: Completes the constant-bracketing program; smoother updates could lengthen the converged plateau the max-statistic harvests.

**Sources**: goal-learnings § Failed Approaches Medium (heat closure); exp-report-021 § Next Steps #1.

**Estimated Effort**: low.

**Risk Assessment**: No comparable-regime evidence; first-order equivalence only — likely re-measures the heat optimum with extra variance. Graceful failure. Prior: lowest.

### 3. Progressive resizing: 24×24 early phase via GPU interpolate, 32×32 late
**Summary**: Downsample batches to 24×24 on GPU for the first ~half of the budget (dt ≈ 0.56×, ~+28% total steps), switch to 32×32 for the remainder; CPU augmentation pipeline unchanged; time-keyed schedule self-adapts.

**Reasoning**: Front-loads steps into the early schedule (aligned with the deferral law, uniquely among remaining ideas); fast.ai/DAWNBench evidence — but ImageNet-only; at CIFAR's native 32×32, 24×24 destroys proportionally far more information, and neither cifar10-fast nor airbench uses it.

**Sources**: knowledge/README (cifar10-fast, airbench — notable ABSENCE of the technique at this resolution); fast.ai DAWNBench reports (ImageNet).

**Estimated Effort**: medium — phase switch, two compile graphs (double cold-compile startup), BN-stat transient at the switch.

**Risk Assessment**: Wrong-resolution-regime evidence; multiple confounds (BN transient, augmentation-statistics shift at low res, recompile); plausible crash/cap-bust modes. Worst risk profile of the three.

## Idea Evaluation

**Evidence strength**: Idea 1 is the only candidate standing on an in-project measurement (EXP-012's clean +12-epoch throughput at this exact batch in the validated numerics regime) plus a canonical literature rule targeting that run's specific diagnosed failure. Idea 2 has no evidence. Idea 3's evidence is from the wrong resolution regime — the exact transfer failure mode (wrong-regime external evidence) that has now burned four experiments.

**Mechanism clarity**: Idea 1: √-scaling preserves gradient-noise scale → trajectory tracks baseline → +12 epochs convert at ~+0.02pp/epoch → clears the bar. Quantified end-to-end, with the EXP-012 linear point already bracketing the hot side of the 1024 LR axis. Idea 2: vague. Idea 3: clear direction but confounded.

**Expected impact**: Idea 1: +0.15–0.25pp if the trajectory holds — the only candidate whose success scenario clears the bar with margin. Ideas 2–3: likely within noise.

**Risk profile**: Ideas 1–2 graceful; Idea 3 has crash/cap modes. Idea 1's cold-side risk is real but informative either way: it adds the third point on the 1024-LR curve (0.566 between the untried-cold and the measured-hot 0.8), closing the axis definitively if it misses.

**Feasibility**: Ideas 1–2 are two-constant diffs; Idea 3 is a medium build.

## Chosen Idea
**Selected**: Batch 1024 with √-scaled peak LR (PEAK_LR 0.566)

**Why this idea**:
It revisits the campaign's closest miss (−0.05pp, EXP-012) with the canonical fix for that run's specific diagnosed defect, keeps every component inside the validated execution regime (default compile numerics — the EXP-021 lesson), and is the only remaining candidate whose success arithmetic clears the +0.1 bar with margin. Sixteen misses in, the best EV available is a measured near-miss plus a targeted one-variable correction.

**Hypothesis**:
√-scaled peak 0.566 at batch 1024 removes the hot bouncy phase that linear 0.8 caused: mid-run trajectory within ~1pp of the baseline family (vs EXP-012's ~18pp deficit), ~150 epochs (EXP-012 measured 151), signatures on the EXP-012 family (dt ~41ms, VRAM ~2.6GB), and the +12 epochs convert under unchanged numerics — **best_test_acc ≥ 96.81**. A converged miss closes the batch axis permanently (hot/middle/linear points all measured) and routes the campaign to the momentum trade.
