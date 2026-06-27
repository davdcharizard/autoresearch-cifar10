# Brainstorm EXP-017
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Designing Network Design Spaces (RegNet)** (arXiv 2003.13678, CVPR 2020 — https://arxiv.org/abs/2003.13678)
  Population-level analysis of thousands of trained networks: the best design spaces allocate MOST blocks to the third stage with few in the first — "higher flop models have a large number of blocks in the third stage". Uniform per-stage depth (our [3,3,3]) is NOT what emerges when depth allocation is optimized; later-stage-heavy allocations dominate at matched compute. Directly supports redistributing depth toward stage 3 at constant FLOPs.
- **Wide Residual Networks** (knowledge/README.md → arXiv 1605.07146)
  Width-vs-depth on CIFAR is well mapped (width wins at fixed depth-budget), but WRN only studied UNIFORM stage depths — per-stage allocation is orthogonal to its result and unprobed in-project.
- **Existing knowledge base** (knowledge/README.md)
  why-warmup-lr.md and optimal-linear-decay.md document that the schedule axis is closed in every dimension (EXP-014/016); cifar10-airbench and cifar10-fast references confirm the throughput/pipeline axis is exhausted at this scale. Nothing in the knowledge base bears on per-stage allocation — this is a genuine gap.

## Experimental History Review

- **Current best**: 96.71 @ 1990397 (EXP-006 recipe: bf16 + channels_last + compile, batch 512 nesterov SGD, selective WD 5e-4, LS 0.1, TA+RE aug, time-keyed one-cycle 0.4/0.15 cosine). Ten single-constant probes (EXP-007…015) bracket it as a certified LOCAL OPTIMUM of single-knob changes (goal-learnings § Patterns High; project-insights § Medium).
- **Eleven consecutive no-improvements (007–016)**. Closed axes: uniform capacity scaling both directions (EXP-002/005/007/008 — starvation or convergence-deficit), regularization both sides (EXP-009/013/015), batch/throughput (EXP-012), smoothing (EXP-011 — max-statistic punishes variance reduction), LR heat both sides (EXP-010/014/015), warmup length (EXP-014), anneal family (EXP-016 — cosine's converged plateau is load-bearing for the max-statistic).
- **What is NOT closed**: per-stage allocation of depth or width at constant alignment. EXP-008 (the only depth experiment) changed TOTAL depth (20→14) uniformly; EXP-002/005/007 changed width UNIFORMLY. No experiment has ever moved capacity BETWEEN stages at constant FLOPs/dt. exp-report-016.md § Next Steps ranks this the top remaining direction.
- **Hard lessons that shape any structural attempt**: (1) channels must stay multiples of 32 (project-insights High — H20 tensor-core alignment); (2) projected epochs must be ≥~70, verified from measured compiled dt, never extrapolated (goal-learnings § Failed Approaches High + project-insights Medium); (3) early dt-gate at step ~100 is the validated cheap kill-switch (EXP-008, goal-learnings Protocol Medium).

## Candidate Ideas

### 1. Per-stage depth redistribution [3,3,3] → [2,3,4] at constant FLOPs
**Summary**: Keep depth 20, width 4x, and every training constant identical; change only the per-stage block counts from uniform (3,3,3) to (2,3,4) — one block moved from stage 1 (64ch, 32×32) to stage 3 (256ch, 8×8). By ResNet's halve-spatial/double-width construction, per-block FLOPs are EQUAL across stages (36,864 params × 1024 px = 589,824 params × 64 px ≈ 75.5M MACs/block), so total FLOPs are unchanged by construction, while params go 4.29M → ~5.39M (+1.11M, +26%): the stage-1 block carries 74k params, the stage-3 block 1.18M. Implementation is a ~6-line change: `NUM_BLOCKS = (2, 3, 4)`, pass per-stage counts to the three `_make_layer` calls, fix the depth print to `2 + 2*sum(NUM_BLOCKS)`.

**Reasoning**: This is "capacity where it is cheap". The capacity axis was closed only for UNIFORM scaling, which always paid in epochs; this move adds 26% params at zero FLOPs delta, so the ~139-epoch trajectory — which every closed-axis probe confirmed is the binding resource — is preserved. RegNet's population result (best allocations are third-stage-heavy, first-stage-light) is exactly this direction. Memory traffic even favors it: the removed stage-1 block had the largest activations (64×32×32), the added stage-3 block the smallest (256×8×8), so dt should be ≤ baseline if kernel efficiency holds at 256ch/8×8 (alignment fine: 256 = 8×32).

**Sources**: arXiv 2003.13678; reports/exp-report-016.md § Next Steps 1; goal-learnings § Failed Approaches High (uniform-width closure + the ≥70-epoch rule); project-insights § High (alignment), § Medium (measured-dt rule).

**Estimated Effort**: low (~6-line diff; single 300s run + dt gate).

**Risk Assessment**: Compile may price 256ch 8×8 convs differently than the FLOPs model predicts (project-insights Medium: never project across regimes) — mitigated by the validated step-100 dt-gate (kill if projected epochs < ~125, i.e. dt > ~25ms). Two blocks in stage 1 may under-serve early features; ResNet-18's [2,2,2,2] precedent says 2 blocks/stage is workable. Worst case: clean no-improvement, axis gets its first data point.

### 2. Width asymmetry 64/128/320 at constant alignment
**Summary**: Keep [3,3,3] and all constants; widen only stage 3 from 256 → 320 (= 10×32, aligned). Stage-3 conv params scale ×(320/256)² = 1.5625, adding ~2.0M params; stage-3 FLOPs rise the same factor, but stage 3 is only ~⅓ of total FLOPs, so total FLOPs rise ~+19% — measured compiled dt must confirm epochs stay ≥ ~120.

**Reasoning**: Same "capacity where it is cheap" logic as Idea 1 via widths, and stage-3 8×8 features are FLOPs-cheap per param. But it is NOT FLOPs-neutral, so it re-enters the closed capacity-vs-epochs trade at reduced dose; and EXP-007 measured inductor gains shrinking with width — the +19% FLOPs may cost more than +19% time.

**Sources**: reports/exp-report-016.md § Next Steps 2; project-insights § High (alignment), § Medium (compiled-dt rule); goal-learnings § Failed Approaches High.

**Estimated Effort**: low (2-line diff + dt gate).

**Risk Assessment**: Pays in epochs (~139 → ~125 at best) — the resource every prior failure shows is binding. If compiled dt scales super-linearly (as in EXP-007), epochs drop further and it reruns the uniform-widening failure at lower dose. Safest failure mode is a clean no-improvement, but the mechanism overlaps a 3x-failed axis.

### 3. Squeeze-and-Excitation blocks on stages 2–3
**Summary**: Add SE modules (global-avg-pool → FC(C→C/16) → ReLU → FC(→C) → sigmoid gate) to the residual branch of stage-2/3 blocks. ~+0.3M params, negligible FLOPs; SENet reports +0.5–1pp on CIFAR-scale nets.

**Reasoning**: Genuinely structural (adds a new computational mechanism, not a constant change) with strong literature support (arXiv 1709.01507). But: per-block global pooling inserts reduction ops that fragment the compiled graph — dt risk is real and unmodelable in advance; SE also alters optimization dynamics, and every dynamics-adjacent change (EXP-009/010/011/013/014/015/016) has lost on this recipe; gains in the literature are measured at fixed epochs, not fixed time.

**Sources**: arXiv 1709.01507; project-insights § Medium (regime-transfer warning); goal-learnings § Failed Approaches (dynamics changes consistently lose).

**Estimated Effort**: medium (~30-line diff, new module, compile interaction unknown).

**Risk Assessment**: Highest dt uncertainty of the three (pool/FC/sigmoid per block × 6–7 blocks); if compile fails to fuse, the throughput tax converts to fewer epochs and the run loses for infrastructure-shaped reasons that contaminate the structural signal.

## Idea Evaluation

**Evidence strength**: Idea 1 has the strongest convergent evidence: RegNet's population-level result (third-stage-heavy allocations dominate at matched compute) plus the in-project finding that epochs are the binding resource (so a params-up/FLOPs-flat move is the only capacity move that doesn't touch the binding constraint). Idea 2 shares the directional logic but is FLOPs-positive, partially re-entering a 3x-failed axis. Idea 3 has good external evidence but measured at fixed epochs — the exact transfer condition that has failed in EXP-010/014/016.

**Mechanism clarity**: Idea 1 is the cleanest: +26% params concentrated where per-param FLOPs cost is 16x lower, at provably-equal MACs, holding every training constant fixed — if it moves the metric, the attribution is unambiguous (pure allocation effect). Idea 2's mechanism is confounded by the epoch cost. Idea 3's mechanism (channel attention) is real but its time-budget interaction is opaque.

**Expected impact**: EXP-001 measured accuracy as steeply width/param-sensitive when epochs are preserved (+2.07pp for 4x). Idea 1 is the only candidate that adds params while preserving epochs — if param count (not FLOPs) is what stage-3 representation needs, this is the largest available lever. Ideas 2/3 buy less capacity per epoch lost.

**Risk profile**: Idea 1 risks only a clean no-improvement, with the dt-gate as a cheap abort; Idea 2 risks repeating a closed failure mode; Idea 3 risks an uninterpretable infra-confounded result.

**Feasibility**: Ideas 1–2 are trivial diffs; Idea 3 is a real module with compile-interaction unknowns.

Idea 1 dominates on all five criteria.

## Chosen Idea
**Selected**: Per-stage depth redistribution [3,3,3] → [2,3,4] at constant FLOPs

**Why this idea**:
It is the only untried direction that adds capacity WITHOUT spending the resource every prior failure identified as binding (epochs/FLOPs): per-block MACs are equal across stages by ResNet's construction, so [2,3,4] buys +1.11M params (+26%) at zero FLOPs delta and ~unchanged dt — uniquely sidestepping the starvation mechanism that closed uniform scaling 3x. RegNet's design-space analysis independently found that optimized depth allocations are exactly this shape (third-stage-heavy, first-stage-light). The diff is 6 lines, every training constant stays at its certified-optimal value, and the validated step-100 dt-gate bounds the downside.

**Hypothesis**:
With per-stage blocks (2,3,4), measured compiled dt stays within ~5% of the 22.4ms baseline (projected ≥ ~130 epochs), and the +1.11M params concentrated in stage 3 lift best_test_acc to ≥ 96.81% (baseline 96.71 + 0.1), with num_params ≈ 5.39M and the converged-plateau signature (final ≈ best, flat tail) intact. If instead dt rises > ~12% (projected < 125 epochs), the dt-gate kills the run early and the result is recorded as a throughput-priced allocation, not a capacity verdict.
