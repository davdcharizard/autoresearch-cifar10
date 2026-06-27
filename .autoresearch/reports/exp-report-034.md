# Experiment Report EXP-034: Depth-for-width at matched compute — ResNet-26 at stage widths 56/112/224

- **Date**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-034.md
- **Plan**: plans/plan-034.md
- **Exp-log**: logs/exp-log-034.md
- **Verdict**: **no-improvement** (96.01 vs bar 96.81; baseline 96.71, Δ −0.70 ≈ 3.5σ below the run-level mean — decisively real)

## Goal
Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget, train.py only. Baseline 96.71 @ 1990397 (mean ≈96.57, σ ≈0.16); bar 96.81.

## Idea & Hypothesis
**Idea**: Probe the last unmeasured capacity direction — depth — with every confound held out: ResNet-26 (NUM_BLOCKS 4, 12 blocks) at stage widths 56/112/224 = 4× widths × √(3/4), giving conv FLOPs 1.021× and params +4.3% (4,469,538) vs baseline. Recipe byte-identical. Supporting datum: EXP-008 (shallower-but-wider at ≈matched compute) LOST, the only depth-direction sign on record; mechanism: composition depth buys decision-boundary expressivity (EXP-032 diagnosed the ceiling as boundary-limited) where width buys parallel features.

**Hypothesis**: At matched dt (≤24.5ms gate) and matched params, the deeper net converges to a HIGHER plateau → best ≥ 96.81. Falsified by dt gate-kill or a plateau within/below the baseline noise band.

## Approach
Four-site edit in train.py (6 insertions / 6 deletions): `NUM_BLOCKS = 4`; `WIDTH_MULT` → `STAGE_WIDTHS = (56, 112, 224)`; `ResNet.__init__` takes a widths tuple; construction/print updated. Param count hand-computed AND verified by CPU instantiation (4,469,538 exact; method validated by reproducing baseline's 4,286,026). Watchdog added an early-dt GATE (>24.5ms ×3 consecutive windowed ticks within first 10 → kill) on top of the standard contention/NaN/divergence/wall-cap kills.

## Execution
Three runs, all clean-load, escalating by design:
- **Run 1** (56/112/224): GATE_KILL at ~75s — windowed dt **30.7–31.5ms** (predicted 22.9–24). The +38% dt against +2.1% FLOPs refuted the FLOPs-scaling premise outright.
- **Run 2** (pre-registered fallback 48/96/192, FLOPs 0.75×): GATE_KILL at ~75s — dt **27.0–27.7ms**. Two-point decomposition: cutting 27% of FLOPs bought only 3.6ms ⇒ ∂dt/∂FLOPs ≈ 13.3ms/unit and the 3 extra blocks cost ≈ +8.3ms regardless of width — per-block cost (~2.3–2.8ms) is launch/memory-bound at these widths, nearly width-independent. A 12-block net would need ~0.53× baseline FLOPs (≈2.3M params) to fit 24.5ms.
- **Run 3** (56/112/224 full, dt gate lifted, contention rescaled to >37ms ×4 — autopilot adjustment recorded in exp-log): the matched-dt premise was dead, but the LEVEL hypothesis was still unmeasured and an honest verdict needs a metric. Clean completion: rc=0, 102 epochs (= exactly 139×22.4/30.5), 195 windows mean 30.5ms / 0 slow, total 418.3s, VRAM 1815.8MB.

## Results
**best_test_acc 96.01 (ep98); final 95.96; final_test_loss 0.1964. −0.70 vs baseline. Both halves of the depth question now have measured answers, and both are negative:**

1. **Hardware: depth is structurally expensive at this model scale.** Step time is per-block bound, not FLOPs-bound — each block costs ~2.5ms almost independent of width in the 48–64-per-stage-1 range. The "matched-compute depth trade" design point does not exist on this GPU: depth cannot be bought with width at constant dt without halving capacity. This single fact gate-screens ALL deeper variants (ResNet-32+) permanently.
2. **Statistics: even granting depth its dt cost, the LEVEL is lower, not higher.** The full run is the EXP-005 analogue for depth: a converged, flat, clean plateau (last 15 evals 95.9–96.0, test_loss settled ~0.196) sitting −0.56 below the baseline mean. Notably the trajectory was NOT merely "fewer epochs of the same curve": test_loss converged to 0.196 vs baseline's ~0.185 — a worse basin, like EXP-028's Muon plateau. At this width-heavy shape (4× wide, depth 20), moving capacity into depth degrades both the boundary metric and the loss.
3. **The EXP-008 sign does not invert symmetrically**: shallower-but-wider lost (EXP-008), and now deeper-but-narrower loses too — depth 20 at 4× width is a measured LOCAL OPTIMUM of the depth-width shape plane, matching the He-2015 ladder logic only at its original width (16/32/64), not at 4×. Wide nets saturate depth's marginal value (WRN's core claim, now confirmed in-project at fixed wall clock).

**Capacity axis status**: magnitude bracketed (2×/4×/5×/6×/8× width: EXP-001/002/005/007), allocation closed (EXP-017), depth closed both directions (EXP-008, EXP-034). The architecture-shape space reachable at this dt is now measured-closed in every probed dimension around the 4×-wide ResNet-20.

## Verification
- Condition 1 (best ≥ 96.81): **FAIL** — 96.01. Pre-condition profile PASS at rescaled thresholds (195 win, mean 30.5ms, 0 >37ms; epochs 102 exactly consistent with dt; params exact; evals = epochs). Single clean full run; two deliberate gate-kills documented, neither contamination.
- Trustworthiness: high — dt rock-steady (30.0/30.9 alternation, zero drift), epoch arithmetic exact, plateau flat and converged.
- Verdict basis: clean miss, real deficit, no constraint violated → **no-improvement**.

## Key Learning
Depth is doubly closed at this scale: (1) hardware — step time is per-block launch/memory-bound (~2.5ms/block, width-independent), so 12-block nets cost +8ms regardless of width compensation and matched-dt depth trades don't exist; (2) statistics — granting the dt cost, ResNet-26 @ 56/112/224 converges to a flat LOWER plateau (96.01, test_loss 0.196 vs 0.185 — a worse basin, not a transit deficit). Depth-20 at 4× width is a measured local optimum of the shape plane; width saturates depth's marginal value (WRN confirmed under fixed wall clock).

## Unexplored Avenues
- **ResNet-32 and deeper**: closed by the per-block cost law (each +3 blocks ≈ +8ms ≈ −35 epochs) compounding with the now-measured negative level effect. No design point survives the arithmetic.
- **Width-profile reshape at constant FLOPs (e.g., stage-1-heavy)**: cannot hold params and FLOPs simultaneously (stage-3 dominates params, stage-1 dominates per-channel spatial cost); EXP-017's allocation loss plus this experiment's local-optimum finding give it a low prior — the 64/128/256 doubling profile is the classic equal-FLOPs-per-stage design and it sits at the measured optimum.
- **Stem/head micro-changes**: head measured (EXP-030 concat-pool, no gain); CIFAR stem is already the 3×3 minimal form. Nothing of this class can plausibly clear +0.3.

## Next Steps
1. **Strategic reckoning for the next brainstorm**: 29 consecutive misses; every axis is now measured-closed — recipe constants (both directions), schedule (family/shape/heat), optimizer (internal + geometry), data (augmentation dose/schedule/resolution), eval-side (BN, weight averaging), capacity (magnitude/allocation/depth/head). The remaining honest moves are (a) replicate-based variance exploitation is BANNED (no seed hacking), (b) combination plays of near-misses (but composition of single-axis losses has no measured positive interaction to exploit), (c) genuinely novel mechanism classes not yet in the bracketed set. Confidence: high as an assessment.
2. **Candidate novel-mechanism probes worth screening**: (i) collapse the two eval-time costs the budget structure charges nothing for — e.g., overlap H2D with compute via a prefetch CUDA stream INSIDE the timed step to cut charged dt (~1–2ms H2D at batch 512 is charged today; a real throughput lever, EXP-021-class, must survive numerics-equivalence); (ii) per-sample gradient-noise shaping via batch-size schedule keyed to the anneal (noise axis was bracketed via momentum/batch but never TIME-VARYING). Confidence: low-medium (i), low (ii).
3. **If pursuing (i)**: gate by profile first 2 minutes; the conversion law (EXP-006/021) says saved dt converts only via plateau LENGTH — at ~6% dt saving expect ~+8 epochs ≈ within-noise alone; pair with nothing else changed for a clean read. Confidence: low on clearing the bar, medium on positive sign.

## Exit Action Results
(no exit actions defined for this goal)
