# Brainstorm EXP-049
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external fetches; sources from the project record + model knowledge:

- **reports/exp-report-010.md § Unexplored Avenues** (the direct anchor for this loop): "PEAK_LR 0.3 (downward probe): the search is bracketed only from above; 0.4 could still be slightly above optimum" — written when 0.6 lost −0.57 and never executed in the 38 loops since. EXP-010 also establishes the protocol facts reused here: a pure LR-constant change leaves dt/params/epochs byte-identical (pure-LR effect isolation), and "warmup makes epoch-1 insensitive to the peak" (ep1 34.39 there — confirming EXP-048's ep1 scatter judgment).
- **One-cycle/super-convergence literature (model knowledge, Smith 2018)**: test-accuracy vs peak-LR curves on CIFAR ResNets are asymmetric — steep above the optimum (instability/forgetting), shallow below it (slower convergence partially repaid by the anneal). With the up-side measured steep here (−0.57 at ×1.5), the optimum plausibly sits at or slightly below 0.4; a ×0.75 down-probe samples the shallow side.
- **Post-EXP-048 state**: project-insights absorption entry (0-for-14 external transfers) is irrelevant here — this is an INTERNAL dose-response bracket of the recipe's own constant, the same class as EXP-010/014/015/036, with no external anchor needed.

## Experimental History Review

State after 49 indexed experiments: baseline 96.71 @ 1990397, bar ≥ 96.81; mean ≈ 96.57, σ ≈ 0.16; 42 consecutive non-improvements/invalids. Frontier after EXP-048:

- **Every mechanism class is measured-closed**: structural (absorbed-null 046 / active-negative 047, 030 / cost-priced 020, 037, 040–045), throughput (048: charged step 99.3% irreducible kernel math — EXP-000/006 mechanism exhausted; cudagraphs bounded out), regularization dose-response peaked both sides, noise optimum bracketed, weight/function averaging closed, init both directions, BN constants both directions, activations, attention, routing.
- **The ONE genuinely unmeasured bracket in the recipe's own constants**: integrated LR heat DOWNWARD. The map today: peak 0.6 → −0.57 (EXP-010); warmup 0.08 (hotter everywhere) → −0.22 (EXP-014); heat-constant noise trades both directions negative (EXP-023/024); heat-DISTRIBUTION swap (linear) → −0.50 (EXP-016). Every probe is above or sideways; below is open, and exp-report-010 explicitly queued it.
- **Honest detectability arithmetic (exp-report-048 directive to state plainly)**: bar = mean + 1.5σ; a true effect must be ≥ +0.3 for one-draw detection; cosine-peak optima are flat near the optimum, so the realistic best case for this probe is small-positive — its primary value is CLOSING the last open bracket so the recipe's constants are certified bracketed-both-directions, completing the audit honestly.
- Protocol carry-overs: dual launch gates, D0 gate (26ms), ≥200-step windows, replicate band 96.70–96.80, step-count ledger (new, EXP-048), trajectory-based numerics criterion (ep1 single reads unreliable — EXP-010/048).

## Candidate Ideas

### 1. PEAK_LR 0.4 → 0.3 — the heat-down bracket (single-constant probe)
**Summary**: One-line change: `PEAK_LR = 0.3` (comment updated). Warmup fraction, momentum, WD, batch, schedule shape, architecture all byte-identical. Time-keyed one-cycle means every progress point runs at 0.75× the baseline LR; integrated heat scales by exactly 0.75.

**Reasoning**: This is the only dose-response bracket left open on any recipe constant, queued by exp-report-010 itself and never executed. Mechanism (if positive): 0.4 was linearly scaled in EXP-000 for an UNAUGMENTED 1x net; the current net is 4x wide (lower gradient noise per parameter) under heavy aug — the optimum may have drifted below the scaled value; the literature's down-side flatness means a 0.75× probe risks little. Mechanism (if negative): less exploration at heat → lower plateau, mirroring EXP-023's reduced-noise loss. Free in every currency: dt/params/epochs/VRAM identical (EXP-010 demonstrated pure-LR isolation), no kernel change, no noise-source change beyond LR's intrinsic effect (which IS the variable under test). Either outcome closes the heat axis BOTH directions — the final certification of the recipe-constant audit.

**Sources**: reports/exp-report-010.md (§ Results, § Unexplored Avenues); goal-learnings § Failed Approaches (heat-raising count 2; momentum trades count 2); model knowledge (Smith 2018 one-cycle asymmetry).

**Estimated Effort**: trivial (one constant + standard gated composite; no new sanity surface).

**Risk Assessment**: Failure modes: (a) shallow-side loss −0.1..−0.3 (most likely by the asymmetry argument) → clean no-improvement, axis closed from below; (b) mean-band null → axis closed flat, 0.4 certified within a measured-flat region; (c) GATE/contention — standard screens. No destabilizing mode; signatures fully baseline.

### 2. WARMUP_FRAC 0.15 → 0.25 — heat-down via longer warmup (runner-up)
**Summary**: Reduce integrated heat by stretching the warmup instead of lowering the peak (cosine then anneals from 75% progress).

**Reasoning (and why not the lead)**: Same heat-down intent, but it confounds TWO variables — integrated heat AND the anneal's time support (the cosine tail compresses from 85% to 75% of budget, and EXP-016 proved tail SHAPE matters via the max-statistic's need for converged evals). EXP-014's mirror (0.08, hotter everywhere, −0.22) suggests warmup timing is the weaker, messier operator. Peak-LR is the clean single-variable form of the same question.

**Sources**: reports/exp-report-014.md; exp-report-016.md (cold-tail value); goal-learnings heat entries.

**Estimated Effort**: trivial.

**Risk Assessment**: Confounded attribution on any outcome — dominated by Idea 1.

### 3. LR-floor tail (anneal to ε > 0 so plateau weights keep random-walking) — documented rejection
**Summary**: End the cosine at ~2% of peak instead of ~0, widening plateau-eval scatter so the max over ~10 plateau draws rises.

**Reasoning (and why rejected)**: The gain mechanism is pure max-statistic scatter farming — mean strictly lower, best higher only through wider noise. The project's own MAX-STATISTIC law defines only converged plateau LEVEL as legitimate payment, and the eval-variance-harvesting precedent (brainstorm-041) classifies scatter-driven bests as reward hacking. Adversarial test fails: this approach would "improve" the metric while making the model strictly worse on average. Recorded so it is never re-derived.

**Sources**: goal-learnings max-statistic entries (EXP-011/032); brainstorm-041 rejection precedent.

**Estimated Effort**: trivial — and rejected regardless.

**Risk Assessment**: Integrity-failing by construction; would be classified invalid at analysis. Rejected.

## Idea Evaluation

- **Evidence strength**: Idea 1 is the program's own queued unexplored avenue with a measured one-sided bracket (0.6 → −0.57) and clean single-variable isolation demonstrated by EXP-010. Idea 2 asks the same question with two confounds. Idea 3 fails integrity outright.
- **Mechanism clarity**: Idea 1's is a textbook dose-response: sample the shallow side of an asymmetric optimum whose steep side is measured. Interpretation is unambiguous on every branch.
- **Expected impact**: honestly small (flat-near-optimum prior; needs ≥ +0.3 for one-draw detection) — but it is the LAST open bracket on any recipe constant; closing it completes the audit that 42 loops of negatives have been building, and the small-positive case is not excluded by any measured law (unlike every other remaining candidate).
- **Risk profile**: the safest possible probe — byte-identical signatures, graceful failure, zero new code surface.
- **Feasibility**: one line.

Idea 1 dominates. Ideas 2/3 recorded to pre-empt re-derivation.

## Chosen Idea
**Selected**: Idea 1 — PEAK_LR 0.4 → 0.3 (heat-down bracket)

**Why this idea**:
It is the only remaining dose-response bracket the project's own audit left open, explicitly queued by exp-report-010 and untouched for 38 loops; every other candidate class is excluded by a measured law. It is a one-line, signature-identical, single-variable probe whose every pre-registered branch produces a clean, final piece of the recipe certification — and the small-positive case (optimum drifted below the EXP-000-era linear scaling) is the last outcome no law forbids.

**Hypothesis**:
Lowering the one-cycle peak to 0.3 (0.75× integrated heat at identical noise sources, schedule shape, and signatures) samples the shallow side of the LR optimum: best_test_acc ≥ 96.81 if 0.4 sits above the optimum. Pre-registered branches: (i) best ≥ 96.81 → improvement (replicate pair first if 96.70–96.80); (ii) mean band (96.42–96.72) → heat axis closed flat-below; 0.4 certified within a measured-flat region; recipe-constant audit complete; (iii) < 96.42 → shallow-side loss measured; heat axis closed from below with 0.4 at the interior optimum; audit complete; (iv) GATE_KILL/contention → infra per standard screens (no legitimate path for this change to alter dt).
