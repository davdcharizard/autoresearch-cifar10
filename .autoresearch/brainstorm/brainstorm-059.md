# Brainstorm EXP-059
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external search. The one candidate class left on the books is anchored by a known paper: **"Don't decay the learning rate, increase the batch size"** (Smith, Kindermans, Ying, Le — ICLR 2018, arXiv:1711.00489): replacing tail LR decay with batch growth keeps the gradient-noise scale annealing while taking larger, more parallel steps. Their result is fixed-EPOCH; our fixed-TIME budget changes the accounting (larger batch also buys per-image throughput on this GPU: 1024 runs ~41ms vs 512 at ~22.5ms — measured EXP-012/022).
- Standing internal sources: brainstorm-056 Idea 2 (the original frontier entry), goal-learnings noise closures (EXP-011/012/022/023/024), tail-pressure law (EXP-025/033/055), dual-shape compile-warmup pattern (EXP-055, infra-errors).

## Experimental History Review

- Current best: 96.71 @ 1990397; bar 96.81; σ ≈ 0.16, mean ≈ 96.57 (EXP-027). **53 experiments, 52 consecutive non-improvements.**
- EXP-057/058 closed the final per-layer constant with a two-sided bracket (fc WD: 0 → −1.3σ, 5e-4 → mean, 2e-3 → −2.1σ — interior optimum at the default). Every recipe constant ever dosed now sits at a measured local optimum; every structural class, throughput flavor, loss path, averaging scheme, and per-layer treatment is measured-closed.
- **The single remaining documented corner**: the noise SCHEDULE. The noise law closed constant LEVELS both directions (batch 1024 under two LR rules, EXP-012/022; horizon/noise trades, EXP-023/024), but a LATE-ONLY noise reduction — annealing noise by batch in the tail while LR continues its own anneal — was never run. exp-report-058 Next Steps explicitly promotes it to "arguably the default next run."
- Priors against: three adjacent negative closures (constant-1024 loses ~0.05–0.15 at both canonical LRs; EXP-055 showed a delivered tail-step surplus cannot pay for a tail-dynamics change). Priors for: under fixed TIME, batch 1024 is ~9% cheaper per image (41ms vs 2×22.5ms), so the tail conversion adds genuine extra data passes — the only lever class where throughput and noise annealing point the same way; and the gradient-noise optimum was measured at CONSTANT noise, which does not bound a schedule that keeps the early phase at the optimum and only descends late.
- Infra notes that shape any plan: dual-shape compile warmup required (EXP-055 silent-no-op lesson: pre-warm BOTH batch shapes uncharged); GPU probe mandatory (graph/shape change — EXP-056 protocol); the standing watchdog thresholds assume ~22.5ms windows and would false-kill 41ms tail windows → launcher needs phase-aware bands; step-ledger integrity bands must be recomputed for the two-phase step accounting (EXP-058 lesson: the step ledger is the binding contention gate).

## Candidate Ideas

### 1. Late batch-size step: 512 → 1024 at p ≥ 0.75, LR schedule unchanged
**Summary**: Train at batch 512 until elapsed-budget fraction 0.75, then feed 1024-sample steps by pairing two consecutive loader batches (concat in the fetch path, BEFORE the charged-region t0, so pairing cost is uncharged), LR schedule untouched. Pre-warm both shapes in the uncharged compile warmup; probe both shapes before launch; launcher revised with phase-aware dt bands (~22.5ms pre-switch, ~41ms post-switch).

**Reasoning**: The last un-bracketed degree of freedom. Mechanism has two additive parts: (a) tail noise-scale reduction stacked on the cosine's own anneal — the Smith et al. mechanism; (b) a fixed-time-only throughput dividend: at 41ms/1024 imgs vs 45ms/1024 imgs (2×22.5), the tail converts ~9% more images per charged second (~+1.5–2 equivalent epochs of tail data). Law check: heat unchanged (lr(t) identical); tail pressure MAINTAINED (all params keep training, data keeps flowing — unlike EXP-055's freeze); numerics unchanged; aug pipeline unchanged (absorption-safe: noise schedule is not view-suppliable); dt charged per step rises but per-image falls — deferral law does not directly price it. Honest expectation: ≤ 0 by the three adjacent closures (noise level closed both ways; the cosine already anneals noise via lr→0; EXP-055's tail-conversion null) — but every branch closes the schedule class terminally, and after this run the documented frontier is EMPTY, which is itself the highest remaining closure value.

**Sources**: Smith et al. 2018 (arXiv:1711.00489); brainstorm-056 Idea 2; goal-learnings EXP-012/022/023/024 (noise level), EXP-055 (tail conversion), EXP-058 (step-ledger gate); infra-errors (dual-warmup, probe protocol).

**Estimated Effort**: medium — paired-batch fetch wrapper, dual-shape warmup, two-shape GPU probe, launcher band revision, recomputed integrity ledger.

**Risk Assessment**: Graceful failure modes, all pre-registerable: probe shows 1024-shape dt ≫ 41ms (cost-closure before launch); recompile-at-switch despite dual warmup (visible as a one-off multi-second step — caught by watchdog telemetry; the EXP-055 detach-flag lesson says verify by dt signature); family-band or sign-down read (schedule class closed). Largest operational risk is the launcher revision itself — new bands must be derived from the probe, not guessed.

### 2. Late batch-size step DOWN: 512 → 256 at p ≥ 0.75 (noise-UP tail — the mirror)
**Summary**: The opposite schedule: halve the batch at the tail, raising gradient noise late.

**Reasoning (and why not the lead)**: Completes the schedule axis from the other side, but every prior points one way: tail noise-UP fights the anneal (the cosine exists to reduce tail noise-to-signal); constant-256 was never the optimum (batch axis closed AT 512 with 1024 worse — EXP-012/022 — and smaller batches lose throughput outright: 256 runs ~12–13ms ≈ 8% slower per image); and the max-statistic law says a noisier tail mostly adds plateau scatter, which the MEAN protocol explicitly refuses to harvest. Run only if Idea 1 reads POSITIVE (then the schedule axis becomes interesting enough to bracket).

**Sources**: same noise closures; EXP-027 (scatter vs mean protocol).

**Estimated Effort**: medium (same machinery).

**Risk Assessment**: graceful but dominated — strictly weaker prior than Idea 1 on every component.

### 3. Multi-step batch ramp: 512 → 768 → 1024 (p ≥ 0.6 / p ≥ 0.8)
**Summary**: A smoother schedule shape — two switches instead of one.

**Reasoning (and why not the lead)**: Schedule-shape refinement of Idea 1. Costs a third compiled shape (768 is off the {512, 1024} power-of-2 grid — and the kernel-lattice law (EXP-044/045) warns that off-power-of-2 SHAPES can land on slow kernel tiers, so 768 likely misprices), a third probe point, and finer launcher bands — all to refine a mechanism whose single-step version is unmeasured. If the single step reads null, the shape refinement inherits the null (SE-dose logic, EXP-037); if it reads positive, shape tuning becomes a follow-up.

**Sources**: kernel-lattice law (EXP-044/045); EXP-037 dose-closure logic.

**Estimated Effort**: medium-high.

**Risk Assessment**: graceful but premature — strictly dominated by Idea 1 as the first measurement of the class.

## Idea Evaluation

All three candidates are members of the one remaining unmeasured class, so the evaluation is about which measurement to take first. Idea 1 is the canonical class representative: it matches the published mechanism exactly (Smith et al.'s replace-decay-with-growth, restricted to the tail), is the only variant where BOTH mechanism components point favorably (noise-down with the anneal + a fixed-time throughput dividend unique to this budget structure), and uses only on-lattice shapes (512, 1024 — both measured fast). Idea 2 fights the anneal and the throughput floor simultaneously; Idea 3 adds an off-lattice shape and a third graph for a refinement that is meaningless before the class has its first datum. Evidence strength: all expected ≤ 0, but Idea 1's negative priors are about noise LEVELS, not schedules, leaving it the only candidate with a genuinely unmeasured mechanism component. Risk: Idea 1's failure modes are all pre-registerable and terminal; its operational surface (dual-shape warmup, probe, launcher bands) is the known EXP-055/056 machinery. Closure value: a null on Idea 1 closes the entire schedule class (Ideas 2 and 3 inherit by mechanism and dose logic) and empties the documented frontier — maximal. Idea 1 dominates.

## Chosen Idea
**Selected**: Idea 1 — Late batch-size step 512 → 1024 at p ≥ 0.75, LR unchanged

**Why this idea**:
It is the last documented unmeasured class on the frontier, the canonical representative of that class (published mechanism, on-lattice shapes, favorable two-component accounting unique to the fixed-time budget), and a null terminally closes the whole class including its mirror and shape variants — after which the experimental record supports the measured-ceiling conclusion with no remaining documented gaps. The machinery it needs (dual-shape warmup, two-shape probe, phase-aware watchdog) is all validated from EXP-055/056.

**Hypothesis**:
If a late-only gradient-noise reduction adds value beyond the cosine's own anneal — helped by ~9% more tail images from the 1024-shape throughput dividend — the converged plateau level rises and best_test_acc reads ≥ 96.81 (TRUE effect ≥ +0.3), to be confirmed by replicate-pair MEAN. Expected under the adjacent closures: a family-band read in [96.41, 96.73] at two-phase clean signatures → the noise axis closes in BOTH level and schedule, and the documented frontier is empty. A read < 96.41 → the late noise drop actively harms (tail noise at lr→0 is load-bearing regularization), closing the class from below. Probe branch: if the 1024-shape compiled dt prices above ~46ms (no per-image dividend), the cost side of the mechanism is dead and the launch decision re-evaluates against a noise-only expectation. All branches terminal.
