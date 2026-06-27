# Brainstorm EXP-016
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/{slug}.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Optimal Linear Decay Learning Rate Schedules and Further Refinements (Defazio et al., 2023)** (https://arxiv.org/abs/2310.07831)
  The most comprehensive schedule evaluation to date (10 diverse problems incl. vision, LLMs, logistic regression): warmup + LINEAR decay to zero is the best non-adaptive schedule overall, outperforming cosine; the linear shape (lr ∝ 1 − t/T) arises directly from theory, not heuristics. Their refined schedules also show "rapid annealing near the end" — cosine's flat tail is theoretically suboptimal.
- **Straight to Zero: Why Linearly Decaying the LR to Zero Works Best (2025)** (https://arxiv.org/abs/2502.15938)
  Independent confirmation in budgeted LLM training: linear-to-zero beats cosine at matched budgets.
- **cifar10-fast (davidcpage)** (https://github.com/davidcpage/cifar10-fast — knowledge/README.md References)
  The canonical short-budget CIFAR speedrun used a PIECEWISE-LINEAR schedule (ramp + linear decay), not cosine — in-domain precedent at almost exactly our regime (batch 512, one-cycle, mixed precision).
- **Transfer check (papers/why-warmup-lr.md caveat)**: EXP-014 taught that fixed-iteration LENGTH intuitions invert under time-keying. This candidate is immune: the swap is a SHAPE change in progress-space at byte-identical throughput (progress ≡ t/T when dt is flat), and the LR-time integral is IDENTICAL — ∫(1−q)dq = ∫0.5(1+cos πq)dq = 0.5 over the anneal — so total heat is unchanged by construction.

## Experimental History Review

- Current best: **96.71** (EXP-006 recipe @ 1990397). Ten consecutive no-improvements (EXP-007…015).
- **New High pattern (EXP-015)**: the recipe is a CERTIFIED LOCAL OPTIMUM of single-constant changes — pressure ±, heat ±, capacity ±, batch, smoothing all measured worse. Single-knob tuning is exhausted; remaining space is MULTI-constant trades at held quantities or STRUCTURAL changes.
- Heat axis (count-2 Medium failed approach + EXP-015 cold point): total integrated LR heat is at optimum from both sides. Any candidate that changes ∫lr·dt is re-measuring a closed axis. **A schedule-FAMILY swap at constant integral is the one schedule move that does not touch it.**
- Max-statistic (EXP-011, project-insights Medium): best_test_acc rewards variance at convergence; smoothing loses, and (relevant here) linear decay holds MORE LR than cosine in the late-middle (q > 0.5: e.g. 0.25·peak vs 0.146·peak at q = 0.75) — more late gradient noise, variance-preserving rather than variance-collapsing.
- Protocol: composite launcher + inline watchdog (EXP-014), post-hoc windowed profile, `git clean -e data/` (EXP-015), no added per-image CPU work (EXP-013).

## Candidate Ideas

### 1. LINEAR-TO-ZERO ANNEAL: swap the cosine post-warmup branch for lr = PEAK_LR × (1 − q)
**Summary**: One-line change in `lr_at()`: after the unchanged 15% linear warmup, replace `PEAK_LR * 0.5 * (1 + cos(πq))` with `PEAK_LR * (1 - q)`. Same peak (0.4), same warmup, same time-keying, same endpoint (0). Total LR-time integral is mathematically identical (both anneal shapes integrate to 0.5·peak over the anneal); only the DISTRIBUTION changes: cosine over-weights the just-after-peak region (q<0.5) and starves the late-middle with a flat near-zero tail; linear spreads heat evenly and reaches zero with non-vanishing slope.

**Reasoning**: This is the strongest-evidenced structural move available. (a) External: Defazio et al.'s 10-problem evaluation crowns warmup+linear-decay over cosine, with theory; "Straight to Zero" replicates at matched budgets; cifar10-fast — the in-domain speedrun precedent — used piecewise-linear at our exact regime. (b) Internal: it is the unique schedule change that holds total heat constant (the closed axis) while changing shape (never probed — every prior schedule experiment changed the integral). (c) Max-statistic: higher late-middle LR preserves eval variance near convergence — the direction the metric rewards (EXP-011 inverted lesson) achieved through a legitimate, literature-backed schedule choice, not variance fishing: the mechanism claim is a better MEAN tail per theory, with variance preservation as a bonus.

**Sources**: arXiv 2310.07831, 2502.15938; knowledge/README References (cifar10-fast); goal-learnings § Failed Approaches Medium (heat — untouched by construction), § Patterns High (local-optimum certification → structural moves); project-insights § Medium (max-statistic).

**Estimated Effort**: low — one-line diff in `lr_at()`, ~510s runtime, baseline signatures expected (139 epochs, dt ~22.3ms, 1613MB).

**Risk Assessment**: Graceful failure (no-improvement). Main risk: cosine's early flatness (holding near-peak LR right after warmup) may be load-bearing on THIS recipe — linear starts descending immediately, so the just-after-peak region is ~15% cooler even though the total is equal; if early-mid progress is disproportionately valuable, the run lands slightly behind. Counter: heavier late heat repays in the second half at equal total. Worst case ≈ −0.2pp converged; no crash/cap exposure.

### 2. COMPENSATED SCHEDULE RESHAPE: WARMUP_FRAC 0.08 + PEAK_LR 0.35
**Summary**: Two-constant trade from exp-report-014/015 § Unexplored Avenues: shorter warmup with a lowered peak chosen to hold integrated heat roughly at baseline, isolating the anneal-length component that EXP-014's heat confound masked.

**Reasoning**: The cleanest follow-up WITHIN the cosine family. But the compensation is eyeballed (0.35 chosen by rough integral matching), so a null is ambiguous between "anneal length is worthless" and "compensation was off" — and EXP-014's result already suggests the anneal-length effect, if any, is smaller than the ±0.1pp noise floor.

**Sources**: reports/exp-report-014.md and -015.md § Unexplored Avenues / Next Steps; goal-learnings § Failed Approaches Medium.

**Estimated Effort**: low (two-constant diff).

**Risk Assessment**: Graceful; weakest inference per run (two variables, approximate invariant).

### 3. FLOPS REDISTRIBUTION AT CONSTANT SCALE: move one block from stage 1 to stage 3 (NUM_BLOCKS per-stage [2,3,4])
**Summary**: Structural architecture probe outside the closed uniform-scaling axis: keep ~params and alignment, redistribute depth toward the low-resolution stage (stage 3 blocks cost 1/4 the FLOPs of stage 1 blocks per layer at 4x width — wait, stage 3 has 4x channels at 1/16 spatial: FLOPs per block are equal across stages in ResNet by design; params grow 4x per stage). Moving a block 1→3 raises params (~+1.2M) at ~equal FLOPs/dt — capacity without throughput cost.

**Reasoning**: goal-learnings closed capacity for UNIFORM scaling only; per-stage reshaping at constant time is unprobed. ResNet stages have equal per-block FLOPs but 4x params per stage going deeper — a 1→3 move adds parameters "for free" in time. However: EXP-008 (depth-for-width at 6x) showed depth changes lose by convergence quality, and per-stage rebalancing folklore (more blocks late) is weaker-evidenced than the schedule literature.

**Sources**: goal-learnings § Failed Approaches High (uniform-width) and Low (EXP-008); project-insights § High (alignment — preserved: widths unchanged).

**Estimated Effort**: medium — requires reworking `_make_layer` calls to per-stage block counts; throughput must be re-measured (dt-gate mandatory).

**Risk Assessment**: Graceful but throughput-uncertain (compile may not keep dt flat across the depth rebalance); the params-up direction also fights the EXP-008 "ResNet-20 4x is the topology optimum" reading.

## Idea Evaluation

- **Evidence strength**: Idea 1 is overwhelming by this project's standards: a 10-problem benchmark study with theory (the only schedule-shape result of that rigor anywhere in our knowledge base), an independent replication, AND the in-domain speedrun precedent. Ideas 2/3 rest on internal extrapolations with known weaknesses (eyeballed invariant; EXP-008's depth pessimism).
- **Mechanism clarity**: Idea 1's mechanism is exact — same total heat, different distribution; the theory says cosine's flat tail under-anneals and its early plateau over-invests. It is also the only candidate whose interaction with EVERY closed axis is provably zero by construction (heat integral identical, throughput identical, pressure untouched). Idea 3's mechanism (free params) is clean but its throughput claim needs measurement; idea 2's mechanism is confounded by its own approximation.
- **Expected impact**: Defazio report consistent but modest gains for linear over cosine (fractions of a point on vision tasks) — exactly the +0.1pp-or-better scale this goal needs. Ideas 2/3 have no external magnitude prior at all.
- **Risk profile**: Ideas 1/2 are signature-identical scalar/shape changes (no cap/loader exposure); idea 3 carries throughput risk and a medium-effort diff.
- **Sequencing**: If idea 1 wins, the new schedule becomes the base for a possible follow-up (Defazio's "refined" schedules anneal even faster at the end). If it loses, the cosine-vs-linear question closes cleanly (same-integral comparison) and idea 3 becomes the lead structural probe next loop.

Idea 1 dominates on every criterion.

## Chosen Idea
**Selected**: LINEAR-TO-ZERO ANNEAL: swap the cosine post-warmup branch for lr = PEAK_LR × (1 − q)

**Why this idea**:
It is the best-evidenced untried change left: a theory-backed, 10-problem-validated, in-domain-precedented schedule-family swap that — uniquely among all schedule moves — holds the certified-optimal total heat EXACTLY constant while redistributing it away from cosine's theoretically-suboptimal shape (over-invested early plateau, starved late-middle, flat tail). It is a one-line diff with baseline-identical signatures, graceful failure, and it probes the only schedule dimension (family/shape) never touched in 16 experiments.

**Hypothesis**:
At identical total LR-time integral, warmup, and peak, the linear-to-zero anneal redistributes heat to the late-middle where the converged-tail refinement happens: the trajectory will run slightly behind baseline in the early-mid schedule (cooler just-after-peak region), cross over in the second half, and best_test_acc will reach ≥ 96.81 (baseline 96.71 + 0.1) at byte-identical throughput signatures (~139 epochs, dt ~22.3ms, ~1613MB, 4,286,026 params). Secondary prediction: late-epoch eval variance is at least baseline-level (linear holds 0.25·peak at q=0.75 vs cosine's 0.146·peak), so the max-statistic is not smoothed away.
