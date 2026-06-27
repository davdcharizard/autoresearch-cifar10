# Brainstorm EXP-040
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external searches; the deciding evidence is internal. Standing sources: WRN paper (knowledge README — width beats depth on CIFAR, gains persist to 8–10× at fixed epochs); EXP-001 (in-regime: 1×→4× width = **+2.07 at the LEVEL** under this exact budget/recipe — the largest single gain the program ever measured); EXP-007/005/002 (the three width FAILURES, all starvation: 52–55 epochs, below the ~70-epoch floor); EXP-034 (per-block launch cost ~2.5ms is width-independent "at these widths" — the 4× model's 22.4ms ≈ 9 × 2.5ms is essentially FULLY launch-bound, implying width has headroom before compute-bound onset); EXP-027 (σ ≈ 0.16, bar = mean + 1.5σ).
- Absorption-law note: WRN's fixed-epoch crop+flip numbers get the standard external discount — but unlike SE/SAM/LS, the width axis has its own IN-REGIME level evidence (EXP-001), so the open question is only WHERE the curve saturates, not whether the mechanism exists in this regime.

## Experimental History Review

- 41 experiments, 6 improvements (last EXP-006); baseline 96.71 @ 1990397 (mean ≈96.57, σ ≈0.16); bar 96.81; 34 consecutive misses.
- **EXP-039 closed the program's last in-regime measured slope** (BN momentum two-sided optimum). Per exp-report-039 Next Steps, this loop must source from the escalation path: re-read files for unexamined angles, recombine near-misses, or radical-but-law-compliant architecture.
- **The width curve is the one axis with a measured in-regime LEVEL gain and an UNMEASURED interior.** Datapoints at fixed budget: 1× → 95-ish, 4× → 96.57 mean @139ep (optimum so far), 6× → 95.86 @55ep. The 6× point is confounded: 55 epochs is below the ~70-epoch starvation floor, so it tests starvation, not converged width-level. "Width>4× AND converged" has never been measured. The High-Importance recurring failure entry itself writes the re-entry condition: "Do NOT retry unless measured compiled dt projects ≥70 epochs first" — a precondition that EXP-008/026's validated early dt gate can check in ~90 seconds.
- **The launch-bound discovery changes the dt forecast.** EXP-007's 2.59× slowdown at 6× (widths 96/192/384) is the compute-bound regime; EXP-034 measured per-block cost width-INDEPENDENT at widths ≤256, and the 4× model's 22.4ms is ≈ 9 blocks × 2.5ms — pure launch overhead. 5× (widths 80/160/320) sits between: if still mostly launch-bound, dt lands ~25–30ms → 100–120 epochs, comfortably above every measured convergence point (EXP-034 converged flat at 102ep; EXP-008 at 83ep); if compute-bound like 6×, the gate kills it at ~90s for trivial cost.
- **Epochs past convergence are worth ~0** (max-statistic law; EXP-031's +46 epochs converted at zero) — so the 4×@139ep operating point may be over-paying in epochs for under-capacity. The right comparison is converged-level vs converged-level.
- Screens still binding: absorption (out-regime imports), deferral, numerics equivalence, gradient-noise optimum, dt pricing. Note: stage-3-ONLY widening remains screened (prior brainstorm: asymmetry deficit + dt); this candidate is UNIFORM widening, governed by the gate.

## Candidate Ideas

### 1. Uniform 5× width (80/160/320) behind the validated early dt gate — measure the width curve's unmeasured interior at convergence
**Summary**: One-constant change: `WIDTH_MULT = 4 → 5` (params 4,286,026 → 6,693,850, +56%). Composite run with the EXP-008/026 early dt gate: measure compiled dt from the first watchdog windows; kill by tick ~5 if projected epochs < ~85 (window dt > 36ms). If dt lands ≤ ~30ms (launch-bound scenario), the run gets 100+ epochs — above every measured convergence point — and reads converged 5× level vs converged 4× level for the first time.

**Reasoning**: This is the only axis where the program has measured a large in-regime LEVEL gain (+2.07, EXP-001) and never measured where it saturates under convergence. All three prior width failures were starvation artifacts (≤55 epochs); the failure law itself specifies the gate as the lawful re-entry. The hardware story has also changed since those failures: the 4× model is now measured fully launch-bound (22.4ms ≈ 9×2.5ms width-independent block cost, EXP-034), so the marginal dt of 5× may be a few ms, not the 2.59× of compute-bound 6×. Honest effect range: if converged width-level still rises past 4× (WRN direction), +0.1–0.4 and a bar shot; if saturated-or-reversed (EXP-034's worse-basin pattern), −0.2–0.4 — a genuinely two-sided distribution, which after 34 expected-zero nulls is itself valuable. Risks priced: gradient-noise law (batch unchanged; per-param noise shifts with width but EXP-001 establishes width moves are exempt — the law binds noise-only changes); heat (time-keyed anneal completes at any dt); numerics (same kernels family, default compile); VRAM ~2.5GB (soft constraint, fine).

**Sources**: exp-report-001.md (in-regime width level); exp-report-007/034.md (dt regimes; per-block law); goal-learnings § Failed Approaches High (gate precondition verbatim); knowledge README WRN row.

**Estimated Effort**: low — one constant; composite launcher with the EXP-026-style dt-gate variant; abort costs ~90s if compute-bound.

**Risk Assessment**: Gate-protected: the only new failure mode vs baseline is a ~90s killed run if dt projects starvation. A full run that converges below baseline closes the width axis at convergence (upgrade of the recurring-failure entry from "starvation" to "level"); a gain is the first improvement in 34 experiments. No closed law is violated.

### 2. Derandomized alternating horizontal flip (2× virtual dataset)
**Summary**: Replace iid RandomHorizontalFlip with deterministic coverage: dataset of virtual length 100k (index < 50k unflipped, ≥ 50k flipped), shuffle over the union; an "epoch" becomes one pass over both orientations (≈ 2 baseline epochs, ~70 evals).

**Reasoning**: The only fully unmeasured data-ORDER candidate, zero-dt. But the mechanism (flip-sampling variance) is calibrated to ~10-epoch budgets; at 139 epochs each image already samples ~70 flips per orientation (residual imbalance ~4%), so the coverage guarantee adds ~nothing. Halving eval count also costs ~−0.05 of max-statistic harvest. Implementation must dodge the persistent-workers epoch-state gotcha (hence the virtual-length design). Expected ≈ 0 with mild negative skew.

**Sources**: airbench (knowledge README); EXP-027 max-statistic arithmetic; brainstorm-039 screening notes.

**Estimated Effort**: medium (Dataset wrapper + epoch-semantics changes ripple into eval cadence and log parsing).

**Risk Assessment**: Safe but expected-zero; changes epoch semantics, which complicates every downstream signature check.

### 3. Sigmoid/BCE objective in place of CE + label smoothing
**Summary**: Replace `F.cross_entropy(..., label_smoothing=0.1)` with multi-label BCE-with-logits (big-vision-style sigmoid loss), zero-dt.

**Reasoning**: "Objective shaping" is a named out-of-recipe area and BCE has published small gains on ImageNet classification. But the evidence is fixed-epoch, lighter-aug, different-scale — exactly the profile the absorption law has killed 4 consecutive times (SAM, LS, SE, and the anchor-transfer record 0-for-14); and EXP-036 measured the loss-target axis FLAT under TA+RE. No in-regime evidence exists or can be cheaply obtained.

**Sources**: exp-report-036.md (loss-target flat); project-insights absorption entry.

**Estimated Effort**: low.

**Risk Assessment**: Fails the in-regime screen outright; predicted-null by the absorption law — information-poor.

## Idea Evaluation

Evidence strength: Candidate 1 dominates — the width axis owns the program's largest measured in-regime level gain (+2.07) and its three failures are all confounded by starvation that the gate now lawfully removes; Candidates 2 and 3 are predicted ≈ 0 by coverage arithmetic and by the absorption law respectively. Mechanism clarity: capacity raises the converged plateau level (measured 1×→4×); the open empirical question — does it continue past 4× when convergence is guaranteed? — is exactly what the run answers. Expected impact: the only candidate with a plausible ≥ +0.3 branch (vs strictly ~0 for the others), and the launch-bound dt discovery (EXP-034) gives the favorable-dt scenario real probability rather than hope. Risk profile: best of the three — the gate converts the historical failure mode (wasted full run) into a ~90s abort, and even a converged loss permanently upgrades the width-closure from "starvation artifact" to "measured at convergence". Feasibility: a one-constant diff with an already-validated launcher pattern. Candidate 1 is the clear choice and squarely matches the directive's "more radical architectural changes" escalation.

## Chosen Idea
**Selected**: Uniform 5× width (80/160/320) behind the early dt gate (Candidate 1)

**Why this idea**:
It is the only remaining candidate whose mechanism has in-regime LEVEL evidence (EXP-001's +2.07) with an unmeasured interior, the only one with a genuine upside branch, and its historical failure mode (epoch starvation) is removed by a gate the failure law itself prescribes. The launch-bound block-cost measurement (EXP-034) makes the favorable dt scenario (≤30ms → 100+ epochs) physically plausible for the first time.

**Hypothesis**:
At 5× width the compiled step stays near launch-bound (dt ≤ ~30ms → ≥ ~100 epochs, above every measured convergence point), the time-keyed anneal completes, and the converged plateau level exceeds 4×'s (best_test_acc ≥ 96.81) because the width-level curve measured at +2.07/3-steps from 1×→4× has not yet saturated. Falsified by (a) GATE_KILL: dt > 36ms (projected < ~85 epochs) — width 5× is compute-bound and the axis stays closed on dt grounds; or (b) a full run converging at-or-below the baseline band — converged width-level is saturated at 4×, closing the width axis at its strongest form (level, not starvation). Diagnostics: dt/epoch count (regime determination), ep5/10/20 (transit), last-15 plateau mean/spread, final_test_loss (basin quality), VRAM (~2.5GB expected).
