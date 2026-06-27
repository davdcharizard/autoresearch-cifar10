# Brainstorm EXP-058
**Created**: 2026-06-11
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external search this loop. The standard-modernization audit is complete (exp-report-056; external transfer 0-for-16), and this loop's candidates are dose-response continuations of the project's OWN fresh measurement (EXP-057), not imports. Knowledge base consulted via goal-learnings/project-insights; no entry needed re-reading.
- Model knowledge anchor: weight-decay-as-margin-control on BN-free output layers (the classifier-norm literature around LS/calibration, e.g., Müller et al. 2019) — used only to frame the mechanism; the operative evidence is EXP-057's in-vivo slope.

## Experimental History Review

- Current best: 96.71 @ 1990397; bar 96.81; σ ≈ 0.16, recipe mean ≈ 96.57 (EXP-027). **52 experiments, 51 consecutive non-improvements.**
- **The new datum that anchors this loop (EXP-057)**: removing WD from fc.weight read 96.36 = mean−1.3σ, below the family floor, at byte-identical signatures and FAMILY test_loss. The classifier's WD margin cap is load-bearing under CE+LS + heavy aug. This is the first measured SLOPE pointing in a specific direction in dozens of loops: along the fc-decay-pressure axis, accuracy RISES from λ_fc=0 to λ_fc=5e-4 (~+0.2pp over the interval). The 5e-4-and-above side of that axis is unmeasured — the global reg-dose closures dosed GLOBAL knobs (aug, LS-bearing recipe constants), never fc-specific pressure above default.
- Standing laws unchanged and binding: absorption (mechanism must not be aug-suppliable — fc decay pressure is classifier-side, not view-side), deferral (zero added dt required), two-sided tail pressure, heat/noise/loss/numerics/structure/order all closed, max-statistic (plateau LEVEL is the target).
- Adjacent closures that bound interpretation: EXP-050 (forced margin-UP in the loss loses −2.4σ) does NOT cover margin-cap-TIGHTER via the optimizer path; EXP-051 (suppressing hard views) is unrelated mechanism; logit-scale RELIEF is closed (EXP-057).
- Remaining documented corner: late batch-size schedule 512→1024 at p≥0.75 (brainstorm-056 Idea 2; three adjacent negative closures, medium infra effort) — still on the books as fallback.
- Protocol assets: composite launcher /tmp/exp046_composite.sh; no GPU probe needed for optimizer-only diffs (validated in-vivo by EXP-057: D0 22.7, family signatures throughout); replicate-pair MEAN for any read ≥ 96.81; reads in (96.73, 96.81) are no-improvement.

## Candidate Ideas

### 1. Increase classifier weight decay ×4 (fc.weight WD 5e-4 → 2e-3) — dose-response continuation along the measured slope
**Summary**: Keep the EXP-057 param-group split (fc.weight isolated) but set its weight_decay to 2e-3 instead of 0, leaving conv weights at 5e-4 and BN/bias at 0. Same ~6-line optimizer-only diff shape as EXP-057; graph, schedule, loader, loop byte-identical; no GPU probe needed (validated by EXP-057's clean in-vivo signatures for exactly this diff class).

**Reasoning**: EXP-057 measured the first directional slope on the frontier: fc decay pressure 0 → 5e-4 gains ~0.2pp (96.36 → family mean), meaning the margin cap on the one BN-free layer is genuinely load-bearing — under heavy TA+RE augmentation, a tighter logit-scale cap regularizes per-view overconfidence. The natural follow-up is the unmeasured side: does the gain continue above 5e-4? This is NOT a retry of any closure: EXP-050 forced margins WIDER through the loss (opposite direction, different path); the global regularization-dose closures never dosed fc-specific pressure; per-layer WD coverage measured fc at {0, 5e-4} only. ×4 is chosen to be detectable if real (a ×2 dose risks sub-resolution; equilibrium fc norm scales sub-linearly in λ, so ×4 is a moderate, not extreme, cap tightening). Law check: zero dt/heat/noise/numerics change, full tail pressure (fc keeps training; WD term scales with lr so it self-anneals like all decay here), classifier-side mechanism that augmentation cannot supply. All outcome branches close the axis: up → improvement protocol; flat → 5e-4 sits on a plateau at/past the optimum, axis closed with three measured points; down → optimum bracketed in (0, 2e-3) with 5e-4 the measured best, axis closed.

**Sources**: reports/exp-report-057.md § Results (the slope datum and the load-bearing-cap interpretation); goal-learnings § Failed Approaches Medium (loss-axis entry, count 3 — distinguishes loss-path margin-up from optimizer-path cap-tighter); plan-057 (diff shape, sanity assets reusable: /tmp/exp057_sanity.py needs only the WD-value assert changed).

**Estimated Effort**: trivial — same diff class as EXP-057 (~6 lines), sanity script reuse, no probe, standard composite launch.

**Risk Assessment**: Safest failure mode available: signatures byte-identical by construction, attribution airtight. Worst case is a sign-down read → brackets the optimum and terminally closes the last open axis with a measured interior maximum. Main assumption at risk: the 0→5e-4 slope may already be saturated AT 5e-4 (the default could sit on the plateau's edge); then the read is family-band and the axis closes flat.

### 2. fc-specific LR multiplier ×0.5 — tighten the cap via the adaptation path
**Summary**: Give fc.weight (and fc.bias) lr = 0.5 × lr_at(progress) while all other params keep the full schedule; WD structure unchanged at the recipe default.

**Reasoning (and why not the lead)**: A slower-moving classifier also limits logit growth — same direction as Idea 1 but through adaptation speed rather than norm penalty. Two weaknesses: (a) mechanism entanglement — fc LR↓ changes BOTH the equilibrium scale AND the classifier's tracking of the feature drift over the anneal (a tail-pressure-adjacent risk: the head partially lags the features it must classify, the EXP-055 lesson in miniature); (b) the WD path (Idea 1) tests the cap hypothesis cleanly because the SGD decay term lr·λ·w preserves the recipe's effective-LR trajectory shape, whereas an LR multiplier distorts it. If Idea 1 reads up, an LR-path confirmation becomes interesting; running it first would muddy attribution.

**Sources**: exp-report-057.md § Unexplored Avenues; brainstorm-056 Idea 3 (the original fc-LR framing and its margin-pressure-down caveat).

**Estimated Effort**: trivial-to-low (third param group + per-step lr assignment by tag — the EXP-055 tag pattern).

**Risk Assessment**: graceful, but a down-read would be ambiguous between "cap tightening past optimum" and "head lag" — strictly worse closure value than Idea 1 at equal cost.

### 3. Late batch-size schedule: 512 → 1024 at p ≥ 0.75, LR unchanged (carried from brainstorm-056/057)
**Summary**: Tail-only noise-scale reduction by doubling the per-step sample count at p=0.75 (concatenate two loader batches; dual-shape compile warmup), LR schedule untouched.

**Reasoning (and why not the lead)**: The lone un-bracketed noise DOF (schedule vs level), but it duplicates the cosine's own tail annealing, carries three adjacent negative closures (EXP-022 constant-1024 both LR rules, EXP-024 horizon trades, EXP-055 tail-conversion), and the largest infra surface on the menu (two compiled shapes, recompile risk, probe + band revision). Stays on the books strictly behind any candidate with a measured positive prior.

**Sources**: brainstorm-056 Idea 2 (full frontier entry); Smith et al. 2018 (arXiv:1711.00489); goal-learnings EXP-012/022/023/024, EXP-055.

**Estimated Effort**: medium.

**Risk Assessment**: graceful but expected ≤ 0 on three adjacent closures; dominated by Idea 1 on evidence, effort, and risk.

## Idea Evaluation

Evidence strength is, for the first time in many loops, not uniform across candidates: Idea 1 rests on a measured in-vivo slope from the immediately preceding experiment (fc decay pressure 0 → 5e-4 = ~+0.2pp at airtight attribution), while Ideas 2 and 3 rest on triangulated priors (Idea 2 shares Idea 1's direction but through an entangled mechanism; Idea 3 has three adjacent negative closures). Mechanism clarity likewise favors Idea 1: the cap-tightening hypothesis makes a specific, falsifiable prediction (plateau LEVEL rises if 5e-4 is below the cap optimum) through a pathway (optimizer-side norm penalty on the one BN-free layer) that no prior closure covers — EXP-050 closed loss-path margin-UP, EXP-057 closed optimizer-path relief, and the global dose closures never touched fc-specific pressure above default. Expected impact is honestly modest (the slope may saturate at the default), but every branch terminally closes the last axis with a measured directional datum — the highest closure-value-per-token available. Risk ordering: 1 < 2 < 3 (Idea 1 inherits EXP-057's zero-infra profile wholesale). Feasibility: Idea 1 reuses the EXP-057 diff shape, sanity script, and launcher verbatim. Idea 1 dominates on all five criteria.

## Chosen Idea
**Selected**: Idea 1 — Increase classifier weight decay ×4 (fc.weight WD 5e-4 → 2e-3)

**Why this idea**:
It is the only candidate on the frontier backed by a measured positive slope rather than a triangulated prior: EXP-057 established in vivo that the classifier's WD margin cap is load-bearing (+0.2pp from 0 to 5e-4 at byte-identical signatures), and the region above the default is the one unmeasured stretch of the last open axis. The run costs the same near-zero infra as EXP-057 (optimizer-only diff, no probe, family signatures guaranteed), and all three outcome branches — up, flat, down — terminally resolve the axis with an interior measurement.

**Hypothesis**:
If the fc margin-cap optimum lies above the default 5e-4, tightening the cap ×4 (fc WD = 2e-3) raises the converged plateau level and best_test_acc reads ≥ 96.81 (TRUE effect ≥ +0.3), to be confirmed by replicate-pair MEAN. If 5e-4 already sits at/past the optimum: a family-band read in [96.41, 96.73] closes the axis flat (three measured points: 0 ↓, 5e-4 ✓, 2e-3 →), or a read < 96.41 brackets the optimum inside (0, 2e-3) with 5e-4 the measured best — closed from above. All branches terminal at family signatures (dt 22.0–22.8ms, 138–140 epochs, params 4,286,026).
