# Brainstorm EXP-039
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external searches. The in-regime-evidence screen (project-insights, EXP-035/036/037 absorption law) holds, and this loop's lead candidate is a direct dose-response follow-up to OUR OWN fresh measurement (EXP-038) — the strongest in-regime evidence class available.
- Standing sources: EXP-038 (lag dominates variance in BN running stats: m 0.1→0.02 cost −0.30 with 4× plateau scatter); EXP-029 (constants must come from the augmented training stream); EXP-027 (σ ≈ 0.16, bar = mean + 1.5σ); PyTorch BN semantics (`running = (1−m)·running + m·batch`).
- Statistical note grounding the candidate (framework arithmetic, not transfer): at batch 512, an m=0.25 EMA averages ~4 batches ≈ 2,048 samples per channel — the standard error of per-channel mean/var estimates at that sample size is tiny, so the VARIANCE cost of much fresher constants is near-zero; the LAG cost, by contrast, was just measured to be large and dominant in the other direction.

## Experimental History Review

- 40 experiments, 6 improvements (last EXP-006); baseline 96.71 @ 1990397 (mean ≈96.57, σ ≈0.16); bar 96.81; **33 consecutive misses**.
- **EXP-038 changed the local landscape**: unlike the three exact-deficit nulls before it (SAM/LS/SE — dial inert), BN momentum is a LIVE dial with a measured steep slope: −0.30 at 5×-longer horizon, mechanism cleanly attributed (constants lag the drifting weights at hot phase AND through the still-drifting "converged" tail — best-at-final-epoch pattern across EXP-036/037/38 confirms the tail drifts to the last eval).
- The dial is bracketed from ONE side only (below). Whether m=0.1 is the optimum or merely the first point on the rising freshness side is unmeasured. The lag mechanism operates at m=0.1 too — 10 batches at hot LR is a real weight distance (EXP-038's ep5 reading implies enormous per-batch drift early), and the tail drift that damaged 0.02 also ages 0.1's constants, just 5× less.
- Everything else remains closed: explicit+implicit constants audited; capacity/depth/head/init/shortcut/activation/attention closed; schedule/optimizer/noise bracketed; regularization peaked; data order has no in-regime evidence; throughput at numerics floor; absorption law screens out-regime imports.
- Screens: in-regime evidence (this candidate = our own dose-response), free-in-all-currencies (zero dt/heat/noise/numerics — same fused op), σ-aware (effect honestly small; zero cost makes the null free).

## Candidate Ideas

### 1. BN running-stat momentum 0.1 → 0.25 (freshness direction — complete the dose-response of the one LIVE dial)
**Summary**: Same 4-line shape as EXP-038 with the opposite sign: `BN_MOMENTUM = 0.25` (effective ~4-batch horizon vs default ~10) at all three BN construction sites. Execution signatures byte-identical to baseline (dt 22.4ms, ~139 epochs, params 4,286,026, VRAM 1613MB).

**Reasoning**: EXP-038 measured the dial's slope from below: smoothing (longer horizon) costs −0.30 via lag, with the mechanism visible at every phase including the plateau (the cosine tail drifts weights to the final eval — best-at-last-epoch in 3 consecutive experiments). The same mechanism predicts m=0.1's constants are also stale, 5× less so. Freshening to m=0.25 buys ~2.5× less lag at near-zero variance cost (2,048 samples per estimate — SE negligible at BN's per-channel granularity). This is the only candidate in the program backed by a measurement taken THIS SESSION on THIS recipe — the purest in-regime evidence possible. Honest effect size: +0.0–0.15 if 0.1 is already near-flat; up to +0.3 only if lag at 0.1 is still the dominant constants-error term (falsifiable). Either way the read completes the dose-response curve: {0.02: −0.30, 0.1: 0, 0.25: ?} — a flat/negative read establishes m=0.1 as the measured optimum from BOTH sides and closes the dial permanently.

**Sources**: exp-report-038.md (lag mechanism + plateau-drift evidence); exp-report-029.md (constants sensitivity); EXP-027 σ; PyTorch BN EMA semantics.

**Estimated Effort**: low — identical shape to EXP-038's diff; standard composite run, baseline watchdog.

**Risk Assessment**: (a) variance side-effect at ~4-batch horizon — arithmetic says negligible, and the run measures it directly (plateau scatter is the diagnostic); (b) flat read = free null (zero deficit currency spent), closes the dial; (c) no risk in any closed currency.

### 2. Derandomized alternating horizontal flip (airbench-style coverage guarantee)
**Summary**: Replace iid RandomHorizontalFlip with deterministic per-epoch alternation so every image is seen flipped and unflipped in equal measure.

**Reasoning**: The last unmeasured micro-dial on the data axis, and zero-dt. But (a) the effect is calibrated to ~10-epoch budgets where flip-sampling variance matters — at 139 epochs iid sampling already covers both orientations ~70× per image (expected effect ≈ 0 by coverage arithmetic); (b) implementation is NOT clean here: per-epoch state does not propagate to persistent DataLoader workers (`persistent_workers=True`), so it needs a virtual-length-100k dataset or worker-state plumbing — complexity and epoch-semantics churn for an expected-zero read; (c) the absorption law's anchor-mismatch discount applies (airbench regime).

**Sources**: airbench (knowledge README row); EXP-035/036/037 absorption entry.

**Estimated Effort**: medium (worker-state gotcha).

**Risk Assessment**: Expected-zero with implementation risk — dominated by Candidate 1 on every axis.

### 3. Classifier-head zero-init
**Summary**: fc weight zero-init (uniform logits at step 0), one line.

**Reasoning**: Third appearance on the candidate list; still screened — the init law (EXP-018/019: nothing init-time moves a 139-epoch converged plateau, measured both directions) predicts the null in advance, and no in-regime evidence contradicts it. Kept only as the zero-cost slot-filler of last resort.

**Sources**: goal-learnings init entries; brainstorm-037/038 screening.

**Estimated Effort**: low.

**Risk Assessment**: Free but information-free — re-measures a measured law.

## Idea Evaluation

Evidence strength: Candidate 1 stands alone — it follows a slope measured 30 minutes ago on this exact recipe (the definition of in-regime), where Candidates 2 and 3 are predicted-zero by coverage arithmetic and by a twice-measured law respectively. Mechanism clarity: precise and quantitative — constants-staleness scales with horizon × weight-drift-rate; EXP-038 measured the staleness cost at 50 batches; 0.25 tests whether 10 batches is still inside the damage zone. Expected impact: honestly modest (the plateau-scatter arithmetic caps the likely gain below the bar), but it is the only candidate whose UPSIDE scenario (+0.3 if lag dominates at 0.1) is consistent with an actual measurement rather than hope. Risk: zero-cost in every closed currency; both outcomes are informative (gain → first improvement in 33 experiments on a genuinely live dial; flat/negative → m=0.1 certified optimal from both sides, dial closed with a complete curve). Feasibility: identical diff shape to the experiment just run. Candidate 1 wins decisively.

## Chosen Idea
**Selected**: BN running-stat momentum 0.1 → 0.25 (Candidate 1)

**Why this idea**:
It is the unique candidate backed by in-regime evidence with a measured sign: EXP-038 proved the BN-constants dial is live and lag-dominated, the best-at-final-epoch pattern proves the tail still drifts (so even 10-batch-old constants are stale), and the variance penalty of a 4-batch horizon is arithmetically negligible at batch 512. Zero cost in every closed currency means even the null is free and completes the dose-response bracket around the default.

**Hypothesis**:
Fresher constants (m=0.25, ~4-batch horizon) reduce the residual staleness that m=0.1 carries through the hot phase and the drifting tail, raising plateau evals — predicting best_test_acc ≥ 96.81 in the strong form (residual lag cost at 0.1 is ≥ +0.25). Execution signatures byte-identical to baseline (dt 22.4ms, 139±4 epochs, params 4,286,026). Falsified by a plateau within the baseline band (96.4–96.7) → m=0.1 is the measured optimum from both sides ({0.02: −0.30, 0.25: ~0}), the dial closes, and the diagnostic suite (hot-phase evals ep5/10/20, plateau mean/scatter, test_loss) decomposes whichever way it lands.
