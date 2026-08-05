# Brainstorm EXP-012 (Quick)
**Created**: 2026-06-29

<!-- Goal/metric/constraints: goals/maximize-cifar10-test-accuracy/01-definition.md. Baseline 96.38 (EXP-008), bar ≥96.48, in 04-results.tsv. -->

## Web Search & Literature Review

- **Bag of Tricks (He et al., CVPR 2019, arXiv:1812.01187)** (consulted EXP-011): exclude BN γ/β + biases from weight decay ("no bias decay"); the dominant effect in a BN net is on the EFFECTIVE LR (BN scale-invariance → conv-weight norm doesn't change block output), isolated returns mixed/diminishing — a modest-confidence tunable. Cited in `knowledge/references/fast-cifar10-recipes.md`.
- **fast-CIFAR lineage** (`knowledge/references/fast-cifar10-recipes.md`): airbench96/hlb use cutout=12 and LS≈0.1; DavidNet/johanwind use LS up to 0.2 — establishes 0.2 as the high end now that our aug matches airbench's cutout=12.
- **ReZero** (`knowledge/references/rezero-identity-init.md`, arXiv:2003.04887): the learnable α=0 gate; uniform weight decay applies a restoring force toward 0 on α, fighting the capacity ramp (net-specific WD-shaping motivation).
- No new external search this loop — the candidate levers were all grounded in the EXP-011 thorough brainstorm (proposals reused below).

## Experimental History Review

Current best **96.38% (EXP-008)**: DavidNet/ResNet-9 + whitening + ReZero(256)@layer2 + EMA(0.998) + flip-TTA, SGD-Nesterov lr 0.4 / wd 5e-4 / LS 0.2, time-based one-cycle, Cutout(12)+RandomErasing, ~150 ep/300s.

- **Long plateau since EXP-008**: 009 Muon (94.11, diverged), 010 Muon-sweep (96.33, ties SGD), 011 CutMix (96.40, +0.02 within noise). Six of last seven experiments no-improvement.
- **Converged diagnosis** (multi-axis exhaustion): optimizer axis exhausted (Muon ties SGD, EXP-009/010); capacity adds under-anneal (EXP-005 deepen, EXP-007 widen 256→384 cut epochs 150→94); eval-side TTA exhausted (EXP-006); **input-space augmentation now saturating** (EXP-011 — a 2nd strong aug, CutMix, only ties; the 1st, EXP-008, won +0.38).
- **Untried throughput-free levers** (different mechanism than the saturated input-aug): (a) **weight-space** WD-shaping — no-WD on BN γ/β + ReZero α (uniform 5e-4 currently fights the α gate; EXP-004 measured α.grad≈0.0179); (b) **target-space** LS retune 0.2→0.1; (c) **schedule shape** — cosine decay / pct_start / peak-LR (one-cycle shape set EXP-001, never revisited); (d) loss-geometry SAM (moonshot, ~25-30%, deprioritized by the EXP-010/011 "near generalization ceiling" prior).
- **Higher-upside-but-risky untried**: **mild capacity at the proven 8×8 stage** (widen layer2 256→**320**, the pre-registered EXP-007 follow-up at ~1.25× cost vs the failed 1.5×) — capacity, not regularization; higher ceiling but under-anneal risk.

## Diagnose What Limits the Objective

The net is **regularization-bound near its generalization ceiling at 300s**, with input-space augmentation now empirically saturating (EXP-011). The remaining headroom, if any, is in regularization mechanisms NOT yet refreshed — weight-space (WD allocation) and target-space (LS) — and in the schedule shape that governs the low-LR tail where accuracy is set. These are throughput-free (cannot under-anneal). A separate, higher-variance possibility is that the net is mildly capacity-bound at the proven 8×8 stage and a *small* capacity step (256→320) could land on the right side of the capacity/epoch curve that the 256→384 step (EXP-007) overshot. Levers are individually modest (~0.1pp noise floor), so the highest-value move is the one combining a genuinely-different mechanism, low downside, and a net-specific reason to expect a non-zero shift — which points at WD-shaping (the α-decoupling is specific to this net and untested).

## Candidate Ideas

### 1. Recipe-scalar refresh — weight-decay shaping + label-smoothing retune
**Summary**: Split the SGD optimizer into param groups so weight decay (5e-4) applies ONLY to conv/fc weight matrices and is removed (wd=0) for BN γ/β and the ReZero α scalar; and drop LABEL_SMOOTHING 0.2→0.1. Clean `p.ndim<=1` split (verified: no bias params exist; frozen whitening conv stays excluded via `requires_grad`). Best run as a small read: cell-A (WD-shaping only, LS held) and cell-B (WD-shaping + LS 0.1), sharing a same-session baseline; hold PEAK_LR fixed. Full proposal: `experiments/011/proposals/idea-02.md` (reused; reviewer-scored 6.5/5 last loop). Zero throughput cost.

**What it targets**: the regularization *allocation* on the regularization-bound net via a mechanism (weight-space + target-space) DISTINCT from the now-saturated input-space aug. The scalars are stale (set EXP-001/002 at 95.2-95.7%, never retuned after EXP-008's aug change); uniform WD also actively fights the ReZero α capacity gate (net-specific, untested).

**Reasoning**: Throughput-free regularization-shaping is the proven lever class (EXP-008 +0.38), explicitly listed in EXP-008's Next Steps; the α-decoupling is not priced into generic bag-of-tricks accounting. Cannot under-anneal. After CutMix showed a 2nd *input-space* aug ties, switching to a *different* regularization axis is the natural next move.

**Sources**: `experiments/011/proposals/idea-02.md`; arXiv:1812.01187; EXP-004 (α.grad≈0.0179); EXP-008 Next Steps; EXP-011 saturation finding (`03-experiment-learnings.md` Low-Importance CutMix entry, project-insights refinement).

**Estimated Effort**: low (one ~10-line optimizer partition + one constant; 1-3 full runs).

**Risk Assessment**: each knob individually sub-noise on this saturated base → possibly unprovable single-run; WD-shaping is per literature a mixed/diminishing effective-LR knob. Worst case: flat at normal epochs (cannot regress via under-anneal). Honest: modest-confidence, low-downside.

### 2. One-cycle schedule reshape (cosine decay)
**Summary**: Replace the linear post-peak LR decay with a **cosine** decay (same time-based endpoints: peak at pct_start, ~0 at budget end), throughput-free. Optionally probe pct_start 0.15→0.10 (longer tail) and peak 0.4→0.5 as cheap riders. Full proposal: `experiments/011/proposals/idea-03.md` (reused). Schedule shape set in EXP-001, never revisited across 4 recipe generations.

**What it targets**: the low-LR tail where accuracy concentrates (EXP-001 Pattern) — a smoother cosine approach to 0 gives the weight-EMA a lower-variance tail to average (EMA-synergy). Throughput-free, cannot under-anneal.

**Reasoning**: canonical super-convergence finish (Smith one-cycle, SGDR); never tuned here. Distinct lever from regularization. But honestly small: cosine-vs-linear on an already-fully-annealing one-cycle is typically ≤0.1-0.2pp — coin-flip on the bar.

**Sources**: `experiments/011/proposals/idea-03.md`; EXP-001 (tail Pattern), EXP-002 (EMA).

**Estimated Effort**: low (~5-line LR-block edit; 1-4 trials if sweeping shape).

**Risk Assessment**: most likely within-noise; peak 0.5 carries a small stability risk (SGD robust, but watch ep25). Lower upside than #1's net-specific α-decoupling.

### 3. Mild capacity step — widen layer2 (8×8) 256→320
**Summary**: Widen the proven layer2/8×8 stage from 256→320 channels (≈1.25× cost, the pre-registered EXP-007 follow-up after 256→384 at 1.5× overshot). Adds capacity at the full-throughput 8×8 stage (no 4×4 kernel penalty). Target ~120-135 epochs (vs EXP-007's 94).

**What it targets**: a possible mild capacity bound at the proven stage — a *different* axis (capacity, not regularization) with a higher ceiling than sub-noise scalar tweaks, IF the net isn't already capacity-saturated.

**Reasoning**: EXP-004 (ReZero@layer2) was the last capacity win (+0.13); EXP-007's failure was magnitude (epochs 150→94), not location — a milder step might land on the right side of the capacity/epoch curve. Materially different from the failed 256→384 (justified retry).

**Sources**: EXP-007 analysis (`03-experiment-learnings.md` Medium under-anneal entry, "try a milder 256→320"); EXP-004 (layer2 capacity win).

**Estimated Effort**: low-medium (a few-line channel-count change + downstream stem adjust; one full run, watch num_epochs).

**Risk Assessment**: HIGHER risk — capacity adds under-anneal here (recurring failure, count 2). 256→320 still costs epochs; if it drops below ~135 it likely under-anneals and loses. Also the net may simply be capacity-saturated (EXP-005/007 both failed). Higher ceiling but a real chance of regression.

## Review

Cross-model review (Codex) in `01-idea-review.md`. **Pick: Idea 1, Recipe-scalar refresh, WD-shaping as headline** (evidence/reasoning 7/10, impact 5.5/10) over cosine schedule (6/4.5) and mild capacity widen (5.5/**7** — highest ceiling but "repeated capacity under-anneal is the dominant local evidence"). No hard-constraint issues. Three refinements raised, all adopted:

1. **WD-shaping is the headline, not LS** — EXP-011 already tested LS-0.1 (under CutMix) and it lowered the ceiling (96.32 vs 96.40); learnings call the annealed optimum "LS-insensitive" in this range. That test was CutMix-confounded, so a CLEAN LS-0.1-only cell is still informative, but it must not carry the experiment. **Resolution**: cell ordering puts WD-shaping-only first as the primary test; LS-0.1 is a separate isolated cell, not bundled into the headline.
2. **Instrument ReZero α** — the α-decoupling argument cites α.grad but not actual α magnitude. **Resolution**: print the final (and EMA) ReZero α in the summary so we can verify WD-shaping actually lets α grow larger than baseline (the mechanism), not just attribute a number to it.
3. **Same-session baseline control** — fixed seed still varies with host throughput/epoch count (~0.1pp floor); stored 96.38 alone is too weak for a +0.05-0.10pp read. **Resolution**: run a same-session baseline cell (current EXP-008 recipe, unchanged) and read all cells as a cross-cell ranking at matched host load, not just vs the stored baseline.

Cell plan (for the planner): **cell-0** same-session baseline (EXP-008 recipe, byte-identical) · **cell-A (headline)** WD-shaping only (no-WD on BN γ/β + ReZero α), LS 0.2 held · **cell-B** WD-shaping + LS 0.1 · **cell-C** LS 0.1 only (clean, no WD-shaping). Hold PEAK_LR fixed throughout. A win = a cell beating same-session cell-0 by ≥0.10pp AND clearing the stored 96.48 bar, with α instrumentation corroborating the WD-shaping mechanism.

## Idea Evaluation

Adopt the reviewer's pick (Recipe-scalar refresh, WD-shaping headline). It best fits the post-EXP-011 diagnosis: after input-space aug saturated (CutMix tied) and the optimizer axis tied (Muon), a throughput-free regularization-*allocation* change on a DIFFERENT axis (weight-space) with a net-specific, untested reason (uniform WD fights the ReZero α gate) is the highest-value remaining low-downside move. Idea 3 (capacity widen) has the highest ceiling (impact 7) but the dominant local evidence is repeated capacity under-anneal, so it is deferred to a future loop if the throughput-free levers stall (and only with num_epochs pre-registered as the first decision metric). Idea 2 (cosine) is the cheapest but lowest-impact; fold any schedule tweak in as a free rider on a future training-side win. Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: Recipe-scalar refresh — **weight-decay shaping** (headline) + label-smoothing retune (secondary), `experiments/011/proposals/idea-02.md` with the three review refinements above.

**Why this idea**:
Input-space augmentation saturated (EXP-011 CutMix tied) and the optimizer axis is exhausted (EXP-009/010 Muon ties SGD), so the next move is a regularization change on a genuinely different axis. WD-shaping (remove decay from BN γ/β and the ReZero α; keep it on conv/fc) is throughput-free (cannot under-anneal — the failure mode that sank every capacity experiment), is standard practice never applied here, and has a net-specific untested mechanism: uniform 5e-4 currently applies a restoring force toward 0 on the ReZero α capacity gate (EXP-004 measured α.grad≈0.0179, non-negligible vs the gate's data gradient), so decoupling α should let the layer2 block reach a larger steady-state capacity. The reviewer ranked it first on evidence and as the best-aligned lever.

**Hypothesis**:
Removing weight decay from BN γ/β and the ReZero α (cell-A, LS held at 0.2) raises `best_test_acc` above a same-session baseline by ≥0.10pp and clears the 96.48 bar, with the final ReZero α measurably larger than baseline (mechanism corroboration) and `num_epochs` unchanged (~142-150, throughput-free). Falsifiable predictions: (a) if α is essentially unchanged vs baseline, the α-decoupling mechanism didn't fire and any delta is from BN-γ/β decoupling (effective-LR effect) instead; (b) if all cells land within noise of same-session cell-0, the regularization allocation is already near-optimal on this saturated net (no-improvement), and the remaining throughput-free lever (schedule) and the higher-risk capacity lever are next; (c) LS-0.1-only (cell-C) is expected ≈ baseline or slightly below, confirming the EXP-011 "LS-insensitive" read in a clean (CutMix-free) setting.
