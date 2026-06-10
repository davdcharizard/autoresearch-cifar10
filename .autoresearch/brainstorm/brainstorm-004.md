# Brainstorm EXP-004
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
Grounded in our own throughput data + standard WRN practice:

- **Wide ResNets (Zagoruyko & Komodakis 2016, arXiv:1605.07146)**: width is the most compute-efficient
  capacity knob; WRN-28-10 (~36M) reaches ~96% on CIFAR-10. Standard WRN uses weight decay 5e-4 and often
  dropout between convs. We currently run WD 1e-4 (k=1-era value).
- **Throughput observation (our runs)**: per-step time barely grew with width — k=1 ~8.6 ms (EXP-000), k=4
  ~10.0 ms (EXP-003) despite 16× params/FLOPs. The H20 is memory-bandwidth / kernel-launch bound at this
  scale, NOT compute-bound → **widening is nearly free in wall-clock** until compute saturates. Implies k=6
  should still fit many epochs.

## Experimental History Review
Source: experiment-indices/improve-cifar10-test-accuracy.tsv, goal-learnings, project-insights, exp-reports.

- **Current best / baseline**: **96.00%** (EXP-003, commit f59de56).
- **Trajectory**: 91.73 (BASE) → 92.06 (EXP-000 recipe) → 94.90 (EXP-001 widen k=4) → 95.42 (EXP-002 Cutout)
  → 96.00 (EXP-003 GPU Cutout / throughput recovered, 77 epochs).
- **Goal-learnings (High)**: widening is the dominant lever and nearly free on H20 (capacity was the ceiling);
  Cutout regularizes the wide model; under the 300s budget, op/aug efficiency is itself an accuracy lever.
- **Project-insights (High)**: VRAM essentially free (k=4 peak 492 MB / 98 GB); 300s wall-clock binds.
- **Untried gaps**: width beyond k=4 (k=6/8), WD 5e-4 (WRN standard, never tried), mixup, dropout, deeper-wider mix.
- **No failed approaches** — four straight improvements.

## Candidate Ideas
First principles: width gave by far the biggest single jump (+2.84) and per-step cost barely rises with width
(H20 memory/launch-bound), so more capacity is likely still the strongest lever — and it is now backed by
Cutout regularization + an efficient pipeline. The main risk (fewer epochs) is smaller than naive FLOP-scaling
suggests. All ideas edit only train.py and keep the EXP-003 recipe (k=4, Cutout, bf16, cosine, etc.) as the base.

### 1. Increase width to k=6 ({96,192,384})
**Summary**: Set WIDTH_MULT 4→6 (~9.7M params), everything else fixed (Cutout, recipe). Adds capacity to the
proven-dominant lever.

**Reasoning**: Width drove the +2.84 jump (EXP-001); per-step time barely grows with width on the H20
(k=1→k=4 was only ~16% slower for 16× FLOPs), so k=6 should still fit ~55–70 epochs — enough to converge with
cosine. Cutout (EXP-002/003) now regularizes the larger model, mitigating overfit. VRAM is free.

**Reasoning/Sources**: WRN (1605.07146); reports/exp-report-001.md (+2.84 from width), exp-report-003.md
(throughput/epoch data); goal-learnings (width dominant & cheap).

**Estimated Effort**: low (one constant: WIDTH_MULT=6).

**Risk Assessment**: Low–medium. If k=6 saturates compute and epochs drop too far (<~40), it could underfit →
graceful no-improvement. No crash risk; VRAM ample. The time-fraction cosine anneals fully regardless of epoch count.

### 2. Weight decay 1e-4 → 5e-4 (WRN standard)
**Summary**: Raise WEIGHT_DECAY to 5e-4 on the current k=4+Cutout model.

**Reasoning**: WRN-standard decay we have never used; stronger L2 curbs overfitting of the 4.3M model. Cheap, isolated, low-risk.

**Sources**: WRN (1605.07146); goal-learnings (regularization is a live lever).

**Estimated Effort**: low (one constant).

**Risk Assessment**: Low. Modest expected magnitude (+0.1–0.3); over-regularization possible but unlikely at 77 epochs.

### 3. Add mixup (α≈0.2) alongside Cutout
**Summary**: Add mixup (convex combos of input/label pairs) to training, stacked with Cutout.

**Reasoning**: Strong CIFAR regularizer; complements Cutout for the high-capacity model.

**Sources**: mixup (Zhang et al. 2017, arXiv:1710.09412).

**Estimated Effort**: medium (mix inputs + blend the (already label-smoothed) loss; care with bf16/label smoothing interaction).

**Risk Assessment**: Medium. mixup often needs more epochs to pay off; at the 300s budget it can slow
convergence and even hurt. Higher uncertainty than 1/2; more moving parts.

## Idea Evaluation
All respect hard constraints (train.py only, no deps, single GPU/300s, eval once/epoch, no seed hacking).

- **Evidence strength**: Idea 1 (k=6) is backed by the strongest project-specific evidence — width gave the
  dominant gain and our throughput data shows it stays cheap; capacity scaling is the through-line of this
  goal. Idea 2 (WD) is well-evidenced but a smaller lever. Idea 3 (mixup) is evidenced generally but
  budget-risky here.
- **Mechanism clarity**: Idea 1 — more capacity on a memory-bound GPU at low wall-clock cost, regularized by
  existing Cutout; crisp. Idea 2 — stronger L2; clear but small. Idea 3 — extra regularization but uncertain
  sign at short budget.
- **Expected impact**: Idea 1 highest upside; Idea 2 reliable but modest; Idea 3 high variance.
- **Risk profile**: 1 and 2 fail gracefully; 3 most likely to disappoint at this budget.
- **Feasibility**: 1 and 2 are one-line changes; 3 is more code. Idea 1's upside + cheap implementation win.

Idea 1 (k=6) wins: capacity is the proven dominant lever, widening is nearly free in wall-clock on the H20, and
the larger model is now regularized by Cutout. WD 5e-4 (Idea 2) is the natural cheap follow-up; mixup (Idea 3)
later if regularization is still the ceiling.

## Chosen Idea
**Selected**: Idea 1 — Increase width to k=6 ({96,192,384}, ~9.7M params), EXP-003 recipe + Cutout otherwise fixed.

**Why this idea**:
Width has been by far the dominant lever (the +2.84 pp jump at k=4), and our own throughput data shows per-step
time barely grows with width on the H20 (memory/launch-bound, not compute-bound) — so k=6 should still fit
enough epochs (~55–70) to converge, while VRAM stays trivial. The larger model is now regularized by Cutout
(EXP-002/003), reducing the overfit risk that pure capacity adds. One-line change, fails gracefully.

**Hypothesis**:
Increasing WIDTH_MULT to 6 (~9.7M params) with Cutout and the recipe unchanged will raise best_test_acc above
the 96.00% baseline (expected ~+0.2–0.6 pp toward ~96.3–96.5%) by adding capacity at low wall-clock cost. If
the epoch count drops far enough to underfit (e.g. <~40 epochs), the result lands flat (no-improvement),
which would itself signal the capacity/epoch trade-off has turned — informing whether to stop widening.
