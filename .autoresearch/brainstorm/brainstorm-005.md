# Brainstorm EXP-005
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
Standard WRN/CIFAR regularization practice (no new external search needed):

- **Wide ResNets (Zagoruyko & Komodakis 2016, arXiv:1605.07146)**: the canonical CIFAR WRN recipe pairs width
  with **weight decay 5e-4** (and often dropout). We currently run WD **1e-4** — a value inherited from the
  k=1 ResNet-20 era and never re-tuned for the 4.3M-param wide model. Heavier WD curbs overfitting of large
  CIFAR nets and is the most-cited single knob alongside Cutout.
- **Bag of Tricks (He et al. 2019, arXiv:1812.01187)**: WD in the 1e-4–5e-4 range; for higher-capacity nets
  the upper end generally generalizes better given enough regularization headroom.

## Experimental History Review
Source: experiment-indices, goal-learnings, project-insights, exp-reports.

- **Current best / baseline**: **96.00%** (EXP-003, commit f59de56) — k=4 WRN + GPU Cutout, WD 1e-4, 77 epochs.
- **Trajectory**: 91.73 → 92.06 (recipe) → 94.90 (k=4 width) → 95.42 (Cutout) → 96.00 (GPU Cutout). EXP-004
  (k=6) REGRESSED to 95.26 (no-improvement).
- **Goal-learnings**: (High) widening dominant **but only to k=4** at this budget — do NOT widen further;
  (High) Cutout regularizes the wide model; (High) op/aug efficiency is an accuracy lever under the 300s budget.
- **Failed Approaches (Low)**: k=6 underfits (compute-bound, 35 epochs) — squeeze k=4 via regularization/recipe instead.
- **Untried gaps**: WD 5e-4 (never tried), Cutout size/probability tuning, mixup, peak-LR sweep, dropout.

## Candidate Ideas
First principles: k=4 is the capacity sweet spot (EXP-004); the lever now is better generalization of that
fixed model. The most-cited, never-tried single knob is weight decay (WRN uses 5e-4 vs our 1e-4). All ideas
edit only train.py and keep the EXP-003 model/recipe otherwise fixed (full 77-epoch budget preserved).

### 1. Weight decay 1e-4 → 5e-4
**Summary**: Set WEIGHT_DECAY 1e-4 → 5e-4 on the k=4 + Cutout model; everything else fixed.

**Reasoning**: WRN-standard decay we have never used; the 4.3M model with only WD 1e-4 is under-regularized on
the L2 axis. Stronger decay is the most-cited generalization knob for wide CIFAR nets, costs nothing in
throughput (epochs stay ~77), and is a one-line isolated change.

**Sources**: WRN (1605.07146), Bag of Tricks (1812.01187); goal-learnings (regularization is the live lever).

**Estimated Effort**: low (one constant).

**Risk Assessment**: Low. 5× WD could over-regularize, but at 77 epochs with a 4.3M model it is well within the
normal range; worst case graceful no-improvement. No crash/eval risk.

### 2. Tune Cutout (size 16 → 12, or probability p<1)
**Summary**: Reduce Cutout hole to 12px (or apply with p=0.5) to soften the current regularization.

**Reasoning**: 16px on 32×32 erases 25% of the image — possibly slightly strong at 77 epochs; a smaller/
probabilistic hole may shift the bias-variance point favorably.

**Sources**: Cutout (1708.04552); reports/exp-report-002/003.

**Estimated Effort**: low (one constant / add a probability gate in cutout_batch).

**Risk Assessment**: Low. Could go either way (less reg → more overfit); small magnitude.

### 3. Add mixup (α≈0.2) alongside Cutout
**Summary**: Add mixup (convex input/label blends) stacked with Cutout.

**Reasoning**: Strong CIFAR regularizer complementary to Cutout for high-capacity nets.

**Sources**: mixup (Zhang et al. 2017, arXiv:1710.09412).

**Estimated Effort**: medium (blend inputs + (label-smoothed) loss; care with bf16).

**Risk Assessment**: Medium. mixup often needs more epochs to pay off; at 77 epochs it can slow convergence; more moving parts.

## Idea Evaluation
All respect hard constraints (train.py only, no deps, single GPU/300s, eval once/epoch, no seed hacking).

- **Evidence strength**: Idea 1 (WD 5e-4) is the most-cited, directly-applicable knob and is conspicuously
  un-tuned in our recipe — strongest, cleanest evidence. Idea 2 is plausible but lower-evidence (16px is the
  validated default). Idea 3 well-evidenced generally but budget-risky.
- **Mechanism clarity**: Idea 1 — stronger L2 → less overfit of the 4.3M model; crisp, throughput-neutral.
  Idea 2 — adjust reg strength; uncertain sign. Idea 3 — extra reg but uncertain at the budget.
- **Expected impact**: Idea 1 modest but reliable; Idea 2 small; Idea 3 high variance.
- **Risk profile**: 1 and 2 fail gracefully; 3 most likely to disappoint at 77 epochs.
- **Feasibility**: 1 and 2 are one-liners; 3 is more code. Idea 1's evidence + cleanliness win.

Idea 1 (WD 5e-4) wins: the canonical, never-tried, throughput-neutral regularization knob for the wide model,
isolated for clean attribution. Cutout tuning (2) and mixup (3) are follow-ups.

## Chosen Idea
**Selected**: Idea 1 — Weight decay 1e-4 → 5e-4 on the k=4 + Cutout model.

**Why this idea**:
k=4 is the capacity sweet spot (EXP-004 showed more width regresses), so the lever is generalization of that
fixed model. Weight decay is the most-cited WRN regularizer and our value (1e-4) is a leftover from the
original ResNet-20; the standard WRN value is 5e-4. It is throughput-neutral (epochs stay ~77), a one-line
isolated change, and fails gracefully.

**Hypothesis**:
Raising WEIGHT_DECAY to 5e-4 (k=4 + Cutout, recipe otherwise fixed) will raise best_test_acc above the 96.00%
baseline (expected ~+0.1–0.4 pp) by better regularizing the 4.3M-param model, completing cleanly within the
300s budget. If it over-regularizes and lands flat/below, that bounds the useful WD range and points to
Cutout-tuning / mixup instead.
