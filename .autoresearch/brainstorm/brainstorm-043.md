# Brainstorm EXP-043
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review

- **AdamW — Decoupled Weight Decay Regularization (Loshchilov & Hutter, ICLR 2019)**: decoupling weight
  decay from the adaptive gradient update fixes Adam's generalization deficit on image models and makes
  AdamW a competitive (sometimes superior) optimizer with cosine schedules. Canonical modern default;
  the task explicitly invites "modernizing the baseline up to date" and lists the optimizer as fair game.
- **timm / modern-recipe practice**: small image models trained from scratch with AdamW + cosine + warmup
  typically use peak lr ≈ 1e-3–3e-3 and decoupled wd ≈ 0.05; warmup matters for early-step Adam stability.
- No new knowledge-base entry needed yet — AdamW is standard; a distillation will be added if it informs
  future loops after the result.

## Experimental History Review

Current best = baseline 96.22 (EXP-012, TA+Cutout, commit 6c417a4). Bar = 96.32 (+0.1pp). 32 consecutive
no-improvements since EXP-012 (now including EXP-042 deep supervision, −0.31pp).

- **The optimizer FAMILY has never been changed.** All 44 runs used `optim.SGD(lr=0.2, momentum=0.9,
  nesterov=True, wd=1e-4)`. The three "optimizer" experiments only modified SGD's gradients/objective:
  Gradient Centralization (EXP-030/031, loss-only), SAM (EXP-036, −0.33pp/compute-confounded), PolyLoss
  (EXP-041, loss-only). → AdamW (adaptive per-parameter step sizes) is the single largest untested axis.
- **CLOSED axes (do not retry):** capacity (k both ways + reallocation); augmentation family entirely incl.
  cooldown; LR schedule (peak/floor/shape/restarts); regularizer-adds (dropout/SAM); architecture
  (preact/blurpool/ResNet-D/SE/multi-scale-head); activations; weight-averaging (EMA/SWA);
  classifier-head; large-batch (compute-bound); bag-of-tricks; objective/loss-shape; cheap throughput
  (cudnn.benchmark); **intermediate-feature-routing (multi-scale-head EXP-032 + deep-supervision EXP-042)**.
- **TTA off-limits** (integrity-rejected EXP-027/032).
- **High Importance polish-vs-top1 wall (project-insights L61):** axis-independent, EXPLICITLY covers the
  optimizer axis — "top-1 gains require capacity or fundamentally different generalization, NOT
  optimization/objective polish." This predicts AdamW is most likely a regression/null. The probe's value
  is therefore primarily MAP-COMPLETION (definitively closing the optimizer-family question), with a small
  chance the adaptive-optimizer + strong-aug combination behaves differently than the SGD-modification runs.
- **Fairness:** dt must stay ~8ms for an epoch-neutral test; AdamW's two `_foreach_` moment buffers are
  sub-ms on 4.3M params → expected dt-neutral. Watch the dt distribution (EXP-042 Run 1 lesson).

## Candidate Ideas

### 1. AdamW optimizer-family swap (decoupled WD, retuned peak LR, same schedule/aug/seed)
**Summary**: Replace `optim.SGD(lr=PEAK_LR=0.2, momentum=0.9, nesterov=True, weight_decay=1e-4)` with
`optim.AdamW(lr=PEAK_LR=2e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.05)`. Keep everything else
identical: the time-fraction cosine schedule with 5% warmup (scales the new peak), Cutout(16),
TrivialAugmentWide, label smoothing 0.1, batch 128, channels_last, bf16, `torch.compile(reduce-overhead)`,
seed 42, frozen eval. Only `PEAK_LR`, `WEIGHT_DECAY`, and the optimizer constructor change (`MOMENTUM`
becomes unused).

**Reasoning**: The single major axis the project has never tested. AdamW is the canonical modern optimizer,
explicitly fair game, and its adaptive per-parameter steps can converge faster within the fixed 300s budget
under strong augmentation. Compute-neutral (sub-ms `_foreach_` moment updates → dt ~8ms → epoch-neutral
fair test). Clean single-axis change with clear attribution. Whatever the sign, the result closes the
optimizer-family question that has sat open for the entire project.

**Sources**: AdamW (Loshchilov & Hutter, ICLR 2019); knowledge/papers (GC EXP-030/031, SAM EXP-036, PolyLoss
EXP-041 — SGD-modification optimizer runs); project-insights L61 (polish wall); TASK.md (optimizer fair game).

**Estimated Effort**: low — change three lines (PEAK_LR, WEIGHT_DECAY, the optimizer constructor); verify dt.

**Risk Assessment**: High prior of regression/null — the axis-independent polish wall explicitly covers the
optimizer axis, plus the well-known adaptive-optimizer generalization gap on CIFAR ConvNets, plus a mild
LR-retuning confound (2e-3 is a co-change; mitigated by using a literature-standard config and documenting
it). Slightly higher VRAM (2 moment buffers — fine, soft, 98GB). Worst case: graceful no-improvement. No
scope/stability risk (warmup + LS guard early divergence).

### 2. AdamW with a lighter decoupled WD (lr 2e-3, wd 0.02) — alternative tuning
**Summary**: Same as idea 1 but wd=0.02 (lighter), in case wd=0.05 over-regularizes the already
strongly-augmented 90-ep recipe.
**Reasoning**: Hedges the WD choice; the recipe already has TA+Cutout+LS, so a smaller decoupled WD may
generalize better with AdamW.
**Sources**: same as idea 1.
**Estimated Effort**: low.
**Risk Assessment**: This is a tuning variant of idea 1, not a distinct axis — better treated as a follow-up
if idea 1 is a near-miss than as a separate first probe. Same polish-wall prior.

### 3. ResNeXt-style grouped-conv blocks (cardinality at iso-FLOPs) — radical architecture gamble
**Summary**: Convert BasicBlock 3×3 convs to grouped convolutions with increased width at matched FLOPs
(Xie et al. 2017) — more representational power per FLOP, a "fundamentally different generalization" change.
**Reasoning**: The polish wall says top-1 needs capacity/generalization, not optimization; ResNeXt is a
genuine inductive-bias change with real CIFAR top-1 upside at equal FLOPs.
**Sources**: ResNeXt (Xie et al., CVPR 2017); EXP-038 (reallocation not wall-clock-neutral).
**Estimated Effort**: high — block rewrite + width bookkeeping.
**Risk Assessment**: High. Grouped convs are frequently memory-bound on GPU → dt likely rises → epoch-wall
underfit confound (the failure mode that killed k=5/6, BlurPool, fat-head, SAM, deep-supervision Run 1).
Iso-FLOPs ≠ iso-dt here. Highest upside but very likely a dt-confounded, uninformative regression.

## Idea Evaluation

**Evidence strength**: Idea 1 targets the one demonstrably-untested major axis with a standard technique the
task invites. Idea 2 is a tuning variant of idea 1 (not a separate axis). Idea 3 has top-1 upside in theory
but is repeatedly undercut here by the iso-FLOPs≠iso-dt reality (EXP-038), making a clean test unlikely.

**Mechanism clarity**: Idea 1 has the cleanest single-axis attribution (optimizer family, all else fixed) and
a concrete budget-relevant mechanism (adaptive convergence in fixed time). Idea 3's mechanism is real but its
prerequisite (dt stays flat with grouped convs) is unlikely.

**Risk / value**: Idea 1 is compute-neutral, low-risk, fast, graceful-failing, and — crucially — closes the
largest open question in the map regardless of sign. Idea 3 is high-risk and most likely produces a
dt-confounded regression that teaches little (we already know iso-FLOPs≠iso-dt). Idea 2 is better as a
follow-up than a first probe. Idea 1 is the best risk-adjusted, highest-map-value next step; if it (as the
polish wall predicts) does not gain, the optimizer-family axis is definitively closed and EXP-044 can take
the higher-variance architecture route with a clearer map.

## Chosen Idea
**Selected**: Idea 1 — AdamW optimizer-family swap

**Why this idea**:
With every generalization-class lever now closed (capacity, augmentation, classifier-head, and as of EXP-042
intermediate-feature-routing), the remaining genuinely-untested major axis is the optimizer family itself.
All 44 experiments used SGD; AdamW has never been tried as a whole optimizer (only SGD-gradient/objective
modifications). It is a clean, compute-neutral, single-axis change explicitly invited by the task. The
honest prior (axis-independent polish wall + adaptive generalization gap) is regression/null, but actually
TESTING it — rather than assuming — definitively closes the optimizer-family question and is the rational,
low-risk next probe before resorting to high-risk architecture gambles.

**Hypothesis**:
Swapping SGD+Nesterov for AdamW (lr 2e-3, decoupled wd 0.05) under the identical cosine+warmup schedule and
TA+Cutout+LS recipe will, via faster adaptive convergence in the fixed 300s budget, push `best_test_acc`
above the 96.32 bar at a throughput-neutral ~90 ep. Honest prior: a small regression/within-noise null is
most likely (≈95.8–96.2; tuned SGD usually wins final CIFAR accuracy). Falsified if best_test_acc lands at or
below baseline within noise — which itself closes the optimizer-family axis and licenses an architecture
gamble next loop.
