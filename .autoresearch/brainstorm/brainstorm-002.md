# Brainstorm EXP-002
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
Canonical CIFAR-10 augmentation/WRN literature (heavily-trodden; citations standard):

- **DeVries & Taylor, "Improved Regularization of CNNs with Cutout" (2017)** (arXiv:1708.04552)
  Cutout (mask one random square region per image) is the canonical regularizer paired with WideResNets:
  WRN-16/28 + Cutout reaches ~95.5–96.9% on CIFAR-10 vs ~95% without. Standard hole size for CIFAR-10 is
  16×16. Pure regularization — no inference/eval change, no extra params, negligible compute.
- **Zagoruyko & Komodakis, "Wide Residual Networks" (2016)** (arXiv:1605.07146)
  The canonical strong-CIFAR recipe pairs width with weight decay 5e-4 (we currently use 1e-4) and dropout
  between conv layers; Cutout/heavier WD are how wide nets reach their ceiling.
- **EXP-001 (current best)**: k=4 WideResNet ({64,128,256}, 4.3M params) + projection shortcuts + EXP-000
  recipe → 94.90% (reports/exp-report-001.md).

## Experimental History Review
Source: experiment-indices/improve-cifar10-test-accuracy.tsv, goal-learnings, project-insights, exp-reports.

- **Current best / baseline**: **94.90%** (EXP-001, commit 0086a21).
- **Trajectory**: 91.73 (BASE ResNet-20) → 92.06 (EXP-000 modern recipe) → 94.90 (EXP-001 widen k=4).
- **EXP-000**: bf16+channels_last+cosine+Nesterov+label-smoothing recipe. **EXP-001**: WRN-style widening
  (the dominant lever; +2.84 pp at only ~28% fewer epochs).
- **Goal-learnings (High)**: widening is the dominant lever and is nearly free on H20; capacity was the
  binding ceiling at k=1. **Project-insights (High)**: VRAM headroom enormous (wide net only 490 MB / 98 GB);
  binding budget is the 300s wall-clock.
- **Untried gaps**: any extra augmentation (Cutout/mixup), weight-decay / dropout tuning for the wide net,
  pushing width further (k=6/8), depth×width tradeoffs, recipe (peak LR) re-tuning.
- **No failed approaches yet** — all three experiments improved.

## Candidate Ideas
First principles: at ~95% with a 4.3M-param WRN trained on only pad-crop-flip augmentation, the model has the
capacity to overfit, so the next ceiling is most likely *regularization/augmentation*, not raw capacity. The
canonical, highest-evidence complement to a WideResNet is Cutout. All ideas keep the EXP-001 model + EXP-000
recipe and edit only train.py.

### 1. Cutout augmentation
**Summary**: Add Cutout to the training pipeline — for each training image, zero out one random 16×16 square
(standard CIFAR-10 hole size), applied after normalization in the train transform (eval transform untouched).
Keep the k=4 model and all recipe knobs fixed. Implement as a small transform appended to `train_tf` (a
custom callable or `transforms.RandomErasing`-style op operating on the tensor) — no new dependencies.

**Reasoning**: Cutout is THE canonical regularizer paired with WideResNets and reliably adds ~0.5–1.5 pp on
CIFAR-10 ResNets (DeVries & Taylor). It directly addresses the now-likely overfitting of the larger model,
costs ~0 compute and 0 params (so the epoch budget is unaffected), touches only the train transform, and
cannot affect the frozen eval. Lowest-risk, best-evidenced next increment.

**Sources**: Cutout (1708.04552); WRN (1605.07146); reports/exp-report-001.md; goal-learnings.

**Estimated Effort**: low (one small transform appended to `train_tf` in train.py).

**Risk Assessment**: Low. At the 79-epoch budget, an aggressive hole could slow convergence slightly; 16px is
the validated CIFAR-10 value and Cutout generally helps even at moderate budgets. Worst case: graceful
no-improvement. No crash/eval risk.

### 2. Push width further (k=6)
**Summary**: Increase WIDTH_MULT 4→6 ({96,192,384}, ~9.7M params), recipe + shortcuts unchanged.

**Reasoning**: If still capacity-bound, more width may keep paying; VRAM/throughput headroom remains.

**Sources**: WRN (1605.07146); reports/exp-report-001.md.

**Estimated Effort**: low (one constant).

**Risk Assessment**: Medium. k=4→6 grows FLOPs ~2.25× → fewer epochs (~50?), and without more augmentation a
bigger model is more prone to overfit — diminishing returns are likely once regularization, not capacity, is
the ceiling. Could underfit at the reduced epoch count.

### 3. Weight-decay increase (1e-4 → 5e-4)
**Summary**: Raise WEIGHT_DECAY to the WRN-standard 5e-4, everything else fixed.

**Reasoning**: Wide nets are typically trained with 5e-4; stronger decay curbs overfitting of the large model.

**Sources**: WRN (1605.07146).

**Estimated Effort**: low (one constant).

**Risk Assessment**: Low–medium. A pure-WD change is a smaller, less reliable lever than Cutout, and 5× WD at
a short 79-epoch budget could over-regularize/underfit. Best as a cheap follow-up or combined with Cutout later.

## Idea Evaluation
All three respect hard constraints (train.py only, no new deps, single GPU/300s, eval once/epoch, no seed hacking).

- **Evidence strength**: Idea 1 (Cutout) has the strongest, most directly-applicable evidence — it is the
  canonical WRN companion with documented CIFAR-10 gains, and our model is now in the regime (high capacity,
  light aug) where it helps most. Idea 3 (WD) is evidenced but a weaker/again-regularizing lever. Idea 2
  (more width) risks the diminishing-returns/overfit regime.
- **Mechanism clarity**: Idea 1 — adds input-space regularization → less overfit → better test acc; crisp and
  compute-free. Idea 2 — more capacity, but likely past the capacity ceiling. Idea 3 — stronger weight penalty,
  but a blunt single knob.
- **Expected impact**: Idea 1 highest given the regime; Idea 2 uncertain (could even regress via fewer epochs);
  Idea 3 modest.
- **Risk profile**: 1 and 3 fail gracefully; 2 has underfit risk from fewer epochs.
- **Feasibility**: all trivial-to-low effort; Idea 1's superior evidence/impact dominates.

Idea 1 (Cutout) wins: best evidence, clear compute-free mechanism, targets the current (regularization) ceiling,
and is isolated to the train transform for clean attribution. WD tuning (Idea 3) and more width (Idea 2) are
natural follow-ups once Cutout's effect is known.

## Chosen Idea
**Selected**: Idea 1 — Cutout augmentation (16×16, one hole) on top of the EXP-001 k=4 WideResNet + EXP-000 recipe.

**Why this idea**:
After widening took us to 94.90%, the model has ample capacity and is trained on only pad-crop-flip
augmentation, so overfitting/regularization is the most likely remaining ceiling. Cutout is the canonical,
best-evidenced regularizer for WideResNets, adds ~0 compute and 0 params (so the 79-epoch budget is
preserved), edits only the train transform, and cannot touch the frozen eval — a clean, low-risk, isolated change.

**Hypothesis**:
Adding Cutout (16×16, one hole) to the training augmentation, with the k=4 model and recipe unchanged, will
raise best_test_acc above the 94.90% baseline (expected ~+0.3–1.0 pp, into the ~95% range) by reducing
overfitting, while completing cleanly within the 300s budget (Cutout adds negligible per-step cost). If it
slows convergence at this budget and lands flat, WD tuning (5e-4) or a smaller hole are the fallbacks.
