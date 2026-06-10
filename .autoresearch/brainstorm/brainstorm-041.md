# Brainstorm EXP-041
**Created**: 2026-06-09
**Goal**: goals/improve-cifar10-test-accuracy.md

<!-- Ideation only. Metric/direction/constraints/verification live in the goal file;
     baseline (96.22, commit 6c417a4) lives in experiment-indices/improve-cifar10-test-accuracy.tsv. -->

## Web Search & Literature Review

- **PolyLoss (Leng et al., ICLR 2022, arXiv:2204.12511)** — views classification losses as a polynomial
  expansion of `(1−p_t)`; cross-entropy = Σ (1/j)(1−p_t)^j. **Poly-1** perturbs only the leading term:
  `L_Poly1 = CE + ε·(1 − p_t)`, where `p_t` = softmax prob of the true class. Reported gains over CE on
  2D image classification (ImageNet ResNet), detection, segmentation — "sometimes by a large margin".
  ε is dataset/task-dependent (requires tuning); ImageNet ResNet optimum ≈ +1 to +2. Compute-free (one
  gather+exp on logits already computed), convergence-neutral, no new params/deps.
  Source: https://arxiv.org/abs/2204.12511 ; formula confirmed via PyTorch issue #76732 and ICLR PDF.
- knowledge/README.md: all optimizer/aug/schedule/polish/head papers — CLOSED here. PolyLoss is a NEW,
  unexplored objective-shaping lever (distinct from the closed label-smoothing knob EXP-023 and the
  closed SAM objective EXP-036).

## Experimental History Review

**Current best / baseline**: 96.22% (EXP-012, 6c417a4); bar 96.32 (+0.1). 41 experiments; plateau very
well-confirmed. Recent cluster (036–040) ALL sub-baseline nulls.

**Three established walls (project-insights):**
1. **Compute/epoch wall** — any non-trivial FLOP add OR wall-clock-heavier change → fewer epochs →
   under-train → regress (down to the sub-ms-op scale, EXP-039).
2. **Polish-vs-top1** — compute-neutral OPTIMIZATION polish (EMA/SWA/GC/LS-down/bag-of-tricks) lowers
   loss/flatness not top-1.
3. **Regularizer-adds underfit at 300s** — dropout/CutMix/Mixup/SAM(cost) all underfit/regress.

**CLOSED axes (~33)**: capacity BOTH ways; ALL augmentation; ENTIRE LR schedule (peak/floor/shape,
EXP-016/017/029 — but WARMUP FRACTION specifically never swept); regularizer-adds; architecture
(SiLU/preact/ResNet-D/BlurPool/multi-scale-head/SE); optimizer dynamics (GC) + objective (SAM);
weight-averaging; classifier head BOTH sub-levers (feature-agg EXP-032, cosine-geometry EXP-039); input
std-norm INFEASIBLE; large-batch; bag-of-tricks; cheap throughput (cudnn.benchmark EXP-040, conv dt floor
already reached by torch.compile).

**The loss FUNCTION itself is almost untouched.** The only objective experiments were: label-smoothing
DOWN (EXP-023 — lowered CE loss, top-1 −0.19pp, a polish null) and SAM (EXP-036 — closed on compute
cost). The **leading-polynomial reshaping of the CE objective (PolyLoss) has never been tried** — it is
compute-free, convergence-neutral, and changes per-example gradients (top-1-affecting, not loss-only
polish). With ε>0 it AMPLIFIES the gradient on hard/low-`p_t` examples → behaves as a mild convergence
ACCELERATOR, which directly fits the strongest standing hypothesis (the net is convergence-bound at
~91 ep / 300s). Untried gap.

## Candidate Ideas

First-principles: the metric is plateaued at 96.2; capacity/aug/schedule/regularizer/optimizer/head are
closed. The remaining unexplored compute-free axis is the **training objective's gradient shape**. Of the
objective levers, PolyLoss has the best literature support and a mechanism (hard-example gradient
amplification) that aligns with convergence-bound short-budget training.

### 1. PolyLoss Poly-1 (ε·(1−p_t) added to the CE+label-smoothing loss)
**Summary**: Replace `loss = F.cross_entropy(outputs, targets, label_smoothing=0.1)` (train.py L234-236)
with `loss = CE_with_LS + EPSILON_POLY * (1 - p_t).mean()`, where `p_t` is the softmax probability of the
true (hard) class (`logp = F.log_softmax(outputs,1); pt = logp.gather(1, targets[:,None]).exp().squeeze(1)`).
Keep label smoothing 0.1 unchanged; add a single hyperparameter `EPSILON_POLY = 1.0` (conservative;
paper's ImageNet ResNet optimum ≈ +1 to +2 — pick the low end for a short CIFAR budget to avoid
over-perturbation). Everything else unchanged. Compute-free (gather+exp on already-computed logits),
params unchanged, convergence-neutral.
**Reasoning**: After 41 experiments the objective's polynomial shape is the one untouched compute-free
degree of freedom. Poly-1 with ε>0 increases the loss gradient on examples where the model is unconfident
on the true class (low `p_t`) → accelerates learning of hard examples → more effective convergence within
the fixed 300s/~91-ep budget. This dodges all three walls (compute-neutral → wall #1; objective gradient
reshape that moves the decision boundary → not loss-only polish, wall #2; an accelerator not a
convergence-slowing regularizer-add → not wall #3). Distinct from the closed LS knob (EXP-023 scaled the
target distribution; PolyLoss adds a separate `(1−p_t)` gradient term) and from SAM (which was a compute
cost, not an objective-shape, failure).
**Sources**: PolyLoss ICLR 2022 (arXiv:2204.12511); train.py L232-236 (loss); project-insights
(convergence-bound hypothesis; polish-vs-top1; regularizer-underfit).
**Estimated Effort**: low — ~3-line loss change + one constant. One run.
**Risk Assessment**: MEDIUM. Clean failure mode (no crash, compute-neutral → no epoch confound). Main
risk: ε mistuned — too large destabilizes/over-amplifies hard (possibly noisy/augmented) examples at the
short budget → underfit/regress; ε=1.0 is the conservative choice. Interaction with label smoothing
(LS softens targets while Poly-1 sharpens hard-example gradients) could partially cancel → null. Honest
expectation: modest gain or within-noise null (objective tweaks LS-down/cosine were null here), but a
genuinely unmapped, cited, compute-free, wall-dodging probe.

### 2. Warmup-fraction sweep (`WARMUP_FRAC` 0.05 → 0.02 or 0.10)
**Summary**: The LR-schedule axis is "closed" for PEAK_LR (EXP-016/017) and shape/restarts (EXP-029), but
the linear-warmup FRACTION (currently 5% of the budget) was never swept. Try a shorter (0.02) or longer
(0.10) warmup.
**Reasoning**: With a high PEAK_LR=0.2 + BN, warmup length trades early stability vs. high-LR exploration
epochs; a different fraction could marginally help convergence.
**Sources**: train.py L24, L35-41; goal-learnings (LR-schedule axis closed — but warmup-fraction sub-lever
not explicitly listed).
**Risk Assessment**: LOW risk / LOW EV. Almost certainly within the "LR schedule fully closed" finding;
warmup over 300s is ~15s (~4-5 ep) so the sweepable effect is tiny. Likely null. Deprioritized.

### 3. BatchNorm momentum tuning / end-of-training BN-stat recompute
**Summary**: Tune `BatchNorm2d` momentum (default 0.1) or recompute running mean/var over training data
in eval mode after the cosine-to-0 tail before final eval.
**Reasoning**: Stale BN running stats after a near-zero-LR tail could mildly hurt eval.
**Sources**: train.py L71/75/83 (BN), L267 (eval); papers/swa.md (BN-recompute).
**Risk Assessment**: LOW EV. Over ~35k updates at momentum 0.1 (effective window ~10 batches) BN stats
are well-converged, not stale → recompute is likely a no-op; momentum tuning likely null. Deprioritized.

## Idea Evaluation

- **#1 (PolyLoss)** is the clear lead: strongest evidence (ICLR 2022 with reported image-classification
  gains), clearest mechanism (hard-example gradient amplification = convergence accelerator, matching the
  convergence-bound hypothesis), targets the one untouched compute-free axis (objective polynomial shape),
  dodges all three walls, and has a clean compute-neutral failure mode. Honest ceiling is modest (sibling
  objective tweaks were null) but it is the best-positioned untried lever after 41 experiments.
- **#2 (warmup)** and **#3 (BN)** are low-EV: both fall under axes already characterized as closed/saturated
  (LR schedule; BN stats converge), with weak mechanisms. They remain as fallbacks if PolyLoss is chosen
  against, but neither has #1's evidence or mechanism.

**#1 wins** on evidence strength + mechanism clarity + wall-avoidance.

## Chosen Idea
**Selected**: PolyLoss Poly-1 — add `ε·(1 − p_t)` (ε=1.0) to the existing CE+label-smoothing loss, where
`p_t` is the softmax probability of the true class. Single compute-free objective-shape change; everything
else in the recipe unchanged.

**Why this idea**:
After 41 experiments closing capacity, augmentation, LR schedule, regularizers, optimizer dynamics/
objective(SAM), weight-averaging, both classifier-head sub-levers, and cheap throughput, the training
objective's polynomial GRADIENT SHAPE is the one untouched compute-free degree of freedom. PolyLoss Poly-1
is a cited (ICLR 2022) technique with reported image-classification gains; with ε>0 it amplifies gradients
on hard/low-`p_t` examples (a convergence accelerator that fits the convergence-bound hypothesis), is
compute-neutral and convergence-neutral, and changes the decision boundary (top-1-affecting, not loss-only
polish) — so it dodges all three established walls.

**Hypothesis**:
Adding `1.0·(1 − p_t)` to the CE+LS loss amplifies the per-example gradient on hard examples → more
effective convergence within the fixed 300s/~91-ep budget → best_test_acc above the bar 96.32 at
throughput-neutral ~91 ep, params unchanged. Honest most-likely outcome: within-noise (~96.0–96.3), since
sibling objective tweaks (label-smoothing-down EXP-023, cosine head EXP-039) were null and label smoothing
may partially cancel the poly term; a clean null then closes the objective-polynomial-shape sub-lever
(with a possible ε sweep, e.g. +2, noted as a follow-up before fully closing it).
