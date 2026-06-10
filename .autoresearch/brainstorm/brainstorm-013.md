# Brainstorm EXP-013
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- **TrivialAugment (Müller & Hutter, ICCV 2021)** (knowledge/papers/trivialaugment.md; arXiv:2103.10158): now a
  *validated* lever here (EXP-012, 96.00→96.22). Standard CIFAR-WRN strong-aug recipes pair auto-augmentation with
  Cutout — but the Cutout hyperparameter (hole size) in those recipes is co-tuned WITH the auto-aug, not inherited
  from a no-auto-aug baseline. This project's Cutout(16) was tuned *before* TA existed (EXP-002/003).
- **DeVries & Taylor 2017 (Cutout)**: the 16px hole is the standard value for CIFAR-10 *without* heavy auto-aug.
  When combined with a strong photometric+geometric policy, the *total* augmentation strength rises, so the optimal
  occlusion size typically shrinks (less marginal regularization needed; over-regularization slows convergence at a
  fixed budget — the dominant failure mode on this 300s benchmark).

## Experimental History Review
- **Current best 96.22%** (EXP-012, commit 6c417a4): k=4 WRN + GPU-Cutout(16) + **TrivialAugmentWide** + bf16/
  channels_last/cosine/Nesterov/LS + compile. Fair 91-epoch run, final_test_loss 0.195 (< prior 0.204).
- **Just-validated lever** (goal-learnings High / project-insights High): strong, diverse augmentation breaks the
  plateau; a *weak* one (Mixup, EXP-011) did not. Correction recorded: don't declare an aug axis closed from
  weak-variant nulls.
- **Closed axes (do NOT revisit)**: width ≥k5 (epoch wall, EXP-004/009), weight-decay (EXP-005), EMA/SWA with
  cosine-to-0 (EXP-006), more-epochs-alone (EXP-007), SE/channel-attention (EXP-008), activation/SiLU (EXP-010),
  *weak* Mixup-on-old-recipe (EXP-011).
- **Dominant constraint**: 300s wall-clock epoch wall — over-regularization (too-strong aug) risks underfit. TA did
  NOT cost throughput (8ms/step ≈ compiled-k4; single PIL op, no GPU sync).
- **Untried gap opened by EXP-012**: the Cutout hole size was never co-tuned with TA; the occlusion sweet spot
  likely shifted. Also untried: TA-only (drop Cutout), RandAugment(N=2), complementary regularizer on top of TA.

## Candidate Ideas

### 1. Reduce Cutout hole size 16→8px under the TA+compile recipe
**Summary**: On the new EXP-012 baseline (TA + Cutout(16) + compile), change `CUTOUT_SIZE = 16` → `8`. Single-variable
tune of the occlusion strength now that TrivialAugment supplies substantial additional augmentation diversity.
**Reasoning**: The 16px hole was tuned with NO auto-augmentation (EXP-002/003). TA adds photometric+geometric
augmentation, raising *total* augmentation strength; the standard effect is that the optimal Cutout size shrinks —
a 16px occlusion stacked on TA may now over-regularize and slow convergence against the 300s budget (the project's
dominant failure mode). A smaller 8px hole keeps orthogonal occlusion benefit while reducing total regularization,
potentially letting the model fit better in the fixed epoch budget. Directly exploits the fresh win.
**Sources**: EXP-012 report (TA win); EXP-002/003 (Cutout(16) tuned pre-TA); knowledge/papers/trivialaugment.md;
project-insights High (over-regularization → underfit at fixed budget).
**Estimated Effort**: low (one-constant change; compile already in baseline).
**Risk Assessment**: Likely-graceful — worst case is a noise-scale null (lands ~96.0–96.2). Two-sided risk: if the
16px hole was already optimal under TA, 8px slightly under-regularizes (small loss). The +0.1 bar over a 96.22
baseline (→ ≥96.32) is demanding and the delta may fall inside the ~0.2pp noise band, so corroborate any gain with
final_test_loss (should drop) and the late-eval cluster, not the single best epoch.

### 2. Drop Cutout entirely (TrivialAugment-only)
**Summary**: Remove the GPU Cutout op, keep TA + compile. Tests whether TA *subsumes* Cutout's regularization.
**Reasoning**: If TA's diversity already covers what Cutout provided, dropping Cutout removes redundant
over-regularization → faster convergence → possible gain; it also isolates TA's standalone contribution.
**Sources**: EXP-012 report; DeVries & Taylor 2017.
**Estimated Effort**: low (delete one line).
**Risk Assessment**: Higher downside than Idea 1 — Cutout's occlusion is partly orthogonal to TA (TA's "Cutout"-like
ops are not guaranteed each step), so removing it could lose the proven ~+0.5pp Cutout gain and regress below 96.0.
Idea 1 (shrink, not remove) is the safer exploitation of the same hypothesis.

### 3. RandAugment (num_ops=2, magnitude≈9) instead of TrivialAugment
**Summary**: Swap `TrivialAugmentWide()` for `RandAugment(num_ops=2, magnitude=9)` — a *two-op* policy (stronger than
TA's single op).
**Reasoning**: More augmentation ops per image = stronger regularization/diversity; if the model has headroom, could
beat TA.
**Sources**: RandAugment (Cubuk et al. 2020); knowledge/papers/trivialaugment.md (TA ≈ RA in lit).
**Estimated Effort**: low (one-line swap).
**Risk Assessment**: Two ops is stronger aug → higher underfit risk at the 300s budget, and TA already matches/beats
RA in the literature, so the expected upside over the working TA recipe is low. Mainly a sideways comparison.

## Idea Evaluation
All three exploit the EXP-012 win and are single-line, low-effort, train.py-only changes. Idea 1 (shrink Cutout) has
the clearest **mechanism** (co-tuning the occlusion size with the newly-added auto-aug; total-augmentation-strength
argument is textbook) and the **safest failure mode** (shrinking, not removing, keeps the proven occlusion benefit —
graceful null at worst). Idea 2 (drop Cutout) tests a related hypothesis but with materially higher downside (could
forfeit the proven Cutout gain and regress below the *old* 96.0). Idea 3 (RandAugment) is a sideways swap with low
expected upside (TA ≈ RA in lit) and higher underfit risk from two ops.

Evidence + mechanism clarity + safe failure mode select **Idea 1**. It is the canonical "co-tune the occlusion with
the auto-aug" step and the natural follow-on to the EXP-012 improvement. Idea 2 is the logical next probe if Idea 1
shows that *less* occlusion helps (it would motivate testing the zero-occlusion endpoint).

## Chosen Idea
**Selected**: Reduce Cutout hole size 16→8px under the TA+compile recipe

**Why this idea**:
It directly exploits the fresh EXP-012 win through the highest-clarity mechanism: the Cutout(16) sweet spot was tuned
*before* TrivialAugment existed, so adding TA raised total augmentation strength and very likely shifted the optimal
occlusion size downward. Shrinking (rather than removing) the hole is the safe, single-variable test of that
hypothesis — it retains Cutout's orthogonal occlusion benefit while reducing the over-regularization that, at this
300s budget, manifests as underfit. Low-effort, train.py-only, clean attribution (compile is null per EXP-007).

**Hypothesis**:
Reducing `CUTOUT_SIZE` 16→8 on the TA+compile recipe will reduce total augmentation strength enough to improve
convergence within the 300s budget, lifting `best_test_acc` above the 96.32 bar (expected ~96.3–96.5%), with a
corroborating drop in final_test_loss below 0.195. If acc is flat/down and loss unchanged, the 16px hole was already
near-optimal under TA and the occlusion-size axis is settled.
