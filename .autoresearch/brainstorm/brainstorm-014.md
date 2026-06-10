# Brainstorm EXP-014
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- **RandAugment (Cubuk et al., 2020)** (https://arxiv.org/abs/1909.13719): applies `num_ops` randomly-chosen
  augmentation ops per image at a single shared `magnitude`. The canonical strong-aug recipe for CIFAR WideResNets is
  **RandAugment + Cutout**. torchvision `transforms.RandAugment()` defaults to `num_ops=2, magnitude=9` (the
  standard CIFAR setting) — drop-in, no new dependency (verified torchvision 0.24.1).
- **TrivialAugment vs RandAugment** (knowledge/papers/trivialaugment.md): TA (1 op, uniformly-random strength)
  matches/beats *tuned* RA in the literature. RA's distinction is **tunability** (num_ops, magnitude are explicit
  knobs) and that **num_ops=2 applies two ops per image** — strictly *more* augmentation per sample than TA's single
  op. Given EXP-012/013 show the augmentation-strength axis is live and points toward MORE aug, RA(2,9) is the
  natural probe of "more ops per image."

## Experimental History Review
- **Current best 96.22%** (EXP-012, commit 6c417a4): k=4 WRN + Cutout(16) + **TrivialAugmentWide** + compile, loss 0.195.
- **Augmentation-strength axis is LIVE and points UP** (goal-learnings High + Failed-Low): EXP-012 ADDED strong
  diverse aug (TA) → +0.22pp; EXP-013 REDUCED aug (Cutout 16→8) → −0.30pp, loss rose 0.195→0.202 (under-regularized).
  So more/stronger augmentation is the productive direction; Cutout sweet spot under TA is ≥16px.
- **Closed axes (do NOT revisit)**: width ≥k5, weight-decay, EMA/SWA (cosine-to-0), more-epochs-alone, SE, SiLU,
  *weak* Mixup on the OLD (Cutout-only) recipe, *shrinking* Cutout.
- **Dominant constraint**: 300s epoch wall — *too-strong* aug risks underfit. RA(2,9) is moderate (2 ops at mag 9/31),
  not extreme; TA did not cost throughput and RA (also CPU PIL ops, no GPU sync) should match.
- **Untried gaps**: stronger/tunable auto-aug (RandAugment), larger Cutout (≥16px), complementary regularizer on the
  TA recipe.

## Candidate Ideas

### 1. RandAugment(num_ops=2, magnitude=9) replacing TrivialAugmentWide (keep Cutout(16) + compile)
**Summary**: Swap `transforms.TrivialAugmentWide()` → `transforms.RandAugment()` (defaults num_ops=2, magnitude=9) in
the train pipeline. Keeps Cutout(16) and compile. RA(2,9) is the standard strong CIFAR-WRN auto-aug; with 2 ops/image
it applies *more* augmentation per sample than TA's single op — a direct probe of the demonstrated "more aug helps" axis.
**Reasoning**: EXP-012 (TA gained) + EXP-013 (less-aug lost) establish that the augmentation-strength axis is live and
points up. RA(2,9) is the canonical, tunable way to push augmentation strength beyond TA's single op, with strong
literature support as a top CIFAR recipe. Unlike TA it exposes explicit knobs, so if it helps, magnitude/num_ops can
be tuned in follow-ups. Clean attribution: compile is null (EXP-007); Cutout/compile unchanged → any delta is RA-vs-TA.
**Sources**: RandAugment (arXiv:1909.13719); knowledge/papers/trivialaugment.md (TA≈RA, RA tunable); EXP-012 (TA win),
EXP-013 (less-aug regression); goal-learnings High/Failed-Low.
**Estimated Effort**: low (one-line transform swap; RA defaults are the CIFAR setting).
**Risk Assessment**: Two-sided. (a) Lit says TA≈RA → likely-null (lands ~96.1–96.3, within noise of TA's 96.22) →
the +0.1 bar over 96.22 (≥96.32) may not be cleared; (b) 2 ops could over-augment → mild underfit (loss↑). Downside
bounded: baseline 96.22 (TA) holds on no-improvement. Corroborate any gain with final_test_loss (< 0.195) + late-eval
cluster, not a lone epoch. Throughput: RA is CPU PIL ops like TA (no GPU sync) → expect ~8ms/step, fair ~90-epoch run.

### 2. Larger Cutout (20px) under the TA recipe
**Summary**: On the EXP-012 baseline, raise `CUTOUT_SIZE` 16→20. EXP-013 showed 8px < 16px (under-regularized), so
the occlusion sweet spot under TA is ≥16px — test whether it's >16.
**Reasoning**: Continues the EXP-013 gradient (8<16) in the direction the data points (larger), single-variable, cheap.
**Sources**: EXP-013 report (8px under-regularized); DeVries & Taylor 2017.
**Estimated Effort**: low (one constant).
**Risk Assessment**: Low ceiling — 16px is the textbook CIFAR-10 Cutout optimum even *without* auto-aug, and TA already
adds strength, so 20px likely over-regularizes → underfit (loss↑). Occlusion is the same mechanism already near its
peak; less information than probing a different/stronger aug policy.

### 3. Mild Mixup (α=0.2) stacked on TA + Cutout(16)
**Summary**: Add per-batch Mixup on top of the TA recipe (a third, interpolation-based mechanism).
**Reasoning**: Mixup is orthogonal to TA (photometric/geometric) and Cutout (occlusion); the EXP-011 Mixup null was on
the OLD Cutout-only recipe, so the overfit/underfit balance differs now.
**Sources**: EXP-011 (Mixup null on old recipe); Mixup (Zhang et al. 2018).
**Estimated Effort**: low-medium (re-add the EXP-011 Mixup block).
**Risk Assessment**: Higher underfit risk — three stacked augmentations at a 300s budget likely slows convergence; Mixup
also raises test loss (soft targets). EXP-011 already showed α=0.2 Mixup null; adding it onto an *already* strongly
augmented recipe is more likely to underfit than help. Lower evidence than RA.

## Idea Evaluation
The augmentation-strength axis is the one demonstrated-live lever (EXP-012/013), so all three candidates stay on it.
Idea 1 (RandAugment 2,9) has the strongest **evidence** (canonical CIFAR-WRN strong-aug, RA+Cutout is standard) and
the clearest **mechanism** for "more augmentation" (2 ops/image > TA's 1) while being a clean single-line swap with
**bounded downside** (TA baseline holds). Its honest weakness is that TA≈RA in the literature, so it may null — but it
also opens explicit tunable knobs for follow-ups, giving it the best information value. Idea 2 (larger Cutout) is a
safe single-variable continuation but low-ceiling (16px is already the occlusion optimum; pushing higher likely
over-regularizes). Idea 3 (Mixup-on-TA) has the highest underfit risk and the weakest evidence (EXP-011 α=0.2 null).

Evidence + mechanism clarity + tunability + bounded downside select **Idea 1**. If it nulls, the explicit knobs
(magnitude, num_ops) and Idea 2 (larger Cutout) give clear bracketing follow-ups.

## Chosen Idea
**Selected**: RandAugment(num_ops=2, magnitude=9) replacing TrivialAugmentWide (keep Cutout(16) + compile)

**Why this idea**:
EXP-012 and EXP-013 together show the augmentation-strength axis is live and points toward *more* augmentation.
RandAugment at its standard CIFAR setting (2 ops/image at magnitude 9) is the canonical, well-evidenced way to push
augmentation strength beyond TrivialAugment's single op, and unlike TA it exposes explicit knobs for follow-up tuning.
It is a clean one-line swap on the validated recipe (Cutout(16) + compile unchanged), giving clean attribution and a
bounded downside (the 96.22 TA baseline holds if it nulls).

**Hypothesis**:
Replacing TrivialAugmentWide with RandAugment(2, 9) will apply more augmentation per image and improve generalization,
lifting `best_test_acc` above the 96.32 bar (expected ~96.3–96.6%) with a corroborating final_test_loss ≤ 0.195 at a
fair ~90-epoch converged run. If acc is within noise of 96.22 with loss ≈ 0.195, TA and RA are equivalent here (the
single-op vs two-op distinction doesn't matter) and the auto-aug *policy* axis is saturated; if loss rises and acc
drops, RA(2,9) over-augments at this budget and the sweet spot is TA's lighter single-op regime.
