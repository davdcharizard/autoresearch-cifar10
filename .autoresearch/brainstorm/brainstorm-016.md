# Brainstorm EXP-016
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
- Knowledge base (`knowledge/README.md`): only `papers/trivialaugment.md` so far (the EXP-012 win). No
  prior LR-schedule or batch-size reference saved.
- **LR / warmup / cosine schedule** — standard SGD practice (Goyal et al. 2017, "Accurate, Large Minibatch
  SGD", arXiv:1706.02677; Loshchilov & Hutter 2017, SGDR/cosine, arXiv:1608.03983): linear warmup then cosine
  anneal-to-0 is the modern default (already implemented). The linear-scaling rule ties peak LR to batch size;
  for batch 128 the textbook WRN/ResNet peak is ~0.1, but strongly-augmented recipes routinely run a *higher*
  effective LR because heavy augmentation curbs overfitting and lets the optimizer explore more aggressively
  before annealing. Our peak (0.2) was set heuristically in EXP-000 and never re-tuned after the k=4 widen,
  Cutout, or TrivialAugment were added — i.e. the schedule has never been matched to the current (much more
  regularized) recipe.
- **CutMix** (Yun et al., ICCV 2019, arXiv:1905.04899): cut-and-paste a rectangular patch from another image,
  mix labels by patch-area fraction. A strong, *regional* augmentation+regularizer that consistently improves
  CIFAR/ImageNet in long-budget recipes; mechanism is spatial (closer to Cutout) but with label interpolation
  (like Mixup). Note: label-mixing augmentations typically need *many* epochs to pay off.

## Experimental History Review
- **Current best 96.22%** (EXP-012): k=4 WRN-style (post-act v1 blocks) + Cutout(16) + TrivialAugment + compile,
  test loss 0.195, fair ~91-epoch run. 17 experiments, 6 improvements.
- **Closed / saturated axes** (goal-learnings + project-insights):
  - Capacity (width ≥k5, depth) — monotone epoch wall, compile can't lift it once compute-bound (EXP-004/009).
  - Block micro-architecture / activation ordering (pre-activation) — no gain + throughput confound (EXP-015).
  - Activation function (SiLU), channel attention (SE), EMA/SWA, weight-decay, more-epochs-alone — all null.
  - Augmentation POLICY (RA≈TA) saturated (EXP-014); shrinking Cutout under-regularizes (EXP-013); **weak Mixup
    null** (EXP-011, label-mixing, ~88 ep).
- **Genuinely UNTRIED axes**:
  - **(a) LR schedule** — only weight-decay was ever swept (EXP-005, on the *old* recipe). Peak LR / warmup never
    re-tuned for the current TA+Cutout+compile recipe. Compute-free → a perfectly fair test (no throughput confound,
    which just burned EXP-015).
  - **(b) Batch size** — fixed at 128 the entire time; never swept. On a launch-bound net larger batches can raise
    img/s, but epochs are already saturated and large-batch generalization is finicky.
  - **(c) A different aug *mechanism*** — CutMix (regional label-mixing) is the one well-evidenced strong aug not yet
    tried; larger Cutout (≥16) likely over-regularizes (brainstorm-015 Idea 3).
- **Dominant constraint**: 300s epoch wall; throughput must not drop (EXP-015 lesson: verify realized epoch count,
  FLOPs-neutral ≠ wall-clock-neutral under compile). Sub-~0.2pp deltas are noise.

## Candidate Ideas

### 1. LR-schedule micro-tuning — raise peak LR 0.2 → 0.3 (strong-aug-tolerates-higher-LR)
**Summary**: Change the single constant `PEAK_LR` 0.2 → 0.3, keeping the existing 5% linear warmup and cosine
anneal-to-0. Everything else (k=4, batch 128, Nesterov, WD 1e-4, LS 0.1, Cutout(16), TrivialAugment, compile,
seed 42) is unchanged.
**Reasoning**: The LR schedule is the highest-value knob *never tuned on the current recipe* — peak 0.2 was a
heuristic from EXP-000, set before the k=4 widen, Cutout, and TrivialAugment all piled on regularization. Heavy,
diverse augmentation (TA+Cutout) suppresses overfitting, which typically lets the optimizer run a *more aggressive*
peak LR and explore a wider region before the cosine anneals to 0 — landing in a flatter, better-generalizing
minimum. We already sit at 2× the textbook batch-128 peak (0.1) and it *helped* (EXP-000), evidence the model
likes a higher-than-default LR; pushing modestly further (0.3) tests whether there's more headroom. It is
compute-free, so — unlike EXP-015 — the epoch count is guaranteed unchanged and the test is perfectly fair.
**Sources**: train.py L23 (`PEAK_LR=0.2`), L35-41 (`lr_at_fraction`); Goyal 2017 (arXiv:1706.02677); Loshchilov
2017 (arXiv:1608.03983); EXP-000 (heuristic peak); goal-learnings (LR untried, only WD swept).
**Estimated Effort**: trivial (one constant).
**Risk Assessment**: Direction is somewhat blind — a higher peak could overshoot and slightly *underfit* within
the ~91-epoch budget instead of helping. Mitigated by the 5% warmup + BN tolerance + Nesterov (EXP-000's 0.2 was
stable; 0.3 with warmup should be too). Failure mode is graceful (no-improvement, baseline holds). If 0.3 nulls or
regresses, the immediate next probe is the *opposite* direction (0.1/0.15, textbook). Low-to-medium ceiling but the
cleanest fair test available.

### 2. CutMix (regional label-mixing augmentation), GPU-vectorized per batch
**Summary**: Add CutMix (Yun 2019) as a per-batch GPU op alongside the existing Cutout: with some probability per
batch, paste a random rectangular patch from a shuffled copy of the batch and set the target to the area-weighted
mix of the two labels (cross-entropy over mixed soft targets). Implemented like `cutout_batch` (vectorized, no CPU
sync) to preserve throughput.
**Reasoning**: The augmentation-strength axis is the one that broke the plateau (TA, EXP-012), and CutMix is the
strongest, most-evidenced aug mechanism not yet tried — *regional* (spatially local, akin to Cutout) with label
interpolation, distinct from TA's photometric/geometric transforms and from Cutout's pure occlusion. It targets
the generalization gap directly.
**Sources**: Yun et al. ICCV 2019 (arXiv:1905.04899); train.py L44-57 (`cutout_batch` template), L223 (per-batch
GPU aug pattern); goal-learnings (aug-strength axis live; CutMix listed as remaining move).
**Estimated Effort**: medium (vectorized patch-paste + label-mix + soft-target loss; train.py-only, GPU op).
**Risk Assessment**: Real risk of a null/regression: (a) CutMix is a *label-mixing* augmentation, the same family
as the **weak Mixup that already nulled** (EXP-011) on this regularization-saturated net; (b) label-mixing augs
characteristically need *many* epochs (CutMix papers use 200-300) to pay off, and we only fit ~91 — it may
*underfit* within budget. Test loss will rise (soft-target artifact) as with Mixup; judge on acc only. Higher
ceiling than Idea 1 *if* it works, but lower probability given the Mixup precedent and tight budget.

### 3. Larger batch (256) + linear LR scaling (peak 0.4)
**Summary**: Double `BATCH_SIZE` 128 → 256 and scale `PEAK_LR` 0.2 → 0.4 (linear-scaling rule), keeping warmup/cosine.
**Reasoning**: The net is launch-bound at batch 128 (8ms/step); a larger batch amortizes per-step launch overhead,
potentially raising img/s, and gives lower-variance gradients. VRAM is a non-issue (~0.5 GB of 98 GB).
**Sources**: Goyal 2017 (arXiv:1706.02677); project-insights (launch-bound k=4, VRAM headroom); train.py L22.
**Estimated Effort**: low (two constants).
**Risk Assessment**: Multiple headwinds: epochs are already saturated (more img/s → more epochs won't help, EXP-007);
large-batch SGD has a known generalization gap unless carefully tuned; couples two changes (batch + LR) confounding
attribution. Blind and low-confidence; best deferred.

## Idea Evaluation
The aug axis has now produced a string of nulls/regressions (weak Mixup EXP-011, RA≈TA EXP-014, smaller-Cutout
EXP-013) with only the *input-space* TA actually gaining — and the one strong aug left (CutMix, Idea 2) is a
label-mixing cousin of the failed Mixup that additionally wants far more epochs than our budget allows. So Idea 2 is
higher-ceiling but lower-probability and risks repeating a known null. Idea 3 couples two variables, fights the
saturated-epochs finding, and is blind. **Idea 1 (LR peak retuning)** is the only *genuinely untried, compute-free*
lever: the schedule was never matched to the current heavily-regularized recipe, the mechanism (strong aug → tolerate
a more aggressive peak → flatter minimum) is defensible, and crucially it is a *perfectly fair test* — zero throughput
confound, exactly the failure that muddied EXP-015. Its ceiling is modest and its direction is somewhat blind, but it
has the best evidence-to-risk ratio and, win or lose, cleanly maps the LR axis (a higher-peak result immediately tells
us which direction to probe next). Evidence-to-risk + fair-test cleanliness select **Idea 1**; Idea 2 is the
higher-ceiling fallback if the LR axis proves flat.

## Chosen Idea
**Selected**: LR-schedule micro-tuning — raise peak LR 0.2 → 0.3

**Why this idea**:
The LR schedule is the single highest-value knob never tuned on the *current* recipe (only weight-decay was swept,
EXP-005, on the old recipe). Peak 0.2 was a pre-widen, pre-augmentation heuristic; after k=4 + Cutout + TrivialAugment
piled on regularization, the optimizer can likely tolerate — and benefit from — a more aggressive peak before the
cosine anneals to 0. It is a one-constant, compute-free change, so it is a perfectly fair same-budget test with no
throughput confound (the exact issue that muddied EXP-015), and the result cleanly maps the LR axis regardless of sign.

**Hypothesis**:
Raising `PEAK_LR` 0.2 → 0.3 (same 5% warmup, cosine-to-0) will let the heavily-augmented k=4 model explore a wider
region before annealing and settle in a flatter, better-generalizing minimum, lifting `best_test_acc` above the 96.32
bar (expected ~96.3–96.5) at unchanged throughput/epochs (~91). If acc is within noise of 96.22 or lower, the peak is
already near-optimal (or 0.3 overshoots within the budget), and the next probe is the opposite direction (peak 0.1–0.15).
