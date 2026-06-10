# Brainstorm EXP-046
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **SGDR: Stochastic Gradient Descent with Warm Restarts** (`knowledge/papers/sgdr-cosine-schedule.md`)
  Cosine annealing is a low-risk replacement for abrupt step drops under a fixed step/time horizon. The saved project note recommends testing no-restart cosine before restart complexity.
- **Local sibling run observation** (`../v2.9.6-opus-4-8/run.log`, `../v2.9.6-opus-4-8/train.py`)
  A separate local run used a time-fraction cosine decay and reached high late accuracy, but it bundled many architecture and regularization changes. Treat it as directional evidence for budget-matched cosine scheduling, not as a clean result for this goal.

## Experimental History Review

- Current best is EXP-038 at `best_test_acc=93.97%`; the explicit improvement threshold is 94.07% because increases below +0.10 percentage points are too noisy to count.
- The current anchor is `STAGE_WIDTHS=(28, 56, 112)`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, reflection crop padding, `label_smoothing=0.05`, FP32 compile plus channels-last, and a first LR drop at step 21,000.
- High-importance failures rule out isolated second LR drops, weight averaging variants, and widening beyond the current 28/56/112 anchor as first choices.
- Medium-importance failures argue against changing scalar initial LR, label smoothing, batch size, or cutout-style regularization in isolation.
- Isolated mild RandAugment reached 93.83% but did not clear the 94.07% threshold, so augmentation-only changes need a more targeted mechanism.
- The main untested gap is replacing abrupt step milestones with a schedule shape that is calibrated to the fixed wall-clock budget rather than to a brittle realized-step estimate.

## Candidate Ideas

### 1. Time-Budget-Matched No-Restart Cosine on Current Anchor
**Summary**: Replace the current `MultiStepLR([21000, 64000])` with a no-restart cosine decay driven by elapsed training-time fraction. Keep the model width, batch size, initial LR, momentum, weight decay, reflection crop, label smoothing, FP32 compile, and channels-last settings unchanged. Update LR once per training step using `total_training_time / TIME_BUDGET_S`, with optional zero warmup or very short warmup only if planning finds it necessary.

**Reasoning**: The current schedule depends on hitting a fixed step milestone under a wall-clock budget where throughput can vary with contention. A time-fraction cosine schedule uses the whole training budget and reaches low LR near the end regardless of whether the run completes 23k, 35k, or 41k steps. This directly targets schedule shape without retuning scalar LR or adding regularization.

**Sources**: `knowledge/papers/sgdr-cosine-schedule.md`; EXP-038 current anchor; failed schedule-only second drops in EXP-003, EXP-024, EXP-030; local sibling time-fraction cosine observation.

**Estimated Effort**: low

**Risk Assessment**: Cosine may decay too gradually compared with the proven 21k step drop, leaving too little low-LR refinement. The worst case is a valid no-improvement run below 94.07%, with clean attribution to the schedule shape.

### 2. Targeted Color-Only Augmentation on Current Anchor
**Summary**: Add a mild photometric augmentation such as small `ColorJitter` after crop/flip and before tensor conversion, while preserving all optimizer, architecture, and schedule settings. Avoid geometric or erasure augmentation, which already failed in nearby forms.

**Reasoning**: RandAugment mixed photometric and geometric policy changes and added overhead; cutout-style masking over-regularized. A narrower color-only perturbation could improve invariance without changing crop geometry or erasing pixels. It also preserves the current optimization dynamics.

**Sources**: EXP-044 RandAugment no-improvement; EXP-005 and EXP-009 cutout failures; current augmentation anchor from `train.py`.

**Estimated Effort**: low

**Risk Assessment**: The effect may be too small to clear +0.10 points, or added CPU augmentation overhead could reduce step budget. If it fails, it likely fails as a clean no-improvement rather than an invalid run.

### 3. High-Capacity Budget-Matched Cosine Bundle
**Summary**: Test a larger WRN-style width multiplier with a budget-fraction cosine schedule and stronger regularization, inspired by the local sibling run. This would be a bundled architecture, schedule, augmentation, and loss experiment inside `train.py`.

**Reasoning**: The sibling run suggests a sufficiently retuned high-capacity recipe can use the fixed budget well and reach substantially higher accuracy. However, many ingredients in that bundle failed locally when isolated, so the causal attribution would be poor.

**Sources**: `../v2.9.6-opus-4-8/train.py`; `../v2.9.6-opus-4-8/run.log`; local failures for width increases, BF16/TF32, cutout, projection shortcuts, and Nesterov.

**Estimated Effort**: medium

**Risk Assessment**: This has the highest possible upside but the least clean attribution. It risks violating the established local lesson that widening beyond 28/56/112 is a poor first choice under the current recipe. It is better reserved for after the isolated schedule test.

## Idea Evaluation

Candidate 1 has the best balance of evidence, mechanism clarity, and low implementation risk. It is directly supported by the saved SGDR note and targets a real protocol weakness: the current schedule is step-based while the benchmark is time-budgeted. It also avoids known recurring failures around weight averaging, extra width, second-drop tweaking, and scalar LR retuning.

Candidate 2 is plausible but has weaker evidence after EXP-044. A narrow color-only augmentation may be less harmful than RandAugment, but the expected effect size is likely small relative to the 94.07% threshold.

Candidate 3 has the largest upside from a local sibling observation, but it is a broad bundle that contradicts several local negative findings in isolation. It should be considered later if clean schedule and targeted augmentation tests fail, because a bundled success would be useful but harder to analyze.

## Chosen Idea
**Selected**: Time-Budget-Matched No-Restart Cosine on Current Anchor

**Why this idea**:
It is the cleanest untested schedule-shape change and respects every current anchor setting that has survived local bracketing. Driving cosine by elapsed training-time fraction makes the schedule robust to variable realized steps while preserving the fixed wall-clock harness.

**Hypothesis**:
A budget-matched no-restart cosine decay on the EXP-038 anchor will improve late optimization stability and raise `best_test_acc` to at least 94.07%, because it replaces the abrupt 21k drop with smoother decay that still reaches low LR before the fixed budget ends.
