# Report EXP-013: Reflection padding for RandomCrop
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md
- **Plan**: plans/plan-013.md
- **Log**: logs/exp-log-013.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397. Specific question: does replacing the crop augmentation's zero padding (the last 2016-era recipe component) with reflection padding convert artifact-invariance capacity into signal?

## Idea & Hypothesis

Chosen idea: one-argument change `padding_mode="reflect"` on the existing RandomCrop — framed as a DATA-QUALITY fix (border crops get natural statistics instead of black bands) rather than added regularization, therefore expected to sidestep the saturated regularization axis and compose with the measured optimum at zero cost. Hypothesis: best_test_acc ≥ 96.81 at byte-identical throughput (~139 epochs, dt 22ms, VRAM 1613MB). Runner-ups: batch 1024 @ sqrt LR, WD 2.5e-4. (TTA was integrity-flagged and discarded.)

## Approach

1-line diff in train.py: `transforms.RandomCrop(32, padding=4)` → `transforms.RandomCrop(32, padding=4, padding_mode="reflect")`. No deviations from plan.

## Execution

One run, no retries (task bxk0hsvm4, GPU 0, fully idle node). GPU-side execution was byte-identical to baseline as predicted: 139 epochs / 13418 steps, cum dt 22.4ms flat, zero contention-detector events, VRAM 1613.0MB, params unchanged — the cleanest contention-sanity pass yet. The failure arrived through a channel the plan considered settled: total wall clock hit 626.9s (> 600s cap), because reflect padding's extra per-image PIL cost pushed the 8-worker pipeline below the GPU's ~23k img/s demand, growing loader-fetch stalls from ~50s (baseline) to ~197s. Stalls land outside the timed budget (epochs unaffected) but consume the 10-minute envelope.

## Results

- **Primary metric**: best_test_acc = 96.57% (baseline: 96.71, delta: −0.14pp, −0.14%); fails first on the wall-clock condition (626.9s > 600)
- **Observations**: Double refutation. (1) MECHANISM: the trajectory ran ~6–8pp BELOW baseline through the mid-schedule (ep 20: 79.8 vs ~88; ep 60: 85.4 vs ~92; ep 100: 92.1 vs ~96) at identical throughput — reflection padding does NOT behave as free signal; it behaves as STRONGER augmentation (reflected borders are more diverse and lack the easy black-band cue, so each crop is a harder example). The tail converged (96.46–96.57 plateau, final 96.52 ≈ best) 0.14pp short — precisely the dose-response curve's prediction for another increment of augmentation pressure on the saturated axis (RE +0.83 → TA +0.17 → mixup −0.46 → reflect −0.14 fits between TA and mixup). Notably epoch-1 eval was HIGHER than baseline (39.3 vs ~35) — early on, cleaner borders help; the pressure cost compounds later. (2) SYSTEM: the EXP-004 "PIL cost is worker-absorbed" precedent silently expired when EXP-006's compile raised GPU demand ~22%; the worker margin is now so thin that even a cheap per-image operation tips marginal batches into stalls, and the wall-clock envelope — not the timed budget — becomes the binding constraint.
- **Analysis**: The reframing in the brainstorm ("data quality, not regularization") was wrong, and the result strengthens the saturation learning into a sharper form: ON THIS RECIPE, ANY increase in train-set diversity — however it is framed — is regularization, and the axis is saturated. The augmentation surface is now mapped on four points with a clean monotone dose-response crossing zero between TA and reflect. Separately, the run exposed a new binding constraint for future ideas: the CPU loader sits at ~97% of GPU demand, so any candidate that adds per-image CPU work is wall-clock-infeasible regardless of its accuracy merit (and conversely, ideas that LIGHTEN the loader free wall clock but cannot add epochs — the budget is GPU-time-keyed). Eight consecutive misses; every axis (capacity, regularization/augmentation, peak LR, batch, smoothing, schedule shape untouched except warmup) now has measured boundaries.
- **Key Learning**: On a regularization-saturated recipe, augmentation "quality" changes that increase crop diversity are indistinguishable from added regularization pressure — reflect padding slots cleanly into the dose-response curve at −0.14pp; and CPU-side transform cost is now wall-clock-critical (worker margin ~3%), making the 600s envelope a first-class design constraint.

## Verification

- **Conditions**: pre-condition contention sanity CLEAN (139/139 epochs, detector silent — the cap bust is the change's own reproducible cost, not interference); condition 1 FAILED (total_seconds 626.9 > 600); conditions 2–3 skipped per first-failure stop (observed informally: 96.57 would also have missed; 139 eval lines = 139 epochs)
- **Review Notes**: trustworthy — metric consistent with the eval trail; the wall-clock failure is correctly attributed to the experiment (not infra): training was clean and the stall growth is mechanistically tied to the added PIL work
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid run; wall-clock necessary condition violated; metric also below bar)

## Unexplored Avenues

- **Tensor-side reflect padding on GPU** (F.pad in the training loop after H2D): would avoid the CPU cost — but the accuracy result makes the point moot (the mechanism itself lost −0.14pp).
- **padding_mode="edge"**: milder diversity increase than reflect; same saturation logic applies — expected between zeros and reflect, i.e. within noise. Low value.
- **Reducing an existing regularizer to make room** (e.g. RE p 0.5→0.35 + reflect): re-balancing the saturated budget rather than adding to it; combinatorial and weakly-informative single probes. Low confidence.

## Next Steps

1. **WARMUP_FRAC 0.15 → 0.08** — the last untouched schedule-shape constant; pure budget redistribution with no CPU/loader/regularization interaction (immune to both failure modes just observed). Confidence: low-medium.
2. **WEIGHT_DECAY 5e-4 → 2.5e-4** — the other never-revisited constant; the now-four-point saturation curve argues total regularization is past optimum, and explicit WD is the one component that can be REDUCED without touching the data pipeline. Confidence: low-medium — arguably raised by this loop: if the recipe is over-regularized at the margin, lowering WD is the only untried lever that moves pressure DOWN.
3. **Loader-margin audit before any future data-pipeline idea** — protocol step, not an experiment: any candidate adding per-image CPU work must first estimate stall growth vs the ~120s wall-clock headroom (now codified in goal-learnings/infra-errors). Confidence: high (as protocol).

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
