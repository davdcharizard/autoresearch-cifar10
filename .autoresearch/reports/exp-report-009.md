# Report EXP-009: Mixup (α=0.2) on the compiled 4x recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-009.md
- **Plan**: plans/plan-009.md
- **Log**: logs/exp-log-009.md

## Goal

Maximize `best_test_acc` (%) of the CIFAR-10 ResNet within the fixed 300s training budget (higher is better). Baseline at experiment time: **96.71%** @ 1990397 (4x ResNet-20, compiled TA+RE+LS recipe, 139 epochs). Specific question: with capacity closed bidirectionally (EXP-007/008), does input-label-space regularization (mixup) reduce the residual train/test gap enough to clear +0.1pp?

## Idea & Hypothesis

Chosen idea: per-batch mixup at the mild α=0.2 setting on the otherwise-frozen recipe. Rationale: mixup is the strongest published lever (+0.5–1pp on CIFAR WRNs at α=1.0/200ep) in a mechanism class distinct from the spatial transforms already stacked (RE, TA), the 139-epoch schedule was the longest yet to absorb its slower convergence, and α=0.2 hedged the over-regularization risk. Hypothesis: best_test_acc ≥ 96.81. Runner-up candidates: EMA eval, compile max-autotune.

## Approach

+13/−1 lines in train.py: `MIXUP_ALPHA = 0.2` constant; per-step `lam ~ Beta(0.2,0.2)`, batch permutation, `torch.lerp` input mixing with explicit channels_last re-contiguation, two-term lam-weighted label-smoothed CE. All mixing outside the compiled module (no recompile; warmup untouched); eval path untouched. No deviations from plan.

## Execution

One run, no retries (task b52gvbaa9, GPU 0). All health checks passed: params 4,286,026 unchanged, dt 23ms (mixing ~free; 137 epochs vs 139 baseline), epoch-1 eval 33.87% (vs 35.11 baseline — no bug signature), train-loss trace elevated as expected from mixed targets. Run completed cleanly: 482.8s total, startup 11.1s (warm inductor cache), VRAM 1613 MB.

## Results

- **Primary metric**: best_test_acc = 96.25% (baseline: 96.71, delta: −0.46pp, −0.48%)
- **Observations**: The run tracked 3–5pp below EXP-006's trajectory through the whole mid-schedule (ep 40: 75.3 vs ~85; ep 120: 95.58) and the final anneal did NOT close the gap — the last five epochs are a noisy converged plateau at 96.13–96.25. final_test_loss 0.3042 vs baseline 0.1837 (softer predictions under mixup; the loss gap also reflects genuinely worse fit, not just calibration).
- **Analysis**: Clean negative result for the regularization axis. The pre-registered ambiguity (over-regularization vs too-short schedule) resolves toward over-regularization: the run converged (plateau, not climbing), so more epochs were not the binding factor — the recipe with four regularizers (LS + TA + RE + mixup) simply fits the data worse than with three. Combined with the EXP-003→004 trend (+0.83 → +0.17), the regularization dose-response curve at 4.29M params has now been mapped through its maximum: RE was the big win, TA was marginal, mixup is negative. Regularization is saturated at this capacity, exactly as capacity is closed at this time budget. What remains untried is qualitatively different territory: optimization/averaging tricks (EMA), inference-side gains within eval rules, throughput (max-autotune, GPU-resident data pipeline — airbench's central trick), and hyperparameter refinement of the existing recipe (peak LR, WD, batch) which has never been re-tuned since EXP-000 despite the recipe around it changing completely.
- **Key Learning**: The regularization axis is saturated at 4x/137ep — the dose-response curve (RE +0.83, TA +0.17, mixup −0.46) has crossed zero; remaining headroom lies in optimization quality, throughput, or re-tuning the never-revisited base hyperparameters (PEAK_LR/WD set in EXP-000 for a different recipe).

## Verification

- **Conditions**: condition 2 failed (best_test_acc 96.25 < 96.81 = baseline + 0.1pp); condition 1 passed (clean exit, 482.8s ≤ 600); condition 3 skipped per first-failure stop (informally compliant: 137 eval lines = 137 epochs)
- **Review Notes**: results confirmed trustworthy — metric consistent with the full eval trail; training_seconds exactly 300.0; params unchanged; mixing verified outside eval path (clean test data, eager base_model); no constraint violations
- **Verdict**: no-improvement
- **Verdict Basis**: condition failure (valid run, metric below baseline + 0.1pp)

## Unexplored Avenues

- **Mixup with label smoothing removed**: LS and mixup both soften targets; their composition may double-soften. Removing LS when adding mixup is a defensible variant — but the measured −0.46pp deficit is large for that interaction alone to explain, so expected value is low.
- **CutMix instead of mixup**: patch-based rather than blend-based; but it overlaps RE (occlusion) the way mixup overlaps LS — the saturation finding likely transfers.
- **Trading a regularizer**: swap RE out for mixup rather than stacking — tests substitution rather than dose. Low priority given RE's +0.83 is the largest banked regularization gain.

## Next Steps

1. **Re-tune PEAK_LR (and optionally WD) on the current recipe** — the base hyperparameters were set in EXP-000 for an unaugmented 1x net at 345 epochs and never revisited; the recipe now has 4x width, heavy augmentation, and 137 epochs, where a different peak LR (e.g. 0.6 or 0.3) could plausibly clear +0.1pp. Cheap, single-constant, high information. Confidence: medium.
2. **EMA weight averaging for eval** — last remaining optimization-quality lever; small expected gain but the bar is only +0.1pp and the final-epoch eval noise (±0.1) suggests averaging could harvest it. Confidence: low-medium.
3. **GPU-resident data pipeline / batch growth probe** — airbench's core insight (knowledge/README.md): host DataLoader is the residual bound (project-insights Medium); moving augmentation+data to GPU or raising batch to 1024 with LR rescale could buy epochs at fixed architecture. Larger diff, but the only remaining big-throughput idea. Confidence: low-medium.

## Exit Action Results
<!-- Leave empty if no exit actions defined. -->
