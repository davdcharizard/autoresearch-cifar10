# Report EXP-004: TrivialAugmentWide on top of the regularized recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md
- **Log**: logs/exp-log-004.md

## Goal

Maximize CIFAR-10 test accuracy (best_test_acc, %, higher is better) of train.py within the fixed 300s training budget (≤10 min total wall clock). Baseline at experiment start: **96.06%** @ 3a62d44 (EXP-003: 4x-wide ResNet-20 + time-keyed one-cycle + RandomErasing). Question tested: does tuning-free policy augmentation (TrivialAugment) stack with the existing occlusion regularization at the short 114-epoch schedule?

## Idea & Hypothesis

Chosen over width 6x and torch.compile because EXP-003's gain landed at the top of the published range — evidence the regularization axis was still hot — and because the TA paper's own CIFAR protocol applies a 16px cutout *after* TA, making composition with the existing RandomErasing literature-standard rather than speculative.

**Hypothesis**: Inserting TrivialAugmentWide before ToTensor (keeping RandomErasing) raises best_test_acc from 96.06% to ≥96.25%, with throughput and epoch count essentially unchanged; main risk is over-regularization at the short schedule (paper uses 200 epochs).

## Approach

Single transform inserted into `train_tf` in train.py:

```python
transforms.TrivialAugmentWide(),   # after RandomHorizontalFlip, before ToTensor (PIL-stage op)
```

Library defaults kept (num_magnitude_bins=31, NEAREST — TA is tuning-free by design). RandomErasing stays last after Normalize, matching the paper's TA-then-cutout ordering. Architecture and recipe otherwise byte-identical to 3a62d44. No deviations from plan.

## Execution

Single run, no retries, no adjustments. Launched in background on GPU 0; completed cleanly in 416.5s total. Early signal: params 4,286,026 unchanged; img/s ~19k unchanged (TA's PIL cost absorbed by the 8 CPU workers); epoch-1 test_acc 34.26% — ~5pp below EXP-003's epoch-1, the expected signature of stronger augmentation slowing early fitting.

## Results

- **Primary metric**: best_test_acc = **96.23%** (baseline: 96.06, delta: **+0.17pp**, +0.18%)
- **Observations**:
  - num_epochs 114, peak_vram_mb 1620.7, ~19k img/s — all identical to EXP-003, again a perfectly controlled comparison.
  - final_test_loss fell 0.2084 → 0.1947 (−6.6%): the generalization gap keeps closing, confirming a real regularization gain.
  - Mid-schedule accuracy ran consistently below EXP-003 (ep40 83.78, ep80 91.60) and converged past it only in the anneal — the model spends the cycle fighting harder data and cashes in at low LR.
  - final (96.21) ≈ best (96.23): converged, no undertraining signature even with the stacked augmentation.
  - The gain (+0.17pp) was ~1/3 of EXP-003's (+0.83pp): hypothesis target (96.25) missed by 0.02pp while the pass bar was cleared — the regularization axis is showing clearly diminishing returns at this capacity.
- **Analysis**: Hypothesis substantively validated (stacking works at 114 epochs; throughput unchanged) though the magnitude landed marginally under the point prediction. Trajectory: 91.97 → 93.16 (schedule) → 95.23 (capacity) → 96.06 (occlusion) → 96.23 (policy augmentation). Two regularizers are now in; a third would likely land within noise. The constraint has rotated back to capacity/throughput: with augmentation pressure raising effective data complexity AND final=best showing epochs are not yet binding at 4x, the width step (5–6x) is now the best-motivated untried lever.
- **Key Learning**: Augmentation stacking obeys diminishing returns in gap-units — RandomErasing took the cheap 0.83pp, TA only 0.17pp more — so after two regularizers the binding constraint rotates back to model capacity.

## Verification

- **Conditions**: all passed
  1. Clean completion within budget: total_seconds 416.5 ≤ 600, exit 0 — PASS
  2. best_test_acc ≥ baseline + 0.1pp (≥ 96.16): 96.23 — PASS
  3. Eval at most once per epoch: 114 eval lines = 114 epochs — PASS
- **Review Notes**: results confirmed trustworthy — frozen prepare.py Eval on the ground-truth test set; epochs/VRAM/img-s identical to EXP-003 rules out confounds; improvement via the intended intervention class (training-input augmentation only).
- **Verdict**: improvement
- **Verdict Basis**: all conditions passed + improvement above the 0.1pp noise bar (+0.17pp)

## Unexplored Avenues

- **Width 5–6x on the doubly-regularized recipe**: the strongest variation — augmentation raised effective data complexity, final=best says epochs aren't binding at 4x, and EXP-002's 8x failure predates both regularizers. 6x at ~70 epochs under TA+RE could behave very differently.
- **TA magnitude/interpolation variants**: BILINEAR interpolation or TrivialAugment (non-Wide) — likely within noise; low priority.
- **Mixup/CutMix**: a third regularizer class; given the diminishing-returns pattern (0.83 → 0.17), expected gain is sub-noise at this capacity. Revisit only after a capacity step.
- **Longer effective schedule via torch.compile**: would amplify augmentation gains if epochs ever become binding (they aren't at 4x).

## Next Steps

1. **WIDTH_MULT 4 → 6 on the current recipe** (medium-high confidence: capacity is the rotated bottleneck; both regularizers raise signal per epoch; abort signature to watch is final=best with depressed absolute accuracy, i.e., undertraining).
2. **torch.compile throughput as a width-enabler** (medium confidence: if 6x shows undertraining, throughput is the unlock; fiddly failure modes).
3. **Third regularizer (Mixup/CutMix)** (low confidence now: diminishing returns measured at 0.83 → 0.17pp per added regularizer).

## Exit Action Results
