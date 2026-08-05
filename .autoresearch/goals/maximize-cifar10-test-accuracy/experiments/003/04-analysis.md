# Report EXP-003: Early CutMix With a Hard-Label Tail
- **Created**: 2026-07-24

## Goal

Increase CIFAR-10 `best_test_acc` above the 94.07% moving baseline by at least 0.10 percentage points under the fixed 300-second training budget, testing whether localized mixed-sample regularization outperforms the accepted early mixup schedule.

## Idea & Hypothesis

Replace alpha-0.2 mixup during the first 65% of counted training with shared-rectangle `Beta(1,1)` CutMix, recompute target mass from exact pasted area, and retain the validated final 35% hard-label tail. The hypothesis was that crisp local composition would improve spatial invariance while preserving one-forward-pass throughput, reaching at least 94.17%.

## Approach

Only `train.py` changed. A deterministic CPU generator sampled the uniform retained-area coefficient and rectangle center without adding a CUDA scalar synchronization; the permutation remained device-local. One clipped rectangle was pasted into a cloned destination batch, its exact area determined the two-label loss, and the existing hard-label path resumed at 65%. The WRN-16-2 architecture, optimizer, LR schedule, seed, loader, crop/flip transforms, evaluation cadence, and final output schema remained unchanged.

## Execution

One full run completed without retries. Pixel, target, alias-safety, zero-area, finite-backward, parameter-count, lint, and scope checks passed. Alternating synthetic timing projected 121.8 CutMix passes versus 121.6 for matched mixup, a 1.0017 ratio. The full run switched once at 195.0 seconds and step 17,886 with mean pasted fraction 0.3099, then completed 27,831 steps in 340.1 seconds total.

## Results

- **Primary metric**: 93.72% (baseline: 94.07%, delta: -0.35 percentage points, -0.37%)
- **Observations**: The run realized 142.5 data passes, slightly more than EXP-002's 141.9, so throughput did not explain the regression. Accuracy was 88.10% at the last pre-switch evaluation, recovered through the hard-label tail, peaked at 93.72% at epoch 140, and ended at 93.70%; the 0.02-point best/final gap shows a stable lower endpoint rather than a missed transient peak.
- **Analysis**: The hypothesis is not supported for this formulation. CutMix executed as intended and preserved compute exposure, but uniform shared rectangles pasted 31% of pixels on average and area-weighted targets were less effective than whole-image alpha-0.2 interpolation on this small WRN/CIFAR setting. The hard-label tail recovered optimization but could not close the representation/generalization deficit. This discredits the tested strength and shared-rectangle formulation, not every possible CutMix mixture or patch distribution.
- **Key Learning**: Shared-rectangle CutMix preserves exposure but trails early mixup, suggesting area-weighted spatial labels are less effective on this WRN.

## Verification

- **Conditions**: metric improvement condition failed
- **Review Notes**: Results confirmed trustworthy. The process exited 0 on one H20, used exactly 300.0 counted seconds, stayed below 600 seconds total, evaluated at most once per epoch, modified only `train.py`, and retained normal exposure.
- **Verdict**: no-improvement
- **Verdict Basis**: The run was valid, but 93.72% was 0.35 points below the 94.07% baseline and 0.45 points below the required 94.17% threshold.

## Unexplored Avenues

- Mix CutMix with the validated mixup path at a fixed low probability so spatial composition adds diversity without discarding the proven regularizer.
- Use smaller donor patches or per-example rectangles; either could reduce semantic label mismatch or increase spatial diversity relative to the shared 31%-area formulation.

## Next Steps

- **High confidence**: return to the accepted early-mixup baseline and tune one variable such as alpha or the 65% cutoff.
- **Medium confidence**: test a short-horizon late EMA on the accepted mixup model, recognizing its likely small headroom.
- **Medium confidence**: revisit WRN-16-3 only with validated mixup and a stricter exposure gate so added capacity is not confounded by severe update loss.

## Exit Action Results

No exit actions were defined for this local-only run.
