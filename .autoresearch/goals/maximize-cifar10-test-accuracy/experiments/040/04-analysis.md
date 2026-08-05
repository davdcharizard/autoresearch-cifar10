# Report EXP-040: Equalize Effective Classifier Row Norms
- **Created**: 2026-07-27

## Goal

Maximize fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.48% baseline under the frozen 300-second counted-training contract. EXP040 tested whether removing class-specific classifier radii, without changing total classifier scale, could improve the accepted pooled-head learner by at least 0.10 points.

## Idea & Hypothesis

The accepted classifier starts with a 6.96% row-norm CV and a 1.2725 max/min ratio. The candidate replaced each effective row by its direction times the differentiable RMS raw-row norm, preserving instantaneous classifier Frobenius norm while forcing all ten effective rows to a common radius. The hypothesis required best accuracy at least 94.58% and at least 127 passes; final accuracy at least 94.45% and loss at most 0.2456 were corroboration.

## Approach

Only the final affine operation in `WideResNet.forward()` changed. Raw classifier weight and bias parameters, initialization bytes, state keys, optimizer membership, coupled `5e-4` decay, pooled residual MLP, model body, data/RNG, augmentation transitions, schedule, evaluator, seed, and budget remained accepted. The transformation preserved raw row directions and raw/effective Frobenius norm at each forward, but deliberately changed gradient conditioning and coupled all radial motion through one RMS scale.

## Execution

Two disposable verifier defects were corrected before timing or scoring: a tensor-valued RMS was changed from an invalid `full_like` fill value to an expanded comparison tensor, and an independently ordered FP32 full-logit check received a measured numerical tolerance. Production code did not change. Semantic qualification then proved the intended geometry, analytic gradients, transformed forward, RNG preservation, and fresh/preseeded Nesterov updates. Timing passed narrowly with 0.978220 retention, 127.466 projected passes, maximum 0.70% CV, and 610.16 MiB candidate peak memory.

The sole score completed without retry or runtime error. Mixup stopped at step 15,947 and 195.0 seconds; RandAugment stopped after the epoch-82 iterator exhausted at step 15,990. The run produced 26 unique every-fifth plus final evaluations.

## Results

- **Primary metric**: 93.91% (baseline: 94.48%, delta: -0.57 points, -0.60%)
- **Observations**: Final accuracy was 93.87%, and final loss worsened from accepted 0.2456 to 0.2622. The run delivered 24,910 steps, 127.53920 passes, 128 epochs, 1,096.4 MiB peak VRAM, and 1,003,482 parameters in 300.0 counted / 344.7 wall seconds.
- **Analysis**: The exact intended geometry was active at normal exposure, so compute loss or implementation contamination cannot explain the regression. Equal class-vector radii removed useful boundary freedom or introduced harmful conditioning/shared-scale coupling; this experiment cannot distinguish those mechanisms. It rejects only the exact differentiable RMS equal-row map and algebraic equivalents, not learned gain, feature normalization, angular penalties, or other normalized-classifier parameterizations.
- **Key Learning**: Preserve the accepted affine classifier's class-specific row radii; exact Frobenius-preserving equalization worsens both top-1 and CE at normal exposure.

## Verification

- **Conditions**: Completion/resource contract passed; primary metric improvement failed.
- **Review Notes**: Results are trustworthy. One H20, a single fixed-seed score, `train.py`-only scope, independent geometry/gradient/update/RNG checks, 127.539 passes, correct transitions, once-per-epoch cadence, and complete timing/summary evidence all passed.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid 93.91% score missed baseline by 0.57 points and the required 94.58% threshold by 0.67 points, with worse loss.

## Unexplored Avenues

- A learned common classifier gain remains formally distinct, but it adds optimizer and scale choices without evidence that common-radius geometry is beneficial.
- Feature normalization and centered-simplex regularization remain untested, but both impose broader angular geometry and need an independently justified scale.
- A one-time momentum reset at the hard-label boundary remains a clean state intervention, though its direct effect decays below 1% within about 44 updates.

## Next Steps

- **High confidence**: Preserve the ordinary affine classifier with accepted bias and class-specific row radii.
- **Medium confidence**: Seek a low-cost mechanism outside local classifier-radius, classifier-decay, and tail-LR tuning, preferably one with a diagnosed interaction with the successful pooled head.
- **Low confidence**: Use the one-time momentum reset only if a broader offline review finds no stronger orthogonal candidate; do not combine it with failed geometry as a rescue.
