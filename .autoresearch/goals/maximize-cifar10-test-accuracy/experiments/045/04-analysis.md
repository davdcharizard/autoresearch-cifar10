# Report EXP-045: ResNet-D Projection Shortcuts
- **Created**: 2026-07-27

## Goal

Increase fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.48% frontier under the unchanged 300-second single-H20 budget. The experiment tested whether phase-selective stride-2 shortcut projections limited invariance and boundary quality.

## Idea & Hypothesis

At both spatial transitions, average each non-overlapping 2x2 preactivated cell before the unchanged pointwise shortcut map while retaining the main stride-2 convolutions. The hypothesis predicted at least 94.58% best accuracy and 127 realized passes if single-phase bypass sampling materially limited the accepted image-invariance learner.

## Approach

Changed only `PreActBlock`: stride-2 projections became stride 1 after a parameterless `AvgPool2d(2,2)`. Parameter construction order, all parameter and buffer bytes, layer1 direct projection, four raw identity shortcuts, main branches, data path, loss, optimizer, schedule, pooled head, and evaluator remained exact. An ignored evaluator-blocked harness proved topology, state/RNG identity, independent four-phase forward/backward equations, unchanged controls and Nesterov updates, then qualified exposure with 16 interleaved one-arm-at-a-time windows.

## Execution

Two semantic attempts stopped on preflight-only defects: candidate-only attribute access on the accepted control, then a shape probe that updated candidate BatchNorm buffers. Both were repaired without changing production code or launching a score; the third semantic attempt passed. Timing passed with median retention 0.981565 and projected 127.902 passes. The sole scored run completed normally in 340.5 wall seconds with correct 65% transitions, 26 every-fifth evaluations, one H20, and no runtime error.

## Results

- **Primary metric**: 94.11% (baseline: 94.48%, delta: -0.37 points, -0.39%)
- **Observations**: The local treatment was exact and affordable: all four shortcut phases received equal forward and backward weight, fixed-probe shortcut RMS ratios stayed near one (`1.0008` at both transitions), and realized exposure was 25,215 steps or 129.1008 passes. Nevertheless final accuracy/loss worsened from accepted 94.45%/0.2456 to 94.06%/0.2512.
- **Analysis**: This was neither a null treatment nor a compute failure. Averaging removed shortcut phase selection while preserving main branches and almost all accepted exposure, yet both best and endpoint quality declined. The evidence indicates the learned stride-2 main path and accepted augmentation already handle the relevant invariance, while shortcut phase selection preserves useful fine spatial evidence or a favorable residual decomposition. Because the full exact treatment failed at normal exposure, one-transition, kernel, order, gain, and main-branch variants are immediate post-result rescues without an independent diagnosis and are closed.
- **Key Learning**: Averaging all four downsampling-shortcut phases slightly harms boundary quality; the accepted single-phase projections are not the current invariance bottleneck.

## Verification

- **Conditions**: Completion/resource contract passed; primary improvement and hypothesis accuracy failed.
- **Review Notes**: Results confirmed trustworthy: only `train.py` changed, frozen files and common state were exact, analytic oracles and controls passed, timing/exposure were normal, and the valid score was unique.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid 94.11% score failed baseline 94.48% and threshold 94.58% at 129.1008 passes.

## Unexplored Avenues

- A learned anti-aliasing mechanism could preserve discriminative detail better than a fixed box average, but it changes capacity and needs a separate causal rationale.
- Translation-equivariant padding or phase-ensemble objectives target related behavior outside the shortcut itself, but evaluator-frozen evidence does not currently diagnose them.
- One-transition pooling, blur kernels, pool/projection order, shortcut gain, or matching main-branch filtering are immediate variants explicitly closed by the preregistered normal-exposure miss.

## Next Steps

- **High confidence**: Preserve accepted projection shortcuts and seek a distinct nearly free generalization mechanism outside closed readout, gradient, and anti-aliasing treatments.
- **Medium confidence**: Revisit early target regularization only with a mechanism that is demonstrably nonredundant with batch-shared mixup.
- **Low confidence**: Explore a training-only diagnostic of class boundary or calibration errors before choosing another objective or classifier intervention.
