# Report EXP-041: Training-Only Direct-Path Auxiliary Cross-Entropy
- **Created**: 2026-07-27

## Goal

Maximize fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.48% baseline under the frozen 300-second counted-training contract. EXP041 tested whether a weak training-only objective on the raw pooled representation could improve the accepted pooled-head learner by at least 0.10 points without changing inference.

## Idea & Hypothesis

The candidate optimized `0.9 * CE(fc(z + 0.1*h(z))) + 0.1 * CE(fc(z))` throughout training, using the same accepted mixed targets early and hard targets late. The default inference path remained the accepted refined representation. The hypothesis required best accuracy at least 94.58% and at least 127 passes; final accuracy at least 94.45% and loss at most 0.2456 were corroboration.

## Approach

`WideResNet.forward()` gained an explicit default-false dual-logit mode. It always computed accepted main logits first and, only for training opt-in, called the shared classifier again on raw pooled features. Both inline loss branches used one input, one target set, and the exact convex 90/10 blend. Parameters, initialization/state, optimizer groups, coupled decay, data/RNG, schedule, augmentation transitions, evaluator, seed, and default inference remained accepted. The objective intentionally reduced pooled-head data gradient by 10% while leaving decay unchanged.

## Execution

One disposable verifier tolerance was corrected before timing or scoring: separate cloned FP32 CUDA convolution backward reductions differed by up to `1.2893e-4`, so the decomposition oracle received a measured reduction tolerance while exact source/logit and all-parameter update checks remained strict. Semantic qualification then proved default accepted inference, ordered main/direct classifier inputs, loss/gradient formulas, RNG preservation, and fresh/preseeded Nesterov updates. Main/direct gradients were strongly but not perfectly aligned, with sampled cosines `0.976-0.989`. Timing retained 0.984301 exposure, projected 128.258 passes, stayed below 0.66% CV, and peaked at 610.18 MiB.

The sole score completed without retry or runtime error. Mixup stopped at step 16,103 and 195.0 seconds; RandAugment stopped after the epoch-83 iterator exhausted at step 16,185. The run produced 26 unique every-fifth plus final evaluations.

## Results

- **Primary metric**: 94.26% (baseline: 94.48%, delta: -0.22 points, -0.23%)
- **Observations**: Final accuracy equaled best at 94.26%, and final loss worsened from accepted 0.2456 to 0.2529. The run delivered 25,105 steps, 128.53760 passes, 129 epochs, 1,096.4 MiB peak VRAM, and 1,003,482 parameters in 300.0 counted / 343.5 wall seconds.
- **Analysis**: The intended extra representation gradient was active and nonidentical at normal exposure, so compute loss or implementation contamination cannot explain the miss. Its strong gradient alignment suggests the auxiliary added limited new direction, while the exact treatment also weakened the successful pooled head's data gradient and required one classifier to serve raw and refined features. The score cannot isolate redundancy, head suppression, or shared-boundary conflict, but it rejects their complete 90/10 combination as a frontier improvement.
- **Key Learning**: Weak shared-classifier direct-path supervision is mostly gradient-aligned with the accepted objective yet worsens top-1 and CE; preserve sole refined-path CE.

## Verification

- **Conditions**: Completion/resource contract passed; primary metric improvement failed.
- **Review Notes**: Results are trustworthy. One H20, one fixed-seed score, `train.py`-only scope, exact default inference, independent source/loss/gradient/update/RNG checks, 128.538 passes, correct transitions, once-per-epoch cadence, and complete timing/summary evidence all passed.
- **Verdict**: no-improvement
- **Verdict Basis**: The valid 94.26% score missed baseline by 0.22 points and the required 94.58% threshold by 0.32 points, with worse loss.

## Unexplored Avenues

- Intermediate spatial supervision is formally distinct but would add adapters and spatial backward work without a diagnosed failure, so it remains low priority.
- A separate auxiliary classifier or detached direct path changes both inference-state ownership and gradient mechanism; result-conditioned versions are closed absent independent evidence.
- Other loss weights or temporal cutoffs remain mathematically possible but are immediate rescues of this miss and should not be tuned.

## Next Steps

- **High confidence**: Preserve the accepted sole refined-path CE, classifier, decay allocation, and pooled residual head.
- **Medium confidence**: Return to an orthogonal input or representation mechanism with a stronger generalization rationale and measured >=127-pass contract; avoid another local classifier/loss constraint.
- **Low confidence**: A one-time hard-boundary momentum reset remains a clean but very transient state intervention; use only if a fresh offline brainstorm finds no higher-upside mechanism.
