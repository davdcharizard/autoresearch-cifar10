# Report EXP-036: Scaled Pooled-Feature Residual MLP Head
- **Created**: 2026-07-27

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` from 94.32% to at least 94.42%
within 300 counted training seconds, one H20, and the frozen evaluator contract.
The experiment tested whether a cheap nonlinear remapping of accepted pooled
features could improve generalization without adding spatial convolution cost.

## Idea & Hypothesis

Add a bias-free `128 -> 64 -> 128` ReLU MLP after global average pooling,
combine it with the direct pooled vector at fixed scale 0.1, and preserve the
accepted classifier and complete spatial learner. EXP027's capacity-plus-early-
invariance interaction supplied indirect local support, while post-pooling
placement made the treatment distinct from failed spatial bottlenecks. The
hypothesis required at least 130 passes and 94.42% best accuracy; final accuracy
at least 94.32% and loss at most 0.2523 were corroboration.

## Approach

The branch adds exactly 16,384 parameters/MACs per image and raises total
parameters to 1,003,482. It is constructed only after the accepted model and
classifier are fully initialized. A CPU-only default-generator seed 36036 is
used inside a restoring CPU RNG fork, preserving all 987,098 common tensor
bytes and both global CPU/CUDA RNG states. Both new matrices use Kaiming-normal
initialization and accepted matrix decay, LR, momentum, and Nesterov settings.

An ignored evaluator-free oracle verified source scope, exact common state,
branch reconstruction, forward formula, direct path, optimizer membership,
RNG neutrality, temporal controls, and finite early/hard updates. Initial
residual/direct feature norm ratio was 0.120864 and logit RMS perturbation was
0.069719. Balanced complete-step timing gated the sole score at 130 passes.

## Execution

The first semantic attempt exposed a code bug before timing: the accepted
linear initializer assumes a bias tensor, while the new linears are bias-free.
The exact Kaiming-normal matrix operation was applied directly to both weights;
topology, seed, and treatment were unchanged. Semantic qualification then
passed. Timing retained 0.982799 of accepted exposure, projecting 130.71955
passes with all CVs below 0.44%.

The sole score completed without runtime or integrity errors. Mixup disabled
at step 16,336/195.0 seconds and RandAugment at the epoch-84 boundary 44 steps
later. The run completed 25,450 steps, 131 epochs, 130.304 passes, and 27 unique
evaluations in 343.9 wall seconds.

## Results

- **Primary metric**: 94.48% (baseline: 94.32%, delta: +0.16 points, +0.17%)
- **Observations**: Final accuracy was 94.45%, +0.23 points over the accepted 94.22 endpoint, and final loss improved from 0.2523 to 0.2456. Best occurred at epoch 130 and the final epoch-131 score remained within 0.03 points. The head surrendered 2.70336 passes versus accepted but stayed above its preregistered normal-exposure floor; VRAM remained effectively unchanged at 1,096.4 MiB.
- **Analysis**: The treatment achieved its intended local effect: a small active nonlinear remapping after pooling improved both top-1 and test loss despite modestly lower exposure. Stable best/final behavior and better loss make a sparse-evaluation accident less plausible. This supports pooled channel-co-occurrence remapping as an efficient use of capacity under the accepted early-invariance recipe. It does not establish that the prior affine classifier was a formal bottleneck, because the backbone trains end to end, nor does one fixed scale/init establish a tunable neighborhood.
- **Key Learning**: A 16k-parameter residual MLP after global pooling improves both accuracy and loss, adding useful nonlinear capacity without spatial-compute cost.

## Verification

- **Conditions**: All passed - valid completion, correct hard constraints, `best_test_acc=94.48% >=94.42%`, and both endpoint corroboration checks.
- **Review Notes**: Results are trustworthy. Only `train.py` changed, common initialization/RNG and frozen evaluation were protected, one H20 and one fixed seed were used, counted/wall limits and once-per-epoch cadence held, and realized exposure was 130.304 passes.
- **Verdict**: improvement
- **Verdict Basis**: The primary metric improved by 0.16 points, exceeding the required margin, while final accuracy and loss independently improved.

## Unexplored Avenues

- Different pooled-head widths, scales, activations, biases, zero initialization, or seeds remain technically possible but are not justified by this one operating point; immediate tuning would be an adjacent post-success sweep.
- A separately motivated normalized classifier or decay allocation could compose with the head, but should first establish an independent mechanism and preserve the new 130-pass frontier.
- The learned head's representation behavior could be diagnosed with training-only feature statistics in a future design, provided diagnostics are preregistered and do not use frozen test data for tuning.

## Next Steps

- **High confidence**: Preserve the exact pooled residual head as the new base and search only orthogonal mechanisms with a measured >=127-pass operating contract.
- **Low confidence**: Reconsider classifier-specific decay only as an isolated allocation test on the new base; its directional evidence remains weak.
- **Low confidence**: Explore normalized logits only after deriving a scale from training-only invariants prospectively, avoiding a result-conditioned sweep.
