# Report EXP-037: Exclude Terminal Classifier Weight From Decay
- **Created**: 2026-07-27

## Goal

Raise fixed-seed CIFAR-10 best accuracy from 94.48% to at least 94.58% under
the accepted 300-second, one-H20, frozen-evaluator contract.

## Idea & Hypothesis

Move only the 1,280-element `fc.weight` tensor from continuous `5e-4` decay to
the existing zero-decay group, preserving every convolution and both successful
pooled-head matrices. The low-prior hypothesis was that terminal class-vector
shrinkage constrained useful boundary fitting on the nonlinear pooled features.

## Approach

Only the two optimizer comprehensions changed. An independent accepted-source
preflight proved exact 999,856/3,626 element groups, model/RNG/source identity,
and first/preseeded Nesterov updates against per-arm coupled-decay oracles.
Only `fc.weight` differed directly, by about `1.45e-5` in each fixture.

## Execution

Semantic and timing gates passed without retry. Timing retention was 1.001687,
projecting 130.52384 passes with maximum CV 0.31%. The sole score exited 0,
completed 131.72736 passes, and produced correct transitions and 27 unique
evaluations without numerical, CUDA, worker, or evaluator errors.

## Results

- **Primary metric**: 94.41% (baseline 94.48%, delta -0.07 points, -0.07%)
- **Observations**: Final accuracy fell from 94.45% to 94.38%, while final loss worsened from 0.2456 to 0.2786. Best-final gap remained 0.03. The candidate completed 25,728 steps in 300.0 counted / 345.9 wall seconds with accepted-like VRAM.
- **Analysis**: The allocation achieved its exact local effect with slightly more exposure, so throughput and incomplete optimization do not explain the miss. Worse loss is consistent with classifier shrinkage contributing useful confidence/boundary regularization after the pooled head. The small top-1 regression alone might be noise-scale, but the 0.033 loss deterioration provides directional corroboration against under-decay.
- **Key Learning**: Continuous decay on the terminal classifier supports the pooled-head frontier; removing it preserves exposure but worsens loss and slightly lowers accuracy.

## Verification

- **Conditions**: Integrity passed; 94.58% primary threshold and both endpoint corroboration checks failed.
- **Review Notes**: Trustworthy exact-scope, normal-exposure fixed-seed result.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid 94.41% score was below baseline and threshold with worse endpoint loss.

## Unexplored Avenues

Increased classifier decay is an opposite-direction mechanism and is not closed,
but the accepted loss already improved strongly and there is no exact value
rationale. Intermediate/later under-decay, LR compensation, pooled-head decay
changes, and seed/head rescues are closed by the preregistered normal-exposure miss.

## Next Steps

- **Low confidence**: Derive a training-only angular-classifier scale before reconsidering normalized logits.
- **Low confidence**: Explore a new post-spatial representation mechanism that is not an adjacent head width/scale modification.
- **Low confidence**: Keep classical momentum rejected absent evidence of endpoint instability.
