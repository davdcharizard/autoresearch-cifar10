# Report EXP-016: Fixed-MAC Stage-Depth Redistribution
- **Created**: 2026-07-26

## Goal

Raise fixed-seed CIFAR-10 `best_test_acc` above the accepted 94.07% WRN-16-2 baseline within 300 counted training seconds. Success required at least 94.17% with all local H20, scope, timing, seed, and evaluation constraints intact.

## Idea & Hypothesis

Dense 8x8 capacity was the only repeated positive neighborhood: stage-3 width and depth reached 94.11% and 94.15% despite reduced exposure. EXP-016 tested whether the accepted six-block budget was misallocated by moving one equal-MAC identity block from stage 1 to stage 3, yielding `[1,2,3]`. The hypothesis was that retaining the dense late transform while recovering EXP-011's exposure would cross 94.17%.

## Approach

The production constructor built and fully initialized the accepted `[2,2,2]` model first. It then removed `layer1[1]` and appended one `128->128` block as `layer3[2]`. The new block used the exact accepted initializer inside a restored CPU RNG fork with preregistered seed 16016; every surviving parameter/buffer and the post-construction CPU/CUDA RNG state matched an accepted reference bitwise. Final topology had six blocks, 968,538 parameters, and exactly the accepted 101,106,944 convolution/linear MACs per image. All training settings remained accepted.

## Execution

Semantic preflight passed topology, state, RNG, seed-oracle, MAC, gradient, and optimizer checks without correction. Matched timing showed a substantial unexpected advantage: weighted step time fell from 10.671999 to 8.853656 ms, retention 1.205378, with all timing CVs below 0.0053. The sole scored run then completed cleanly on one H20. Mixup disabled once at step 21,454 and 195.0 seconds; 35 evaluation epochs were unique; no error signature appeared.

## Results

- **Primary metric**: 93.82% (baseline: 94.07%, delta: -0.25 points, -0.27%)
- **Observations**: The candidate completed 33,535 steps, or 171.6992 dataset-equivalent passes, versus about 141.9 accepted. It used only 968.2 MiB peak VRAM and finished in 346.2 wall seconds. Best and final accuracy were both 93.82%; final loss was 0.2778 versus 0.2432 accepted.
- **Analysis**: The operational premise succeeded: moving the block late preserved static MACs and increased exposure by about 21%. The accuracy premise failed decisively. More late transformation and nearly 30 extra passes could not recover the representation/generalization supplied by the removed 32x32 block. The worse final loss, despite near-zero training loss and extra optimization, indicates a quality loss rather than undertraining. EXP-011's +0.08 signal therefore depends on adding late capacity while retaining early depth; it cannot be converted into a free fixed-MAC exchange.
- **Key Learning**: Keep both 32x32 stage-1 blocks; equal-MAC late reallocation accelerates training but loses essential high-resolution representation quality.

## Verification

- **Conditions**: Completion and integrity passed; primary metric condition failed because 93.82% is below 94.17%.
- **Review Notes**: Results are trustworthy. Only `train.py` changed, common state/RNG was controlled, the fixed seed was preregistered, the run completed once, and log/topology counts match the preflight.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid complete run with no hard-constraint violation, but accuracy fell 0.25 points below baseline and 0.35 below threshold.

## Unexplored Avenues

- **`[2,1,3]` redistribution**: preserves both high-resolution blocks while trading middle-stage refinement. It is a distinct allocation and may retain local features, but the hierarchy risk and EXP-016 result weaken the prior.
- **Late feature selection without removing depth**: neutral stage-3 attention remains untested and targets the positive late-capacity signal at much lower structural disruption.
- **Kernel-aware architecture search**: the 20% latency difference at equal MACs suggests room for throughput-aware topology design, but accuracy constraints must explicitly preserve early representation capacity.

## Next Steps

- **Medium confidence - neutral stage-3 feature selection**: test the already developed identity-initialized SE proposal while retaining `[2,2,2]` and accepted exposure.
- **Medium confidence - mild data invariance**: consider RandAugment only with a design that does not undermine the validated hard-label tail.
- **Low confidence - measured update conditioning**: diagnose gradient structure before applying a targeted conditioning rule rather than projecting arbitrary directions.
