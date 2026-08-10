# Brainstorm EXP-025
**Created**: 2026-08-06

## Web Search & Literature Review

- **ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks** (`knowledge/papers/eca-net.md`; CVPR 2020)
  Tiny channel-axis interactions after global descriptors can improve ResNets without an SE bottleneck. The transfer is indirect, but zero-start `2*sigmoid` gates can preserve the exact accepted initial function and shortcut geometry that EXP-024 disturbed.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`; BMVC 2016)
  Width remains the strongest local capacity prior, but EXP-023/024 now disfavor both deleting depth and abruptly widening the final Option-A stage. Conditional reuse of accepted width is a distinct response.
- **CutMix** (`knowledge/papers/cutmix.md`)
  Area-proportional regional targets remain the accepted regularizer. Any global descriptor mechanism must prove bounded gate recruitment on exact soft-target batches rather than assume ImageNet attention transfers safely.

## Experimental History Review

- The frontier remains EXP-010 at 94.15% with width-2 ResNet-20, standard momentum/decay, N1/M7 plus probability-0.5 alpha-1 CutMix through 80%, and a hard weak tail. Its 89.73% switch, 93.16% first-weak checkpoint, 0.1934 final NLL, and 26,898 updates remain the reference trajectory.
- EXP-023 preserved numerical health but lost 0.15 point after trading three blocks for global width 3. EXP-024 preserved depth but the 64-to-160 final stage produced a candidate-only 98.44% class transient on its first CutMix step. Static width allocation is no longer the best immediate capacity test.
- The exact EXP-024 review already developed and compared ECA, bounded GeM, and final-stage widening. Widening won that review but is now vetoed; ECA is the pre-reviewed fallback and uniquely preserves the accepted step-zero function, block count, widths, and Option-A ratios.
- Backward remains 75.46% of measured step cost. Three final-stage attention paths can still be launch-bound: EXP-012's nine-block SE probe cost 1.23324x. A strict paired timing gate is therefore load-bearing, not ceremonial.
- Repeated lower-loss class transients across optimizer, shortcut, precision, and now architecture changes make immutable production-batch recruitment evidence mandatory. Exact initialization alone is insufficient after EXP-014; first-update gate scale must be bounded explicitly.

## Collected Ideas

## Combinations

## Candidate Ideas

### Channels-Last Accepted Model
**Summary**: Convert the unchanged width-2 ResNet-20 parameters and every GPU input batch to channels-last physical storage, preserving mathematical operations, optimizer, RNG, data, schedule, and evaluator. Production would use the same accepted graph and parameter count; only memory format changes.

**What it targets**: The 75.46% convolution-backward systems bottleneck. If H20 kernels accelerate, extra in-budget updates could improve the accepted recipe without changing representation or optimizer geometry.

**Reasoning**: H20 wider kernels scaled sublinearly in EXP-023, so device layout deserves measurement rather than inference. This idea is orthogonal to the transient-prone model changes. However, the accepted model already processes 26,898 updates, and no local experiment proves more exposure alone raises top-1; tiny FP32 NCHW kernels may already be optimal.

**Sources**: `02-system-understanding.md`; EXP-013 and EXP-023; `project-notes/project-insights.md`.

**Estimated Effort**: medium — simple production edits, exact-layout/identity checks and paired real-batch timing.

**Risk Assessment**: Input/layout conversions may erase kernel gains, evaluator inputs need consistent conversion without harness changes, and a speedup has only an indirect accuracy mechanism. Timing failure would yield no scored evidence.

### Identity-Initialized Final-Stage ECA
**Summary**: Add ECA gates to all three `layer3` residual branches. Each gate averages the 128-channel 8x8 residual, applies a bias-free length-5 `Conv1d`, and multiplies by `2*sigmoid(logits)` immediately before the unchanged shortcut addition. Zero kernels give exact unit gates, adding only 15 parameters while preserving every shared weight, output, width, block, and Option-A ratio at step zero. The detailed reviewed design is in `experiments/024/proposals/idea-01.md`.

**What it targets**: Conditional allocation of accepted semantic channels without another static width/depth or transition change. It aims to improve weak-tail generalization while maintaining strong fit and at least 96.66% exposure.

**Reasoning**: EXP-007 proved capacity matters; EXP-023/024 show static reallocations introduce depth or transition costs. ECA-Net supplies a lightweight conditional alternative, and the exact-function start directly addresses EXP-024's architecture discontinuity. Three sequential attention paths may still fail timing, while global CutMix descriptors and arbitrary channel adjacency are semantic risks.

**Sources**: `knowledge/papers/eca-net.md`; EXP-007, EXP-010, EXP-012, EXP-014, EXP-023, EXP-024; `experiments/024/proposals/idea-01.md`.

**Estimated Effort**: high — compact tracked change, substantial identity/recruitment/timing verification.

**Risk Assessment**: Zero-start gates can move sharply at LR 0.1; require bounded first-update weights/gates and no candidate-only concentration. Three pool/Conv1d/sigmoid/multiply chains may exceed the 3.5% timing allowance. CutMix descriptors may conflate pasted regions, and channel locality may be meaningless.

### Batch-96 Accepted Recipe
**Summary**: Reduce `BATCH_SIZE` from 128 to 96 while leaving LR 0.1 and the accepted data/optimizer/schedule unchanged. The test seeks more optimizer updates and higher gradient noise under the same 300-second budget, with evaluation cadence constrained against observation-count bias.

**What it targets**: Short-horizon generalization and update exposure rather than static capacity. A smaller batch may add useful noise and updates without changing the model or momentum path.

**Reasoning**: Batch 256 failed the image-throughput gate, but the smaller direction remains untested. The H20 is lightly loaded in memory; fewer examples per step can be faster, and more updates may benefit the hard weak tail. Yet image exposure likely falls, LR 0.1 may be too high for 96, and altered epoch length changes the number of evaluation opportunities unless elapsed checkpoints are capped.

**Sources**: `02-system-understanding.md`; EXP-010 and EXP-013; `knowledge/papers/large-minibatch-sgd.md` (directional scaling caveat).

**Estimated Effort**: medium — one constant, paired image/update throughput and evaluation-count controls.

**Risk Assessment**: The candidate may see fewer total images, suffer over-noisy CutMix gradients, or gain max-over-checkpoint opportunities. A fair plan must pre-register equal elapsed evaluation count and refuse LR tuning or batch rescue after timing.

## Review

The independent reviewer selected identity-initialized final-stage ECA (evidence/reasoning 8/10, potential impact 6/10) over channels-last (4/10, 3/10) and batch 96 (3/10, 3/10). It judged ECA the only candidate aimed directly at feature allocation/generalization rather than the still-unproven exposure-to-accuracy link. I adopted its request to report gate recruitment and distributions separately on hard and CutMix batches, which will make a null result interpretable. I also retain the tight timing gate and explicit channel-locality/CutMix caveats.

I do not adopt the suggestion to make switch/NLL signatures extra acceptance requirements: the user-approved goal defines success by the three existing necessary conditions, and brainstorming cannot invent a new quality gate. A bare 94.25% pass remains formally valid but will be described as weak causal evidence unless the registered trajectory supports the mechanism. Full critique: `01-idea-review.md`.

## Idea Evaluation

Adopt **Identity-Initialized Final-Stage ECA**. Channels-last and batch 96 may change work performed, but neither has local evidence that exposure is the current accuracy limiter; batch 96 also risks fewer images and biased evaluation count. ECA is a bounded, pre-reviewed conditional-capacity hypothesis that preserves the exact accepted graph function and every static width/depth/shortcut ratio at initialization.

## Chosen Idea
**Selected**: Identity-Initialized Final-Stage ECA

**Why this idea**:
EXP-007/010 show that the accepted representation benefits from capacity, while EXP-023/024 show static capacity reallocations can lose depth or disturb transition geometry. Three zero-start unit ECA gates add input-conditional channel allocation only at 8x8 without changing the step-zero function or accepted shortcut ratios. The design already has an exact implementation and strict recruitment/timing protocol in `experiments/024/proposals/idea-01.md`; its main risks—launch overhead, CutMix descriptor ambiguity, and arbitrary channel adjacency—are explicit and measurable.

**Hypothesis**:
Three length-5 identity-scale ECA gates on `layer3` will remain numerically safe, retain at least 26,000 projected updates, preserve the accepted strong-phase trajectory, and raise `best_test_acc` from 94.15% to at least 94.25% by improving conditional semantic-channel allocation. Gate statistics split by hard versus CutMix batches and late NLL will distinguish meaningful recruitment from a marginal checkpoint fluctuation.
