# Brainstorm EXP-005
**Created**: 2026-08-05

## Web Search & Literature Review

- **RandAugment** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/randaugment.md`): the method deliberately reduces augmentation tuning to operation count and shared magnitude; EXP-004 validated one point in that space locally.
- **SGDR** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/sgdr.md`): keep the now twice-validated 80% high-LR plateau and low-LR cosine tail fixed while tuning augmentation.

## Experimental History Review

- The moving baseline is `92.30%` at commit `11f8469`; EXP-005 must reach `92.40%`.
- EXP-002 established the optimizer horizon. EXP-003 showed synchronized loss overhead and target smoothing were poor next levers. EXP-004 added `RandAugment(num_ops=1, magnitude=7)` through 80%, switched to crop/flip, preserved 99.3% of steps, and gained 0.47 points.
- EXP-004's final strong checkpoint was only 84.60%, the first weak epoch recovered to 91.43%, and the peak arrived late at epoch 98. This makes augmentation strength and clean-refinement duration the narrow remaining parameters; worker lifecycle, evaluation cadence, model, and LR timing should remain fixed.

## Collected Ideas

<!-- Quick pass: candidate space is deliberately narrow; no separate seed collection. -->

## Combinations

<!-- Quick pass: candidates each change one augmentation policy dimension. -->

## Candidate Ideas

### Increase Magnitude to 9
**Summary**: Change only `RandAugment` magnitude from 7 to 9, retaining one operation, the 80% switch, the weak tail, worker lifecycle, and all optimizer/model controls.

**What it targets**: Remaining invariance headroom. EXP-004's conservative setting improved by 0.47 points, so a modestly stronger version may create more useful perturbation diversity before the validated clean tail.

**Reasoning**: Magnitude 9 is torchvision's default strength and lies close to the successful point, making this the simplest local exploitation test. Strong-loader throughput should remain essentially unchanged because operation count is fixed.

**Sources**: RandAugment knowledge note; EXP-004 analysis and accepted code.

**Estimated Effort**: low.

**Risk Assessment**: More severe transforms may destroy 32x32 semantics, depress representation quality rather than just strong-view accuracy, or require a longer weak tail. A single-run gain could also include inherent augmentation-stream variance.

### Use Two Operations at Magnitude 5
**Summary**: Change the strong phase from one magnitude-7 operation to two magnitude-5 operations, retaining the 80% weak switch and every other accepted choice.

**What it targets**: Breadth of composed invariances rather than severity of one perturbation. Two mild transforms may cover richer neighborhoods while avoiding magnitude-9 extremes.

**Reasoning**: RandAugment exposes operation count and magnitude precisely because their tradeoff matters. Worker preflight can determine whether the doubled PIL work preserves the fixed-time and step protocol.

**Sources**: RandAugment knowledge note; EXP-004 loader protocol and results.

**Estimated Effort**: low to medium.

**Risk Assessment**: Composed transforms may be more destructive than one stronger operation, loader throughput may fall, and the fixed horizon may underfit the harder distribution.

### Switch Augmentation Off at 75%
**Summary**: Keep one operation at magnitude 7 and the optimizer's `LR_HOLD_FRACTION=0.8`, but introduce a distinct augmentation switch fraction of 0.75. This gives 5% of the budget to weak crop/flip training at `lr=0.1` before the unchanged low-LR tail.

**What it targets**: Clean-objective adaptation and BatchNorm resettling. EXP-004 recovered sharply after the switch and peaked near termination, suggesting the weak phase may still be refinement-limited.

**Reasoning**: This isolates phase duration without changing augmentation strength or the validated LR schedule. The added 15 counted seconds of weak high-LR training may bridge the strong-to-clean distribution shift before annealing.

**Sources**: EXP-004 trajectory in `experiments/004/04-analysis.md`; EXP-002 schedule pattern.

**Estimated Effort**: low.

**Risk Assessment**: Shortening strong augmentation may sacrifice invariance learning; weak high-LR updates may overwrite useful features; adding a second fraction slightly complicates the phase logic.

## Review

Mandatory external Claude review completed successfully with no fallback (`01-idea-review.md`). It selected the 75% switch because it isolates one boundary with no extra throughput or distortion risk. Adopted the review's mechanism correction: the evidence is EXP-004's 6.83-point strong-to-weak jump and the opportunity for clean high-LR/BatchNorm adaptation, not the unsurprising late cosine peak. Did not adopt its proposed multi-seed variance runs because the user-defined goal forbids rerolling seeds; the experiment stays at the single fixed seed 42, and a near-threshold pass will be described cautiously rather than used for a strong causal claim.

## Idea Evaluation

Adopt the external winner. Claude scored the 75% switch `6/10` for evidence/reasoning and `5/10` for impact, versus two operations at `5/10` and `4/10`, and magnitude 9 at `4/10` on both. The selected idea changes only augmentation phase duration, preserves the accepted magnitude/operation count and LR schedule, and creates a distinct 75-80% interval of crop/flip training at `lr=0.1`.

## Chosen Idea
**Selected**: Switch Augmentation Off at 75%

**Why this idea**:
It is grounded in the accepted run's sharp distribution-switch recovery and is the cleanest way to give the weak objective and BatchNorm statistics high-LR adaptation before annealing. It adds no loader work, model work, or new worker lifecycle transition and changes only one phase boundary.

**Hypothesis**:
With `AUG_SWITCH_FRACTION=0.75` and `LR_HOLD_FRACTION=0.8`, five percent of counted training will use crop/flip at `lr=0.1` before the unchanged cosine tail, raising `best_test_acc` from `92.30%` to at least `92.40%` while preserving throughput and all other EXP-004 behavior.
