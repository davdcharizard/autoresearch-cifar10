# Brainstorm EXP-038
**Created**: 2026-07-27

## Web Search & Literature Review

- Existing local regularization notes (`knowledge/papers/time-matters-regularization.md`, `label-smoothing.md`, `mixup.md`) were consulted. No network or remote source was used.

## Experimental History Review

- EXP036 remains the 94.48% baseline with its exact pooled residual head, 94.45% endpoint, 0.2456 loss, and 130.304 passes.
- EXP037 moved only `fc.weight` to zero decay, retained 131.727 passes, scored 94.41%, and worsened loss to 0.2786. This normal-exposure result rejects classifier under-decay and supplies limited directional evidence for stronger shrinkage.
- Global late decay removal failed severely in EXP007. Mixup strength/duration and additive soft/masking regularizers are closed or adverse in their tested forms. The solution space for a clean follow-up is therefore narrow.

## Collected Ideas

## Combinations

## Candidate Ideas

### Early Epsilon-0.05 Label Smoothing Alongside Mixup
**Summary**: During the accepted mixup window, use `label_smoothing=0.05` in both paired cross-entropies, then preserve the exact hard-label tail.

**What it targets**: Confidence and boundary quality through a cheap target prior rather than parameter shrinkage.

**Reasoning**: Local literature supports smoothing for overconfidence, but paired mixup targets already regularize labels and no calibration metric diagnoses a need.

**Sources**: `knowledge/papers/label-smoothing.md`; EXP002, EXP035-037.

**Estimated Effort**: low

**Risk Assessment**: Likely stacks redundant soft-target regularization and epsilon is ungrounded; a miss would be harder to interpret than decay allocation.

### Double Only Terminal Classifier Decay
**Summary**: Add a third optimizer group so only `fc.weight` uses continuous `1e-3` coupled decay; keep every convolution and both pooled-head matrices at accepted `5e-4`, and all biases/BN affine tensors at zero.

**What it targets**: The loss/confidence deterioration exposed by zero classifier decay, testing the opposite one-point bracket without changing graph, initialization, or pre-step gradients.

**Reasoning**: EXP037 provides direct directional evidence that classifier shrinkage matters. Doubling is a prospective symmetric magnitude step around accepted `5e-4`, while preserving all representation regularization. Attribution remains exact before optimizer step.

**Sources**: EXP036/037 reports; `02-system-understanding.md`; installed SGD semantics.

**Estimated Effort**: low

**Risk Assessment**: The negative zero-decay result does not imply monotonic benefit; `1e-3` is still a bracket choice and may underfit class vectors. A valid miss must close classifier-decay strength tuning in both directions.

### Increase All Matrix Decay to 7.5e-4
**Summary**: Change `WEIGHT_DECAY` globally from `5e-4` to `7.5e-4` for convolutions, pooled head, and classifier, retaining all other behavior.

**What it targets**: The remaining train/test generalization gap through stronger representation-wide norm control.

**Reasoning**: The learner nearly interpolates and zero classifier decay worsened loss, but no global stronger-decay point has been isolated. The 1.5x value is weaker than doubling to limit underfitting.

**Sources**: EXP007, EXP036/037, system diagnosis.

**Estimated Effort**: low

**Risk Assessment**: It changes over one million parameter updates, has no local magnitude calibration, and confounds successful pooled-head/backbone regularization; a miss is less diagnostic than classifier-only decay.

## Review

The offline critic selected double classifier decay because EXP037 explicitly left increased decay open and `1e-3` is a prospective symmetric opposite step around accepted `5e-4`. I adopted the crucial limitation: zero decay's loss regression is directional evidence that shrinkage matters, not monotonic evidence that more is better. Preserve all representation/head allocations; a normal-exposure miss closes classifier-decay strength in both directions. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the pick. Classifier-only `1e-3` scored 5/10 evidence and 3/10 impact. Global decay has more possible effect but poor attribution; label smoothing is redundant and underdetermined.

## Chosen Idea
**Selected**: Double Only Terminal Classifier Decay

**Why this idea**:
Create a third optimizer group containing only `fc.weight` at continuous `1e-3` decay. Keep every convolution and both pooled-head matrices at `5e-4`, all rank-below-2 tensors at zero, and preserve the exact accepted graph, initialization, RNG, data, seed, and evaluator.

**Hypothesis**:
If stronger terminal class-vector shrinkage improves confidence/boundary generalization on the pooled representation, `1e-3` classifier decay will retain at least 127 projected and realized passes and raise fixed-seed best accuracy from 94.48% to at least 94.58%, with final accuracy >=94.45% and loss <=0.2456 as corroboration. A valid normal-exposure miss closes classifier decay below and above `5e-4` without intermediate values or schedules.
