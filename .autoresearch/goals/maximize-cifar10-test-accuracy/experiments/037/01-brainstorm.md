# Brainstorm EXP-037
**Created**: 2026-07-27

## Web Search & Literature Review

- Existing local knowledge on mixup, early regularization, label smoothing, RandAugment, averaging, and WRNs was consulted through `knowledge/README.md`. No network or remote source was used.

## Experimental History Review

- EXP036 moved the frontier to 94.48% with a fixed scale-0.1 pooled residual MLP, 94.45% final accuracy, 0.2456 loss, and 130.304 passes. Preserve its exact width, scale, activation, seed, and decay unless an independent mechanism directly requires otherwise.
- The remaining limiter is generalization under a tighter 130-pass compute envelope. Spatial compute, batch scaling, masking, averaging, mixup tuning, SAM, and SE are closed or costly in their tested forms.
- The new head is positive evidence for post-spatial nonlinear remapping, not evidence for immediate width/scale tuning. Orthogonal optimizer or classifier geometry changes remain unmeasured but weakly evidenced.

## Collected Ideas

- **Classifier-only zero decay** - exempt only `fc.weight` while preserving decay on the successful pooled head and every convolution; exceptionally isolated and compute-neutral, but adverse confidence evidence and only 1,280 affected parameters limit upside.
- **Classical momentum 0.9** - change only `nesterov=True` to false, testing whether the new nonlinear head benefits from less current-gradient amplification. It is cost-neutral but the accepted LR was built around Nesterov and stable endpoint evidence argues against it.
- **Fixed-scale cosine classifier** - normalize pooled features and class weights at fixed scale 10. It directly changes angular geometry after pooling, but bundles bias removal, normalization, and uncalibrated gradient scale.
- **Direct/full logit consistency** - cheaply regularize the head against the direct path. Detailed development rejected it because it penalizes the exact correction that produced EXP036 and has no defensible teacher, coefficient, or window.
- **Early label smoothing atop mixup** - smooth both paired CE targets before the hard tail. This is cheap but stacks soft-target regularization after alpha strength/duration were tightly bracketed, with no calibration diagnosis.
- **Gradient centralization** - subtract per-output-channel gradient means for matrix parameters before SGD. It is a representation/optimizer hybrid with low arithmetic but no local evidence and changes every accepted update.
- **Remove classifier bias** - simplify the final boundary after normalized pooled features. Likely too small, construction-RNG isolation is awkward, and it lacks a mechanism beyond redundancy.

## Combinations

- **Cosine classifier + classifier under-decay** is internally coherent because normalized weights make radial decay indirect, but this compounds two uncertain optimizer geometries and is not appropriate before isolation.
- **Pooled head + classifier-only zero decay** preserves the successful remapping and changes only terminal vector shrinkage, making it cleaner than any head modification.

## Candidate Ideas

### Classical Momentum at Coefficient 0.9
**Summary**: Change only PyTorch SGD from Nesterov to classical momentum while retaining coefficient 0.9 and every LR/model/data choice.

**What it targets**: Whole-model update dynamics, testing whether the nonlinear pooled head benefits from removing Nesterov's extra current-gradient term.

**Reasoning**: Nesterov was bundled into EXP001 and never isolated on the new learner, but the LR and stable endpoint favor keeping it. Full contract: `proposals/idea-02.md`.

**Sources**: EXP001, EXP036, installed PyTorch SGD semantics, `proposals/idea-02.md`.

**Estimated Effort**: low

**Risk Assessment**: Materially reduces first-step displacement without LR compensation and can under-optimize; a miss closes exact classical 0.9.

### Exclude Only Terminal Classifier Weight From Decay
**Summary**: Move only the 1,280-element `fc.weight` from `5e-4` decay into the existing zero-decay group for the full run; preserve the pooled head and all other matrices.

**What it targets**: Terminal class-vector fitting at no graph cost under the new nonlinear pooled representation.

**Reasoning**: EXP007 does not isolate this allocation, while exact pre-step identity gives unusually clean attribution. Direction and ceiling are weak. Full contract: `proposals/idea-01.md`.

**Sources**: EXP007, EXP036, `02-system-understanding.md`, `proposals/idea-01.md`.

**Estimated Effort**: low

**Risk Assessment**: May inflate confidence and worsen loss; effect may be below the ten-example margin. A miss closes under-decay only.

### Fixed-Scale Cosine Classifier
**Summary**: Replace affine logits with `10 * linear(normalize(features), normalize(fc.weight))`, removing classifier bias while preserving the accepted pooled head.

**What it targets**: Angular decision geometry after the successful pooled remapping.

**Reasoning**: It has larger upside than tiny decay allocation but scale 10 and bias removal are uncalibrated bundled choices. Detailed prior contract: `experiments/036/proposals/idea-03.md`, updated to baseline 94.48 and threshold 94.58.

**Sources**: EXP036 analysis/system diagnosis; `experiments/036/proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: Changes gradient scale throughout the backbone and may destabilize training; a miss closes only exact scale 10.

## Review

The offline critic selected classifier-only zero decay solely as the cleanest causal closure test. I adopted its constraints: preserve both successful pooled-head matrices at `5e-4`; do not infer that the train/test gap diagnoses excessive classifier shrinkage; and close intermediate classifier decay, schedules, LR compensation, seeds, and head tuning after a valid miss. Cosine classification was rejected for bundled uncalibrated scale/normalization, and classical momentum for uncompensated update-amplitude change without instability evidence. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the pick despite low expected impact. Classifier under-decay scored only 4/10 evidence and 3/10 impact, but it is the only finalist that changes no graph, initialization, RNG, loss, or update before the optimizer step. That makes one fixed-seed result interpretable; the alternatives are higher-upside but underdetermined systems.

## Chosen Idea
**Selected**: Exclude Only Terminal Classifier Weight From Decay

**Why this idea**:
Move only `fc.weight` into the existing zero-decay optimizer group for the full run. Keep every convolution and both pooled-head matrices continuously decayed, and retain the exact accepted model, head, schedule, data, seed, and evaluator. This cleanly resolves whether terminal class-vector shrinkage is unnecessary on the new nonlinear representation without reopening global decay timing.

**Hypothesis**:
If coupled decay on only the terminal class vectors constrains useful boundary fitting, zero-decay `fc.weight` will retain at least 127 projected and realized passes and raise fixed-seed `best_test_acc` from 94.48% to at least 94.58%, with final accuracy at least 94.45% and loss at most 0.2456 as corroboration. A valid normal-exposure miss closes classifier under-decay and all adjacent rescues.
