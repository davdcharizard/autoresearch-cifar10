# Brainstorm EXP-020
**Created**: 2026-08-06

## Web Search & Literature Review

- **Gradient Centralization: A New Optimization Technique for Deep Neural Networks** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/gradient-centralization.md`): ECCV 2020 reports positive CIFAR-100 SGDM results across several architectures, defines the official coupled-L2-before-projection order, and states that convolution-only GC is sufficient for small-resolution CIFAR. Its conventional CIFAR-100 setting differs materially from this saturated CIFAR-10 CutMix recipe, so it supplies a mechanism prior rather than a promised gain.

## Experimental History Review

- The lineage BASE -> EXP001 -> EXP002 raised accuracy from 91.51% to 95.23% through WRN-16-4 and then CutMix/clean-tail scheduling. The goal-wide best is EXP011 at 95.61%, but EXP020 deliberately returns to EXP002 to isolate GC without SAM/EMA interactions.
- EXP019 was a valid, full-dose test of official-order GC on all convolution and classifier weights. It preserved exposure and removed 22.05% of eligible gradient energy, yet scored 95.07%, 0.16 points below EXP002. Its most diagnostic split was a 41.89% convolution norm removal versus 93.21% classifier norm removal (`experiments/019/04-analysis.md`).
- EXP018 showed that frequent full-run Lookahead smoothing also lowered the stable tail, while other children of EXP002 either failed or supplied no evidence that generic optimizer smoothing is the current answer. Repeating full eligible GC or Lookahead is therefore closed.
- The limiting quality gap at this base is stable test generalization, not compute or memory (`02-system-understanding.md`). Any GC follow-up must preserve approximately the parent's exposure and change the harmful projection hypothesis rather than merely reduce runtime.
- The unresolved gap is eligibility/order/phase attribution: convolution-only official-order GC is directly recommended for small images and removes the path with the strongest measured distortion; raw-gradient GC would instead preserve coupled-decay common-mode directions; phase-limited GC would leave the clean endpoint unconstrained. No prior child has tested any of these exact rules.

## Collected Ideas

## Combinations

## Candidate Ideas

### Raw-gradient convolution-only centralization
**Summary**: Centralize only the raw data gradients of convolution weights immediately after backward, then add ordinary coupled L2 to all parameters and apply unchanged momentum/Nesterov. This preserves the L2-induced common-mode component instead of projecting the sum of data gradient and weight decay.

**What it targets**: The possibility that EXP019 harmed optimization by projecting regularization directions rather than data-gradient directions. It narrows eligibility to convolutions while explicitly preserving the parent's coupled-decay geometry.

**Reasoning**: EXP019 proves the official-order projection is strongly active but does not decompose removed energy into data-gradient versus L2 components. Since this model already depends on weight decay and CutMix, retaining L2 common-mode motion could avoid one destructive interaction while still applying the core GC transformation to convolution learning signals.

**Sources**: `experiments/019/04-analysis.md`; `knowledge/papers/gradient-centralization.md` (official order, which this candidate intentionally changes); `00-navigate.md`.

**Estimated Effort**: low

**Risk Assessment**: This changes both eligibility and update ordering at once, so a gain would not isolate which factor mattered. It has weaker literature support than official-order GC and may preserve exactly the mean component GC is intended to remove. A negative result would also be harder to generalize.

### Early-phase convolution-only centralization
**Summary**: Apply official-order convolution-only GC only during the first 75% high-LR phase, then use the exact EXP002 optimizer in the final clean quarter. Audit calls by phase and preserve all scheduling and evaluation behavior.

**What it targets**: A phase-specific tradeoff: GC may improve early conditioning yet constrain the low-LR clean endpoint that determines the stable accuracy tail. Disabling it late gives the model an unconstrained refinement period.

**Reasoning**: Projected updates have their strongest optimization rationale at high learning rates, while this repository's prior knowledge shows regularization can be relaxed late. EXP019's stable but lower final tail is consistent with persistent projection bias, so a clean tail could recover useful directions.

**Sources**: `experiments/019/04-analysis.md`; `knowledge/papers/gradient-centralization.md`; `knowledge/papers/time-matters-regularization.md`; `02-system-understanding.md`.

**Estimated Effort**: low

**Risk Assessment**: EXP019 did not measure phase-resolved removed energy, so the phase diagnosis is indirect. This adds a duration choice and weakens dose, making a miss ambiguous between an ineffective mechanism and insufficient exposure. It is less clean than the eligibility ablation.

### Convolution-only official-order gradient centralization
**Summary**: After each backward, add the inherited coupled L2 term to every gradient, but centralize only the 16 convolution weight gradients over their non-output dimensions. Leave the classifier, biases, and normalization parameters unprojected, then apply unchanged Nesterov momentum. Keep the EXP019 inventory, path, decomposition, and cadence audits narrowed to convolution eligibility so the run proves the intended ablation.

**What it targets**: EXP019's most concrete unresolved failure mechanism: the full rule removed 93.21% of classifier direction norm, plausibly suppressing useful class-boundary motion, while small-image GC literature says convolution-only eligibility is sufficient. The classifier is only 2,560 of 2,745,264 eligible elements, so the hypothesis is explicitly about disproportionate *output-layer sensitivity*, not classifier dominance of aggregate removed energy. It preserves GC's convolutional conditioning hypothesis while removing the most strongly distorted layer by relative norm.

**Reasoning**: This is the narrowest causal follow-up to a complete negative experiment. It changes exactly eligibility, retains the paper's official update order, costs slightly less than EXP019's measured 1.008x median overhead, and should preserve parent-like exposure. The run will additionally decompose removed row-mean energy by training phase and by raw-gradient versus L2 contribution without changing the applied update. A valid result at or below 95.23% will close the literature-supported official-order GC rules on this base; the audits can establish whether raw-order or phase-limited variants are numerically distinct, but cannot causally close those untested rules.

**Sources**: `experiments/019/04-analysis.md`; `knowledge/papers/gradient-centralization.md`; `00-navigate.md`.

**Estimated Effort**: low

**Risk Assessment**: Convolutional projection still removed 41.89% of norm in EXP019 and almost certainly accounts for most absolute removed energy; it may itself discard useful directions in this already-regularized recipe. The output-layer-sensitivity attribution is observational within one run, and the retained EXP019 summary does not preserve the classifier's absolute energy components, so excluding it may be insufficient. Worst case is another valid no-improvement near or below 95.23%, but the ablation remains decisive and cheap.

## Review

Claude selected convolution-only official-order GC (evidence 7/10, impact 5/10), identifying it as the only single-variable follow-up with direct literature support. I adopted four material refinements from `01-idea-review.md`: the rationale now distinguishes output-layer sensitivity from aggregate energy dominance; expected upside is framed as a modest close-out ablation in a regime where local optimizer perturbations have hurt; the applied run will audit phase buckets and raw-gradient-versus-L2 removed row-mean energy; and a valid result at or below the 95.23% parent preregisters closure of the literature-supported official-order GC rules on EXP002. A later plan review correctly narrowed this from all conceivable GC variants because magnitude audits alone cannot prove untested order/phase rules harmful. The requested classifier absolute-energy calculation cannot be recovered from EXP019's retained aggregate/norm-ratio transcript after raw-log cleanup, so the plan states that limitation rather than inventing a value.

## Idea Evaluation

Adopt the independent verdict. Raw-gradient convolution-only GC rests on an unmeasured claim that the `1e-4` L2 common-mode is material and changes both order and eligibility. Early-phase convolution-only GC likewise changes two factors and has no phase-resolved evidence. The official-order full-run convolution-only rule cleanly isolates eligibility; its non-interventional audits can answer whether either weaker sequel deserves reconsideration without spending another run.

## Chosen Idea
**Selected**: Convolution-only official-order gradient centralization

**Why this idea**:
It is the sole one-factor ablation against EXP019, follows the paper's small-image eligibility recommendation, preserves measured throughput, and converts a valid miss into a useful closure result. Its realistic upside is modest because EXP019 is the strongest in-domain evidence and was negative; if it succeeds on EXP002, transfer to the global-best EXP011 stack would still require a separate experiment.

**Hypothesis**:
Removing classifier GC while retaining official-order convolution GC will preserve useful class-boundary motion and produce a complete valid `best_test_acc >=95.33%` from EXP002 without meaningful exposure loss. A result at or below 95.23% closes the literature-supported official-order GC rules on this base; phase-resolved and raw-gradient-versus-L2 row-mean audits will measure whether timing or update order is numerically distinct, without pretending to test their causal accuracy effects.
