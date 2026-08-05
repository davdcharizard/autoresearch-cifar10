# Brainstorm EXP-036
**Created**: 2026-07-27

## Web Search & Literature Review

- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`): shallow-wide CIFAR residual representations are compute-effective, but local results require preserving the accepted depth/invariance composition rather than adding broad spatial compute.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): soft targets can reduce overconfidence, but the note warns against stacking soft-target methods without calibration and gives no project-specific epsilon.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): early regularization can be removed for late convergence, supporting the accepted temporal structure but not interchangeability of regularizers.

No network or remote source was consulted. This offline thorough pass used the
existing per-goal knowledge base, source, system measurements, and 35 completed
experiment records.

## Experimental History Review

- EXP027 remains the 94.32% frontier: early worker-private N1/M5 RandAugment composes positively with the otherwise sub-margin `(2,2,3)` depth change. Preserve the full residual blocks, early invariance, alpha-0.2 batch-shared mixup through 65%, FP32, batch 256, schedule, decay, and seed unless directly testing one element.
- The current limiter is decision quality rather than memory, input delivery, or wall headroom. Forward/backward consumes about 98% of counted step time and the run nearly interpolates training, while test loss remains 0.2523. New spatial compute directly trades against the narrow 133-pass regime.
- Immediate strength/duration changes to mixup, batch scaling, parameter averaging, feature masking, SAM, SE attention, precision/layout, padding, late freezing, schedule floor, weight-decay timing, and nearby capacity reallocations are closed or infeasible in their tested forms.
- A low-dimensional classifier or pooled-feature intervention remains materially underexplored. It can change class-boundary geometry without paying high-resolution convolution cost, but lacks local evidence and needs stronger semantic checks than a one-line regularization closure.
- The system is not proven saturated: the accepted score is far below a known CIFAR ceiling and prior interactions produced a 0.25-point jump after standalone near misses. The evidence supports a narrower search, not an exhaustion claim.

## Collected Ideas

- **Classifier-specific weight-decay allocation** - move only `fc.weight` out of the matrix-decay group while retaining `5e-4` on every convolution. This targets the final angular/radial boundary fit at effectively zero compute and clean attribution. It differs from EXP007, which removed all matrix decay only in the tail, but the direction is weakly supported and may simply inflate logit norms.
- **Exact-neutral pooled residual MLP head** - after global average pooling, add a small `128 -> 128 -> 128` residual transformation before the classifier, initialized to the identity through a zero terminal projection and trained throughout. It spends capacity after expensive spatial processing, directly targets boundary representation, and preserves all high-resolution gradients; prior zero residual endpoints and a post-stage spatial bottleneck failed, so initialization and family closure must be strict.
- **Cosine-normalized classifier** - L2-normalize pooled features and class weights and use one preregistered fixed scale, replacing affine radial logits with angular class boundaries. It adds little compute and directly attacks boundary geometry, but the scale is an uncalibrated hyperparameter and removing feature/weight norm information substantially changes SGD dynamics.
- **Higher Nesterov momentum** - change momentum from 0.9 to 0.95 while keeping the exact time-based LR curve. This is a zero-graph-cost optimization lever that may smooth noisy short-horizon updates, but it changes effective step dynamics across warmup and the hard tail without local momentum evidence; a one-point miss would close only the exact value.
- **Smooth activation substitution** - replace residual ReLUs with SiLU so small negative and near-zero signals can shape features. This is a representation-level change and may improve function quality, but it abandons the validated preactivation WRN recipe and adds elementwise work in stage 1, the most expensive spatial stage.
- **Early label smoothing instead of mixup** - replace mixed images and paired labels with epsilon-0.05 cross entropy through 65%, then use the exact hard tail. The local paper note supports confidence control at negligible cost, but this discards EXP002's strongest causal gain without any measured calibration diagnosis and leaves epsilon underdetermined.
- **Remove classifier bias** - make the final linear layer bias-free while preserving its weight, initialization stream through an explicit zero-width-equivalent RNG contract, and all training choices. This simplifies a redundant offset after final BatchNorm and pooling, but its plausible effect is too small and changing construction RNG makes clean isolation more complex than the intervention warrants.
- **Pooled-feature prototype regularization moonshot** - maintain classwise pooled-feature means during the early window and add a small attraction/separation term, then remove it for the hard tail. This targets intra-class compactness explicitly after cheap pooling, but introduces state, an arbitrary coefficient, mixed-label prototype semantics, and extra synchronization; it is too compound without new evidence.

## Combinations

- **Pooled residual head + classifier-specific decay**: put the extra low-cost boundary capacity under ordinary matrix decay while exempting only the terminal classifier. The combination could separate representation shaping from final radial shrinkage better than either alone, but it compounds two unsupported choices and should follow isolated evidence.
- **Cosine classifier + early label smoothing**: angular logits could make uniform target entropy act on directions rather than feature norms, potentially stabilizing cosine optimization. It is more coherent than either generic treatment alone, but scale and epsilon create an unjustified two-dimensional operating point and discard accepted mixup.
- **Pooled residual head + accepted early invariance**: retain the full EXP027 training recipe and add capacity only after pooling. This is the natural isolated form of the head proposal rather than a new compound treatment: early invariance continues shaping spatial features while the new head can refine their class geometry at negligible MAC cost.

## Candidate Ideas

### Exclude Only Classifier Weight From Decay
**Summary**: Keep continuous `5e-4` decay on all 983,472 convolution parameters but move only the 1,280-element `fc.weight` tensor into the accepted zero-decay group for the entire run. All architecture, initialization, LR, momentum, data, regularization timing, seed, and evaluator behavior remain exact.

**What it targets**: The boundary-quality gap identified in `02-system-understanding.md`, allowing final class vectors to fit the already-regularized representation without weakening spatial feature decay or adding counted compute.

**Reasoning**: EXP007 proves continuous convolutional decay is necessary but does not isolate the classifier: it removed decay from every matrix only in the final 35% and sharply worsened loss. The proposed allocation changes 0.13% of decayed parameters from the first step and has exceptional causal and systems isolation. Full contract: `proposals/idea-01.md`.

**Sources**: `experiments/007/04-analysis.md`; `experiments/027/04-analysis.md`; `02-system-understanding.md`; `proposals/idea-01.md`.

**Estimated Effort**: low

**Risk Assessment**: Accepted classifier decay may control logit norms and confidence; removing it can worsen loss, and only 1,280 parameters change, so upside may be below the ten-example margin. A valid miss closes only classifier under-decay, not increased decay.

### Fixed-Scale Cosine Classifier
**Summary**: Remove classifier bias and compute logits as `10 * linear(normalize(features), normalize(fc.weight))`, retaining the accepted class-weight tensor, decay group, backbone, training recipe, and seed. No learned scale, margin, or auxiliary loss is allowed.

**What it targets**: The same boundary-quality gap through a different representation: angular class decisions that cannot exploit feature/weight norms to interpolate training.

**Reasoning**: It operates after pooling at negligible spatial cost and directly addresses classifier geometry, an untested gap after regularization and averaging failures. Its evidence is mechanistic rather than local; fixed scale 10 is intentionally exposed as the central uncertainty. Full contract: `proposals/idea-03.md`.

**Sources**: `02-system-understanding.md`; EXP027; `proposals/idea-03.md`.

**Estimated Effort**: medium

**Risk Assessment**: Scale 10 is uncalibrated and changes gradient magnitude throughout the network; normalization makes weight decay's forward effect indirect; bias removal and near-small feature norms add confounds. A miss closes only this exact operating point.

### Scaled Pooled-Feature Residual MLP Head
**Summary**: Add a bias-free `128 -> 64 -> 128` ReLU MLP after global pooling and before the accepted classifier, using a fixed `0.1` residual scale and a prospectively fixed isolated initialization seed. It adds 16,384 parameters and MACs per image while preserving the entire spatial backbone and direct pooled path.

**What it targets**: Boundary representation capacity after the 98%-of-step spatial forward/backward bottleneck, seeking nonlinear class separation without paying for another 8x8 convolution or removing high-resolution gradients.

**Reasoning**: EXP027 shows extra capacity can become useful under early invariance, while EXP012's spatial rank-64 bottleneck failed at over 200x this candidate's added MACs. The nonzero scaled initialization explicitly avoids EXP014's delayed exact-zero branch opening. Full contract: `proposals/idea-02.md`.

**Sources**: `experiments/012/04-analysis.md`; `experiments/014/04-analysis.md`; `experiments/027/04-analysis.md`; `02-system-understanding.md`; `proposals/idea-02.md`.

**Estimated Effort**: medium

**Risk Assessment**: The arbitrary scale and branch seed weaken attribution, extra nonlinear capacity can worsen the train/test gap, and tiny GEMMs may be launch-bound. A miss must close the whole nearby pooled-MLP neighborhood rather than invite width/scale/init rescue.

## Review

The offline adversarial reviewer selected the scaled pooled-feature residual MLP as the best balance of local evidence and upside. I adopted its main corrections: pooled-head capacity is exploratory rather than a diagnosed classifier bottleneck; the treatment tests a cheap nonlinear remapping of accepted pooled features, not whether image decision boundaries are globally linear; and initial branch/direct norm ratio, logit perturbation, and backbone/classifier/head gradient norms must be measured as non-tuning diagnostics. Scale 0.1, width 64, ReLU, and isolated seed 36036 remain fixed, and a valid miss closes their nearby rescue neighborhood. Full review: `01-idea-review.md`.

## Idea Evaluation

Adopt the reviewer's pick. The pooled head scored 3.5/5 for both evidence and impact, ahead of cosine classification's uncalibrated bundled geometry and classifier under-decay's weak direction/ceiling. Its evidence is still indirect: EXP027 supports capacity under early invariance, while post-pooling placement and nonzero initialization make it materially distinct from EXP012/014. That is enough for one preregistered exploratory score, not a claim that the final affine layer is the established limiter.

## Chosen Idea
**Selected**: Scaled Pooled-Feature Residual MLP Head

**Why this idea**:
Add exactly one bias-free `128 -> 64 -> 128` ReLU branch after accepted global pooling, combine it with the direct pooled vector at fixed scale 0.1, and keep every accepted spatial, optimization, augmentation, seed, and evaluator choice. The branch adds only 16,384 parameters/MACs per image after the dominant convolutional work, preserves the accepted direct representation, starts actively rather than reproducing failed exact-zero opening, and extends the only locally validated capacity-plus-invariance interaction.

**Hypothesis**:
If the accepted deeper-plus-early-invariance learner benefits from a cheap nonlinear remapping of pooled channel co-occurrences, then the fixed scale-0.1 residual MLP will retain at least 130 projected and realized passes and raise fixed-seed `best_test_acc` from 94.32% to at least 94.42%, with `final_test_acc >=94.32%` and `final_test_loss <=0.2523` as corroboration. A valid normal-exposure miss closes immediate pooled-MLP width, scale, activation, bias, zero-init, learnable-scale, head-seed, and optimizer rescues.
