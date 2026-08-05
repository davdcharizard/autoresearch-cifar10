# Brainstorm EXP-015
**Created**: 2026-07-26

## Web Search & Literature Review

- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup.md`): convex example-wise interpolation regularizes decision boundaries with little compute overhead; the accepted recipe already validates the mechanism, but currently applies one shared strength to the whole batch.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`): regularization can be concentrated early and removed for late convergence, supporting retention of the accepted 65% transition to hard labels.
- **When Does Label Smoothing Help?** (`knowledge/papers/label-smoothing.md`): soft targets can improve overconfident classifiers, but stacking target regularizers requires care. No network access was used; this offline run relies on the project knowledge base.

## Experimental History Review

- EXP-001 and EXP-002 established the current best of 94.07%: WRN-16-2 with time-based cosine scheduling and alpha-0.2 batchwise mixup through 65%, followed by hard labels.
- EXP-003 through EXP-009 show that more regularization, altered regularization timing, lower late learning rate, or faster reduced-precision exposure all regress. The accepted balance is sensitive, and raw training exposure is not the limiter.
- EXP-010 and EXP-011 suggest low-resolution capacity has weak positive signal (94.11% and 94.15%), but both miss the required 94.17%; EXP-012 shows a cheaper low-rank substitute does not preserve it.
- EXP-013 and EXP-014 show that late whole-state averaging and identity-biased initialization do not produce enough top-1 boundary changes. Lower loss and cleaner residual startup do not translate directly to the primary metric.
- The limiting gap is generalization at a near-converged top-1 boundary: training loss approaches zero, exposure remains about 142 passes, and several transformations move loss or capacity without crossing 94.17%. Untried space remains in example-level target diversity and the geometry of individual SGD updates.

## Collected Ideas

- **Per-example mixup strengths** — draw one Beta(0.2, 0.2) coefficient per example, broadcast it over pixels, and compute unreduced paired cross-entropies with the same vector. This preserves the validated mixup distribution and 65% cutoff while preventing an entire batch from sharing one regularization strength, plausibly yielding more diverse boundary constraints per update.
- **Gradient centralization** — after backpropagation and before SGD, subtract each convolutional or linear weight gradient's mean over its non-output dimensions. This hyperparameter-free projection changes update geometry without altering the model, data, loss, schedule, or exposure materially, and may reduce feature-wise gradient bias in an overfit model.
- **Early-window SAM** — use a sharpness-aware two-pass update only during an early fraction of counted time, then return to the exact accepted SGD path. This imports an explicit flatness objective and follows the evidence that early interventions can persist, but trades substantial early exposure for potentially stronger generalization.
- **Stage-2 auxiliary supervision** — add a temporary pooled classifier after stage 2 with a small loss weight during the mixup phase, then disable it for the hard-label tail and discard it at evaluation. Direct intermediate supervision may improve representation quality, although its additional soft-target pressure and compute could recreate the over-regularization pattern.
- **Deranged mixup pairing** — replace unconstrained `randperm` with a cyclic random shift so no sample mixes with itself. This simplifies away identity pairings that waste a fraction of already scarce mixed-example constraints while preserving the loss and coefficient law; the expected effect may be too small at batch size 256.
- **Class-different mixup pairing** — construct pairings that prefer a different class, increasing the fraction of coefficients that constrain inter-class boundaries. It directly targets classification geometry but changes the effective task more aggressively and may over-regularize semantically nearby images.
- **Adaptive gradient clipping** — cap per-parameter update norms relative to parameter norms during the high-learning-rate phase. This orthogonal optimizer lever could suppress rare destabilizing steps without affecting steady updates, but there is no evidence that gradient outliers currently limit this stable run.
- **Cosine classifier moonshot** — normalize final features and classifier weights and train a learned or fixed logit scale. This representation-level change explicitly makes angular separation determine the decision boundary, but it disrupts the well-converged classifier and likely requires tuning beyond one experiment.

## Combinations

- **Per-example coefficients + deranged pairing**: every item receives an independently strong or weak interpolation and is guaranteed a non-self partner. The cross removes wasted identity constraints while maximizing within-batch target diversity, plausibly stronger than either small change alone, though it makes attribution less clean.
- **Gradient centralization + early-only activation**: centralize gradients only through the accepted 65% mixup window, then restore exact SGD for hard-label refinement. This could capture a smoother early representation without constraining late class-boundary fitting, combining update conditioning with the validated temporal-regularization pattern.
- **Early SAM + accepted hard-label tail**: confine expensive flatness-seeking steps to an initial short window and spend the majority of the budget on ordinary SGD. The combination is more feasible than full-run SAM and more targeted than generic early regularization, but still loses meaningful data passes.

## Candidate Ideas

### Mixup-Window Gradient Centralization
**Summary**: After backward and before the existing SGD step, subtract each `ndim > 1` parameter gradient's mean over all non-output dimensions, only while mixup is active. This affects the 16 convolution weights and classifier weight, leaves vector parameters untouched, and stops at the accepted 65% transition so the hard-label tail uses the original update rule (`proposals/idea-02.md`).

**What it targets**: The diagnosed need for a useful change in update geometry rather than more exposure, capacity, or loss reduction. The projection aims to remove common-mode filter-gradient components during representation formation.

**Reasoning**: It adds no target softness, forward compute, state, or new hyperparameter and is orthogonal to the tested architecture and regularization changes. Applying it only in the mixup window protects the locally validated late optimizer rule. The mechanistic case is plausible for convolutional filters followed by normalization, but there is no direct result in the offline project knowledge base.

**Sources**: EXP-002, EXP-007 through EXP-009, and EXP-013 in `04-results.tsv`; `03-experiment-learnings.md`; accepted `train.py`; `proposals/idea-02.md`.

**Estimated Effort**: low

**Risk Assessment**: The all-ones coefficient direction is not an exact symmetry, especially for the classifier, so projection can discard useful signal or compound mixup regularization. Seventeen reductions may also impose nontrivial kernel-launch overhead despite low FLOPs.

### Per-Example Mixup Strengths
**Summary**: Replace the one batch-shared `Beta(0.2, 0.2)` coefficient with 256 independent coefficients, broadcast them over each image, and use unreduced paired cross-entropies weighted by the same per-example vector. Keep the permutation, alpha, 65% cutoff, hard-label tail, and every other accepted setting unchanged. Do not combine it with deranged pairing in this experiment (`proposals/idea-01.md`).

**What it targets**: The diagnosed top-1 generalization boundary and high batchwise correlation of the accepted regularizer. It increases interpolation-constraint diversity within each update without increasing expected target softness or sacrificing exposure.

**Reasoning**: EXP-002 validates alpha-0.2 mixup, while EXP-003/005/006 show that changing its semantic strength or stacking regularizers hurts. Independent draws preserve the exact marginal coefficient law and expected mixup objective while reducing batch-average coefficient variance by about 16x at batch 256. This is materially different from stronger alpha or CutMix and adds only batch-length tensors.

**Sources**: `knowledge/papers/mixup.md`; `knowledge/papers/time-matters-regularization.md`; EXP-002 through EXP-006 in `04-results.tsv`; `proposals/idea-01.md`.

**Estimated Effort**: low

**Risk Assessment**: Batch-shared strength variation may be useful optimizer noise, and diverse coefficients alter BatchNorm statistics even though the marginal objective is unchanged. The effect may also be smaller than the metric threshold. RNG consumption necessarily changes but remains a single fixed-seed treatment.

### Ten-Percent Early-Window SAM
**Summary**: Use non-adaptive SAM with rho 0.05 only while pre-step counted progress is below 10%, then return to the exact accepted SGD path for the remaining 90%. Reuse the same mixed batch for both passes, preserve exactly one BatchNorm running-stat update, and perform one optimizer step from restored weights (`proposals/idea-03.md`).

**What it targets**: The diagnosed solution-geometry gap. SAM explicitly seeks a neighborhood with lower worst-case loss, a qualitatively different route to generalization than extra capacity, averaging, or stronger augmentation.

**Reasoning**: Early-only activation follows the saved evidence that early interventions can have persistent effects and leaves the entire hard-label tail untouched. It has a larger conceptual upside than the incremental candidates, but the repo has no direct SAM evidence and the 300-second budget makes its second pass expensive.

**Sources**: `knowledge/papers/time-matters-regularization.md`; EXP-002, EXP-009 through EXP-014 in `04-results.tsv`; accepted `train.py`; `proposals/idea-03.md`.

**Estimated Effort**: high

**Risk Assessment**: A 10% time window still projects roughly 5% fewer unique minibatches and fewer updates near peak LR. SAM can compound early mixup regularization, and correct perturbation/BatchNorm restoration is substantially more failure-prone. Rho 0.05 lacks local calibration.

## Review

The blind review selected **Per-Example Mixup Strengths**. I accept its central caution that the proposal's causal assumption remains unverified: shared coefficient variation may be useful SGD noise, and no evaluator-free preflight can settle that question. The treatment therefore remains standalone, with no deranged pairing, alpha change, or rescue variant. I also adopt the review's direction to treat the predicted range as directional rather than calibrated. Full concerns and scores are in `01-idea-review.md`.

## Idea Evaluation

Adopt the verdict in `01-idea-review.md`. Per-example mixup has the strongest combination of local evidence, negligible exposure cost, and single-variable attribution. SAM's higher ceiling is offset by an uncalibrated radius/window and lost early updates, while gradient centralization lacks a sound invariance argument for several of its mandated tensors.

## Chosen Idea
**Selected**: Per-Example Mixup Strengths

**Why this idea**:
It refines the only regularizer already shown to improve this accepted model while preserving the per-example coefficient distribution, expected objective, alpha, 65% window, and hard-label tail. Unlike the other finalists, it adds almost no compute and changes one explicit correlation structure, so either outcome is directly reusable.

**Hypothesis**:
Replacing the batch-shared mixup coefficient with independent per-example `Beta(0.2, 0.2)` coefficients will retain at least 95% matched mixup throughput and raise fixed-seed `best_test_acc` from 94.07% to at least 94.17% within the 300-second budget by increasing within-update interpolation diversity without increasing expected target softness.
