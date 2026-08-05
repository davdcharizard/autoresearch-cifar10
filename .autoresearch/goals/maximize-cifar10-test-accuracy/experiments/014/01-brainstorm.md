# Brainstorm EXP-014
**Created**: 2026-07-26

## Web Search & Literature Review
- `knowledge/papers/wide-residual-networks.md`: residual architecture can trade depth and width, but three local low-resolution allocations are now closed.
- `knowledge/papers/label-smoothing.md` and `time-matters-regularization.md`: generalization interventions exist, but repeated local soft-target/additive regularization failures weaken them.

No network search was performed because this autoresearch session is offline.

## Experimental History Review
- Accepted WRN-16-2 plus early mixup remains 94.07% at 141.9 passes.
- Capacity probes reached 94.11/94.15 before the efficient bottleneck regressed to 93.74; raw architecture allocation is no longer the leading immediate direction.
- BF16, EMA, stronger/shorter mixup, CutMix, dropout, late decay removal, and cosine-to-zero all failed. The live accepted graph still has one untested initialization geometry intervention with unchanged compute.

## Diagnosis
The accepted network fits to near-zero loss with adequate exposure, so the remaining gap is the optimization basin/generalization geometry rather than basic capacity or throughput. A deterministic initialization-only change is more isolated than another regularizer or model expansion.

## Collected Ideas
Quick pass; finalists are below.

## Combinations
Quick pass; combinations are excluded for attribution.

## Candidate Ideas

### True CIFAR Per-Channel Standardization
**Summary**: Replace unit standard deviations with canonical CIFAR-10 channel standard deviations while retaining the existing mean, crop, flip, architecture, and optimizer.

**What it targets**: Input conditioning and channel-scale imbalance.

**Reasoning**: Standardization is conventional and free at runtime, but the accepted LR is calibrated to unit-scale subtraction; this nominal data fix implicitly changes effective first-layer step sizes.

**Sources**: accepted `train.py` normalization comment and CIFAR practice; no new external source.

**Estimated Effort**: low.

**Risk Assessment**: It may destabilize the fixed LR or merely rescale a representation BatchNorm already handles. Exact canonical values also introduce a configuration choice.

### Safe Zero-Initialized Residual Endpoints
**Summary**: After unchanged Kaiming initialization, overwrite all six `PreActBlock.conv2.weight` tensors with zero. Keep all BN scales at one; zeroing pre-ReLU `bn2` would permanently kill branch gradients. Endpoint convolutions learn on step one and upstream branch parameters on step two.

**What it targets**: Early optimization geometry and basin selection without changing graph, parameter count, throughput, RNG consumption, targets, or schedule.

**Reasoning**: This is the remaining fully developed orthogonal treatment (`experiments/012/proposals/idea-03.md`). It preserves roughly 141.9 passes and can be verified through exact first/second-step gradient semantics.

**Sources**: `experiments/012/proposals/idea-03.md`; accepted `PreActBlock.forward`; EXP-011/012/013 analyses.

**Estimated Effort**: low.

**Risk Assessment**: The shallow WRN may not benefit from identity-biased initialization, and removing early random residual features may mismatch the accepted warmup/LR trajectory.

### Replace ReLU With SiLU
**Summary**: Use `F.silu` at all seven current ReLU sites while preserving topology and training recipe.

**What it targets**: Smooth gradient flow and representation quality rather than capacity.

**Reasoning**: A smooth nonlinearity can improve optimization without parameters, but it changes every residual and final activation and may add kernel cost under the fixed budget.

**Sources**: accepted activation structure; general activation-function mechanism only.

**Estimated Effort**: low.

**Risk Assessment**: Weak local evidence, broad graph-wide intervention, lower throughput, and the accepted Kaiming-ReLU initialization is no longer perfectly matched.

## Review
The blind offline critic selected safe zero-initialized endpoints (8.5 evidence / 7.0 impact). I adopt its correction that prior failures make optimization geometry plausible but do not prove it is the limiter. Exact all-six `conv2` zeroing, RNG preservation, gradient semantics, and >=135-pass gate are mandatory. Full review: `01-idea-review.md`.

## Idea Evaluation
Adopt the reviewer pick. SiLU's actual 13 runtime activations make it broader and less evidenced; channel standardization is coupled to effective LR and largely buffered by early BN.

## Chosen Idea
**Selected**: Safe Zero-Initialized Residual Endpoints

**Why this idea**:
It is the only remaining candidate with an architecture-correct, directly falsifiable mechanism that preserves graph, parameters, RNG consumption, and throughput.

**Hypothesis**:
Zeroing all six `PreActBlock.conv2.weight` tensors after accepted initialization will retain at least 97% throughput / 135 projected and realized passes and reach `best_test_acc >=94.17%` in one fixed-seed run.
