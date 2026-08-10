# Brainstorm EXP-009
**Created**: 2026-08-05

## Web Search & Literature Review

- **Wide Residual Networks** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/wide-residual-networks.md`): wider CIFAR residual networks are commonly paired with stronger regularization, but their published `5e-4` decay uses a much longer and architecturally different regime, so the local operating point must be measured.

## Experimental History Review

- EXP-002 established the long `lr=0.1` plateau and low-LR cosine tail; EXP-004 added N1/M7 RandAugment through the 80% boundary and reached 92.30%.
- EXP-005 and EXP-006 showed that shortening strong augmentation or replacing it with fixed Cutout loses accuracy, so the accepted augmentation lifecycle should remain fixed.
- EXP-007 widened ResNet-20 by 2x and raised the moving baseline to 93.55% despite 29.2% fewer updates, establishing capacity as valuable under strong views.
- EXP-008 increased coupled decay from `1e-4` to `5e-4`. Accuracy fell to 93.38%, but final NLL improved from 0.2196 to 0.1988 while the strong checkpoint collapsed from 90.08% to 81.29% and the final tail was still rising.
- The limiting error mode is therefore an operating-point tradeoff: accepted `1e-4` may leave some generalization available through norm control, while `5e-4` suppresses strong-view fitting too severely in 300 seconds. The untested gap is whether a lower scalar or selective parameter targeting can preserve fit while retaining part of the NLL benefit.

## Collected Ideas

## Combinations

## Candidate Ideas

### Coupled Weight Decay 2e-4
**Summary**: Change only `WEIGHT_DECAY` from the accepted `1e-4` to `2e-4`, keeping the single SGD parameter group and every other training, data, architecture, timing, and evaluation setting fixed.

**What it targets**: The width-2 regularization operating-point gap exposed by EXP-008. It adds modest norm pressure without approaching the `5e-4` setting that severely suppressed strong-view fit.

**Reasoning**: EXP-008 improved final NLL but lost top-1 and showed unmistakable underfit at `5e-4`; this conservative interpolation tests whether some of that confidence/generalization benefit can be retained while substantially reducing the fit penalty. It has exact attribution and effectively no added compute.

**Sources**: EXP-007 and EXP-008 reports; `03-experiment-learnings.md`; `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: low

**Risk Assessment**: The response may be non-monotonic or the accepted `1e-4` may already be optimal. A single fixed-seed result near the 0.10-point gate could also be dominated by run noise, though rerolls are prohibited.

### Coupled Weight Decay 3e-4
**Summary**: Change only `WEIGHT_DECAY` from `1e-4` to `3e-4` in the existing single SGD group, preserving the accepted width-2 recipe exactly otherwise.

**What it targets**: The same norm-control versus finite-horizon fit tradeoff, but chooses the arithmetic midpoint of the tested scalar interval to retain more of EXP-008's lower-NLL effect.

**Reasoning**: If the top-1 optimum lies broadly between `1e-4` and `5e-4`, `3e-4` is more likely than `2e-4` to produce a measurable regularization shift. The intervention remains compute-neutral and cleanly attributable.

**Sources**: EXP-008 `04-analysis.md`; `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: low

**Risk Assessment**: The 8.79-point strong-checkpoint collapse at `5e-4` suggests a steep response; `3e-4` may still over-regularize the short plateau and repeat the top-1 failure at reduced magnitude.

### Selective Weight-Only Decay 5e-4
**Summary**: Split optimizer parameters into weight tensors (`ndim > 1`) with `5e-4` decay and normalization affine parameters plus biases with zero decay. Keep the model, schedule, augmentation, seed, and evaluator unchanged.

**What it targets**: A possible mechanism behind EXP-008's underfit: applying strong coupled decay indiscriminately to BatchNorm scales and biases, rather than limiting norm control to convolution and classifier weight matrices.

**Reasoning**: The accepted optimizer currently applies decay to every trainable tensor. Selective decay could preserve BN affine flexibility during strong RandAugment while retaining stronger regularization of the high-capacity kernels; it directly separates parameter-targeting from scalar strength.

**Sources**: Local `train.py` optimizer construction; EXP-008 `04-analysis.md`; `knowledge/papers/wide-residual-networks.md`.

**Estimated Effort**: medium

**Risk Assessment**: EXP-008 does not isolate BN/bias decay as the cause, so the hypothesis is weaker than scalar interpolation. Most parameters and norm pressure remain in convolution weights, meaning strong-view underfit may persist. Parameter-group logic also expands the change surface.

## Review

Claude completed the mandatory external idea review with exit code 0; no fallback reviewer was used. It found both scalar interpolations low-impact because the measured endpoints do not establish a top-1 optimum above the accepted `1e-4`, and it judged `3e-4` dominated by `2e-4`. It selected parameter targeting as the only off-curve mechanism with a credible path above the gate, while identifying the proposed `5e-4` kernel scalar as self-defeating. Its initial refinement proposed `2e-4` on weights and zero on one-dimensional parameters. The later mandatory Claude plan review exposed that refinement as a two-lever confound that still pushes fit-limited kernels toward EXP-008's failure mode. The final executable refinement therefore preserves the accepted `1e-4` on weight tensors and changes only BN affine and bias decay to zero. Full critiques are in `01-idea-review.md` and `02-plan-review.md`.

## Idea Evaluation

Adopt the reviewer's selective-decay pick, refined after plan review into an isolated targeting test. Kernel decay remains at the accepted `1e-4`; only one-dimensional BN affine parameters and biases move from `1e-4` to zero. This removes the scalar confound and tests whether directly shrinking those functional scale/offset parameters contributes to finite-horizon underfit.

## Chosen Idea
**Selected**: Selective Weight-Only Decay 5e-4 (reviews refined kernel scalar to baseline 1e-4)

**Why this idea**:
Claude scored the selective mechanism highest on potential impact because it is the only candidate not confined to interpolation on the already-declining all-parameter scalar curve. The plan critic then correctly rejected raising kernel decay at all: it would confound targeting with scalar strength and contradict the fit-limited diagnosis. The executable version keeps `1e-4` on tensors with `ndim > 1` and sets zero on BN affine and biases, isolating the selected targeting mechanism.

**Hypothesis**:
Removing decay from BN affine and bias parameters while preserving accepted `1e-4` kernel decay will improve strong-view fitting without sacrificing kernel norm control, reaching at least 93.65% best test accuracy under the fixed 300-second run. The 80% checkpoint should remain near or above EXP-007's 90.08%; a lower value would contradict the proposed mechanism.
