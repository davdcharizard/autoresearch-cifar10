# Brainstorm EXP-019
**Created**: 2026-08-06

## Web Search & Literature Review

- **Gradient Centralization: A New Optimization Technique for Deep Neural Networks** (`knowledge/papers/gradient-centralization.md`)
  ECCV 2020 and the official CIFAR optimizer support coefficient-free centralization of eligible regularized directions in the exact order `data gradient + coupled L2 -> GC -> momentum/Nesterov`, with no extra model forward.
- **Deeply-Supervised Nets** (`knowledge/papers/deeply-supervised-nets.md`)
  AISTATS 2015 reports CIFAR gains from training-only intermediate companion objectives, offering one-backbone-forward representation supervision but weak architecture matching to this shallow BN residual network.
- **A Unified Analysis of Mixed Sample Data Augmentation** (`knowledge/papers/mixed-sample-analysis.md`)
  NeurIPS 2022 explains distinct Mixup and CutMix input-gradient regularization, supporting an additive hybrid only if the validated CutMix dose and clean-tail convergence remain intact.

## Experimental History Review

- The lineage reached 91.51 -> 94.62 through the time-aware WRN and 94.62 -> 95.23 through front-loaded probability-0.5 CutMix. EXP002 completed 27,950 steps, ended at 95.19%, and reduced final CE to 0.2044 (`tree.sh log 002`; `experiments/002/04-analysis.md`).
- EXP004's late periodic SAM is the only successful child of EXP002, adding 0.17 points before EXP011's separate clean-tail EMA lifted the global tip to 95.61. This shows optimizer geometry can matter, but the remaining signal is small and selection noise near 0.17 points requires a plausible 0.25-0.30-point stable effect (`02-system-understanding.md`).
- EXP003 rejected a selected CutMix/drop-path scalar change, EXP009 found four FP32 SE gates 20.7% too slow, EXP010's equal-MAC back-loaded depth raised exposure but not accuracy, and EXP018's exact low-overhead canonical Lookahead reduced the stable tail to 94.72 mean. Do not repeat those packages (`tree.sh children 002`).
- EXP017 is not negative GC evidence. It passed exact eligible inventory, official update ordering, RNG, excluded-update, and numerical checks, but its disposable preflight retained 1,024 CUDA loss scalars after the sole repair and emitted no timing or accuracy (`experiments/017/04-analysis.md`; `experiments/017/04-result-review.md`).
- EXP018 validated the missing procedural recipe: fixed device diagnostics plus baselines taken after persistent optimizer/audit state held allocation exactly constant for a 1,056-step trace. It also showed sparse foreach parameter-state work can preserve or improve exposure on this H20 (`experiments/018/04-analysis.md`; `03-experiment-learnings.md`).
- The limiter at base 002 is a stable generalization improvement without sacrificing the validated 27,950-step data exposure. GC targets conditioning without another forward or stochastic draw; the companion head targets intermediate features but may overconstrain them; complementary Mixup attacks a distinct data-space regularization axis but reduces clean identity exposure.

## Collected Ideas

## Combinations

## Candidate Ideas

### Clean Reference-Ordered Gradient Centralization Retry
**Summary**: Re-run EXP017's unchanged scientific mechanism from the clean EXP002 parent. After every backward, add coupled L2 to all 44 FP32 gradients, subtract each eligible output row's mean from exactly the 16 convolution weights and classifier weight, then use unchanged PyTorch momentum/Nesterov with internal decay disabled. Rebuild the accuracy-blind harness around EXP018's validated fixed-scalar and post-state-baseline pattern; no metric, coefficient, phase gate, stochastic draw, extra forward, or eligibility change is permitted.

**What it targets**: Stable generalization through optimizer conditioning while preserving EXP002's data exposure. The system has memory headroom but needs a mechanism with negligible forward cost and plausible 0.25-point effect (`02-system-understanding.md`).

**Reasoning**: ECCV 2020 and the official CIFAR implementation give exact ordering and directly relevant vision evidence. EXP017 already proved implementation mathematics, RNG neutrality, and excluded-update parity, but failed only because its temporary harness retained device loss tensors; Claude independently classified GC as untested. EXP018 then demonstrated the corrected allocation discipline on the same model/hardware, so this is a procedurally new experiment resolving an unanswered scientific question rather than retrying a failed result.

**Sources**: `knowledge/papers/gradient-centralization.md`; `experiments/017/04-analysis.md`; `experiments/017/04-result-review.md`; `experiments/018/04-analysis.md`; `03-experiment-learnings.md` § Protocol Findings.

**Estimated Effort**: medium.

**Risk Assessment**: Medium scientific risk and low-to-medium throughput risk. BatchNorm may already remove functionally similar common-mode directions, making GC redundant, while 17 per-step reductions can be launch-bound. The harness must avoid repeating EXP017's failure and a complete valid result below 95.33% must finally count as negative evidence for this exact composition.

### Training-Only Middle-Stage Companion Classifier
**Summary**: Attach one small pooled `Linear(128,10)` companion head after the fourth residual block, train it on the exact hard or area-weighted CutMix targets with a fixed modest coefficient that tapers to zero by the 75% clean-tail boundary, and omit it from evaluation. Isolate its initialization RNG and preserve one backbone forward, parent augmentation streams, and the clean final quarter.

**What it targets**: Intermediate representation discriminativeness rather than optimizer conditioning. It uses direct supervised signal to shape the first four blocks without the multi-gate forward overhead that sank EXP009 (`02-system-understanding.md`; `experiments/009/04-analysis.md`).

**Reasoning**: Deeply-Supervised Nets reports CIFAR gains from companion objectives, and one 1,290-parameter head is cheaper than four spatial SE paths. It is orthogonal to CutMix and later SAM/EMA. However, this six-block pre-activation WRN is not known to be gradient-starved, and CutMix targets at an intermediate pooled representation may constrain useful nonlinear features.

**Sources**: `knowledge/papers/deeply-supervised-nets.md`; `experiments/002/04-analysis.md`; `experiments/009/04-analysis.md`; `02-system-understanding.md`.

**Estimated Effort**: medium.

**Risk Assessment**: High scientific and medium implementation risk. Attachment point and coefficient are weakly calibrated, auxiliary supervision may over-regularize a shallow network, and a small pooling/GEMM path could erase a marginal benefit through lower exposure.

### CutMix-Complementary Early Mixup
**Summary**: Preserve every existing CutMix draw and convert a fixed subset of otherwise-clean batches before the 75% boundary into Mixup using a dedicated RNG stream, with the final quarter left fully clean. Use one model forward and exact mixed-target cross-entropy, preregistering the Mixup gate and alpha before any metric access.

**What it targets**: The remaining generalization gap through a data-space regularizer whose input-gradient effect differs from CutMix, while retaining the proven CutMix mechanism and wall-clock schedule (`knowledge/papers/mixed-sample-analysis.md`; `02-system-understanding.md`).

**Reasoning**: Mixed-sample analysis predicts complementary behavior, and CutMix itself supplied EXP002's largest recipe gain. An additive hybrid avoids a substitution test that identifies only the difference between mechanisms. Yet it necessarily reduces clean identity exposure during early training and risks repeating EXP003's lesson that small regularization-balance changes are dominated by variance or over-regularization.

**Sources**: `knowledge/papers/mixed-sample-analysis.md`; `experiments/002/04-analysis.md`; `experiments/003/04-analysis.md`; `03-experiment-learnings.md` § Patterns.

**Estimated Effort**: low-to-medium.

**Risk Assessment**: Medium-high scientific risk and low implementation risk. The hybrid adds two new hyperparameters, may over-regularize alongside drop path, and could lower clean-identity exposure enough to worsen the stable tail even if throughput remains unchanged.

## Review

Claude selected the clean reference-ordered GC retry, scoring it 8/10 for evidence and 6/10 for impact versus 4/5 for complementary Mixup and 3/5 for the companion classifier (`01-idea-review.md`). I adopted its central concern that prior procedural investment does not establish effect size: the primary ECCV paper is now distilled in `knowledge/papers/gradient-centralization.md`, showing ten-run CIFAR-100 SGDM gains of +0.37 to +1.95 points across five architectures, +0.91 for ResNet-50, and reported ~0.85% epoch overhead. These results make an effect above 0.10 plausible but do not guarantee transfer to this saturated CIFAR-10 WRN/CutMix recipe.

I also adopted the requirement to reuse EXP018's preflight structure literally: fixed device-scalar finiteness, allocation baselines only after persistent state, identical parent/candidate charged regions, calibrated relative dispersion, and no novel accuracy-blind gate without necessity. The installed Torch 2.9.1 has no `_foreach_mean`, so the reviewer's proposed fused foreach reduction is infeasible; planning must retain 17 explicit row reductions, use one heterogeneous broadcast `torch._foreach_sub_`, and profile the real launch cost rather than claiming fusion. A complete valid result below 95.33% remains binding negative evidence. Any local pass must be treated as single-seed evidence before composition onto EXP011.

## Idea Evaluation

Adopt the reviewer verdict. GC is the only finalist with exact mechanism correctness already demonstrated, a diagnosed and now-correctable procedural failure, official quantitative CIFAR evidence, no extra forward, and durable value from either outcome. The companion head's evidence is architecture-mismatched and its mixed intermediate targets unvalidated; the Mixup hybrid raises total mixed-batch dose along the same regularization axis that EXP003 failed to confirm and needs a sweep rather than a blind two-parameter shot.

## Chosen Idea
**Selected**: Clean Reference-Ordered Gradient Centralization Retry

**Why this idea**:
It directly resolves the only high-confidence unanswered mechanism at EXP002. EXP017 proved the exact official-order update but produced no metric; EXP018 proved the corrected fixed-scalar/post-state allocation methodology; and the ECCV paper reports multi-architecture CIFAR gains large enough to clear this goal's local gate in less saturated regimes. The experiment is coefficient-free and preserves every parent stochastic/data path, making a clean result unusually interpretable.

**Hypothesis**:
Full-run official-order GC on all 16 convolution weights plus the classifier weight will improve conditioning without meaningful exposure loss, producing a complete valid `best_test_acc >=95.33%` from EXP002. A result at least 95.53% with a final-16 plateau near the best is stronger but still single-seed composability evidence; 95.61% matches the global best and 95.71% clears it. A complete valid result below 95.33% definitively rejects this exact GC composition, while removed-energy fraction and achieved dose distinguish BN redundancy from an underexecuted mechanism.
