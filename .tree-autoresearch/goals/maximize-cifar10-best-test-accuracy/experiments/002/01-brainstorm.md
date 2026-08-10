# Brainstorm EXP-002
**Created**: 2026-08-05

## Web Search & Literature Review

- **A Unified Analysis of Mixed Sample Data Augmentation** (`knowledge/papers/mixed-sample-analysis.md`)
  Mixup and CutMix act as efficient but distinct input-gradient regularizers, supporting a one-forward-pass mixed-sample intervention on the winning WRN.
- **Time Matters in Regularizing Deep Networks** (`knowledge/papers/time-matters-regularization.md`)
  Augmentation and weight-decay effects are strongest during an early critical period and can be relaxed late without losing their generalization benefit.
- **Using Mixup as a Regularizer Can Surprisingly Improve Accuracy & OOD Robustness** (`experiments/001/papers/regmixup.md`)
  Retaining clean-example supervision alongside mixed examples is useful; probabilistically interleaving clean and CutMix batches captures that principle without RegMixup's second forward pass.

## Experimental History Review

- The lineage is BASE 91.51% -> EXP-001 94.62%. EXP-001 replaced the baseline with a time-scheduled BF16 PreAct WRN-16-4 and is the global-best tip.
- EXP-001 has no failed children. It completed 148 epochs with late smoothed training loss near zero, best accuracy 94.62%, final accuracy 94.52%, and only 1,178.9 MiB peak VRAM.
- The validated architecture and time schedule are not saturated by compute or memory. The immediate limiter is generalization: the model fits training extremely well while test accuracy remains 5.38 points below perfect classification.
- Goal-wide memory says to preserve charged-time schedules across throughput changes and prioritize generalization before more capacity.
- Untried gaps are mixed-sample augmentation, checkpoint averaging, stronger image corruption, and ablation of existing stochastic depth.

## Collected Ideas

Quick pass - candidate approaches are recorded directly below.

## Combinations

Quick pass - no combinations were developed beyond the bounded candidate recipes.

## Candidate Ideas

### Front-Loaded Probabilistic CutMix
**Summary**: Keep the complete EXP-001 architecture, optimizer, stochastic depth, BF16 path, and time-based LR schedule. During the first 75% of charged training time, apply CutMix to each batch with probability 0.5 using `lambda ~ Beta(1, 1)` implemented as a uniform draw, one shared on-device rectangle, an area-corrected label weight, and two-term cross-entropy on paired labels. Copy source pixels safely before assignment, record mixed/eligible batch counts, use ordinary clean batches otherwise, and use exclusively clean batches during the final 25%. This adds no second forward pass and no dependency.

**What it targets**: EXP-001's near-zero late training loss indicates overfitting/generalization, not insufficient optimizer exposure. CutMix broadens local input/label support while the clean late phase preserves direct fitting during low-LR refinement.

**Reasoning**: Mixed-sample analysis supports CutMix as a low-arithmetic-cost input-gradient regularizer, and critical-period evidence supports applying it early then removing it late. The 0.5 probability, `Beta(1,1)`, and 75% cutoff are fixed hypotheses rather than evidence-backed optima. Alternating clean and mixed batches is motivated by RegMixup's clean-supervision result but is not equivalent to its simultaneous objective; the one-forward-pass compromise preserves the fixed compute budget.

**Sources**: `knowledge/papers/mixed-sample-analysis.md`; `knowledge/papers/time-matters-regularization.md`; `experiments/001/papers/regmixup.md`; EXP-001 `04-analysis.md`.

**Estimated Effort**: low

**Risk Assessment**: CutMix may underfit or interact redundantly with stochastic depth. A 0.5 gate, unmodified weight decay, and a fully clean final quarter limit the risk. Rectangle/lambda mismatches are the main code-integrity hazard.

### Sparse EMA Checkpoint Smoothing
**Summary**: Preserve the EXP-001 training recipe and maintain a detached EMA model after 20% time progress, updating every 10 optimizer steps with an effective decay of 0.99 per update. Copy online BatchNorm buffers before each epoch-end evaluation and evaluate only EMA after activation to respect the one-validation-per-epoch rule.

**What it targets**: EXP-001's best-to-final gap was 0.10 points. EMA targets checkpoint noise and may make the late low-LR solution more stable without changing examples, capacity, or optimization.

**Reasoning**: The mechanism is cheap on a 2.75M-parameter model and memory headroom is ample. It is highly isolated, but its upside is naturally bounded by the small observed best/final gap and it lacks direct support in the current paper packet.

**Sources**: EXP-001 `04-analysis.md` and `03-execute.md`.

**Estimated Effort**: low

**Risk Assessment**: EMA can lag genuine late improvements, and online checkpoints become invisible after activation because evaluating both would violate the cadence constraint. Incorrect BatchNorm-buffer handling can erase any benefit.

### Early Cutout with Light Label Smoothing
**Summary**: Add a single random square mask to training images during the first 75% of charged time and use cross-entropy label smoothing 0.05 throughout. Keep all EXP-001 architecture, optimizer, schedule, and drop-path settings unchanged.

**What it targets**: Cutout increases occlusion invariance without mixed labels, while light label smoothing discourages the extreme confidence implied by EXP-001's near-zero training loss.

**Reasoning**: This recipe is simpler than mixed-sample target handling and keeps one hard target per image. It is orthogonal to stochastic depth, but the current high-signal sources support mixed-sample augmentation more directly than this combination.

**Sources**: EXP-001 `04-analysis.md`; `knowledge/papers/time-matters-regularization.md` for early regularization timing.

**Estimated Effort**: low

**Risk Assessment**: Two regularizers are bundled, their interaction is not directly evidenced here, and masking plus label smoothing may over-regularize the short fixed-budget run.

## Review

The fallback adversarial review (`01-idea-review.md`) selected CutMix and found no hard-constraint or reward-hacking issue. Its actionable concerns were adopted: the exact gate/alpha/cutoff are described as hypotheses, not optima; RegMixup is motivation rather than an equivalence claim; the implementation uses one shared on-device rectangle, safe source copying, clipped-area lambda correction, and one forward pass; and the run logs realized mixed-batch exposure. No hyperparameter changes or reruns may be selected from test accuracy.

## Idea Evaluation

Front-Loaded Probabilistic CutMix scored 8/10 for both evidence and potential impact because it directly targets EXP-001's measured generalization gap with relevant external support and little arithmetic overhead. EMA scored 4/10 on both criteria due weak evidence, small expected upside, and BatchNorm/model-selection hazards. Cutout plus label smoothing scored 4/10 evidence and 6/10 impact because it bundled underspecified, less-supported regularizers. The reviewer's pick is adopted.

## Chosen Idea
**Selected**: Front-Loaded Probabilistic CutMix

**Why this idea**:
It is the most direct next intervention on the 94.62% parent: the winning model already fits training nearly perfectly, while CutMix adds input-level regularization without discarding the validated architecture, schedule, or throughput regime. A probabilistic early phase preserves ordinary clean batches, and a fully clean final quarter preserves low-LR refinement.

**Hypothesis**:
Adding fixed `p=0.5`, `Beta(1,1)` CutMix during the first 75% of charged training time to the unchanged EXP-001 recipe will complete within 600 total seconds and achieve `best_test_acc >= 94.72%`, because mixed spatial supervision will reduce overfitting while clean late training preserves final convergence.
