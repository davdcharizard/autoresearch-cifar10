# Brainstorm EXP-073
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  CutMix is already the validated anchor mechanism in this repo: regional patch replacement adds label-space regularization without the global interpolation behavior that made mixup variants weak.
- **Mixup regularization background** (`knowledge/papers/mixup-beyond-erm.md`)
  Mixed-example methods can improve generalization but add training distribution noise; under this fixed 300s budget, any schedule should preserve first-LR-drop timing and step coverage.
- **Probabilistic CutMix baseline report** (`reports/exp-report-064.md`)
  Static `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and endpoint smoothing 0.05 is the current best recipe at 94.11%, so new ideas should preserve the validated post-drop CutMix anchor.

## Experimental History Review

- Current best remains EXP-064: probabilistic CutMix reached `best_test_acc=94.11%`; the active improvement threshold is `94.21%`.
- Static CutMix strength is locally bracketed: `p=0.25` reached 94.09%, `p=0.75` tied 94.11% but missed the noise guard, `alpha=0.5` reached 94.07%, and `alpha=2.0` reached 94.00%.
- Post-drop CutMix weakening failed: EXP-069 tapering `p=0.5 -> 0.25` after step 21000 peaked at 93.73%, so the post-drop refinement phase should keep the static `p=0.5` anchor.
- Fan-out Conv2d initialization was a clean near miss: EXP-072 reached 94.16% but did not clear the 94.21% threshold, so isolated initialization changes are probably sub-threshold.
- Strong repeated failures discourage isolated policy augmentation, label-smoothing deviations, direct mixup, cutout, SE, batch-size changes, weight averaging, scalar LR changes, BN/bias decay exceptions, and further simple schedule-only second-drop work.
- A short early CutMix warmup remains untested and is distinct from EXP-069: it would reduce mixed-label noise only during the initial representation phase while preserving full CutMix during the validated post-drop refinement window.

## Candidate Ideas

### 1. Short Clean Warmup Before CutMix
**Summary**: Train with the existing clean label-smoothed loss for a short initial warmup, then enable the validated `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, `CUTMIX_LABEL_SMOOTHING=0.05` recipe for the rest of the run.

**Reasoning**: Static CutMix is validated, but mixed-label patches from the first updates may slow early feature formation. EXP-069 shows post-drop CutMix weakening is harmful, so the safer temporal variant is the opposite: preserve the full post-drop CutMix anchor and only remove CutMix during the earliest high-instability phase. This targets a specific gap without touching architecture, optimizer, schedule, batch size, or evaluation.

**Sources**: `reports/exp-report-064.md`, `reports/exp-report-069.md`, `reports/exp-report-072.md`, `knowledge/papers/cutmix-regularization.md`, goal learnings for CutMix brackets and post-drop taper.

**Estimated Effort**: low

**Risk Assessment**: The warmup may simply reduce useful regularization and underperform the static anchor. If the warmup window is too long, it could resemble weakening CutMix and regress like EXP-069. Worst case is a valid no-improvement run with clean attribution.

### 2. CutMix Probability Linear Ramp
**Summary**: Instead of a hard off/on warmup, linearly ramp the effective CutMix probability from 0.0 to 0.5 over an early step window, then keep `p=0.5` for the remainder.

**Reasoning**: A ramp is a smoother version of early warmup and could avoid an abrupt distribution shift. It directly tests whether early mixed-label noise should be introduced gradually while preserving the validated late CutMix behavior.

**Sources**: `reports/exp-report-064.md`, `reports/exp-report-069.md`, CutMix probability bracket EXP-065/066 in the experiment index.

**Estimated Effort**: low

**Risk Assessment**: This adds more schedule code than the binary warmup and introduces an extra degree of freedom. If the hard warmup mechanism is wrong, the ramp likely also fails; as a first test, the hard warmup gives cleaner attribution.

### 3. CutMix Endpoint Label-Smoothing Probe
**Summary**: Keep clean-batch label smoothing at 0.05 and static `p=0.5`, but reduce label smoothing only inside the two CutMix endpoint cross-entropy losses.

**Reasoning**: CutMix labels already soften targets through lambda weighting, so endpoint label smoothing might over-soften mixed batches. This is technically distinct from global label-smoothing deviations because clean batches keep the validated smoothing anchor.

**Sources**: `reports/exp-report-033.md`, `reports/exp-report-057.md`, `reports/exp-report-064.md`, goal learnings for label-smoothing failures.

**Estimated Effort**: low

**Risk Assessment**: Label-smoothing deviations are a high-importance failed family, and EXP-057 showed late hard-label sharpening overfits. This probe is more targeted than prior failures but lower confidence than early CutMix warmup.

## Idea Evaluation

Short clean warmup before CutMix has the strongest mechanism-to-history fit. It preserves the validated static CutMix recipe during the post-drop window, avoids the failed post-drop taper direction, and tests an untried point in the temporal regularization space. It also has a simple implementation and clean failure mode.

The linear ramp is closely related and may be smoother, but it is a less interpretable first probe because the ramp length and ramp shape add degrees of freedom. It is better as a follow-up if a binary warmup shows a useful signal or a near miss.

The CutMix endpoint smoothing probe is interesting because mixed labels and endpoint smoothing may compound, but the goal learnings strongly warn against label-smoothing deviations. It should be deprioritized until stronger evidence suggests the mixed-batch loss itself is the bottleneck.

## Chosen Idea
**Selected**: Short Clean Warmup Before CutMix

**Why this idea**:
It is the clearest remaining targeted CutMix interaction that has not been bracketed and does not retry a recurring failed family. It changes only the early availability of the validated regional regularizer, while preserving the current architecture, optimizer, schedule, label smoothing, unit-std normalization, and full post-drop CutMix anchor.

**Hypothesis**:
If early CutMix injects too much mixed-label spatial noise before stable low-level features form, then a short clean warmup followed by the validated static CutMix recipe will improve `best_test_acc` from 94.11% to at least 94.21% without harming the step-21000 LR drop.
