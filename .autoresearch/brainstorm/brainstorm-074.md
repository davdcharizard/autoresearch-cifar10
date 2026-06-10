# Brainstorm EXP-074
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix regularization** (`knowledge/papers/cutmix-regularization.md`)
  CutMix replaces a rectangular region and mixes labels by area. The local implementation already follows this pattern and is the current best mechanism.
- **Mixup regularization background** (`knowledge/papers/mixup-beyond-erm.md`)
  Mixed-label training intentionally softens targets. Prior direct mixup variants are below anchor, but they motivate asking whether additional endpoint label smoothing inside mixed losses can over-soften supervision.
- **Probabilistic CutMix anchor report** (`reports/exp-report-064.md`)
  Static `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and endpoint smoothing 0.05 reached the current 94.11% baseline.
- **Recent CutMix schedule reports** (`reports/exp-report-069.md`, `reports/exp-report-073.md`)
  Post-drop tapering and early hard warmup both missed the 94.21% threshold, so temporal CutMix weakening is not the strongest next lever.

## Experimental History Review

- Current best remains EXP-064: probabilistic CutMix reached `best_test_acc=94.11%`; improvement now requires `best_test_acc >= 94.21%`.
- CutMix strength is locally bracketed: `p=0.25`, `p=0.75`, `alpha=0.5`, and `alpha=2.0` all failed to clear the threshold. Keep `CUTMIX_PROB=0.5` and `CUTMIX_ALPHA=1.0`.
- CutMix timing is now also weakly bracketed: post-drop tapering peaked at 93.73%, and a 2000-step early clean warmup peaked at 94.14%, so preserving static CutMix exposure is preferable.
- Fan-out Conv2d initialization reached 94.16% but stayed inside the noise band. It is compatible but likely too small alone.
- Label-smoothing deviations are a recurring failed family, but prior failures changed global clean-label behavior or direct mixup behavior. CutMix endpoint smoothing is a narrower untested interaction because clean batches can keep `label_smoothing=0.05`.
- Recurring failures discourage isolated policy augmentation, direct mixup, cutout, SE, batch-size changes, scalar LR changes, no-decay parameter groups, weight averaging, and more schedule-only second-drop work.

## Candidate Ideas

### 1. CutMix Endpoint Hard Labels
**Summary**: Keep clean batches at `label_smoothing=0.05`, but use `label_smoothing=0.0` for the two endpoint cross-entropy terms inside CutMix batches.

**Reasoning**: CutMix already softens supervision by mixing two labels with area-adjusted lambda. Applying endpoint smoothing to both targets may compound target softness and reduce final fit. Unlike prior global label-smoothing deviations, this preserves the validated clean-batch loss and changes only the mixed-batch endpoint losses. It also preserves the validated static CutMix exposure, `p=0.5`, `alpha=1.0`, architecture, schedule, batch size, and transforms.

**Sources**: `reports/exp-report-064.md`, `reports/exp-report-060.md`, `knowledge/papers/cutmix-regularization.md`, `knowledge/papers/mixup-beyond-erm.md`, goal learnings for label smoothing and CutMix brackets.

**Estimated Effort**: low

**Risk Assessment**: The label-smoothing family has repeated failures, and direct mixup without endpoint smoothing did not help. The narrower CutMix-only scope gives a clean attribution path, but this can easily regress if endpoint smoothing is part of what made CutMix robust.

### 2. Fan-Out Conv Initialization Plus Static CutMix Anchor
**Summary**: Reapply EXP-072's Conv2d fan-out ReLU Kaiming initialization while leaving Linear initialization and all CutMix-anchor settings unchanged.

**Reasoning**: EXP-072 reached 94.16%, the best no-improvement result so far. It is compatible with the current anchor and does not weaken the post-drop CutMix behavior. A repeat would test whether the near-miss is stable, but the goal's +0.10pp guard means isolated initialization is unlikely to clear the threshold unless the previous run under-sampled its true effect.

**Sources**: `reports/exp-report-072.md`, `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: This risks chasing noise because the isolated result was only +0.05pp over baseline. It should be deprioritized unless combined with another distinct mechanism or used as a reproducibility check, neither of which is the strongest next single experiment.

### 3. Shorter CutMix Warmup Probe
**Summary**: Use a shorter clean warmup, such as 500 or 1000 updates, before enabling the same static CutMix branch.

**Reasoning**: EXP-073's 2000-step warmup was too weak to clear the threshold but did not collapse the recipe. A shorter warmup might remove only the most unstable first updates while restoring CutMix earlier. This is a narrower temporal variant than EXP-073.

**Sources**: `reports/exp-report-073.md`, `reports/exp-report-069.md`, goal learnings for CutMix timing.

**Estimated Effort**: low

**Risk Assessment**: The 2000-step warmup already suggests timing changes have small effect sizes. A shorter warmup is likely even smaller and risks another sub-threshold result without teaching much.

## Idea Evaluation

CutMix Endpoint Hard Labels is the strongest candidate because it targets a distinct unresolved interaction inside the successful regional-mixing mechanism. It does not retry static CutMix strength brackets, does not weaken post-drop CutMix exposure, and does not change the clean-batch label smoothing that prior experiments established as useful. The mechanism is concrete: mixed labels are already soft, so removing endpoint smoothing may recover sharper class supervision only on mixed batches.

Fan-out initialization has the best prior numeric near-miss, but as an isolated repeat it is mostly a noise check. It may become useful as a coupled ingredient later, but the current experiment should test an unresolved mechanism rather than repeat a sub-threshold one.

Shorter warmup is lower priority because EXP-073 already tested the temporal-clean-warmup family and produced only a +0.03pp signal. A shorter version might be less harmful, but the expected impact is too small and the learning value is limited.

## Chosen Idea
**Selected**: CutMix Endpoint Hard Labels

**Why this idea**:
It is the cleanest remaining CutMix-internal probe that preserves the validated anchor while testing an unbracketed source of possible over-regularization. It changes only the loss applied to CutMix endpoint labels, leaving the clean-batch path and all successful structural settings intact.

**Hypothesis**:
If CutMix's area-weighted label mixture plus endpoint label smoothing over-softens mixed-batch targets, then setting CutMix endpoint label smoothing to 0.0 while keeping clean batches at 0.05 will improve `best_test_acc` from 94.11% to at least 94.21%.
