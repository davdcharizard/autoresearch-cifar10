# Brainstorm EXP-066
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features** (`knowledge/papers/cutmix-regularization.md`)
  CutMix regionally replaces image patches and mixes labels by area, retaining real pixels while adding localized regularization; this directly supports bracketing the successful EXP-064 CutMix recipe.
- **mixup: Beyond Empirical Risk Minimization** (`knowledge/papers/mixup-beyond-erm.md`)
  Whole-image mixup motivates mixed-label regularization but prior repo results show global interpolation underperforms here, strengthening the case for regional rather than global mixing.

## Experimental History Review

- Current best is EXP-064 at `best_test_acc=94.11%`, commit `1119ff8`, using `CUTMIX_ALPHA=1.0`, `CUTMIX_PROB=0.5`, and `CUTMIX_LABEL_SMOOTHING=0.05`.
- The active goal now requires at least +0.10 percentage points over baseline, so EXP-066 must reach `best_test_acc >= 94.21%` to count as an improvement.
- EXP-065 lowered `CUTMIX_PROB` to 0.25 and reached 94.09%, a clean near-miss below the 94.11 baseline. More training steps did not compensate, so less CutMix exposure appears slightly weaker for the best-checkpoint metric.
- Direct mixup variants are a medium-importance failed family, while CutMix is a medium-importance validated pattern. This indicates the regional replacement mechanism should be bracketed before returning to global interpolation or unrelated augmentation.
- Label-smoothing deviations are a repeated failed family, so this brainstorm avoids changing endpoint smoothing inside the next CutMix bracket.

## Candidate Ideas

### 1. CutMix Probability 0.75
**Summary**: Increase `CUTMIX_PROB` from 0.5 to 0.75 while keeping `CUTMIX_ALPHA=1.0`, `CUTMIX_LABEL_SMOOTHING=0.05`, the ResNet-20 `(28,56,112)` architecture, `WEIGHT_DECAY=2e-4`, the 21k-step first LR drop, reflection padding, channels-last compile path, and validation cadence unchanged.

**Reasoning**: EXP-064 proved the regional mixing mechanism can clear the threshold, while EXP-065 showed reducing frequency to 0.25 slightly weakens the peak. The direct opposite bracket tests whether the anchor is under-regularized by regional mixing rather than over-regularized. It is a one-scalar diagnostic with a clean interpretation.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-064.md`; `reports/exp-report-065.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: low

**Risk Assessment**: Stronger CutMix may add too much label noise or reduce clean-image fitting, producing a no-improvement. The worst likely outcome is a valid result below threshold; code risk is minimal because it reuses the existing CutMix path.

### 2. CutMix Alpha 0.5
**Summary**: Keep `CUTMIX_PROB=0.5` and lower `CUTMIX_ALPHA` from 1.0 to 0.5 to alter the patch-area distribution while preserving the successful application frequency.

**Reasoning**: The CutMix paper frames lambda sampling as a core part of the method. Changing alpha may find a better balance between small localized patches and larger regional replacements without touching the rest of the recipe. It is distinct from EXP-065 because frequency remains at the successful value.

**Sources**: `knowledge/papers/cutmix-regularization.md`; `reports/exp-report-064.md`; `reports/exp-report-065.md`.

**Estimated Effort**: low

**Risk Assessment**: A different beta distribution may produce more extreme boxes that destabilize the tuned anchor. The interpretation is slightly less direct than the probability bracket because alpha changes both patch size variance and label mixing strength.

### 3. Post-Drop CutMix Probability Ramp
**Summary**: Keep early training at `CUTMIX_PROB=0.5`, then reduce or disable CutMix after the first LR drop to allow lower-LR refinement on cleaner labels.

**Reasoning**: EXP-064 peaked late but ended with final accuracy well below its best checkpoint, suggesting late dynamics may be sensitive to mixed-label regularization. A schedule could preserve early CutMix invariance and sharpen later. This is a coupled regularization schedule rather than a static smoothing change.

**Sources**: `reports/exp-report-064.md`; `reports/exp-report-065.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md`.

**Estimated Effort**: medium

**Risk Assessment**: Label-smoothing and schedule-only deviations have repeatedly failed, so a CutMix schedule may simply remove useful regional regularization during refinement. It also adds more logic than a one-scalar bracket and is harder to interpret cleanly.

## Idea Evaluation

The strongest evidence supports staying inside the CutMix family: EXP-064 is the only recent intervention to create a new baseline, and the knowledge base supports regional mixing as a distinct mechanism from failed Cutout and direct mixup variants. Among the candidates, `CUTMIX_PROB=0.75` has the clearest experimental logic because EXP-065 already tested the lower-frequency side and missed by only 0.02 points. Testing the higher-frequency side completes the nearest bracket around the successful `p=0.5` anchor.

`CUTMIX_ALPHA=0.5` is also defensible but should come after the probability bracket because alpha changes the area distribution and label weights simultaneously. The post-drop schedule is more speculative: it targets EXP-064's late peak/final gap, but schedule-only and label-smoothing changes are recurring failure modes, and the added conditional logic makes attribution weaker.

The +0.10pp threshold means small numerical wins are no longer enough. The next idea should therefore have a plausible mechanism for more than a tiny noise-level move; stronger CutMix exposure has the highest chance among low-risk brackets because the current best came from this exact mechanism and the weaker side did not improve.

## Chosen Idea
**Selected**: CutMix Probability 0.75

**Why this idea**:
This is the cleanest remaining local bracket around the validated EXP-064 CutMix recipe. It directly tests whether the successful regional-mixing anchor benefits from more frequent CutMix after the lower-frequency bracket underperformed, and it keeps the diff limited to one hyperparameter in `train.py`.

**Hypothesis**:
If the EXP-064 anchor is still slightly under-regularized by regional mixing, increasing `CUTMIX_PROB` to 0.75 will improve the late best checkpoint enough to reach at least `94.21%`; if stronger mixed-label pressure is too noisy, the run will finish as a valid no-improvement and establish `p=0.5` as the local probability optimum.
