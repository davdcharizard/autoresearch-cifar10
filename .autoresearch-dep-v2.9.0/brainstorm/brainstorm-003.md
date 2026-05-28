# Brainstorm EXP-003
**Created**: 2026-05-27
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

No new external sources needed — the WRN paper (https://arxiv.org/abs/1605.07146) is the primary reference and was already cited in brainstorm-001 and brainstorm-002. The key fact: the WRN paper uses weight_decay=5e-4 and Nesterov momentum for all CIFAR-10 experiments (Tables 4-5, experimental setup section). Our current recipe uses WD=1e-4 and no Nesterov, inherited from the original He-2015 ResNet paper.

## Experimental History Review

Current baseline: **92.92%** (EXP-002, width-2x + TrivialAugmentWide + RandomErasing, 1.07M params, 68 epochs, commit 651d57c).

Trajectory:
- **BASE** (91.72%): ResNet-20, 270K, MultiStepLR, no augmentation
- **EXP-000** (88.79%, no-improvement): Bundled 5 recipe changes — cosine T_max mismatch killed it
- **EXP-001** (92.29%, +0.57pp): Width-2x + wall-clock-fractional schedule
- **EXP-002** (92.92%, +0.63pp): TrivialAugmentWide + RandomErasing on width-2x

Total gain from BASE: +1.20pp. Two successful single-axis experiments in a row.

Remaining untouched axes on the current baseline:
- **Weight decay**: 1e-4 (WRN paper uses 5e-4 for wider models)
- **Nesterov**: Off (WRN paper uses it)
- **Label smoothing**: Off (never cleanly isolated)
- **Architecture**: Width-4x not tried
- **Throughput**: No AMP, batch size unchanged at 128

Goal-learnings patterns to respect:
- Wall-clock-fractional schedule is the validated standard (High Importance)
- Augmentation compounds with capacity (Medium Importance)
- Single-axis changes have been reliable; EXP-000's multi-change bundle failed (Low Importance failed approach)

## Candidate Ideas

### 1. Weight decay 5e-4 (aligning with WRN paper)

**Summary**: Change `WEIGHT_DECAY = 1e-4` to `WEIGHT_DECAY = 5e-4` in the hyperparameters block. One constant change. Everything else unchanged.

**Reasoning**: The WRN paper uses WD=5e-4 for all CIFAR-10 experiments with wider models. Our width-2x model (1.07M params) is in the regime where the original He-2015 WD of 1e-4 may be insufficient to prevent overfitting. With augmentation already in place (EXP-002), stronger WD can compound — both are regularization mechanisms targeting different aspects (L2 norm constraint vs data diversity). The training loss at EXP-002's final epochs was 0.39 (vs 0.02 without augmentation in EXP-001), suggesting the model has room for the WD to tighten the regularization without causing underfitting.

**Sources**: WRN paper (https://arxiv.org/abs/1605.07146, experimental setup), reports/exp-report-002.md § Next Steps #1.

**Estimated Effort**: Very low — single constant change.

**Risk Assessment**: Low. WD=5e-4 is the literature standard for WRN-class models. With augmentation making training harder (training loss 0.39 vs 0.02), the increased WD has plenty of margin before causing underfitting. Worst case: no-improvement if WD=5e-4 is too aggressive given the 68-epoch budget (the WRN paper trains for 200 epochs).

### 2. Nesterov momentum

**Summary**: Add `nesterov=True` to the SGD optimizer call. One keyword argument change. Everything else unchanged.

**Reasoning**: Nesterov momentum is a standard improvement over classical momentum for SGD. The WRN paper uses it. Typical gain on CIFAR-10 is +0.1-0.3pp. It's a "free" change with no risk — the lookahead gradient computation slightly improves convergence quality without changing the training dynamics or throughput.

**Sources**: WRN paper experimental setup, standard SGD-Nesterov literature.

**Estimated Effort**: Very low — adding `nesterov=True`.

**Risk Assessment**: Very low. Nesterov is never harmful in practice. The gain is small but reliable.

### 3. Label smoothing 0.1

**Summary**: Add `label_smoothing=0.1` to the `F.cross_entropy()` call. One keyword argument change. Everything else unchanged.

**Reasoning**: Label smoothing prevents the model from becoming overconfident on training examples, acting as a calibration and regularization technique. It was included in EXP-000's bundle but was confounded with 4 other changes, so its individual effect is unknown. On wider models with augmentation, label smoothing typically adds +0.1-0.3pp. The Inception-v3 paper and subsequent CIFAR-10 literature report consistent small gains.

**Sources**: EXP-000 (included but confounded), Inception-v3 paper, Ruseckas CIFAR-10 tricks blog (cited in brainstorm-000).

**Estimated Effort**: Very low — adding `label_smoothing=0.1`.

**Risk Assessment**: Low. Label smoothing 0.1 is well within the safe range. The main risk is that it changes the loss landscape enough that the EMA training loss reported in the log looks different (higher), which is cosmetic. Combined with augmentation, label smoothing could make training too hard for the 68-epoch budget on the width-2x model, but the effect is typically small.

## Idea Evaluation

All three candidates are single-constant changes with very low effort and low risk. The differentiation is in expected impact:

**Evidence strength**: Candidate 1 (WD=5e-4) has the strongest evidence — it's the exact hyperparameter the WRN paper uses for the exact class of model we're running, and the discrepancy between our 1e-4 and their 5e-4 is a clear gap in recipe alignment. Candidates 2 and 3 have weaker per-change evidence on CIFAR-10 specifically.

**Expected impact**: Candidate 1 is expected to have the largest effect (+0.2-0.5pp) because WD directly addresses the overfitting gap of a 1M-param model, and the WRN paper's results were achieved with this exact setting. Candidates 2 and 3 are smaller effects (+0.1-0.3pp each).

**Mechanism clarity**: Candidate 1 has the clearest mechanism — stronger L2 regularization on a wider model that the WRN paper explicitly designed for. Candidate 2 improves gradient estimation. Candidate 3 reduces overconfidence.

**Single-axis principle**: All three are single-axis changes. Candidate 1 addresses the largest identified gap between our recipe and the WRN paper's.

**Selection**: Candidate 1 (WD=5e-4) is the strongest choice — largest expected impact, strongest evidence, clearest mechanism, and the most direct recipe alignment with the WRN paper that produced the literature anchors guiding this goal.

## Chosen Idea

**Selected**: Candidate 1 — **Weight decay 5e-4**

**Why this idea**: It aligns the recipe with the WRN paper's standard for wider CIFAR-10 models. The 5x increase from 1e-4 to 5e-4 is the exact setting the WRN paper used for all its CIFAR-10 experiments, and our current baseline's training loss of 0.39 (with augmentation) indicates the model has ample margin for stronger L2 regularization without underfitting.

**Hypothesis**: Changing WEIGHT_DECAY from 1e-4 to 5e-4 on the width-2x augmented baseline — while keeping all other settings unchanged — will raise best_test_acc from 92.92% to **93.1-93.5%**, a +0.2-0.6pp gain. The mechanism is stronger L2 regularization matching the WRN paper's recipe for this parameter-count regime, compounding with the existing augmentation. The improvement bar of 93.02% (baseline + 0.1pp) is within the expected range.
