# Brainstorm EXP-033
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **Existing CIFAR regularization knowledge** (`knowledge/papers/cutout-cifar-regularization.md`)
  Cutout-style masking can help CIFAR recipes generally, but local cutout variants have already over-regularized this fixed-budget recipe.
- **PyTorch EMA Weight Averaging** (`knowledge/references/pytorch-ema-averaging.md`)
  Weight averaging remains a possible late-stability mechanism, but local attempts show implementation risk and snapshot-staleness risk.
- **Torchvision RandomCrop padding reference** (`knowledge/references/torchvision-randomcrop-padding.md`)
  Reflection padding is now the validated crop-boundary anchor after EXP-029 and EXP-031.
- **EXP-032 report** (`reports/exp-report-032.md`)
  Mild `label_smoothing=0.05` preserved throughput and raised the reflection anchor to 93.70%, making label-smoothing value the most local newly opened tuning axis.

## Experimental History Review

- Current baseline is EXP-032 at `best_test_acc=93.70%`; under the goal's +0.10 percentage-point rule, EXP-033 must reach at least `93.80%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- EXP-032 is the first successful non-architecture improvement after reflection padding. It suggests low-overhead confidence regularization can improve late stability without spending the step budget.
- Schedule-only second LR drops are a high-importance failed family, and width beyond 28/56/112 is a high-importance capacity failure. Candidate ideas should avoid isolated second-drop tuning and further width increases.
- Cutout masking, stronger combined regularization, batch size 96, higher momentum, lower weight decay, no-decay BN/bias, projection shortcuts, and zero-gamma initialization all underperformed the current family.
- Remaining plausible gaps are small local tuning around the successful label-smoothing regularizer, carefully bounded late-stability mechanisms, or very mild stochasticity changes that preserve milestone reachability.

## Candidate Ideas

### 1. Lower Label Smoothing to 0.03
**Summary**: Preserve the full EXP-032 anchor and change only the loss-call smoothing value from `label_smoothing=0.05` to `label_smoothing=0.03`.

**Reasoning**: EXP-032 shows that mild label smoothing is beneficial, but 0.05 may be slightly stronger than necessary for top-1 accuracy. A smaller value could keep the late-stability benefit while reducing underconfidence or slowed class separation, potentially improving the best epoch. This is the narrowest continuation of the successful experiment: one scalar change, no throughput cost, no evaluation change, and clean attribution.

**Sources**: `reports/exp-report-032.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Patterns; current `train.py`.

**Estimated Effort**: low

**Risk Assessment**: The benefit from 0.05 may rely on the full regularization strength, so 0.03 could regress toward the 93.58 reflection-only baseline or land in the noise band below 93.80. Failure should be a clean no-improvement run.

### 2. Raise Label Smoothing to 0.08
**Summary**: Preserve the full EXP-032 anchor and increase only the smoothing value from `0.05` to `0.08`.

**Reasoning**: EXP-032's final accuracy stayed close to its best accuracy, suggesting confidence regularization improved late stability. A slightly stronger smoothing value might further reduce late overconfident updates and lift the post-drop plateau. This tests the opposite side of the same newly validated axis.

**Sources**: `reports/exp-report-032.md`; `brainstorm/brainstorm-032.md`; current `train.py`.

**Estimated Effort**: low

**Risk Assessment**: Stronger smoothing can reduce top-1 peak accuracy and resembles the failed strong-regularization direction from EXP-000, although EXP-000 combined multiple disruptive changes. If 0.08 is too strong, it should underfit cleanly without invalidating the run.

### 3. Mild Batch Size 112 on Label-Smoothed Anchor
**Summary**: Preserve reflection padding, `label_smoothing=0.05`, architecture, optimizer, and the 21k first LR drop, but reduce `BATCH_SIZE` from 128 to 112.

**Reasoning**: The current anchor may benefit from a small increase in gradient noise after label smoothing. EXP-025 showed batch size 96 was too aggressive and lost too much useful budget, but 112 is a milder stochasticity change that may preserve enough step throughput and still hit the first LR drop.

**Sources**: `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` rows EXP-025 and EXP-032; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches.

**Estimated Effort**: low

**Risk Assessment**: This directly neighbors a known failed family. It changes both stochasticity and throughput, so attribution is less clean than label-smoothing value tuning. If throughput drops enough to reduce post-drop refinement, it will likely miss the 93.80 threshold.

## Idea Evaluation

The strongest evidence is for continuing along the label-smoothing axis because EXP-032 just showed that this intervention class can produce a valid improvement while preserving throughput and all hard constraints. Between the two scalar probes, lowering to 0.03 is the better first test: it asks whether the beneficial regularization can be kept with less top-1 suppression, and it avoids moving toward the stronger-regularization region that has failed in cutout and combined-regularization experiments. Raising to 0.08 is still plausible but has higher underfitting risk.

Batch size 112 is a reasonable future candidate, but it has weaker evidence because batch size 96 already failed by losing useful budget. It also changes two coupled mechanisms at once: gradient noise and step/image throughput. A label-smoothing scalar probe is easier to interpret and safer operationally.

EXP-033 should therefore test `label_smoothing=0.03` on the full current anchor. If it fails, the next brainstorm can try the stronger-smoothing side or move to a tightly controlled late-stability mechanism.

## Chosen Idea
**Selected**: Lower Label Smoothing to 0.03

**Why this idea**:
It is the most local exploitation of the new successful EXP-032 pattern. It preserves all validated anchor choices while testing whether the label-smoothing benefit peaks at a slightly milder value than 0.05.

**Hypothesis**:
Changing the loss to `label_smoothing=0.03` will retain the late-stability benefit of EXP-032 while improving peak class separation enough to raise `best_test_acc` from 93.70% to at least 93.80%.
