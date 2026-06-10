# Brainstorm EXP-034
**Created**: 2026-06-08
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

<!-- This file is focused on IDEATION only.
     Goal statement, primary metric, direction, hard constraints, and verification criteria
     live in the goal file (see pointer above). Baseline lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     Do not duplicate those fields here — always point to the source of truth. -->

## Web Search & Literature Review

- **SGDR / schedule knowledge** (`knowledge/papers/sgdr-cosine-schedule.md`)
  LR schedule shape can matter under fixed step budgets, but local cosine and second-drop variants have underperformed when they disrupt useful refinement.
- **EXP-032 report** (`reports/exp-report-032.md`)
  `label_smoothing=0.05` is the current successful anchor, reaching 93.70% with preserved throughput and a 21k first LR drop.
- **EXP-033 report** (`reports/exp-report-033.md`)
  Lowering smoothing to 0.03 reached 93.79%, but the +0.09 point gain did not clear the +0.10 threshold, suggesting the smoothing axis is close but not enough alone.

## Experimental History Review

- Current baseline is EXP-032 at `best_test_acc=93.70%`; under the goal's +0.10 percentage-point rule, EXP-034 must reach at least `93.80%`.
- The current anchor is `STAGE_WIDTHS = (28, 56, 112)`, reflected `RandomCrop`, `label_smoothing=0.05`, `BATCH_SIZE = 128`, `LR = 0.1`, `MOMENTUM = 0.9`, `WEIGHT_DECAY = 1e-4`, `LR_MILESTONES = [21000, 64000]`, FP32, channels-last, cuDNN benchmark, and `torch.compile`.
- EXP-033 showed lower smoothing is directionally promising but not large enough to count. The anchor should keep 0.05 until a distinct lever improves it.
- Isolated second LR-drop tuning is a high-importance failed family, so EXP-034 should not move or activate the second 64k milestone.
- First-drop local brackets were explored before reflection padding and label smoothing: 20k was slightly early, 23k was too late, and 21k became the pre-label-smoothing anchor. The addition of label smoothing may shift the best first-drop point slightly later.
- Width beyond 28/56/112, projection shortcuts, zero-gamma initialization, lower weight decay, no-decay BN/bias, higher momentum, cutout, and batch size 96 have all underperformed.

## Candidate Ideas

### 1. Move First LR Drop to 22k on the Label-Smoothed Anchor
**Summary**: Preserve the full EXP-032 anchor and change only `LR_MILESTONES` from `[21000, 64000]` to `[22000, 64000]`.

**Reasoning**: Label smoothing reduces overconfident updates and can slow class separation slightly. The current 21k first drop was calibrated before the successful label-smoothing change; a later 22k drop may give the smoothed model a little more high-LR fitting time before refinement. This is not a retry of the failed second-drop family because the second milestone remains unreachable, and it is not the old 23k pre-label-smoothing test because the regularization and augmentation anchor have changed.

**Sources**: `reports/exp-report-032.md`; `reports/exp-report-033.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Patterns and Failed Approaches; current `train.py`.

**Estimated Effort**: low

**Risk Assessment**: Prior first-drop evidence favors 21k, so 22k may reduce refinement time and miss the threshold. The failure mode should be a clean no-improvement result with preserved throughput and easy attribution.

### 2. Raise Label Smoothing to 0.08
**Summary**: Preserve the current anchor and increase only the smoothing value from `0.05` to `0.08`.

**Reasoning**: EXP-032 benefited from mild confidence regularization, and EXP-033's lower value nearly cleared the next bar. Testing the stronger side would determine whether late stability, rather than sharper class separation, is the better direction.

**Sources**: `reports/exp-report-032.md`; `reports/exp-report-033.md`; `brainstorm/brainstorm-033.md`.

**Estimated Effort**: low

**Risk Assessment**: Stronger smoothing can suppress top-1 peaks and resembles the failed strong-regularization direction from EXP-000, although EXP-000 combined multiple disruptive changes. If 0.08 is too strong, it should underfit cleanly.

### 3. Mild Batch Size 112 on the Label-Smoothed Anchor
**Summary**: Preserve reflection padding, `label_smoothing=0.05`, architecture, optimizer, and schedule, but reduce `BATCH_SIZE` from 128 to 112.

**Reasoning**: A small increase in gradient noise might help the already regularized anchor find a better late peak. EXP-025 showed batch size 96 was too costly, but 112 is a milder variant that may still hit the 21k first drop.

**Sources**: `experiment-indices/maximize-cifar10-best-test-accuracy.tsv` rows EXP-025, EXP-032, and EXP-033; `goal-learnings/maximize-cifar10-best-test-accuracy.md` § Failed Approaches.

**Estimated Effort**: low

**Risk Assessment**: This neighbors a known failed approach and changes both stochasticity and throughput. If step budget drops materially, the run will likely miss the 93.80 threshold.

## Idea Evaluation

The best next probe is the 22k first-drop retune because it is a distinct no-overhead lever that composes with the validated label-smoothing anchor. It has a clear mechanism: 0.05 label smoothing may need slightly more LR 0.1 fitting before the LR 0.01 refinement phase. It also avoids the lower-smoothing micro-probe that just landed in the noise band.

Raising smoothing to 0.08 is simple but moves toward stronger regularization, which has a weaker local track record. Batch size 112 is plausible but less clean because it changes throughput and optimization noise together, and batch size 96 already failed by losing too much useful budget.

EXP-034 should therefore test `LR_MILESTONES = [22000, 64000]` while preserving `label_smoothing=0.05` and all other anchor choices.

## Chosen Idea
**Selected**: Move First LR Drop to 22k on the Label-Smoothed Anchor

**Why this idea**:
It is a low-risk retune of the current successful anchor that tests whether label smoothing shifts the optimal first LR drop later. It preserves throughput, architecture, augmentation, regularization value, and validation cadence.

**Hypothesis**:
Moving the first LR drop from 21000 to 22000 will give the label-smoothed model enough extra high-LR fitting time to raise `best_test_acc` from 93.70% to at least 93.80% without changing step throughput or violating any hard constraints.
