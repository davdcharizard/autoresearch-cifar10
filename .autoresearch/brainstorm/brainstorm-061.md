# Brainstorm EXP-061
**Created**: 2026-06-09
**Goal**: goals/maximize-cifar10-best-test-accuracy.md

## Web Search & Literature Review

- **Existing knowledge base** (`knowledge/README.md`)
  The saved knowledge now covers throughput, crop padding, width scaling, mixup, RandAugment, cosine schedules, stochastic depth, SE attention, and shortcut/downsampling tweaks. No new external search was needed for this loop because the next gap is directly identified by EXP-060 and the accumulated failure map.
- **Wide Residual Networks** (`knowledge/papers/wide-residual-networks.md`)
  Wider compact residual networks can improve CIFAR accuracy, but this project has already found that width beyond 28/56/112 loses the fixed-budget tradeoff without additional design changes.
- **Squeeze-and-Excitation Networks** (`knowledge/papers/squeeze-and-excitation-networks.md`)
  SE provides lightweight channel recalibration, but EXP-058 showed all-block SE underperforms here; any future SE test should be narrower and lower overhead.

## Experimental History Review

- Current best remains EXP-038 at `best_test_acc=93.97%`; EXP-061 must reach at least `94.07%` to count as an improvement under the explicit +0.10 percentage-point threshold.
- The validated anchor is `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, full-run label smoothing 0.05, FP32 compile, channels-last, and once-per-epoch validation.
- EXP-060 closed the most plausible mixup/smoothing near-miss: removing endpoint label smoothing from mild mixup still peaked at 93.81%, so direct label-interpolation variants are no longer the lead.
- Repeated failed families now include schedule-only second drops, weight averaging, batch-size deviations, label-smoothing deviations, cosine schedules, mild mixup, BN/bias no-decay, LR scalar deviations, residual branch scaling, isolated augmentation, shortcut tweaks, and broad SE attention.
- Untried gaps that still respect `train.py`-only scope are now narrower: head-only regularization, very localized architecture changes, or a more structural compact residual redesign. The recent evidence favors the smallest untried intervention first because broader regularizers and architectural changes have mostly lost the anchor tradeoff.

## Candidate Ideas

### 1. Final Classifier Dropout
**Summary**: Add a small dropout layer immediately before the final `fc` classifier, likely `p=0.1`, active only during training. Preserve the residual body, optimizer, schedule, augmentation, batch size, label smoothing, weight decay, compile path, and evaluation loop. Add a startup print such as `Classifier dropout p: 0.1` for verification.

**Reasoning**: The current anchor has resisted broad residual regularizers, label-space regularizers, and augmentation changes. A final-head dropout is a much narrower regularizer: it perturbs only the pooled feature vector before the classifier, avoiding residual-path undertraining and avoiding target softening. It may reduce late classifier overfit without changing the convolutional representation or step budget meaningfully. EXP-060's next-steps section explicitly identifies this as the most direct remaining low-overhead lever.

**Sources**: `reports/exp-report-060.md`; `goal-learnings/maximize-cifar10-best-test-accuracy.md` entries on failed broad regularizers; current `train.py` head structure (`adaptive_avg_pool2d -> view -> fc`).

**Estimated Effort**: low

**Risk Assessment**: Isolated regularization has a weak recent record, and even small dropout may simply underfit the classifier. The likely failure mode is a valid no-improvement rather than a crash, and the implementation surface is small.

### 2. Stage-3-Only SE Gate
**Summary**: Add SE channel gates only to the three final-stage residual blocks (`layer3`) instead of every block. Keep stages 1 and 2 unchanged and preserve all optimizer, schedule, augmentation, and loss settings.

**Reasoning**: EXP-058 rejected all-block SE, but that result does not fully rule out a narrower final-stage gate. Late channels encode higher-level features, so final-stage recalibration could help classification while avoiding overhead and early-feature disruption from all-block SE. The idea is still architecture-side and has a clearer mechanism than another scalar optimizer bracket.

**Sources**: `knowledge/papers/squeeze-and-excitation-networks.md`; `reports/exp-report-058.md`; goal-learning "SE channel attention underperforms the current block."

**Estimated Effort**: medium

**Risk Assessment**: This remains close to a recent negative result. It may add enough overhead or gating noise to miss the first-drop/late-refinement balance, and implementation requires stage-aware block construction.

### 3. Compact WRN-Style ResNet-14
**Summary**: Reduce depth to two residual blocks per stage (`NUM_BLOCKS=2`, ResNet-14) while increasing widths modestly or preserving 28/56/112. The goal is to trade depth for more completed optimization steps or potentially support a slightly wider-but-faster compact model.

**Reasoning**: The width-scaling history shows compact wider residual networks can work until the 28/56/112 anchor, but further width lost step budget. A shallower variant could test whether the current depth is not the best fixed-budget tradeoff. This is more structural than final-head dropout and may expose a different capacity/throughput frontier.

**Sources**: `knowledge/papers/wide-residual-networks.md`; experiment index EXP-011 through EXP-020; goal-learnings on width scaling and failures beyond 28/56/112.

**Estimated Effort**: medium

**Risk Assessment**: Reducing depth may remove useful representational capacity and regress sharply. Any width adjustment would need schedule calibration, and the result may be hard to attribute in one run.

## Idea Evaluation

Final Classifier Dropout has the lowest implementation risk and targets a still-untried part of the model: the classifier head after global pooling. It is not a repeat of stochastic depth, mixup, label-smoothing changes, or broad augmentation because it leaves the residual representation and labels untouched. Its expected impact is modest, but the current search space has many closed broad levers; a small, localized test is a reasonable next move.

Stage-3-Only SE has a plausible representation mechanism, but it is adjacent to EXP-058's all-block SE miss. It should remain a future option, but after all-block SE reached only 93.71%, a narrower version needs stronger evidence before outranking the lower-overhead head dropout test.

Compact WRN-style restructuring has the largest possible upside because it changes the depth/throughput frontier, but it also has the largest attribution risk. The current anchor already represents a carefully calibrated width/schedule recipe, and prior capacity experiments show that seemingly small architectural moves can lose the fixed-budget balance.

The lead candidate is therefore Final Classifier Dropout. It is the smallest unexplored regularization lever left, has a simple implementation path, and directly tests whether the current anchor's final classifier can benefit from mild feature-level noise without disturbing the rest of the recipe.

## Chosen Idea
**Selected**: Final Classifier Dropout

**Why this idea**:
It is the most targeted untried lever after EXP-060 closed mild mixup variants. It touches only the pooled classifier input, adds minimal overhead, preserves all anchor settings, and has a clear failure mode if the anchor is already optimally regularized.

**Hypothesis**:
Adding `CLASSIFIER_DROPOUT_P = 0.1` before the final `fc` layer will preserve step count and the step-21000 LR drop while reducing classifier overfit enough to raise `best_test_acc` to at least `94.07%`.
