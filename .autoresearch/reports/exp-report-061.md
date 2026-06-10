# Report EXP-061: Final Classifier Dropout
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-061.md
- **Plan**: plans/plan-061.md
- **Log**: logs/exp-log-061.md

## Goal
Maximize CIFAR-10 `best_test_acc` under the fixed evaluation harness and fixed training budget while modifying only `train.py`. The active baseline remains 93.97% at commit `755be2c`; because the goal requires at least +0.10 absolute percentage points to count, EXP-061 needed `best_test_acc >= 94.07%`.

## Idea & Hypothesis
The chosen idea was to add a small training-only dropout layer before the final classifier. This was selected as the narrowest remaining regularization lever after broader target, residual, augmentation, schedule, and attention variants had underperformed. The hypothesis was that `CLASSIFIER_DROPOUT_P = 0.1` could reduce classifier-head overfit without disrupting residual representation learning, step count, or first LR-drop timing enough to clear 94.07%.

## Approach
`train.py` was modified only to add `CLASSIFIER_DROPOUT_P = 0.1`, create `self.classifier_dropout = nn.Dropout(p=CLASSIFIER_DROPOUT_P)` in `ResNet.__init__`, and apply it to the flattened pooled feature vector immediately before `self.fc(out)`. A startup print reported `Classifier dropout p: 0.1`. All anchor settings were preserved: `STAGE_WIDTHS=(28, 56, 112)`, batch size 128, LR 0.1, momentum 0.9, weight decay 2e-4, first LR milestone 21000, reflection crop padding, label smoothing 0.05, compile, channels-last, and once-per-epoch validation.

## Execution
One local foreground run executed on GPU0 with output captured to `run.log`. Startup confirmed CUDA, `ResNet-20 | params: 822,790`, `Classifier dropout p: 0.1`, and `Batches per epoch: 390`. The first LR drop was reached cleanly at `step 21000 ep 54` with `lr: 0.0100`; there were no tracebacks, OOMs, NaNs, or runtime errors. The run completed within the wall-clock limit with numeric final metrics.

## Results
- **Primary metric**: 93.54% (baseline: 93.97%, delta: -0.43pp, -0.46%)
- **Observations**: Pre-drop best reached 88.55% at epoch 53, then post-drop refinement climbed to 93.54% by epoch 78. Accuracy then plateaued below threshold and finished at 93.14%. The run completed 39,806 steps and 103 epochs in 401.8 total seconds, with unchanged 822,790 parameters and 660.4 MB peak VRAM.
- **Analysis**: Final-head dropout preserved attribution quality: it did not change parameter count, batch geometry, or LR timing. The result still trailed the anchor by 0.43pp, suggesting the current recipe is not primarily limited by final-classifier overfit. This also fits the broader pattern that isolated regularizers tend to underfit or soften the fixed-budget anchor rather than improve generalization.
- **Key Learning**: Head-only classifier dropout preserves throughput and LR timing but weakens the current anchor, so final-feature dropout is not a promising isolated lever.

## Verification
- **Conditions**: all passed except the improvement threshold
- **Review Notes**: Results are trustworthy: only `train.py` was modified, compile and ruff checks passed, startup configuration matched the plan, the LR drop occurred at step 21000, parameter count stayed 822,790, and final summary metrics were present.
- **Verdict**: no-improvement
- **Verdict Basis**: The run was valid, but `best_test_acc=93.54%` is below both the 93.97% baseline and the 94.07% improvement threshold.

## Unexplored Avenues
- Lower dropout such as `p=0.05` could reduce underfitting, but after multiple isolated regularizer misses, a smaller scalar retune is lower priority than a distinct mechanism.
- Dropout only after the first LR drop could target late overfit more directly, but it adds scheduler-conditioned behavior and should be considered only with stronger evidence that late classifier overfit is the bottleneck.

## Next Steps
- High confidence: deprioritize isolated regularization levers and keep the 28/56/112, 2e-4, label-smoothed reflection anchor unchanged.
- Medium confidence: test a more structural compact architecture move, such as a shallower ResNet-14 or stage-limited capacity redistribution, because scalar regularizers are mostly closed.
- Medium confidence: test a cheaper stage-3-only attention/gating variant only if it avoids all-block SE overhead and has a clear local mechanism.

## Exit Action Results
