# Report EXP-054: Very Mild Residual Stochastic Depth
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-054.md
- **Plan**: plans/plan-054.md
- **Log**: logs/exp-log-054.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` harness, modifying only `train.py`. The active baseline is 93.97% at commit `755be2c`, and the goal verification requires at least +0.10 percentage points, so EXP-054 needed `best_test_acc >= 94.07%` to count as an improvement.

## Idea & Hypothesis
The selected idea was very mild residual stochastic depth: randomly drop residual branches during training with a linearly increasing per-block probability and use the full deterministic model at evaluation. The hypothesis was that a conservative maximum drop probability of 0.03 would regularize residual co-adaptation without changing parameter count, throughput, or evaluation behavior, lifting the current 93.97% anchor above the 94.07% threshold.

## Approach
`train.py` was modified to add `STOCHASTIC_DEPTH_MAX_P = 0.03`, store a `drop_prob` per `BasicBlock`, and apply a per-sample `(batch, 1, 1, 1)` residual-branch mask after `bn2(conv2(out))` and before shortcut addition. The mask was active only in training mode and was scaled by keep probability to preserve expected residual magnitude.

Drop probabilities were assigned linearly across the nine residual blocks, ending at 0.03 in the final block. The experiment preserved the anchor widths, batch size, optimizer, LR milestones, weight decay, augmentation, label smoothing, compile path, channels-last memory format, once-per-epoch validation, and unchanged `num_params=822,790`.

## Execution
One local foreground run was launched on GPU1 with output captured to `run.log`. Preflight passed: the tracked code diff was limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported no issues.

Startup confirmed CUDA, `num_params=822,790`, `Stochastic depth max p: 0.03`, and `Batches per epoch: 390`. The run reached the first LR transition at `step 21000` with `lr: 0.0100`, completed cleanly within the 10-minute cap, and produced numeric final metrics. No traceback, CUDA OOM, NaN, or Inf patterns were observed.

## Results
- **Primary metric**: 93.40% (baseline: 93.97%, delta: -0.57 percentage points, -0.61%)
- **Observations**: The run reached 93.12% by epoch 60 after the LR drop, then plateaued at 93.40% from epoch 71 onward. Final test accuracy was 92.61%, final test loss was 0.2717, training time was 300.0 seconds, total time was 398.4 seconds, and the run completed 39,018 steps across 101 epochs.
- **Analysis**: The hypothesis was not supported. Stochastic depth preserved the important execution conditions, so the negative result is attributable to the intervention rather than a missed LR milestone or run failure. In this shallow fixed-budget ResNet-20, even very mild residual branch dropping appears to weaken late fitting or refinement rather than improve generalization.
- **Key Learning**: Isolated very mild residual stochastic depth is a weak residual regularizer for the current anchor, reaching only 93.40% despite a clean LR-drop run.

## Verification
- **Conditions**: all execution and integrity conditions passed; the metric condition failed because 93.40% is below 94.07%.
- **Review Notes**: Results are trustworthy. The run used a single GPU, modified only `train.py`, completed without crash, reported numeric metrics, preserved parameter count and batch geometry, and hit the step-21000 LR drop.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but `best_test_acc=93.40%` did not exceed the 93.97% baseline by the required +0.10 percentage points.

## Unexplored Avenues
- A lower maximum probability such as 0.01 could reduce undertraining, but the 0.03 result is already far below the threshold, making this low priority.
- Stage-specific stochastic depth limited to the final stage might be less disruptive, but local residual regularization evidence is weak after EXP-028, EXP-051, and EXP-054.
- Reliable mild mixup remains more scientifically open because EXP-042 crashed before a final metric, whereas stochastic depth now has a clean negative result.

## Next Steps
- Reliable mild mixup retry, medium confidence: foreground execution is now reliable, and EXP-042 never produced a final metric, so mixup remains an unmeasured augmentation/label-space regularizer.
- Distinct non-residual mechanism, medium confidence: prioritize ideas outside isolated schedule, batch-size, width, weight-decay, scalar-LR, and residual-regularization brackets.
- Further stochastic-depth variants, low confidence: only revisit if paired with a stronger mechanism that explains why the 93.40% plateau should not repeat.

## Exit Action Results
