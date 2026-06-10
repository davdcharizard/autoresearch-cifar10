# Report EXP-021: Sparse Post-Drop Weight Averaging
- **Created**: 2026-06-08
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-021.md
- **Plan**: plans/plan-021.md
- **Log**: logs/exp-log-021.md

## Goal
EXP-021 targeted higher CIFAR-10 `best_test_acc` under the fixed harness and fixed 300s training budget. The current experiment-index baseline was 93.23% from commit `f187edf`, so the goal's +0.10 percentage-point rule required at least 93.33% to count as an improvement.

## Idea & Hypothesis
The chosen idea was sparse post-drop weight averaging on the current 28/56/112 ResNet-20 anchor with the proven `[21000, 64000]` LR milestones. The hypothesis was that averaging late low-LR weights once per epoch would smooth the current best recipe's best/final gap without the per-step overhead that hurt EXP-004.

## Approach
`train.py` imported `AveragedModel`, preserved `STAGE_WIDTHS = (28, 56, 112)` and `LR_MILESTONES = [21000, 64000]`, and kept a raw uncompiled model as the optimizer and averaging source while using the compiled wrapper for training. After step 21000, the run updated the averaged model once per epoch and evaluated exactly one model per epoch: live before averaging started, averaged afterward.

The planned `use_buffers=True` path crashed on integer BatchNorm bookkeeping buffers in Run 1, so Run 2 switched to `AVG_USE_BUFFERS = False`. In this PyTorch version that averages parameters while copying buffers from the live model.

## Execution
Run 1 reached the first LR drop and produced the first averaged evaluation, then crashed on the second averaged update with `RuntimeError: Integer division with addcdiv is no longer supported`. The error came from default averaging over integer BatchNorm buffers.

Run 2 completed cleanly after switching to `use_buffers=False`. It reached the 21k first LR drop, activated averaged evaluation at epoch 54, passed the previous crash point, and ran to the fixed 300s training budget.

## Results
- **Primary metric**: 91.85% (baseline: 93.23%, delta: -1.38 points, -1.48%)
- **Observations**: Averaged evaluation peaked early at epoch 59 with 91.85%, then degraded steadily as snapshots accumulated. Late averaged evaluations fell to 58.31% at epoch 103 and 51.61% at epoch 110.
- **Analysis**: The result rejects the hypothesis for naive equal post-drop averaging. The collapse suggests that averaging a long post-drop trajectory, especially with copied live BatchNorm buffers, produces weights whose predictions become badly mismatched even though the underlying training loss remains low.
- **Key Learning**: Naive post-drop equal averaging peaked early then collapsed as snapshots accumulated, so it is unsuitable without a short window or BN recalibration.

## Verification
- **Conditions**: All process and hard-constraint checks passed; the metric improvement condition failed.
- **Review Notes**: Results are trustworthy. The run completed without traceback, OOM, NaN, or Inf patterns; only `train.py` changed; validation remained once per epoch; and total wall-clock was 394.1 seconds.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid result, but `best_test_acc=91.85%` is below the 93.23% baseline and the required 93.33% threshold.

## Unexplored Avenues
- Short-window averaging after the first LR drop might avoid the long-horizon collapse by keeping only recent compatible snapshots.
- EMA-style averaging only after the first LR drop could smooth late weights without EXP-004's full-run per-step overhead, though update frequency must be low enough to preserve throughput.
- BatchNorm recalibration might repair averaged-weight/buffer mismatch, but it would need careful validation against the fixed-budget and evaluation-cadence constraints before use.

## Next Steps
Try a cleaner low-cost anchor perturbation next. High confidence: avoid further width increases above 28/56/112 and avoid long equal averaging. Medium confidence: test a schedule-only 20k first LR drop on the 28/56/112 anchor, or a mild weight-decay reduction, because both preserve throughput and stay within the currently successful architecture.
