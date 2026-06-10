# Report EXP-053: Batch Size 160 Probe
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-053.md
- **Plan**: plans/plan-053.md
- **Log**: logs/exp-log-053.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed harness, with higher accuracy better. The current experiment-index baseline is 93.97% at commit `755be2c`; because the goal requires at least +0.10 percentage points to count, EXP-053 needed `best_test_acc >= 94.07%`.

## Idea & Hypothesis
The chosen idea was a modest larger-batch probe: increase `BATCH_SIZE` from 128 to 160 while preserving the successful 28/56/112 model, step LR schedule, `WEIGHT_DECAY=2e-4`, reflection crop padding, label smoothing, compile, channels-last, and validation cadence. The hypothesis was that batch 160 would preserve the step-21000 LR drop while improving image coverage and gradient stability enough to clear 94.07%.

## Approach
The implementation changed only `train.py`, setting `BATCH_SIZE = 160`. No model, optimizer, LR milestone, regularization, augmentation, compile, memory-format, or evaluation code changed. Keeping step-based milestones unchanged made the run a direct test of batch geometry rather than a coupled schedule experiment.

## Execution
One local foreground run was launched on GPU0 with output captured to `run.log`. Startup confirmed CUDA, `num_params=822,790`, the 300s training budget, and `Batches per epoch: 312`. The run was healthy, reached the first LR milestone at `step 21000` with `lr: 0.0100`, and exited cleanly with final summary metrics. No traceback, OOM, NaN, or inf pattern was observed.

## Results
- **Primary metric**: 93.71% (baseline: 93.97%, delta: -0.26 pp, -0.28%)
- **Observations**: Batch 160 completed 118 epochs and 36,597 steps, reached 93.71% at epoch 89, then drifted downward to a final accuracy of 93.31%. Peak VRAM rose modestly to 785.4 MB.
- **Analysis**: The hypothesis was not supported. The larger batch preserved the first LR drop and increased epoch count versus the batch-128 anchor, but the reduced update count and changed SGD noise did not improve generalization. Together with the prior batch-96 and batch-112 failures, this closes the obvious isolated batch-size bracket around the current anchor.
- **Key Learning**: Isolated batch-size deviations from 128 now fail on both sides; the current anchor's batch geometry is better than nearby smaller or larger probes.

## Verification
- **Conditions**: all passed
- **Review Notes**: results confirmed trustworthy. The run modified only `train.py`, produced numeric metrics, stayed under the 10-minute total limit, reported expected batch geometry, preserved parameter count, and reached the step-21000 LR drop.
- **Verdict**: no-improvement
- **Verdict Basis**: all process conditions passed, but `best_test_acc=93.71%` is below the 94.07% improvement threshold.

## Unexplored Avenues
- Batch-size changes coupled to a schedule retune remain possible, but isolated schedule-only experiments are already a recurring failed space, so this is low priority.
- Gradient accumulation could decouple physical batch size from optimizer batch size, but it would likely reduce update throughput under the fixed time budget.

## Next Steps
- Reliable mild mixup retry: medium confidence. EXP-042 crashed before producing a metric, so mixup remains scientifically unmeasured despite known throughput risk.
- Very mild residual drop-path: low to medium confidence. It targets residual co-adaptation without changing capacity, but residual-branch interventions have local failure history.
- Try a distinct optimizer-dynamics lever rather than more batch, scalar LR, or decay brackets: medium confidence. The isolated local hyperparameter brackets around the anchor are mostly closed.

## Exit Action Results
