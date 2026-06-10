# Report EXP-055: Reliable Mild Mixup Retry
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-055.md
- **Plan**: plans/plan-055.md
- **Log**: logs/exp-log-055.md

## Goal
Improve CIFAR-10 `best_test_acc` under the fixed `prepare.py` harness, modifying only `train.py`. The active baseline is 93.97% at commit `755be2c`, and the goal verification requires at least +0.10 percentage points, so EXP-055 needed `best_test_acc >= 94.07%` to count as an improvement.

## Idea & Hypothesis
The selected idea was a reliable retry of the EXP-042 mild mixup intervention. EXP-042 crashed before producing a final metric, so mixup remained scientifically unmeasured rather than disproven. The hypothesis was that a completed foreground run with `MIXUP_ALPHA=0.1` would preserve the step-21000 LR drop and possibly improve late generalization enough to clear 94.07%.

## Approach
`train.py` was modified to add `MIXUP_ALPHA = 0.1`, print the active alpha at startup, sample one beta-distributed lambda per batch, permute the batch on-device, mix inputs, and train with weighted two-target cross entropy while preserving `label_smoothing=0.05`.

The experiment preserved the current anchor: `STAGE_WIDTHS=(28, 56, 112)`, `BATCH_SIZE=128`, `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=2e-4`, `LR_MILESTONES=[21000, 64000]`, reflection crop padding, FP32 compile, channels-last, and once-per-epoch validation. Evaluation stayed unchanged through the fixed `Eval.evaluate()` path.

## Execution
One local foreground run was launched on GPU0 with output captured to `run.log`. Preflight passed: the tracked code diff was limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported no issues.

Startup confirmed CUDA, `num_params=822,790`, `Mixup alpha: 0.1`, and `Batches per epoch: 390`. The run reached the first LR transition at `step 21000` with `lr: 0.0100`, completed cleanly within the 10-minute cap, and produced numeric final metrics. No traceback, CUDA OOM, NaN, or Inf patterns were observed.

## Results
- **Primary metric**: 93.85% (baseline: 93.97%, delta: -0.12 percentage points, -0.13%)
- **Observations**: Mixup completed 37,547 steps across 97 epochs. Post-drop convergence climbed quickly to 93.37% by epoch 58, then plateaued mostly between 93.3% and 93.8%, peaking at 93.85% at epoch 95. Final test accuracy was 93.48%, final test loss was 0.2594, training time was 300.0 seconds, and total time was 394.5 seconds.
- **Analysis**: The retry resolves EXP-042's open question: mild alpha-0.1 mixup is executable under the reliable foreground launch path, reaches the LR drop, and produces a clean result. The hypothesis was not supported because the peak remained 0.12 percentage points below the active anchor and 0.22 percentage points below the required improvement threshold. Mixup is not catastrophically harmful here, but the added interpolation regularization appears to soften or slow the final fit enough to stay below the label-smoothed, stronger-weight-decay anchor.
- **Key Learning**: Mild mixup alpha 0.1 is now a clean no-improvement rather than an infrastructure unknown, peaking at 93.85% after reaching the LR drop.

## Verification
- **Conditions**: all execution and integrity conditions passed; the metric condition failed because 93.85% is below 94.07%.
- **Review Notes**: Results are trustworthy. The run used a single GPU, modified only `train.py`, completed without crash, reported numeric metrics, preserved parameter count and batch geometry, and hit the step-21000 LR drop.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid completed run, but `best_test_acc=93.85%` did not exceed the 93.97% baseline by the required +0.10 percentage points.

## Unexplored Avenues
- Lower-strength mixup (`alpha=0.05`) may reduce over-regularization, but it would not address the step-overhead tradeoff and now has lower priority after alpha 0.1 missed the bar cleanly.
- Coupling mixup with a different schedule or smoothing value could change the regularization balance, but scalar LR, smoothing, and schedule brackets are already weak locally.
- Final-layer dropout remains a low-overhead regularizer candidate, but recent isolated regularizers often underperform this anchor, so it should be treated as low-to-medium confidence.

## Next Steps
- Try a distinct optimizer-dynamics or normalization-state mechanism, medium confidence: avoid replaying isolated schedule, batch-size, width, decay, scalar-LR, residual-regularization, and mixup brackets.
- Test a very small final classifier dropout, low-to-medium confidence: it is low overhead and distinct from residual dropping, but isolated regularization has a weak recent record.
- Search for a new architecture-neutral stability mechanism, medium confidence: the current anchor is robust, so future ideas need a clearer causal path than more generic regularization.

## Exit Action Results
