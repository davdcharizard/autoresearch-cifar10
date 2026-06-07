# Goal: Maximize CIFAR-10 Test Accuracy
**Created**: 2026-05-28

## Goal Statement
Maximize the CIFAR-10 test accuracy (best_test_acc) by modernizing the ResNet-20 baseline training in train.py. The baseline is a faithful ResNet-20 implementation from He et al. 2015 achieving ~91.81% accuracy. The objective is to apply modern techniques (architecture changes, optimizer improvements, augmentation strategies, regularization, etc.) to push accuracy as high as possible within the fixed 300-second training time budget.

## Primary Metric
- **Metric**: best_test_acc (%)
- **Direction**: higher is better

<!-- Current baseline is intentionally NOT tracked here — it lives in experiment-indices/maximize-cifar10-test-accuracy.tsv.
     All autoresearch goals must be quantitative. -->

## Hard Constraints

- **Only modify train.py**: `prepare.py` is read-only. It contains the fixed evaluation harness and time budget. No other source files may be created or modified for the experiment.
- **No new dependencies**: Only packages already in `pyproject.toml` can be used. No `pip install` or new deps.
- **Single GPU, 300s training budget**: Must run on a single GPU within the TIME_BUDGET_S=300 wall-clock training time defined in `prepare.py`. Total run time (including startup/eval) should not exceed 10 minutes.
- **No eval tampering**: Cannot modify the `Eval.evaluate()` method or evaluation harness. Validation must be called at most once per epoch.

## Verification

### Necessary Conditions

- best_test_acc improves over baseline by at least 0.1 percentage points (e.g., if baseline is 91.81%, experiment must achieve >= 91.91%)
- Training run completes without crashing and finishes within the 300s time budget
- Validation (evaluator.evaluate) is called no more than once per epoch

### Procedure

1. Run the experiment: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
2. Extract metrics: `grep "^best_test_acc:\|^peak_vram_mb:" run.log`
3. If grep output is empty, the run crashed — check `tail -n 50 run.log` for the stack trace
4. Get current baseline via `exp-index.sh baseline`
5. Compare best_test_acc against baseline + 0.1%

### Informational Metrics (Optional)

- final_test_acc: final epoch test accuracy (may differ from best)
- final_test_loss: final epoch test loss
- training_seconds: actual wall-clock training time
- total_seconds: total run time including startup and eval
- startup_seconds: time before training starts
- peak_vram_mb: peak GPU memory usage
- num_epochs: total epochs completed
- num_steps: total training steps completed
- num_params: model parameter count

## Experiment Index
See: experiment-indices/maximize-cifar10-test-accuracy.tsv

## Exit Actions

<!-- No exit actions configured. -->
