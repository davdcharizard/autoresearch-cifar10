# Goal: Maximize CIFAR-10 Best Test Accuracy
**Created**: 2026-06-08

## Goal Statement
Improve the CIFAR-10 training recipe in this repository to obtain the highest possible `best_test_acc` from the fixed evaluation harness. Experiments should focus on meaningful changes to `train.py`, including model architecture, optimizer, learning-rate schedule, augmentation, regularization, batch size, and training-loop choices that can improve accuracy under the fixed wall-clock budget.

The baseline is the provided ResNet-20 training setup. The objective is to beat the current best baseline while respecting the task constraints and avoiding changes that would alter the benchmark itself.

## Primary Metric
- **Metric**: best_test_acc (%)
- **Direction**: higher is better

<!-- Current baseline is intentionally NOT tracked here — it lives in experiment-indices/maximize-cifar10-best-test-accuracy.tsv.
     All autoresearch goals must be quantitative. For binary goals like "make feature X work", use `status`
     as the metric with values 0 (not working) and 1 (working). There are no qualitative-only goals. -->

## Hard Constraints
- Experiments may modify `train.py` only.
- Do not modify `prepare.py`; it contains the fixed constants, evaluation function, and wall-clock training budget.
- Do not modify the evaluation harness; `Eval.evaluate()` in `prepare.py` is the ground-truth metric.
- Do not install new packages, add dependencies, or modify dependency lock/config files.
- Each experiment must run on a single GPU, targeting the available NVIDIA H20 class GPU.
- Training must use the fixed time budget defined in `prepare.py`; validation time is excluded by the harness.
- Do not run validation more than once per epoch.
- Seed hacking is not an optimization move; do not re-roll seeds merely to chase a lucky metric.
- If a run exceeds 10 minutes total, kill it and treat it as a failure.
- VRAM is a soft constraint: increases are acceptable only when justified by meaningful `best_test_acc` gains.

## Verification

### Necessary Conditions
- `uv run train.py` completes without crashing.
- The run reports a numeric `best_test_acc`.
- `best_test_acc` improves over the current experiment-index baseline by at least +0.10 percentage points in absolute percent units (for example, a 91.95% baseline requires `best_test_acc >= 92.05%`). Smaller increases are treated as `no-improvement` because they are too noisy to count.
- The implementation respects all hard constraints, especially modifying only `train.py` and preserving the fixed evaluation harness.

### Procedure
1. Confirm a single GPU is selected for the experiment.
2. Remove any stale `run.log` before starting a new experiment.
3. Run `uv run train.py > run.log 2>&1`.
4. If the process exceeds 10 minutes total, terminate it and classify the experiment as a failure.
5. Extract metrics with `grep "^best_test_acc:\|^peak_vram_mb:" run.log`; if no `best_test_acc` is present, inspect the tail of the log for the crash.
6. Compare the reported `best_test_acc` against the current baseline from `experiment-indices/maximize-cifar10-best-test-accuracy.tsv`; classify it as an improvement only if it is at least +0.10 percentage points higher than that baseline.
7. Remove temporary run logs after the experiment result is captured.

### Informational Metrics (Optional)
- final_test_acc: helps identify overfitting or late-epoch degradation.
- final_test_loss: supports accuracy changes with calibration/loss context.
- training_seconds: checks whether runs used the fixed time budget.
- total_seconds: monitors startup and validation overhead.
- peak_vram_mb: tracks the soft VRAM tradeoff.
- num_epochs: shows how training-loop changes affect epoch throughput.
- num_steps: shows optimization-step budget under the fixed wall-clock training window.
- num_params: records architecture size changes.

## Experiment Index
See: experiment-indices/maximize-cifar10-best-test-accuracy.tsv

## Exit Actions

<!-- Optional. The instructions below are followed by the agent once an experiment (loop)
     finishes on either failure or success. Runs after every loop with this goal. Non-blocking.

     Examples:
     - "Comment on GitHub issue myorg/myrepo#42 with the experiment ID, primary metric delta,
        key learning, and a one-sentence approach summary. On success, also include the PR URL.
        On failure, include what was tried and why it failed."
     - "Send a Feishu message to the #research channel summarizing the outcome."
     - "Only notify on improvement: post a one-paragraph summary to issue #42 including the
        metric delta and the PR URL; do nothing on no-improvement, invalid, or crash." -->
