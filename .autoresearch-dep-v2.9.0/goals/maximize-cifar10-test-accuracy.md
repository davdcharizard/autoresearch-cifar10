# Goal: Maximize CIFAR-10 Test Accuracy
**Created**: 2026-05-27
**Status**: active  <!-- active | inactive -->

## Goal Statement
Maximize the best test accuracy (best_test_acc) of a CIFAR-10 classifier by modifying `train.py` — architecture, optimizer, data augmentation, hyperparameters, training loop, batch size, model size, model type, etc. Starting from a ResNet-20 baseline (~91.89%), apply modern techniques to push accuracy as high as possible within the fixed time budget.

## Primary Metric
- **Metric**: best_test_acc (%)
- **Direction**: higher is better

<!-- Current baseline is intentionally NOT tracked here — it lives in experiment-indices/maximize-cifar10-test-accuracy.tsv.
     All autoresearch goals must be quantitative. -->

## Hard Constraints
- Only `train.py` may be modified — `prepare.py` is read-only
- Each experiment must run on a single GPU
- Training time budget is 300s, hard-set in `prepare.py` (`TIME_BUDGET_S`) — do not override or shadow this value
- Training must complete within the fixed time budget defined in `prepare.py` (300s wall clock)

## Verification

### Necessary Conditions
- best_test_acc must be strictly higher than the current baseline by at least 0.1 percentage points
- The training script must complete without crashing and print the full summary block
- Validation must not run more than once per epoch

### Procedure
N/A — conditions are self-evaluating. Run `uv run train.py > run.log 2>&1`, then `grep "^best_test_acc:" run.log` to extract the primary metric.

### Informational Metrics (Optional)
- training_seconds: wall-clock training time to check budget compliance
- peak_vram_mb: GPU memory usage
- final_test_acc: final epoch accuracy (vs best)
- final_test_loss: final epoch loss
- num_epochs: how many epochs completed in budget
- num_steps: total training steps
- num_params: model parameter count

## Experiment Index
See: experiment-indices/maximize-cifar10-test-accuracy.tsv

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

DO NOT COMMIT ANY RUN LOGS. Check for any `*.log` files found in the repository, and delete them from git tracking and clean them out.