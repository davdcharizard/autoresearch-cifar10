# Goal: Maximize CIFAR-10 Test Accuracy
**Created**: 2026-06-10

## Goal Statement
Maximize `best_test_acc` of the ResNet-20 CIFAR-10 baseline (He et al. 2015) within the fixed wall-clock training budget defined in `prepare.py`, by modernizing `train.py` — model architecture, optimizer, learning-rate schedule, data augmentation, regularization, batch size, training loop, etc. The baseline is a faithful 2016-era ResNet-20 setup; the task (per TASK.md) is to bring it up to date and squeeze the highest test accuracy possible under the same compute budget.

## Primary Metric
- **Metric**: best_test_acc (%)
- **Direction**: higher is better

<!-- Current baseline is intentionally NOT tracked here — it lives in experiment-indices/maximize-cifar10-test-accuracy.tsv.
     All autoresearch goals must be quantitative. For binary goals like "make feature X work", use `status`
     as the metric with values 0 (not working) and 1 (working). There are no qualitative-only goals. -->

## Hard Constraints

- Only `train.py` may be modified. `prepare.py` (fixed constants, evaluation function, time budget) is read-only; the `Eval.evaluate()` method is the ground-truth metric and must not be touched.
- No new packages or dependencies — only what is already in `pyproject.toml` may be used.
- Fixed compute: single GPU, always GPU 0 (two H20s on the node; if GPU 0 is busy, wait for it to free up rather than using GPU 1). Fixed wall-clock training budget from `prepare.py`. Kill any run exceeding 10 minutes total and treat it as a failure.
- No seed hacking — re-rolling seeds to bump the metric without a genuine enhancement is not an optimization move.
- Validation must run at most once per epoch.
- VRAM is a soft constraint — some increase is acceptable for a meaningful gain in best_test_acc.

## Verification

### Necessary Conditions

- best_test_acc exceeds the current baseline by at least 0.1 percentage points (absolute).
- The run completes without crashing within the time budget (≤ 10 minutes total wall clock).
- Validation is executed at most once per epoch.

### Procedure

1. Confirm GPU 0 is free (`nvidia-smi`); if busy, wait for it to free up.
2. Run the experiment on GPU 0: `uv run train.py > run.log 2>&1` (redirect everything — do not tee or stream output).
3. Extract metrics: `grep "^best_test_acc:\|^peak_vram_mb:" run.log`. Empty grep output means the run crashed — read `tail -n 50 run.log` for the stack trace.
4. Get the current baseline via `exp-index.sh baseline` and compare.
5. Delete `run.log` (and any renamed variant) once the experiment concludes, before starting a new one.

### Informational Metrics (Optional)

- peak_vram_mb: VRAM headroom and cost of the change (soft constraint awareness)
- num_epochs: how many epochs fit in the fixed time budget (throughput proxy)
- num_params: model size trade-off vs accuracy

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
