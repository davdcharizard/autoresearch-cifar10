# Goal: Maximize CIFAR-10 Best Test Accuracy
**Created**: 2026-08-05

## Goal Statement
Modernize the 2016 ResNet20 CIFAR-10 training baseline by changing only `train.py` and maximize the highest test accuracy reached during a run. Improvements must be genuine training enhancements under the fixed evaluation, hardware, and wall-clock protocol rather than seed selection or evaluation changes.

## Primary Metric
- **Metric**: best_test_acc (%)
- **Direction**: higher is better

## Hard Constraints
- Modify only `train.py`; all other tracked project files are read-only during experiments.
- Do not modify `prepare.py` or the `Eval.evaluate()` ground-truth evaluation harness.
- Do not install packages, add dependencies, or change `pyproject.toml` or `uv.lock`.
- Run every baseline and experiment on one NVIDIA H20 GPU with 98 GB VRAM, confirmed before execution.
- Keep the fixed wall-clock training budget defined in `prepare.py`; startup, compilation, and validation remain excluded as implemented by the harness.
- Do not run validation more than once per epoch.
- Do not use seed hacking or reroll seeds to obtain a favorable metric.
- Kill any run exceeding 10 minutes and treat it as a failure.
- Redirect all training output to `run.log`; do not use `tee` or stream full training output into the agent context.
- Remove completed experiment log files before starting the next experiment.

## Verification

### Necessary Conditions
- `best_test_acc` must exceed the current baseline by at least 0.1 percentage points.
- The run must complete without crashing and print the expected numeric summary.
- The run must respect the fixed training time budget and finish within 10 minutes total.

### Procedure
1. Confirm that exactly one NVIDIA H20 GPU with approximately 98 GB VRAM is available for the run.
2. For the first run, execute the unmodified baseline. For later runs, obtain the current moving baseline from `04-results.tsv` using `exp-index.sh baseline`.
3. Ensure no stale `run.log` or renamed run-log variant remains from a completed experiment.
4. Run `uv run train.py > run.log 2>&1`, monitor it, and terminate it as a failure if it exceeds 10 minutes.
5. Extract the primary and VRAM metrics with `grep "^best_test_acc:\|^peak_vram_mb:" run.log`. If this is empty, inspect `tail -n 50 run.log` for the crash.
6. Parse the final summary, compare `best_test_acc` with the moving baseline, record the result, and remove the log before the next experiment.

### Informational Metrics (Optional)
- final_test_acc (%): accuracy at the final evaluation.
- final_test_loss: loss at the final evaluation.
- training_seconds (s): time counted against the fixed training budget.
- total_seconds (s): end-to-end runtime.
- startup_seconds (s): startup and compilation overhead.
- peak_vram_mb (MB): peak accelerator memory consumption and soft VRAM-cost signal.
- num_epochs: completed training epochs.
- num_steps: completed optimizer steps.
- num_params: model parameter count.

## Exit Actions

<!-- Optional. The instructions below are followed by the agent once an experiment (loop)
     finishes on either failure or success. Runs after every loop with this goal. Non-blocking.

     Examples:
     - "Comment on GitHub issue myorg/myrepo#42 with the experiment ID, primary metric delta,
        key learning, and a one-sentence approach summary.
        On failure, include what was tried and why it failed."
     - "Send a Feishu message to the #research channel summarizing the outcome."
     - "On improvement: open a PR from the experiment branch to main as a permanent record,
        then post a one-paragraph summary to issue #42 including the metric delta and the
        PR URL; do nothing on no-improvement, invalid, or crash." -->
