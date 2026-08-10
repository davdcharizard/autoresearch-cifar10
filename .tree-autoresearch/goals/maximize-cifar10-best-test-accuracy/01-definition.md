# Goal: Maximize CIFAR-10 Best Test Accuracy
**Created**: 2026-08-05

## Goal Statement
Modernize the ResNet-20 CIFAR-10 training baseline to achieve the highest possible `best_test_acc` within the fixed training-time budget. Improvements must come from genuine changes to the model or training setup in `train.py`, not from evaluation changes or seed rerolling.

## Primary Metric
- **Metric**: best_test_acc (%)
- **Direction**: higher is better

<!-- Baseline and best values are intentionally NOT tracked here - they live in goals/maximize-cifar10-best-test-accuracy/04-results.tsv
     (BASE row + best: header), maintained by tree.sh. -->

## Hard Constraints

- Experiments may modify only `train.py`; `prepare.py`, the evaluation harness, and all other project files are read-only.
- Do not install packages, add dependencies, or use anything outside the dependencies already declared in `pyproject.toml`.
- Each experiment must run only on physical GPU 0, an NVIDIA H20 with 98 GB memory, by setting `CUDA_VISIBLE_DEVICES=0`.
- Preserve the fixed `TIME_BUDGET_S` wall-clock training budget from `prepare.py`; startup, compilation, and validation are excluded from that training budget.
- Run validation no more than once per epoch.
- Seed rerolling is not an optimization method; metric gains must reflect a genuine training or model enhancement.
- Kill any run that exceeds 10 minutes and classify it as a failure.
- VRAM is a soft consideration rather than a pass/fail constraint; increases are acceptable when justified by meaningful accuracy gains.

## Verification

### Necessary Conditions

- `best_test_acc` must be at least 0.10 percentage points higher than baseline, where baseline means the experiment's parent node metric.
- The run must complete without crashing, respect the fixed training-time budget, and print the complete final summary.

### Procedure

1. Confirm that physical GPU 0 is an NVIDIA H20 with approximately 98 GB memory and expose only that GPU with `CUDA_VISIBLE_DEVICES=0`.
2. Read the parent metric using `tree.sh show <base>` for experiment comparisons; for initial goal setup, run the unchanged `train.py` to establish the BASE metric.
3. From the repository root, run `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` and enforce a 10-minute outer timeout.
4. Extract the primary and memory metrics with `grep "^best_test_acc:\\|^peak_vram_mb:" run.log`; an empty result means the run crashed, in which case inspect `tail -n 50 run.log`.
5. Parse the complete final summary, verify all necessary conditions, then remove `run.log` or any renamed variant before beginning another experiment.

### Informational Metrics (Optional)

- `final_test_acc` (%): accuracy at the final evaluation.
- `final_test_loss`: cross-entropy loss at the final evaluation.
- `training_seconds` (s): measured time spent in training operations.
- `total_seconds` (s): end-to-end runtime including startup and evaluation.
- `startup_seconds` (s): setup time excluded from the training budget.
- `peak_vram_mb` (MiB): peak allocated GPU memory.
- `num_epochs`: completed training epochs.
- `num_steps`: completed optimization steps.
- `num_params`: trainable model size.

## Exit Actions

<!-- No exit actions were requested. -->
