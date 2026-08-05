# Goal: Maximize CIFAR-10 test accuracy within a fixed 300s training budget

**Created**: 2026-06-28

## Goal Statement
Improve the test accuracy of a CIFAR-10 image classifier over the ResNet-20 (He et al. 2015)
baseline, modifying only `train.py`. The training script runs under a fixed wall-clock training
budget of 300 seconds (defined in `prepare.py`, excluding startup/compilation and validation time).
Within that fixed compute budget, the objective is to push `best_test_acc` (the best test accuracy
observed across epochs) as high as possible by improving the model architecture, optimizer, learning
rate schedule, data augmentation, regularization, batch size, and training loop — anything inside
`train.py`. Modern training techniques are expected to substantially beat the ~91.9% baseline.

This matters as an autoresearch test bench: the eval harness and time budget are frozen, so the only
lever is genuinely better training code, making improvements directly attributable to method quality.

## Primary Metric
- **Metric**: best_test_acc (%)
- **Direction**: higher is better

## Hard Constraints

- **Only `train.py` may be modified.** It is the sole editable file (model architecture, optimizer,
  LR, augmentation, regularization, training loop, batch size, model size/type, etc. are all fair game).
- **`prepare.py` is read-only / frozen.** It contains the fixed evaluation function (`Eval.evaluate`)
  and the 300s time budget. It is hook-protected and must not be modified. The eval harness is the
  ground-truth metric and must not be altered or circumvented.
- **No new dependencies.** Only packages already in `pyproject.toml` may be used; do not install or add
  any new packages.
- **No seed hacking.** Do not re-roll / search random seeds to bump the metric without a genuine method
  improvement. Seed manipulation is not a valid optimization move.
- **At most one validation run per epoch.** Do not call the evaluator more than once per epoch.
- **Fixed compute.** Each experiment runs on a single GPU (NVIDIA H20). For this environment all runs
  use **GPU 1** (`CUDA_VISIBLE_DEVICES=1`), since GPU 0 is in use. The 300s training-time budget is
  fixed and must not be extended.
- **VRAM is a soft constraint.** Some increase is acceptable in exchange for a meaningful accuracy gain;
  there is leeway on this dataset.

## Verification

### Necessary Conditions

- The run completes without crashing and finishes within the fixed time budget (no manual extension);
  the summary prints a valid `best_test_acc`. A run exceeding 10 minutes wall-clock is killed and
  treated as a failure.
- `best_test_acc` improves over the current baseline (from `04-results.tsv`) by **at least 0.1
  percentage points** (the TASK.md minimum-improvement criterion). Gains smaller than 0.1% do not count
  as an improvement.
- The improvement comes from genuine training/method changes in `train.py` only — no seed hacking, no
  modification of or circumvention of the frozen eval harness, and at most one validation per epoch.

### Procedure

1. Get the current baseline via `exp-index.sh baseline` on `goals/maximize-cifar10-test-accuracy/04-results.tsv`.
2. Run the experiment: `CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1` (redirect — do not flood context).
3. Extract metrics: `grep "^best_test_acc:\|^peak_vram_mb:\|^training_seconds:" run.log`. Empty grep ⇒ crash; read `tail -n 50 run.log`.
4. Compare `best_test_acc` to baseline; require ≥ +0.1pp to count as an improvement.
5. Remove `run.log` after recording results to keep the working tree clean.

### Informational Metrics (Optional)
- peak_vram_mb: VRAM headroom used (soft constraint awareness).
- training_seconds / num_epochs / num_steps: confirms the run used the full budget and how many epochs fit.
- num_params: model size, for understanding the accuracy/compute trade-off.

## Exit Actions

<!-- None defined at creation (autopilot). Can be added later in a copilot session via /research-goal. -->
