# Goal: Maximize CIFAR-10 ResNet-20 Test Accuracy Within a Fixed Training Budget
**Created**: 2026-06-08

## Goal Statement
Improve on a faithful ResNet-20 (He et al. 2015) CIFAR-10 baseline by modernizing the
training setup. The model is trained under a fixed wall-clock training budget
(`TIME_BUDGET_S = 300`s, excluding startup/compilation and validation) on a single GPU,
and evaluated on the CIFAR-10 test set by the frozen `Eval.evaluate()` harness in
`prepare.py`. The objective is to maximize `best_test_acc` (the best test accuracy over
all epochs of a run).

Everything inside `train.py` is fair game — model architecture, optimizer, LR schedule,
data augmentation, regularization, batch size, model size/type, and the training loop —
as long as the run finishes within the budget without crashing and the evaluation harness
is left untouched. The baseline scores 91.73% under this budget; the aim is to push test
accuracy as high as possible with modern techniques while respecting the constraints below.

## Primary Metric
- **Metric**: best_test_acc (%)
- **Direction**: higher is better

<!-- Current baseline is intentionally NOT tracked here — it lives in experiment-indices/improve-cifar10-test-accuracy.tsv. -->

## Hard Constraints
<!-- Authoritative source for all rules: TASK.md in the project root. Summarized here. -->

- **Only `train.py` may be modified.** `prepare.py` is read-only (it holds the fixed
  constants, time budget, and the ground-truth `Eval.evaluate()` harness). The evaluation
  harness must not be altered or circumvented. (Enforced via `.autoresearch/protected-files.json`.)
- **No new dependencies.** Only packages already present in `pyproject.toml` may be used.
  No installing or adding packages.
- **Single GPU, fixed time budget.** Each experiment runs on one GPU (NVIDIA H20, 98GB).
  Training wall-clock is fixed at `TIME_BUDGET_S = 300`s. The run must complete without
  crashing and print the summary. Any run exceeding 10 minutes total must be killed and
  treated as a failure.
- **Validation at most once per epoch.** Do not call `evaluate()` more than once per epoch.
- **No seed hacking.** Re-rolling random seeds to bump the metric without a genuine
  algorithmic improvement is not a valid optimization move.
- **VRAM is a soft constraint.** Some increase is acceptable in exchange for a meaningful
  accuracy gain.
- Full authoritative rules: see `TASK.md`.

## Verification

### Necessary Conditions
<!-- ALL must hold. If ANY fails, the experiment is no-improvement (or invalid for constraint breaches). -->

- **`best_test_acc` improves over the current baseline by at least 0.1 percentage points**
  (i.e. `best_test_acc >= baseline + 0.1`), where baseline is read from the experiment index.
- **The run completes cleanly within budget**: `train.py` finishes without crashing, prints
  the final summary block (`best_test_acc:` etc.), respects `TIME_BUDGET_S`, and total
  wall-clock does not exceed 10 minutes.
- **No hard-constraint violations**: only `train.py` was modified, `prepare.py`/eval harness
  untouched, `evaluate()` called at most once per epoch, no new dependencies, no seed hacking.

### Procedure
1. Get the current baseline via `exp-index.sh baseline` on `experiment-indices/improve-cifar10-test-accuracy.tsv`.
2. Apply the experiment's changes to `train.py` only.
3. Run `uv run train.py > run.log 2>&1` on a single GPU (e.g. `CUDA_VISIBLE_DEVICES=0`).
4. Extract metrics: `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:" run.log`.
   Empty `best_test_acc` ⇒ the run crashed (inspect `tail -n 50 run.log`).
5. Compare `best_test_acc` to baseline; confirm clean completion within budget and no constraint violations.
6. Remove `run.log` before starting the next experiment (keep the working tree clean).

### Informational Metrics (Optional)
- peak_vram_mb: VRAM headroom used (soft constraint awareness)
- num_epochs / num_steps: how much training fit in the budget (signals throughput vs. compute trade-offs)
- training throughput (img/s): efficiency of the training setup under the fixed budget

## Experiment Index
See: experiment-indices/improve-cifar10-test-accuracy.tsv

## Exit Actions

<!-- Optional. None defined for this goal. Add later via /research-goal "Modify this goal" in a copilot session. -->
