# Plan EXP-002: 80%-Hold Cosine with Standard Momentum
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Implement the reviewed schedule and protocol controls
- [x] Modify only `train.py` to hold `lr=0.1` through 80% of counted time, then step to `0.01` and cosine-decay to `1e-4` over the final 20%.
- [x] Keep standard SGD momentum (`nesterov=False`) and remove the absolute-step `MultiStepLR`.
- [x] Enable persistent training-loader workers; evaluate at 20/40/60/70%, then every completed epoch from 80% through termination.
- [x] Guarantee at most one evaluation per epoch and unconditional terminal metric initialization.
- [x] Verify with `uv run python -m py_compile train.py`.

### Milestone 2: Validate isolation and code quality
- [x] Run `uv run ruff check train.py`.
- [x] Run `uv run pre-commit run --files train.py` until it exits `0`.
- [x] Confirm `git diff --check -- train.py` and inspect the diff for only schedule, persistent-worker, and evaluation-cadence changes.
- [x] Confirm no tracked file other than `train.py` is modified.

### Milestone 3: Execute and verify
- [x] Confirm exactly one NVIDIA H20 with approximately 98 GB VRAM and current baseline `91.67%`.
- [x] Require no stale `run.log`, then launch with full redirection under a 600-second timeout.
- [x] Monitor compact process/GPU/fatal-pattern health without streaming progress.
- [x] Verify necessary conditions in order; collect informational metrics only if all pass.

## Code Changes
- **`train.py`**: import `math`; define `ANNEAL_START_LR=0.01`, `MIN_LR=1e-4`, `LR_HOLD_FRACTION=0.80`, and four early evaluation checkpoints; set `persistent_workers=True`; remove `MultiStepLR` construction and the loop's `scheduler.step()` call; compute and assign LR before each update from prior `total_training_time`; and gate the unchanged evaluator at checkpoint/dense-tail/terminal epoch boundaries. The optimizer remains standard SGD with momentum `0.9`, and all model/data/loss/seed/resource controls remain unchanged.

Learning-rate function:

```text
progress = clamp(total_training_time / TIME_BUDGET_S, 0, 1)
if progress <= 0.80:
    lr = 0.1
else:
    cosine_progress = (progress - 0.80) / 0.20
    lr = 1e-4 + 0.5 * (0.01 - 1e-4) * (1 + cos(pi * cosine_progress))
```

Evaluation policy at each epoch boundary:

```text
training_done = total_training_time >= TIME_BUDGET_S or step >= MAX_STEPS
checkpoint_due = first unconsumed threshold in [0.20, 0.40, 0.60, 0.70] is reached
dense_tail_due = progress >= 0.80
evaluate exactly once iff checkpoint_due or dense_tail_due or training_done
```

Consume all reached early thresholds after each evaluation. Preserve the exact `"  eval ep {epoch:3d}"` prefix. Initialize `test_loss=test_acc=None`, make `training_done` an unconditional evaluation trigger, and assert immediately before the summary that terminal values are non-null; the last printed `final_test_*` values must therefore come from the terminal epoch, not a stale checkpoint. Keep `gc.collect()` after epoch 1 independent of evaluation.

## Configuration Changes
- Learning-rate scheduler: absolute steps `[32000, 48000]` with gamma `0.1` -> 80% hold, step to `0.01`, then 20% elapsed-time cosine to `1e-4`.
- Training-loader lifecycle: non-persistent -> `persistent_workers=True` (validated EXP-001 protocol control).
- Validation cadence: every epoch -> 20/40/60/70% plus every completed epoch from 80% through terminal. This densely samples the only region where the new schedule can beat or overfit the baseline while retaining ample runtime margin.
- Standard momentum: unchanged at `0.9`; explicitly do not enable Nesterov, removing the EXP-001 confound.
- Unchanged: `LR=0.1`, `WEIGHT_DECAY=1e-4`, `BATCH_SIZE=128`, `MAX_STEPS=64000`, seed `42`, architecture, initialization, crop/flip, hard-label loss, evaluator, dependencies, and fixed 300-second budget.

The dominant research risk is a compressed deep-refinement phase: the full final 20% is at or below `0.01`, but the cosine reaches very small values only near termination. This is intentional; the experiment preserves the baseline's known-good plateau and low-LR tail while adding a gradual deeper phase without repeating EXP-001's over-annealing.

## Execution Environment
- Method: `timeout --signal=TERM 600s uv run train.py > run.log 2>&1` from the project root.
- Resources: exactly one NVIDIA H20 with approximately 97,871 MiB; existing environment only.
- Estimated runtime: approximately 350-430 seconds total. EXP-001 took `321.7s` with seven evaluations; adding roughly 13-17 dense-tail evaluations remains well below 600 seconds.
- Log output: all stdout/stderr in `run.log`; inspect only compact fatal signals and final fields.
- Tool skill: `/research-execute`; no remote submission or WandB.

## Abort Criteria
- GPU or baseline gate differs from the planned single H20 / `91.67%` reference.
- Any tracked file other than `train.py` changes.
- Compile, Ruff, pre-commit, or diff checks fail after targeted fixes.
- Process exits non-zero, reports CUDA/OOM/traceback/non-finite loss, or lacks a final summary.
- Process remains active at 600 seconds, validates an epoch twice, or violates the 300-second counted budget.

## Verification Protocol

### Verification Procedure
1. **Environment/baseline** (30s): run `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader` and `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`; require one `NVIDIA H20`, approximately `97871 MiB`, and `baseline=91.67`. Concrete threshold: `91.77%`.
2. **Clean execution** (600s): require `test ! -e run.log`; run `timeout --signal=TERM 600s uv run train.py > run.log 2>&1`; exit `0` passes execution, `124` is timeout, other non-zero is crash.
3. **Accuracy** (10s): parse `grep '^best_test_acc:' run.log`; pass only if `>=91.77%`. CIFAR-10 accuracy is quantized in exact 0.01-point increments over 10,000 examples. On failure, stop verification with `no-improvement`.
4. **Valid summary** (10s): require exactly one finite numeric occurrence of each of the ten fields matching `grep -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log`.
5. **Budget/integrity** (10s): require `300.0 <= training_seconds < 301.0`, `total_seconds < 600.0`, at least 15 `rg '^  eval ep '` lines, all epoch numbers unique, and the last evaluated epoch equal to summary `num_epochs`. Missing lines fail rather than pass vacuously.
6. Persist all analysis evidence, then remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- final_test_acc (%): final summary `final_test_acc`.
- final_test_loss: final summary `final_test_loss`.
- training_seconds (s): final summary `training_seconds`.
- total_seconds (s): final summary `total_seconds`.
- startup_seconds (s): final summary `startup_seconds`.
- peak_vram_mb (MB): final summary `peak_vram_mb`.
- num_epochs: final summary `num_epochs`.
- num_steps: final summary `num_steps`.
- num_params: final summary `num_params`.
