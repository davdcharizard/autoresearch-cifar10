# Plan EXP-001: Budget-Aligned Cosine SGD with Nesterov
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Implement the reviewed training policy
- [x] Modify only `train.py` to add a 15%-hold, elapsed-time cosine learning-rate policy from `0.1` to `1e-4`.
- [x] Enable Nesterov on the existing SGD optimizer and remove `MultiStepLR` without changing model, batch size, transforms, loss, seed, weight decay, or maximum steps.
- [x] Enable persistent training-loader workers; diagnostics measured epoch 2 at `1.025s` with persistence versus `18.975s` without it.
- [x] Replace per-epoch evaluation with checkpoints at 20%, 40%, 60%, 70%, 80%, and 90% of counted training time, plus an unconditional terminal evaluation.
- [x] Ensure each epoch is evaluated at most once and that `test_loss`/`test_acc` are always initialized before the final summary.
- [x] Verification: `uv run python -m py_compile train.py` exits `0`.

### Milestone 2: Validate code quality and experiment isolation
- [x] Run `uv run ruff check train.py` and resolve all findings within `train.py`.
- [x] Run `uv run pre-commit run --files train.py`; accept only changes to `train.py` and rerun until it exits `0`.
- [x] Inspect `git diff -- train.py` and confirm the diff contains only the reviewed scheduler, Nesterov, persistent-worker, and validation-cadence changes.
- [x] Confirm no tracked file other than `train.py` is modified with `git status --short --untracked-files=no`.

### Milestone 3: Execute and verify the experiment
- [x] Confirm exactly one NVIDIA H20 with approximately 98 GB VRAM using `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader`.
- [x] Query the moving baseline using `exp-index.sh baseline`; require `baseline=91.67` for this plan's precomputed threshold.
- [x] Ensure `run.log` does not exist, then run the experiment under a 600-second hard timeout with all output redirected.
- [x] Monitor process health without streaming the training log; on completion extract only the final summary.
- [x] Verify all necessary conditions in order and collect informational metrics only if they pass.

## Code Changes
- **`train.py`**: import `math`; add `MIN_LR = 1e-4` and `LR_HOLD_FRACTION = 0.15`; enable persistent DataLoader workers and `nesterov=True` in the existing SGD constructor; remove `MultiStepLR` construction and stepping; compute the LR before every optimizer update from prior accumulated `total_training_time`; and gate validation by elapsed-budget checkpoints while guaranteeing one terminal evaluation. This tests whether a long low-LR tail improves `best_test_acc` while addressing the measured loader/evaluation timeout risks.

The learning-rate function is unambiguous:

```text
progress = clamp(total_training_time / TIME_BUDGET_S, 0, 1)
if progress <= 0.15:
    lr = 0.1
else:
    cosine_progress = (progress - 0.15) / 0.85
    lr = 1e-4 + 0.5 * (0.1 - 1e-4) * (1 + cos(pi * cosine_progress))
```

Validation policy at each epoch boundary:

```text
training_done = total_training_time >= TIME_BUDGET_S or step >= MAX_STEPS
checkpoint_due = first not-yet-consumed threshold in [0.20, 0.40, 0.60, 0.70, 0.80, 0.90] is reached
evaluate exactly once iff checkpoint_due or training_done
```

After an evaluation, consume every checkpoint already passed. The `training_done` condition guarantees evaluation of the terminal model even if training ends during a partial epoch. Preserve the exact `"  eval ep {epoch:3d}"` log prefix for non-vacuous validation checks. `gc.collect()` remains after epoch 1 and is independent of whether that epoch is evaluated.

## Configuration Changes
- SGD Nesterov: `False` -> `True` (reviewed low-cost complement to the chosen cosine schedule).
- Learning-rate scheduler: step milestones `[32000, 48000]`, gamma `0.1` -> elapsed-time 15% hold followed by monotone cosine decay (the baseline never reaches the second milestone).
- Minimum learning rate: unreachable effective `0.001` -> terminal target `0.0001` (provides a genuine refinement phase).
- Training-loader lifecycle: restart eight workers every epoch -> `persistent_workers=True` (measured second-epoch loader time `18.975s` -> `1.025s`; augmentation distribution is unchanged, though the exact fixed-seed worker RNG stream differs).
- Validation cadence: every completed epoch -> 20/40/60/70/80/90% checkpoints and terminal model (seven observations cover the annealing trajectory while bounding the measured `17.271s` per-evaluation cost).
- Unchanged controls: `LR=0.1`, `MOMENTUM=0.9`, `WEIGHT_DECAY=1e-4`, `BATCH_SIZE=128`, `MAX_STEPS=64000`, seed `42`, ResNet20 architecture, initialization, crop/flip transforms, hard-label cross-entropy, and fixed evaluator.

The dominant metric risk is the shorter high-LR phase: the baseline stays at `0.1` for about 83% of counted time, while this experiment holds it for 15% before annealing. This is an intentional, pre-registered optimization bet rather than a free change. Nesterov is a second dynamics variable; if the result fails, the next discriminating run uses the same schedule with standard momentum. Although the exact endpoint LR is reached only at termination, the cosine tail supplies roughly the final 17% of updates below `0.01`, which is the missing refinement region being tested.

## Execution Environment
- Method: local single-process run from the project root: `timeout --signal=TERM 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one NVIDIA H20 GPU with approximately 97,871 MiB; existing CPU/data-loader allocation; no new packages.
- Estimated runtime: approximately 430-550 seconds total; always abort at 600 seconds. Diagnostics measured persistent loader epochs near 1 second after the first and fixed evaluations near 17.3 seconds; seven evaluations contribute about 121 seconds. Counted training remains approximately 300 seconds.
- Log output: all stdout/stderr redirected to `run.log`; only compact process health and final summary fields enter agent context.
- Tool skill: `/research-execute` owns implementation, execution, and monitoring; no remote submission skill is needed.

## Abort Criteria
- GPU check does not show exactly one NVIDIA H20 with approximately 98 GB VRAM.
- Baseline query is not `91.67`; stop and recompute the concrete success threshold from the current moving baseline before execution.
- Any tracked file other than `train.py` becomes modified.
- Static compilation, Ruff, or pre-commit fails after targeted correction attempts.
- The process exits non-zero, reports a CUDA/runtime exception, emits non-finite loss, or produces no expected final summary.
- The process remains active at 600 seconds; `timeout` must terminate it and the experiment is a failure.
- The run performs validation more than once for any epoch or violates the fixed 300-second counted training budget.

## Verification Protocol

### Verification Procedure
1. **Environment and baseline** (timeout: 30 seconds):
   - Run `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader`; pass only if exactly one row reports `NVIDIA H20` and approximately `97871 MiB`.
   - Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`; pass only if `baseline=91.67`. The concrete accuracy threshold is `91.77%`.
2. **Clean setup and execution** (timeout: 600 seconds):
   - Run `test ! -e run.log` before launch.
   - Run `timeout --signal=TERM 600s uv run train.py > run.log 2>&1`.
   - Exit `124` is a timeout failure; any other non-zero exit is a crash. Exit `0` proceeds to verification.
3. **Necessary condition 1, accuracy improvement** (timeout: 10 seconds):
   - Run `grep '^best_test_acc:' run.log` and parse the numeric percentage.
   - Pass only if `best_test_acc >= 91.77`, which is exactly 0.10 percentage points above the current `91.67` baseline. CIFAR-10 has 10,000 test examples, so accuracy is quantized in exact 0.01-point increments and two-decimal output cannot hide an intermediate 91.765 value. Failure immediately yields `no-improvement`.
4. **Necessary condition 2, valid completion** (timeout: 10 seconds):
   - Run `grep -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log`.
   - Pass only if all nine fields occur exactly once and contain finite numeric values; otherwise classify as crash/invalid according to the observed failure.
5. **Necessary condition 3, fixed budget and wall limit** (timeout: 10 seconds):
   - Parse `training_seconds` and `total_seconds` from the same summary.
   - Pass only if `training_seconds` is at least `300.0` and below `301.0`, and `total_seconds < 600.0`.
   - Run `rg '^  eval ep ' run.log`; require exactly seven matching lines using the pinned prefix, then verify all seven epoch numbers are unique. Zero or missing matches fail verification rather than passing vacuously; any duplicate validation for one epoch makes the experiment invalid.
6. After recording the result and analysis evidence, remove `run.log` before any subsequent experiment.

### Informational Metrics (Optional)
- final_test_acc (%): final summary field `final_test_acc` in `run.log`.
- final_test_loss: final summary field `final_test_loss` in `run.log`.
- training_seconds (s): final summary field `training_seconds` in `run.log`.
- total_seconds (s): final summary field `total_seconds` in `run.log`.
- startup_seconds (s): final summary field `startup_seconds` in `run.log`.
- peak_vram_mb (MB): final summary field `peak_vram_mb` in `run.log`.
- num_epochs: final summary field `num_epochs` in `run.log`.
- num_steps: final summary field `num_steps` in `run.log`.
- num_params: final summary field `num_params` in `run.log`.
