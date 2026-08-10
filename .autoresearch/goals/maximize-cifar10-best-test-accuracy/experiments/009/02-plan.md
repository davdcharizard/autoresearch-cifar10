# Plan EXP-009: Exclude BN and Bias from Weight Decay
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Implement isolated optimizer grouping
- [x] Preserve `WEIGHT_DECAY = 1e-4` and partition trainable parameters by dimensionality: tensors with `ndim > 1` retain `1e-4`; tensors with `ndim <= 1` receive zero decay.
- [x] Construct the existing SGD optimizer from those two groups while preserving global LR, momentum, and all other semantics.
- [x] Assert at construction that the two groups exhaustively cover every trainable parameter exactly once; verify syntax, style, exact diff scope, group counts, total elements, and decay values.

### Milestone 2: Execute one fixed-protocol H20 run
- [x] Confirm exactly one idle NVIDIA H20 with approximately 98 GB memory.
- [x] Confirm no stale `run.log` exists, then execute one seed-42 run with stdout/stderr redirected only to `run.log` and a 600-second timeout.
- [x] Monitor process health and terse log tails without streaming the full log or rerunning on a valid result.

### Milestone 3: Verify result and mechanism
- [x] Parse the numeric summary and evaluation trajectory from `run.log`.
- [x] Verify completion, 300-second counted budget, sub-600-second total runtime, single evaluation per epoch, one augmentation switch, worker shutdown, exact parameter count, and optimizer-group invariants.
- [x] Compare `best_test_acc` to the moving 93.55% baseline; improvement requires at least 93.65%.
- [x] Compare the final strong-view checkpoint and weak-tail recovery with EXP-007 and EXP-008 to determine whether excluding one-dimensional parameters from decay preserved fit.

## Code Changes
- **`train.py`**: Materialize trainable parameters once in their existing `model.parameters()` traversal order, partition them into two exhaustive, disjoint groups, and assert tensor-count plus element-count coverage before optimizer construction. Matrix/kernel tensors (`ndim > 1`) retain accepted `1e-4` coupled decay; one-dimensional normalization affine parameters and biases receive zero. No model, data, schedule, timer, RNG, loss, evaluator, or logging logic changes.

## Configuration Changes
- `WEIGHT_DECAY`: remains `1e-4` (the plan critic rejected increasing decay on fit-limited kernels and identified the combined scalar/targeting edit as an attribution confound).
- Weight-tensor decay: inherited all-parameter `1e-4` -> explicit `1e-4` for 20 tensors / 1,071,200 parameters.
- BN-affine and bias decay: inherited `1e-4` -> explicit `0` for 39 tensors / 2,762 parameters.
- Unchanged: width multiplier 2, 1,073,962 total parameters, batch 128, LR/momentum, 80% high-LR and N1/M7 plateau, deterministic weak-loader tail, seed 42, 300-second counted budget, and evaluator cadence.

## Adversarial Review Response
- Mandatory Claude plan review completed with exit code 0; no fallback reviewer was used.
- Accepted concerns 1-2: preserving kernel decay at `1e-4` removes the diagnosis inversion and two-lever attribution confound from the initial plan. The experiment now isolates one semantic change versus baseline.
- Accepted concerns 4-5: the preflight is an external read-only Python expression, and optimizer construction materializes the trainable list and asserts both tensor-count and element-count coverage.
- Concern 3 remains an honest experimental risk: the one-seed 0.10-point gate may exceed the effect size. It cannot be mitigated without violating the fixed protocol, so the plan records a valid near-baseline result as no-improvement and forbids reruns.

## Execution Environment
- Method: local `timeout 600s uv run train.py > run.log 2>&1` from the project root.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB total VRAM; no other accelerator.
- Estimated runtime: approximately 332-340 seconds total based on EXP-007/008; hard ceiling 600 seconds.
- Log output: complete stdout and stderr only in project-root `run.log`; monitoring uses process status and bounded `tail`/targeted patterns, never `tee`.
- Tool skill: none; local execution.

## Abort Criteria
- Stop on a non-zero process exit, traceback, CUDA OOM, non-finite loss, unavailable/wrong GPU, or evidence another process occupies the required H20 before launch.
- `timeout` terminates the run at 600 seconds total; classify that outcome as a failure.
- Do not abort merely for a weak 80% checkpoint: it is a pre-registered mechanism diagnostic, not a goal condition, and EXP-008 demonstrated substantial tail recovery.
- Run exactly once for a valid completed result. Do not reroll seed, restart for accuracy, or alter the plan after observing metrics.

## Verification Protocol

### Verification Procedure
1. Query the baseline with `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`; require `baseline=93.55` and resolve the success threshold as 93.65%.
2. Run `nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader` and `nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader`; require one NVIDIA H20 near 97,871 MiB and no conflicting compute process.
3. Before launch, require `test ! -e run.log`. Run `timeout 600s uv run train.py > run.log 2>&1`; exit 0 is necessary, exit 124 is the explicit total-time failure, and any other non-zero exit is a crash.
4. Run `grep -E '^(best_test_acc|final_test_acc|final_test_loss|training_seconds|total_seconds|startup_seconds|peak_vram_mb|num_epochs|num_steps|num_params):' run.log`; require every numeric summary key, `training_seconds` approximately 300.0, `total_seconds < 600`, and `num_params = 1,073,962`.
5. Parse `best_test_acc`; require `best_test_acc >= 93.65` for improvement. A completed lower result is valid `no-improvement`, not an infrastructure retry.
6. Check `grep -c '^  eval ep' run.log` against the number of unique parsed evaluation epochs; require equality so there was at most one evaluation per epoch. Require exactly one `augmentation_switch:` line with `randaugment->base`, progress near 80%, and eight workers stopped.
7. Run a separate one-line `uv run python -c` inspection that imports `train.ResNet`, reconstructs the same `requires_grad` and dimensionality partition without calling `main()`, and prints counts/elements; do not redirect it to `run.log` and do not modify `train.py` for diagnostics. Require groups `20/1,071,200` and `39/2,762`, total 1,073,962, decay values `{1e-4, 0}`, and the in-code coverage assertions. Also require `git diff --name-only` to contain only `train.py` and inspect `git diff -- train.py` for the planned optimizer-only change.
8. For mechanistic interpretation, parse the last evaluation before `augmentation_switch:`. Accuracy near or above EXP-007's 90.08% and clearly above EXP-008's 81.29% supports the claim that removing one-dimensional decay preserves strong-view fit; this observation cannot override the primary accuracy gate.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: project-root `run.log` final summary lines.
- Strong/weak trajectory: `grep -E '^  eval ep|^augmentation_switch:' run.log` — compare the 80% checkpoint, first weak checkpoint, best epoch, and final-tail slope with EXP-007/008.
- Lifecycle integrity: `grep -c '^augmentation_switch:' run.log` and parsed evaluation epochs — verify one switch and unique per-epoch evaluation.
