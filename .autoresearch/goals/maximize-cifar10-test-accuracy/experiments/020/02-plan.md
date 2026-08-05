# Plan EXP-020: Extend Mixup to 75 Percent
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Change only the mixup cutoff
- [x] Modify only `train.py`, changing `MIXUP_END_FRACTION` from 0.65 to exactly 0.75; preserve alpha 0.2, batch-shared sampling, model, optimizer, schedule, data, and evaluation.
- [x] Compile and audit the diff against `eb08811`; exactly one numeric constant changed and no other production file.

### Milestone 2: Pass deterministic semantic preflight
- [x] Create ignored `experiments/020/preflight.py` with dummy evaluator; verified the candidate source differs only at the cutoff constant and resolves to 0.75 while every other exported hyperparameter matches accepted.
- [x] Simulate progress boundaries; mixup is active below 0.75, hard labels at/above 0.75, exactly one transition occurs, and the learning-rate oracle is unchanged.

### Milestone 3: Execute once and verify
- [x] Confirm one H20, remove stale `run.log`, and execute exactly `timeout 600s uv run train.py > run.log 2>&1` once. A throughput preflight was unnecessary because only phase duration changed.
- [x] Require exit 0, finite summary, `training_seconds >=300.0`, total below 600, exactly one transition with `225.0 <= training_seconds <226.0`, unique evaluation epochs, 691,674 parameters, and no error signature.
- [x] Compare `best_test_acc` to 94.17; observed 93.82, a valid no-improvement that will not be adjusted or rerun.

## Code Changes
- **`train.py`**: change only `MIXUP_END_FRACTION = 0.65` to `MIXUP_END_FRACTION = 0.75`. This extends accepted batch-shared alpha-0.2 mixup from 195 to 225 counted seconds and retains a 75-second hard-label tail.

## Configuration Changes
- `MIXUP_END_FRACTION`: 0.65 -> 0.75, chosen before scoring from the measured 50%-to-65% directional result.
- All other model, optimizer, LR/floor, weight decay, batch, augmentation, mixup alpha/coefficient sharing, fixed seed 42, workers, and evaluation settings unchanged.

## Execution Environment
- Method: offline local command; no remote, network, package install, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20, local CIFAR-10, existing `uv`, eight persistent workers.
- Estimated runtime: semantic preflight under 1 minute; scored run about 342 seconds wall, hard limit 600 seconds.
- Log output: scored stdout/stderr exclusively in project-root `run.log`, retained through analysis.
- Tool skill: none.

## Abort Criteria
- Abort before scoring if the diff contains anything beyond the cutoff literal; cutoff semantics, LR oracle, compile, scope, or H20 checks fail.
- During scoring abort/classify on nonzero exit, timeout, OOM, error/non-finite output, missing summary, wrong parameter count, duplicate eval epoch, or missing/multiple transition. Do not change to 70%, tune alpha, or rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Query the results index; require baseline 94.07 at `eb08811`, threshold 94.17. Timeout 10 seconds.
2. Run H20, git scope/status/untracked/root-Python, exact diff, `git diff --check`, and `uv run python -m py_compile train.py` audits; require one H20 and only the one approved constant changed. Timeout 30 seconds.
3. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/020/preflight.py`; require exact source/hyperparameter equality except cutoff, correct 0.75 boundary, one simulated transition, and unchanged LR values at boundary probes. Timeout 60 seconds.
4. Remove stale `run.log`; execute exactly `timeout 600s uv run train.py > run.log 2>&1` once and never rerun after valid completion.
5. Parse using `rg`/`awk`; require one finite summary, `best_test_acc >=94.17`, `training_seconds >=300.0`, total <600, 691,674 parameters, exactly one transition with logged time in `[225.0,226.0)`, unique eval epochs, and no errors. CIFAR-10's 10,000 examples make 94.17 exactly +0.10 over 94.07. Stop at first necessary-condition failure.
6. Audit `git diff eb08811 -- train.py`; confirm the sole production difference remains `0.65 -> 0.75`.

### Informational Metrics (Optional)
- If necessary conditions pass, collect final accuracy/loss, timing, epochs, steps/passes, VRAM, and parameters from `run.log`.
