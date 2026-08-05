# Plan EXP-008: Decoupled Cosine-to-Zero Floor
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Isolated schedule change and static verification
- [x] Create experiment branch `autoresearch/maximize-cifar10-test-accuracy-008` from clean accepted commit `eb08811`.
- [x] Modify only `train.py`: separate the accepted warmup-start LR from the post-warmup cosine floor, leaving model, optimizer family, decay groups, mixup, seed, loader, and evaluator cadence unchanged.
- [x] Run `uv run python -m py_compile train.py` and exact schedule assertions at 0%, 5%, 65%, 90%, 95%, and 100% counted progress.
- [x] Review the complete `train.py` diff against an exact four-edit allowlist, verify `prepare.py` remains byte-identical to `eb08811`, and verify the optimizer is initialized with LR 0.002.

### Milestone 2: Single fixed-seed scored execution
- [x] Confirm exactly one NVIDIA H20 is visible and no stale `run.log` exists.
- [x] Run exactly once with `timeout 600s uv run train.py > run.log 2>&1`; do not reroll or tune from an interim result.
- [x] Monitor locally for startup failure, non-finite loss, GPU/resource errors, or timeout and capture the process exit status.

### Milestone 3: Result and protocol audit
- [x] Require logged `Device: cuda`, a complete final summary with rounded `training_seconds: 300.0`, `num_steps < 64000`, at most 600 total seconds, and no more than one evaluation per epoch.
- [x] Extract primary and informational metrics, compute data passes from `num_steps * 256 / 50000`, and compare exposure against EXP-002's 27,735 steps / 141.9 passes.
- [x] Confirm mixup disables exactly once near 195 counted seconds with LR about 0.0598, continuous matrix decay remains in code, and accept only `best_test_acc >= 94.17%`.

## Code Changes
- **`train.py`**: add `WARMUP_START_LR = 0.002`, change `MIN_LR` from `0.002` to `0.0`, use `WARMUP_START_LR` in the linear warmup formula, and initialize SGD with `lr=WARMUP_START_LR`. This preserves the accepted 0-5% trajectory and isolates only the post-warmup cosine endpoint. No other file or training behavior changes.

## Configuration Changes
- `WARMUP_START_LR`: new constant at `0.002` (preserves accepted optimizer initialization and 0-5% warmup).
- `MIN_LR`: `0.002 -> 0.0` (post-warmup cosine reaches zero at 300 counted seconds).
- `LR`, `WARMUP_FRACTION`, `MOMENTUM`, `WEIGHT_DECAY`, `MIXUP_ALPHA`, `MIXUP_END_FRACTION`, architecture, batch size, seed, and evaluation cadence: unchanged.

## Execution Environment
- Method: local, offline execution from the project root using `timeout 600s uv run train.py > run.log 2>&1`.
- Resources: one NVIDIA H20; existing local CIFAR-10 data and declared dependencies only; no network, remote job, or GitHub operation.
- Estimated runtime: about 341 seconds total, including 300 seconds of counted training and periodic evaluation; hard timeout 600 seconds.
- Log output: complete stdout/stderr redirected to project-root `run.log`, which is the source of truth during execution and is removed after analysis.
- Tool skill: none; execution is fully local.

## Abort Criteria
- Stop and classify the run if `timeout` returns 124 or wall time reaches 600 seconds.
- Stop on a Python traceback, CUDA error, out-of-memory error, missing H20, or non-finite training loss.
- Do not stop for low interim accuracy: sparse evaluations and the preregistered one-run rule require the completed trajectory unless an infrastructure or numerical failure occurs.
- Do not retry, reroll, or alter the floor after a valid run below 94.17%; that is a no-improvement result.

## Verification Protocol

### Verification Procedure
1. Query the accepted baseline with `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require `baseline=94.07`, hence success threshold `94.17`.
2. Before execution, run `nvidia-smi --query-gpu=name --format=csv,noheader`; require exactly one visible line containing `NVIDIA H20`. Run `git diff --name-only eb08811`; require only `train.py`, and run `git diff --exit-code eb08811 -- prepare.py`; require exit 0. Review the complete `git diff eb08811 -- train.py` and reject any hunk outside this exact allowlist: add `WARMUP_START_LR = 0.002`, set `MIN_LR = 0.0`, substitute `WARMUP_START_LR` for `MIN_LR` in the warmup return, and substitute `WARMUP_START_LR` for `MIN_LR` in SGD initialization.
3. Run `uv run python -m py_compile train.py`. Then import the production `learning_rate` and `TIME_BUDGET_S`, call `learning_rate(progress * TIME_BUDGET_S)`, and assert within `1e-9`: progress 0%=0.002, 2.5%=0.101, 4.9%=0.19604, 5%=0.2, 65%=0.0598304575, 90%=0.0054182758, 95%=0.0013638697, 100%=0.0. The interior and immediately-pre-boundary values prove the full warmup formula remains accepted. Also inspect the optimizer construction diff to require `lr=WARMUP_START_LR`; the complete-diff allowlist ensures both decay groups remain unchanged (`5e-4` matrix, zero bias/norm).
4. Remove any stale log, then execute the only scored run: `rm -f run.log` followed by `timeout 600s uv run train.py > run.log 2>&1`. Require exit status 0; otherwise inspect `tail -n 50 run.log` and classify timeout/crash without rerun.
5. Require exactly one `Device: cuda` line in `run.log`, tying the inventory check to the actual execution device. Parse the final summary and require present `best_test_acc`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params`. Because training stops after the first step crossing the time budget and the summary rounds to one decimal, require reported `training_seconds=300.0`, `num_steps < 64000`, `total_seconds <= 600`, finite loss, and `num_params=691674`.
6. Audit `run.log` evaluation lines: each evaluated epoch must be unique, so validation occurs at most once per epoch. Require exactly one `Mixup disabled` line near 195.0 training seconds / 65.0% and transition LR rounding to 0.0598. Compare steps and computed passes to accepted exposure before mechanism attribution.
7. Necessary-condition verdict: require `best_test_acc >= 94.17%`. A lower loss or final-at-best behavior cannot rescue a lower accuracy. On completion, retain `run.log` only through analysis and then remove it before the next experiment.

### Informational Metrics (Optional)
- `peak_vram_mb`: final summary line in `run.log`.
- `final_test_acc`: final summary line in `run.log`; compare whether final equals or trails best.
- `final_test_loss`: final summary line in `run.log`; compare to accepted 0.2432.
- `training_seconds`, `total_seconds`: final summary lines in `run.log`.
- `num_epochs`, `num_steps`, `num_params`: final summary lines in `run.log`; compute passes as `num_steps * 256 / 50000`.
- Transition LR/time: the single `Mixup disabled` line; expected approximately 195 seconds and LR 0.0598.
