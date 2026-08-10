# Plan EXP-003: Modest Label Smoothing
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Isolated loss intervention implemented
- [x] Create experiment branch `autoresearch/maximize-cifar10-best-test-accuracy-003` from integration commit `5016cc4`.
- [x] Add `LABEL_SMOOTHING = 0.05` beside the training constants in `train.py`.
- [x] Pass only that constant as `label_smoothing=LABEL_SMOOTHING` to the existing training `F.cross_entropy`; do not change evaluator, model, transforms, optimizer, schedule, loader, evaluation cadence, seed, or any other tracked file.
- [x] Verify the locked PyTorch exposes the `label_smoothing` argument using `uv run python -c "import inspect; import torch.nn.functional as F; assert 'label_smoothing' in inspect.signature(F.cross_entropy).parameters"`.
- [x] Verify with `uv run python -m py_compile train.py`, `uv run ruff check train.py`, and the repository pre-commit checks.
- [x] Verify `git diff -- train.py` contains only the constant and training-loss argument, and `git status --short` contains no other tracked modification.

### Milestone 2: Fixed-budget run launched and monitored
- [x] Confirm a single selected GPU is an NVIDIA H20 with approximately 98 GB, negligible used memory, and no compute process using `nvidia-smi`; expose only that idle index through `CUDA_VISIBLE_DEVICES`.
- [x] Confirm `run.log` and completed renamed log variants do not exist; preserve the intentional untracked `data/` cache.
- [x] Run exactly once with fixed seed 42 under a 600-second supervisor: `CUDA_VISIBLE_DEVICES=<index> timeout 600s uv run train.py > run.log 2>&1`.
- [x] Monitor process liveness and compact log signals without streaming the training log; record PID, start/end UTC timestamps, selected GPU, and exit code in `03-execute.md`.

### Milestone 3: Result integrity and threshold verified
- [x] Parse the unique final summary from `run.log` and require `best_test_acc >= 91.93%` against moving baseline `91.83%`.
- [ ] Require exit code 0, all ten numeric summary fields exactly once, `300.0 <= training_seconds < 310.0`, and `total_seconds < 600`.
- [ ] Confirm evaluation epochs are unique, no epoch is evaluated more than once, at least one evaluation occurred, and the final evaluated epoch equals `num_epochs`.
- [ ] Only after all necessary conditions pass, collect the informational metrics and compare throughput/evaluation trajectory with EXP-002; do not compare the smoothed training-loss magnitude with EXP-002's hard-label loss.

## Code Changes
- **`train.py`**: Add `LABEL_SMOOTHING = 0.05` and pass it to the existing training-only `F.cross_entropy`. This tests target-space regularization while preserving every validated EXP-002 mechanism and the fixed evaluator. The built-in PyTorch implementation avoids dependencies and should have negligible throughput or VRAM impact. The principal risk is underfitting or an NLL/calibration improvement without the required top-1 gain.

## Configuration Changes
- `LABEL_SMOOTHING`: implicit `0.0` -> `0.05` (conservative strength below the commonly used `0.1`, chosen to limit over-regularization in the approximately 100-epoch fixed-time horizon).
- All EXP-002 settings remain fixed: `BATCH_SIZE=128`, `LR=0.1`, `LR_HOLD_FRACTION=0.8`, `ANNEAL_START_LR=0.01`, `MIN_LR=1e-4`, standard momentum `0.9`, weight decay `1e-4`, seed 42, persistent workers, and current evaluation checkpoints/dense tail.
- Expected smoothed-loss floor is approximately `0.28` for ten classes, so training-loss values are not directly comparable with the prior hard-label run. Fixed-evaluator test loss remains comparable.

## Execution Environment
- Method: local one-run execution from the project root with `CUDA_VISIBLE_DEVICES=<selected-H20-index> timeout 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one NVIDIA H20 GPU with approximately 98 GB VRAM; existing environment and dependency lock; cached CIFAR-10 data under untracked `data/`.
- Estimated runtime: approximately 300 seconds counted training and 335-345 seconds end to end; hard stop at 600 seconds.
- Log output: all stdout/stderr redirected only to project-root `run.log`; monitoring uses process state, compact `rg`/`tail` checks, and never `tee` or full-log context streaming.
- Tool skill: local execution; no job-submission skill or external service.

## Abort Criteria
- Abort before launch if the selected device is not an NVIDIA H20 with approximately 98 GB, is occupied by another compute process or material memory allocation, if more than one GPU is exposed to training, if a stale completed experiment log exists, or if tracked files outside `train.py` are modified.
- Kill and classify as failure if the supervisor reaches 600 seconds, the process exits non-zero, or `run.log` contains an unrecoverable Python traceback, CUDA OOM, illegal-memory-access error, or dependency/data integrity error.
- Treat no training-step output after 120 seconds as a likely startup/data-loader failure: inspect only the compact log tail, and abort if the process is not making progress. Do not change the experimental setting in response.
- Do not abort based on intermediate test accuracy or the expected elevated smoothed training loss; the predeclared primary verdict uses the completed run.

## Verification Protocol

### Verification Procedure

1. Query the baseline immediately before launch:

   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   ```

   Require `baseline=91.83` and `baseline_commit=5016cc4`; otherwise recompute the threshold as baseline plus `0.10` before running rather than using stale values.

2. Confirm hardware and scope:

   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
   nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
   git diff -- train.py
   git status --short
   ```

   Select one idle H20 index, require no compute process and negligible used memory on it, and expose only it through `CUDA_VISIBLE_DEVICES`. Require the tracked diff to contain only the planned `train.py` loss change. The untracked `data/` cache is allowed and must not be committed or removed.

3. Launch under the 600-second timeout and record the shell exit code. A valid run requires code `0`; timeout code `124` is a failure.

4. Extract the primary result and compact summary:

   ```bash
   rg '^best_test_acc:|^peak_vram_mb:' run.log
   rg '^best_test_acc:|^final_test_acc:|^final_test_loss:|^training_seconds:|^total_seconds:|^startup_seconds:|^peak_vram_mb:|^num_epochs:|^num_steps:|^num_params:' run.log
   ```

   Parse numeric values, require each of the ten keys exactly once and finite, and apply conditions in goal-file order. First require `best_test_acc >= 91.93`; if it fails, classify `no-improvement` and skip remaining formal necessary-condition checks. If it passes, require clean exit and the complete expected numeric summary. Then require `300.0 <= training_seconds < 310.0` and `total_seconds < 600`. The upper sanity bound allows the final timed step to overshoot the 300-second loop boundary without false-failing the run.

5. Verify evaluation integrity after the accuracy condition passes by parsing lines beginning with `eval ep`: require at least one, no duplicate epoch number, and maximum/final evaluated epoch equal to summary `num_epochs`. This demonstrates no more than one validation per epoch and terminal coverage.

6. Timeout for the complete verification workflow is 600 seconds for execution plus 30 seconds for parsing. Do not rerun a valid completed experiment, change seed, or tune smoothing based on the result.

7. Interpret the binary verdict narrowly. A `0.10`-point boundary gain is exactly ten additional correct examples on the 10,000-image evaluator and satisfies the declared protocol, but one fixed-seed result alone does not establish a broad causal effect; record this limitation in execution and analysis.

### Informational Metrics (Optional)
- final_test_acc (%): final summary key `final_test_acc` in `run.log`.
- final_test_loss: final summary key `final_test_loss` in `run.log`; directly comparable because the evaluator remains hard-label and fixed.
- training_seconds (s): final summary key `training_seconds`.
- total_seconds (s): final summary key `total_seconds`.
- startup_seconds (s): final summary key `startup_seconds`.
- peak_vram_mb (MB): final summary key `peak_vram_mb`.
- num_epochs: final summary key `num_epochs`.
- num_steps: final summary key `num_steps`; compare with EXP-002's 38,629 only after all necessary conditions pass.
- num_params: final summary key `num_params`; require informational equality with EXP-002's 269,722 to confirm architecture isolation.
