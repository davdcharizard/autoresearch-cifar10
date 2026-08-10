# Plan EXP-023: FP32 Width-3 ResNet-14 Depth-Width Rebalance
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and structurally verify the architecture
- [x] Create experiment branch `autoresearch/maximize-cifar10-best-test-accuracy-023` from integration commit `7c1e7d8`.
- [x] Modify only tracked `train.py`: set `NUM_BLOCKS = 2` and `WIDTH_MULTIPLIER = 3`, updating the nearby ResNet comment without altering model logic or any training/evaluation setting.
- [x] Assert ResNet-14 with stages 48/96/192, two blocks per stage, 13 convolutions, two unchanged Option-A transitions, `Linear(192,10)`, and exactly 1,540,474 trainable parameters.
- [x] Verify hard and CutMix probability-target forward/backward/update paths, FP32 parameter/buffer/momentum state, stage shapes, and side-effect-free finite evaluation.
- [x] Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, `git diff --check`, and `git diff --name-only`; only `train.py` may be tracked-modified.

### Milestone 2: Pass immutable-corpus safety and H20 timing gates
- [x] Create ignored experiment-local `preflight_rebalance.py`, resolving and prepending the project root before imports. Materialize and persist 200 exact post-N1/M7/CutMix batches once, with SHA-256 and hard/soft counts saved before assertions.
- [x] Train fresh accepted and candidate arms on the same persisted tensors for 200 steps. Controllers must instantiate control explicitly as `ResNet(3, 10, 2)` and candidate as `ResNet(2, 10, 3)`—never from mutated module constants. Require finite loss/state and no repeated candidate-only >95% prediction concentration while control remains <=90%; serialize evidence before assertions. Record loss EMA and size-normalized gradient/update norms as cross-architecture diagnostics, not vetoes.
- [x] Create ignored `timing_rebalance.py`; run one unscored device-conditioning subprocess, then five alternating fresh-process control/candidate pairs. Each arm uses 100 warmups and at least 1,000 complete synchronized measured steps split 80/20 across strong hard/soft and weak hard paths.
- [x] Require weighted candidate/control mean step ratio <=1.345, every pair <1.38, per-arm trial-mean CV <=2%, candidate p95 <=1.45x control mean, peak allocation <1.25 GiB, historical projected exposure >=20,000 updates, and projected total runtime <540 seconds.

### Milestone 3: Run the scored fixed-budget experiment once
- [x] Confirm exactly one idle NVIDIA H20 near 97,871 MiB and no competing compute process; confirm baseline remains 94.15%.
- [x] Ensure no `run*.log` exists, then launch exactly once with `timeout 600s uv run train.py > run.log 2>&1`; never use `tee` or stream the full log.
- [x] Monitor file growth, process/GPU status, and error patterns without dumping the log. Kill on non-finite loss, CUDA/resource/worker error, prolonged no-output plus no GPU activity, or 600-second timeout.
- [x] After exit, parse the summary and trajectory. Require one augmentation switch near 80%, eight workers stopped, hard weak-tail targets, at most one evaluation per epoch, no more than the accepted 19 looks, exactly 1,540,474 parameters, and at least 20,000 actual updates.

### Milestone 4: Verify and preserve mechanism evidence
- [ ] Require exit zero, numeric summary, approximately 300 counted training seconds from unchanged `TIME_BUDGET_S`, total <600 seconds, finite metrics, and `best_test_acc >=94.25%`.
- [x] Record switch accuracy versus 89.73%/87.08%, first-weak accuracy versus 93.16%, final NLL versus 0.1934, final/best gap, strong/weak step counts, epochs, and evaluation count even if the primary metric passes.
- [x] Write all commands, hardware, controller artifacts, decisions, failures, and inline metric values to `03-execute.md`; leave `run.log` until analysis captures evidence, then remove it before another experiment.

## Code Changes

- **`train.py`**: Change `NUM_BLOCKS = 3` to `2`, making `6 * NUM_BLOCKS + 2` report ResNet-14 and causing `_make_layer` to build exactly two blocks per stage.
- **`train.py`**: Change `WIDTH_MULTIPLIER = 2` to `3`, yielding stage channels 48, 96, and 192 and a 192-to-10 classifier.
- **`train.py`**: Update only the descriptive model comment from ResNet-20 to the generic CIFAR ResNet variant/ResNet-14 candidate. Do not modify `BasicBlock`, Option-A shortcut code, initialization, pooling, classifier logic, optimizer, data, schedule, timer, workers, evaluation, seed, or logging schema.
- **Ignored experiment artifacts**: Preflight/timing controllers and PT/JSON reports live only under experiment `023`; no other tracked path may change.

## Configuration Changes

- `NUM_BLOCKS`: `3 -> 2` (remove one block from each stage to reduce sequential depth).
- `WIDTH_MULTIPLIER`: `2 -> 3` (increase channels from 32/64/128 to 48/96/192).
- Model: width-2 ResNet-20, 1,073,962 parameters -> width-3 ResNet-14, exactly 1,540,474 parameters.
- Preserve batch 128, FP32, ordinary SGD momentum 0.9, all-parameter decay `1e-4`, LR 0.1 through 80%, 0.01-to-1e-4 cosine tail, N1/M7 plus p=0.5 alpha-1 CutMix through 80%, hard weak tail, seed 42, and fixed evaluator.

## Execution Environment

- Method: local single-process controllers and production run from project root.
- Resources: exactly one NVIDIA H20 with approximately 98 GB VRAM, eight existing loader workers, no new dependencies.
- Estimated runtime: safety plus paired timing approximately 2-4 minutes; production approximately 5.5-9 minutes total and must remain below 10 minutes.
- Log output: production stdout/stderr only in `run.log`; preflight/timing data in experiment-local reports. Never use `tee`.
- Tool skill: none; no remote job platform.
- Infrastructure safeguards: ignored controllers prepend the resolved project root; deterministic diagnostics launch with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; one unscored fresh subprocess conditions the device before timing.

## Abort Criteria

- Stop before timing/production on wrong graph/shape/parameter count, non-FP32 state, changed optimizer/data/evaluator settings, off-scope tracked changes, non-finite state, or repeated candidate-only >95% concentration while aligned control remains <=90%. Do not use 200-step loss/gradient ratios as pass/fail thresholds: width/depth/parameter-count changes make those cross-architecture scales non-equivalent.
- Stop before production if weighted step ratio >1.345, any pair >=1.38, timing CV >2%, p95 >1.45x control mean, peak allocation >=1.25 GiB, historical projected exposure <20,000, or projected total >=540 seconds. Do not rescue with precision, batch, LR, decay, another width/depth point, or extra evaluations.
- Stop production on non-finite loss, OOM/CUDA/worker error, silent loss of GPU activity, or 600-second timeout. Do not rerun a valid seed-42 job.

## Verification Protocol

### Verification Procedure

1. Query the moving baseline and require 94.15% (timeout 30 seconds):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   ```
   The formal threshold is 94.25%.

2. Confirm hardware before each GPU command:
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```
   Pass only with exactly one visible H20 near 97,871 MiB and no competing compute application.

3. Verify implementation and scope (timeout 60 seconds):
   ```bash
   uv run python -m py_compile train.py
   uv run ruff check train.py
   uv run ruff format --check train.py
   git diff --check
   git diff --name-only
   ```
   All exit zero; the final command prints only `train.py`.

4. After execution creates the registered controller, run safety preflight (timeout 180 seconds):
   ```bash
   CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 180s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/023/preflight_rebalance.py
   ```
   Pass only if the serialized report covers 200 exact batches, explicit `ResNet(3,10,2)` control and `ResNet(2,10,3)` candidate construction, required graph/parameter counts and target paths, finite state, and the repeated concentration gate. Loss EMA and normalized gradient/update ratios must be recorded but are diagnostic only.

5. Run paired timing (timeout 300 seconds):
   ```bash
   timeout 300s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/023/timing_rebalance.py
   ```
   Pass only with five alternating pairs after conditioning and ratio <=1.345, every pair <1.38, CV <=2%, p95 <=1.45x, peak <1.25 GiB, advisory projection >=20,000 steps, and total projection <540 seconds. Only actual production `num_steps` is conclusive exposure evidence.
   The candidate's estimated 1.456x MAC ratio makes timing no-go a likely legitimate result; do not relax the gate or infer accuracy failure if H20 kernel efficiency cannot close the gap to 1.345x.

6. Confirm no stale logs, then run once (timeout 600 seconds):
   ```bash
   find . -maxdepth 1 -type f -name 'run*.log' -print
   timeout 600s uv run train.py > run.log 2>&1
   ```
   The first command must print nothing; the production command exits zero before timeout.

7. Extract the summary without streaming the log:
   ```bash
   grep "^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:" run.log
   ```
   If incomplete, inspect only `tail -n 50 run.log`. Parse numerically. Pass only if `best_test_acc >=94.25`, `training_seconds` is near the unchanged 300-second budget, `total_seconds <600`, `num_steps >=20000`, `num_params ==1540474`, and all values are finite.

8. Verify integrity from log/artifacts: exactly one near-80% loader switch, eight stopped workers, hard weak targets, seed 42, no full-run retry, at-most-once-per-epoch unique evaluations, and evaluation count <=19. Record switch, first-weak, NLL, exposure, and evaluation diagnostics regardless of pass/fail so a near-floor miss is not overinterpreted.
   A timing pass followed by an accuracy miss is a likely outcome because the candidate simultaneously loses depth and roughly one quarter of updates; analyze it as the net fixed-time architecture result, not as isolated evidence that width is saturated.

### Informational Metrics (Optional)

- final_test_acc: `grep '^final_test_acc:' run.log`.
- final_test_loss: `grep '^final_test_loss:' run.log`.
- training_seconds: `grep '^training_seconds:' run.log`.
- total_seconds: `grep '^total_seconds:' run.log`.
- startup_seconds: `grep '^startup_seconds:' run.log`.
- peak_vram_mb: `grep '^peak_vram_mb:' run.log`.
- num_epochs: `grep '^num_epochs:' run.log`.
- num_steps: `grep '^num_steps:' run.log`.
- num_params: `grep '^num_params:' run.log`.
- mechanism diagnostics: switch accuracy, first-weak accuracy, final NLL, final/best gap, strong/weak exposure, and evaluation count from `run.log` and controller reports.
