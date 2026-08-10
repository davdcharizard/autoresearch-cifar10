# Plan EXP-022: Lookahead-Wrapped Momentum SGD
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the isolated Lookahead intervention
- [x] Create the experiment branch through the execute-phase workflow and confirm the starting commit is the 94.15% integration baseline `7c1e7d8`.
- [x] Modify only tracked `train.py`: add fixed Lookahead configuration (`k=5`, `alpha=0.5`), create detached slow parameter copies after SGD construction, and synchronize after every fifth ordinary optimizer step with fused foreach operations while retaining momentum state.
- [x] Keep synchronization arithmetic before the existing CUDA synchronization so its cost is included in `total_training_time`; preserve every model, data, schedule, worker, timer, seed, and evaluation setting.
- [x] Add concise diagnostics (`lookahead_syncs` and final mean relative fast/slow distance) without adding evaluations or changing the required numeric summary.
- [x] Verify syntax/style and scope with `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `git diff --check`, and `git diff --name-only`; the only tracked modified path must be `train.py`.

### Milestone 2: Prove recurrence, safety, and counted-cost viability
- [x] Create ignored experiment-local controller `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight_lookahead.py`; resolve and prepend the project root before importing `train` to avoid the recorded path-import failure.
- [x] Materialize and serialize at least 200 exact post-transform production-distribution batches once, including hard and CutMix probability targets; both aligned arms must consume the same tensors rather than independently replay forkserver seeds.
- [x] Starting from byte-identical seed-42 models and SGD state, prove steps 1-4 match ordinary SGD, prove step-5 interpolation equals a manual reference, and prove momentum buffers persist and drive step 6.
- [x] Through step 50, record every synchronization and following update: prediction histogram, loss, total update norm, momentum norm, fast/slow distance, five-step committed displacement versus aligned SGD, and an isolated zero-data-gradient decay-only recurrence. Continue the exact-corpus safety comparison through 200 steps.
- [ ] Require finite losses, gradients, parameters, BatchNorm buffers, slow tensors, and momentum state; no candidate-only prediction concentration above 95%; candidate terminal loss EMA no more than 1.5 times control; and unchanged parameter membership/count and RNG consumption outside the intended optimizer trajectory.
- [ ] Create ignored controller `.autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/timing_lookahead.py` and run one unscored device-conditioning subprocess, then five alternating fresh-process control/candidate H20 pairs with 100 warmup and at least 1,000 measured production-region steps per arm.
- [ ] Measure strong and weak production regions separately and combine their current-machine ratios with fixed 80/20 weights. Require weighted candidate/control mean step ratio at most 1.01, the advisory historical projection `floor(26898 * control_mean / candidate_mean) >= 26629`, trial-mean CV at most 2%, peak allocation below 650 MiB, and projected total runtime below 540 seconds. Save diagnostics before assertions; only the full production `num_steps` is conclusive exposure evidence.

### Milestone 3: Run the fixed-budget experiment once
- [ ] Confirm exactly one visible NVIDIA H20 with approximately 97,871 MiB memory and no competing compute process.
- [ ] Remove stale `run.log` or renamed run-log variants, then run exactly once with `timeout 600s uv run train.py > run.log 2>&1`; do not use `tee` or stream the full log.
- [ ] Monitor file growth/process status without reading the full log; kill and classify failure if wall time reaches 10 minutes, CUDA/resource errors occur, the process exits without a summary, or the log stops growing for an abnormal interval while the process is not using the GPU.
- [ ] Extract the final summary and Lookahead diagnostics only after exit. Confirm one augmentation switch, eight workers stopped, hard weak-tail targets, no more than one evaluation per epoch, and no candidate-specific additional evaluation look.

### Milestone 4: Verify and hand results to analysis
- [ ] Query the moving baseline from `04-results.tsv` and confirm it remains 94.15%; calculate the acceptance threshold as 94.25%.
- [ ] Require exit zero, numeric summary, approximately 300 counted training seconds, total runtime below 600 seconds, at least 26,629 steps, finite metrics, and `best_test_acc >= 94.25%`.
- [ ] Record full execution decisions, preflight/timing artifact paths, failures or dead ends, command, hardware, and parsed metrics in `03-execute.md`; remove `run.log` after the analysis phase has captured all required evidence.

## Code Changes

- **`train.py`**: Add `LOOKAHEAD_K = 5` and `LOOKAHEAD_ALPHA = 0.5`. Immediately after constructing accepted SGD, clone each trainable parameter with `parameter.detach().clone()` into an ordered slow-weight list and initialize `lookahead_syncs = 0`.
- **`train.py`**: Immediately after every accepted `optimizer.step()` and before the existing `torch.cuda.synchronize()`, test `(step + 1) % LOOKAHEAD_K == 0`. Under `torch.no_grad()`, use the verified PyTorch 2.9.1 APIs `torch._foreach_lerp_(slow_weights, fast_parameters, LOOKAHEAD_ALPHA)` followed by `torch._foreach_copy_(fast_parameters, slow_weights)`; increment the synchronization count. This avoids roughly 100+ tiny per-tensor launches every sync. Do not touch optimizer state, BatchNorm buffers, non-parameter buffers, gradients, or LR groups.
- **`train.py`**: Keep `torch.cuda.synchronize()`, `dt`, `total_training_time`, and `step += 1` in their accepted order after the new block so sync cost is counted and syncs occur on completed steps 5, 10, 15, etc.; an incomplete final segment is allowed and must yield `floor(num_steps/5)` syncs. Print the synchronization count and final mean relative fast/slow distance near the final summary, and append `lookahead_offset: {step % LOOKAHEAD_K}` to each existing evaluation line so weight/BN-buffer phase is attributable without adding evaluations. Preserve all required summary keys.
- **Experiment-local ignored controllers**: The execute phase may add preflight/timing scripts and JSON/PT artifacts only under experiment `022`; these are metadata, not tracked project code. They must not modify `prepare.py`, the evaluator, package metadata, or production settings.

## Configuration Changes

- Lookahead synchronization period: absent -> `5` (paper-supported robust default and reviewer-selected operating point).
- Lookahead slow interpolation: absent -> `0.5` (paper-supported fixed default; no tuning or rescue within EXP-022).
- Inner optimizer state: ordinary accepted momentum SGD -> the same SGD with momentum buffers retained across synchronization.
- Everything else remains accepted: width 2, batch 128, LR 0.1 through 80%, 0.01-to-1e-4 cosine tail, momentum 0.9, coupled all-parameter decay `1e-4`, N1/M7 plus alpha-1 CutMix at p=0.5 through 80%, hard weak tail, seed 42, and unchanged evaluator/checkpoints.

## Execution Environment

- Method: local single-process run from the project root; production command `timeout 600s uv run train.py > run.log 2>&1`.
- Resources: exactly one visible NVIDIA H20 with approximately 98 GB VRAM (expected `memory.total` near 97,871 MiB), eight existing DataLoader workers, no new dependencies.
- Estimated runtime: preflight and paired timing approximately 2-3 minutes total; production training approximately 5.5 minutes and must remain below 10 minutes.
- Log output: all production stdout/stderr only in `run.log`; preflight/timing summaries in experiment-local JSON/text artifacts. Never use `tee`.
- Tool skill: none; execution is local, not a Volcano job.
- Infrastructure safeguards: experiment-local scripts prepend the resolved project root to `sys.path`; any deterministic diagnostic subprocess sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launch; one unscored conditioning process precedes paired timing.

## Abort Criteria

- Stop before production if any tracked path other than `train.py` changes, required accepted settings differ, the recurrence/reference test fails, slow/fast parameter ordering differs, momentum is reset, or any non-finite state appears.
- Stop before production if a candidate-only class exceeds 95% of predictions on the exact-corpus safety run, candidate terminal loss EMA exceeds 1.5 times control, or a synchronization/following update shows catastrophic loss or update-norm escalation. The exact-corpus test is a representative production-distribution screen, not proof of the future scored seed-42 trajectory; production monitoring remains authoritative. A merely smaller committed displacement is diagnostic unless it causes one of these safety failures; do not compensate LR or decay.
- Stop before production if paired timing exceeds 1.01 candidate/control, projected exposure is below 26,629, timing CV exceeds 2%, peak allocation is at least 650 MiB, or projected total runtime is at least 540 seconds.
- Stop production on non-finite loss, CUDA/OOM/resource error, unexpected worker failure, missing progress with no GPU activity, or 600-second timeout. Do not rerun a valid seed-42 experiment and do not tune `k`, alpha, LR, decay, or momentum as a rescue.

## Verification Protocol

### Verification Procedure

1. Confirm the baseline and threshold (command exists at the shared skill path):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv
   ```
   Pass only if `baseline=94.15`; the required experiment threshold is `94.25`.

2. Confirm hardware immediately before every GPU controller/production command:
   ```bash
   nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```
   Pass only with exactly one visible NVIDIA H20 near 97,871 MiB and no competing compute application. Treat unavailable/mismatched hardware as infrastructure failure, not an accuracy result.

3. After Milestone 1, verify code and scope (timeout: 60 seconds):
   ```bash
   uv run python -m py_compile train.py
   uv run ruff check train.py
   git diff --check
   git diff --name-only
   ```
   All commands must exit zero, and `git diff --name-only` must print only `train.py`.

4. After the execute phase creates the registered controller, run semantic/safety preflight (timeout: 180 seconds):
   ```bash
   CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 180s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/preflight_lookahead.py
   ```
   Pass only if its saved report records 200 aligned exact-corpus steps and every recurrence, finite-state, concentration, loss-ratio, parameter/RNG, and step-50 diagnostic condition from Milestone 2.

5. Run paired timing (timeout: 240 seconds):
   ```bash
   timeout 240s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/022/timing_lookahead.py
   ```
   Pass only if the saved report contains separate strong/weak measurements for five alternating pairs after conditioning and satisfies weighted ratio <=1.01, advisory historical projected steps >=26,629, CV <=2%, peak <650 MiB, and projected total <540 seconds. This projection uses EXP-010 only to set a conservative no-go threshold; it does not establish actual exposure on the current machine.

6. Ensure no stale logs, then run production once (timeout: 600 seconds):
   ```bash
   find . -maxdepth 1 -type f -name 'run*.log' -print
   timeout 600s uv run train.py > run.log 2>&1
   ```
   The first command must print nothing before launch. The run command must exit zero within 600 seconds.

7. Extract necessary results without streaming the full log:
   ```bash
   grep "^best_test_acc:\|^peak_vram_mb:\|^training_seconds:\|^total_seconds:\|^num_steps:\|^lookahead_syncs:" run.log
   ```
   If empty or incomplete, inspect only `tail -n 50 run.log` and classify crash/failure. First confirm from the unchanged `prepare.py` import that `TIME_BUDGET_S == 300`; `training_seconds` near 300 then checks timer/protocol integrity but is not exposure evidence. Pass only if `best_test_acc >= 94.25`, `total_seconds < 600`, actual `num_steps >= 26629`, Lookahead sync count equals `floor(num_steps/5)`, and peak VRAM is finite. Parse all values numerically rather than lexicographically.

8. Verify protocol integrity from log/artifacts: all validation events occur at most once per epoch; evaluation look count does not exceed the accepted 19; every evaluation records `lookahead_offset` so parameter/BatchNorm-buffer phase is visible; the augmentation switch occurs once near 80%; exactly eight strong-loader workers stop; all post-switch targets are hard; seed is 42; and no full-run retry occurred. Any failure makes the result invalid even if top-1 clears. Treat evaluation-offset correlation as mechanism evidence only—never move or add an evaluation to favor synchronized weights.

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
- mechanism diagnostics: `lookahead_syncs`, fast/slow distance, switch accuracy versus 89.73% and 87.08%, first weak accuracy versus 93.16%, final/best gap, and NLL versus 0.1934 from `run.log` plus preflight reports.
