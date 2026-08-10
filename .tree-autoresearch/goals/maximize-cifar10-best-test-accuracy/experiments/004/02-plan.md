# Plan EXP-004: Clean-Finish Periodic SAM
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Implement and prove the two-pass update path
- [x] Add the fixed SAM configuration, cadence predicate, audit counters, and startup/final logging to `train.py`.
- [x] Add helpers for FP32 global gradient norm, preallocated exact parameter snapshots, perturbation/restoration, and temporary BatchNorm tracking suppression.
- [x] Preserve the existing first-pass loss branches without refactoring; compute the SAM second-pass clean loss in its own autocast context.
- [x] Run GPU helper smokes proving cadence, perturbation radius, exact restoration, one BatchNorm buffer update, one optimizer update, CUDA RNG parity, and a distinct perturbed BF16 loss.
- [x] Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `git diff --check`, and inspect the complete tracked diff.

### Milestone 2: Execute one bounded experiment on physical GPU 0
- [x] Confirm GPU 0 is an NVIDIA H20 with approximately 98 GB memory and is available.
- [x] Remove any stale `run.log`, then launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
- [x] Monitor compact health signals without using intermediate accuracy to stop, retry, or tune the run.
- [x] Confirm SAM remains inactive before 75% charged progress and approaches a 0.50 applied/eligible ratio afterward.

### Milestone 3: Verify and preserve the result
- [x] Require exit code 0 and a complete final summary after approximately 300 charged seconds and less than 600 total seconds.
- [x] Parse `best_test_acc` and compare it with the 95.23% parent; pass requires at least 95.33%.
- [x] Parse all informational metrics and mechanism counters, record them inline in `03-execute.md`, then remove `run.log`.
- [x] Confirm only `train.py` differs from the parent and protected files are unchanged.

## Code Changes
- **`train.py`**: add `SAM_RHO=0.05`, `SAM_START=0.75`, `SAM_PERIOD=2`, and `SAM_EPS=1e-12`. Include them in the existing configuration line.
- **`train.py`**: define scheduling against the upcoming one-based optimizer step: SAM applies exactly when `progress >= SAM_START and (step + 1) % SAM_PERIOD == 0`. Assert a scheduled SAM batch has no CutMix paired target.
- **`train.py`**: collect trainable parameters once and preallocate one `torch.empty_like` snapshot per parameter before the charged timer. Snapshots are not optimizer parameters and add about 11 MiB.
- **`train.py`**: on a SAM pulse, save CUDA RNG state immediately before the first forward, run the unchanged clean loss/backward in one autocast context, compute the global L2 gradient norm in FP32, require `torch.isfinite(norm)` and `norm > 0.0`, copy parameters to snapshots, and perturb by `rho * grad / (norm + eps)` under `torch.no_grad()`. `SAM_EPS` is used only in the denominator; there is no ambiguous near-zero threshold.
- **`train.py`**: zero first-pass gradients, restore the saved CUDA RNG state, disable `track_running_stats` on every BatchNorm module for only the second training-mode forward, and run hard-label cross-entropy in a new autocast context so cached BF16 casts cannot hide the perturbation. Then restore BatchNorm flags and exact parameter snapshots before the sole `optimizer.step()`.
- **`train.py`**: make restoration stage-aware. First backward/norm failures occur before mutation and need no restoration. After snapshots are populated, a `parameters_perturbed` flag guards exact restore; after BatchNorm flags are captured, a separate `bn_tracking_disabled` flag guards their restore. Perturbation helper failures restore snapshots before re-raising. Any failure aborts after restoration rather than silently skipping a pulse.
- **`train.py`**: preserve the ordinary EXP-002 path bit-for-logic when SAM is not scheduled. All second-pass work, snapshot copies, perturbation, restoration, and the optimizer step stay between the existing `t0` and CUDA synchronization so they are charged.
- **`train.py`**: add `sam_eligible_batches`, `sam_applied_batches`, `sam_first_step`, and `sam_first_progress`. Increment eligible on every batch at/above 75%; increment applied only after a successful SAM optimizer update. Emit `sam: applied=... eligible=... ratio=... first_step=... first_progress=...` and include compact counters in periodic progress output after activation without changing required summary keys.

The evaluator, validation cadence, CutMix helper/RNG streams, data transforms, architecture, LR/drop-path schedules, global seed, optimizer hyperparameters, and summary metric flow remain unchanged. No other tracked file is modified.

## Configuration Changes
- `SAM_RHO`: absent -> `0.05` (standard small plain-SAM perturbation used as a fixed, untuned hypothesis).
- `SAM_START`: absent -> `0.75` (begins exactly when CutMix ends and the clean low-LR refinement phase starts).
- `SAM_PERIOD`: absent -> `2` (every second eligible one-based step; stronger than the initial period-four proposal after Claude identified momentum dilution).
- `SAM_EPS`: absent -> `1e-12` (denominator guard only; a nonfinite or exactly zero norm aborts).
- Expected optimizer exposure: 27,950 parent steps -> approximately 25,000-25,800 steps and 2,000-2,400 SAM pulses after including RNG, norm, snapshot, and Python dispatch overhead. These are informational expectations. Mechanism integrity requires at least 24,000 steps and 1,800 successful SAM pulses; lower exposure means the planned dose was not executed faithfully even if the process completed.

## Execution Environment
- Method: local process from repository root: `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
- Resources: physical GPU 0 only, NVIDIA H20 with 97,871 MiB; no use of GPU 1.
- Estimated runtime: 460-500 seconds end to end, including epoch evaluations; hard outer limit 600 seconds.
- Log output: all stdout/stderr redirected to fresh `run.log`; monitoring uses only compact `tail`/`rg` extracts. The log is deleted after metrics are copied into `03-execute.md`.
- Tool skill: none; execution is local.

## Abort Criteria
- The process reaches the 600-second timeout, exits nonzero, or emits a traceback, CUDA error, OOM, nonfinite loss/gradient, or parameter/BatchNorm restoration assertion.
- Startup does not report CUDA and the exact fixed SAM configuration within 60 seconds, or GPU inspection shows the run is not isolated to physical GPU 0.
- `run.log` file size and modification time do not advance for 120 seconds outside evaluation, indicating a hang. Monitoring normalizes carriage returns with `tr '\r' '\n'` rather than relying on newline-oriented `tail` alone.
- Any SAM application occurs before `progress >= 0.75`, any SAM batch overlaps CutMix, or the one-based cadence differs from every second eligible step.
- After at least 100 eligible batches, `sam_applied_batches / sam_eligible_batches` is outside `[0.45, 0.55]`.
- Do not abort, retry, or tune based on intermediate test accuracy. A low metric is a valid completed result.

## Verification Protocol

### Verification Procedure

1. Reconfirm the parent and threshold:
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show \
     .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 002
   ```
   Parent must be 95.23%; the experiment passes accuracy only if `best_test_acc >= 95.33%`.
2. Confirm physical device identity and availability before launch:
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
   ```
   Index 0 must report NVIDIA H20 and approximately 97,871 MiB. Launch with `CUDA_VISIBLE_DEVICES=0` regardless of GPU 1 state.
3. With the local CIFAR payload present, run focused GPU smokes with `CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY' ... PY`; importing `train.py` constructs the evaluator, so no device-independent CPU smoke is claimed:
   - synthetic schedule pairs must prove no SAM below 0.75 and exactly even upcoming one-based steps at/above 0.75;
   - a tiny BatchNorm/stochastic module must perform two forwards but increment `num_batches_tracked` once;
   - the perturbation L2 norm must equal 0.05 within tolerance, snapshots must restore parameters exactly before the optimizer step, and momentum state must update once;
   - first and second passes must use separate autocast contexts, and with fixed RNG the perturbed second-pass loss or gradients must differ from an unperturbed replay so BF16 cast caching cannot nullify SAM;
   - fixed-logit clean and CutMix losses from the unchanged first-pass branches must match the parent formulas exactly;
   - from the same initial CUDA RNG state, a SAM two-pass forward with replay must leave the same final RNG state as one ordinary forward;
   - injected failures after perturbation and during the second pass must restore all parameters and BatchNorm flags; a pre-snapshot failure must be a no-op-safe abort.
4. Launch once with the command in Execution Environment. The outer timeout is 600 seconds. If metric extraction is empty after process completion, classify as crash and inspect `tail -n 50 run.log`.
5. Parse the full summary:
   ```bash
   rg '^best_test_acc:|^final_test_acc:|^final_test_loss:|^training_seconds:|^total_seconds:|^startup_seconds:|^peak_vram_mb:|^num_epochs:|^num_steps:|^num_params:|^sam:' run.log
   ```
   Require every standard summary field, `training_seconds` near 300, `total_seconds < 600`, finite metrics, and `best_test_acc >= 95.33%` for improvement.
6. Verify process integrity:
   - exactly one evaluation line per epoch;
   - fixed seed 42 and unchanged CutMix exposure near 0.50 before its cutoff;
   - `sam_first_progress >= 0.75`, `sam_first_step` satisfies the one-based even cadence, final SAM ratio near 0.50, and at least 1,800 successful pulses (2,000-2,400 expected);
   - unchanged `num_params=2,748,890` and at least 24,000 total steps (25,000-25,800 expected);
   - no traceback, NaN/Inf, CUDA, timeout, or restoration error.
7. Run `git diff --check`, `uv run ruff check train.py`, and inspect `git status --short`; only `train.py` may be modified. Record all results before removing `run.log`.

### Informational Metrics (Optional)
- `final_test_acc`: final summary in `run.log`.
- `final_test_loss`: final summary in `run.log`.
- `training_seconds`: final summary; expected approximately 300 seconds.
- `total_seconds`: final summary; must remain below 600 seconds.
- `startup_seconds`: final summary.
- `peak_vram_mb`: final summary; expected only modestly above 1,178.9 MiB.
- `num_epochs`: final summary.
- `num_steps`: final summary; expected approximately 25,000-25,800, with 24,000 as the mechanism-integrity floor.
- `num_params`: final summary; expected 2,748,890.
- SAM exposure: final `sam:` audit line with applied, eligible, ratio, first applied step, and first applied progress.
