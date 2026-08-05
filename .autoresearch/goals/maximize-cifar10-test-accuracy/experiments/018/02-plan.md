# Plan EXP-018: Final-Block-Only Neutral SE
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement one exact-neutral final gate
- [x] Modify only `train.py`: add a ratio-16 `Stage3SE(128)` and apply it only to the residual branch of `layer3[1]` after `conv2` and before shortcut addition.
- [x] Construct and initialize the accepted WRN first; then attach the gate inside a CPU-only restored RNG fork using the project's fixed seed 42, Kaiming first projection, zero biases, and exact-zero second projection. Never vary this seed.
- [x] Keep all accepted model/training/data/evaluation settings unchanged and add no scored diagnostics or summaries.

### Milestone 2: Pass evaluator-free semantic preflight
- [x] Create ignored `experiments/018/preflight.py` with a dummy `prepare.Eval`; verify one operational gate only at `layer3[1]`, 693,858 parameters, exact initial scales/logits, accepted common state, and CPU/CUDA RNG preservation.
- [x] Verify the fixed-seed-42 parameter oracle, correct device/dtype and optimizer grouping, nonzero second-projection gradient on step one, zero first-projection gradient on step one, and nonzero first-projection gradient after one opening update.
- [x] Run compile, diff/scope, protected-file, and root-Python audits; require only production `train.py` changed.

### Milestone 3: Pass matched throughput preflight
- [x] Time accepted and candidate training steps using identical synthetic data, optimizer logic, batch-shared mixup and hard-label paths; warm at least 25 steps and measure three balanced windows of at least 50 steps per regime.
- [x] Require every timing CV <=5%, weighted candidate retention >=97%, finite losses/parameters, and no extra persistent tensor state. Observed retention 98.58%, worst CV 0.078%, and finite synthetic projection 121.34 passes.

### Milestone 4: Execute exactly once and verify
- [x] Confirm one H20, remove stale `run.log`, then execute exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require exit 0, a complete finite summary, 300 counted seconds, total below 600 seconds, one mixup transition near 195 seconds, unique evaluation epochs, 693,858 parameters, and no error signature.
- [x] Compare `best_test_acc` to the fixed 94.17 threshold; observed 93.67, a valid no-improvement that will not be rerun or tuned.

## Code Changes
- **`train.py` / `Stage3SE`**: global-average-pool the signed residual, apply biased `Linear(128,8)`, ReLU, biased `Linear(8,128)`, and `2*sigmoid`; multiply the residual by the per-example channel scale. The module contains only learned parameters, no diagnostic buffers or control-flow effects.
- **`train.py` / `PreActBlock`**: initialize `self.se = None`; after `conv2`, apply `self.se(out)` only when non-`None`, before adding the unchanged shortcut.
- **`train.py` / `WideResNet`**: preserve accepted construction plus whole-model initialization exactly, then inside `torch.random.fork_rng(devices=[])` seed only `torch.random.default_generator` with the existing project seed 42, create/explicitly initialize one gate, and assign it to `layer3[1].se` before the caller's `.to(device)`. This deterministic seed is never tuned or exposed as a new constant.

## Configuration Changes
- Attention: none -> one exact-neutral ratio-16 SE gate on `layer3[1]` only.
- Parameters: 691,674 -> 693,858; added 2,184 (0.316%).
- Gate initialization: `fc1.weight` Kaiming-normal fan-in/ReLU; `fc1.bias`, `fc2.weight`, and `fc2.bias` exact zero; scale range `(0,2)` and exact initial scale 1; existing fixed seed 42, never varied.
- Model topology `[2,2,2]`, all accepted common weights/buffers, global CPU/CUDA RNG, FP32, optimizer, LR/floor, weight decay, batch, crop/flip, batch-shared alpha-0.2 mixup through 65%, clean tail, seed 42, persistent workers, and evaluation cadence remain unchanged.

## Execution Environment
- Method: offline local execution; no remote, network, package install, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20 with local CIFAR-10 and existing `uv`; expected peak VRAM about 1.1 GiB.
- Estimated runtime: preflights under 4 minutes; scored run about 340 seconds wall with a hard 600-second timeout.
- Log output: scored stdout/stderr exclusively in project-root `run.log`, retained until analysis.
- Tool skill: none.

## Abort Criteria
- Abort before scoring for syntax/scope failure; gate count/placement/shape/init/device/dtype mismatch; parameter count other than 693,858; changed accepted common state; CPU or CUDA RNG mismatch; non-identical initial logits; shortcut gating; wrong optimizer group; failed two-step opening; any diagnostic state in production; or non-finite values.
- Abort before scoring if a timing window CV exceeds 5%, weighted throughput retention is below 97%, or the H20 is unavailable. Synthetic projected exposure is informational because the real input pipeline can shift realized passes.
- During scoring abort/classify on nonzero exit, 600-second timeout, OOM, non-finite/error output, missing summary, duplicate evaluation epoch, wrong parameter count, or missing/multiple mixup transition. A complete valid score is never rerun, even if it narrowly misses.

## Verification Protocol

### Verification Procedure

1. Query `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; require baseline 94.07 at `eb08811`, hence threshold 94.17. Timeout 10 seconds.
2. Run `nvidia-smi --query-gpu=name --format=csv,noheader`, `git status --short --untracked-files=all`, `git diff --name-only eb08811 --`, `git diff --check`, root Python-file audit, and `uv run python -m py_compile train.py`; require exactly one H20 and only tracked production `train.py` changed. Timeout 30 seconds.
3. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/018/preflight.py --semantics`; require one `128->8->128` gate only on the second stage-3 block, 693,858 parameters, CPU-only fixed-seed-42 oracle, bitwise equal common state and separately serialized post-construction CPU/CUDA RNG, bitwise equal initial logits, exact unit scale, correct placement/device/dtype/groups, and the preregistered two-step gradient behavior. Timeout 120 seconds.
4. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/018/preflight.py --throughput`; require at least 25 warm steps, three at-least-50-step balanced windows per regime, CVs <=0.05, weighted retention >=0.97, finite projected exposure, and explicit pass. This is an experiment-feasibility guard, not goal verification. Timeout 180 seconds.
5. Remove stale `run.log` and run exactly `timeout 600s uv run train.py > run.log 2>&1`; require exit 0 and no second scored run after any valid completion.
6. Parse `run.log` using `rg`/`awk`. Require one finite summary; `best_test_acc >=94.17`; `training_seconds` at least 300.0; `total_seconds <600`; 693,858 parameters; exactly one transition near 195 seconds; unique evaluation epochs; and no traceback, runtime error, OOM, NaN, or Inf. CIFAR-10 has 10,000 test examples, so 94.17 is exactly 9,417 correct and exactly +0.10 over the 94.07 baseline, not a rounded sub-threshold value. Stop verification at the first necessary-condition failure.
7. Audit `git diff eb08811 -- train.py` and confirm the diff contains only the approved one-gate mechanism and no evaluator/data/schedule/metric changes.

### Informational Metrics (Optional)
- If all necessary conditions pass, collect `peak_vram_mb`, `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `num_epochs`, `num_steps`, and `num_params` from the single final summary in `run.log`.
