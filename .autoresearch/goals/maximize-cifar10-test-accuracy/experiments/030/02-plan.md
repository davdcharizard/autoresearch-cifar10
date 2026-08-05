# Plan EXP-030: Early Drop-Path on the Added Stage-3 Block
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement exact targeted private-RNG drop-path
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-030` from accepted commit `67c8e98`; modify only `train.py` in production and keep `prepare.py` frozen.
- [x] Add a strict inverted per-example whole-residual helper and configure only `model.layer3[2]` with `p=0.05` and a non-registered private device generator seeded once at 28028 after accepted model construction/device transfer.
- [x] Route the production cutoff through a small `maybe_disable_drop_path` controller invoked immediately after computing `use_mixup` and before the first hard-label forward; disable exactly once when `use_mixup=False`, and require eval mode and p=0 train mode to skip mask allocation/RNG/scaling and reproduce accepted computation.
- [x] Preserve accepted topology/state/init, global RNG, early RandAugment/cutoff, alpha-0.2 mixup, batch-256 FP32 SGD/schedule/decay, seed 42, loader, budget, and evaluator; compile and audit the diff.

### Milestone 2: Prove identity, isolation, and exposure
- [x] Create ignored `experiments/030/preflight.py` with a fail-closed evaluator and independent `git show 67c8e98:train.py` oracle; prove exact state/init/global RNG/optimizer identity, one-block placement, constants, and 987,098 parameters.
- [x] Prove p=0 train and active-p eval logits/loss/gradients/update are bitwise accepted and consume neither global nor private RNG; prove active p=0.05 uses `[B,1,1,1]` masks with empirical rate 4-6%, values `{0,1/0.95}`, deterministic private replay, finite gradients, and unchanged global RNG.
- [x] Exercise the same production `maybe_disable_drop_path` controller to prove the first hard-label step disables before forward, the private state then remains fixed, only one transition/no re-enable exists, and accepted RandAugment worker/cutoff/clean-tail semantics remain unchanged.
- [x] Run balanced fresh-snapshot H20 timing across accepted/candidate mixup and hard paths, including pinned-host copies and full scored bodies; emit metrics before assertions and require every CV <=5%, exact fixed-time retention >=0.9774, and >=130 projected passes from 133.00736.

### Milestone 3: Run the sole fixed-seed score
- [x] Confirm baseline 94.32 at `67c8e98`, one idle H20, local data, frozen evaluator, clean scope, and passing preflights; remove stale `run.log` and launch exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Monitor numerical/CUDA/worker health and mixup/drop-path/RandAugment transitions without reacting to interim accuracy; never rerun a valid completion or change mask probability/seed/block/cutoff.
- [x] Require exit 0, 300 counted seconds, total <600, 987,098 parameters, one private-mask transition, and unique accepted-cadence evaluations. Record realized exposure separately; an otherwise valid `<130`-pass score is operationally inconclusive for the intended mechanism but still counts and may never be rerun.

### Milestone 4: Verify targeted robustness
- [x] Classify objective improvement solely by `best_test_acc >=94.42%`; separately classify predetermined endpoint corroboration by `final_test_acc >=94.32%` without allowing it to override the primary metric.
- [x] Record final loss versus accepted 0.2523, best-final gap, steps/epochs/passes, transition times/lags, VRAM, counted/wall time, and final diff.
- [x] Accept into the objective frontier when the primary metric and hard task constraints pass. Report exposure and corroboration as separate mechanism verdicts. A valid normal-exposure primary-metric miss closes targeted early masking of the added block; never tune adjacent probability, generator seed, block placement, mask granularity, cutoff, or rescaling.

## Code Changes
- **`train.py` / constants**: add `DROP_PATH_P=0.05` and `DROP_PATH_SEED=28028` as fixed experiment settings. Do not add a second cutoff constant: the actual controller consumes the existing `use_mixup` predicate.
- **`train.py` / helper and block**: add `apply_drop_path(residual, probability, generator, training)` with an immediate identity return when not training or probability is zero. Active mode samples one private-generator uniform per example into `[B,1,1,1]`, thresholds at 0.05, casts to residual dtype, and multiplies by `mask/0.95`. `PreActBlock` holds non-parameter attributes defaulting to p=0 / generator=None and applies the helper only to its residual after `conv2`, before shortcut addition.
- **`train.py` / model setup and transition**: add `maybe_disable_drop_path(block, enabled, use_mixup, epoch, step, training_time, progress, lr)`, returning the new boolean state and mutating/logging only on the single `enabled and not use_mixup` transition. After direct accepted model construction and `.to(device)`, assign p=0.05 and a newly created device generator seeded 28028 to `layer3[2]` only. Invoke the controller before `optimizer.zero_grad`/forward at the existing mixup boundary and emit one `Drop-path disabled` line at the same epoch/step/time. Generator creation/seeding must not touch global CPU/CUDA RNG.
- **`.autoresearch/.../experiments/030/preflight.py`**: ignored verification-only harness for accepted identity, helper/mask/private-RNG semantics, transition source/runtime behavior, unchanged worker augmentation, and balanced complete-step timing. It stubs `prepare.Eval` and never constructs evaluator/test data.

## Configuration Changes
- Early training computation: accepted third-block residual -> per-example accepted residual times Bernoulli(0.95)/0.95; every other block unchanged.
- Hard-tail/eval computation: exact accepted residual addition with no mask draw or scale.
- New private seed 28028 controls only drop-path masks and is never rerolled; global seed 42 and all accepted global/worker streams remain unchanged.
- Model parameters/state dict/MACs: unchanged 987,098 and accepted branch compute; only tiny mask generation/multiply overhead while active.

## Execution Environment
- Method: offline local semantic/timing preflights, then one local score only on pass; no remote, network, installs, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20, local CIFAR-10, installed environment, eight persistent forkserver workers.
- Estimated runtime: preflights under 3 minutes; scored run about 345-380 seconds wall, hard timeout 600.
- Log output: scored stdout/stderr only in root `run.log`, retained through analysis then removed.
- Tool skill: none.

## Abort Criteria
- Abort before timing on any accepted state/init/global-RNG/optimizer mismatch; wrong block/parameter count; p=0 or eval non-identity/RNG draw; active mask shape/value/rate/private replay/global leakage error; non-finite gradient; wrong transition ordering; hard-tail private RNG advance; worker cutoff/tail leak; evaluator/test access; scope/syntax failure.
- Abort before scoring on any non-finite/error/OOM timing arm, any CV >5%, exact fixed-time retention <0.9774, or projected passes <130. Print all measurements before gate assertions and never repeat a stable miss.
- During score stop/classify on timeout, nonzero exit, OOM/worker/resource error, non-finite loss, no output for 60 seconds, malformed/missing summary, wrong topology/counts, invalid/repeated transition, duplicate eval epoch, or total >=600. Never rerun a valid score; if realized passes are `<130`, retain the score and label only the mechanism exposure operationally inconclusive.

## Verification Protocol

### Verification Procedure

1. Query `exp-index.sh baseline`; within 10 seconds require 94.32 at `67c8e98`, so the primary threshold is 94.42.
2. Within 30 seconds query `nvidia-smi`, local CIFAR presence, branch/status, `git diff --check`, full diff, `git diff --exit-code 67c8e98 -- prepare.py`, and `uv run python -m py_compile train.py .../experiments/030/preflight.py`. Require one idle H20, only tracked `train.py`, and no frozen-file drift.
3. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/030/preflight.py --semantics`. Require independent accepted/candidate model state, logits, post-construction global RNG, optimizer groups, and 987,098 parameters; exact constants/one-block placement; private generator creation without global RNG drift; and no registered generator state.
4. In the same command, require direct helper mask shape/rate/value/replay checks; p=0 train and active-p eval accepted identity including gradients/update and both RNG states; active finite gradients/private-state advance with fixed global RNG; direct calls to the production `maybe_disable_drop_path` controller proving no early mutation, exactly one boundary mutation/log before a traced hard forward, no re-enable, and hard-tail private-state stability; unchanged EXP-027 worker transform order/private RNG/exhausted cutoff/clean replay; and no evaluator/test construction.
5. Run `timeout 240s uv run python .../experiments/030/preflight.py --throughput`. For mixup and hard separately, run three fresh deterministic accepted/candidate fixtures in balanced order, each with >=20 warmups and >=50 measured complete pinned-H2D-through-synchronize FP32 SGD steps. Emit all windows/medians/CVs first; compute exact fixed-time retention as `(0.65/candidate_early_ms + 0.35/candidate_hard_ms) / (0.65/accepted_early_ms + 0.35/accepted_hard_ms)` and `projected_passes=133.00736*retention`, then require CV <=.05, retention >=.9774, and passes >=130.
6. Reconfirm audit and one idle H20, remove stale log, execute exactly `timeout 600s uv run train.py > run.log 2>&1` once, record PID/start, and never launch a second valid score.
7. Parse with `rg`; require one finite summary, 300.0-300.1 counted seconds, total <600, 987,098 parameters, and no traceback/OOM/non-finite/worker errors. Record `num_steps*256/50000`; `<130` does not invalidate or authorize rerunning the sole completed score, but makes the intended-exposure mechanism verdict operationally inconclusive.
8. Require mixup and drop-path disable exactly once at the same first >=195-second epoch/step before forward; require one later RandAugment disable after iterator exhaustion with step lag `[0,195)` and no re-enable. Require unique every-fifth-epoch evaluations plus one final partial epoch.
9. Classify goal success only by `best_test_acc >=94.42%`. Independently report whether `final_test_acc >=94.32%` corroborates the mechanism; a corroboration miss cannot overturn primary-metric success. Audit final source and record final loss relative to 0.2523 regardless of verdict.

### Informational Metrics (Optional)
- Final summary: `run.log` - best/final accuracy, final loss, counted/total/startup seconds, epochs, steps, passes, VRAM, and parameter count.
- Transitions: `run.log` - mixup/drop-path epoch/step/time, RandAugment epoch/step/time, and lag.
- Preflight: direct output - mask rate/values, per-regime timing windows/CVs, retention, and projected passes.
- Mechanism: final loss delta from 0.2523 and endpoint delta from 94.22; lower values support robustness but cannot override top-1 gates.
