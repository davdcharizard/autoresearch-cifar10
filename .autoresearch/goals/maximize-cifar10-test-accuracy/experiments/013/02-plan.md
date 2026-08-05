# Plan EXP-013: Late Whole-State EMA
- **Created**: 2026-07-24

## Milestones

### Milestone 1: Exact EMA implementation
- [x] Create experiment branch from `eb08811`; modify only source `train.py`.
- [x] Add 65%-start, decay-0.999 detached FP32 whole-state EMA updated after eligible SGD steps; evaluate EMA only after initialization via exception-safe in-place swap/restore.
- [x] Compile/diff audit; pass state coverage, identity, optimizer reference, normal/exception restoration, cadence, finite-state, and update arithmetic tests.

### Milestone 2: Matched feasibility gate
- [x] Run named fail-closed evaluator-free `experiments/013/preflight.py` on one H20 with accepted/candidate production paths.
- [x] Require every CV ratio <=0.05, retention >=0.95, projection >=134.8 passes, identical live updates, and exact 691,674 parameters.

### Milestone 3: Single scored run and audit
- [x] Run exactly once: `rm -f run.log` then `timeout 600s uv run train.py > run.log 2>&1`.
- [x] Require exit 0, 300.0-300.5 counted seconds, total <=600, one mixup/EMA initialization, accepted cadence, finite state, and complete summary.
- [x] Accept only `best_test_acc >=94.17%`; no decay/window/BN policy/recalibration/live-eval fallback.

## Code Changes
- **`train.py`**: add `EMA_START_FRACTION=0.65`, `EMA_DECAY=0.999`; implement a plain external helper (not `nn.Module`, never registered) holding detached clones for every original state key, interpolating floating tensors and copying integral buffers after eligible optimizer steps. Initialize exactly once after the first hard-label update. At scheduled evaluation, clone live state and enter `try/finally` before the first EMA copy; restore in `finally`; never replace objects. Preserve the original model key set. Log init step/pre-step time and update count; scan whole EMA/live state for finiteness at evaluations/final.
- **`experiments/013/preflight.py`**: ignored research artifact containing exact semantic and matched timing checks; never imported by production, source diff remains `train.py` only.

## Configuration Changes
- EMA: none -> start 0.65, decay 0.999, whole inference state; accepted model/training hyperparameters unchanged.
- No architecture, smoothing, initialization, optimizer, LR, mixup, data, seed, cadence, or BN recalibration change.

## Execution Environment
- Local/offline; one H20; existing dependencies/data; no remote/GitHub. Preflight under one minute, scored total about 340 seconds. Output to `run.log`, removed after analysis.

## Abort Criteria
- Do not score on any state/swap/identity/live-update/CV/95%-retention/134.8-pass preflight failure.
- Stop on timeout 124, traceback, OOM, non-finite live/EMA state, wrong update/init count, missing H20, or >=600 wall seconds. Never stop for interim accuracy or retry a valid score.

## Verification Protocol

### Verification Procedure
1. Query index baseline 94.07/`eb08811`; require threshold 94.17, one H20, local data. Compile; require only `train.py` source diff and unchanged `prepare.py`.
2. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/013/preflight.py` with fail-closed `prepare.Eval`. Assert one key per state-dict item, detached FP32 floating shadows, exact integral copies, finite values, and no optimizer state in EMA.
3. Assert update arithmetic on known tensors; no EMA before 65%; first eligible post-SGD step direct-copies live state; later steps use `s=0.999*s+0.001*x`. Assert live model/optimizer state is unchanged by EMA updates.
4. Require EMA helper remains external/unregistered and original `state_dict` keys are unchanged. For normal evaluator failure and an injected failure after a partial EMA-to-live copy, begin `try/finally` before copying and assert all live values restore bitwise, object IDs and optimizer references remain identical, mode/cadence is preserved, and only one evaluator call occurs. No real evaluator/test data.
5. Time accepted vs candidate with fixed pinned inputs and matched models/optimizers. Before every timed window restore that path's private CPU/CUDA RNG snapshots used by global `Beta.sample`/`randperm`, then capture updated snapshots afterward. Warm 25; measure three 50-step windows per path/regime in fixed interleaved order. Require bitwise equality after each matched cumulative window for the complete live model state including BN buffers, optimizer momentum/state, and param-group values. Use 65/35 medians, CV=`pstdev/mean`, retention=`accepted/candidate`, projection=`141.9*retention`; require CV <=0.05, retention >=0.95, projection >=134.8, finite `[256,10]` logits.
6. Run sole scored command. Define `init_step` as the first step whose pre-step `total_training_time>=195.0` and previous pre-step time was `<195.0`; initialization direct-copies state after that step and counts as one EMA update. Require one transition/init, `ema_update_count=final_step-init_step`, EMA-only later evaluations at fifth-plus-terminal cadence, scheduled/final finite scans, counted `[300.0,300.5]`, total <=600, steps<64000, complete summary.
7. Compute passes `steps*256/50000`; record against 134.8/141.9. Accept only >=94.17%. A lower valid score is no-improvement and rejects exact 65%/0.999/whole-state EMA only.

### Informational Metrics (Optional)
- Summary metrics, passes, best epoch/gap, evaluation count, EMA update count, preflight timings/CVs/retention/projection.
