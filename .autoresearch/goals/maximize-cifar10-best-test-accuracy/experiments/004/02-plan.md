# Plan EXP-004: Plateau-Only Conservative RandAugment
- **Created**: 2026-08-05

## Milestones

### Milestone 1: Phase-aware augmentation implemented
- [x] Create branch `autoresearch/maximize-cifar10-best-test-accuracy-004` from integration commit `5016cc4`.
- [x] Split the accepted crop/flip transform from an augmented transform that inserts `transforms.RandAugment(num_ops=1, magnitude=7)` before `ToTensor`.
- [x] Factor loader construction into a local helper and begin training with the augmented transform.
- [x] Hold an explicit loader iterator. After the batch whose measured duration first takes counted time to 80%, break immediately before another batch; explicitly call the persistent iterator's `_shutdown_workers()`, clear loader iterator references, verify old worker PIDs exit, run `gc.collect()`, construct a fresh crop/flip loader, and print one provenance line. Keep hard labels and every EXP-002 optimizer/evaluator choice fixed.
- [x] Pass constructor/API smoke checks, Python compilation, Ruff, pre-commit, diff, and tracked-scope checks.

### Milestone 2: Loader feasibility established
- [x] Create a temporary `/tmp/exp004_loader_bench.py` diagnostic outside the repository using the exact loader settings and explicit `multiprocessing_context="forkserver"`. Its candidate mode warms/times the strong loader, deterministically shuts it down, verifies its worker PIDs exit, constructs the weak loader in the same process, and measures restart-to-first-batch plus warm weak throughput.
- [x] Run a control process and the strong-to-weak transition process, record all batch rates, real transition latency, and worker-exit result; verify 390 batches/epoch and delete the script after results are persisted in `03-execute.md`.
- [x] Require strong-loader slowest rate at least 100 batches/s, slowest no more than 20% below its median, all old workers exited, measured transition at most 30 seconds, and projected total runtime at most 500 seconds. State explicitly that this gates feasibility only, not expected accuracy.

### Milestone 3: One fixed-budget run executed
- [x] Confirm the moving baseline remains `91.83%`, no stale run log exists, and the only visible idle device is an H20 with approximately 98 GB and no compute process.
- [x] Launch exactly once with `CUDA_VISIBLE_DEVICES=0 timeout 600s uv run train.py > run.log 2>&1`; monitor compact liveness/error signals without streaming the log.
- [x] Confirm exactly one augmentation-switch provenance line, continued training after worker reconstruction, and no orphaned old worker set.

### Milestone 4: Ordered verification completed
- [x] Require `best_test_acc >= 91.93%`; stop formal verification on failure.
- [x] If accuracy passes, require clean exit, each of ten finite summary keys exactly once, `300.0 <= training_seconds < 310.0`, and `total_seconds < 600`.
- [x] If prior conditions pass, require unique evaluation epochs, terminal evaluation equal to `num_epochs`, exactly one phase switch at/just after 80%, and unchanged `num_params=269722`.
- [x] Collect informational metrics only after all necessary conditions pass, including steps, phase-switch time, total time, and candidate/control loader rates.

## Code Changes
- **`train.py`**: Define weak crop/flip and strong crop/flip-plus-RandAugment transforms; factor identical `forkserver` DataLoader creation into a helper; retain an explicit iterator; train with strong augmentation during high-LR exploration; break immediately after the crossing batch; deterministically shut down the old persistent iterator; then rebuild with weak augmentation before low-LR refinement. This implements Claude's underfit/BatchNorm refinement while preserving hard labels, schedule math, architecture, and evaluator. RandAugment necessarily consumes RNG and changes later augmentation draws even at fixed seed; this is part of the method, not a selectable reroll, and a boundary-sized pass must not be overclaimed causally.

## Configuration Changes
- `RandAugment`: absent -> `num_ops=1, magnitude=7`, active only until the first batch at or above `LR_HOLD_FRACTION=0.8`.
- Tail training transform: unchanged accepted random crop, horizontal flip, tensor conversion, and normalization.
- Loader settings remain `BATCH_SIZE=128`, `NUM_WORKERS=8`, pinning, shuffle, `drop_last=True`, and persistent workers; `multiprocessing_context="forkserver"` is made explicit to match the Python 3.14 default and the diagnostic.
- All EXP-002 optimization remains fixed: hard-label cross-entropy, standard momentum 0.9, weight decay `1e-4`, `lr=0.1` through 80%, then step to `0.01` and cosine to `1e-4`, seed 42, model, and evaluation cadence.

## Execution Environment
- Method: local preflight in two fresh CPU processes, followed by one local GPU training run under `timeout 600s`.
- Resources: one idle NVIDIA H20 (97,871 MiB observed), eight persistent loader workers per active loader, existing locked dependencies, cached CIFAR-10 data.
- Estimated runtime: preflight approximately 1-2 minutes; full run approximately 355-420 seconds total including one worker restart, with 300 seconds counted GPU training.
- Log output: preflight rates recorded inline in `03-execute.md`; all training stdout/stderr redirected only to project-root `run.log`.
- Tool skill: local execution; no remote submission.

## Abort Criteria
- Do not launch the full run if the candidate loader/transition fails any feasibility gate, produces other than 390 batches per epoch, leaves old worker PIDs alive, or the temporary diagnostic fails.
- Abort before launch on stale log, non-idle/wrong GPU, more than one exposed GPU, tracked changes outside `train.py`, or an unreviewed diff.
- Kill and classify as failure on 600-second timeout, non-zero exit, traceback, OOM, illegal memory access, data integrity error, no training steps after 120 seconds, missing phase switch by 85% progress, or failure to resume steps within 60 wall seconds after the switch.
- Do not abort on intermediate accuracy, stronger-augmentation train loss, or an expected one-time worker restart pause.

## Verification Protocol

### Verification Procedure

1. Query the baseline with `exp-index.sh baseline`; require `91.83` at `5016cc4` or recompute the acceptance threshold as current baseline plus 0.10 before launch.

2. Run the loader feasibility diagnostic. Both processes use seed 42, cached data, batch 128, shuffle, eight pinned persistent workers, and explicit forkserver context. The control warms and times weak loading. The transition process warms and times three strong epochs, records old worker PIDs, explicitly shuts down and clears the iterator, requires those PIDs to exit, builds the weak loader, and records restart-to-first-batch plus a timed weak epoch. Require exactly 390 batches each full epoch. Compute:

   ```text
   projected_total = 336.0 + max(0, 0.8 * 38629 / slowest_strong_rate + 0.2 * 38629 / weak_rate - 300.0) + measured_transition_seconds
   ```

   Require strong slowest rate `>=100 batches/s`, stability within 20% of median, weak rate positive, every old worker exited, transition `<=30s`, and projection `<=500s`. These measurements validate wall feasibility and lifecycle only, never the accuracy mechanism.

3. Verify source and environment:

   ```bash
   uv run python -m py_compile train.py
   uv run ruff check train.py
   uv run pre-commit run --all-files
   git diff -- train.py
   git status --short
   nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
   nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
   ```

   Require only the reviewed `train.py` diff, the intentional untracked `data/` cache, and one idle H20 exposed to training.

4. Launch once under the 600-second supervisor. Record PID, timestamps, selected GPU, exit code, and compact phase-switch/worker observations in `03-execute.md`. Never rerun a valid completed result or change seed/magnitude adaptively.

5. Parse the summary with `rg` for all ten required keys. First require `best_test_acc >= 91.93%`; on failure record `no-improvement` and skip remaining formal necessary conditions. If it passes, require exit 0, unique finite keys, counted-time band, and wall limit.

6. After prior conditions pass, parse `eval ep` lines and the augmentation-switch provenance line. Require unique eval epochs, terminal evaluation equal to `num_epochs`, exactly one switch on the batch that crosses 80% and before any subsequent low-LR batch, and `num_params=269722`.

7. Verification timeout is 600 seconds execution plus 30 seconds parsing. The single-run threshold is protocol-valid but should not be overinterpreted causally when the delta is exactly ten test examples or attributed solely to RandAugment because the intervention necessarily changes the fixed-seed augmentation stream.

### Informational Metrics (Optional)
- final_test_acc and final_test_loss: final summary values from `run.log`.
- training_seconds, total_seconds, startup_seconds: final summary values.
- peak_vram_mb, num_epochs, num_steps, num_params: final summary values.
- phase switch: provenance line in `run.log`, including epoch and counted progress.
- loader feasibility: three control and three candidate batch rates plus projection, persisted in `03-execute.md` before temporary script deletion.
