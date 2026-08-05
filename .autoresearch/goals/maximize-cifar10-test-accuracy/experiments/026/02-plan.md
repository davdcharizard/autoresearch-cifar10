# Plan EXP-026: Worker-Safe Early-Only RandAugment
- **Created**: 2026-07-26

## Milestones

### Milestone 1: Implement the fixed temporal image policy
- [x] Create `autoresearch/maximize-cifar10-test-accuracy-026` from accepted commit `eb08811`; modify only `train.py` in production.
- [x] Add top-level picklable `EarlyRandAugment` wrapping torchvision 0.24.1 `RandAugment(num_ops=1, magnitude=5, num_magnitude_bins=31, interpolation=BILINEAR, fill=[125,123,114])` after accepted crop/flip and before `ToTensor`; swap a lazily cloned per-worker RNG stream around only this call and restore accepted crop/flip RNG in `finally`.
- [x] Create one unlocked shared byte from an explicit `multiprocessing.get_context()` also passed to the DataLoader, retain one eight-worker persistent loader with explicit `prefetch_factor=2`, and disable the byte exactly once only after exhausting the first full epoch whose counted time reaches 65%.
- [x] Preserve all accepted model, optimizer, schedule, mixup, seed, evaluation, and summary semantics; compile and audit the production diff.

### Milestone 2: Prove semantics, cutoff isolation, and wall feasibility
- [x] Create ignored `experiments/026/preflight.py` with a fail-closed evaluator; verify fixed transform arguments/order, unchanged parent and worker crop/flip CPU RNG plus CUDA RNG and model state/logits/parameters, unchanged optimizer/constants, and the installed 14-operation policy.
- [x] With a marker dataset and the production multiprocessing context/workers/prefetch, exhaust an active epoch, flip the shared byte, then require every item in the next epoch to be inactive; verify the source can flip only at an exhausted epoch boundary and cannot re-enable.
- [x] Benchmark accepted and active-candidate real-data loaders in fresh balanced `A-B-B-A` arms, each with one warmup and three paced complete epochs; require CV <=5%, finite correct batches, clean worker exit, no abnormal transition stall, and both historical-differential and live-absolute projected total wall times <=500 seconds.

### Milestone 3: Run exactly one scored experiment
- [x] Confirm exactly one NVIDIA H20, remove stale `run.log`, and launch exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Monitor output for worker errors/stalls, non-finite loss, CUDA/resource failures, malformed or repeated transitions, and timeout; never adapt the policy or rerun a valid completion.
- [x] Require exit 0, 300 counted seconds, total below 600 seconds, exactly one mixup transition at/after 195 seconds, exactly one later epoch-boundary RandAugment transition with a nonnegative lag below 195 optimizer steps, unique evaluation epochs, and at least 139.0 realized passes.

### Milestone 4: Verify the accuracy hypothesis
- [x] Parse `best_test_acc` and require `>=94.17%` against the indexed 94.07% baseline; observed 94.12%, so the metric condition failed.
- [x] Record best/final accuracy, final loss, epochs, steps/passes, VRAM, wall/counting time, and both transition points; audit the final production diff.
- [x] Accept only on all conditions. The valid lower score closes this exact early `N=1,M=5` epoch-boundary policy; no policy, seed, or rerun change was made.

## Code Changes
- **`train.py` / `EarlyRandAugment`**: add a top-level picklable callable holding the fixed standard torchvision transform, an externally owned shared byte, and a worker-local RNG-state tensor initialized lazily by cloning the worker's current state. On each active call, save the accepted worker RNG, swap in/update the isolated RandAugment stream, and restore the accepted state in `finally`. Return only the image; add no seed, marker, lock, per-sample logging, custom operation, or evaluator interaction. This preserves subsequent accepted crop/flip draws exactly while retaining standard stochastic RandAugment operations.
- **`train.py` / training data construction**: import `multiprocessing`, bind `mp_context = multiprocessing.get_context()`, create `mp_context.Value("b", 1, lock=False)` after fixed torch seeding, insert the wrapper after crop/flip, and pass the same context to DataLoader with explicit `prefetch_factor=2`. The verified host method is `forkserver`; preflight and production must agree. Construction must not consume torch CPU/CUDA RNG or alter model initialization/shuffle state.
- **`train.py` / epoch boundary**: track a main-process Boolean and, after the `for inputs, targets in train_loader` iterator returns normally, flip the shared byte once when `total_training_time >= 0.65 * TIME_BUDGET_S`. Print one line with epoch, step, counted seconds/fraction, and `iterator_exhausted=true`. Do not flip on a budget/step-cap break, during a live iterator, or before mixup's existing per-batch transition.
- **`.autoresearch/.../experiments/026/preflight.py`**: ignored evaluator-free semantic, marker-cutoff, and real-loader timing harness; no test data/evaluator metrics or production changes.

## Configuration Changes
- Training image policy: accepted crop/flip -> crop/flip plus fixed `N=1,M=5` RandAugment during complete early epochs, then accepted crop/flip only.
- RandAugment operation: 14-operation torchvision 0.24.1 space, 31 bins, bilinear interpolation, CIFAR mean-color uint8 fill `[125,123,114]`; no probability wrapper or filtering.
- Temporal cutoff: off after the first exhausted epoch ending at or after 65% counted time. Mixup remains the accepted per-batch 65% transition, so a preregistered hard-label-plus-RandAugment lag of less than one epoch is intentional.
- Loader prefetch: implicit default 2 -> explicit 2; batch, workers, shuffle, pinning, drop-last, and persistence unchanged.
- WRN-16-2, 691,674 parameters, FP32, batch 256, LR/floor/warmup, SGD/Nesterov/decay, alpha-0.2 batch-shared mixup, seed 42, fixed time budget, and evaluation cadence remain unchanged.

## Execution Environment
- Method: offline local preflights followed by one local scored command only on pass; no remote, network, package installation, W&B, GitHub, or `gh`.
- Resources: one NVIDIA H20; local CIFAR-10; installed torch 2.9.1/torchvision 0.24.1; eight persistent workers using the explicitly shared runtime context (`forkserver` on this host).
- Estimated runtime: preflights 1-3 minutes; scored run about 345-430 seconds wall, hard limit 600 seconds. The comparison equalizes 300 counted GPU-training seconds, not total CPU/wall work; wall time remains a hard feasibility constraint.
- Log output: scored stdout/stderr exclusively in project-root `run.log`; retain through analysis and remove before the next experiment.
- Tool skill: none.

## Abort Criteria
- Abort before scoring for wrong transform type/order/arguments/operation space, parent RNG drift, model/logit/parameter/optimizer/config drift, non-picklable shared state, worker failure, any next-epoch active marker after the exhausted-boundary flip, a possible mid-iterator/re-enable path, or evaluator/test access.
- Abort before scoring if any loader arm CV exceeds 5%, tensors/labels are malformed or non-finite, worker cleanup fails, boundary stall is abnormal, or projected total wall exceeds 500 seconds. Do not weaken or retime the policy.
- During scoring abort/classify on nonzero exit, timeout, OOM/resource/worker error, non-finite loss, stale/missing summary, duplicate evaluation epoch, missing/repeated/misordered transition, negative or >=195-step RandAugment lag after the mixup transition, fewer than 139.0 realized passes, or wall time >=600 seconds. Never rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Run `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.3/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-test-accuracy/04-results.tsv`; within 10 seconds require `baseline=94.07`, commit `eb08811`, and threshold 94.17.
2. Run `nvidia-smi --query-gpu=name --format=csv,noheader`, `git diff --check`, `git diff --name-only eb08811 --`, `git status --short --untracked-files=all`, and `uv run python -m py_compile train.py`; within 30 seconds require one H20 and only tracked production change `train.py`.
3. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/026/preflight.py --semantics`; within 180 seconds require exact transform order/arguments and 14-operation space, explicit production/preflight context equality, unchanged parent CPU/CUDA RNG, accepted 691,674-parameter model state/logits and optimizer groups, unchanged constants/loader settings, and marker proof that an exhausted active epoch followed by a parent flip yields a wholly inactive next epoch with no re-enable path. In paired fresh processes, require identical labels and accepted crop/flip RNG/tensors after cutoff, proving the private RandAugment stream does not perturb the clean trajectory.
4. Run `uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/026/preflight.py --loader-timing`; within 300 seconds run fresh balanced `A-B-B-A` real-training-loader arms with one warmup plus three complete paced epochs each. Require each path's measured epoch CV <=0.05, finite `[256,3,32,32]` tensors and valid labels, clean workers, and a normal active-to-inactive boundary. Require both `341.2 + max(0, candidate_epoch_median - accepted_epoch_median) * 143 <= 500` and `41.2 + 143 * candidate_epoch_median <= 500`; the constants come from accepted EXP-002's 341.2-second total and 300 counted seconds.
5. Remove stale `run.log`, record launch time/PID in `03-execute.md`, and run exactly `timeout 600s uv run train.py > run.log 2>&1` once. Require exit 0 and no result-conditioned rerun.
6. Parse summary, evaluations, transitions, and errors with `rg`. Require one finite complete summary; `training_seconds` approximately 300; `total_seconds <600`; 691,674 parameters; `num_steps * 256 / 50000 >=139.0`; one mixup transition at/after 195 seconds; one subsequent RandAugment transition with `iterator_exhausted=true` and step lag in `[0,195)`; unique evaluation epochs; and no traceback/error/non-finite markers. Also verify the transition occurred in a normally exhausted epoch well before the terminal budget-break epoch.
7. Require `best_test_acc >=94.17`. On failure stop verification immediately. On pass audit `git diff eb08811 -- train.py` for only the approved worker-safe temporal policy and unchanged evaluator/training recipe.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `num_epochs`, `num_steps`, `peak_vram_mb`, and `num_params`: collect inline from the final summary after all necessary conditions pass.
- Effective passes and transitions: compute `num_steps * 256 / 50000`; record mixup and RandAugment epoch/step/time lines plus the bounded lag.
- Loader feasibility: record all epoch times, CVs, medians, projected wall time, and boundary timing from the preflight.
