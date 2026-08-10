# Plan EXP-011: Cadence-31 charged-time EMA
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the isolated EMA state machine
- [x] Add fixed `EMA_START=0.75`, `EMA_UPDATE_EVERY=31`, and `EMA_TAIL_HALF_LIVES=4.0`; compute `EMA_HALF_LIFE_S=(1-EMA_START)*TIME_BUDGET_S/EMA_TAIL_HALF_LIVES` and assert/log the expected 18.75 seconds under the frozen 300-second budget.
- [x] Build name-aligned, non-gradient shadow tensors for every parameter, floating persistent buffer, and non-floating persistent buffer; also preallocate exact online-restore and previous-sample states. Assert their key set against a freshly materialized `model.state_dict()` key set, then assert shape/dtype/device coverage and optimizer/SAM exclusion.
- [x] After the sole optimizer step and any SAM restoration, sample only when pre-batch progress is >=0.75 and one-based step is divisible by 31. Pin `sample_time` to the exact `total_training_time` value captured at step entry and used to compute that step's progress. First-copy online state; thereafter use `2**(-(sample_time-last_sample_time)/EMA_HALF_LIFE_S)`, EMA-lerp floating tensors, and copy integer buffers.
- [x] Keep EMA update and consecutive-sample L2 work before the existing CUDA synchronization and `dt` calculation so every training-side operation is charged. Accumulate no-grad GPU scalar distances without `.item()` or an extra synchronization, read them only after charged training, and never sample perturbed SAM weights or consume RNG.

### Milestone 2: Implement one-source EMA evaluation and audits
- [x] Before EMA activation, retain the parent live evaluation. After activation, snapshot the full live state, copy the full EMA state into the same model, call the frozen evaluator exactly once, and restore live state plus every module training flag in `finally`.
- [x] Prove bitwise live restoration by comparing a fresh post-restore `model.state_dict()` enumeration against the restore snapshot, independently of the lists used to copy EMA state. Also require unchanged optimizer parameter/momentum identities, no RNG movement from EMA operations, and exactly one evaluator call per epoch. Never evaluate live and EMA together or recalibrate BatchNorm.
- [x] Record update count; first/last step/progress/time; ordinary/SAM sample parity; decay min/mean/max; total span and oldest-state coefficient; live/EMA evaluation counts; consecutive-sample and EMA-to-live parameter distances; BN mean/variance distances and per-BN variance ratios; coverage and restoration failures.
- [x] Preserve the complete EXP-004 online CutMix/SAM path and counters. Static checks must show only `train.py` differs from commit `1a8d0de`.

### Milestone 3: Pass correctness and physical-GPU-0 gates
- [x] Verify closed-form time EMA under cadence partitions, first-copy behavior, integer-copy behavior, no shadow gradients/optimizer ownership, and fixed-seed RNG parity on scalar/tiny-module tests.
- [x] On the full WRN, test both ordinary- and SAM-parity EMA samples, post-optimizer ordering, SAM perturbation norm/replay/BN semantics/restoration, finite nonzero consecutive distance, and exception-safe evaluation restore.
- [x] Confirm physical GPU 0 is the approximately 97,871 MiB NVIDIA H20 and run a same-harness parent/candidate BF16/channels-last latency benchmark with five alternating rounds. Across the complete benchmark, include at least 200 ordinary and 100 production-faithful SAM steps per arm plus enough cadence sequences for at least 30 candidate EMA updates; measure swap/evaluate/restore separately.
- [x] Proceed only if full-run-weighted candidate training latency is <=1.02x parent, projected steps are >=25,200, evaluation-with-swap projects total runtime below 600 seconds, peak allocation is <1.30 GiB, and all state/RNG/BN/restore assertions pass. The first valid measurement is decisive.

### Milestone 4: Execute one fixed-seed metric run
- [x] Remove stale `run.log` and launch once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- [x] Monitor process/GPU activity and error signatures without pruning on intermediate accuracy. Abort on exception, OOM/CUDA error, nonfinite state, restoration failure, 120 seconds without process/GPU progress, or the 600-second timeout.
- [x] Durably copy the complete metric summary, CutMix/SAM/EMA audits, evaluation-source counts, and trajectory/BN diagnostics into `03-execute.md` before deleting transient logs.

### Milestone 5: Verify the result
- [x] Require exit 0, 299.5-301.0 charged seconds, total runtime <600 seconds, one evaluation per epoch, 2,748,890 trainable parameters, complete summary, and only `train.py` changed.
- [x] Apply fixed mechanism classifications: <25,200 steps, <145 EMA samples, or identically zero finite distances is a valid dose/mechanism failure (`no-improvement` even if accuracy clears); parity difference >1, restore/coverage/nonfinite failure, extra evaluator call, or CutMix/SAM semantic mismatch makes the result untrustworthy (`invalid`, metric `NaN`).
- [x] Formal improvement is `best_test_acc >=95.50%` versus EXP-004 at 95.40%; record separately whether the stronger >=95.70% target is reached. Below 95.50 is `no-improvement`, not a retry trigger.

## Code Changes

- **`train.py`**: Add a sparse charged-time EMA over model parameters and persistent buffers, previous-sample and evaluation-restore storage, post-optimizer cadence logic, exception-safe one-source evaluation routing, and compact durable audit output. The online architecture, data path, optimizer, schedules, CutMix, and SAM computations remain unchanged.

No other tracked file may change. Temporary accuracy-blind smoke/benchmark harnesses may exist outside the repository while executing and must be removed after their results are durably recorded.

## Configuration Changes

- `EMA_START`: new `0.75`, aligned with the CutMix-to-clean/SAM transition.
- `EMA_UPDATE_EVERY`: new `31`; odd cadence alternates ordinary/SAM parity instead of selecting the even SAM subsequence.
- `EMA_TAIL_HALF_LIVES`: new `4.0`; the 75-second tail contains four half-lives.
- `EMA_HALF_LIFE_S`: computed as `(1-0.75)*TIME_BUDGET_S/4`, expected `18.75` seconds; step-entry charged-time deltas determine each decay.
- EMA state: all parameters plus persistent floating buffers use the same decay; persistent integer buffers copy latest. No optimizer state, gradients, RNG, SAM snapshot, counters, or nonpersistent state is averaged.
- Evaluation source: live before first EMA sample, EMA only afterward; exactly one frozen evaluator call per epoch.
- All EXP-004 model, batch, seed, optimizer, LR, drop-path, CutMix, SAM, timing, and summary settings remain fixed.

## Execution Environment

- Method: local implementation/smokes, paired local latency preflight, then one local full run.
- Resources: physical GPU 0 only through `CUDA_VISIBLE_DEVICES=0`; NVIDIA H20 with approximately 98 GB; existing `uv` environment; no new dependency.
- Estimated runtime: under 3 minutes for smokes/preflight plus approximately 460 seconds for the full run; hard outer limit 600 seconds.
- Log output: full stdout/stderr to repository-root `run.log`; exact preflight and terminal evidence copied to `experiments/011/03-execute.md`; transient files removed before analysis.
- Tool skill: local execution; no remote submission skill.

## Abort Criteria

- Wrong physical/visible GPU, a tracked change outside `train.py`, or any change to seed, evaluator, data stream, online optimizer/CutMix/SAM semantics, budget, or validation cadence.
- State inventory mismatch against fresh `model.state_dict()` keys, shadow tensor entering autograd/optimizer/SAM lists, EMA sampling before optimizer/SAM restoration, unexpected RNG movement, nonfinite distance/state, or EMA update parity not alternating as defined.
- Evaluation calls both live and EMA in one epoch, performs BN recalibration/data replay, fails to restore live state/mode exactly, or changes optimizer identities/state.
- First valid paired preflight exceeds 1.02x weighted latency, projects <25,200 steps or >=600 seconds total, exceeds 1.30 GiB peak allocation, or fails any correctness assertion. Before viewing candidate gates, compute parent drift as `max(parent_round_medians)/min(parent_round_medians)-1`; >0.075 discards the full first measurement and permits exactly one complete rerun after recording all parent rounds. The second measurement is decisive even if still noisy.
- Full run exception, CUDA/OOM/nonfinite error, restore/coverage failure, 120 seconds without process/GPU progress, or 600-second timeout. Intermediate accuracy/loss cannot trigger abort unless nonfinite.

## Verification Protocol

### Verification Procedure

1. Confirm parent and thresholds:
   `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 004`.
   Require `metric=95.40`; formal threshold is 95.50% and meaningful target is 95.70%.
2. Before every GPU command, run `nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader` and `CUDA_VISIBLE_DEVICES=0 uv run python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)"`. Require one visible NVIDIA H20 with approximately 98 GB.
3. Run `python -m py_compile train.py`, `git diff --check`, and `git diff --name-only 1a8d0de`; require exit 0 and only `train.py`. Execute the exact arithmetic, inventory, parity, SAM-integration, swap/restore, RNG, and BN smokes in Milestones 1-2.
4. From repository-root cwd, materialize the parent with `git show 1a8d0de:train.py > /tmp/exp011_parent_train.py`, add the repository root to `sys.path`, and load it under a unique non-`__main__` importlib name; import the candidate normally. Assert both modules resolve identical `prepare` budget/data/worker constants and that neither import invokes `main()`. Reseed immediately before each model construction, use fixed synthetic batch-256 inputs, and query no accuracy. Run under `timeout 300s env CUDA_VISIBLE_DEVICES=0 uv run python -`; exit 124 is an infeasible preflight and stops the experiment without a metric run. Apply the Milestone-3 gates and parent-only drift rule, then remove the snapshot/harness.
5. After a passing preflight, run exactly once: `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`. Treat exit 124 as timeout and any other nonzero exit as crash.
6. Extract `grep "^best_test_acc:\|^peak_vram_mb:" run.log`, the complete tail summary, and EMA/CutMix/SAM audits. Formal necessary conditions require `best_test_acc >=95.50%`, training seconds in `[299.5,301.0]`, total seconds <600, complete summary, and clean completion. Apply the preregistered classifications: dose/mechanism shortfall -> `no-improvement`; state/evaluation/online-parent integrity failure -> `invalid` with `NaN`; otherwise the formal metric decides improvement versus no-improvement.
7. Require evaluation-line count = `num_epochs`; scan `rg -n -i 'traceback|cuda error|out of memory|(^|[^a-z])(nan|inf)([^a-z]|$)' run.log`; verify only `train.py` differs. Record exact values and source lines in `03-execute.md`, then delete `run.log` before advancing.

### Informational Metrics (Optional)

- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params`: terminal summary in `run.log`.
- EMA mechanism: update count/window/decays/parity, oldest-state coefficient, live/EMA evaluation counts, consecutive-sample and EMA/live distances, BN mean/variance distances and variance ratios, state coverage, restoration checks.
- Parent mechanism preservation: CutMix and SAM applied/eligible ratios, first SAM step/progress, exact evaluation count, and projected/realized exposure.
