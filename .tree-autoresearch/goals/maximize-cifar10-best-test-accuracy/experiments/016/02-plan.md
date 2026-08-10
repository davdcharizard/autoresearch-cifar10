# Plan EXP-016: 106-State Trailing Uniform Clean-Tail SWA
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement and statically validate the fixed boxcar
- [x] Modify only `train.py` to add the fixed cadence-31, 106-state trailing full-state uniform average and one-source evaluation swap; evaluate live until the window is completely filled.
- [x] Run `PYTHONPYCACHEPREFIX=/tmp/exp016-pycache uv run python -m py_compile train.py` and require exit 0.
- [x] Run CPU arithmetic/restoration smokes covering cumulative fill, first eviction, wraparound, floating buffers, latest-copy integers, non-aliasing, RNG neutrality, and injected evaluation failure.
- [x] Verify `git diff --name-only 1a8d0de` reports only `train.py` and review `git diff --check`.

### Milestone 2: Pass one decisive accuracy-blind GPU-0 preflight
- [x] Verify physical GPU 0 is the 97,871 MiB NVIDIA H20 and that `CUDA_VISIBLE_DEVICES=0` exposes exactly its UUID.
- [x] Materialize the exact EXP-004 parent at commit `1a8d0de` under `/tmp`, guard the evaluator before any trace, and run five alternating-order paired BF16/channels-last workload rounds plus production-order and boxcar-eviction correctness sequences.
- [ ] Accept only if state, RNG, optimizer, SAM ordering, swap/restore, arithmetic, dose, latency, memory, and total-runtime gates all pass on the first complete numeric output.

### Milestone 3: Run exactly one fixed-seed metric experiment
- [ ] Remove stale `run.log`, reconfirm GPU identity, then run `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1` exactly once.
- [ ] Monitor for abort conditions without changing the intervention or launching another metric run.
- [ ] Transcribe the complete final summary, evaluation tail, and audit lines into `03-execute.md` before removing transient logs.

### Milestone 4: Verify and hand off for analysis
- [ ] Parse `best_test_acc`, the source and window size at the best epoch, integrity fields, phase dose, kernel statistics, timing, memory, and final averaged-source evaluations.
- [ ] Require `best_test_acc >= 95.50%` for formal local improvement over parent EXP-004; separately report whether 95.61% and 95.71% global-context bars are reached.
- [ ] Preserve the raw log until Claude adversarial result review is complete, then remove `run.log` and experiment-owned `/tmp` artifacts.

## Code Changes
- **`train.py` only**: add constants `SWA_START=0.75`, `SWA_UPDATE_EVERY=31`, and `SWA_WINDOW=106`; all inherited architecture, data, loss, schedule, optimizer, CutMix, drop-path, and SAM settings remain byte-for-byte behaviorally unchanged.
- **`train.py` / state manager**: add a `TrailingUniformSWA` class built from stable named parameters and persistent named buffers. Partition floating and non-floating buffers, require exact `state_dict` coverage, and reject alias, shape, dtype, device, gradient, optimizer, or SAM ownership errors.
- **`train.py` / ring representation**: before `t_start_training`, for every parameter and persistent floating buffer allocate one tensor with a leading dimension of 106 plus a same-shaped averaged-state tensor. Ring storage uses the live tensor dtype; production tensors are FP32 in this model. Precompute or construct slot views without allocating new tensor storage. Allocate latest-copy averaged integer buffers and complete live-state restore shadows. No second model or running-sum state is constructed. Startup allocation cost and full-run peak memory are reported but do not consume the charged budget.
- **`train.py` / sampling**: after the sole `optimizer.step()` and exact SAM restoration, but before the existing CUDA synchronization and charged-time increment, sample iff step-entry `progress >= 0.75` and one-based `next_step % 31 == 0`. Copy live floating state into the current ring slot and record its charged timestamp. Before sample 106, do not materialize or evaluate a cumulative average. At sample 106 and every later sample, directly reduce exactly the 106 active ring slots into the averaged floating state and copy current integer buffers. Direct reduction eliminates subtract/add cancellation drift. All copies and reductions are charged and consume no RNG.
- **`train.py` / evaluation**: evaluate live state until the 106-slot ring is full. Thereafter evaluate only the true 106-state trailing uniform parameters and persistent averaged/latest buffers. Snapshot live state and module modes, swap without replacing `Parameter` objects, call the frozen evaluator once, and restore in `finally`. Require exact restoration, unchanged optimizer identities/state ownership, unchanged RNG around state management, and one attempted source per epoch, including injected evaluator failure. Record source, update count, and window size at every epoch and at the best epoch; a best from a pre-fill live evaluation is not evidence for the boxcar mechanism.
- **`train.py` / audits**: record update count, first/last step/progress/charged time, interval min/mean/max, ordinary/SAM counts, window fill/evictions/wraps, final window size/span/mean state age, uniform weight/ESS, consecutive-sample parameter distances, final averaged/live parameter and BN distances and variance ratios, positive running variances as an integrity invariant, newest integer equality, inventory and allocation sizes, live/SWA evaluations, source/window at best, swaps/restores, and zero coverage/alias/RNG/nonfinite/restore failures. Print all available full-window evaluation accuracies (expected about 11), their mean/min/max/final/best premium as report-only context. BN distances/ratios are diagnostic rather than arbitrary acceptance gates.

## Configuration Changes
- `SWA_START`: absent -> `0.75` (matches the validated clean-tail boundary and excludes CutMix states).
- `SWA_UPDATE_EVERY`: absent -> `31` (inherits EXP-011's negligible-cost cadence and alternates period-two ordinary/SAM states).
- `SWA_WINDOW`: absent -> `106` (EXP-011's observed cadence interval was `74.7736 / 159 = 0.470274s`; 106 states predict `(106-1)/2 * 0.470274 = 24.6894s` mean age and ESS 106, close to the validated implemented EMA's 25.13-second center while removing its 6.30% first-state anchor).
- Evaluation source: live-only -> live until the 106th sample, then trailing-uniform full state. This prevents the rejected growing cumulative estimator from entering the max metric and does not add an evaluation or modify `prepare.py`/`Eval`.
- No BN recalibration, time weighting, EMA, live/SWA interpolation, accuracy-conditioned selection, LR change, seed change, or post-result window adjustment is permitted.

## Execution Environment
- Method: local process from the repository root. Accuracy-blind checks use an experiment-owned harness under `/tmp`; the sole metric command is `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- Resources: physical GPU 0 only, NVIDIA H20, 97,871 MiB. Expected ring storage is approximately 106 complete floating model states, about 1.1 GiB, with total peak expected near 2.5 GiB; memory is informational under the goal's soft VRAM constraint and only OOM or insufficient physical headroom aborts. CPU data loading retains the parent configuration.
- Estimated runtime: preflight below 4 minutes; metric run about 7.5 minutes total with exactly 300 charged training seconds, always below the 10-minute outer timeout.
- Log output: preflight JSON/text under `/tmp/exp016-*`; raw metric output in repository-local `run.log` until durable transcription and Claude result review, then delete it per protocol. Set `PYTHONPYCACHEPREFIX=/tmp/exp016-pycache` for transient scripts to avoid shared `/tmp/__pycache__` permissions.
- Tool skill: no remote submission skill; execution is local.

## Abort Criteria
- Stop before metric launch if physical index 0 is not the approximately 97,871 MiB NVIDIA H20, if more than one GPU is visible, or if the visible UUID differs from physical GPU 0.
- A completed numeric preflight is decisive. Abort this leaf if parent drift exceeds 3%, paired-ratio MAD/median exceeds 0.5%, median candidate/parent charged latency exceeds 1.005, any round ratio exceeds 1.02, projected optimizer steps fall below 25,400, or projected total runtime reaches 600 seconds. Candidate peak allocation is reported and checked for ample H20 headroom, but is not a numeric pass/fail gate.
- Abort before metric launch on any arithmetic-reference mismatch, online parent/candidate mismatch before state-only effects, incomplete state coverage, alias, nonfinite state, nonpositive averaged BN variance, RNG movement, optimizer/SAM ownership mutation, cadence/parity error, or failed success/exception restoration check.
- During the metric run, terminate on traceback, CUDA/OOM/device error, nonfinite training/state diagnostics, audit failure, missing progress for 90 seconds after startup, or wall time of 600 seconds. Intermediate accuracy is never an abort or retuning signal.
- A realized full run with fewer than 25,400 steps, fewer than 155 samples, final window size not 106, last sample below 99.5% progress, or sample parity imbalance greater than one is a trustworthy dose shortfall and therefore `no-improvement`, not permission to rerun.

## Verification Protocol

### Verification Procedure

1. **Scope and syntax** (timeout 30 seconds):
   ```bash
   git diff --name-only 1a8d0de
   git status --porcelain --untracked-files=all
   git diff --check
   env PYTHONPYCACHEPREFIX=/tmp/exp016-pycache uv run python -m py_compile train.py
   ```
   Run this before creating `run.log`. Require the diff to print only `train.py`, status to contain only the tracked `train.py` modification (the ignored `.tree-autoresearch/` metadata is expected), and both validation commands to exit 0. Experiment helpers remain under `/tmp`, never in the repository.

2. **GPU identity** (timeout 10 seconds):
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader
   env CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(torch.cuda.device_count(), p.name, p.total_memory, torch.cuda.get_device_uuid(0) if hasattr(torch.cuda, "get_device_uuid") else "uuid-via-nvidia-smi")'
   ```
   Require physical index 0 to be `NVIDIA H20` with approximately 97,871 MiB and exactly one visible CUDA device. The harness additionally compares the UUID with `GPU-b1bc897d-2183-dad2-8302-8800bc02a633` from the pre-plan check.

3. **Deterministic arithmetic and restoration smoke** (timeout 60 seconds):
   ```bash
   env PYTHONPYCACHEPREFIX=/tmp/exp016-pycache uv run python /tmp/exp016_swa_smoke.py
   ```
   Require a machine-readable `PASS` covering one sample without averaged evaluation readiness, 106-sample fill, samples 107 and 212 wraparound, irregular values/timestamps, direct 106-slot reduction agreement with an FP64 reference within dtype-aware tolerance, latest integer buffers, full-state coverage, non-aliasing, RNG neutrality, optimizer identity, live-before-fill routing, and exact successful and injected-failure evaluation restoration.

4. **One decisive accuracy-blind paired preflight** (timeout 240 seconds):
   ```bash
   timeout 240s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp016-pycache PYTHONUNBUFFERED=1 uv run python /tmp/exp016_preflight.py
   ```
   The harness checks out only the parent `train.py` content from `1a8d0de` into an experiment-owned `/tmp` module, monkeypatches both evaluators to raise before data access, and uses real CIFAR training batches with BF16/channels-last. Each of five alternating-order timed rounds contains exactly 248 production-ordered steps: 186 early steps and 62 clean-tail steps with 31 ordinary/31 SAM paths and exactly two cadence updates, one of each parity. This is one SWA update per 124 overall steps, matching the production phase/frequency. A separate state-only sequence reaches the first eviction without thousands of training forwards. The first complete numeric output must meet every abort gate above, including parent drift/dispersion, median and maximum ratios, projected dose/runtime, reported peak allocation, exact pre-state online equality, ring arithmetic, cadence/SAM balance, BN integrity, and restoration. Any numeric failure ends the experiment without a metric launch. An exception, malformed output, or 240-second timeout before numeric gates is a harness failure and permits one recorded repair/retry (timeout-only retry may raise the harness limit to 360 seconds without changing the workload); there is no open-ended retry loop.

5. **Exactly one metric launch** (timeout 600 seconds):
   ```bash
   rm -f run.log
   timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
   ```
   Require exit 0, no error markers, `training_seconds` in `[299.5, 301.0]`, `total_seconds < 600`, exactly one evaluation and one evaluator call per epoch, `num_steps >= 25400`, `num_params = 2,748,890`, at least 155 samples, final window size 106, no SWA-source evaluation before sample 106, last sample progress at least 0.995, ordinary/SAM sample imbalance at most one, nonzero trajectory distance, positive averaged BN variances, exact latest integer state, and zero audit failures. Explicitly verify `train.py` contains no `torch.compile` or CUDA-graph capture; state copies/reductions and evaluation swaps therefore remain outside any compiled/captured region.

6. **Metric decision** (timeout 10 seconds):
   ```bash
   grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:' run.log
   ```
   First query the parent with `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 004` and require metric 95.40. Valid `best_test_acc >= 95.50%` is `improvement`; a valid result below 95.50 is `no-improvement`. Record whether the best epoch used live or the full 106-state boxcar; a live-source formal pass retains the tree verdict but does not support the SWA hypothesis. Report separately whether the result reaches EXP-011's 95.61 global-best level or clears it by the goal resolution at 95.71. All available full-window SWA evaluations, their mean/range/final accuracy, and best-minus-mean premium are scientific context, not extra acceptance gates. Do not retry, alter the window, blend sources, or change the seed after observing accuracy.

### Informational Metrics (Optional)
- `final_test_acc`: `grep '^final_test_acc:' run.log` — terminal averaged-source accuracy.
- `final_test_loss`: `grep '^final_test_loss:' run.log` — frozen evaluator cross-entropy at the terminal averaged source.
- `training_seconds`, `total_seconds`, `startup_seconds`: corresponding final summary lines in `run.log`.
- `peak_vram_mb`: `grep '^peak_vram_mb:' run.log`; compare with the preflight allocation but retain the known candidate-only/full-run calibration caveat.
- `num_epochs`, `num_steps`, `num_params`: corresponding summary lines plus the SWA dose and source audit lines.
- Stable-tail context: final audit line with every available full-window SWA evaluation (expected about 11), mean/min/max/final, and full-window best minus mean.
