# Plan EXP-008: Implementation-Audited Mild RandAugment in the Regularized Phase
- **Created**: 2026-08-05

## Decision Lock

- Parent EXP-004 is `1a8d0de` at 95.40%; formal acceptance is 95.50% and the separate preregistered mechanism target is 95.70%.
- Fixed intervention: one torchvision RandAugment operation at magnitude 5 of 30 (`num_magnitude_bins=31`), nearest interpolation, fill zero, applied to model inputs only while charged progress is `<0.75`.
- Planning audited torchvision 0.24.1's exact 32x32 operation table before any accuracy run. Paper magnitude 2 is not semantically portable: translations truncate to zero and posterization stays at eight bits. Magnitude 5 is the lowest mild bin with nonzero two-pixel translations and seven-bit posterization. No further scalar, phase, seed, or operation-set change is permitted.
- Parent crop/flip, independent identities, CutMix, drop path, clean-tail Euclidean SAM, optimizer, model, evaluator, and validation cadence remain unchanged.

## Milestones

### Milestone 1: Implement paired-view RandAugment with isolated worker RNG
- [x] Add explicit constants `RANDAUGMENT_NUM_OPS=1`, `RANDAUGMENT_MAGNITUDE=5`, `RANDAUGMENT_NUM_BINS=31`, `RANDAUGMENT_END=0.75`, and `RANDAUGMENT_SEED=42`.
- [x] Add a dataset transform that performs the exact parent crop/flip once, then returns a parent-identical clean normalized FP32 tensor and an augmented uint8 tensor from the same PIL image. RandAugment operates on an image copy; augmented normalization occurs on GPU only when selected.
- [x] Lazily create a private generator keyed by the current `worker_info.seed`, worker id, and fixed namespace. Reinitialize on any seed-key change, including worker recreation. Around only `RandAugment.__call__`, swap worker-global torch RNG to/from private state in `try/finally`; capture advanced private state even on exception and restore the global torch state exactly. Preflight must prove Python/NumPy states remain untouched by the unmodified policy.
- [x] Compute `progress` exactly once after batch yield and use that same scalar for view selection, CutMix, and SAM. Select augmented only below 0.75, normalize it on GPU, and transfer no unselected view. Assert direct augmented/SAM overlap is zero.
- [x] Add complete startup config and final `randaugment:` audit output with selected/eligible batches and images, ratio, last selected progress, cutoff, fixed policy, RNG namespace, and zero isolation/overlap failures.

### Milestone 2: Prove parent semantics and transform correctness
- [x] Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, and `git diff --check`; inspect `git diff -- train.py` and require only the approved transform, selection, and audit changes.
- [x] Print and verify the installed torchvision version and resolved 14-operation magnitude table. Over at least 10,000 samples require at least 70% pixel-changing outputs, mean absolute uint8 delta at least 1.0, p99 absolute delta at least 8, and at least 10/14 operation magnitudes structurally nonzero.
- [x] Snapshot worker-global torch, Python, and NumPy RNG around the wrapper. Require private state to advance even on an injected exception, global states to restore bitwise, eight distinct production worker seed keys/initial draws, and new replayable keys after worker recreation.
- [x] Compare actual parent and candidate DataLoaders from restored main RNG states over at least two worker-recreated epochs. Require identical target order and bitwise-equal candidate clean tensors versus parent tensors; require candidate augmented sequences to replay bitwise across repeated fixed-state runs.
- [ ] On fixed inputs, run early CutMix and late ordinary/SAM GPU-0 integration smokes. Require unchanged CutMix decisions/geometry, clean late tensors, model state, parent RNG streams, six replayed drop-path draws, one BatchNorm update, exact SAM restoration, and one Nesterov update.

### Milestone 3: Pass CPU/GPU-0 feasibility gates
- [ ] Confirm physical GPU 0 is the 97,871 MiB NVIDIA H20, branch is EXP-008 at parent `1a8d0de`, only `train.py` differs, and no stale `run.log` or repository-root smoke artifact exists.
- [x] In separate fixed-state processes, time at least five full parent and paired-view candidate DataLoader epochs including the RNG-state wrapper. Report median/p90 inter-arrival and sustained images/second; candidate clean targets/order must remain exact.
- [ ] Benchmark at least 1,000 early candidate training steps and matched parent steps on GPU 0 with production DataLoaders, synchronization, BF16, channels-last, and no evaluation. Separately benchmark at least 200 late clean SAM pulses to confirm the paired loader does not alter GPU hot-path latency.
- [ ] Proceed only if the worst of five candidate loader epochs sustains at least 1.20x the measured early GPU consumption rate, projected optimizer exposure is at least 25,000, worst-case total runtime projects below 550 seconds, peak allocation is finite, and no worker queue stalls, exceptions, or repository processes remain. A failed fixed-package gate rejects the experiment; do not change policy or worker count to pass.

### Milestone 4: Run and verify exactly once
- [ ] Reconfirm GPU/process/log preconditions, then launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`.
- [ ] Monitor health and timeout conditions only. Intermediate accuracy cannot prune, tune, or trigger a retry.
- [ ] Parse configuration, RandAugment/CutMix/SAM exposure, timing, evaluation count, and summary integrity before reading `best_test_acc`.
- [ ] Compare the single metric read with 95.50% for the tree verdict and 95.70% for the mechanism-sized hypothesis; preserve all raw evidence for Claude's adversarial result review before removing `run.log`.

## Code Changes

- **`train.py` only**:
  - Import only standard-library RNG-state helpers if required; use the already installed torchvision transforms. Do not change dependencies or `prepare.py`.
  - Split the parent transform into stochastic crop/flip and deterministic tensor conversion. Invoke crop/flip once, materialize the exact parent clean FP32 normalized tensor, apply fixed RandAugment to a PIL copy, and materialize only uint8 for the augmented view.
  - Initialize a private `torch.Generator` lazily with a mixed 64-bit key derived from current worker seed/id and namespace 42. Track the key and reinitialize when it changes; define a separate explicit `num_workers=0` key. Never touch CUDA or CutMix generators.
  - In `try/finally`, save global torch state, install private torch state, call the unmodified torchvision policy, capture the resulting private state even on failure, then restore global torch state before re-raising. Do not copy Python/NumPy states per sample; source inspection and preflight assert the policy never consumes them.
  - Accept the collated paired batch, compute one `progress` scalar, and use it for selection, CutMix, and SAM. For augmented batches transfer uint8 then convert/divide/subtract the preallocated GPU mean in FP32; for clean batches preserve the parent FP32 transfer. The unselected view remains only until the collated container leaves scope and is never transferred.
  - Keep the exact parent main-loop CutMix predicate and dedicated CPU/CUDA generator draws. Because `RANDAUGMENT_END == CUTMIX_END == SAM_START`, assert a RandAugment-selected batch cannot enter SAM.
  - Preserve model construction/initialization, parameter count, DataLoader sampler/worker/pinning/drop-last settings, all optimizer/LR/drop-path/SAM code, evaluator calls, and required final summary keys.
  - Add fixed config and exposure audits. Model-exposure counters, not worker-generated-but-unused views, define selected RandAugment dose. Print enough precision on the last selected progress to verify `<0.75` directly.

## Configuration Changes

- Per-image policy: parent crop/flip only -> parent crop/flip plus one audited RandAugment operation at magnitude 5 during progress `<0.75`.
- RandAugment: `num_ops=1`, `magnitude=5`, `num_magnitude_bins=31`, `InterpolationMode.NEAREST`, `fill=0`, private namespace seed 42.
- RandAugment cutoff: 0.75, exactly equal to parent `CUTMIX_END` and `SAM_START`; the final quarter remains parent-clean except for its unchanged crop/flip.
- Model/optimizer/SAM/CutMix: unchanged, including 2,748,890 parameters, CutMix probability 0.5, SAM `rho=0.05`, and period two.

## Execution Environment

- Method: local single-process training from the repository root; eight existing DataLoader workers.
- Resources: physical GPU 0 only via `CUDA_VISIBLE_DEVICES=0`; NVIDIA H20 with 97,871 MiB; no dependencies added.
- Estimated runtime: 300 charged seconds and 470-550 total seconds; hard outer timeout 600 seconds.
- Log output: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log` captures stdout/stderr and is the sole metric source.
- Tool skill: none; local GPU execution is monitored through the process session and log.

## Abort Criteria

- Before launch, abort if GPU 0 is not the H20, another process from this repository is active, branch/base/scope is wrong, a stale log exists, static checks fail, the fixed policy/table or pixel-effect gate fails, paired clean tensors differ from the parent, worker streams collapse, RNG isolation/replay fails, CutMix/SAM parent semantics differ, or worst-case feasibility projects fewer than 25,000 steps or at least 550 total seconds.
- During the run, stop on traceback, worker exception/death, CUDA/OOM error, nonfinite loss, RNG restoration failure, RandAugment/SAM overlap, SAM restoration or BatchNorm failure, missing progress for more than 60 seconds, or a trajectory that makes the 600-second timeout unavoidable.
- Do not abort for weak intermediate accuracy. After any evaluation output has been read, no retry is allowed. Before any evaluation output, one deterministic repair is permitted only for a named implementation/infrastructure defect unrelated to accuracy; preserve its evidence.
- Exit 124/nonzero, incomplete/duplicate summary, charged time outside 299.5-301.0 seconds, total time at least 600 seconds, parameter-count change, evaluation-count mismatch, or audit failures is a protocol failure. Any completed step count is valid protocol evidence; below 25,000 is exposure-degraded and falsifies the mechanism exposure target without authorizing a retry.

## Verification Protocol

### Verification Procedure

1. **Parent, scope, and hardware** (20-second timeout):
   - Run `tree.sh show` for node 004 and require metric 95.40 / commit `1a8d0de`; compute thresholds as 95.50 and 95.70.
   - Require branch `tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-008`, parent ancestry, only tracked `train.py` changed, and no stale log.
   - Run `nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader` and the compute-process query. Require physical index 0, H20, 97,871 MiB, no process from this repository, and record unrelated co-tenants.

2. **Static policy and scope** (60-second timeout):
   - Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, and `git diff --check`.
   - Inspect source/config and instantiate the policy. Require the exact five fixed settings, cutoff equality, unchanged 2,748,890 parameters, 195 batches per epoch, unchanged model state from identical seeds, and no global RNG consumption during model construction.

3. **Transform and RNG semantics** (180-second timeout):
   - Exercise the production paired transform under both zero and eight workers plus injected exceptions. Require same-crop construction, clean-parent bitwise parity, deterministic full-stream augmented replay, distinct seed-keyed worker streams, private-state advance on failure, and exact restoration of torch/Python/NumPy global states.
   - Run two parent/candidate DataLoader epochs from restored main states. Require identical target order and clean tensors, 49,920 identities per dropped-last epoch, distinct but replayable augmented tensors, and no main CPU/CUDA/CutMix RNG drift.

4. **Training integration** (180-second timeout):
   - Load actual parent source with `git show 1a8d0de:train.py` into a non-main namespace. On matched inputs/model/RNG, require early candidate transformation is the only semantic difference, while CutMix decisions and geometry remain fixed.
   - At progress at or above 0.75, require the selected candidate tensor is bitwise the parent clean tensor and actual-parent ordinary/SAM outputs, losses, gradients, optimizer state, BatchNorm buffers, CUDA RNG, and exact restore match. Require one bound progress value per step and zero `sam_active && selected_augmented` events.
   - Require all RandAugment-selected batches are below 0.75, all SAM batches are at or above 0.75, and overlap count is zero.

5. **Loader/GPU-0 feasibility** (240-second timeout):
   - Time five complete parent/candidate DataLoader epochs with the production wrapper, then matched 1,000-step early GPU loops and 200 late SAM pulses in separate fixed-seed GPU-0 processes. Synchronize GPU timings and tear down every PID.
   - Preflight runs are separate invocations from the 600-second full-run timeout. Report candidate/parent loader median/p90 and worst-epoch inter-arrival, sustained throughput, charged/total projection, step projection, and peak VRAM. Require worst-epoch loader rate at least 1.20x measured early GPU consumption, at least 25,000 projected steps, and under 550 projected total seconds; never change fixed configuration to pass.

6. **Single full run** (610-second timeout):
   - Reconfirm launch preconditions and execute exactly `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` once. Monitor only health patterns until completion.

7. **Protocol integrity before metric** (30-second timeout):
   - Parse unique startup/audit/summary fields. Require fixed policy/table and cutoff, nondegenerate preflight pixel statistics, positive selected dose only below 0.75, RandAugment selected/eligible ratio exactly one, CutMix ratio `[0.49,0.51]`, period-two SAM ratio `[0.499,0.501]`, zero direct overlap failures, 299.5-301.0 charged seconds, total below 600, exact parameter count, and one evaluation per epoch. Worker-stream distinction and RNG isolation are preflight gates; any production wrapper failure raises and makes the run nonzero.
   - The printed last RandAugment-selected progress must be `<0.75` at sufficient precision. Require complete final summary and no error signature.

8. **Primary and mechanism verdicts** (10-second timeout):
   - Only after protocol integrity passes, run `grep '^best_test_acc:' run.log` and parse exactly one percentage.
   - `>=95.50%` passes the necessary tree condition; `<95.50%` is no-improvement. Independently, `>=95.70%` supports the preregistered mechanism-sized hypothesis. A 95.50-95.69 result is formally improved but below the 0.30-point evidentiary bar. Do not use loss, runtime, or exposure to upgrade the accuracy conclusion.

9. **Evidence and cleanup** (10-second timeout):
   - Record all raw audit and summary values in `03-execute.md` and `04-analysis.md`, obtain Claude's adversarial result review, then remove `run.log`. Require `git status --porcelain --untracked-files=all` to show only the intended tracked `train.py` change before the analysis commit.

### Informational Metrics (Optional)

- `final_test_acc`, `final_test_loss`: final summary, compared with parent 95.40% / 0.1654; informational only.
- `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`: final summary, compared with 300.0 / 457.3 / 1.2 / 1,190.5.
- `num_epochs`, evaluation count, `num_steps`, `num_params`: summary/log, compared with 132 / 132 / 25,560 / 2,748,890.
- RandAugment/CutMix/SAM doses and cutoff timestamps: final audit lines; report exact model exposure and confirm the clean-tail boundary.
- Tail context: report last-five accuracy mean and evaluation count alongside the frozen best metric; these never alter the verdict.
