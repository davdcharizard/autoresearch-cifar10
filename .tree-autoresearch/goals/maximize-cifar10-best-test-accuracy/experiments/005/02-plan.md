# Plan EXP-005: Clean-Gated Last-Mini-Batch Self-Distillation
- **Created**: 2026-08-05

## Goal and Fixed Reference

- Parent node: EXP-004 (`1a8d0de`), `best_test_acc=95.40%`.
- Primary acceptance threshold: `best_test_acc >= 95.50%`.
- Chosen mechanism: DLB with half-overlapping batches, `tau=3`, `alpha=1`, and identity-safe clean-clean gating.
- Expected true gain after discounting the paper result for partial early coverage and the stronger parent: 0.10-0.30 percentage points.
- Explicit confound: the required overlapping sampler changes the data schedule, not merely its order. It halves new-identity introduction, repeats fresh views one step apart, and changes parent global-CPU RNG consumption even though all new sampler and worker generators are fixed at seed 42. Any success supports the combined repeat-view sampler plus clean-gated KL mechanism; isolating KL requires a future `alpha=0` control.

## Milestones

### Milestone 1: Implement an auditable half-overlap stream
- [x] Add fixed DLB constants and small `IndexedDataset` and `OverlappingBatchSampler` helpers in `train.py` only.
- [x] Draw a seed-42 permutation of all 50,000 train indices, retain 49,920 indices, split them into 390 chunks of 128, and emit exactly 389 natural batches `[chunk_i, chunk_(i+1)]`.
- [x] Give sampler ordering and DataLoader worker seeding separate dedicated seed-42 CPU generators; do not consume the global model/drop-path or dedicated CutMix RNG streams.
- [x] Return raw sample indices with every transformed image and assert that each current first half equals the cached prior outgoing indices before applying KL.
- [x] Keep the natural DLB epoch at 389 optimizer steps. Do not redefine or split epochs to increase validation frequency.
- [x] Verification: run the deterministic sampler/transition smoke described below and require 390 unique chunks, 49,920 unique indices, 389 batches, exact half-overlap, reproducibility, and unchanged global RNG states.

### Milestone 2: Integrate clean-gated DLB with CutMix and SAM
- [x] Preserve the current supervised CE/CutMix loss and add FP32 `tau^2 * KL(teacher || student)` only when the current batch is clean and an aligned clean teacher cache exists. Form distributions explicitly as `softmax(cached_logits / tau)` and `log_softmax(current_logits / tau)`.
- [x] Publish detached FP32 primary-forward logits and CPU indices from the outgoing second half only after a successful optimizer update on a clean batch.
- [x] Invalidate both cache tensors after every mixed batch and reset them at each natural epoch boundary.
- [x] On scheduled SAM steps, include the same incoming detached teacher in both unperturbed and perturbed CE+KL objectives; retain EXP-004 RNG replay, second-pass BatchNorm suppression, exact restoration, and one optimizer update.
- [x] Never publish perturbed second-pass logits. Preserve the primary logits until the update succeeds, then publish them.
- [x] Add audit counters for clean batches, active DLB batches/examples, cache publications, mixed invalidations, epoch resets, and overlap mismatches. Add the fixed DLB recipe to the startup config.
- [x] Verification: compile/lint; run loss-direction, temperature-sensitive, cache-transition, BF16/channels-last, and SAM integration smokes; require finite gradients, teacher detachment, exact cache source, zero mismatches, and unchanged SAM invariants.

### Milestone 3: Run one full GPU-0 experiment
- [x] Confirm physical GPU 0 is an NVIDIA H20 with approximately 98 GB memory.
- [x] Confirm the only tracked diff is `train.py`, the branch is the EXP-005 branch at parent commit `1a8d0de`, and no stale `run.log` exists.
- [x] Launch once with the fixed seed and 600-second outer timeout, capturing all output in `run.log`.
- [x] Monitor for completion, timeout, traceback, CUDA error, nonfinite loss, or overlap assertion failure without exposing or using intermediate test accuracy for decisions.
- [x] Verification: require exit 0, a complete summary, approximately 300 charged seconds, total runtime below 600 seconds, natural 389-step epochs, zero overlap mismatches, and an auditable DLB/SAM exposure trace.

### Milestone 4: Verify the preregistered outcome
- [x] Parse `best_test_acc` and compare it once against the fixed 95.50% threshold; do not rerun or adjust a parameter based on the test result.
- [x] Report final-versus-best accuracy, final loss versus the parent's 0.1654, evaluation count, unique-image-rate change, DLB coverage, step count, runtime, VRAM, and model size.
- [x] Interpret flat accuracy with final loss worse than 0.1654 as evidence that the CE+KL SAM tail was over-regularized, not as grounds for a retry.
- [x] Attribute the result only to the combined repeat-view sampler plus KL intervention, not KL alone; record that an `alpha=0` control is required for isolation. Remove `run.log` after its evidence has been recorded in `03-execute.md` and `04-analysis.md`.

## Code Changes

- **`train.py` only**:
  - Add `DLB_TAU=3.0`, `DLB_ALPHA=1.0`, `DLB_HALF=BATCH_SIZE//2`, and `DLB_SEED=42`; assert that `BATCH_SIZE` is even.
  - Add an index-returning wrapper around the existing CIFAR-10 training dataset. It must call the wrapped dataset independently on both appearances so repeated identities receive fresh crop/flip transforms rather than a reused transformed tensor.
  - Add a deterministic batch sampler. Each `__iter__` draws one permutation from its private generator, truncates to 390 full half-batches, and yields 389 overlapping full batches. `__len__` returns 389. Use it through DataLoader `batch_sampler=` while preserving eight workers, pinned memory, and existing transforms.
  - Keep separate seed-42 generators for sampler permutations and DataLoader worker base seeds. Pass the latter explicitly as `DataLoader(generator=dlb_worker_generator)` and use it for nothing else. The existing global seed 42 and dedicated CutMix CPU/CUDA seed-42 generators remain unchanged, but the global CPU RNG trajectory is not claimed to match the parent because the parent loader consumed it for shuffle seeding.
  - Carry sample indices through the loop on CPU. At every potential DLB transition, compare `indices[:DLB_HALF]` exactly against cached outgoing indices; any inequality increments the mismatch counter and raises immediately.
  - Compute `teacher_prob = softmax(cached_logits.float() / DLB_TAU)` and `student_log_prob = log_softmax(current_logits.float() / DLB_TAU)`. Use `F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * DLB_TAU**2`, then add `DLB_ALPHA * dlb_loss` to the full-batch supervised loss. Do not divide by batch size again.
  - Gate KL to clean-clean transitions. A current mixed batch ignores any incoming cache and clears the cache after its update. A clean batch with no cache performs ordinary CE and publishes a new outgoing cache.
  - Preserve primary outputs across SAM's second pass. Recompute the identical DLB term against the same incoming teacher for the perturbed objective, but publish only `primary_outputs[DLB_HALF:]` after exact restoration and the sole optimizer update.
  - Reset the cache once at every outer epoch start. Log the natural 389-batch epoch size and all cache/DLB audit counters without per-step `.item()` synchronizations beyond the existing post-sync loss read.
  - Do not change model architecture, parameter initialization, optimizer, LR/drop-path schedules, batch size, CutMix gate/geometry/RNG/cutoff, SAM rho/start/cadence, evaluator use, time accounting, or final summary keys.

## Configuration Changes

- `DLB_TAU`: absent -> `3.0` (published DLB CIFAR setting; no tuning).
- `DLB_ALPHA`: absent -> `1.0` (published DLB loss weight; no tuning).
- `DLB_HALF`: absent -> `128` (half of the unchanged batch size 256).
- `DLB_SEED`: absent -> `42` (matches the frozen experiment seed while isolating sampler state).
- Training batch construction: 195 independent shuffled batches per parent epoch -> 389 half-overlapping batches per natural DLB epoch. This is the mechanism, not an evaluation-cadence optimization.
- All EXP-004 constants remain unchanged, including `CUTMIX_PROB=0.5`, `CUTMIX_END=0.75`, `MAX_DROP_PATH=0.08`, `SAM_RHO=0.05`, `SAM_START=0.75`, and `SAM_PERIOD=2`.

## Execution Environment

- Method: local single-process run from the repository root.
- Resources: physical GPU 0 only, NVIDIA H20 with approximately 98 GB memory; eight existing DataLoader workers; no new package or dependency.
- Estimated runtime: 300 seconds of charged training and approximately 380-500 seconds total; hard outer limit 600 seconds.
- Log output: `run.log`, populated by redirecting both stdout and stderr. This is the sole full-run evidence source until analysis is recorded.
- Full command: `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Monitoring: inspect the redirected log for process health only. Intermediate test accuracy must not trigger stopping, retrying, or configuration changes.
- Tool skill: use `tree-autoresearch:active-log-monitor` during execution if compatible with the local redirected log.

## Abort Criteria

- Abort before launch if physical GPU 0 is not the approximately 98 GB NVIDIA H20, the branch/parent commit is wrong, a protected tracked file changed, a dependency was added, or any deterministic sampler/loss/SAM smoke fails.
- During the run, stop on a traceback, CUDA/OOM error, NaN/Inf or nonfinite gradient/loss, DLB overlap mismatch, stale-cache assertion, or missing progress for long enough to make the 600-second timeout unavoidable.
- `timeout` exit code 124, any other nonzero exit, no complete summary by 600 seconds, or charged time outside 299.5-301.0 seconds is an execution failure. The inherited loop updates charged time after every synchronized step and breaks immediately, so natural epoch length cannot add an epoch-sized overshoot.
- Do not prune for low or flat intermediate accuracy. Do not abort merely because natural DLB epochs produce about half as many evaluations as the parent.
- No metric-driven retry is allowed. A deterministic implementation or infrastructure defect before any valid final summary may be repaired and documented; after a valid summary, the result is final for EXP-005.

## Verification Protocol

### Verification Procedure

1. **Parent and scope check** (10-second timeout):
   - Run `test -x /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh` and require exit 0, then run `bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 004` and require `metric=95.40`, `commit=1a8d0de`, and `is_extendable=true`.
   - Run `git branch --show-current && git rev-parse --short HEAD && git status --short` and require the EXP-005 branch, parent `1a8d0de`, and only the planned `train.py` tracked diff.
   - Run `git diff --name-only` and require exactly `train.py`; specifically require no diff in `prepare.py` or `pyproject.toml`.

2. **Static checks** (60-second timeout):
   - Run `uv run python -m py_compile train.py` and require exit 0.
   - Run `uv run ruff check train.py` and require exit 0. Do not run `ruff format`: the inherited EXP-004 file is not format-clean, and formatting it would introduce unrelated churn.
   - Inspect `git diff -- train.py` and confirm all modifications implement DLB/auditing only and leave evaluator, fixed time budget, seed 42, CutMix, and SAM constants intact.

3. **Deterministic semantic smoke** (120-second timeout):
   - Run an inline `uv run python` smoke using the implemented sampler with a toy 50,000-index dataset. Require `len(sampler)==389`, 390 chunks covering 49,920 unique indices, `batch_t[:128] == batch_(t-1)[128:]` for all adjacent batches, identical output from fresh seed-42 sampler instances, and unchanged global CPU/CUDA RNG states.
   - Call the wrapped dataset twice for one raw index under controlled transform seeds and verify the wrapper returns the same identity while permitting independently generated transformed tensors.
   - Numerically compare the DLB helper/objective with manual `tau^2 * KL(teacher || student)` using non-identical logits divided by `tau=3`. Require agreement with the tau-three manual result, disagreement with tau one, approximately zero for identical logits, no cached-logit gradient, and KL gradient only on the current repeated-half logits.
   - Exercise clean-clean, clean-mixed, mixed-clean, and epoch-reset cache transitions. Require DLB only on aligned clean-clean transitions, immediate invalidation after mixing, no stale teacher, and zero mismatches.

4. **GPU/SAM integration smoke** (120-second timeout):
   - Expose only physical GPU 0 and run one ordinary clean DLB step and one scheduled BF16/channels-last SAM+DLB step on the full WRN.
   - Require finite objectives/gradients, one/two forwards respectively, the same incoming teacher on both SAM passes, a perturbation norm of 0.05, exact parameter restoration, one BatchNorm-buffer update, one Nesterov update, primary-only cache publication, and unchanged CUDA RNG parity.
   - Require `num_params=2,748,890`; DLB adds no model parameter.

5. **Hardware and full run** (610-second timeout):
   - Run `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader` and require physical index 0 to report NVIDIA H20 and approximately 97,871 MiB.
   - Remove only a stale transient `run.log` if present, then run `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` exactly once.
   - Require exit 0. If metric extraction is empty, inspect `tail -n 80 run.log` and classify the run as failed unless it is a repairable pre-summary implementation/infrastructure defect.

6. **Protocol integrity** (30-second timeout):
   - Run `grep -E "^(config:|Batches per epoch:|dlb:|sam:|cutmix:|---|best_test_acc:|final_test_acc:|final_test_loss:|training_seconds:|total_seconds:|startup_seconds:|peak_vram_mb:|num_epochs:|num_steps:|num_params:)" run.log`.
   - Require `Batches per epoch: 389`, DLB recipe `tau=3 alpha=1 half=128 seed=42`, zero overlap mismatches, nonzero DLB activity, CutMix applied/eligible ratio in `[0.47, 0.53]`, SAM first progress in `[0.7500, 0.7520]`, an even positive SAM first step, SAM applied/eligible ratio in `[0.499, 0.501]`, unchanged 2,748,890 parameters, 299.5-301.0 charged seconds, total runtime below 600 seconds, and every required summary key.
   - Count `eval ep` lines and require exactly `num_epochs`, proving one evaluation per natural epoch and no cadence-oriented epoch redefinition. In the inherited loop, `epoch` increments before the pass and even the budget-truncated final pass is evaluated before exit, so this relation includes the final partial epoch. Roughly 60-70 evaluations versus the parent's 132 is an informational expectation, not a hard range.

7. **Primary verdict** (10-second timeout):
   - Run `grep "^best_test_acc:" run.log` and parse the percentage.
   - `>=95.50%`: necessary metric condition passes; `<95.50%`: no improvement. Do not rerun, tune, or choose another seed.
   - A valid official improvement additionally requires exit 0, the fixed charged budget, complete summary, GPU-0/H20 scope, unchanged evaluator, and zero protocol-integrity failures.
   - `num_steps >= 24,500` is the preregistered throughput prediction and part of full hypothesis support, but it is diagnostic rather than an independent goal-level acceptance condition if every hard constraint and the primary metric pass.

8. **Cleanup** (10-second timeout):
   - After transcribing all evidence into the experiment artifacts, remove `run.log` with `rm -f run.log` and require `git status --short` to contain no log or unintended tracked changes.

### Informational Metrics (Optional)

- `final_test_acc`: final summary in `run.log`; compare with best to quantify checkpoint-max dependence under halved evaluation count.
- `final_test_loss`: final summary; compare with parent 0.1654. Flat accuracy plus worse loss is the preregistered over-regularized-tail diagnostic.
- `training_seconds`, `total_seconds`, `startup_seconds`: final summary; distinguish charged GPU work from longer natural epoch/worker boundaries.
- `peak_vram_mb`: final summary; compare with parent 1,190.5 MiB.
- `num_epochs`, evaluation count, `num_steps`: final summary plus `eval ep` line count; compare with parent 132 epochs and 25,560 steps.
- `num_params`: final summary; require unchanged 2,748,890.
- DLB exposure: final `dlb:` audit line, including active batches/examples, clean batches, cache publications/invalidations/resets, and zero mismatches.
- Unique-image rate: derived from fixed sampler construction, 49,920 unique raw identities per 389-step natural epoch; report the approximately twofold reduction in new identities per optimizer step.
- Attribution: report the result as evidence for the combined repeat-view sampler plus clean-gated KL mechanism; a future fixed-seed `alpha=0` control is required to isolate the KL contribution.
