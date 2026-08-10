# Plan EXP-018: Direct canonical Lookahead on EXP002
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the fixed parameter-only Lookahead state machine
- [x] Add fixed `LOOKAHEAD_K=5` and `LOOKAHEAD_ALPHA=0.5`; clone all 44 trainable model parameters into name/order-aligned, detached FP32 slow tensors after model construction without consuming RNG, and assert 2,748,890 slow elements with no gradients or optimizer ownership.
- [x] Preserve the inherited SGD optimizer, parameter order, momentum, Nesterov, and effective coupled weight decay. Immediately after each ordinary `optimizer.step()`, define `completed_step = step + 1` from the inherited pre-increment counter; if and only if `completed_step % 5 == 0`, run `slow += 0.5 * (fast - slow)` followed by an in-place slow-to-fast copy under `torch.no_grad()`. Assert the first sync is step 5 and every recorded sync index is divisible by five; never reset, interpolate, or rescale momentum.
- [x] Charge interpolation and audit work inside the existing timed step before CUDA synchronization. Apply Lookahead throughout all early CutMix, early-clean, and late-clean batches without a phase gate or extra forward.
- [x] Keep BatchNorm running statistics and integer buffers live and outside slow state. Do not change architecture, initialization, data/CutMix/drop-path RNG, BF16/channels-last execution, LR schedule, evaluator cadence, or best-checkpoint selection.

### Milestone 2: Make slow evaluation and mechanism audits exact
- [x] Evaluate the slow parameters at every inherited evaluation. Whenever `completed_step % 5 != 0` at any evaluation boundary, snapshot fast parameters, copy slow parameters into the model, call the frozen evaluator exactly once, and restore fast parameters and every module training flag in `finally`; otherwise evaluate the already synchronized live parameters. Assert each evaluation is either synchronized or explicitly swapped. With the inherited `drop_last=True` loader, full 195-step epochs should be synchronized because `195 % 5 == 0`, while the budget-truncated final epoch is the expected swap case.
- [x] Prove the temporary evaluation swap restores every fast parameter bitwise using a fresh model-parameter enumeration, independently of the lists used for copying; assert optimizer parameter identities and momentum-buffer identities remain unchanged and the swap consumes no RNG.
- [x] Record inner steps, expected/actual sync count, first/last sync, steps since sync, early-CutMix/early-clean/late-clean sync counts, total slow elements, and synchronized versus swapped evaluations. Require `first_sync == 5`, `last_sync == LOOKAHEAD_K * actual_syncs`, every sync index divisible by five, and every evaluation classified exactly once as synchronized or swapped.
- [x] At syncs, accumulate finite FP64 squared pre-interpolation slow-fast distance and interpolation displacement into fixed device scalars without retaining per-step CUDA tensors or synchronizing. Report normalized mean/max pre-sync distance, cumulative displacement, and final normalized slow-fast distance/phase; also retain the final 16 evaluation accuracies for mean/range/final/best-premium context.

### Milestone 3: Pass deterministic correctness and decisive GPU-0 feasibility gates
- [x] Run syntax/scope checks and deterministic tiny/full-model smokes proving RNG-neutral slow initialization, parent equality through steps 1-4, explicit-reference equality at step 5, retained momentum, BN-buffer exclusion, FP32 slow state, nonfinite detection, and exception-safe evaluation restoration.
- [x] Guard the evaluator and test-loader during all feasibility work. On real CIFAR BF16/channels-last clean and CutMix batches, prove unchanged parent forward/backward behavior before the first sync, correct fifth-step interpolation, exact counters, fixed-scalar diagnostics, and stable allocation after all persistent optimizer/slow/audit state exists.
- [x] Run one accuracy-blind, five-round alternating-order parent/candidate preflight on physical GPU 0 reflecting the 37.5% CutMix, 37.5% early-clean, and 25% late-clean workload. The candidate path must call the exact Lookahead sync/audit helper imported from `train.py`, and both arms must time the same production charged region. Require parent drift <=4%, ratio MAD/median <=1.5%, median candidate/parent charged latency <=1.01, every ratio <=1.03, projected steps >=27,500, projected epochs >=141, projected total runtime <600 seconds, and no persistent allocation growth.
- [x] Treat the first complete numeric preflight as decisive. Permit at most one documented repair only for a malformed/exceptional pre-vector temporary harness; do not repair, retune, or rerun a complete numeric failure.

### Milestone 4: Run exactly one fixed-seed metric experiment and verify it
- [x] Remove stale `run.log`, reconfirm physical GPU 0 and one-device visibility, then run `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1` exactly once.
- [x] Monitor only liveness and integrity abort criteria; finite intermediate loss or accuracy never changes the run. Transcribe the complete final summary, all Lookahead diagnostics, CutMix dose, achieved steps/epochs, and final-16 evaluations into `03-execute.md` before deleting the transient log.
- [x] Require a complete 299.5-301.0-second charged run, total runtime below 600 seconds, one evaluation per epoch, unchanged 2,748,890 model parameters, exact Lookahead reconciliation, finite nonzero displacement, and only tracked `train.py` changed.
- [x] Classify `best_test_acc >=95.33%` as the formal local improvement over EXP002's 95.23%. Report 95.53 as stronger but noise-limited context, 95.61 as global-best match, and 95.71 as resolution-clearing global improvement; tail stability and achieved exposure inform interpretation but never override the frozen verdict.

## Code Changes
- **`train.py` only / fixed state**: add `LOOKAHEAD_K=5` and `LOOKAHEAD_ALPHA=0.5`, then create one detached FP32 slow tensor per one of the model's 44 trainable parameters. Assert name/order/shape/dtype/device coverage, 2,748,890 total elements, no gradients, and no optimizer ownership. Model parameters remain the optimizer's live fast parameters; persistent model buffers are excluded.
- **`train.py` only / canonical update**: immediately after every inherited `optimizer.step()`, compute `completed_step = step + 1`; synchronize if and only if `completed_step % LOOKAHEAD_K == 0` via one `torch._foreach_lerp_(slow_parameters, fast_parameters, 0.5)` and one `torch._foreach_copy_(fast_parameters, slow_parameters)` under `torch.no_grad()`. Put this exact operation in a small helper that the preflight imports, and assert step 5 is the first sync. The pinned `torch==2.9.1` environment was checked to provide both foreach APIs. Retain all SGD momentum buffers unchanged. The interpolation, path counters, and fixed-scalar audits precede the existing CUDA synchronization so their cost is charged.
- **`train.py` only / slow evaluation**: preserve one evaluator call per epoch and the inherited `best_acc=max(test_acc)` rule. At any evaluation boundary with `completed_step % 5 == 0`, evaluate the already-equal live parameters. At every unsynchronized boundary, copy fast parameters into preallocated restore tensors, install slow parameters, evaluate with live BN buffers, and restore fast parameters/module flags in `finally`; independently assert bitwise restoration and unchanged optimizer state identities. This keeps evaluation on slow weights even if loader length or epoch termination changes unexpectedly.
- **`train.py` only / audits and summary**: report sync cadence and phase counts, slow-state inventory, evaluation ownership, normalized slow-fast distance and displacement, finiteness/restoration failures, and final-16 accuracy statistics. Diagnostics are report-only except preregistered integrity assertions and use fixed device accumulators rather than retained loss/distance tensors. Preserve every inherited final-summary key and append diagnostics without changing the evaluator or primary metric.

## Configuration Changes
- `LOOKAHEAD_K`: absent -> `5` (canonical NeurIPS 2019 setting fixed before any accuracy observation).
- `LOOKAHEAD_ALPHA`: absent -> `0.5` (canonical slow/fast interpolation; never tuned from preflight or metric output).
- Slow state: absent -> one FP32 clone of each of 44 trainable parameters, 2,748,890 elements; no BN running or integer buffers.
- Momentum policy: inherited Nesterov buffers -> retained unchanged across Lookahead synchronization, matching the canonical wrapper policy.
- Evaluation ownership: inherited live parameters -> slow parameters; any unsynchronized evaluation uses an exact temporary swap. Under the inherited 195-batch full epoch, only the truncated final epoch is expected to need it, but correctness does not depend on that expectation.
- Interpretation thresholds: 95.33% formal local pass, 95.53% stronger noise-limited signal, 95.61% global-best match, and 95.71% resolution-clearing global lift. Only 95.33% controls the tree verdict.

## Execution Environment
- Method: local execution from the repository root. Deterministic smoke and paired preflight scripts/logs live only under `/tmp/exp018-*`; the sole metric command is `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- Resources: physical GPU 0 only, NVIDIA H20 with approximately 97,871 MiB; exactly one CUDA device visible to every GPU command. Expected model plus slow-state memory remains tiny relative to 98 GB, with OOM or unexplained persistent allocation growth treated as failure.
- Estimated runtime: correctness and decisive preflight under 5 minutes; sole metric run approximately 470 seconds total with exactly 300 charged training seconds and a hard 600-second outer timeout.
- Log output: `/tmp/exp018_preflight.log` for accuracy-blind feasibility and repository-local `run.log` for the one metric run until exact durable transcription. Set `PYTHONPYCACHEPREFIX=/tmp/exp018-pycache` for transient Python commands because shared `/tmp/__pycache__` may be unwritable.
- Tool skill: none; execution is a local single-GPU process.

## Abort Criteria
- Stop before metric launch for a wrong GPU/UUID, more than one visible CUDA device, evaluator/test-loader access during preflight, tracked scope beyond `train.py`, syntax/diff failure, missing pinned foreach APIs, RNG drift, parent mismatch before step five, explicit-reference mismatch at step five, a first sync other than step 5, any sync index not divisible by five, slow-state inventory/ownership/dtype failure, BN-buffer inclusion, momentum mutation, nonfinite state, restoration failure, an unclassified evaluation boundary, counter mismatch, or unexplained persistent allocation growth.
- The first complete paired preflight is decisive. Abort the leaf if parent drift exceeds 4%, ratio MAD/median exceeds 1.5%, median weighted candidate/parent latency exceeds 1.01, any round ratio exceeds 1.03, projected steps are below 27,500, projected epochs below 141, or projected total runtime reaches 600 seconds. Peak allocation is informational unless it OOMs or grows unexpectedly.
- During the metric run terminate on traceback, CUDA/OOM/device error, explicit audit/integrity assertion, nonfinite slow/fast/optimizer state, no progress for 90 seconds after startup, or the 600-second timeout. Never stop or adapt on finite loss or accuracy.
- A valid completed run with fewer than 27,500 steps or 141 epochs retains its formal accuracy verdict; these dose floors gate the preflight and remain mandatory interpretation context, not extra goal conditions. A complete run below 95.33% is a valid no-improvement, not permission to rerun or tune.

## Verification Protocol

### Verification Procedure

1. **Parent, scope, commands, and syntax** (30-second timeout):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.3/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 002
   git diff --name-only a36dc09
   git status --porcelain --untracked-files=all
   git diff --check
   env PYTHONPYCACHEPREFIX=/tmp/exp018-pycache uv run python -m py_compile train.py
   ```
   Require parent metric 95.23, only tracked `train.py` changed, no repository helper artifacts, and successful syntax/diff checks. The `.tree-autoresearch/` metadata is ignored and not experiment code.

2. **GPU identity and visibility** (10-second timeout):
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader
   env CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(torch.cuda.device_count(), p.name, p.total_memory)'
   ```
   Require physical index 0 `NVIDIA H20`, approximately 97,871 MiB, and exactly one visible CUDA device. Record the current physical UUID in `03-execute.md` rather than hard-coding a possibly machine-specific UUID into experiment logic.

3. **Deterministic mathematical and integration smoke** (120-second timeout):
   ```bash
   env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp018-pycache uv run python /tmp/exp018_lookahead_smoke.py
   ```
   Insert the fixed repository root into the temporary script's `sys.path`. Require RNG-neutral cloning; exact 44-tensor/2,748,890-element FP32 slow inventory; no slow gradients or optimizer ownership; parent/candidate equality through steps 1-4; explicit `slow += 0.5*(fast-slow)` equality and fast-to-slow equality at step 5; `first_sync == 5`; every later sync index divisible by five; unchanged momentum identity/value across interpolation; excluded BN buffers; exact sync/path counts; finite positive displacement; and exception-safe bitwise restoration with unchanged RNG and one evaluator call in both synchronized and deliberately unsynchronized cases. Confirm full-model BF16 autocast/channels-last compatibility and presence of `torch._foreach_lerp_`/`torch._foreach_copy_` in pinned torch 2.9.1 without compilation, graph capture, GradScaler, or added dependency.

4. **Decisive accuracy-blind paired preflight** (300-second timeout):
   ```bash
   timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp018-pycache PYTHONUNBUFFERED=1 uv run python /tmp/exp018_preflight.py > /tmp/exp018_preflight.log 2>&1
   ```
   Mechanically load/extract the EXP002 parent and candidate under `/tmp`, replace evaluator access with guards before traces, and never compute test accuracy. Use shared real CIFAR clean/CutMix inputs and aligned RNG to prove pre-sync forward/backward/update parity and fifth-step reference divergence. The parent and candidate arms time the identical region from the inherited per-batch `t0` through the same CUDA synchronization; both execute the same input, forward, loss, backward, and SGD calls, while the candidate additionally calls the exact imported production Lookahead sync/audit helper. After all persistent optimizer/slow/audit state exists, run a long candidate production trace using fixed device-scalar finiteness and release transient losses before allocation snapshots. Then run five alternating-order paired rounds reflecting 37.5% CutMix, 37.5% early-clean, and 25% late-clean work, with enough updates to exercise at least 30 Lookahead syncs.

   Define each round's workload-weighted ratio as `0.375 * cutmix_ratio + 0.375 * early_clean_ratio + 0.25 * late_clean_ratio`; use the median of those five weighted ratios as `R`. Compute `projected_steps = floor(27950 / R)` from EXP002's charged 27,950-step reference and `projected_epochs = ceil(projected_steps / 195)`. Compute `projected_total_seconds = 300 + 1.2 + projected_epochs * ((467.1 - 300 - 1.2) / 144) + measured_final_swap_seconds`, using EXP002's 467.1-second total, 1.2-second startup, and 144 evaluations, with the candidate's separately measured final swap cost. Emit these inputs and outputs in one complete JSON vector and require every preflight threshold and integrity condition above. The 1.5% dispersion gate diagnoses whether the five-round sample is usable; the stricter 1% median and 3% per-round ceilings are the preregistered scientific exposure limits, not confidence intervals. One recorded repair is allowed only if the temporary harness fails before a complete numeric vector; a numeric failure is final.

5. **Exactly one metric launch** (600-second timeout):
   ```bash
   rm -f run.log
   timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
   ```
   Require exit 0, `training_seconds` in `[299.5,301.0]` (the inherited 300-second budget plus at most one bounded final charged step), `total_seconds <600`, one evaluation per epoch, all inherited summary keys, `num_params=2,748,890`, unchanged CutMix/drop-path/LR/optimizer configuration, exact `actual_syncs == floor(num_steps/5)`, `first_sync == 5`, `last_sync == 5 * actual_syncs`, every evaluation synchronized or swapped exactly once, exact phase reconciliation, finite nonzero displacement, zero restoration/integrity failures, and only tracked `train.py` changed. Falling outside the charged-time interval is a frozen-budget integrity failure with no accuracy verdict or rerun, not a no-improvement. Record whether achieved steps and epochs meet 27,500/141 as dose context without overriding an otherwise valid formal result.

6. **Metric decision and durable context** (15-second timeout):
   ```bash
   grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:' run.log
   ```
   Re-query EXP002 and require 95.23. With all integrity conditions satisfied, `best_test_acc >=95.33%` is improvement and anything lower is no-improvement. Lead with that formal verdict; separately report whether 95.53, 95.61, and 95.71 were reached. Transcribe the final 16 epoch accuracies and compute their mean, min/max range, final value, and best-minus-mean premium; report final slow-fast distance and achieved sync/dose. These diagnostics cannot change the verdict. Preserve raw evidence through result review, then remove `run.log` and EXP018-owned `/tmp` artifacts.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: corresponding final summary lines in `run.log`.
- Lookahead mechanism: sync count and phase split, slow-state tensor/element inventory, synchronized/swapped evaluation counts, normalized pre-sync slow-fast distance, interpolation displacement, final normalized distance/phase, and restoration/finiteness failures from appended audit lines.
- Stability: final-16 evaluation values, mean, min/max range, final value, and best-minus-mean premium parsed from `run.log` and durably copied into `03-execute.md` before log removal.
- Exposure: achieved steps/epochs versus EXP002's 27,950/144, realized CutMix applied/eligible ratio, and total runtime; all are interpretation context only after necessary conditions pass.
