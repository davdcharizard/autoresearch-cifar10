# Plan EXP-019: Corrected reference-ordered Gradient Centralization
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the unchanged official-order GC mechanism
- [x] Inventory all 44 trainable parameter tensors after model construction. Select all and only the `weight` parameters of 16 `nn.Conv2d` modules plus the final `nn.Linear`: 17 eligible tensors, 2,745,264 elements, and 2,266 output rows; reconcile the remaining 27 BN-affine/bias tensors and 3,626 elements as excluded.
- [x] After every inherited backward, require every gradient, add coupled `WEIGHT_DECAY=1e-4` to all 44 FP32 directions with one `torch._foreach_add_`, subtract each eligible direction's mean over dimensions `1..ndim-1`, and apply all 17 broadcasts with one `torch._foreach_sub_`; then call unchanged PyTorch SGD momentum/Nesterov with internal `weight_decay=0` so decay is neither omitted nor doubled.
- [x] Keep GC active for every early-CutMix, early-clean, and late-clean step. Add no coefficient, phase gate, clipping, cast, stochastic draw, model parameter, forward/backward, optimizer subclass, compile path, or graph capture. Preserve architecture, initialization, data/CutMix/drop-path RNG, BF16/channels-last execution, LR schedule, and evaluator/max-selection semantics.
- [x] Place coupled decay, the 17 reductions/subtractions, counters, and sparse audit work before the inherited CUDA synchronization and `dt` update so all mechanism cost is charged.

### Milestone 2: Add fixed-scalar mechanism audits without new crash surface
- [x] At one-based step 1 and every 512th step, use only fixed FP64 device accumulators to record total/conv/classifier regularized-direction squared energy, removed row-mean energy, post-GC row-mean residual, decomposition error, and nonfinite status; delete all 17 transient means before allocation observations and never retain per-step losses or CUDA diagnostics.
- [x] Record exact GC calls and early-CutMix/early-clean/late-clean path counts, eligible/excluded inventory, audited row counts, and final parameter/momentum finiteness. Require `gc_calls == num_steps` and exact path/inventory reconciliation.
- [x] Preserve the inherited single live evaluation per epoch and `best_acc=max(test_acc)`. Append only already-computed evaluation accuracies to report the final-16 mean/range/final/best premium; evaluation code, targets, cadence, and selection cannot change.
- [x] Interpret removed/regularized energy as a mechanism diagnostic only: `<=1%` suggests BN redundancy, `1-5%` moderate action, and `>=5%` substantial action. No audit value may alter the run, eligibility, implementation, or verdict.

### Milestone 3: Pass exact math and the EXP018-structured decisive preflight
- [x] Pass syntax/scope checks and a deterministic CPU/GPU smoke proving the 17-tensor inventory, official `data gradient + L2 -> GC -> momentum/Nesterov` ordering, dimensions, reconstruction/orthogonality, post-row-mean residual <=1e-6, FP64 decomposition error <=1e-5, heterogeneous foreach-sub equivalence, RNG neutrality, and excluded update/momentum parity over two steps. Also simulate a seven-step partial final epoch and require exactly seven GC calls, one completed-step increment per call, and exact path/audit reconciliation.
- [x] Reuse EXP018's harness structure literally: import candidate helpers through the existing `if __name__ == "__main__": main()` guard, immediately replace the import-constructed evaluator with a guard before any iteration, mechanically load the parent, align real CIFAR clean/CutMix inputs and RNG, create every persistent optimizer/GC/audit scalar before the allocation baseline, reduce finiteness into one fixed device scalar, and release every loss/mean/temporary. Importing may construct the frozen test dataset/loader but must never call `main`, iterate it, or compute accuracy.
- [x] Run five alternating-order paired rounds with 80 CutMix, 80 early-clean, and 55 late-clean production-faithful steps per arm per round. Both arms time the identical region from inherited `t0` through CUDA synchronization; the candidate additionally imports and calls the exact production GC helper. Require parent drift <=4%, ratio MAD/median <=1.5%, median weighted candidate/parent latency <=1.03, every round ratio <=1.06, projected steps >=27,000, projected epochs >=138, exact counters, zero evaluator calls, and final live allocation no more than 1 MiB above the post-32-step baseline. Projected total runtime remains a reported sanity estimate; the hard 600-second metric timeout is authoritative.
- [x] Treat the first complete numeric preflight as decisive. Permit at most one documented repair only if the temporary harness throws or is malformed before emitting a complete JSON vector; no repair, alternate implementation, or rerun follows a complete numeric failure.

### Milestone 4: Run exactly one fixed-seed metric experiment and verify it
- [x] Remove stale `run.log`, reconfirm physical GPU 0 and one-device visibility, and launch `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp019-pycache PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1` exactly once after all accuracy-blind gates pass.
- [x] Monitor only liveness and integrity abort criteria; finite intermediate loss or accuracy never changes execution. Transcribe the complete inherited summary, GC inventory/path/energy diagnostics, CutMix dose, achieved steps/epochs, and final-16 evaluations into `03-execute.md`; preserve `run.log` unchanged until an adversarial result audit compares the raw values with the transcript, then delete it before the next experiment.
- [x] Require exit 0, 299.5-301.0 charged seconds, total runtime below 600 seconds, one evaluation per epoch, unchanged 2,748,890 model parameters, exact `gc_calls == num_steps`, positive finite removed energy, residual/decomposition limits, zero integrity failures, and only tracked `train.py` changed.
- [x] Classify `best_test_acc >=95.33%` as formal local improvement over EXP002's 95.23%. Report 95.53 as stronger but noise-limited context, 95.61 as global-best match, and 95.71 as resolution-clearing global improvement; a complete valid result below 95.33% definitively rejects this exact GC composition.

## Code Changes
- **`train.py` only / inventory**: derive eligible weights from named `Conv2d`/`Linear` modules, assert 16 convolution plus one classifier tensor, 17 tensors, 2,745,264 elements, and 2,266 rows, and reconcile all 2,748,890 parameters as 2,745,264 eligible plus 3,626 excluded elements. These counts were measured on this exact `a36dc09` model by EXP017's passing full-model smoke; a mismatch therefore indicates wrong eligibility or changed parent code, not spreadsheet uncertainty.
- **`train.py` only / official-order transformation**: after the sole inherited backward, materialize coupled L2 on all 44 gradients with `torch._foreach_add_(all_gradients, all_parameters, alpha=WEIGHT_DECAY)`. For each eligible FP32 direction compute `mean(dim=tuple(range(1, ndim)), keepdim=True)`; pinned Torch 2.9.1 has no `_foreach_mean`, so 17 explicit reductions are required. Apply their heterogeneous broadcasts with one supported `torch._foreach_sub_`, then call inherited SGD with internal decay disabled. The helper used by `train.py` is imported directly by preflight.
- **`train.py` only / sparse fixed-scalar audits**: set `GC_AUDIT_EVERY=512`; audit step 1 and every 512th one-based step. Accumulate regularized/removed energy, residual, decomposition, and nonfiniteness into a fixed set of device scalars, split convolution/classifier energy, and delete transient means before leaving the charged step. Host counters classify calls by early CutMix, early clean, and late clean; there is no per-step `.item()` or synchronized diagnostic.
- **`train.py` only / summary and evaluation preservation**: append exact inventory/dose reconciliation, removed-energy ratios, decomposition/residual limits, final optimizer-state finiteness, and final-16 stability lines. Preserve every inherited final-summary key and the frozen evaluator call/max-selection path; diagnostics are report-only except preregistered integrity assertions.

## Configuration Changes
- `GC_AUDIT_EVERY`: absent -> `512` (sparse integrity/energy audit cadence, not a training hyperparameter).
- Eligible set: absent -> all and only 16 convolution weights plus the final classifier weight (17 tensors / 2,745,264 elements / 2,266 rows), matching the official vision mechanism while excluding BN affine and bias directions.
- Coupled L2 location: SGD internal `weight_decay=1e-4` -> mathematically equivalent external foreach addition with optimizer `weight_decay=0`, enabling the official `data gradient + L2 -> GC -> momentum/Nesterov` ordering without double decay.
- GC coefficient: none; the projection is exact and full-run. No setting is tuned from preflight or accuracy.
- Interpretation thresholds: 95.33% formal local pass, 95.53% stronger single-seed signal, 95.61% global-best match, and 95.71% resolution-clearing global lift. Only 95.33% controls the tree verdict.

## Execution Environment
- Method: local execution from the repository root. Deterministic smoke and paired preflight scripts/logs live only under `/tmp/exp019-*`; the sole metric command is `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp019-pycache PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- Resources: physical GPU 0 only, NVIDIA H20 with approximately 97,871 MiB; exactly one CUDA device visible to every GPU command. Expected memory remains near EXP002's 1,178.9 MiB; VRAM is informational unless OOM or unexplained persistent allocation growth occurs.
- Estimated runtime: correctness and decisive preflight under 5 minutes; sole metric run approximately 470 seconds total with exactly 300 charged training seconds and a hard 600-second outer timeout.
- Log output: `/tmp/exp019_preflight.log` for the complete accuracy-blind JSON vector and repository-local `run.log` for the one metric run. `run.log` must survive through exact durable transcription and the independent result audit, then be removed before the next experiment. Use `PYTHONPYCACHEPREFIX=/tmp/exp019-pycache` for every Python command because shared or repository-local bytecode paths may be unwritable or pollute scope evidence.
- Tool skill: none; local single-GPU execution.

## Abort Criteria
- Stop before metric launch for wrong physical GPU/UUID, more than one visible CUDA device, evaluator/test-loader iteration or accuracy access during preflight, tracked scope beyond `train.py`, syntax/diff failure, inventory/reconciliation mismatch, wrong reduction axes, reconstruction/orthogonality/foreach mismatch, decay duplication/omission, excluded update or momentum mismatch, RNG drift, nonfinite state, residual/decomposition failure, counter mismatch, or unexplained persistent allocation growth.
- The first complete numeric preflight is decisive. Abort this leaf if parent drift exceeds 4%, ratio MAD/median exceeds 1.5%, median weighted candidate/parent latency exceeds 1.03, any round ratio exceeds 1.06, projected steps are below 27,000, projected epochs below 138, final live allocation grows by more than 1 MiB from the post-32-step baseline, or any structural/integrity field fails. Projected total runtime and reserved/cache allocation are informational; the metric's hard 600-second timeout is authoritative.
- During the metric run terminate on traceback, CUDA/OOM/device error, explicit GC integrity assertion, nonfinite state, no progress for 90 seconds after startup, incomplete inventory/path/audit reconciliation, or the 600-second timeout. Never stop, adapt, narrow eligibility, or change implementation based on finite loss, accuracy, or removed-energy magnitude.
- A valid completed run with fewer than 27,000 steps or 138 epochs retains its formal accuracy verdict; those floors gate the accuracy-blind projection and remain dose context, not extra goal conditions. A complete run below 95.33% is no-improvement and cannot be retried or tuned within EXP019.

## Verification Protocol

### Verification Procedure

1. **Parent, scope, commands, and syntax** (30-second timeout):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.3/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 002
   git diff --name-only a36dc09
   git status --porcelain --untracked-files=all
   git diff --check
   env PYTHONPYCACHEPREFIX=/tmp/exp019-pycache uv run python -m py_compile train.py
   ```
   Require parent metric 95.23, only tracked `train.py` changed, no repository helper artifacts, and successful syntax/diff checks. `.tree-autoresearch/` is ignored local metadata.

2. **GPU identity and visibility** (10-second timeout):
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader
   env CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(torch.cuda.device_count(), p.name, p.total_memory)'
   ```
   Require physical index 0 `NVIDIA H20`, approximately 97,871 MiB, and exactly one visible CUDA device; durably record the physical UUID.

3. **Deterministic math/integration smoke** (120-second timeout):
   ```bash
   env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp019-pycache uv run python /tmp/exp019_gc_smoke.py
   ```
   Insert the fixed repository root into the temporary script's `sys.path`; verify import does not call `main`, then replace the constructed evaluator before any iteration. Require exact 44/17/27 tensor and 2,748,890/2,745,264/3,626-element reconciliation; correct 2-D/4-D axes; FP64/FP32 reconstruction and orthogonality; heterogeneous foreach/loop equivalence; post-row-mean residual <=1e-6; decomposition error <=1e-5; GC-only RNG neutrality; explicit two-step Nesterov agreement for empty/nonempty momentum; excluded parameter/momentum parity; and full-model FP32 gradients under BF16 autocast. Simulate a seven-step truncated epoch and require seven calls/steps with exact path and one-based audit counts. Confirm no GradScaler, compile, graph capture, extra dependency, evaluator iteration, or accuracy access.

4. **Decisive EXP018-structured accuracy-blind preflight** (300-second timeout):
   ```bash
   timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp019-pycache PYTHONUNBUFFERED=1 uv run python /tmp/exp019_preflight.py > /tmp/exp019_preflight.log 2>&1
   ```
   Mechanically load EXP002 parent and candidate under `/tmp`, replace evaluator access with guards before traces, and never iterate a test loader or compute accuracy. On shared real CIFAR clean/CutMix inputs with aligned RNG, prove identical forward/raw-backward state followed by exact reference-ordered eligible divergence and excluded update/momentum parity. Import the exact production GC helper into the candidate arm. Create fixed diagnostics before a 32-step allocation baseline, run 1,024 more candidate steps while reducing finiteness into one device scalar, release every loss/mean/temporary, and require `final_allocation - baseline_allocation <= 1,048,576` bytes; report reserved allocation separately.

   Run five alternating-order paired rounds of 80 CutMix, 80 early-clean, and 55 late-clean steps per arm. For each round compute `0.375 * cutmix_ratio + 0.375 * early_clean_ratio + 0.25 * late_clean_ratio`; let median `R` drive `projected_steps=floor(27950/R)` and `projected_epochs=ceil(projected_steps/195)`. Compute `projected_total_seconds=300+1.2+projected_epochs*((467.1-300-1.2)/144)` as informational context only. Emit all raw round times, ratios, formula inputs/outputs, counters, live/reserved allocation, and guard calls in one JSON vector. Require every decisive threshold above. One pre-vector harness repair is allowed; any complete numeric failure is final.

5. **Exactly one metric launch and integrity check** (600-second timeout):
   ```bash
   rm -f run.log
   timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp019-pycache PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
   ```
   Require exit 0, `training_seconds` in `[299.5,301.0]`, `total_seconds <600`, one evaluation per epoch, all inherited summary keys, `num_params=2,748,890`, unchanged CutMix/drop-path/LR/evaluation configuration, `gc_calls == num_steps`, exact inventory/path/audit reconciliation, finite positive removed energy, residual <=1e-6, decomposition error <=1e-5, zero final nonfinite tensors, and only tracked `train.py` changed. Record achieved 27,000-step/138-epoch dose context without overriding a valid accuracy verdict.

6. **Metric decision and durable context** (15-second timeout):
   ```bash
   grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:' run.log
   ```
   Re-query EXP002 and require 95.23. With integrity valid, `best_test_acc >=95.33%` is improvement and anything lower is no-improvement. Lead with the formal verdict; separately report whether 95.53, 95.61, and 95.71 were reached. Transcribe final-16 values/mean/range/final/premium, removed-energy fractions, step/epoch dose, and exact counters. Then run the adversarial result audit against both the transcript and still-present `run.log`; delete raw logs only after the auditor has compared them. None of these diagnostics can override the formal verdict.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: corresponding final summary lines in `run.log`.
- GC mechanism: eligible/excluded tensor/element/row inventory; call/path/audit counts; aggregate, convolution, and classifier removed/regularized energy ratios; maximum residual; decomposition error; final nonfinite count.
- Stability: final-16 evaluation values, mean, min/max range, final value, and best-minus-mean premium from appended report-only diagnostics.
- Exposure: achieved steps/epochs versus EXP002's 27,950/144, realized CutMix ratio, paired preflight ratios, and total runtime; interpretation context only after necessary conditions pass.
