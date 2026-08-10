# Plan EXP-017: Full-Run Eligible-Weight Gradient Centralization
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement reference-ordered GC and deterministic checks
- [x] Modify only `train.py` with a fixed 17-tensor eligible inventory, externally materialized coupled L2, exact per-output row means, one `torch._foreach_sub_`, sparse FP64 audits, and final-16 evaluation context.
- [x] Require `PYTHONPYCACHEPREFIX=/tmp/exp017-pycache uv run python -m py_compile train.py`, `git diff --check`, and pre-launch scope checks to pass.
- [x] Run CPU/GPU mathematical smokes for axes, reconstruction, orthogonality, foreach/loop bitwise parity, excluded tensors, RNG neutrality, and `decay -> centralize -> momentum -> Nesterov` ordering.

### Milestone 2: Pass one decisive accuracy-blind GPU-0 preflight
- [x] Verify physical GPU 0 and its UUID, expose only `CUDA_VISIBLE_DEVICES=0`, guard the evaluator, and assert zero test-loader iterations/accuracy values.
- [ ] Run full-model clean/CutMix integration checks, a 1,024-step production-order candidate trace, and five alternating-order paired latency rounds using exact EXP-002 parent code.
- [ ] Proceed only if every structural/integrity gate and the first complete numeric latency/dose vector pass without changing eligibility, implementation, or thresholds.

### Milestone 3: Run exactly one fixed-seed metric experiment
- [ ] Remove stale `run.log`, reconfirm GPU 0, then run `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1` exactly once.
- [ ] Monitor only liveness/integrity abort criteria; intermediate accuracy never changes execution.
- [ ] Transcribe the complete summary, final-16 evaluations, GC/CutMix dose, energy diagnostics, GPU identity, and scope evidence into `03-execute.md` before cleanup.

### Milestone 4: Verify and hand off
- [ ] Require all frozen-protocol integrity conditions and classify `best_test_acc >=95.33%` as local improvement over EXP-002.
- [ ] Lead with the formal verdict; report 95.53 as noise-limited context, 95.61 as global-best match, and 95.71 as resolution-clearing global improvement separately.
- [ ] Preserve raw evidence through Claude-only result review, then remove `run.log` and EXP017-owned `/tmp` artifacts.

## Code Changes
- **`train.py` only / inventory**: build the eligible list after model construction by selecting exactly each `nn.Conv2d.weight` and `nn.Linear.weight` from named modules. Assert 16 convolution weights plus one classifier weight, 17 tensors, 2,745,264 elements, and 2,266 output rows. Reconcile the 2,748,890 total parameters as 2,745,264 eligible plus 3,626 excluded BN-affine/bias elements.
- **`train.py` only / reference-ordered transformation**: after the sole inherited backward, require every parameter gradient and use one `torch._foreach_add_(all_gradients, all_parameters, alpha=WEIGHT_DECAY)` to materialize the parent's coupled L2 direction. For the 17 eligible FP32 directions, compute each mean over dimensions `1..ndim-1`, then apply all broadcasts with one `torch._foreach_sub_(eligible_directions, row_means)`. Run PyTorch SGD with internal `weight_decay=0` but unchanged parameter order, learning rate, momentum, dampening, Nesterov, and state initialization. This exactly implements the official `data gradient + L2 -> eligible GC -> momentum/Nesterov` ordering without double decay. BN affine/bias directions receive identical effective decay but no GC. No phase gate, coefficient, clipping, casting, optimizer subclass, extra forward, or persistent optimizer state is introduced.
- **`train.py` only / audits**: set `GC_AUDIT_EVERY=512`; audit step 1 and every 512th one-based step. Reuse pre-subtraction means and accumulate regularized, removed, and centralized squared energy plus residual/nonfinite data in GPU FP64 scalars inside the charged timer, read only after charged training. Split convolution and classifier energy. Restrict numeric nonfinite checks to sparse audit steps and final state; add no per-step `.item()` or host synchronization. Host counters classify every call as early CutMix, early clean, or late clean.
- **`train.py` only / summary and evaluation preservation**: print inventory/dose reconciliation, total and split removed/regularized ratios, FP64 decomposition error, maximum post-row-mean residual, zero audited nonfinite count, and final-16 evaluation mean/range/final/best premium. Evaluation transform, batches, cadence, evaluator calls, and `best_test_acc=max(per_epoch_test_acc)` remain byte-for-byte inherited; only already-computed per-epoch accuracies are appended to an in-memory final-16 diagnostic. All diagnostics are report-only except fixed integrity checks.

## Configuration Changes
- `GC_AUDIT_EVERY`: absent -> `512` (sparse fixed audit cadence that does not add host synchronization inside charged training).
- Eligible set: absent -> all and only 16 `Conv2d.weight` plus final `Linear.weight` tensors (ECCV standard per-output centralization; coefficient-free).
- Coupled L2 location: PyTorch SGD internal `weight_decay=1e-4` -> one equivalent pre-step foreach add with optimizer-internal `weight_decay=0`, so the official GC ordering centralizes the regularized eligible direction while excluded tensors retain the parent update.
- Subtraction implementation: absent -> one broadcast-capable `torch._foreach_sub_` across the 17 gradient/mean pairs (locally confirmed to support heterogeneous shapes; preflight must prove bitwise equivalence to loop `sub_`).
- Interpretation: aggregate removed/regularized `<=1%` supports BN redundancy, `1-5%` is ambiguous moderate intervention, and `>=5%` means substantial regularized-direction energy was removed. Classifier energy is reported separately. These readings never alter the run or formal verdict.

## Execution Environment
- Method: local execution from the repository root. Temporary smokes and paired harnesses live only under `/tmp/exp017-*`.
- Resources: physical GPU 0 only, NVIDIA H20 with approximately 97,871 MiB physical memory; expected model memory remains near EXP-002's 1,178.9 MiB. VRAM is informational and only OOM/insufficient headroom aborts.
- Estimated runtime: correctness plus preflight below 5 minutes; sole metric run approximately 470 seconds total with exactly 300 charged seconds and a hard 600-second outer limit.
- Log output: `/tmp/exp017_preflight.log` for the accuracy-blind gate and repository-local `run.log` for the sole metric run until durable transcription and Claude review. Set `PYTHONPYCACHEPREFIX=/tmp/exp017-pycache` for all transient scripts.
- Tool skill: none; local GPU execution.

## Abort Criteria
- Stop before metric launch for wrong physical GPU/UUID, more than one visible CUDA device, any evaluator/test-loader access, tracked scope beyond `train.py`, arithmetic/foreach/reference mismatch, incorrect inventory/axes, excluded update mismatch, RNG drift, decay/GC/Nesterov-order mismatch, nonfinite state, or counter failure.
- The first complete numeric preflight is decisive. Abort this leaf if parent drift exceeds 4%, paired-ratio MAD/median exceeds 1.5%, median candidate/parent charged latency exceeds 1.03, any ratio exceeds 1.06, projected steps fall below 27,000, projected epochs below 138, or projected total reaches 600 seconds. Peak allocation is reported but is not a numeric failure gate.
- During the metric run terminate on traceback, CUDA/OOM/device error, sparse-audit/final-state nonfinite values, no progress for 90 seconds after startup, incomplete reconciliation, or 600-second timeout. Do not add per-step synchronized checks and do not abort or tune on finite loss or accuracy.
- A valid completed run with fewer than 27,000 steps or 138 epochs retains its formal accuracy verdict; those floors are scientific dose context, not extra goal conditions. Incomplete GC dose or failed audit is an integrity failure, never permission for a metric retry.

## Verification Protocol

### Verification Procedure

1. **Parent, scope, commands, and syntax** (30-second timeout):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.2/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 002
   git diff --name-only a36dc09
   git status --porcelain --untracked-files=all
   git diff --check
   env PYTHONPYCACHEPREFIX=/tmp/exp017-pycache uv run python -m py_compile train.py
   ```
   Require parent metric 95.23, only tracked `train.py` changed, no repository helper files, and successful syntax/diff checks. Run before creating `run.log`; `.tree-autoresearch/` is ignored metadata.

2. **GPU identity** (10-second timeout):
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader
   env CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(torch.cuda.device_count(), p.name, p.total_memory)'
   ```
   Require physical index 0 `NVIDIA H20`, physical UUID `GPU-b1bc897d-2183-dad2-8302-8800bc02a633`, approximately 98 GB, and exactly one visible CUDA device.

3. **Deterministic mathematical/integration smoke** (90-second timeout):
   ```bash
   env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp017-pycache uv run python /tmp/exp017_gc_smoke.py
   ```
   Require exact inventory and 3,626-element excluded reconciliation, correct 2-D/4-D axes, FP64/FP32 reconstruction and orthogonality on the regularized direction, bitwise foreach/loop subtraction equivalence, post-row-mean residual at most `1e-6`, FP64 decomposition error at most `1e-5`, GC-only RNG neutrality, explicit two-step Nesterov agreement for empty/nonempty momentum, and excluded parameter/buffer parity between external and parent-internal decay. Confirm parent FP32 grads with BF16 autocast and absence of GradScaler, compile, and graph capture.

4. **Decisive accuracy-blind paired preflight** (300-second timeout):
   ```bash
   timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp017-pycache PYTHONUNBUFFERED=1 uv run python /tmp/exp017_preflight.py > /tmp/exp017_preflight.log 2>&1
   ```
   Extract exact EXP-002 parent code mechanically under `/tmp`; guard evaluators before traces. Verify parent/candidate parity through backward on shared real CIFAR clean/CutMix batches, then reference-ordered eligible divergence plus excluded-update parity. Profile and microbenchmark 1,000 isolated foreach-versus-loop subtractions to report dispatch/kernel/latency evidence; this is diagnostic only and cannot trigger an implementation fallback. Run 1,024 candidate production-order steps and five alternating-order paired rounds of 44 early CutMix, 45 early clean, and 31 late clean steps. Emit one complete JSON vector and require every abort gate above, zero test access, exact counter identities, finite positive removed energy, no post-warmup allocation growth beyond expected optimizer state, and reported memory. An exception/malformed/timeout before numeric output permits one recorded harness repair; a complete numeric failure cannot be rerun.

5. **Exactly one metric launch** (600-second timeout):
   ```bash
   rm -f run.log
   timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
   ```
   Require exit 0, `training_seconds` in `[299.5,301.0]`, `total_seconds <600`, unchanged one evaluation/evaluator call per epoch and max-selection semantics, `num_params=2,748,890`, unchanged CutMix/drop-path settings, `gc_calls == num_steps`, exact path-count reconciliation, positive finite removed energy, FP64 decomposition/residual limits, and zero integrity failures. Record whether `num_steps >=27000` and `num_epochs >=138` as mechanism-dose context, without overriding the formal goal verdict.

6. **Metric decision and context** (10-second timeout):
   ```bash
   grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:' run.log
   ```
   Re-query EXP-002 and require 95.23. With all frozen-protocol integrity conditions, `best_test_acc >=95.33%` is local improvement; below 95.33 is no-improvement. Lead with this formal verdict. Treat `>=95.53%` only as a noise-limited single-seed signal that may motivate a GC-on-EXP-011 follow-up, and separately report whether 95.61 and 95.71 are reached. Final accuracy at least 95.29, CE loss at most 0.2044, step/epoch dose, and final-16 context support interpretation but do not override the formal tree verdict. Never rerun, narrow eligibility, or change implementation after accuracy.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: corresponding final summary lines in `run.log`.
- GC mechanism: terminal audit values for aggregate/conv/classifier removed/regularized energy, decomposition error, residual, path counts, and sparse-audit nonfinite count.
- Stability: final-16 evaluation values, mean, range, final value, and best-minus-mean premium.
