# Plan EXP-020: Convolution-only official-order Gradient Centralization
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the single-variable eligibility ablation
- [x] Inventory all 44 trainable tensors after model construction. Select all and only the weights of the 16 `nn.Conv2d` modules: 16 tensors, 2,742,704 elements, and 2,256 output rows; reconcile 28 excluded tensors and 6,186 excluded elements, including the 2,560-element classifier weight.
- [x] After every inherited backward, require all gradients, add coupled `WEIGHT_DECAY=1e-4` to all 44 FP32 directions with `torch._foreach_add_`, subtract row means only from the 16 convolution directions, then call unchanged SGD momentum/Nesterov with optimizer-internal `weight_decay=0`.
- [x] Preserve every architecture, initialization, data/CutMix/drop-path RNG, LR, BF16/channels-last, evaluator, and max-selection behavior. Add no coefficient, phase gate, stochastic draw, extra forward/backward, compile path, or graph capture.
- [x] Place coupled decay, projection, counters, and sparse audit work before the inherited CUDA synchronization and `dt` update so every mechanism cost is charged.

### Milestone 2: Instrument the closure questions with report-only fixed scalars
- [x] At one-based step 1 and every 512th step, accumulate fixed FP64 device scalars for convolution regularized, removed, and centralized squared energy; post-projection row-mean residual; orthogonal decomposition error; and nonfiniteness. Split regularized/removed energy into CutMix, early-clean, and late-clean phase buckets.
- [x] On the same sparse audits only, save raw-gradient row means before external L2, derive the effective L2 row-mean contribution as `regularized_mean - raw_mean` after the exact FP32 foreach addition, and accumulate raw, L2, and twice-cross removed-energy terms. Report reconstruction error with the hybrid scale `max(abs(regularized), abs(raw)+abs(L2)+abs(cross), 1e-12)`. This diagnostic never changes the applied official-order update, and its failure is a reported interpretation anomaly rather than a metric-run failure.
- [x] Record exact GC calls and early-CutMix/early-clean/late-clean path counts, eligible/excluded inventory, audit counts per phase, and final parameter/momentum finiteness. Require exact reconciliation and no host synchronization inside charged steps beyond the inherited end-of-step synchronize.
- [x] Preserve one live evaluation per epoch and `best_acc=max(test_acc)`. Append only already-computed accuracies for final-16 mean/range/final/best premium. Diagnostics are report-only and cannot alter execution or verdict.

### Milestone 3: Pass deterministic math and one decisive accuracy-blind preflight
- [x] Pass syntax/scope checks and deterministic CPU/GPU smoke for the exact 16-convolution inventory, official `data gradient + L2 -> conv GC -> momentum/Nesterov` order, axes, reconstruction/orthogonality, heterogeneous foreach equivalence, RNG neutrality, and two-step classifier/BN/bias update and momentum parity with ordinary coupled-decay SGD. On identical initial state/gradients, require audit-enabled and audit-disabled calls to yield bitwise-identical applied gradients and optimizer state.
- [x] Prove sparse raw/L2/cross reconstruction, per-phase bucket routing, one-based audit cadence, truncated-epoch counters, post-row-mean residual `<=1e-6`, energy decomposition error `<=1e-5`, and zero evaluator/test-loader access.
- [x] Run a production-faithful parent/candidate preflight on shared real CIFAR batches with aligned model RNG: seven alternating-order 512-step rounds per arm, each using a preregistered deterministic 192 CutMix / 192 early-clean / 128 late-clean fixture schedule (the configured expected 0.375/0.375/0.25 shares) and one realistic cadence-512 audit. Require parent drift `<=5%`, ratio MAD/median `<=2.5%`, median weighted latency ratio `<=1.03`, 90th-percentile round ratio `<=1.06`, projected dose `>=27,000` steps and `>=138` epochs, and post-state live allocation growth `<=1 MiB`.
- [x] Validate harness formulas, arm pairing, fixture counts, source hashes, and JSON schema before inspecting timing ratios. One correction/rerun is permitted only for a demonstrated harness or environmental invalidity, with the original vector retained and the reason documented; no candidate-code change or threshold-driven rerun is allowed. A complete valid numeric failure is final.

### Milestone 4: Run exactly one fixed-seed metric experiment and verify it
- [x] Remove stale `run.log`, reconfirm physical GPU 0 and single-device visibility, then launch `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp020-pycache PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1` exactly once after all accuracy-blind gates pass.
- [x] Monitor only liveness and integrity. Never stop or alter the run based on finite loss, intermediate accuracy, or audit magnitude. Preserve `run.log` through exact transcription into `03-execute.md` and an independent raw-result audit.
- [x] Require exit 0, 299.5-301.0 charged seconds, total runtime below 600 seconds, one evaluation per epoch, unchanged 2,748,890 parameters, exact calls/path/audit/inventory reconciliation, positive finite removed energy, residual/decomposition limits, and zero nonfinite state.
- [x] Classify `best_test_acc >=95.33%` as formal improvement over EXP002's 95.23%. A valid result `<=95.23%` closes the literature-supported official-order full-eligible and convolution-only GC rules on this base; raw-order and phase-limited rules remain untested and merely become low priority absent new causal evidence. `95.24-95.32%` remains formal no-improvement but is reported as unresolved positive movement.

## Code Changes
- **`train.py` only / inventory and applied update**: port the already-validated EXP019 external coupled-decay and GC helper, but include only named `Conv2d.weight` tensors in the projected set. Assert 16/2,742,704/2,256 eligible and 28/6,186 excluded against the unchanged 44-tensor, 2,748,890-parameter model. The classifier still receives ordinary coupled L2 and momentum/Nesterov; it is excluded only from projection.
- **`train.py` only / sparse diagnostics**: narrow the existing cadence-512 energy audit to convolution GC, add fixed phase buckets, and add raw/L2/cross row-mean decomposition. Raw means are read-only and captured before the unchanged single foreach L2 addition; regularized means drive the applied projection. Construct persistent device scalars before timing/allocation baselines; delete transient row means; never retain losses, per-step tensors, or synchronize for diagnostics.
- **`train.py` only / summary and integrity**: append exact inventory, dose, phase, energy, raw/L2/cross reconstruction, residual, nonfinite, and final-16 lines. Preserve all inherited summary keys. Integrity assertions may fail the run only for preregistered structural/numerical violations, never for accuracy or effect size.

## Configuration Changes
- `GC_AUDIT_EVERY`: absent -> `512` (sparse report-only audit cadence).
- Projected eligibility: absent -> exactly 16 convolution weights / 2,742,704 elements / 2,256 rows; final `Linear.weight`, BN affine, and biases remain unprojected.
- Coupled L2 location: SGD internal `weight_decay=1e-4` -> equivalent external addition to all gradients with optimizer `weight_decay=0`, required to apply the paper's `data gradient + L2 -> GC` order without omitting classifier decay or double-decaying any tensor.
- GC dose/coefficient: exact projection on every completed training step in every phase; no tunable coefficient or phase gate.
- Decision thresholds: 95.33% formal local improvement; `<=95.23%` closes official-order GC on EXP002; 95.53% is stronger but single-seed/noise-limited context, 95.61% matches the global best, and 95.71% clears it by the formal resolution gate.

## Execution Environment
- Method: local execution from the repository root. Temporary smoke/preflight scripts and logs use `/tmp/exp020-*`; the sole metric command is `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp020-pycache PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- Resources: physical GPU 0 only, one visible NVIDIA H20 with approximately 97,871 MiB. Expected peak memory is near EXP002's 1,178.9 MiB; VRAM is informational unless OOM or unexplained persistent allocation growth occurs.
- Estimated runtime: correctness and decisive preflight under 5 minutes; the sole metric run approximately 450-480 seconds total, including exactly 300 charged training seconds, with a hard 600-second timeout.
- Log output: `/tmp/exp020_preflight.log` holds the complete accuracy-blind preflight vector; repository-local `run.log` is the metric source of truth. Keep `run.log` unchanged until durable transcription and independent comparison, then remove it before another experiment. Use `/tmp/exp020-pycache` to avoid shared bytecode permissions and repository pollution.
- Tool skill: none; local single-GPU execution.

## Abort Criteria
- Stop before the metric run for wrong GPU/visibility; tracked changes beyond `train.py`; syntax/diff failure; inventory mismatch; evaluator/test-loader iteration or accuracy access in preflight; wrong axes/order; decay omission/duplication; classifier/BN/bias parity failure; RNG drift; nonfinite state; counter, phase, audit, raw/L2/cross, residual, or orthogonal-energy reconciliation failure.
- Abort the leaf on the first structurally valid preflight vector if parent drift exceeds 5%, ratio MAD/median exceeds 2.5%, median weighted candidate/parent latency exceeds 1.03, the 90th-percentile round ratio exceeds 1.06, projected steps fall below 27,000, projected epochs below 138, live allocation grows by more than 1,048,576 bytes after persistent-state baseline, evaluator guard count is nonzero, or any structural field fails. Reserved/cache allocation and projected total runtime are informational. A rerun is allowed only for independently demonstrated harness/environment invalidity and must retain the original vector.
- During the metric run terminate on traceback, CUDA/OOM/device error, explicit integrity failure, nonfinite state, no progress for 90 seconds after startup, or the 600-second timeout. Finite loss/accuracy/audit magnitudes never trigger termination or adaptation.
- A valid metric run below 95.33% is no-improvement and cannot be retried or tuned within EXP020. Realized dose below preflight projections remains context and cannot override the accuracy verdict if the fixed time budget and all goal constraints were respected.

## Verification Protocol

### Verification Procedure

1. **Parent, scope, command, and syntax checks** (30-second timeout):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.3/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 002
   git diff --name-only a36dc09
   git status --porcelain --untracked-files=all
   git diff --check
   env PYTHONPYCACHEPREFIX=/tmp/exp020-pycache uv run python -m py_compile train.py
   ```
   Require parent metric 95.23, only tracked `train.py` changed, no repository helper artifacts, and successful syntax/diff checks. `.tree-autoresearch/` is ignored loop metadata.

2. **GPU identity and visibility** (10-second timeout):
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader
   env CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(torch.cuda.device_count(), p.name, p.total_memory)'
   ```
   Require physical index 0 `NVIDIA H20`, approximately 97,871 MiB, and exactly one visible CUDA device. Record its physical UUID.

3. **Deterministic math/integration smoke** (120-second timeout):
   ```bash
   env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp020-pycache uv run python /tmp/exp020_gc_smoke.py
   ```
   Import the exact production helpers through the existing `if __name__ == "__main__"` guard (already present at EXP002 commit `a36dc09`), replace the import-constructed evaluator before any iteration, and forbid accuracy/test-loader access. Require exact 44/16/28 tensor and 2,748,890/2,742,704/6,186-element inventory; correct 4-D axes; FP64/FP32 reconstruction and orthogonality; heterogeneous foreach/loop equivalence; maximum absolute post-projection row mean `<=1e-6`; applied-energy decomposition error `<=1e-5`; RNG neutrality; two-step Nesterov agreement for empty and populated momentum; classifier/BN/bias update/momentum parity; and bitwise audit-on/off applied-update parity. Report the raw/L2/cross hybrid-scaled error, but do not fail the metric mechanism on that report-only decomposition. Simulate calls across all three phase buckets and a seven-step truncated epoch; require exact call/step/audit reconciliation and no extra dependency, forward/backward, evaluation, compile, or graph capture.

4. **Decisive accuracy-blind preflight** (300-second timeout):
   ```bash
   timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp020-pycache PYTHONUNBUFFERED=1 uv run python /tmp/exp020_preflight.py > /tmp/exp020_preflight.log 2>&1
   ```
   Materialize the parent as the byte-exact output of `git show a36dc09:train.py` under `/tmp`, record its SHA-256, and load it without restructuring through its existing main guard; load the candidate similarly and record its source hash. Guard both evaluators before traces and never iterate a test loader or compute accuracy. On shared real CIFAR clean/CutMix fixtures, restore identical model/drop-path RNG state per arm and prove identical forward/raw-backward state followed by exact official-order convolution divergence and excluded classifier/BN/bias update/momentum parity. Fixture path selection is deterministic and does not claim parity with the parent's stochastic CutMix draw stream; within each chosen path, both arms consume identical model/data randomness. Import the exact production GC helper into the candidate arm. Create all persistent diagnostics before a 32-step allocation baseline, run 1,024 more candidate steps while reducing finiteness into one fixed scalar, release every loss/mean/temporary, and require final live allocation growth `<=1,048,576` bytes; report reserved allocation separately.

   Run seven alternating-order paired rounds of 512 steps per arm: 192 deterministic CutMix, 192 deterministic early-clean, and 128 late-clean fixtures, derived from the configured 75% early phase times 0.5 CutMix gate and 25% late phase. Use one-based step IDs 257..768 so exactly one production-cadence audit occurs per 512 steps. Compute per-path ratios and the equivalent `0.375/0.375/0.25` weighted ratio; use median `R` for `floor(27950/R)` projected steps and `ceil(projected_steps/195)` epochs. Emit raw times, round ratios, dispersion, percentile, formulas, fixture/audit counts, source hashes, decompositions, allocation, and evaluator guard calls in one JSON vector. Validate schema/formulas before evaluating thresholds. Any corrected vector must preserve the original plus a documented proof of harness/environment invalidity; valid numeric failure is final.

5. **Exactly one metric launch and integrity check** (600-second timeout):
   ```bash
   rm -f run.log
   timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp020-pycache PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
   ```
   Require exit 0, `training_seconds` in `[299.5,301.0]`, `total_seconds <600`, exactly one evaluation per completed epoch, all inherited summary keys, `num_params=2,748,890`, unchanged CutMix/drop-path/LR/evaluation configuration, `gc_calls == num_steps`, exact inventory/path/audit/phase reconciliation, positive finite removed energy, applied orthogonal-energy decomposition error `<=1e-5`, maximum absolute post-projection row mean `<=1e-6`, zero nonfinite counts, and only tracked `train.py` changed. Raw/L2/cross reconstruction outside its hybrid tolerance is transcribed as a diagnostic anomaly and disables inference from that audit; it does not invalidate the metric run.

6. **Metric decision, durable context, and evidence cleanup** (15-second parsing timeout):
   ```bash
   grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:' run.log
   ```
   Re-query EXP002 and require 95.23. With integrity valid, `best_test_acc >=95.33%` is improvement and anything lower is no-improvement. Separately report official-order closure (`<=95.23%`) and context thresholds 95.53/95.61/95.71. Transcribe the full final summary, final-16 values, energy/decomposition values by phase, exact dose, and counters into `03-execute.md`. Run the independent result audit against both the transcript and still-present raw log; only then delete `run.log` and `/tmp/exp020-*` artifacts. Diagnostics cannot override the formal verdict or claim causal closure of untested order/phase variants.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: corresponding final summary lines in `run.log`.
- GC mechanism: eligible/excluded inventory; calls and phase counts; aggregate and phase-bucket regularized/removed energy; raw/L2/cross removed-energy decomposition; residual and orthogonal decomposition errors; final nonfinite count.
- Stability: final-16 evaluation values, mean, range, final value, and best-minus-mean premium.
- Exposure: achieved steps/epochs versus EXP002's 27,950/144, CutMix ratio, paired preflight ratios, and total runtime; interpretation context only.
