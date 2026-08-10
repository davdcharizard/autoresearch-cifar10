# Plan EXP-009: Late-Stage Identity-Centered SE with Isolated Initialization
- **Created**: 2026-08-05

## Decision Lock

- Parent EXP-002 is `a36dc09` at 95.23%; formal acceptance is 95.33% and the separate mechanism target is 95.53%.
- Add SE only to blocks 3-6 (128/256 channels), with hidden width 16, per-sample channel-standardized descriptors, `2*sigmoid` gates, dedicated seed-42 initialization, and fixed gate LR scale 5.0 / weight decay 0.
- The four modules add exactly 25,408 parameters, for 2,774,298 total. No placement, hidden width, LR scale, seed, activation, or normalization change is allowed after preflight.

## Milestones

### Milestone 1: Implement RNG-isolated identity-centered SE
- [x] Add a raw-parameter SE module with `weight1 [16,C]`, `bias1 [16]`, zero `weight2 [C,16]`, and zero `bias2 [C]`; initialize only `weight1` from a dedicated CPU generator without consuming global RNG.
- [x] Pool each `conv2` output and, inside an explicit autocast-disabled region, standardize each sample with biased channel variance and `rsqrt(var + 1e-5)`, compute FP32 affine/ReLU/affine excitation, and multiply the same `conv2` tensor by an effective dtype-cast `2*sigmoid` gate before drop path.
- [x] Instantiate modules only for the four 128/256-channel blocks and keep the two 64-channel blocks exact.
- [x] Split optimizer parameters exactly into parent and gate groups; gate parameters use `lr_scale=5.0`, zero weight decay, and otherwise the same SGD/momentum/Nesterov settings. The LR loop multiplies only by the stored scale.
- [x] Add startup inventory and a final aggregate `se:` line before the `---` delimiter: module/tensor/element counts, settings, per-module parameter norms, training-only mean/fraction/max gate deviations, saturation fractions, and device-accumulated nonfinite/inventory failures. No per-step host sync or print is allowed.

### Milestone 2: Prove initialization, gradients, and parent preservation
- [x] Pass `py_compile`, Ruff, `git diff --check`, and scope inspection.
- [x] Construct actual parent and child from restored CPU/CUDA RNG states; require every shared tensor, initial BF16/FP32 forward, post-construction global RNG, CutMix generator state/first draws, and drop-path RNG advancement bitwise equal. Require zero final SE layers after `model.apply`, 2,774,298 parameters, 25,408 gate elements, and four modules.
- [ ] Run FP64 closed-form tests for descriptor standardization, affine layers, identity gate, and gradients. Require constant descriptors remain finite, zero final layer gives exact gate one, `weight2` receives a nonzero first gradient, and `weight1` receives a nonzero gradient after one update.
- [x] On GPU 0, verify BF16/channels-last forward/backward plus FP32 descriptor/logit dtypes inside outer autocast, CutMix parity, unchanged drop-path RNG, finite gradients, exact group ownership, and no parent parameter receives gate settings. Run an FP32 `eval()`/`inference_mode()` forward and require no training activity counters mutate.

### Milestone 3: Pass live-gate and parent-relative latency gates
- [x] Confirm physical GPU 0 is the 97,871 MiB H20, branch/base/scope are exact, no stale log exists, and record co-tenancy.
- [x] In a separate fixed-seed GPU-0 process, train exactly 200 production-faithful steps without evaluation. For every module require mean effective BF16 `|gate-1| >=0.002`, at least 1% of gates beyond 0.015625, saturation fraction (`gate<0.1` or `gate>1.9`) below 1%, and finite nonzero final matrices. Never alter a locked value to pass.
- [x] Benchmark matched parent/candidate batch-256 BF16/channels-last production loops after warmup, at least 300 synchronized iterations each in separate processes. Project only as `27,950 / median_ratio`; require ratio at most 1.075, at least 26,000 steps, and at least 134 epoch evaluations.

### Milestone 4: Run and verify once
- [ ] Launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`; monitor health only.
- [ ] Parse SE/CutMix exposure, gate activity, timing, evaluation count, and summary integrity before reading accuracy.
- [ ] Compare the single metric with 95.33% formal and 95.53% mechanism thresholds; preserve evidence for Claude result review and remove `run.log` only after analysis.

## Code Changes

- **`train.py` only**:
  - Add fixed SE constants: hidden width 16, init seed 42, LR scale 5.0, epsilon `1e-5`, and minimum channel threshold 128.
  - Implement `IdentitySqueezeExcitation` entirely from raw `nn.Parameter` tensors so module construction consumes no RNG. Pass one dedicated CPU generator through model/block construction and call Kaiming initialization with its `generator=` argument only for first-layer weights. Biases and final layer are zero tensors.
  - In `forward`, enter `torch.autocast(device_type=x.device.type, enabled=False)`, cast pooled `conv2` output to FP32, compute biased variance `mean((d-mean)^2)`, standardize with `rsqrt(var+1e-5)`, then apply FP32 `F.linear`/ReLU/`F.linear`. Form `2*sigmoid`, cast to activation dtype, and gate that same residual tensor. Assert production intermediate dtypes in smoke tests.
  - Under training and outside inference mode only, update detached GPU buffers for count, absolute-deviation sum, threshold count, maximum, saturation count, and nonfinite count. Never synchronize or branch on those buffers in `forward`; inspect them after explicit synchronization.
  - Build explicit name/id inventories. Parent group contains every non-SE trainable tensor exactly once with parent weight decay; gate group contains all 16 SE tensors exactly once with zero decay and 5x LR. Update group LR as `lr * lr_scale`.
  - Preserve DataLoader/transforms, model widths/depth, CutMix helpers/generators/dose, drop path, base LR schedule, evaluator, timer boundaries, and summary keys.

## Configuration Changes

- Architecture: EXP-002 WRN-16-4 -> same WRN plus four late-stage identity-centered SE gates.
- Trainable parameters: 2,748,890 -> 2,774,298 (+25,408, +0.924%).
- Gate hidden width / epsilon: 16 / `1e-5`.
- Gate optimizer: parent LR x5, weight decay 0; all parent parameters retain LR x1 and weight decay `1e-4`.
- Data, CutMix, drop path, global seed, evaluation: unchanged.

## Execution Environment

- Method: local process from repository root.
- Resources: physical GPU 0 only via `CUDA_VISIBLE_DEVICES=0`; NVIDIA H20 97,871 MiB; existing eight workers.
- Estimated runtime: 300 charged seconds, 460-540 total; hard timeout 600 seconds.
- Log output: repository-root `run.log`, removed only after analysis evidence is durable.
- Tool skill: none.

## Abort Criteria

- Before launch, abort on wrong branch/base/scope/GPU identity, changed co-tenancy, stale log, static failure, shared initialization/initial-output/RNG/CutMix mismatch, wrong inventory/groups, FP32-path violation, eval-mode failure, nonfinite math, any live-gate floor/ceiling failure, candidate latency ratio above 1.075, projected steps below 26,000, or projected evaluations below 134.
- During the run, stop on traceback, CUDA/OOM, nonfinite loss/gate/gradient, inventory failure, missing output for 60 seconds, or unavoidable 600-second timeout. Never abort for intermediate accuracy.
- One deterministic repair is allowed only before evaluation for an import/shape/traceback/CUDA/OOM defect. It may not change hidden width, seed, LR scale, epsilon, channel threshold, block placement, gate form, or optimizer grouping. No metric-driven retry is allowed.

## Verification Protocol

### Verification Procedure

1. **Parent, scope, and hardware** (20s): query node 002 and require 95.23 / `a36dc09`; require EXP-009 branch and only `train.py` changed; query physical GPU 0 and require H20 97,871 MiB, recording co-tenants.
2. **Static and inventory** (60s): run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, and `git diff --check`; require four modules, 16 gate tensors, 25,408 gate elements, and 2,774,298 total parameters.
3. **Initialization and geometry** (120s): materialize `git show a36dc09:train.py` and harnesses only in temporary paths outside the repository. Compare construction, shared tensors, CutMix state/draws, initial FP32/BF16 outputs, and global states bitwise. Verify post-apply zero layers and independent FP64 geometry/gradients, including exact biased-variance formula and injected nonfinite inputs.
4. **GPU integration/eval** (180s): expose physical GPU 0 and compare its UUID/PCI identity with `nvidia-smi -i 0`. Run actual-parent/candidate clean and fixed CutMix paths inside outer autocast, require FP32 SE intermediates and exact parent parity at identity, then train one step and verify expected gate-only divergence/group updates. Run `eval()` under `inference_mode()` and require no counter mutation.
5. **Live gate** (180s): run 200 fixed-seed production steps without evaluation; for each module require mean absolute deviation `>=0.002`, fraction beyond 0.015625 `>=0.01`, saturation fraction `<0.01`, finite device counters, and nonzero final matrices.
6. **Latency** (180s): with stable recorded co-tenancy, benchmark matched production parent/candidate loops in separate synchronized GPU-0 processes for at least 300 iterations after 50 warmups. Report median/p90/dispersion and project `27,950/median_ratio`; require ratio `<=1.075`, steps `>=26,000`, and evaluations `>=134`. Recheck GPU/co-tenancy immediately before launch.
7. **Single run** (610s): execute exactly `timeout 600s env CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` once, with health-only monitoring.
8. **Protocol before metric** (30s): require fixed config, four active modules, correct groups, finite training-only gate stats, CutMix ratio `[0.49,0.51]`, 299.5-301.0 charged seconds, total below 600, exact parameters, exactly one `best_test_acc:` and `peak_vram_mb:`, and one evaluation per epoch. Record step/evaluation deltas beside every causal conclusion even though a completed run remains formally valid.
9. **Verdict** (10s): parse exactly one `best_test_acc`. `>=95.33` passes the tree condition; `>=95.53` supports the mechanism target. For 95.33-95.52 record exactly: "Formal improvement, but below the preregistered +0.30 evidentiary bar and within observed single-run variation." Do not upgrade accuracy using diagnostics.
10. **Evidence/cleanup** (10s): write raw values to execution/analysis artifacts, obtain Claude adversarial result review, then remove `run.log` and require no transient artifacts.

### Informational Metrics (Optional)

- Final accuracy/loss versus parent 95.19 / 0.2044.
- Training/total/startup seconds, VRAM, epochs/evals, steps, and parameters versus parent 300.0 / 467.1 / 1.2 / 1,178.9 / 144 / 144 / 27,950 / 2,748,890.
- Per-module parameter norms and effective gate deviations; diagnostic only.
