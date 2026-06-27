# Plan EXP-054: fp16 autocast + GradScaler — the never-probed finer-precision direction of the numerics axis
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-054.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): mean ≈ 96.57, σ ≈ 0.16. The single variable: autocast dtype bf16 → fp16 (mantissa step 2^-8 → 2^-11, 8× finer) with `torch.amp.GradScaler` managing fp16's narrower dynamic range. EXP-021 (replicated) proves this axis moves the metric at the 0.2–0.4pp scale; only the coarser direction has been measured.

Projection: params **4,286,026** (no model change); dt 22.3–22.8ms (fp16 tensor-core throughput = bf16 on H20; scaler unscale+inf-check adds ≤ ~0.3ms worst case → ≤ −2 epochs ≈ −0.04 by the deferral law); epochs 133–141; steps ledger informational ~12,950–13,520 (scaler-skipped steps still increment the counter). VRAM ~1.6GB.

**Failed-Approaches check**: EXP-021 ("step-time engineering via max-autotune + cudagraphs + fused SGD", count 1) is the nearest entry — distinguished explicitly: that probe changed kernel selection/reduction orders chasing SPEED; this probe holds kernels/speed fixed and changes PRECISION in the finer direction, the half of the axis EXP-021 never measured. Numerics judged by the EXP-048 trajectory criterion. Gradient-noise law note: if bf16 rounding noise is part of the tuned noise budget, this reads null-to-negative — that outcome CLOSES the axis and is a pre-registered branch, not a surprise.

## Milestones

### Milestone 1: Code change implemented and passing CPU sanity
- [x] `scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")` created after the optimizer
- [x] Warmup loop: `torch.autocast("cuda", dtype=torch.float16)`; backward becomes `scaler.scale(warm_loss).backward()` (no step/update in warmup — grads zeroed after, scale untouched at init 2^16)
- [x] Timed loop: `torch.autocast("cuda", dtype=torch.float16)`; `scaler.scale(loss).backward()` → `scaler.step(optimizer)` → `scaler.update()` (replacing `loss.backward()` / `optimizer.step()`); everything else byte-identical (loss.item() reads the UNSCALED loss — fp32 under autocast)
- [x] Diff check: `git diff --stat` shows 1 file (train.py), ~6 lines across 3 hunks (scaler init, warmup, timed loop)
- [x] CPU sanity `/tmp/exp054_sanity.py` (CUDA_VISIBLE_DEVICES=""): params == 4,286,026; forward (4,3,32,32) → (4,10) finite; fp16-quantization mechanism check (`torch.tensor(x).half()` rounding step ≈ 2^-11 vs bfloat16 ≈ 2^-8 on a probe value — confirms the variable under test); GradScaler(enabled=False) passthrough identity (scale(loss) is loss; step calls optimizer.step — params change; update is a no-op) exercising the EXACT call sequence of the new loop; 2-step train smoke with the scaler-wrapped sequence: finite decreasing loss
- [x] Static check: `uv run python -c "import ast; ast.parse(open('train.py').read())"`

### Milestone 2: Gated launch on GPU 0, completion, pristine check
- [x] Verified `/tmp/exp046_composite.sh`; launched AS-IS
- [x] GATE_DECISION D0 = 22.7ms; completed rc=0; best = 96.54
- [x] Pristine PASS (windows 22.0–23.3ms; 136 ep; 13,181 steps; no NaN; ep1 36.47 in band)

### Milestone 3: Branch resolution (escalation only on a bar-clearing read)
- [x] Escalation NOT triggered (96.54 < 96.81)
- [x] Branch (ii) resolved: 96.54 ∈ mean band at family signatures → precision-null

### Milestone 4: Verification executed (first-failure-stop) and exp-log updated
- [x] Integrity PASS; Condition 1 FAIL (96.54 < 96.81); results in exp-log-054.md

## Code Changes
- **train.py** (only file, ~6 lines, 3 hunks): autocast dtype fp16 at both sites + GradScaler init + scale/step/update in the timed loop (and scale().backward() in warmup). Why this tests the hypothesis: it isolates ONE variable — the mantissa quantization step of every autocast tensor — at identical kernels, layout, schedule, and hyperparameters; the scaler is the standard mechanism making fp16's range usable and its overhead is priced in the projection. Risks: (a) forward/grad overflow at PEAK_LR 0.4 → dynamic scale backoff handles transients, persistent failure = pre-registered branch (iv); (b) scaler-skipped steps inflate the step counter without updates → monitored via loss-EMA spikes + epochs/plateau (a skip storm collapses the trajectory, caught by the divergence guard); (c) +≤0.3ms scaler overhead → priced, integrity band widened to D0 ≤ 23.5ms.

## Configuration Changes
- None. All training constants identical (certified recipe). The autocast dtype is an execution-regime variable, not a recipe constant.

## Execution Environment
- Method: local, `/tmp/exp046_composite.sh` verbatim (dual launch gates: zero GPU-0 compute apps AND load < 60, poll 30s×240 → `rm -f run.log` → background `uv run train.py > run.log 2>&1` → watchdog 44×15s: GATE_KILL D0 > 26ms, CONTENTION_KILL, STARTUP_KILL, NaN, divergence < 15% after ep5, WALL_CAP). One run; a second byte-identical run ONLY on the escalation branch (best ≥ 96.81), with run.log preserved to /tmp/exp054_run1.log before relaunch.
- Resources: GPU 0 only (H20); VRAM ~1.6GB; CIFAR-10 cached in `data/`
- Estimated runtime: ~8–9 min for one clean run (+9 if escalation triggers)
- Log output: run.log + composite stdout (background task output file)
- Tool skill: none (local)

## Abort Criteria
- NaN in log (composite auto-kills) → branch (iv): fp16 unstable at peak 0.4 under this recipe — research failure, axis closed on stability grounds, NO blind retry (one inspect-and-fix retry allowed ONLY for a clear implementation bug, e.g. scaling the printed loss)
- Divergence: eval < 15% after ep5 (auto-kill) → same branch (iv) handling
- GATE_KILL D0 > 26ms → implementation defect (fp16 ≈ bf16 speed): fix-or-fail per code-error rules (max 2)
- CONTENTION_KILL / STARTUP_KILL → infra: relaunch byte-identically when gates clear (max 2, then Outcome failed)
- Wall ≥ ~660s → kill, failure

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition**: pristine profile — ≥200-step windows mean ≤ 23.5ms, none > 27; num_epochs 133–142; params 4,286,026; training_seconds 300.0; evals ≤ num_epochs; numerics by the EXP-048 trajectory criterion (rejoins family, family-level plateau, test_loss ~0.18–0.20; ep1 informational except < 30% defect tripwire); no NaN/EMA spikes in the loss trail (spot-check step prints around epoch boundaries). Contention-tainted → rerun (infra), never analyzed.

1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1pp): `grep "^best_test_acc:" run.log` (timeout 10s; empty ⇒ crash ⇒ `tail -n 50 run.log`). If ≥ 96.81 → ESCALATE per Milestone 3 (improvement iff MEAN of two byte-identical runs ≥ 96.81; pre-registered, max never used). If < 96.81 → no-improvement, classified into branches (ii)/(iii) per hypothesis for the record.
2. **Within budget**: composite rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600 (timeout 10s)
3. **Eval cadence**: `grep -c "eval ep" run.log` ≤ num_epochs (timeout 10s)

Cleanup per goal Procedure: delete run.log (and /tmp/exp054_run1.log if escalation ran) at loop end.

### Informational Metrics (Optional)
- num_epochs / num_steps (`grep "^num_epochs:\|^num_steps:" run.log`): scaler-overhead datum (vs family 138–140 / 13,428–13,515); a step count NEAR family with epochs well BELOW would flag skip-storms
- final_test_loss (family ~0.185; elevated = trajectory degradation signature, cf. EXP-021's worse-basin reading)
- peak_vram_mb (~1.6GB), num_params (4,286,026), D0 (expect 22.3–22.8ms)
