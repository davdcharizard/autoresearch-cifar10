# Experiment Log EXP-054: fp16 autocast + GradScaler — finer-precision direction of the numerics axis

## Execution

- **Created**: 2026-06-11
- **Brainstorm**: brainstorm/brainstorm-054.md
- **Plan**: plans/plan-054.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-054
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Applied the planned ~6-line change on branch autoresearch/exp-054 (diff: 1 file, +7/−5, 3 hunks exactly as planned): (1) `scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")` after the optimizer; (2) warmup autocast dtype → `torch.float16`, backward → `scaler.scale(warm_loss).backward()` (no step/update in warmup; grads zeroed after); (3) timed-loop autocast dtype → `torch.float16`, `loss.backward(); optimizer.step()` → `scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()`. Printed loss remains the UNSCALED autocast output (fp32). CPU sanity `/tmp/exp054_sanity.py` ALL PASS: params 4,286,026; forward finite; mechanism check ulp(1.0) fp16 2^-10 vs bf16 2^-7 (8× finer, the variable under test); GradScaler(enabled=False) passthrough identity exercising the exact new call sequence (scale returns the same tensor, step updates params, update no-op); 2-step smoke decreasing (2.600 → 1.161). AST OK. M1 complete.

### Surprises & Discoveries
- None at implementation time. Note for the record: ulp(1.0) is 2^-10 for fp16 (10 explicit mantissa bits ⇒ relative step 2^-10 at 1.0; the brainstorm's "2^-11" is the half-ulp rounding error) — the 8× ratio vs bf16, which is the tested variable, is unchanged.

### Decisions
- Pre-registered branches per plan-054: (i) best ≥ 96.81 → escalate to a second byte-identical run, improvement iff MEAN ≥ 96.81 (EXP-052 protocol; max never used); (ii) mean-band [96.41, 96.73] at family signatures → precision-null, numerics axis closed both directions; (iii) < 96.41 or elevated plateau scatter at clean dt → negative sign closure; (iv) NaN/divergence → stability closure, research failure, no blind retry (one fix-retry only for a clear implementation bug); (v) gate/contention kills → infra relaunch (max 2).

## Run Log

### Run 1

**Description**: Single gated run of the fp16+GradScaler variant — the only never-probed axis left (finer-mantissa numerics at identical kernels/throughput). Launched via `/tmp/exp046_composite.sh` (dual gates, watchdog with NaN/divergence guards covering the fp16 stability tail risk). Expected: D0 22.3–22.8ms (scaler overhead ≤ 0.3ms), epochs 133–141, family trajectory if numerics are benign; escalation to a replicate pair only on a bar-clearing read.

**Metadata**:
- Job ID: background task b2wncf142 (composite pid 1892013)
- Log file: run.log
- WandB: N/A
- Status: completed (rc=0)
- Started: 2026-06-11 04:54:09
- Ended: 2026-06-11 ~05:02:15

**Observations**: PRISTINE, fp16 fully stable. GATES_CLEAR poll 1 (apps=0, load=12). GATE_DECISION D0=22.7ms (scaler overhead ~0.2–0.4ms over the 22.3–22.5 family, exactly as priced). Windows 22.0–23.3ms, slow_streak 0, no NaN/divergence events; ep1 = 36.47 in band, ep2 dip then normal climb (trajectory criterion satisfied: rejoined family, plateau at family level). Summary: **best_test_acc 96.54**, final 96.49, final_test_loss 0.1877 (family), 300.0s charged, 483.8s total, **136 epochs / 13,181 steps** (scaler cost ≈ 2–3 epochs ≈ 250 steps vs family 138–140/13,428–13,515 — the priced deferral toll), params 4,286,026, evals 136 ≤ 136. Tail converged-flat (96.52/96.49/96.54/96.49). **96.54 ∈ [96.41, 96.73] mean band → pre-registered branch (ii): precision-null at family signatures.** No escalation (read < 96.81).

## Experimental Adjustments

- None — single clean run, no retries.

## Errors & Dead Ends

- None.

## Verification Results

### Conditions Checked

**Integrity pre-condition (PASS)**: windows 22.0–23.3ms (mean ≤ 23.5, none > 27) ✓; epochs 136 ∈ [133,142] ✓; params 4,286,026 ✓; training_seconds 300.0 ✓; evals 136 ≤ 136 ✓; trajectory criterion: ep1 36.47 (>30 tripwire passed), rejoined family, plateau family-level, test_loss 0.1877 ∈ [0.18,0.20] ✓; no NaN/EMA spikes in loss trail ✓. Source: run.log; composite task b2wncf142.

**Condition 1 — best_test_acc ≥ 96.81**: 96.54 → **FAIL** (first-failure-stop). No escalation (escalation branch required ≥ 96.81). Pre-registered branch (ii): 96.54 ∈ [96.41, 96.73] mean band at family signatures → precision-null; the numerics axis is closed in BOTH directions (EXP-021 coarser: −0.20; EXP-054 finer: null at a 2–3-epoch toll).

**Condition 2 — within budget**: rc=0; total_seconds 483.8 ≤ 600 → PASS (informational).

**Condition 3 — eval cadence**: 136 ≤ 136 → PASS (informational).

**Verdict basis**: no-improvement (valid pristine run, condition 1 failed; branch (ii) closure).

## Human Notes

(autopilot — none)
