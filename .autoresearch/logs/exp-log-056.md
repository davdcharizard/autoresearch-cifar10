# Experiment Log EXP-056: Full pre-activation block reorder (ResNet v2 / WRN-native B(3,3))

## Execution

- **Created**: 2026-06-11
- **Brainstorm**: brainstorm/brainstorm-056.md
- **Plan**: plans/plan-056.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-056
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Applied the planned 4-part reorder on branch autoresearch/exp-056 (train.py only, +14/−7): (1) BasicBlock.bn1 → `BatchNorm2d(in_channels)` (normalizes block INPUT); (2) BasicBlock.forward → full pre-activation `relu(bn1(x)) → conv1 → relu(bn2(·)) → conv2 → + raw-x pad shortcut`, NO post-addition ReLU (clean identity end-to-end; pad shortcut takes raw x — He's shared-pre-activation shortcut applies to learnable projections, which this net does not have); (3) ResNet stem bare (stem `bn1` deleted), `bn_final = BatchNorm2d(256)` added; (4) ResNet.forward → conv1 → layers → `relu(bn_final(·))` → GAP → fc. Optimizer/warmup/loop/eval byte-identical to baseline; zero config changes. AST OK. CPU sanity `/tmp/exp056_sanity.py` ALL PASS: (a) params EXACTLY 4,286,026 (the planned BN-size cancellation: −128 stem, −128/−256 transition bn1 shrinkage, +512 bn_final nets to zero); (b) per-block bn1 sizes (64×4, 128×3, 256×2 pattern by position) and bn2 sizes correct, bare stem, bn_final(256); (c) block output takes negative values (no post-add ReLU — clean-identity mechanism check); (d) forward (8,10) finite both modes; (e) all 59 param tensors receive grads; (f) bn_final + moved bn1 running stats track in train mode; (g) 3-step smoke decreasing (2.377 → 1.759). GPU probe `/tmp/exp056_gpu_probe.py` (EXP-055 protocol, gates clear apps=0/load=12): compile warmup 20.0s, **pre-act dt 23.08ms — PROBE PASS (≤ 24.0)**. M1 complete.

### Surprises & Discoveries
- **The op reorder is NOT dt-free: +1.0ms/step vs the 22.04ms family probe** (23.08 vs 22.04, same kernel set — inductor fuses the pre-act order less favorably; plausibly the stem conv now feeds an unfused BN→ReLU→conv chain and the final BN→ReLU adds an unfused epilogue). By the conversion law this prices ≈ −6 epochs ≈ −0.08pp — a real toll the hypothesis read must carry (recorded as a pre-launch plan revision: epochs band [128,138], steps [12,400,13,400], D0 acceptance [22.6, 24.1]).

### Decisions
- Launched despite the +1.0ms toll per the plan's pre-registered probe rule (≤ 24.0 → proceed): the toll is priced and small relative to the (ii)/(iii) decision bands, and a cost-only kill at 23.1ms would close nothing (the class question is the ORDER's statistical effect, which the run still reads cleanly with the toll subtracted in interpretation).
- Pre-registered branches per plan-056: (i) ≥ 96.81 → byte-identical replicate, improvement iff MEAN ≥ 96.81 (EXP-052 protocol; (96.73, 96.81) reads are no-improvement, never promoted); (ii) [96.41, 96.73] family-shaped → absorption-null, block-order class closed, modernization audit COMPLETE; (iii) < 96.41 → post-act ordering load-bearing at shallow depth (He depth-trend-consistent sign closure); (iv) probe/D0 > 26 → cost-closure without/with kill (not triggered: 23.08); (v) infra → relaunch (max 2).

## Run Log

### Run 1

**Description**: Single gated run of the pre-activation reorder — the last un-enumerated structural class (op ORDER) and final standard-modernization entry; anchor matched on dataset+depth+width (WRN B(3,3)). Probe-priced at 23.08ms (+1.0ms toll ≈ −0.08). Expected: D0 ∈ [22.6, 24.1], epochs ~131–134, steps ~12,700–13,000, params 4,286,026 exact, ep1 ≥ 30, family trajectory; startup 25–45s (fresh compile). Escalation to a replicate pair only on a bar-clearing read.

**Metadata**:
- Job ID: background task bh6zakvy6 (composite)
- Log file: run.log
- WandB: N/A
- Status: running
- Started: 2026-06-11 ~06:10

**Observations**: PRISTINE (note: this was the second launch attempt; the first, botched detached launch was killed during startup before any charged work — see Errors & Dead Ends). GATES_CLEAR poll 1 (apps=0, load=11). D0 = 24.0ms ∈ acceptance [22.6, 24.1] (in-run sits ~+0.9 above the 23.08 tight-loop probe, same offset pattern as EXP-055); all windows 23.3–24.0ms, slow_streak 0 throughout, no kill markers, rc=0. Startup 9.5s (FX cache populated by the killed first attempt's compile — uncharged either way). **best_test_acc 96.49**, final 96.40, final_test_loss 0.1887 (family band), 300.0s charged, 456.8s total, **131 epochs / 12,618 steps** (within the probe-revised bands [128,138]/[12,400,13,400] — the +1.0ms toll cost ~7 epochs vs family exactly as priced), params 4,286,026 EXACT (cancellation arithmetic confirmed in vivo), evals 131 ≤ 131, ep1 33.77 (> 30 tripwire; slightly below family ~36–39, healed by ep3), converged plateau (last 8: 96.33–96.49, gentle late climb). **96.49 ∈ [96.41, 96.73] → pre-registered branch (ii): absorption-null.** Toll-adjusted decomposition: raw read = mean − 0.5σ; adding back the priced ~+0.08 epoch toll puts the ORDER effect itself at ≈ 0 — the v2 mechanism contributes nothing detectable under TA+RE at depth 20. No escalation (read < 96.81).

## Experimental Adjustments

- **Pre-launch plan revision (probe-based)**: integrity bands updated for the measured +1.0ms pre-act toll — epochs [128, 138], steps [12,400, 13,400], D0 [22.6, 24.1]. Same protocol as EXP-055's probe-based revision; recorded before launch.

## Errors & Dead Ends

### 2026-06-11 — Botched first launch attempt (operator error, not the experiment)
- Error: `composite launched via "& disown" in a foreground shell with stdout to /dev/null — no task handle, gate/watchdog telemetry lost`
- Root cause: launch discipline slip; the verification protocol REQUIRES the composite stdout (D0, windows). Compounding: the cleanup `pkill -f exp046_composite` matched the cleanup shell's own command line and killed it (exit 144) — kill leftovers by explicit PID instead.
- Source: this session, pre-Run-1; all processes killed within ~60s of launch (train was still in startup/compile, no charged work of consequence), GPU 0 verified clear (no compute apps, load 11.5).
- Do NOT retry: always launch the composite with Bash run_in_background (task handle + captured output); never pkill by a pattern that appears in your own command line.

## Verification Results

### Conditions Checked

**Integrity pre-condition (PASS)**: GATES_CLEAR poll 1; D0 24.0 ∈ [22.6, 24.1] (probe-revised acceptance); windows 23.3–24.0 (mean ≤ 24.5, none > 27) ✓; no kill markers, rc=0 ✓; num_params 4,286,026 EXACT ✓; training_seconds 300.0 ✓; num_epochs 131 ∈ [128, 138] ✓; num_steps 12,618 ∈ [12,400, 13,400] ✓; total_seconds 456.8 ≤ 600 ✓; evals 131 ≤ 131 ✓; ep1 33.77 ≥ 30 ✓ (mild early lag healed by ep3, plateau family-adjacent — trajectory criterion satisfied); final_test_loss 0.1887 family band ✓; no NaN ✓. Source: run.log; composite task bh6zakvy6.

**Condition 1 — best_test_acc ≥ 96.81** (baseline 96.71 + 0.1, re-queried): **96.49 → FAIL** (first-failure-stop). No escalation (< 96.81). **Pre-registered branch (ii)**: 96.49 ∈ [96.41, 96.73] at family signatures with the +1.0ms toll priced — absorption-null; the block-order class is closed and the standard-modernization audit is COMPLETE.

**Condition 2 — within budget**: rc=0, 456.8 ≤ 600 → PASS (informational).

**Condition 3 — eval cadence**: 131 ≤ 131 → PASS (informational).

**Verdict basis**: no-improvement (valid pristine run; condition 1 failed; branch (ii) closure).

## Human Notes

(autopilot — none)
