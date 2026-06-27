# Experiment Log: EXP-037 — SE channel attention (r=16, all 9 blocks) with near-identity init

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-037.md
- **Plan**: plans/plan-037.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-037 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan Milestone 1 in train.py only: (1) new `SEModule` (squeeze = spatial mean; excite = Linear(C→C/16) → ReLU → Linear(C/16→C) → sigmoid; channel-wise multiply) with near-identity init — `fc2.weight` zeroed, `fc2.bias` = 2.0 → gate exactly sigmoid(2)=0.8808 constant at step 0; (2) `self.se = SEModule(out_channels)` in `BasicBlock.__init__` and `out = self.se(self.bn2(self.conv2(out)))` in forward (SE after last BN, before the shortcut add — canonical SENet placement); (3) `_weights_init` now skips modules carrying `skip_kaiming=True` so the global kaiming pass cannot randomize the SE gates. Sanity (CPU, `CUDA_VISIBLE_DEVICES="" uv run`): params 4,319,710 = hand-calc 4,286,026 + 33,684 exactly; fc2 weight absmax 0.0 AFTER ResNet construction (guard verified effective, since `self.apply` runs after `_make_layer`); gate mean 0.8808; forward (4,3,32,32)→(4,10) OK. AST parse OK; diff touches train.py only (+24/−2 lines).

### Surprises & Discoveries
- None functional. Confirmed the ordering subtlety the plan flagged: `ResNet.__init__` calls `self.apply(self._weights_init)` AFTER blocks are built, so without the `skip_kaiming` guard the kaiming pass would have silently overwritten the zero-init fc2 weights (fan_in C/16 → std up to ~0.7, i.e., randomized gates and a violated near-identity premise). The CPU check measured absmax 0.0 post-construction, proving the guard works.

### Decisions
- Used `x.mean(dim=(2,3))` for the squeeze instead of `F.adaptive_avg_pool2d(x,1)` — identical math, one fewer view, fuses well under compile.
- SE applied inside the bn2 output before the residual add (paper placement), not after the add — keeps the identity path untouched, which is the property the deferral-law engineering relies on.

## Run Log

### Run 1
- **Description**: Full composite gated run of SE-ResNet-20 (4× wide, r=16, 9 SE blocks, near-identity init) on GPU 0. Hyperparameters byte-identical to baseline; the only delta is the SE structure. Expected: dt ≈ 24–25.5ms (SE launch cost +1.5–3ms on launch-bound H20), ~123–132 epochs, params 4,319,710, VRAM ~1.62–1.7GB, total ~460–490s. SE-dt GATE: ticks 3–5 all ≥26.5ms at clean load → GATE_KILL (triggers pre-registered stage-3-only fallback); contention 4 consecutive >29ms thereafter; STARTUP_KILL tick 10; NaN; divergence eval<15% after ep5; wall cap 600s. Hypothesis read: converged plateau LEVEL vs bar 96.81.
- **Job ID / PID**: background task bbj6775hx (composite script /tmp/exp037_composite.sh; train PID in task output LAUNCHED line)
- **Log file**: run.log (project root; deleted after analysis per goal constraints)
- **WandB**: n/a
- **Status**: completed (rc=0) — but profile pre-condition FAILED (5 windows >29ms vs ≤2 allowed) → classified contaminated per plan, rerun triggered
- **Started**: 2026-06-10 19:08:55 (GATES_CLEAR poll 1: apps=0, load=5)
- **Ended**: 2026-06-10 ~19:16:48 (PROC_EXITED tick 32; total_seconds 473.2)
- **Observations**: SE-dt GATE PASSED decisively: gate windows 24.0/24.9/24.0ms — measured SE cost +1.7ms on the 22.4ms baseline step (best-case end of the +1.5–3 estimate; ~6 small launches/block well-fused by default compile). Startup 24.4s vs 9.2s baseline (bigger compile graph; uncharged). Near-identity init worked: ep1 35.35 vs family ~38 (mild, not EXP-018-class 20s), ep5 61.48 vs family ~64, ep10 72.05 — slight early lag consistent with gates learning, recovered by mid-run. Plateau onset ep 117. Last-8 evals 96.22–96.34, test_loss 0.188 ≈ family 0.185. Profile: 247 windows mean 24.1ms BUT 5 isolated windows at exactly 30.0ms scattered (steps 450/1900/3250/6350/9300) — total excess ≈1.5s ≈ 0.6 epochs ≈ −0.01pp; watchdog slow_streak never >0. Per plan pre-registration (≤2 slow windows), classified contaminated → Run 2.
- **Key Metrics**: best_test_acc 96.34 | final 96.29 | final_test_loss 0.1882 | 129 epochs / 12,433 steps | dt mean 24.1ms | params 4,319,710 | VRAM 1644.2MB | startup 24.4s | total 473.2s

### Run 2
- **Description**: Identical relaunch of the composite script per the plan's contamination rule ("Contaminated ⇒ rerun once, gates re-cleared first"). The 5 scattered +6ms windows in Run 1 cannot move the verdict (−0.01pp equivalent), but the pre-registered rule is honored to keep the read clean, and the rerun doubles as a second independent draw of the SE plateau level for analysis.
- **Job ID / PID**: background task br87nbox1 (same /tmp/exp037_composite.sh)
- **Log file**: run.log (overwritten by the rerun's `rm -f run.log` + fresh redirect)
- **WandB**: n/a
- **Status**: completed (rc=0, watchdog never triggered)
- **Started**: 2026-06-10 19:18:59 (GATES_CLEAR poll 1: apps=0, load=7)
- **Ended**: 2026-06-10 ~19:26:40 (PROC_EXITED tick 31; total_seconds 456.8)
- **Observations**: Replicates Run 1 within noise: best 96.37 (vs 96.34), 128 epochs (vs 129), dt mean 24.2ms (vs 24.1), test_loss 0.1884 (vs 0.1882), plateau onset ep 118 (vs 117), ep1 34.82 (vs 35.35). Startup 10.0s (compile cache warm vs Run 1's cold 24.4s). Profile awk: 246 windows, mean 24.2ms, slow>29: 9 — MORE than Run 1's 5 at equally clean load, all exactly 30.0ms. **Root cause found — instrument quantization, not contention**: the step-line `pct` prints with 1 decimal (0.1% = 0.3s); a 50-step window at true ~24.2ms spans Δpct ≈ 0.403% and rounds to 0.4 (reads 24.0ms) or 0.5 (reads exactly 30.0ms). The 29ms threshold sits between the 24.0/30.0 quantization rungs, so a clean 24.2ms run necessarily shows scattered "30.0" windows. Coarse watchdog windows (~400–450 steps, ≈0.7ms granularity): max 25.5ms, zero >29, no streaks, both runs. Pre-condition INTENT (uncontaminated read) satisfied on the coarse profile; both runs are clean draws of the SE level.
- **Key Metrics**: best_test_acc 96.37 | final 96.36 | final_test_loss 0.1884 | 128 epochs / 12,391 steps | dt mean 24.2ms | params 4,319,710 | VRAM 1615.6MB | startup 10.0s | total 456.8s

## Experimental Adjustments
- **Run 2 trigger (2026-06-10)**: Run 1 completed rc=0 with all watchdog windows 24.0–24.9ms, but the post-hoc 50-step profile found 5 isolated windows at 30.0ms (>29 threshold, plan allows ≤2). Magnitude analysis says the excess (~1.5s charged, ~0.6 epochs, ~−0.01pp) cannot explain the −0.23-vs-mean shortfall, so this is jitter rather than run-corrupting contention — but the rule was pre-registered precisely so that marginal calls do not get judgment-called away. Rerun once per plan; Run 1's numbers are retained in this log for the two-draw comparison in analysis.
- **Run 2 profile resolution (2026-06-10)**: Run 2 (equally clean load, watchdog max 25.5ms, zero streaks) showed 9 such windows, all EXACTLY 30.0ms — diagnosed as pct-print quantization (0.1% = 0.3s; 50-step windows at true 24.2ms round to 0.4%→24.0ms or 0.5%→30.0ms; the 29ms threshold sits between rungs). The fine-profile criterion is instrument-invalid at dt ≈ 24ms; the pre-condition is evaluated on the coarse watchdog windows instead (0 exceedances both runs). No further reruns — the plan's one contamination rerun is spent and the contamination hypothesis is affirmatively refuted. PROTOCOL NOTE for future plans: set fine-profile slow-window thresholds OFF the 6ms quantization rungs (e.g., 31ms, not 27–30) or profile on ≥200-step windows.

## Errors & Dead Ends
(none yet)

## Verification Results

### Conditions Checked
First-failure-stop protocol per plan-037 (bar = baseline 96.71 + 0.1 = 96.81). Verified on Run 2 (the protocol-clean run); Run 1 corroborates.

1. **best_test_acc ≥ 96.81** — **FAIL**. `grep "^best_test_acc:" run.log` → **96.37%** (Run 1: 96.34 — two concordant draws). Pre-condition (profile): fine 50-step awk reads mean 24.2ms but its slow-window count (9 >29ms) is instrument-invalid at this dt — all exceedances are exactly 30.0ms pct-quantization rungs (see Experimental Adjustments); evaluated on coarse watchdog windows: 29 windows 24.0–25.5ms, zero >29, no streaks → uncontaminated, read honest. Integrity sub-checks pass: num_params 4,319,710 ✓ (= planned SE total); training_seconds 300.0 ✓; eval_lines 128 = num_epochs 128 ✓. 96.37 < 96.81 → condition fails on its merits.
2. **Completes within budget** — skipped (aborted after prior failure). [Incidental: rc=0, total_seconds 456.8 ≤ 600.]
3. **Validation ≤ once/epoch** — skipped (aborted after prior failure). [Incidental: 128 = 128.]

**Verdict basis**: valid clean replicated result, necessary condition 1 failed → no-improvement. Decomposition: SE cost +1.7ms → −11 epochs → expected deficit-only level ≈ 96.57 − 0.15 ≈ 96.42; measured 96.34/96.37 ≈ deficit-only within noise → SE's retained gain ≈ 0 under this recipe.

### Informational Metrics
- **Measured SE dt cost**: +1.7ms (22.4 → 24.1/24.2ms) for 9 SE modules ≈ 0.19ms/module — far below the 2.5ms/block full-block launch cost; small fused pointwise+matmul chains are cheap under default compile (refines the EXP-026/034 pricing law).
- **Early-heat trace**: ep1 35.35/34.82 vs family ~38, ep5 61.5/58.9 vs ~64 — the near-identity init avoided EXP-018-class deferral (no 20s-class ep1) but a mild early lag remained while gates learned; recovered by mid-run.
- **Plateau**: onset ep 117/118 (family ~120 at 139-epoch pace); last-8 evals 96.22–96.37 tight; final_test_loss 0.1882/0.1884 vs family ~0.185 — basin level statistically indistinguishable from baseline family.
- **VRAM**: 1644.2 / 1615.6MB vs 1613.0 baseline (+0–2%). Startup: 24.4s cold compile / 10.0s warm cache.

## Human Notes
(autopilot — none)
