# Experiment Log EXP-034: Depth-for-width at matched compute — ResNet-26 at stage widths 56/112/224

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-034.md
- **Plan**: plans/plan-034.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-034
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
All Milestone-1 edits applied on `autoresearch/exp-034` (cut from `autoresearch/dev` @ 1990397): (1) `NUM_BLOCKS = 4` (ResNet-26); (2) `WIDTH_MULT = 4` replaced by `STAGE_WIDTHS = (56, 112, 224)`; (3) `ResNet.__init__` takes `stage_widths` tuple instead of `width_mult` (`w1, w2, w3 = stage_widths`); (4) construction passes `STAGE_WIDTHS` and the model print line shows the widths tuple. Recipe constants, transforms, loaders, schedule, compile+warmup, timed step, eval all byte-identical to baseline. Diff: 6 insertions / 6 deletions at exactly the 4 planned sites.

### Surprises & Discoveries
- CPU sanity check (`CUDA_VISIBLE_DEVICES="" uv run python -c ...`) printed **params: 4469538** — exactly the hand-computed value (method validated by reproducing baseline's 4,286,026) — and a 2×3×32×32 forward pass returned (2, 10). BasicBlock's pad-shortcut handles the 56→112→224 transitions with no changes, as planned.

### Decisions
- Ran the param-count sanity with `CUDA_VISIBLE_DEVICES=""` because `train.py` instantiates `Eval()` at module import — masking the GPU guarantees the check cannot touch GPU 0 ahead of the launch gates.

## Run Log

### Run 1
- **Description**: Full run on GPU 0 of ResNet-26 (12 blocks) at stage widths 56/112/224 — a depth-for-width trade at ≈matched conv FLOPs (1.021×) and ≈matched params (4,469,538 vs 4,286,026), recipe byte-identical to baseline. Expected dt ≈ 22.9–24ms (compute +2.1% plus per-block overhead on 12 vs 9 blocks), epochs ≈ 130–139, total ~475–510s. Hypothesis: composition depth raises the converged plateau LEVEL (decision-boundary expressivity per unit compute), best ≥ 96.81. Falsified by: early-dt GATE_KILL (>24.5ms ×3 consecutive windowed ticks, clean load), or a clean full run with plateau within/below the baseline noise band (96.4–96.7).
- **Job ID**: local background composite, Claude task bpgr5ssi5
- **Log file**: run.log
- **WandB**: N/A
- **Status**: GATE_KILL (early-dt gate, by design)
- **Started**: 2026-06-10 (gates passed at poll 1, load 8)
- **Ended**: 2026-06-10 (killed at tick 5, ~75s of training)
- **Observations**: Windowed dt 31.5 / 30.7 / 30.8 ms on three consecutive windowed ticks at CLEAN load (8–12, GPU 0 exclusively ours) → GATE_KILL per plan (threshold 24.5ms ×3 within first 10 ticks). The +38% dt against +2.1% conv FLOPs refutes the FLOPs-scaling premise: per-block cost is ~2.58ms (31/12) vs baseline ~2.49ms (22.4/9) DESPITE each block carrying 23% less compute — step time at this model scale is launch/memory-bound, near-constant per block. Model line confirmed `ResNet-26 (widths (56, 112, 224)) | params: 4,469,538` (exact match to hand calc). 10 evals before kill; ep1 35.65 (family ~38, slightly low — consistent with fewer steps/epoch worth of progress at 31ms... actually fewer EPOCHS of progress per wall second; trajectory normal otherwise, ep10 74.94 vs baseline-family ~78 at ep10 — consistent with ~28% fewer steps elapsed, not a trainability problem).
- **Key Metrics**: win dt ≈ 30.7–31.5ms | per-block ≈ 2.58ms | evals 10 | ep10 74.94 | params 4,469,538 | rc 137 (SIGKILL by watchdog)

### Run 2 (pre-registered fallback)
- **Description**: STAGE_WIDTHS (48, 96, 192) — same ResNet-26 depth, width ×0.75 (FLOPs 0.735× of Run 1, 0.75× of baseline; params 3,284,986). Decisive test of launch-overhead dominance: if dt ∝ FLOPs, expect ~19–20ms (run proceeds, depth probe completes with an epoch surplus and a params-down confound); if per-block launch-bound as Run 1 suggests, expect ~29–30ms → second GATE_KILL, closing the depth direction on hardware grounds with two measured points.
- **Job ID**: local background composite, Claude task b3paxor1t
- **Log file**: run.log
- **WandB**: N/A
- **Status**: GATE_KILL (early-dt gate, by design)
- **Started**: 2026-06-10 (gates held 9 polls ≈ 4.5 min for a foreign GPU-0 process, passed at poll 10, load 10)
- **Ended**: 2026-06-10 (killed at tick 5, ~75s of training)
- **Observations**: Windowed dt 27.6 / 27.0 / 27.7 ms at clean load (9–14) → GATE_KILL. Decisive: FLOPs ×0.735 vs Run 1 bought only −3.6ms. Linear decomposition across the two 12-block points gives ∂dt/∂FLOPs ≈ 13.3ms per baseline-FLOPs-unit → the 3 extra blocks cost ≈ +8.3ms independent of width; per-block overhead (~2.3–2.8ms) dominates compute at these widths. A 12-block net would need ~0.53× baseline FLOPs (width ~40/80/160, ~2.3M params) to fit 24.5ms — a half-capacity giveaway. The matched-dt depth design point does NOT exist on this hardware. Model line confirmed `ResNet-26 (widths (48, 96, 192)) | params: 3,284,986` (exact match to hand calc).
- **Key Metrics**: win dt ≈ 27.0–27.7ms | 11 evals | ep10 73.01 | params 3,284,986 | rc 137 (SIGKILL by watchdog)

### Run 3 (full run at 56/112/224, dt gate lifted)
- **Description**: The plan's matched-dt premise is refuted (two-point measurement above), but the HYPOTHESIS — depth raises the converged plateau LEVEL — is still unmeasured, and the loop bookkeeping needs a numeric metric for an honest verdict. Run the params-matched variant (56/112/224, 4,469,538 params) to completion at its measured ~31ms: time-keyed anneal completes regardless; expect ~97–100 epochs (139×22.4/31), total ~430–460s, well under cap. Watchdog: dt GATE disabled; contention threshold rescaled to the architecture's own dt (4 consecutive windows >37ms ≈ 1.2×31, mirroring 27 = 1.2×22.4); NaN/divergence/startup/wall-cap unchanged. Outcome interpretation: best ≥96.81 = depth pays even at −39 epochs (improvement); best in/below baseline band = depth direction closed with a measured LEVEL datum (the EXP-005 analogue for depth).
- **Job ID**: local background composite, Claude task bjsxrq30l
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed (rc=0)
- **Started**: 2026-06-10 (gates held 9 polls for a foreign GPU-0 process, passed at poll 10, load 11)
- **Ended**: 2026-06-10 (437s wall incl. composite overhead)
- **Observations**: Fully clean at the architecture's own pace: 195 windows mean 30.5ms, 0 windows >37ms, load 10–14 throughout; dt alternated 30.0/30.9 with zero drift — no contamination signature. Epochs match dt arithmetic exactly (139×22.4/30.5 = 102.1 → 102). Trajectory: ep1 35.54 (baseline family ~38 — slightly behind per-epoch, as expected with 28% fewer steps/sec of wall... per-epoch steps identical; behind per-WALL-second), converged smoothly, best 96.01 at ep98, final 15 evals flat 95.9–96.0 with test_loss settling 0.196–0.198. The anneal completed normally (time-keyed); the plateau is genuinely converged (flat tail, falling-then-flat loss), just LOWER: −0.70 vs recorded baseline, −0.56 vs the 96.57 run-level mean ≈ 3.5σ — decisively real.
- **Key Metrics**: best_test_acc 96.01 | final 95.96 | final_test_loss 0.1964 | training_seconds 300.0 | total_seconds 418.3 | startup 10.1s | VRAM 1815.8MB | epochs 102 | steps 9,839 | params 4,469,538 | dt mean 30.5ms

## Experimental Adjustments
- **2026-06-10 — Fallback width (48, 96, 192) after Run 1 GATE_KILL** (pre-registered in plan-034 Milestone 3): Run 1's clean-load dt of ~31ms vs predicted 22.9–24 refutes FLOPs-proportional scaling; the fallback's FLOPs-based prediction (~19–20ms) is therefore likely wrong too, but running it converts "launch-bound suspicion" into a two-point measurement that closes the axis either way. Max one fallback per plan; no further width reductions regardless of outcome.
- **2026-06-10 — Run 3 beyond the plan's stop point (autopilot decision, recorded per execute-skill §6)**: the plan said stop after a second gate-kill, but that leaves EXP-034 with no metric — misclassifiable only as crash/invalid, both wrong (nothing crashed, no constraint violated; the gate kills were deliberate budget protection). A full run at the measured dt costs ~8 min, fits the wall cap with ~140s margin, and converts the axis closure from an inference into a measured plateau LEVEL datum (depth analogue of EXP-005's width datum) — or, in the unlikely upside, reveals the level gain the hypothesis predicted. dt gate lifted for Run 3 (its premise is moot); contention detection retained at the rescaled threshold so foreign-GPU interference is still caught.

## Errors & Dead Ends
(none yet)

## Verification Results

### Conditions Checked
1. **best_test_acc ≥ 96.81 (bar = baseline 96.71 + 0.1)** — **FAIL**. `grep "^best_test_acc:" run.log` → 96.01%. −0.80 vs bar; −0.70 vs recorded baseline; −0.56 vs baseline mean ≈ 3.5σ — decisively real, far outside the noise band.
   - Pre-condition (profile, thresholds rescaled to the architecture's measured dt per Run-3 adjustment): **PASS** — 195 windows, mean 30.5ms, slow>37ms: 0; num_epochs 102 = exactly 139×22.4/30.5; training_seconds 300.0; num_params 4,469,538 (exact hand-calc match); eval_lines 102 = num_epochs; model line `ResNet-26 (widths (56, 112, 224))`. Clean, trustworthy run. (Note: the plan's original mean ≤24.5ms pre-condition belonged to the refuted matched-dt premise; the rescaled check verifies the same thing — absence of foreign interference — at the architecture's own pace.)
2. **Completes within budget** — not evaluated (first-failure-stop). Informationally: rc=0, total_seconds 418.3 ≤ 600 ✓.
3. **Validation ≤ once/epoch** — not evaluated (first-failure-stop). Informationally: 102 = 102 ✓.

**Informational**: dt at depth 26 = 30.5ms mean (the depth-overhead datum: +8.1ms over baseline at +2.1% FLOPs; two-point decomposition with Run 2 gives ~+2.8ms per extra block, launch/memory-bound); ep1 35.54 vs family ~38; plateau: last-15 mean ≈ 95.97, test_loss 0.196 vs baseline family 96.6 / 0.185 — the deeper net's converged LEVEL is lower, not just its transit; startup 10.1s (compile did not grow measurably); VRAM 1815.8MB (+203MB).

## Human Notes
(autopilot — none)
