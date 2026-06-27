# Experiment Log: EXP-040 — Uniform 5× width (80/160/320) behind the early dt gate

## Execution

- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-040.md
- **Plan**: plans/plan-040.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-040
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (both pre-registered width attempts GATE_KILLed at 54ms — planned screen outcome, no full run; pre-registered verdict `invalid`)

## Implementation Notes

### Summary
Milestone 1 as planned: on `autoresearch/exp-040` (cut from `autoresearch/dev` @ 1990397), single-constant change `WIDTH_MULT = 4 → 5` (stage widths 64/128/256 → 80/160/320). `git diff --stat`: train.py only, 1 insertion / 1 deletion. CPU sanity: params 6,693,850 exactly as computed at brainstorm time; forward on (2,3,32,32) returns (2,10). Composite launcher `/tmp/exp040_composite.sh` extends the exp039 baseline-variant with the EXP-040 dt-gate: the first 3 valid watchdog windows form D0 (median); GATE_KILL (exit 47) if D0 > 36ms (projected < ~86 epochs); otherwise contention threshold becomes D0 × 1.25 for the remainder — the dt-adaptive variant the plan requires because the 5× step time is itself the experiment's first measurement.

### Surprises & Discoveries
- **The dt landscape above 4× is a CLIFF, not a slope**: 4.5× (max-width 288) and 5× (max-width 320) both measured exactly 54ms — 2.4× the 4× model's 22.4ms — although their FLOPs differ from 4× by only 1.27× and 1.56×. The compute/kernel crossover is a hard boundary in (256, 288] channels (likely cuDNN/inductor kernel-selection or tensor-core tiling at the 256 boundary under channels_last+bf16). EXP-007's 6× 58ms now reads as "just past the same cliff", and EXP-034's "per-block cost width-independent" holds only up to 256 channels.
- Fallback param estimate in the plan (5,433,514) was off by 10,392 vs the constructed model (5,423,122) — plan corrected to the measured value before launch; sanity asserted on widths + forward instead.

### Decisions
- Gate set at 36ms (≥ ~86 projected epochs), stricter than the Failed-Approaches re-entry minimum (70 epochs / 44ms), per plan rationale: the smallest measured CONVERGED run is 83 epochs (EXP-008) and a wider model needs more convergence margin, not less.
- STARTUP_KILL moved from tick 10 → 12 (180s): the 5× model compiles a larger graph; startup is uncharged so the extra 30s of patience costs nothing.
- Pre-registered fallback on GATE_KILL: explicit widths 72/144/288 (≈4.5×, params 5,433,514), one relaunch under the same gate. Both-kill ⇒ verdict `invalid` (NaN) with the two measured dts as the learning.

## Run Log

### Run 1
- **Description**: Full budget-matched run of the 5× model on GPU 0 via `/tmp/exp040_composite.sh`. The dt gate decides the regime in the first ~5 ticks: if D0 ≤ 36ms the run continues to completion (launch-bound scenario predicts D0 ~25–31ms → 100–120 epochs); if D0 > 36ms the run is killed at ~90s as compute-bound (the historical starvation failure mode, now caught before it wastes a slot). Hypothesis: converged 5× plateau exceeds the 4× family if the width-level curve has not saturated; expected signatures if gate passes: ~95–120 epochs, VRAM ~2.4–2.6GB, total ≤ ~560s.
- **Job ID / PID**: background task bbh8hf20f (`/tmp/exp040_composite.sh`); train PID in composite stdout
- **Log file**: run.log (project root); watchdog + gate decision via composite stdout (task output file)
- **WandB**: N/A
- **Status**: GATE_KILL (exit 47, by design)
- **Started**: 2026-06-10 20:24:02 (gates clear at poll 1: GPU-0 apps=0, load=8)
- **Ended**: 2026-06-10 ~20:25:30 (killed at tick 5 after gate decision)
- **Observations**: All three gate windows read exactly 54.0ms (ticks 3–5, 200–250-step windows; task bbh8hf20f output L3–7). GATE_DECISION D0=54.0ms, projected 57 epochs < 86 → killed ~90s in. 5× (80/160/320) is deep in the COMPUTE-BOUND regime — 54ms is nearly 6×'s 58ms (EXP-007), not an interpolation toward 4×'s launch-bound 22.4ms. The launch-bound plateau (per-block cost width-independent, EXP-034) ends at-or-just-above 4× widths: the H20's kernel-time-vs-launch crossover for these 3×3 convs sits between 256 and 320 max-width.
- **Key Metrics**: D0 = 54.0ms; projected epochs 57 (would have replicated the EXP-007 starvation point). No training metrics (killed pre-eval, by design).

## Experimental Adjustments

- **Run 1 GATE_KILL → pre-registered fallback activated** (plan-040 § Configuration Changes): widths changed from `16/32/64 × WIDTH_MULT` arithmetic to explicit `w1, w2, w3 = 72, 144, 288` (≈4.5×). CPU sanity: params **5,423,122** (the plan's 5,433,514 was a planning-time estimate — corrected in plan-040 to the constructed value), forward (2,10) OK. Same gate applies: D0 > 36ms → both-kill branch (verdict `invalid`, dt data as learning). Evidence: task bbh8hf20f L3–7 (54.0ms × 3 windows).

### Run 2
- **Description**: Fallback run at widths 72/144/288 (≈4.5×, params 5,423,122) under the identical composite gate. Interpolation guess is now pessimistic: 5× landed at 54ms (compute-bound onset is between max-width 256 and 320), so 4.5× (max-width 288) could land anywhere in ~28–48ms; the gate decides at tick ~5. If D0 ≤ 36ms the run completes at ≥ ~86 epochs and reads the converged 4.5× level; if killed, the width axis closes on dt grounds with the crossover bracketed at (256, 288].
- **Job ID / PID**: background task bw1c08tob (`/tmp/exp040_composite.sh`)
- **Log file**: run.log
- **WandB**: N/A
- **Status**: GATE_KILL (exit 47, by design)
- **Started**: 2026-06-10 20:27:36 (gates clear at poll 1: apps=0, load=6)
- **Ended**: 2026-06-10 ~20:29:00 (killed at tick 5)
- **Observations**: Gate windows 54.0 / 52.8 / 54.0ms → D0 = 54.0ms, projected 57 epochs (task bw1c08tob L3–7). run.log header confirms the right model ran (params 5,423,122, 97 batches/epoch); printed dt 53–54ms and img/s ~9,540 corroborate the windows. **4.5× is byte-equal in speed to 5×**: max-width 288 and 320 both run at 54ms while max-width 256 runs at 22.4ms — a 2.4× CLIFF between 256 and 288 channels, not FLOPs scaling (FLOPs ratio 4.5×/4× is only 1.27). The H20/inductor/channels_last kernel path changes character above 256 channels (tile/kernel-selection boundary).
- **Key Metrics**: D0 = 54.0ms; projected 57 epochs. No training metrics (killed pre-eval, by design).

## Errors & Dead Ends

### 2026-06-10 — Both width attempts gate-killed (planned screen, not an error)
- Error: `GATE_KILL: D0=54.0ms > 36ms (projected 57 epochs < ~86)` — identical for 5× (80/160/320) and 4.5× (72/144/288)
- Root cause: hard kernel-performance cliff above 256 channels on H20 + torch.compile(default) + channels_last + bf16 — dt jumps 22.4 → 54ms between max-width 256 and 288; both widths would replicate the EXP-007 starvation point (57 epochs)
- Source: task bbh8hf20f L3–7; task bw1c08tob L3–7; run.log header (params 5,423,122, dt 53–54ms prints)
- Do NOT retry: ANY uniform or stage-3 width with >256 channels anywhere in the net on this stack — the cliff prices it at ~54ms regardless of how little the width exceeds 256. Width interior between 4× and the cliff does not exist (4× IS max-width 256).

## Verification Results

### Conditions Checked

Not run — no completed run exists (`Outcome: failed`). Both pre-registered attempts were killed by the plan's GATE_KILL screen (the screen working as designed). Per plan-040 § Configuration Changes, the both-kill branch pre-registers verdict `invalid` (metric NaN) with the measured dts as the learning.

### Informational Metrics

Not collected (no completed run). The experiment's data product is the dt measurements: 5× → 54.0ms, 4.5× → 54.0ms (52.8–54.0 windows), vs 4× → 22.31ms (EXP-039, same day, same gates).

## Human Notes

(autopilot — none)
