# Experiment Log EXP-028: Muon optimizer for conv weights (airbench-anchored hybrid)

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-028.md
- **Plan**: plans/plan-028.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-028 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (verification condition 1: best 96.53 < 96.81 bar; clean Run 2 stands, Run 1 discarded as contaminated)

## Implementation Notes

### Summary
Plan-028 Milestone 1 implemented in train.py (+42/−4): (1) constants `MUON_PEAK_LR = 0.24`, `MUON_MOMENTUM = 0.6`; (2) module-level `zeropower_via_newtonschulz5` (Jordan's quintic NS-5, bf16, transpose-if-tall); (3) parameter split — 19 conv weights (ndim==4) to Muon with per-param momentum buffers, fc weight (ndim==2) keeps SGD+WD, 39 ndim≤1 params keep SGD without WD; (4) timed step: `optimizer.step()` (SGD groups) followed by an inline no_grad Muon block (nesterov buffer update → NS5 on the (C_out, C_in·k·k) reshape → scale √max(1, rows/cols) → decoupled WD `p.mul_(1−lr·wd)` → `p.add_(O, alpha=−muon_lr_now·scale)`); `muon_lr_now = MUON_PEAK_LR × lr_at(progress)/PEAK_LR` reuses the time-keyed one-cycle shape exactly. Sanity: AST OK; split = 19/1/39 tensors; total params 4,286,026 exact; NS5 smoke test → singular values [0.67, 1.04] (loose orthogonalization, as designed).

### Surprises & Discoveries
- **zero_grad coverage bug caught at implementation**: `optimizer.zero_grad()` only clears params in the optimizer's groups — the Muon conv params are outside it, so their grads would have ACCUMULATED across steps (and from the 3 warmup backwards). Fixed by replacing both call sites with `model.zero_grad(set_to_none=True)` (module-level, clears everything). This bug would have produced a silent slow divergence — worth remembering for any hybrid-optimizer setup.

### Decisions
- `.reshape` (not `.view`) for the 2D flatten — channels_last conv weights are non-contiguous; reshape's copy cost is honestly inside the timed region.
- `o.to(p.dtype)` on the NS5 output (bf16 → fp32) before the param update.
- Warmup loop unchanged (forward/backward only, no Muon state mutation — buffers stay zero until the timed loop), so compile warmup still lands in startup.

## Run Log

### Run 1 (MUON_PEAK_LR = 0.24)
- **Description**: Full 300s-budget run on GPU 0. First change of optimizer FAMILY in the campaign: Muon (orthogonalized nesterov momentum, NS-5) on the 19 conv weight matrices, SGD baseline path for fc/BN/biases. Anchor hyperparameters from airbench (lr 0.24, momentum 0.6). Expected: dt 23–27ms (NS overhead, early-dt gate kills >27), epochs ~115–135, trajectory at/above baseline family at matched progress, plateau ≥ +0.25 above the baseline MEAN (96.57) — bar 96.81. Divergence or ep10 <70% triggers pre-authorized Run 2 at peak 0.12.
- **Job ID**: local background composite, task b7legfb3q (early-dt gate kill at 3 consecutive >27ms in first 7 ticks; contention kill 4 consecutive >32ms; startup gate tick 10; NaN/inf guard)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed rc=0 BUT post-hoc profile fails the strict pre-condition (4 of 234 windows >32ms vs ≤2 allowed) → discarded per contention protocol, rerun once (Run 2). NOTE: epochs 122 vs expected 122.2 — nothing meaningfully stolen; spikes may be intrinsic Muon-step jitter (19-iteration Python loop), which Run 2 will discriminate.
- **Started**: 2026-06-10T14:22:37Z
- **Ended**: 2026-06-10T14:30:08Z
- **Observations**: Muon dt overhead = +2.8ms exactly as estimated (mean win 25.5ms incl. spikes, watchdog steady 24.7–25.5) → 122 epochs (−17). **The sample-efficiency mechanism is REAL and visible**: ep1 37.4 / ep5 73.7 / ep10 84.5 vs baseline family ~38 / ~64 / ~78 — Muon ran WAY ahead at matched epochs mid-run too (ep30 93.2, ep60 94.6/best 95.0, ep90 96.0). But the converged plateau capped at 96.3–96.4 (final-7 median 96.32, best 96.42 at −0.29/−0.15 vs draw/mean) with final_test_loss 0.1944 vs baseline ~0.184–0.189. Early advantage did not convert into a higher plateau — the tail converged to a slightly worse basin (test_loss gap) PLUS the 17-epoch deficit (~−0.26 by EXP-006 arithmetic).
- **Key Metrics**: best 96.42 | final 96.31 | final_test_loss 0.1944 | total 443.4s | startup 12.7s | VRAM 1630.9 (+18 = Muon buffers) | 122 epochs / 11758 steps | params 4,286,026 ✓ | final-7 median 96.32. Source: task b7legfb3q + run.log.

### Run 2 (contamination rerun, same code — MUON_PEAK_LR = 0.24)
- **Description**: Byte-identical relaunch of Run 1 (no code change) per the contention protocol ("contaminated ⇒ rerun once"). Run 1's 4 slow windows >32ms exceeded the ≤2 limit even though epochs matched expected exactly (122 vs 122.2). This run both supplies the clean measurement and discriminates the spike origin: foreign contention should not repeat; intrinsic Muon-step jitter (19-iteration eager Python loop, GC) would reproduce ~2–4 spikes. Same gates as Run 1.
- **Job ID**: local background composite, task bdx6pkrn3
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed rc=0, profile CLEAN — `windows>32ms: 0 of 236 | mean win 25.3 ms | expected epochs 123.1` vs 123 actual. Pre-condition passes; THIS run stands for verification. Side finding: Run 1's 4 spikes did NOT reproduce → they were transient foreign contention, not intrinsic Muon-step jitter.
- **Started**: 2026-06-10T14:35:34Z
- **Ended**: 2026-06-10T14:43:20Z
- **Observations**: Replicates Run 1's shape closely: real early sample efficiency (ep10 85.71 vs baseline family ~78; ep30 92.24; ep60 94.86; ep90 95.93) but the converged plateau caps at 96.45–96.53 (final-7 median 96.47, spread 0.08 — tight), final_test_loss 0.1932 vs baseline ~0.184–0.189. Two clean draws now agree: Muon plateau ≈96.4–96.5 vs baseline mean 96.57 — the orthogonalized geometry lands at/below the baseline distribution even before counting the 16-epoch NS tax. dt overhead +2.9ms (25.3 vs 22.4).
- **Key Metrics**: best 96.53 | final 96.50 | final_test_loss 0.1932 | total 447.1s | startup 11.0s | VRAM 1630.9 | 123 epochs / 11852 steps | params 4,286,026 ✓ | final-7 median 96.47. Source: task bdx6pkrn3 + run.log.

## Experimental Adjustments

## Errors & Dead Ends

## Verification Results

Verified on Run 2 (the clean run; Run 1 discarded per contention protocol — Milestone-3 Run-2-at-0.12 trigger did NOT fire, so the rerun kept MUON_PEAK_LR=0.24). First-failure-stop order per plan-028.

### Conditions Checked
1. **best_test_acc ≥ 96.81**: **FAIL** — `grep "^best_test_acc:" run.log` → 96.53%. Pre-condition (post-hoc profile) PASSED: 0 of 236 windows >32ms, mean win 25.3ms, expected epochs 123.1 vs 123 actual. Gap to bar −0.28 (≈1.8σ below); vs baseline recorded best −0.18; vs baseline mean (96.57) −0.04 → plateau statistically AT the baseline mean despite 16 fewer epochs, but nowhere near the bar. First failure → stop.
2. **Completes within budget**: skipped (would pass: rc=0, total_seconds 447.1 ≤ 600).
3. **Validation ≤ once/epoch**: skipped (would pass: eval_lines 123 = num_epochs 123).

### Informational Metrics
- Muon dt overhead on H20: mean win 25.3ms − 22.4 = **+2.9ms/step** (~13%) — eager NS-5 on 19 conv tensors; costs 16 epochs (123 vs 139).
- num_epochs 123 | VRAM 1630.9 (+17.9 = Muon momentum buffers) | startup 11.0s (NS adds no compile time).
- Deferral check: ep1 37.5 (family ~38 ✓ free at init), ep5 72.6, ep10 85.7 vs family ~78 → genuinely AHEAD early; advantage decays to ~0 by plateau. Final-7 median 96.47 vs best 96.53 — tight plateau, no variance spike.
- Cross-run agreement: Run 1 best 96.42 / Run 2 best 96.53 — both inside the baseline noise band around the 96.57 mean; effect ≈ 0 net (sample-efficiency gain ≈ NS time tax + slightly worse basin test_loss 0.193–0.194 vs ~0.185).

## Human Notes
