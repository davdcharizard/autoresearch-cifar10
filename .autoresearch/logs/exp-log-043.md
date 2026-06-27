# Experiment Log: EXP-043 — Full-alternation two-member ensemble (2 × 4x ResNet-20, per-step alternation, logit-mean inference)

## Execution

- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-043.md
- **Plan**: plans/plan-043.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-043
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed (run clean; verification Condition 1 failed on merits — 96.07 < 96.81)

## Implementation Notes

### Summary
Milestone 1 as planned (72 insertions / 31 deletions, train.py only): `ResNet` untouched; new `MeanEnsemble(nn.Module)` whose forward is `(m1(x)+m2(x))/2`; two members constructed sequentially under seed 42 (independent Kaiming draws), `base_model = MeanEnsemble(model1, model2)` is the eager eval reference, `c1/c2 = torch.compile(model1/model2)`; `make_optimizer()` builds the baseline selective-WD SGD per member; compile warmup runs 3 iters on each compiled member then zeroes both optimizers; the timed loop selects `(c1, opt1)` on even steps and `(c2, opt2)` on odd (parity before increment), sets the time-keyed LR on the ACTIVE optimizer only, and `base_model.train()` re-arms both members after evals; eval thinning evaluates only when `progress ≥ 0.6 or epoch % 3 == 1`. All five CPU sanities passed: (A) members' stem and deep conv weights differ; (B) with the real selection logic, step 0 mutated only member 1's state_dict (params AND BN stats) and step 1 only member 2's; (C) eval-mode `base_model(x)` is a single (4,10) tensor equal to the directly-computed mean logits; (D) constructed params 8,572,052 = 2 × 4,286,026, `git diff --stat` train.py only; (E) thinning sweep: evals at epochs 1,4,7,…, every epoch from ~ep84, total ~84 evals/139 epochs.

### Surprises & Discoveries
- None at implementation time — the loop restructure was mechanical. Noted for the run: the printed loss now alternates between the two members step-to-step (both single-CE scale), so the EMA-smoothed print blends two similar trajectories; cosmetic only.

### Decisions
- `model, optimizer = (c1, opt1) if step % 2 == 0 else (c2, opt2)` reuses the existing variable names inside the loop so the step body (zero_grad/forward/backward/step, loss print) is byte-identical to baseline — minimizes diff and keeps the per-member dynamics audit trivial.
- Warmup order c1-then-c2 (not interleaved): the second compile hits the inductor cache for the identical graph; interleaving would add nothing.

## Run Log

### Run 1
- **Description**: Full budget-matched run of the alternating two-member ensemble on GPU 0 via `/tmp/exp043_composite.sh` (exp041 baseline-threshold watchdog with STARTUP_KILL moved to tick 12 for the double compile; contention >27ms, NaN/divergence/wall guards). dt expected ≈ 22.4ms (dense baseline kernels; alternation adds no kernel work), ~139 loader epochs, ~6,700 steps/member (~70 epoch-equivalents each), ~84 evals (thinned below 60% progress), total ~510–570s. Hypothesis: function-space averaging of two fully-independent members (init + disjoint batch streams) lifts the converged plateau LEVEL above the bar (96.81) if the 2-member decorrelation gain survives member starvation. Pre-registered branches: (ii) sub-bar best with final_test_loss ≤ ~0.165 = mechanism real, starvation-limited → mid-fork next loop; (iii) in-band best with family-equal loss (~0.185) = averaging dichotomy closed both halves. Early ensemble evals EXPECTED below family (members at half steps) — not a defect.
- **Job ID / PID**: background task b1uqntkeh (`/tmp/exp043_composite.sh`)
- **Log file**: run.log (project root); watchdog via composite stdout (task output file)
- **WandB**: N/A
- **Status**: completed (rc=0, no watchdog trigger)
- **Started**: 2026-06-10 21:24:52 (gates clear at poll 1: apps=0, load=6)
- **Ended**: 2026-06-10 ~21:32:45 (PROC_EXITED at tick 32; total_seconds 470.3)
- **Observations**: Pristine run: 29 watchdog windows 21.7-22.8ms, slow_streak never above 0 (task b1uqntkeh); 200-step quantization-safe windows: 66, mean 22.34ms, max 22.5, 0 > 27ms — dense-kernel claim confirmed exactly (alternation costs nothing in dt). Startup 11.5s (second compile hit the inductor cache as predicted). 139 epochs / 13,426 steps (~6,713/member); 85 evals (thinning predicate exact: ep 1,4,7,... then every epoch from ~ep84). Eval wall cost fully absorbed: total 470.3s, BELOW the baseline family's ~473-500s despite double-cost evals. peak_vram 1655.3MB.
- **Key Metrics**: best_test_acc 96.07 (ep137), final 96.06, final_test_loss 0.1961 (family ~0.185 — slightly WORSE, not better). Trajectory: ep10 69.79 (family ~75), ep22 82.18 (family ~81 at ep20), consistently ~1.5-5pp below family mid-run — the half-step starvation signature. Plateau last-15: mean 95.915, min 95.59, max 96.07, spread 0.48 (family ~96.5/+-0.15): plateau LEVEL ~0.6 below family mean with ~3x scatter.

## Experimental Adjustments

(none yet)

## Errors & Dead Ends

(none yet)

## Verification Results

### Conditions Checked

First-failure-stop per plan-043 § Verification Protocol; baseline at verification time 96.71 (bar 96.81).

- **Pre-condition — run integrity**: PASS. Profile pristine: 200-step quantization-safe windows (66) mean 22.34ms ∈ [22.0, 23.5], max 22.5, 0 > 27ms; num_epochs 139 ∈ [130, 145]. Integrity: num_params 8,572,052 ✓; training_seconds 300.0 ✓; eval-line count 85 (expected 80–90, ≤ 139 epochs) ✓. Source: run.log greps; composite stdout (task b1uqntkeh).
- **Condition 1 — best_test_acc ≥ 96.81**: **FAIL on merits.** `grep "^best_test_acc:" run.log` → 96.07 < 96.81 (also below the baseline band 96.4–96.7). Clean profile, full epoch count, exact thinning — no false-failure risk. Verification stopped per first-failure-stop.
- **Condition 2 — completes within budget**: skipped per protocol (incidental: rc=0, total 470.3 ≤ 600).
- **Condition 3 — validation ≤ once/epoch**: skipped per protocol (incidental: 85 ≤ 139).

**Diagnostics (pre-registered interpretive branches)**: NEITHER branch (ii) nor a clean (iii) — the result is a THIRD, more informative shape: best 96.07 sits ~0.5 BELOW the baseline band (not in it) and final_test_loss 0.1961 is slightly worse than family (~0.185), not better. Decomposition: single 4x members at ~6,713 steps (~70 epoch-equivalents) price at ~95.5–95.7 on the starvation ladder (EXP-002/005/007 interpolation), so the ensemble's 96.07 implies a REAL function-space decorrelation gain of roughly +0.3–0.5 over its members — consistent with the literature range — that is fully consumed (2–3×) by the half-step starvation cost (~−0.9 vs the 96.57 single-model mean). The averaging-axis bracket is now complete: fork-at-0 (this run) = 96.07; fork-at-1 (zero diversity: SWA EXP-032) = 96.60; EMA (EXP-011) = 96.46 — every interior point (mid-fork) interpolates to ≲96.6, sub-bar. Elevated plateau scatter (0.48) is the starved-members signature: at the anneal end each member still moves more per remaining step.

### Informational Metrics

- peak_vram_mb: 1655.3
- num_epochs: 139 (family-equal); num_steps 13,426 (~6,713/member)
- num_params: 8,572,052 (= 2 × 4,286,026)

## Human Notes

(autopilot — none)
