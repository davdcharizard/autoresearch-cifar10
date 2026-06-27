# Experiment Report EXP-055: FreezeOut-style tail freezing of stem+stage1 — the first compute-reallocation construction

- **Date**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-055.md
- **Plan**: plans/plan-055.md
- **Exp-log**: logs/exp-log-055.md
- **Verdict**: no-improvement
- **Metric**: 96.32 vs baseline 96.71 (bar 96.81)

## Goal

Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget; higher is better. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16.

## Idea & Hypothesis

After 48 nulls closed every catalogued axis, the one untouched dimension was the ALLOCATION of compute across layers and time. FreezeOut (Brock et al. 2017, arXiv:1706.04983) freezes early layers after a per-layer compressed anneal with ~no accuracy loss at fixed epochs; under this project's fixed-TIME budget the freeze's compute saving converts to extra steps the paper threw away. Construction: group A = conv1+bn1+layer1 (~⅓ conv FLOPs, only 5.2% of params) on `lr_A(p) = lr_at(min(p/0.70, 1))` (unscaled compressed one-cycle, completes at p = 0.70), frozen at p ≥ 0.70 with BN stats still tracking; saved backward (~22–31% of step) → extra tail steps for layers 2–3 plus extra plateau evals. Hypothesis: if FreezeOut's freeze-cost ≈ 0 transfers, the read lands above the recipe mean; ≥ 96.81 if the conversion is worth ≥ +0.3. Pre-registered branches: (i) ≥ 96.81 → replicate-pair (MEAN decides); (ii) [96.41, 96.73] with delivered step surplus → freeze free but tail steps sub-σ; (iii) < 96.41 → early-layer tail refinement load-bearing, parameter-side tail-pressure law; (iv) recompile leak → one fix; (v) infra → relaunch.

## Approach

train.py only (1 file, +43/−7 final): `FREEZE_FRAC = 0.70`; optimizer rebuilt as 4 param groups (B-decay/B-nodecay/A-decay/A-nodecay, `tag` keys, B first so the printed LR tracks the live group); per-step per-group LR (`lr_a` compressed, `lr_b` baseline); one-shot flip at p ≥ 0.70 with a FREEZE marker print; compile warmup extended to pre-cache BOTH graph variants (3 unfrozen + 2 frozen iters, random data, no optimizer.step — the established EXP-006 uncharged pattern).

**The implementation finding of the loop**: the first mechanism — mid-run `p.requires_grad_(False)` on group A — is a SILENT NO-OP under torch.compile: Run 1's flip printed its marker but post-freeze windows stayed 22.0–22.7ms, no recompile fired, and startup 12.5s revealed the "frozen variant" warmup had compiled nothing (the cached graph does not guard on param requires_grad here). Fix (the plan's single permitted attempt): a graph-visible `self.freeze_stage1` bool gating `out = out.detach()` at the stage-1/stage-2 boundary — dynamo guards the flag, the dual warmup genuinely caches two variants, and detach provably cuts backward at the boundary. Validated by CPU sanity v2 (A grads None, B grads flow, BN stats track, eval forward flag-invariant) and a ~90s GPU probe: unfrozen 22.04ms, frozen 15.15ms (**31.3% saving** — better than the 22% estimate because detach also drops the input-grad chain), post-flip first step 0.016s (no recompile). Probe-revised signature bands were written into the plan before relaunch.

## Execution

Run 1 (rejected): mechanism no-op as above PLUS an unrelated contention episode (windows 33/46/48ms at ticks 6–8, slow_streak 3) — 12,893 steps, 96.26, not a valid hypothesis read. Run 2 (decision run): PRISTINE — gates poll 1, D0 22.5ms, every pre-freeze window 22.0–22.7ms, FREEZE at step 9355/progress 0.700, transition window 19.3ms, then 15.3–16.5ms to the end; **15,026 steps / 155 epochs** (+~1,550 steps, +~16 epochs over family 13,400–13,500/138–140), params 4,286,026, 300.0s charged, 551.4s total, 155 evals ≤ 155, ep1 35.79 in band, zero NaN, converged-flat plateau (last 8: 96.16–96.32), final_test_loss 0.1921. **best_test_acc 96.32.**

## Results

Branch (iii), and unusually informative because every link of the causal chain was instrumented and DELIVERED — only the last one (accuracy) inverted:

1. **The conversion mechanism works perfectly and is now measured**: freezing ⅓ of conv FLOPs cut charged dt 22.5 → 15.8ms (31%) with zero recompile toll, yielding +1,550 tail steps and +16 plateau evals. This is the largest legitimate throughput delta the project has ever produced — and it bought nothing.
2. **The read is a real negative, not a null**: 96.32 = mean − 1.6σ at otherwise family-shaped signatures. Decomposed: the surplus steps + extra evals should have HELPED (more max-statistic draws on a longer plateau); instead the level dropped ~0.25 below mean. The freeze package (compressed 0.70×-heat anneal for A + frozen A through the last 30%) costs ≈ −0.3 to −0.4 of plateau LEVEL, overwhelming the surplus.
3. **The tail-pressure law is now PARAMETER-SIDE too**: EXP-025/033 showed the DATA distribution must stay at full pressure to the last step; EXP-055 shows the same for the PARAMETER set — stem/layer1 refinement during the anneal's tail is load-bearing, even though those layers hold only 5.2% of params and their own anneal had completed. The low-LR tail is not "polishing the head on frozen features"; it is a coupled whole-network descent that breaks when a third of the FLOPs stop moving. (Confound noted honestly: A's compressed anneal — hotter early, 0.70× integrated heat — is part of the package; per pre-registration the package, not the isolated freeze, is what branch (iii) closes.)
4. **FreezeOut's transfer failure extends the absorption record to 0-for-15** and adds a new failure mode: prior failures were techniques whose mechanism the heavy-aug recipe absorbed; this one delivered its mechanism in full and lost on a regime mismatch the paper could not see (fixed-epoch "no accuracy loss" ≠ fixed-time "the freed time is worth more than the frozen layers' tail refinement" — here it measured the OPPOSITE sign).
5. Engineering dividends banked: the silent requires_grad/compile no-op (infra-errors), the graph-visible flag+detach pattern with dual-variant warmup (validated, reusable for any mid-run topology change), and the GPU-probe-before-relaunch protocol that converted a would-be wasted run into a 90-second measurement.

Trajectory: 49 consecutive non-improvements. The first compute-reallocation construction closes with a measured negative sign; allocation-in-TIME (freeze schedules) is now priced alongside allocation-in-SPACE (width/depth, closed earlier).

## Verification

- Integrity pre-condition: PASS on Run 2 (gates, D0, window bands pre/post-freeze, FREEZE marker at 0.700, steps/epochs in probe-revised bands, params exact, 300.0s, 551.4 ≤ 600, evals ≤ epochs, trajectory criterion, no NaN). Run 1 rejected (contention + mechanism no-op) and excluded from the decision.
- Condition 1 (best ≥ 96.81): FAIL — 96.32. First-failure-stop; no escalation (< 96.81). Branch (iii) closure.
- Conditions 2–3: PASS informationally (551.4s ≤ 600; 155 ≤ 155).
- Trust review: fresh log, clean parse, watchdog cross-check, mechanism verified by an independent GPU probe AND the in-run dt drop. Verdict basis: valid result below bar → **no-improvement**.

## Unexplored Avenues

- **Amplitude-scaled FreezeOut** (A's peak × 1/0.70 ≈ 0.57 to preserve integrated heat): the documented alternative — but it violates the certified-peak heat law (EXP-010: hotter peak −0.57 globally), and with the unscaled package at −1.6σ the heat shortfall would need to explain the entire deficit to flip the sign. Low prior; only worth it if a future result implicates A's heat specifically.
- **Later freeze (p = 0.85–0.90)**: shrinks both the cost and the benefit proportionally; the measured −0.25 level deficit vs +0.05-ish surplus value at p=0.70 means a later freeze interpolates toward zero from below. Closed by interpolation logic.
- **Freeze-without-schedule-compression** (A keeps the baseline anneal, freeze at 0.70 mid-anneal): isolates the freeze from the compressed-anneal confound, but truncating A's anneal is a worse-credentialed package (EXP-016: the anneal's completion is load-bearing). Not worth a run for attribution alone.
- **The detach-flag + dual-warmup pattern** is validated engineering for ANY future mid-run graph change (e.g., progressive architecture, conditional paths) — the pattern survives even though this use of it lost.

## Next Steps

1. **Treat allocation-in-time as closed; keep enumerating un-catalogued construction classes** (high): the brainstorm sweep that produced EXP-055 (rejecting stratified batching, lookahead, stochastic depth, logit temperature, LARS) should continue from the updated frontier — the surviving space is constructions free in heat/epochs/noise/numerics AND keeping all params + full data pressure to the last step.
2. **Mine the new instrument** (medium): the freeze mechanism is a measured 31% step-time lever that is now known to be NET-NEGATIVE when bought with tail refinement — but it prices what the tail refinement of stem+layer1 is WORTH (≥ ~0.3), the first direct measurement of late-schedule per-layer value; future per-layer ideas must beat that number, not hand-wave it.
3. **Do not retry freeze variants** (high confidence): scaled/later/uncompressed variants are interpolations of a measured −1.6σ package against laws already closed (heat, anneal-shape).

## Key Learning

The first compute-reallocation experiment delivered its entire causal chain — 31% step-time saving, +1,550 tail steps, +16 plateau evals, zero engineering toll — and still lost 1.6σ: under a fixed time budget, the anneal tail's refinement of even the smallest-parameter early layers is worth more than a third of the network's compute converted into extra steps for the rest. The tail-pressure law is two-sided now (data AND parameters): nothing — distribution or weights — may stop moving before the budget ends. Separately: mid-run `requires_grad` flips are silently ignored by compiled graphs; graph-visible flags + detach + dual-variant warmup is the validated pattern, and a 90-second GPU probe before relaunch is what kept the fix from costing a second wasted run.
