# Experiment Report EXP-053: Cross-axis compound of certified-free components — anti-aliased shortcut + de-overhead prefetch (n=2, MEAN decision)

- **Date**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-053.md
- **Plan**: plans/plan-053.md
- **Exp-log**: logs/exp-log-053.md
- **Verdict**: no-improvement
- **Metric**: 96.445 (pre-registered MEAN of best_A=96.61, best_B=96.28) vs baseline 96.71 (bar 96.81)

## Goal

Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget; higher is better. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16, σ_mean(n=2) ≈ 0.113.

## Idea & Hypothesis

After EXP-052 resolved the last single-mechanism datum, the only unfalsified positive-direction region was compound interventions of individually-certified components. EXP-053 ran the sharpest available test: combine the only two free components with non-negative measured point estimates — the anti-aliased shortcut (+0.11 pooled over n=3 pristine draws; function-quality axis) and the EXP-048 de-overhead pair (collate-side channels_last + side-stream H2D prefetch; numerics-identical, +0.15ms/step → +1 epoch; throughput axis). The components share no measurable currency (heat, noise, params, numerics all untouched), so additivity (≈ +0.13 combined) was the default expectation, distinguishing this from the EXP-009 same-axis stacking failure. Pre-registered decision: MEAN of two byte-identical runs ≥ 96.81; branches (i) pass → improvement; (ii) mean ∈ [96.61, 96.80] → weak-positive-closed; (iii) mean ≤ 96.60 → sub-additive/null-closed; (iv) infra relaunch.

## Approach

One train.py (+52/−7): the EXP-046/052 shortcut diff (one logic line in `BasicBlock.forward`) plus the EXP-048 diff (module-level `collate_channels_last` wired via `collate_fn=`; module-level `CUDAPrefetcher` with side stream, wait_stream/record_stream, CPU passthrough; loop iterates the prefetcher with in-step `.to()` calls deleted). Zero params (4,286,026 exact), timer semantics untouched (synchronize fences all streams). Merged CPU sanity all-pass: shortcut semantics + pad sites; collate value-identity + channels_last contiguity; prefetcher sequence-identity 7/7 over two passes; 2-epoch DataLoader smoke decreasing. No deviations from plan.

## Execution

Both runs pristine on first launch via `/tmp/exp046_composite.sh`, no retries:
- **Run A**: D0 = 22.3ms; windows 21.7–22.7ms; ep1 = 37.66 (inside the 36–41 prefetcher-defect band); 139 ep, 13,428 steps, 300.0s charged, 506.3s total; **best_A = 96.61**, final_test_loss 0.1818, converged-flat tail.
- **Run B**: D0 = 22.5ms; windows 21.7–22.7ms; 139 ep, 13,434 steps, 300.0s, 495.2s total; **best_B = 96.28**, final_test_loss 0.1915, converged-flat tail.

**MEAN = 96.445** → branch (iii).

## Results

The compound did not add — it subtracted. Three observations stack:

1. **The pair reads at/below the recipe mean**: 96.445 = mean − 0.8σ (−1.1 σ_mean). The shortcut-alone pool sat at +0.11 (96.65/96.84/96.56); adding the prefetch produced draws of 96.61 and 96.28. The compound's point estimate is −0.13, refuting additivity at the only effect size that mattered: whatever small positive the shortcut may carry, the compound does not preserve it above noise.
2. **The prefetch's throughput payoff did not reproduce**: steps 13,428/13,434 — exactly the 046-family figure, not EXP-048's 13,515. The +87-step (+0.65%) saving measured once was evidently at the top of its own scatter; with ~0 extra steps delivered, the compound's *expected* gain collapses to the shortcut term alone, and the observed reads are consistent with shortcut-noise minus nothing. The "free" prefetch component brought no measurable benefit while adding code surface.
3. **Run B is a low draw with slightly elevated test_loss (0.1915 vs family ~0.185)** — within scatter, plateau converged-flat, so no defect signature; but it pulls the pooled compound view (n=2: 96.445) clearly below the shortcut-alone pool (n=3: 96.68). At these sample sizes that gap (≈0.24) is ~1.6 σ_mean-difference — not proof of an interaction penalty, but the honest reading is: no evidence of additivity, weak evidence against.

Trajectory context: 47 consecutive non-improvements. The compound-of-frees region — the last catalogued unfalsified positive-direction space — is now closed with a measured negative datum. What remains is genuinely uncatalogued territory: constructions that pass all standing laws and were never enumerated, plus the one never-probed axis recorded in brainstorm-053 (fp16-vs-bf16 mantissa precision).

## Verification

- Integrity pre-condition: PASS both runs (windows 21.7–22.7ms; 139 ep ∈ [136,143]; steps within band; params exact; 300.0s; evals 139 ≤ 139; ep1 tripwire passed Run A; family trajectories and plateaus).
- Condition 1 (MEAN ≥ 96.81): FAIL — 96.445. First-failure-stop. Branch (iii) pre-registered closure. Max never used as a decision input.
- Conditions 2–3 (budget; eval cadence): PASS both runs (informational).
- Trust review: fresh per-run logs, summary blocks parsed cleanly, charging semantics intact (timer untouched; step ledger shows ~0 extra steps — if anything the prefetch under-delivered, ruling out hidden work-shifting). Verdict basis: valid result below bar → **no-improvement**.

## Unexplored Avenues

- **Shortcut + a hypothetical future free positive**: the compound failed partly because the second component delivered nothing this time; if a future loop ever certifies a free component with a ≥ +0.1 point estimate, pairing it with the shortcut remains constructible — but there is no such component today, and the burden of proof is now higher (one negative compound datum).
- **fp16 autocast precision probe** (brainstorm-053 Idea 2): the only never-probed axis left — numerics measured SENSITIVE (EXP-021) but only in the faster-but-different direction; fp16 offers 8× finer mantissa at identical throughput, with GradScaler risk at peak LR 0.4. Unsigned prior.
- **n≈15 resolution of the compound** — unaffordable and pointless below the bar; recorded as resolution limit, not an avenue.

## Next Steps

1. **fp16 (GradScaler) precision probe** (medium confidence in feasibility, low-medium in sign): last never-probed axis; trajectory-criterion verification per EXP-048; abort on scaler-skip storms at peak heat.
2. **Re-read train.py + the standing laws for uncatalogued constructions** (medium): with every catalogued class closed, the next brainstorm must construct from first principles — candidates must be free in heat/epochs/noise/numerics AND argue a mechanism heavy-aug cannot supply (absorption record 0-for-14).
3. **Do not** re-run compounds of the current component pool, re-sample the shortcut, or revisit any closed class (high confidence).

## Key Learning

"Certified free + certified free" does not compose into "free positive": the compound of the project's only two non-negative free components read mean − 0.8σ, and the prefetch's once-measured +87-step saving regressed to zero on both replicates — a single throughput measurement at the top of its scatter is not a bankable component. The compound-of-frees region closes the last catalogued positive-direction space; from here, only never-probed axes (fp16 precision) and first-principles constructions remain.
