# Experiment Report EXP-056: Full pre-activation block reorder — the last structural class and the final standard-modernization entry

- **Date**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-056.md
- **Plan**: plans/plan-056.md
- **Exp-log**: logs/exp-log-056.md
- **Verdict**: no-improvement
- **Metric**: 96.49 vs baseline 96.71 (bar 96.81)

## Goal

Maximize CIFAR-10 best_test_acc (%) within the fixed 300s charged budget; higher is better. Baseline 96.71 @ 1990397; bar ≥ 96.81. σ context (EXP-027): recipe mean ≈ 96.57, σ ≈ 0.16.

## Idea & Hypothesis

Block operation ORDER was the last structural class never enumerated (every prior structural experiment changed block CONTENT), and full pre-activation (He et al. 2016, arXiv:1603.05027) was the last unread standard-modernization entry — with the best-matched anchor in project history: WRN, the source of our 4× widths, uses pre-act B(3,3) natively at our depth/width range. Construction: BN(in)→ReLU→conv→BN(out)→ReLU→conv added to a CLEAN identity (raw-x pad shortcut, no post-addition ReLU), bare stem, final BN→ReLU before GAP; params exactly preserved by BN-size cancellation; zero config changes. Hypothesis: if the v2 clean-identity mechanism survives heavy-aug absorption at depth 20, the read lands above the recipe mean (≥ 96.81 if ≥ +0.3). Pre-registered branches: (i) ≥ 96.81 → replicate-pair (MEAN); (ii) [96.41, 96.73] family-shaped → absorption-null, class closed, audit complete; (iii) < 96.41 → post-act order load-bearing at shallow depth; (iv) probe/D0 > 26ms → cost-closure; (v) infra.

## Approach

train.py only (+14/−7, 4 hunks): BasicBlock.bn1 → `BatchNorm2d(in_channels)`; pre-act forward with raw-x shortcut and no post-add ReLU; ResNet stem bare (stem bn1 deleted), `bn_final = BatchNorm2d(256)` + `relu(bn_final(·))` before GAP. Optimizer/warmup/loop byte-identical. CPU sanity ALL PASS including the exact-param cancellation (4,286,026 — −128 stem, −128/−256 transition bn1 shrinkage, +512 bn_final nets to zero) and a clean-identity mechanism check (block outputs take negative values). **GPU probe before launch (EXP-055 protocol): pre-act graph = 23.08ms vs 22.04 family — a +1.0ms op-order/fusion toll (~−7 epochs ≈ −0.08pp by the conversion law), recorded as a pre-launch plan revision** (epochs [128,138], steps [12,400,13,400], D0 acceptance [22.6,24.1]); launched per the pre-registered probe rule (≤ 24.0 → proceed).

## Execution

One operator error before any charged work: the first composite launch was mistakenly detached with stdout to /dev/null (no telemetry); killed during startup and relaunched properly (recorded in Errors & Dead Ends; lesson: always `run_in_background`, never `pkill` by a pattern in your own command line). Run 1 proper: PRISTINE — gates poll 1, D0 24.0ms ∈ acceptance, windows 23.3–24.0 throughout, slow_streak 0, rc=0; **best 96.49**, final 96.40, final_test_loss 0.1887 (family), 300.0s charged, 456.8s total, **131 epochs / 12,618 steps** (the +1.0ms toll cost ~7 epochs exactly as priced), params 4,286,026 exact, evals 131 ≤ 131, ep1 33.77 (above the 30 tripwire, healed by ep3), converged plateau 96.33–96.49.

## Results

Branch (ii) — a clean two-part closure:

1. **The order effect is ≈ 0 after the toll.** Raw read 96.49 = mean − 0.5σ; adding back the priced ~+0.08 epoch deficit puts the pre-activation mechanism itself at zero. The plateau is family-shaped, test_loss family-band — no level change in either direction. He's v2 gain is a depth phenomenon (vanishes at depth 20), and whatever residue remained is absorbed by TA+RE like every other imported technique. External transfer is now 0-for-16.
2. **The op reorder is NOT free at the kernel level — +1.0ms/step (4.5%) at an identical op multiset.** Inductor fuses the post-act order (conv→BN→ReLU epilogues) better than the pre-act order (BN→ReLU→conv prologues + an unfused final BN-ReLU). This extends the pointwise-cost law (EXP-026): not just WHICH ops, but their ORDER is a throughput variable; "same ops, same FLOPs" does not imply same dt. Any future reorder must be GPU-probed first (the ~90s probe priced this for pennies and made the read interpretable).
3. **The standard-modernization audit is COMPLETE.** Every technique in the modern CIFAR/ResNet toolkit has now been measured on this recipe: one-cycle/warmup/LS/TA/RE/nesterov/selective-WD (in the recipe), width (the one big win), compile/bf16/channels_last (wins), and the full null/negative ledger (mixup, EMA, SWA, SAM, SE, blurpool, zero-γ, whitening, projection shortcuts, pre-act, FreezeOut, depth variants, activations, pooling, ensembles, precision both ways, throughput flavors). Nothing on the standard menu remains unread — 50 consecutive nulls all decompose into the standing laws.
4. The exact-param cancellation arithmetic (−128 −128 −256 +512 = 0) verified in vivo — a small but satisfying confirmation that the BN-position accounting was right.

Trajectory: 50 consecutive non-improvements. The honest state: the certified recipe sits at a measured local optimum of every named axis, class, and ordering; remaining upside, if any, lives in constructions outside the published toolkit that simultaneously pass all standing laws — or in accepting the recipe as the measured ceiling of this architecture/budget pair.

## Verification

- Integrity pre-condition: PASS (D0 24.0 ∈ [22.6, 24.1] probe-revised; windows ≤ 24.0 none > 27; params exact; 300.0s; 456.8 ≤ 600; 131 evals ≤ 131; ep1 33.77 > 30; family trajectory/plateau/test_loss; no NaN). The botched first launch never produced charged work and does not enter the record as a run.
- Condition 1 (best ≥ 96.81): FAIL — 96.49. First-failure-stop; no escalation. Branch (ii).
- Conditions 2–3: PASS informationally (456.8s; 131 ≤ 131).
- Trust review: fresh log, watchdog cross-check, probe-vs-D0 consistency (+0.9 offset matches EXP-055's probe-to-run offset). Verdict basis: valid result below bar → **no-improvement**.

## Unexplored Avenues

- **Pre-act WITHOUT the toll** (hand-fused or channels-last-friendly ordering): would isolate the pure order effect — but the toll-adjusted effect already reads ≈ 0, so a free version would read mean-band at best. Closed by arithmetic.
- **Partial pre-act (transition blocks only)**: a dose of a zero-effect mechanism inherits the null (same logic as SE-dose closure, EXP-037).
- **The probe-before-launch protocol** is now twice-validated (EXP-055 mechanism check, EXP-056 cost pricing) and should be standing practice for ANY graph change.

## Next Steps

1. **Acknowledge the measured ceiling in the next brainstorm** (high): with the modernization audit complete and 50 nulls decomposing into laws, candidate generation must move outside the published toolkit — constructions designed from the laws themselves (free in dt/heat/noise/numerics, full tail pressure, mechanism not aug-suppliable) — and weight closure-value heavily, since improvement probability per run is now demonstrably small.
2. **Possible law-derived corners still technically open** (medium): per-layer WD/LR on the one BN-free layer (fc) carries a loss-geometry negative prior; late batch-size schedule carries a noise-law negative prior — both documented in brainstorm-056 as runnable-but-expected-null; only worth a run if the next sweep produces nothing better.
3. **Do not revisit**: block order (this), any structural class, any throughput flavor, precision, schedule family, noise, averaging, regularization dose (all measured-closed; high confidence).

## Key Learning

Pre-activation — the best-matched external anchor ever imported (same dataset, depth range, and width family as WRN) — reads exactly zero after pricing its surprise +1.0ms fusion toll: He's v2 gain is a depth phenomenon, absorbed like all 15 prior imports under heavy augmentation. Two durable instruments came out of the null: op ORDER is a throughput variable even at an identical op multiset (inductor fuses post-act epilogues better — extend the pointwise law), and the 90-second GPU probe is now the standing pre-launch gate for any graph change. With this, the standard-modernization audit is complete: every entry on the modern toolkit menu has been measured on this recipe, and the remaining search space is constructions no paper has named.
