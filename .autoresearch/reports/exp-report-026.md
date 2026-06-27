# Experiment Report EXP-026: Activation modernization — ReLU → GELU/SiLU/hardswish

- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Exp-log**: logs/exp-log-026.md
- **Verdict**: **no-improvement** (96.61 vs baseline 96.71, Δ −0.10)

## Goal

Maximize CIFAR-10 `best_test_acc` (%) of the wide ResNet-20 within the fixed 300s timed training budget; higher is better. Baseline at experiment time: **96.71** @ 1990397 (EXP-006 recipe). Success bar: ≥ 96.81 (+0.1pp absolute).

## Idea & Hypothesis

**Idea**: Probe the one recipe component never touched in 26 experiments — the activation function. Replace ReLU with GELU at all three activation sites, motivated by both regime-matched budget-race lineages independently adopting smooth activations (airbench: GELU; cifar10-fast: CELU) and by the GELU/SiLU literature's consistent small CIFAR gains.

**Hypothesis**: GELU improves per-step optimization from step 0 at near-baseline signatures (dt 22.4±0.3ms, 137–139 epochs): early trail at/above family, converged plateau ≥ +0.1pp, best_test_acc ≥ 96.81. Screened as the first candidate passing all four campaign laws affirmatively (no deferral, ~no epoch cost, numerics law inapplicable, noise-neutral).

## Approach

Three one-token edits in train.py (the only nonlinearity sites: BasicBlock post-bn1, BasicBlock post-residual-add, stem). Init/constants/loaders/schedule/optimizer/compile/eval untouched. The plan added an **early-dt gate** (kill at 3 consecutive watchdog windows >23.5ms within the first ~7 ticks) to protect the budget if the "pointwise-fuses-free" cost premise failed — which turned out to be the decisive instrument of the experiment.

## Execution

Three runs, all clean (no contention, no crashes):
- **Run 1 — GELU(erf)**: EARLY_DT_KILL at 91s. Steady windows 26.2–27.0ms (+~4.5ms, +20% over baseline 22.4). Projected ~117 epochs — hopeless (deficit ≈ −0.33pp).
- **Run 2 — SiLU** (adjustment per brainstorm's named variants): EARLY_DT_KILL at 90s. Windows 24.8–25.5ms (+~2.8ms, +13%). Projected ~124 epochs.
- **Run 3 — hardswish** (piecewise, no special functions): passed the gate at 23.1–24.0ms (mean 23.5, +1.1ms) and ran to completion: rc=0, total 477.2s, 132 epochs, 12,764 steps, params 4,286,026 exact, VRAM 2015MB (+402 — hardswish autograd saves inputs), startup 24.1s (fresh graph). Profile pristine: 0/254 windows >30ms, expected 132.5 vs 132 actual.

## Results

**best_test_acc 96.61 (−0.10)** — the twenty-first consecutive non-improvement, but an unusually information-dense one. Three findings:

1. **The activation-cost ladder on H20/inductor is real and steep**: erf-GELU +4.5ms, sigmoid-SiLU +2.8ms, piecewise-hardswish +1.1ms per 22.4ms step. Fusion is not the issue — the activation math itself, evaluated forward+backward over 19 large activation tensors per step, consumes 5–20% of the step. The airbench evidence does not transfer because airbench's net is far smaller and its budget regime tolerates the cost; under OUR dt budget every smooth activation must first pay for itself in epochs.

2. **Hardswish's accuracy gain ≈ its own epoch cost, no more**: −7 epochs ≈ −0.10pp by EXP-006 arithmetic, and the run landed exactly −0.10. Any smoothness benefit was fully consumed by the throughput tax — and that is the CHEAPEST smooth activation; GELU/SiLU would have been deeper in the hole.

3. **The swap also paid an early-heat toll**: ep1 test_acc 20.24 vs the baseline family's ~38–39 (recovering to near-family by ep5). The Kaiming √2 gain and BN statistics are tuned to ReLU; hardswish's different effective gain and negative outputs depress the first epochs — a deferral-law cost the screen missed because "active from step 0" is not the same as "calibrated from step 0". Additionally, best landed AT the final epoch (96.61, plateau before it 96.4–96.5): the run was still climbing at cutoff — the 7 lost epochs also truncated the converged-plateau harvest window the max-statistic needs.

Hypothesis validation: refuted at every layer — the cost premise (free fusion) failed twice at the gate; the no-deferral premise failed (init-gain mismatch); the plateau-lift, if any, was smaller than the stacked costs. Root cause is structural, not approach-specific: ReLU is the only activation that is simultaneously free in dt, calibrated to the certified init/heat, and backward-cheap in VRAM. The activation axis is closed on cost grounds for this recipe.

## Verification

- **Pre-condition (contention profile)**: PASS — 0/254 windows >30ms, epochs 132 within ±3 of expected 132.5 (Run 3).
- **Condition 1 (best_test_acc ≥ 96.81)**: **FAIL** — 96.61. Genuine converged result; params exact 4,286,026; eval count = epochs.
- **Conditions 2–3**: skipped per first-failure-stop; both would have passed (rc=0 @ 477.2s; 132 = 132).
- **Verdict basis**: valid run, verification condition failed → `no-improvement`, metric 96.61.

## Key Learning

On a dt-bound recipe, the activation function is not a free hyperparameter — it is a throughput lever. The smooth-activation literature (and even regime-matched speedrun lineages) is calibrated to budgets where +5–20% step time is invisible; under a fixed wall clock the swap must beat baseline by its own epoch deficit PLUS an init-calibration toll before contributing anything. The early-dt gate converted what would have been two wasted full runs into 3 minutes of measurement — the gate is now a validated screening instrument for any architecture-class change.

## Unexplored Avenues

- **Leaky ReLU / parametric slopes**: piecewise like ReLU, near-zero dt cost, but the literature effect on well-regularized BN ResNets is ~0 — not worth a loop on its own.
- **Activation swap + recalibrated init gain**: fixing the ep1 toll (e.g., gain tuned to hardswish) addresses one of the three stacked costs but not the dominant dt cost — the arithmetic still nets negative.
- **Activation in the stem only** (1 of 19 sites — negligible dt): the per-site benefit is presumably also ~1/19th; below noise.

## Next Steps

1. **Baseline variance replicate** (brainstorm-025 idea 3) — after twenty-one misses spanning −0.05…−0.99, measuring run-to-run σ of the unmodified baseline calibrates every past and future near-bar reading; explicitly not seed hacking (cannot move the baseline by construction). Confidence: high as measurement value.
2. **Forward-only BN-stat recalibration micro-phase** (exp-report-025 fragment) — the only surviving piece of the alignment mechanism; budget-charged, cannot overfit; small expected magnitude. Confidence: medium-low.
3. **Width asymmetry (stage-3-only widen to 320)** — the one named-but-untried capacity move (brainstorm-026 idea 2); fights the capacity failure class but is the last unprobed structural direction; would reuse the early-dt gate as its screen. Confidence: low-medium.
